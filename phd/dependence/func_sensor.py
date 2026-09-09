import copy
import json
import os
import threading
from importlib import import_module

import pyvista as pv
import numpy as np
import serial
import serial.tools.list_ports
import time
from pyvistaqt import QtInteractor
from PyQt5.QtCore import QObject, QThread, QTimer, Qt
from PyQt5.QtWidgets import QListWidgetItem
from tqdm import tqdm
from phd.dependence.paths import resource_path, sensor_resource_path
from phd.dependence.sensor_layout import (
    column_major_idx as _column_major_idx,
    flatten_column_major_view as _flatten_column_major_view,
    reshape_sensor_values_to_row_col_matrix,
    row_major_idx as _row_major_idx,
)
from phd.dependence.sensor_geometry import (
    compute_grid_normals,
    heatmap_corner_points_from_centres,
    heatmap_surface_from_corner_lattice,
    interpolate_coarse_deformation,
    normalize_point_array,
    structured_grid_edge_pairs,
    structured_grid_edges,
)
from phd.dependence.sensor_data import SensorDataBuffer
from phd.dependence.goodix_usb_sensor import (
    GOODIX_USB_COLUMNS,
    GOODIX_USB_ROWS,
    GOODIX_USB_SOURCE_ID,
    GOODIX_USB_SOURCE_LABEL,
    GoodixUsbError,
    GoodixUsbReadWorker,
    GoodixUsbSensorClient,
    goodix_usb_connected,
    is_goodix_usb_source,
)
from phd.dependence.humanoid_sensor_registry import (
    humanoid_sensor_annotation,
    load_humanoid_device_assignments,
)
from phd.dependence.sensor_heatmap import (
    DEFAULT_HEATMAP_3D_COLOR_GAIN,
    DEFAULT_HEATMAP_3D_PALETTE,
    DEFAULT_HEATMAP_NOISE_FLOOR_PCT,
    DEFAULT_HEATMAP_RESPONSE_MODE,
    DEFAULT_HEATMAP_SATURATION_PCT,
    DEFAULT_PROXIMITY_KNEE,
    DEFAULT_PROXIMITY_NOISE_FLOOR,
    DEFAULT_PROXIMITY_SATURATION,
    HEATMAP_RESPONSE_PROXIMITY_ENHANCED,
    heatmap_3d_rgb,
    normalize_heatmap_3d_palette,
    normalize_heatmap_response_mode,
)
from phd.dependence.sensor_protocol import (
    DEFAULT_SENSOR_BAUD_RATE,
    DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC,
)
from phd.dependence.sensor_serial import (
    SensorCalibrationBridge,
    SensorPayloadBridge,
    SensorReadWorker,
)

# Persistent storage for per-sensor "always-zero" cell masks. The file holds
# a mapping of <model>_<n_row>x<n_col> -> 2D 0/1 list, so each sensor layout
# remembers which cells the user wants forced to zero across runs.
SENSOR_ZERO_MASK_FILE = os.path.join(
    resource_path("config"),
    "sensor_zero_masks.json",
)
SENSOR_REORDER_LOGIC_FILE = os.path.join(
    resource_path("config"),
    "sensor_reorder_logic.json",
)

REORDER_FACTORY_DEFAULT = "factory"
REORDER_NONE = "none"
POINT_GRID_RESPONSE_ZERO_CENTERED = "zero_centered"
POINT_GRID_RESPONSE_LEGACY_OFFSET = "legacy_offset"
REORDER_LOGIC_OPTIONS = (
    REORDER_FACTORY_DEFAULT,
    REORDER_NONE,
    "row_to_col",
    "row_to_col_flipped",
    "row_to_col_c_flip_only",
    "row_to_col_r_flip_only",
    "col_to_row",
    "col_to_row_flipped",
    "col_to_row_c_flip_only",
    "col_to_row_r_flip_only",
    "vertical_flip",
    "horizontal_flip",
    "flip_and_rotate",
    "rotate_180",
)

# Compatibility aliases keep the established internal and external imports stable.
_SensorReadWorker = SensorReadWorker
_SensorCalibrationBridge = SensorCalibrationBridge
_SensorPayloadBridge = SensorPayloadBridge
data = SensorDataBuffer


class SensorModelFactory:
    """
    A factory class to handle the initialization and data structures of a sensor model.
    It encapsulates the logic for creating the grid, loading mesh/signal files,
    calculating points and normals, applying reordering, and preparing for visualization.
    """

    def __init__(
        self,
        n_row,
        n_col,
        mesh_file=None,
        signal_file=None,
        reorder_logic=None,
        offset_scale=0.2,
        window_size=3,
    ):
        """
        Initializes the sensor model configuration.

        Args:
            n_row (int): Number of rows in the sensor grid.
            n_col (int): Number of columns in the sensor grid.
            mesh_file (str, optional): Path to the OBJ mesh file. Defaults to None.
            signal_file (str, optional): Path to the signal mapping txt file. Defaults to None.
            reorder_logic (str, optional): The reordering strategy. Options: 'row_to_col',
                                         'row_to_col_flipped', 'vertical_flip', None.
            offset_scale (float, optional): Scaling factor for visualization displacement.
            window_size (int, optional): Averaging window size for `*Ave` sensor buffers.
        """
        # Store configuration
        self.n_row = n_row
        self.n_col = n_col
        self.n_node = self.n_row * self.n_col
        self.mesh_file = mesh_file
        self.signal_file = signal_file
        self.reorder_logic = reorder_logic
        self.offset_scale = offset_scale
        self.window_size = max(1, int(window_size))

        # Initialize core data structures
        self._data = data(self.n_row, self.n_col, window_size=self.window_size)
        self.points = np.zeros((self.n_node, 3))
        self.points_origin = np.zeros((self.n_node, 3))
        self.normals = np.zeros((self.n_node, 3))
        self.edges = None
        self.colors_3d = np.ones((self.n_node, 4)) * 0.5
        self._2D_map = None
        self.array_positions = []
        self.colors = None
        self.line_poly = None

    def build(self):
        """Executes the full build pipeline and returns the completed instance."""
        is_row_major_input = str(self.reorder_logic or "").startswith("row_to_col")
        major_order = 'row' if is_row_major_input else 'column'

        self._build_edges(major_order=major_order)

        if self.mesh_file and self.signal_file:
            fine_normals = self._load_files()
        else:  # 2D grid case
            fine_normals = self._generate_2d_grid()

        self._calculate_coarse_grid(fine_normals)

        if self.reorder_logic:
            self._apply_reordering()

        # After any reordering, the final structure is treated as column-major for coloring
        self._setup_visualization_data(major_order='column')

        return self

    def _build_edges(self, major_order='column'):
        """Builds the edge array for the grid based on memory layout."""
        total_edges = (self.n_col - 1) * self.n_row + (self.n_row - 1) * self.n_col
        self.edges = np.zeros((total_edges, 3), dtype=int)
        edge_idx = 0

        if major_order == 'column':
            # Vertical edges
            for col in range(self.n_col):
                for row in range(self.n_row - 1):
                    self.edges[edge_idx] = [
                        2,
                        _column_major_idx(self.n_row, col, row),
                        _column_major_idx(self.n_row, col, row + 1),
                    ]
                    edge_idx += 1
            # Horizontal edges
            for col in range(self.n_col - 1):
                for row in range(self.n_row):
                    self.edges[edge_idx] = [
                        2,
                        _column_major_idx(self.n_row, col, row),
                        _column_major_idx(self.n_row, col + 1, row),
                    ]
                    edge_idx += 1
        elif major_order == 'row':
            # Horizontal edges
            for row in range(self.n_row):
                for col in range(self.n_col - 1):
                    self.edges[edge_idx] = [
                        2,
                        _row_major_idx(self.n_col, row, col),
                        _row_major_idx(self.n_col, row, col + 1),
                    ]
                    edge_idx += 1
            # Vertical edges
            for row in range(self.n_row - 1):
                for col in range(self.n_col):
                    self.edges[edge_idx] = [
                        2,
                        _row_major_idx(self.n_col, row, col),
                        _row_major_idx(self.n_col, row + 1, col),
                    ]
                    edge_idx += 1

    def _generate_2d_grid(self):
        """Generates a flat 2D grid procedurally."""
        print("Done: Initiate the 2D grid construction.")
        fine_scale = 5
        size = 0.05
        fine_row, fine_col = self.n_row * fine_scale, self.n_col * fine_scale
        fine_points = np.zeros((fine_row * fine_col, 3))
        for fine_col_idx in range(fine_col):
            for fine_row_idx in range(fine_row):
                fine_points[fine_col_idx * fine_row + fine_row_idx] = [
                    (fine_col_idx + 0.5) * size / fine_scale,
                    (fine_row_idx + 0.5) * size / fine_scale,
                    0,
                ]

        fine_points -= np.mean(fine_points, axis=0)
        self._2D_map = pv.PolyData(fine_points)
        fine_normals = np.tile([0, 0, 1.0], (self._2D_map.n_points, 1))

        self.array_positions = [[] for _ in range(self.n_node)]
        for col in range(self.n_col):
            for row in range(self.n_row):
                coarse_idx = _column_major_idx(self.n_row, col, row)
                for fine_col_offset in range(fine_scale):
                    for fine_row_offset in range(fine_scale):
                        self.array_positions[coarse_idx].append(
                            (col * fine_scale + fine_col_offset) * fine_row
                            + (row * fine_scale + fine_row_offset)
                        )
        print("Done: Simulated fine mesh and mapping.")
        return fine_normals

    def _load_files(self):
        """Loads mesh and signal files."""
        print(f"Loading mesh from: {self.mesh_file}")
        self._2D_map = pv.read(self.mesh_file)

        print(f"Loading signal from: {self.signal_file}")
        with open(self.signal_file, 'r') as file:
            numbers = [int(line.strip()) for line in file.readlines()]

            # Special case for elbow model's larger initial array
            num_positions = 156 if self.n_col == 13 and self.n_row == 10 else self.n_node
            self.array_positions = [[] for _ in range(num_positions)]
            for idx, num in enumerate(numbers):
                if num != -1:
                    self.array_positions[num].append(idx)

        # Special case processing for elbow model
        if self.n_col == 13 and self.n_row == 10:
            for i in range(self.n_col - 1, -1, -1):
                del self.array_positions[i * 12 + 11]
                del self.array_positions[i * 12]

        print("Done: Load the mesh and signal data.")
        return self._2D_map.extract_surface().point_normals

    def _calculate_coarse_grid(self, fine_normals):
        """Calculates the coarse grid points and normals from the fine mesh."""
        for i in tqdm(range(self.n_node), desc="Averaging fine points"):
            fine_indices = self.array_positions[i]
            if not fine_indices: continue

            self.points[i] = np.mean(self._2D_map.points[fine_indices], axis=0)
            self.normals[i] = np.mean(fine_normals[fine_indices], axis=0)

            norm = np.linalg.norm(self.normals[i])
            if norm > 0: self.normals[i] /= norm

            self.points_origin[i] = self.points[i]
            self.points[i] += self.normals[i] * self.offset_scale

    def _apply_reordering(self):
        """Reorders points, normals, and edges based on the specified logic."""
        reorder_index_map = np.zeros(self.n_node, dtype=int)

        if self.reorder_logic == 'row_to_col':
            for old_idx in range(self.n_node):
                r, c = old_idx // self.n_col, old_idx % self.n_col
                reorder_index_map[old_idx] = _column_major_idx(self.n_row, c, r)

        elif self.reorder_logic == 'row_to_col_flipped':
            for old_idx in range(self.n_node):
                r, c = old_idx // self.n_col, old_idx % self.n_col
                r_flipped, c_flipped = (self.n_row - 1) - r, (self.n_col - 1) - c
                reorder_index_map[old_idx] = _column_major_idx(self.n_row, c_flipped, r_flipped)

        elif self.reorder_logic == 'row_to_col_c_flip_only':
            for old_idx in range(self.n_node):
                r, c = old_idx // self.n_col, old_idx % self.n_col
                c_flipped = (self.n_col - 1) - c
                r_keep = r
                reorder_index_map[old_idx] = _column_major_idx(self.n_row, c_flipped, r_keep)

        elif self.reorder_logic == 'row_to_col_r_flip_only':
            for old_idx in range(self.n_node):
                r, c = old_idx // self.n_col, old_idx % self.n_col
                r_flipped = (self.n_row - 1) - r
                c_keep = c
                reorder_index_map[old_idx] = _column_major_idx(self.n_row, c_keep, r_flipped)

        elif self.reorder_logic == 'col_to_row':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                reorder_index_map[old_idx] = _row_major_idx(self.n_col, r, c)

        elif self.reorder_logic == 'col_to_row_flipped':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                r_flipped = (self.n_row - 1) - r
                c_flipped = (self.n_col - 1) - c
                reorder_index_map[old_idx] = _row_major_idx(self.n_col, r_flipped, c_flipped)

        elif self.reorder_logic == 'col_to_row_c_flip_only':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                c_flipped = (self.n_col - 1) - c
                r_keep = r
                reorder_index_map[old_idx] = _row_major_idx(self.n_col, r_keep, c_flipped)

        elif self.reorder_logic == 'col_to_row_r_flip_only':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                r_flipped = (self.n_row - 1) - r
                c_keep = c
                reorder_index_map[old_idx] = _row_major_idx(self.n_col, r_flipped, c_keep)

        elif self.reorder_logic == 'vertical_flip':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                reorder_index_map[old_idx] = _column_major_idx(self.n_row, c, (self.n_row - 1) - r)

        elif self.reorder_logic == 'horizontal_flip':
            # This logic assumes the input is column-major and flips it left-to-right.
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                reorder_index_map[old_idx] = _column_major_idx(self.n_row, (self.n_col - 1) - c, r)

        elif self.reorder_logic == 'flip_and_rotate':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                new_c = (self.n_col - 1) - c
                reorder_index_map[old_idx] = _column_major_idx(self.n_row, new_c, r)

        elif self.reorder_logic == 'rotate_180':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                c_flipped = (self.n_col - 1) - c
                r_flipped = (self.n_row - 1) - r
                reorder_index_map[old_idx] = _column_major_idx(self.n_row, c_flipped, r_flipped)

        else:
            if self.reorder_logic:
                print(f"Warning: Unknown reorder_logic '{self.reorder_logic}'. No reordering applied.")
            return

        print(f"Applying reordering logic: {self.reorder_logic}")

        # Reorder point-based arrays
        new_points = np.zeros_like(self.points)
        new_normals = np.zeros_like(self.normals)
        new_points_origin = np.zeros_like(self.points_origin)
        new_array_positions = [None] * self.n_node
        new_colors_3d = np.zeros_like(self.colors_3d)

        for old_idx in range(self.n_node):
            new_idx = reorder_index_map[old_idx]
            new_points[new_idx], new_normals[new_idx] = self.points[old_idx], self.normals[old_idx]
            new_points_origin[new_idx] = self.points_origin[old_idx]
            new_array_positions[new_idx] = self.array_positions[old_idx]
            new_colors_3d[new_idx] = self.colors_3d[old_idx]

        self.points, self.normals, self.points_origin, self.array_positions, self.colors_3d = \
            new_points, new_normals, new_points_origin, new_array_positions, new_colors_3d

        # Update edges by remapping endpoint indices
        new_edges = np.copy(self.edges)
        for k in range(new_edges.shape[0]):
            new_edges[k, 1] = reorder_index_map[int(self.edges[k, 1])]
            new_edges[k, 2] = reorder_index_map[int(self.edges[k, 2])]
        self.edges = new_edges

    def _setup_visualization_data(self, major_order='column'):
        """Sets up colors and creates the final PolyData object for visualization."""
        if self._2D_map:
            self.colors = np.ones((self._2D_map.n_points, 4)) * 0.5

        is_2d_procedural_model = not self.mesh_file

        # Color the coarse and fine meshes
        for col in range(self.n_col):
            for row in range(self.n_row):
                idx = _column_major_idx(self.n_row, col, row)

                if is_2d_procedural_model:
                    # --- THIS IS THE MODIFIED LINE ---
                    color = [0.3, 0.3, 0.3, 1.0]  # A dimmer, darker gray
                else:
                    color = [col / self.n_col, row / self.n_row, 0.5, 1]

                self.colors_3d[idx] = color
                if self.colors is not None and idx < len(self.array_positions):
                    for k in self.array_positions[idx]:
                        self.colors[k] = color

        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges


class _FeatureDisabledProxy:
    """Fallback object used when an optional feature has not been initialized yet."""

    def __init__(self, feature_name, reason):
        self.feature_name = feature_name
        self.reason = reason

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            print(f"[{self.feature_name}] {self.reason}")
            return None

        return _noop

    def __bool__(self):
        return False


class _LazyFeatureProxy:
    """Instantiate an optional feature only when the UI actually uses it."""

    def __init__(self, owner, attr_name, feature_name, factory):
        self._owner = owner
        self._attr_name = attr_name
        self._feature_name = feature_name
        self._factory = factory

    def _resolve(self):
        current = getattr(self._owner, self._attr_name, self)
        if current is not self:
            return current

        try:
            helper = self._factory()
        except Exception as exc:
            print(f"[{self._feature_name}] Failed to initialize: {exc}")
            helper = _FeatureDisabledProxy(
                self._feature_name,
                f"Unavailable: {exc}",
            )

        setattr(self._owner, self._attr_name, helper)
        return helper

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __bool__(self):
        return True


class MySensor:
    DEFAULT_2D_GRID_SHAPE = (10, 10)
    VISUALIZATION_TARGET_HZ = 60.0
    SENSOR_AVERAGE_WINDOW_SIZE = 3
    AI_HELPER_ATTRS = {
        "record_gesture_class": "RecordGesture",
        "threelevel_hierarchical_transformer_class": "ThreeLevelTransformer",
        "proximity_control_class": "ProximityControl",
        "direct_finger_motion_class": "DirectFingerMotion",
        "console_control_class": "ConsoleControl",
        "ai_direct_finger_motion_class": "AI_DirectFingerMotion",
        "ai_direct_finger_motion_execution_class": "AI_DirectFingerMotion_execution",
    }
    AI_HELPER_IMPORTS = {
        "record_gesture_class": (
            "phd.dependence.gesture.gesture_logic_recording",
            "RecordGesture",
        ),
        "threelevel_hierarchical_transformer_class": (
            "phd.dependence.gesture.gesture_logic_three_level",
            "ThreeLevelTransformer",
        ),
        "proximity_control_class": (
            "phd.dependence.gesture.gesture_logic_proximity_control",
            "ProximityControl",
        ),
        "direct_finger_motion_class": (
            "phd.dependence.gesture.gesture_logic_direct_finger_motion",
            "DirectFingerMotion",
        ),
        "console_control_class": (
            "phd.dependence.gesture.gesture_logic_console",
            "ConsoleControl",
        ),
        "ai_direct_finger_motion_class": (
            "phd.dependence.gesture.gesture_logic_direct_finger_motion",
            "AI_DirectFingerMotion",
        ),
        "ai_direct_finger_motion_execution_class": (
            "phd.dependence.gesture.gesture_logic_direct_finger_motion",
            "AI_DirectFingerMotion_execution",
        ),
    }
    PREDEFINED_SENSOR_MODELS = {
        "elbow": {
            "n_row": 10,
            "n_col": 13,
            "mesh_file": sensor_resource_path("joint_1", "mesh.obj"),
            "signal_file": sensor_resource_path("joint_1", "signal.txt"),
            "offset_scale": 0.02,
        },
        "kuka": {
            "n_row": 10,
            "n_col": 8,
            "mesh_file": sensor_resource_path("kuka", "knitting_mesh_raw.obj"),
            "signal_file": sensor_resource_path("kuka", "signal.txt"),
            "reorder_logic": "row_to_col_flipped",
            "offset_scale": 0.2,
        },
        "double_curve": {
            "n_row": 10,
            "n_col": 10,
            "mesh_file": sensor_resource_path("dualC", "mesh.obj"),
            "signal_file": sensor_resource_path("dualC", "signal.txt"),
            "offset_scale": 0.2,
        },
        "half_cylinder_surface": {
            "n_row": 10,
            "n_col": 9,
            "mesh_file": sensor_resource_path("half_cylinder_surface", "half_cylinder_2.obj"),
            "signal_file": sensor_resource_path("half_cylinder_surface", "vertex_groups_2.txt"),
            "reorder_logic": "flip_and_rotate",
            "offset_scale": 0.2,
        },
    }
    SENSOR_MODEL_NAMES_BY_INDEX = {
        0: "2d",
        1: "elbow",
        2: "kuka",
        3: "double_curve",
        4: "half_cylinder_surface",
    }
    SENSOR_MODEL_LABELS = {
        "elbow": "Elbow",
        "kuka": "Kuka",
        "double_curve": "Double Curve",
        "2d": "2D",
        "half_cylinder_surface": "Half Cylinder Surface",
    }
    SENSOR_VISUALIZATION_MODE_OPTIONS = (
        ("point_grid", "Point Grid"),
        ("stereo_field", "Stereo Field"),
        ("heatmap_3d", "3D Heatmap"),
    )
    SENSOR_REFERENCE_OPACITY = 0.12

    def __init__(self, parent) -> None:
        self.parent = parent
        self.plotter: QtInteractor = self.parent.plotter_2
        self.actionMesh = None  # no mesh yet
        self.objActor = None
        self.matrixLineActor = None
        self.matrixLinePoly = None
        self.matrixLineColors = None
        self.heatmapActor = None
        self.heatmapPoly = None
        self.heatmapColors = None
        self.heatmapGridActor = None
        self.heatmapGridPoly = None
        self.heatmapGridDisplayPoly = None
        self._heatmap_logical_edges = None
        self._heatmap_tile_vertices = None
        self._last_sensor_visualization_matrix = None
        self._pending_sensor_visualization_matrix = None
        self._pending_multi_port_visualization_matrices = {}
        self._sensor_data_by_port = {}
        self._sensor_profiles_by_port = {}
        self._multi_port_sensor_views = {}
        self._multi_port_label_actors = []
        self._primary_sensor_port = None
        self._calibrated_sensor_ports = set()
        self._actor_name_prefix = ""
        self._heatmap_calibration_override = None
        self.heatmap_saturation_pct = DEFAULT_HEATMAP_SATURATION_PCT
        self.heatmap_noise_floor_pct = DEFAULT_HEATMAP_NOISE_FLOOR_PCT
        self.heatmap_response_mode = DEFAULT_HEATMAP_RESPONSE_MODE
        self.heatmap_proximity_noise_floor = DEFAULT_PROXIMITY_NOISE_FLOOR
        self.heatmap_proximity_knee = DEFAULT_PROXIMITY_KNEE
        self.heatmap_proximity_saturation = DEFAULT_PROXIMITY_SATURATION
        self.heatmap_3d_color_gain = DEFAULT_HEATMAP_3D_COLOR_GAIN
        self.heatmap_3d_palette = DEFAULT_HEATMAP_3D_PALETTE
        self.visualization_use_absolute_signal = True
        self.point_grid_response_mode = POINT_GRID_RESPONSE_ZERO_CENTERED
        self._matrix_visual_actor_mode = None
        self._matrix_line_dense_shape = (0, 0)
        self._matrix_line_base_points = None
        self._matrix_line_normals = None
        self._matrix_line_top_indices = None
        self._matrix_line_base_indices = None
        self._matrix_line_field_height = 0.0
        self._matrix_line_height_variation = None
        self._stereo_field_smoothed_visibility = None
        self._stereo_field_smoothed_color_response = None
        self.stereo_field_ignore_noise_enabled = True
        self.stereo_field_deadband_pct = 0.35
        self.stereo_field_response_scale_pct = 2.0
        self.stereo_field_length_scale = 0.35
        self.stereo_field_smoothing_alpha = 0.82
        self.contactNormalActor = None
        self.contactNormalMesh = None
        self._contact_normal_smoothed_start = None
        self._contact_normal_smoothed_direction = None
        self._contact_motion_previous_center = None
        self._contact_anchor_center = None
        self._contact_motion_smoothed_delta = None
        self.referenceAxisActors = []
        self.actorPlaneXY = None
        self.show_sensor_background_reference = True
        self._sensor_geometry_base_points_origin = None
        self._sensor_geometry_base_normals = None
        self._sensor_geometry_base_fine_points = None
        self.current_sensor_geometry_config = None
        self.sensor_visual_offset_scale = 0.0
        self._contact_normal_missing_frames = 0
        self.sensorPointLabelActor = None
        self.sensorSelectionActor = None
        self.sensorSelectionPoly = None
        self._selected_sensor_cell = None
        self.n_col = 0
        self.n_row = 0
        self.touch_sensitivity_scale = 0.05
        self.sensor_visualization_mode = "point_grid"
        self.show_contact_normal_vector = False
        self.show_sensor_point_labels = False
        self.contact_normal_estimator_mode = "touch_anchor_v4"
        self.contact_normal_threshold_pct = 3.0
        self.contact_normal_cluster_floor_pct = 0.8
        self.contact_normal_tilt_gain = 0.85
        self.contact_normal_residual_gain = 1.2
        self.contact_normal_residual_deadband = 0.06
        self.contact_normal_smoothing_alpha = 0.65
        self.contact_normal_missing_grace_frames = 4
        self.contact_motion_deadband_cells = 0.008
        self.contact_motion_smoothing_alpha = 0.68
        self.contact_motion_tilt_gain = 6.0
        self.contact_motion_max_tilt_deg = 90.0
        self.contact_force_scale_n_per_signal = 0.0
        self.cal_data = []
        self.sensor_transport = "serial"
        self._goodix_client = None
        self.sensor_average_window_size = self.SENSOR_AVERAGE_WINDOW_SIZE
        self.visualization_target_hz = self.VISUALIZATION_TARGET_HZ
        self.main_visualization_enabled = True
        self._heatmap_playback_active = False
        self.creatPlaneXY()
        # Frames are processed event-driven (see _SensorPayloadBridge): the
        # reader thread's signal triggers update_animation() the moment a
        # payload arrives. This timer is only a low-rate safety net so a
        # missed signal can never stall the pipeline; it must NOT run at 0 ms
        # (busy-waiting starves the reader thread via the GIL).
        self._payload_bridge = _SensorPayloadBridge(self)
        self._calibration_bridge = _SensorCalibrationBridge()
        self._calibration_bridge.finished.connect(self._finish_update_cal)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(100)
        self._visualization_timer = QTimer()
        self._visualization_timer.timeout.connect(
            self._render_pending_sensor_visualization
        )
        self._visualization_timer.start(
            max(1, int(round(1000.0 / self.visualization_target_hz)))
        )
        self.is_connected = False
        self.visualization_min_interval_sec = 1.0 / self.visualization_target_hz
        self._last_visualization_time = 0.0
        self._sensor_update_hz = 0.0
        self._sensor_update_tick_count = 0
        self._sensor_update_tick_started_at = time.perf_counter()
        self._visualization_hz = 0.0
        self._visualization_tick_count = 0
        self._visualization_tick_started_at = time.perf_counter()
        self.initChannel()
        self.ser_list = []
        self._sensor_read_thread = None
        self._sensor_read_worker = None
        self._latest_sensor_payloads = {}
        self._last_sensor_api_payloads = {}
        self._sensor_reader_generation = 0
        self._last_sensor_reader_error_log_time = 0.0
        self._sensor_calibration_in_progress = False
        self._sensor_calibration_thread = None
        self._sensor_calibration_stop_event = threading.Event()
        self._is_shutting_down = False
        self._last_sensor_stream_error = ""
        self.current_model_name = None
        self.current_reorder_mode = REORDER_FACTORY_DEFAULT
        self.current_effective_reorder_logic = None
        self.current_sensor_reorder_key = None
        self.cell_zero_mask = np.zeros((0, 0), dtype=bool)

        # Delay AI helper/model creation until a sensor model is selected in buildScene().
        # This avoids startup warnings/errors caused by initializing model-dependent logic
        # with n_row = 0 and n_col = 0.
        self.reset_ai_helpers()

    def saveCameraPara(self):
        self.camera_pos = self.plotter.camera.position
        self.camera_focal = self.plotter.camera.focal_point
        self.camera_view_angle = self.plotter.camera.view_angle

    def loadCameraPare(self, camera_pos=None, camera_focal=None, camera_view_angle=None):
        if camera_pos is None:
            camera_pos = self.camera_pos
        if camera_focal is None:
            camera_focal = self.camera_focal
        if camera_view_angle is None:
            camera_view_angle = self.camera_view_angle

        self.plotter.camera.position = camera_pos
        self.plotter.camera.focal_point = camera_focal
        self.plotter.camera.view_angle = camera_view_angle

    @staticmethod
    def _normalize_sensor_camera_config(config):
        if not isinstance(config, dict):
            return None

        def _finite_vector(name):
            try:
                vector = np.asarray(config.get(name), dtype=float).reshape(-1)
            except Exception:
                return None
            if vector.size != 3 or not np.all(np.isfinite(vector)):
                return None
            return vector

        position = _finite_vector("position")
        focal_point = _finite_vector("focal_point")
        view_up = _finite_vector("view_up")
        if position is None or focal_point is None or view_up is None:
            return None
        view_direction = focal_point - position
        if (
            float(np.linalg.norm(view_direction)) <= 1e-9
            or float(np.linalg.norm(view_up)) <= 1e-9
            or float(np.linalg.norm(np.cross(view_direction, view_up))) <= 1e-9
        ):
            return None

        try:
            view_angle = float(config.get("view_angle", 30.0))
        except Exception:
            view_angle = 30.0
        try:
            parallel_scale = float(config.get("parallel_scale", 1.0))
        except Exception:
            parallel_scale = 1.0
        if not np.isfinite(view_angle):
            view_angle = 30.0
        if not np.isfinite(parallel_scale) or parallel_scale <= 0.0:
            parallel_scale = 1.0

        return {
            "position": position.tolist(),
            "focal_point": focal_point.tolist(),
            "view_up": (view_up / np.linalg.norm(view_up)).tolist(),
            "view_angle": float(np.clip(view_angle, 1.0, 179.0)),
            "parallel_projection": bool(
                config.get("parallel_projection", False)
            ),
            "parallel_scale": parallel_scale,
        }

    def capture_sensor_camera_config(self):
        camera = getattr(self.plotter, "camera", None)
        if camera is None:
            return None
        try:
            config = {
                "position": list(camera.GetPosition()),
                "focal_point": list(camera.GetFocalPoint()),
                "view_up": list(camera.GetViewUp()),
                "view_angle": float(camera.GetViewAngle()),
                "parallel_projection": bool(camera.GetParallelProjection()),
                "parallel_scale": float(camera.GetParallelScale()),
            }
        except Exception:
            try:
                config = {
                    "position": list(camera.position),
                    "focal_point": list(camera.focal_point),
                    "view_up": list(camera.up),
                    "view_angle": float(camera.view_angle),
                    "parallel_projection": bool(camera.parallel_projection),
                    "parallel_scale": float(camera.parallel_scale),
                }
            except Exception:
                return None
        return self._normalize_sensor_camera_config(config)

    def apply_sensor_camera_config(self, config, render=True):
        camera_config = self._normalize_sensor_camera_config(config)
        camera = getattr(self.plotter, "camera", None)
        if camera_config is None or camera is None:
            return False
        try:
            camera.SetPosition(*camera_config["position"])
            camera.SetFocalPoint(*camera_config["focal_point"])
            camera.SetViewUp(*camera_config["view_up"])
            camera.SetViewAngle(camera_config["view_angle"])
            camera.SetParallelProjection(
                int(camera_config["parallel_projection"])
            )
            camera.SetParallelScale(camera_config["parallel_scale"])
        except Exception:
            try:
                camera.position = camera_config["position"]
                camera.focal_point = camera_config["focal_point"]
                camera.up = camera_config["view_up"]
                camera.view_angle = camera_config["view_angle"]
                camera.parallel_projection = camera_config[
                    "parallel_projection"
                ]
                camera.parallel_scale = camera_config["parallel_scale"]
            except Exception:
                return False

        self.saveCameraPara()
        try:
            self.plotter.reset_camera_clipping_range()
        except Exception:
            pass
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass
        return True

    def creatPlaneXY(self):
        self.plotter.camera.position = (1, -1, 1)
        self.saveCameraPara()
        self.referenceAxisActors = []

        # X-axis line
        line_x = pv.Line((-50, 0, 0), (50, 0, 0))
        self.referenceAxisActors.append(
            self.plotter.add_mesh(line_x, color='r', line_width=2, label='X Axis')
        )

        # Y-axis line
        line_y = pv.Line((0, -50, 0), (0, 50, 0))
        self.referenceAxisActors.append(
            self.plotter.add_mesh(line_y, color='g', line_width=2, label='Y Axis')
        )

        # Z-axis line
        # line_z = pv.Line((0, 0, -50), (0, 0, 50))
        # self.referenceAxisActors.append(
        #     self.plotter.add_mesh(line_z, color='b', line_width=2, label='Z Axis')
        # )

        planeXY = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=100,
            j_size=100,
            i_resolution=100,
            j_resolution=100,
        )
        self.actorPlaneXY = self.plotter.add_mesh(planeXY, color='gray', style='wireframe')
        self._apply_sensor_background_reference_visibility(render=False)

    def initChannel(self):
        self.com_options = []
        ports = [
            port
            for port in serial.tools.list_ports.comports()
            if not str(port.name or "").lower().startswith("ttys")
        ]
        self.ser = None
        self.parent.serial_channel.clear()
        if ports:
            humanoid_assignments = load_humanoid_device_assignments()
            # Sort ports to have 'ttyACM' first, then others.
            ports = sorted(ports, key=lambda port: (0, int(port.name.replace('ttyACM', ''))) if port.name.startswith(
                'ttyACM') else (1, port.name))

            acm_ports = [port for port in ports if port.name.startswith("ttyACM")]
            other_ports = [port for port in ports if not port.name.startswith("ttyACM")]
            for port in acm_ports:
                self.com_options.append(port.name)
                # Create a standard, selectable item (no checkbox).
                # Selection mode is already handled in ui_ping.py.
                annotation = humanoid_sensor_annotation(
                    port,
                    humanoid_assignments,
                )
                label = (
                    f"{port.name} - {annotation}"
                    if annotation
                    else port.name
                )
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, port.name)
                self.parent.serial_channel.addItem(item)
        else:
            other_ports = []
        # Keep Goodix immediately after the ACM sensor ports and before other
        # USB serial adapters.
        if goodix_usb_connected():
            self.com_options.append(GOODIX_USB_SOURCE_ID)
            item = QListWidgetItem(GOODIX_USB_SOURCE_LABEL)
            item.setData(Qt.UserRole, GOODIX_USB_SOURCE_ID)
            self.parent.serial_channel.addItem(item)
        for port in other_ports:
            self.com_options.append(port.name)
            item = QListWidgetItem(port.name)
            item.setData(Qt.UserRole, port.name)
            self.parent.serial_channel.addItem(item)
        if self.parent.serial_channel.count() > 0:
            first_item = self.parent.serial_channel.item(0)
            self.parent.serial_channel.setCurrentRow(0)
            first_item.setSelected(True)
        else:
            print("No serial ports found. Please connect the device and retry.")
            return

    def reset_ai_helpers(self, reason="Build the scene first to initialize AI features."):
        for attr_name, feature_name in self.AI_HELPER_ATTRS.items():
            if attr_name == "console_control_class":
                continue
            setattr(self, attr_name, _FeatureDisabledProxy(feature_name, reason))
        self.console_control_class = self._make_lazy_helper(
            "console_control_class",
            "ConsoleControl",
            lambda: self._construct_ai_helper("console_control_class"),
        )

    def _safe_create_helper(self, feature_name, factory):
        try:
            return factory()
        except Exception as exc:
            print(f"[{feature_name}] Failed to initialize: {exc}")
            return _FeatureDisabledProxy(feature_name, f"Unavailable: {exc}")

    def _make_lazy_helper(self, attr_name, feature_name, factory):
        return _LazyFeatureProxy(self, attr_name, feature_name, factory)

    def get_initialized_helper(self, attr_name, default=None):
        """Return an already-created optional helper without resolving a proxy."""
        helper = self.__dict__.get(str(attr_name), default)
        if isinstance(helper, _LazyFeatureProxy):
            return default
        return helper

    def _load_ai_helper_class(self, attr_name):
        module_name, symbol_name = self.AI_HELPER_IMPORTS[attr_name]
        module = import_module(module_name)
        return getattr(module, symbol_name)

    def _construct_ai_helper(self, attr_name):
        helper_class = self._load_ai_helper_class(attr_name)
        if attr_name == "record_gesture_class":
            return helper_class(self)
        if attr_name == "threelevel_hierarchical_transformer_class":
            return helper_class(self.parent, self, self.n_row, self.n_col)
        return helper_class(self.parent, self)

    def _ai_helper_factories(self):
        return {
            attr_name: (
                lambda attr_name=attr_name: self._construct_ai_helper(attr_name)
            )
            for attr_name in self.AI_HELPER_ATTRS
        }

    def initialize_ai_helpers(self):
        if self.n_row <= 0 or self.n_col <= 0:
            self.reset_ai_helpers()
            return

        factories = self._ai_helper_factories()
        for attr_name, feature_name in self.AI_HELPER_ATTRS.items():
            setattr(
                self,
                attr_name,
                self._make_lazy_helper(attr_name, feature_name, factories[attr_name]),
            )

    def get_ai_direct_finger_motion_execution_default_model_path(self):
        helper = getattr(self, "ai_direct_finger_motion_execution_class", None)
        path = getattr(helper, "model_checkpoint_path", "") if helper is not None else ""
        if isinstance(path, str):
            return path
        return ""

    def _record_sensor_update_tick(self):
        self._sensor_update_tick_count += 1
        now = time.perf_counter()
        elapsed = now - self._sensor_update_tick_started_at
        if elapsed >= 1.0:
            self._sensor_update_hz = self._sensor_update_tick_count / elapsed
            self._sensor_update_tick_count = 0
            self._sensor_update_tick_started_at = now

    def _should_refresh_visualization(self):
        now = time.perf_counter()
        if (now - self._last_visualization_time) < self.visualization_min_interval_sec:
            return False
        self._last_visualization_time = now
        return True

    def _record_visualization_tick(self):
        self._visualization_tick_count += 1
        now = time.perf_counter()
        elapsed = now - self._visualization_tick_started_at
        if elapsed >= 1.0:
            self._visualization_hz = self._visualization_tick_count / elapsed
            self._visualization_tick_count = 0
            self._visualization_tick_started_at = now

    def get_sensor_average_window_size(self):
        data_obj = getattr(self, "_data", None)
        if data_obj is None:
            return int(self.sensor_average_window_size)
        return int(getattr(data_obj, "windowSize", self.sensor_average_window_size))

    def set_sensor_average_window_size(self, window_size):
        self.sensor_average_window_size = max(1, int(window_size))
        data_objects = list(
            getattr(self, "_sensor_data_by_port", {}).values()
        )
        primary_data = getattr(self, "_data", None)
        if primary_data is not None and not data_objects:
            data_objects.append(primary_data)
        seen = set()
        for data_obj in data_objects:
            if data_obj is None or id(data_obj) in seen:
                continue
            seen.add(id(data_obj))
            data_obj.setWindowSize(self.sensor_average_window_size)

    def get_visualization_target_hz(self):
        return float(self.visualization_target_hz)

    def set_visualization_target_hz(self, hz):
        self.visualization_target_hz = max(1.0, float(hz))
        self.visualization_min_interval_sec = 1.0 / self.visualization_target_hz
        timer = getattr(self, "_visualization_timer", None)
        if timer is not None:
            timer.setInterval(
                max(1, int(round(1000.0 / self.visualization_target_hz)))
            )

    def _render_pending_sensor_visualization(self):
        if self._is_shutting_down or not self.is_connected:
            return
        if not bool(getattr(self, "main_visualization_enabled", True)):
            return
        if bool(getattr(self, "_heatmap_playback_active", False)):
            return
        sensor_matrix = self._pending_sensor_visualization_matrix
        secondary_matrices = dict(
            getattr(self, "_pending_multi_port_visualization_matrices", {})
        )
        if sensor_matrix is None and not secondary_matrices:
            return
        self._pending_sensor_visualization_matrix = None
        pending_secondary = getattr(
            self, "_pending_multi_port_visualization_matrices", None
        )
        if pending_secondary is not None:
            pending_secondary.clear()
        self._last_visualization_time = time.perf_counter()
        self.saveCameraPara()
        for port_name, matrix in secondary_matrices.items():
            view = self._multi_port_sensor_views.get(port_name)
            if view is not None:
                view.update_visualization(matrix, render=False)
        if sensor_matrix is not None:
            self.update_visualization(sensor_matrix)
        elif secondary_matrices:
            self.plotter.render()

    def _sensor_actor_name(self, base_name):
        prefix = str(getattr(self, "_actor_name_prefix", "") or "").strip()
        return str(base_name) if not prefix else f"{base_name}_{prefix}"

    def _ensure_main_sensor_visualization_actors(self):
        """Create main-plotter actors after an external-only scene build."""
        sensor_map = getattr(self, "_2D_map", None)
        if (
            getattr(self, "objActor", None) is None
            and sensor_map is not None
            and int(getattr(sensor_map, "n_points", 0) or 0) > 0
        ):
            self.objActor = self.plotter.add_mesh(
                sensor_map,
                show_edges=True,
                scalars=self.colors,
                rgb=True,
                opacity=self.SENSOR_REFERENCE_OPACITY,
                name=self._sensor_actor_name("sensor_reference"),
            )

        if (
            getattr(self, "actionMesh", None) is None
            and getattr(self, "line_poly", None) is not None
        ):
            self.actionMesh = self.plotter.add_mesh(
                self.line_poly,
                scalars=self.colors_3d,
                point_size=10,
                line_width=3,
                render_points_as_spheres=True,
                rgb=True,
                name=self._sensor_actor_name("sensor_point_grid"),
            )

        if self.actionMesh is not None or self.matrixLineActor is not None:
            self._rebuild_matrix_visualization_actor(render=False)
        if self._is_heatmap_visualization_mode(
            getattr(self, "sensor_visualization_mode", "point_grid")
        ):
            self._ensure_heatmap_visualization_actor(render=False)
        self._refresh_sensor_point_label_actor()

    def set_main_visualization_enabled(self, enabled: bool, render: bool = True):
        """Enable/disable only the main sensor plotter rendering.

        Sensor acquisition and frame processing continue while disabled. This
        lets another view (for example the robot-model sensor overlay) consume
        the same single serial stream without rendering the sensor twice.
        """
        enabled = bool(enabled)
        self.main_visualization_enabled = enabled
        if enabled:
            visualization_was_built = any(
                actor is not None
                for actor in (
                    getattr(self, "objActor", None),
                    getattr(self, "actionMesh", None),
                    getattr(self, "matrixLineActor", None),
                    getattr(self, "heatmapActor", None),
                )
            )
            self._ensure_main_sensor_visualization_actors()
            self._refresh_sensor_visualization_mode_actors()
            if not visualization_was_built:
                self.restore_saved_sensor_perspective(render=False)
            self._apply_sensor_background_reference_visibility(render=False)
            self._set_actor_visible(
                getattr(self, "contactNormalActor", None),
                bool(getattr(self, "show_contact_normal_vector", False)),
            )
            self._set_actor_visible(
                getattr(self, "sensorPointLabelActor", None),
                bool(getattr(self, "show_sensor_point_labels", False)),
            )
        else:
            for actor in (
                getattr(self, "objActor", None),
                getattr(self, "actionMesh", None),
                getattr(self, "matrixLineActor", None),
                    getattr(self, "heatmapActor", None),
                    getattr(self, "contactNormalActor", None),
                    getattr(self, "sensorPointLabelActor", None),
                    getattr(self, "sensorSelectionActor", None),
                    getattr(self, "actorPlaneXY", None),
            ):
                self._set_actor_visible(actor, False)
            for actor in getattr(self, "referenceAxisActors", []) or []:
                self._set_actor_visible(actor, False)
            mesh_functions = getattr(
                getattr(self, "parent", None), "mesh_functions", None
            )
            if mesh_functions is not None and hasattr(
                mesh_functions, "set_secondary_background_reference_enabled"
            ):
                try:
                    mesh_functions.set_secondary_background_reference_enabled(
                        False, render=False
                    )
                except Exception:
                    pass
        for view in getattr(self, "_multi_port_sensor_views", {}).values():
            view.set_replica_visibility(enabled)
        for actor in getattr(self, "_multi_port_label_actors", []):
            self._set_actor_visible(actor, enabled)
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def get_last_sensor_stream_error(self):
        return str(getattr(self, "_last_sensor_stream_error", "") or "")

    def _set_sensor_stream_error(self, message):
        self._last_sensor_stream_error = str(message or "")

    def start_external_visualization_stream(self):
        """Build/calibrate one sensor stream without main-plotter rendering."""
        self._set_sensor_stream_error("")
        if self._sensor_calibration_in_progress:
            return True
        if self._sensor_reader_is_running() and self.is_connected:
            self.set_main_visualization_enabled(False, render=True)
            return True

        self.buildScene(show_main_visualization=False)
        if not self.ser_list or self._data is None:
            return False
        self.updateCal()
        return True

    def stop_external_visualization_stream(self):
        """Stop a stream started for an external visualization."""
        if self._sensor_calibration_in_progress:
            return False
        stopped = self._close_serial_ports()
        if stopped:
            self.is_connected = False
        return bool(stopped)

    def get_sensor_visualization_modes(self):
        return list(self.SENSOR_VISUALIZATION_MODE_OPTIONS)

    def set_sensor_visualization_mode(self, mode):
        valid_modes = {key for key, _label in self.SENSOR_VISUALIZATION_MODE_OPTIONS}
        if mode not in valid_modes:
            mode = "point_grid"
        previous_mode = str(getattr(self, "sensor_visualization_mode", "point_grid"))
        self.sensor_visualization_mode = mode
        if mode != previous_mode:
            self._stereo_field_smoothed_visibility = None
            self._stereo_field_smoothed_color_response = None
        if (
            self._is_matrix_visualization_mode(mode)
            and (
                self.matrixLineActor is None
                or getattr(self, "_matrix_visual_actor_mode", None) != mode
            )
        ):
            self._rebuild_matrix_visualization_actor(render=False)
        if self._is_heatmap_visualization_mode(mode):
            self._ensure_heatmap_visualization_actor(render=False)
            current_matrix = getattr(self, "_last_sensor_visualization_matrix", None)
            if current_matrix is None:
                current_matrix = self._current_heatmap_sensor_matrix()
            if current_matrix is not None:
                self._update_heatmap_visualization(current_matrix)
        self._refresh_sensor_visualization_mode_actors()
        for view in getattr(self, "_multi_port_sensor_views", {}).values():
            view.set_replica_visualization_mode(mode)
        try:
            self.plotter.render()
        except Exception:
            pass

    @staticmethod
    def _is_matrix_visualization_mode(mode):
        return str(mode) == "stereo_field"

    @staticmethod
    def _is_heatmap_visualization_mode(mode):
        return str(mode) == "heatmap_3d"

    @staticmethod
    def _set_actor_visible(actor, visible):
        if actor is None:
            return
        try:
            actor.SetVisibility(1 if visible else 0)
            return
        except Exception:
            pass
        try:
            actor.visibility = bool(visible)
        except Exception:
            pass

    def _refresh_sensor_visualization_mode_actors(self):
        mode = str(getattr(self, "sensor_visualization_mode", "point_grid"))
        enabled = bool(getattr(self, "main_visualization_enabled", True))
        point_grid_mode = enabled and mode == "point_grid"
        matrix_mode = enabled and self._is_matrix_visualization_mode(mode)
        heatmap_mode = enabled and self._is_heatmap_visualization_mode(mode)
        self._set_actor_visible(getattr(self, "objActor", None), point_grid_mode)
        self._set_actor_visible(getattr(self, "actionMesh", None), point_grid_mode)
        self._set_actor_visible(getattr(self, "matrixLineActor", None), matrix_mode)
        self._set_actor_visible(getattr(self, "heatmapActor", None), heatmap_mode)
        self._set_actor_visible(
            getattr(self, "heatmapGridActor", None), heatmap_mode
        )
        self._set_actor_visible(
            getattr(self, "sensorSelectionActor", None),
            enabled and getattr(self, "_selected_sensor_cell", None) is not None,
        )

    @staticmethod
    def _heatmap_axis_half_vector(point_grid, row, col, axis):
        rows, cols = point_grid.shape[:2]
        if axis == 0 and rows > 1:
            if 0 < row < rows - 1:
                return (point_grid[row + 1, col] - point_grid[row - 1, col]) * 0.25
            neighbour = row + 1 if row == 0 else row - 1
            direction = 1.0 if row == 0 else -1.0
            return (point_grid[neighbour, col] - point_grid[row, col]) * 0.5 * direction
        if axis == 1 and cols > 1:
            if 0 < col < cols - 1:
                return (point_grid[row, col + 1] - point_grid[row, col - 1]) * 0.25
            neighbour = col + 1 if col == 0 else col - 1
            direction = 1.0 if col == 0 else -1.0
            return (point_grid[row, neighbour] - point_grid[row, col]) * 0.5 * direction
        return np.zeros(3, dtype=float)

    @staticmethod
    def _heatmap_tangent_vector(vector, normal):
        vector_np = np.asarray(vector, dtype=float)
        normal_np = np.asarray(normal, dtype=float)
        tangent = vector_np - normal_np * float(np.dot(vector_np, normal_np))
        if np.all(np.isfinite(tangent)):
            return tangent
        return np.zeros(3, dtype=float)

    def _custom_heatmap_grid_vectors(self):
        """Return independently edited heatmap centres and normals, if enabled."""
        geometry = self._normalize_sensor_geometry_config(
            getattr(self, "current_sensor_geometry_config", None)
        )
        if not bool(geometry.get("use_custom_heatmap_shape", False)):
            return None, None

        expected_count = int(self.n_row) * int(self.n_col)
        local_points = normalize_point_array(
            geometry.get("custom_heatmap_points", []), expected_count
        )
        if local_points is None:
            return None, None

        effective_geometry = self._effective_sensor_geometry_config(geometry)
        local_sensor_points, _local_sensor_normals = self._sensor_local_geometry(
            effective_geometry
        )
        if local_sensor_points is None:
            return None, None
        local_normals = compute_grid_normals(
            local_points,
            self.n_row,
            self.n_col,
            normal_flip=bool(effective_geometry.get("normal_flip", False)),
        )
        if local_normals is None:
            return None, None

        rotation_pivot = self._sensor_geometry_rotation_pivot(local_sensor_points)
        world_points, world_normals = self._rotate_sensor_geometry(
            local_points,
            effective_geometry.get("rotation_deg", [0.0, 0.0, 0.0]),
            normals=local_normals,
            pivot=rotation_pivot,
        )
        return (
            self._coarse_grid_vectors(world_points),
            self._coarse_grid_vectors(world_normals),
        )

    def _custom_heatmap_corner_grid(self):
        """Return a rotated shared-corner heatmap grid, if one is enabled."""
        geometry = self._normalize_sensor_geometry_config(
            getattr(self, "current_sensor_geometry_config", None)
        )
        if not bool(geometry.get("use_custom_heatmap_shape", False)):
            return None

        corner_rows = int(self.n_row) + 1
        corner_cols = int(self.n_col) + 1
        local_corners = normalize_point_array(
            geometry.get("custom_heatmap_corners", []),
            corner_rows * corner_cols,
        )
        if local_corners is None:
            return None

        effective_geometry = self._effective_sensor_geometry_config(geometry)
        local_sensor_points, _local_sensor_normals = self._sensor_local_geometry(
            effective_geometry
        )
        if local_sensor_points is None:
            return None
        rotation_pivot = self._sensor_geometry_rotation_pivot(local_sensor_points)
        world_corners = self._rotate_sensor_geometry(
            local_corners,
            effective_geometry.get("rotation_deg", [0.0, 0.0, 0.0]),
            pivot=rotation_pivot,
        )
        corner_grid = np.empty((corner_rows, corner_cols, 3), dtype=float)
        for col in range(corner_cols):
            for row in range(corner_rows):
                corner_grid[row, col] = world_corners[col * corner_rows + row]
        return corner_grid

    def _custom_heatmap_curve_offsets_world(self):
        geometry = self._normalize_sensor_geometry_config(
            getattr(self, "current_sensor_geometry_config", None)
        )
        if not bool(geometry.get("use_curved_heatmap_edges", False)):
            return None
        edge_count = len(
            structured_grid_edge_pairs(
                int(self.n_row) + 1,
                int(self.n_col) + 1,
            )
        )
        local_offsets = normalize_point_array(
            geometry.get("custom_heatmap_edge_offsets", []), edge_count
        )
        if local_offsets is None:
            local_offsets = np.zeros((edge_count, 3), dtype=float)
        rotation = self._sensor_geometry_rotation_matrix_deg(
            self._effective_sensor_geometry_config(geometry).get(
                "rotation_deg", [0.0, 0.0, 0.0]
            )
        )
        return local_offsets @ rotation.T

    def _build_connected_heatmap_polydata(self, corner_grid):
        """Build taxel quads whose neighboring edges share exact coordinates."""
        corner_rows = int(self.n_row) + 1
        corner_cols = int(self.n_col) + 1
        corner_points = np.asarray(
            [
                corner_grid[row, col]
                for col in range(corner_cols)
                for row in range(corner_rows)
            ],
            dtype=float,
        )
        edge_offsets = self._custom_heatmap_curve_offsets_world()
        surface = heatmap_surface_from_corner_lattice(
            corner_points,
            corner_rows,
            corner_cols,
            edge_offsets=edge_offsets,
            curve_resolution=6 if edge_offsets is not None else 1,
        )
        if surface is None:
            return None, None, None
        tile_points = surface["points"]
        faces = surface["faces"]
        tile_vertices = surface["tile_vertices"]

        logical_edges = []
        for pair, samples in zip(
            surface["edge_pairs"], surface["edge_samples"]
        ):
            first, second = (int(pair[0]), int(pair[1]))
            first_col, first_row = divmod(first, corner_rows)
            second_col, second_row = divmod(second, corner_rows)
            adjacent_cells = []
            if first_col == second_col:
                edge_col = first_col
                edge_row = min(first_row, second_row)
                candidates = (
                    (edge_row, edge_col - 1),
                    (edge_row, edge_col),
                )
            else:
                edge_col = min(first_col, second_col)
                edge_row = first_row
                candidates = (
                    (edge_row - 1, edge_col),
                    (edge_row, edge_col),
                )
            for row, col in candidates:
                if 0 <= row < self.n_row and 0 <= col < self.n_col:
                    adjacent_cells.append((row, col))
            logical_edges.append(
                {
                    "points": np.asarray(samples, dtype=float),
                    "cells": adjacent_cells,
                }
            )
        self._heatmap_logical_edges = logical_edges

        poly = pv.PolyData(
            np.asarray(tile_points, dtype=float),
            np.asarray(faces, dtype=np.int_),
        )
        colors = np.full((len(tile_points), 4), 255, dtype=np.uint8)
        mask = self._cell_zero_mask_for_plotter()
        for col in range(self.n_col):
            for row in range(self.n_row):
                if mask[row, col]:
                    colors[tile_vertices[row, col], 3] = 0
        poly.point_data["heatmap_colors"] = colors
        poly.set_active_scalars("heatmap_colors")
        return poly, colors, tile_vertices

    def _build_visible_heatmap_grid_polydata(self):
        """Return grid edges belonging to at least one unmasked taxel."""
        heatmap_poly = getattr(self, "heatmapPoly", None)
        if heatmap_poly is None:
            return pv.PolyData()
        mask = self._cell_zero_mask_for_plotter()
        logical_edges = getattr(self, "_heatmap_logical_edges", None)
        if logical_edges:
            line_points = []
            lines = []
            for edge in logical_edges:
                if not any(
                    not bool(mask[row, col])
                    for row, col in edge.get("cells", [])
                ):
                    continue
                samples = np.asarray(edge.get("points", []), dtype=float)
                if samples.ndim != 2 or samples.shape[0] < 2:
                    continue
                first_index = len(line_points)
                line_points.extend(samples)
                lines.extend(
                    [
                        len(samples),
                        *range(first_index, first_index + len(samples)),
                    ]
                )
            if not line_points:
                return pv.PolyData()
            grid_poly = pv.PolyData(np.asarray(line_points, dtype=float))
            grid_poly.lines = np.asarray(lines, dtype=np.int_)
            return grid_poly

        visible_cell_ids = [
            col * int(self.n_row) + row
            for col in range(int(self.n_col))
            for row in range(int(self.n_row))
            if not bool(mask[row, col])
        ]
        if not visible_cell_ids:
            return pv.PolyData()
        visible_surface = heatmap_poly.extract_cells(
            np.asarray(visible_cell_ids, dtype=int)
        ).extract_surface()
        return visible_surface.clean().extract_all_edges()

    def _build_heatmap_polydata(self):
        self._heatmap_logical_edges = None
        custom_corner_grid = self._custom_heatmap_corner_grid()
        if custom_corner_grid is not None:
            return self._build_connected_heatmap_polydata(custom_corner_grid)

        point_grid = self._coarse_grid_vectors(getattr(self, "points_origin", None))
        normal_grid = self._coarse_grid_vectors(getattr(self, "normals", None))
        custom_point_grid, custom_normal_grid = self._custom_heatmap_grid_vectors()
        if custom_point_grid is not None and custom_normal_grid is not None:
            point_grid = custom_point_grid
            normal_grid = custom_normal_grid
        if point_grid is None or normal_grid is None:
            return None, None, None

        finite_points = point_grid[np.all(np.isfinite(point_grid), axis=2)]
        if finite_points.size:
            sensor_span = float(
                np.linalg.norm(np.max(finite_points, axis=0) - np.min(finite_points, axis=0))
            )
        else:
            sensor_span = 0.05
        fallback_half_size = max(sensor_span * 0.025, 0.001)
        surface_lift = max(sensor_span * 0.002, 0.0001)
        tile_scale = 0.94

        tile_points = []
        faces = []
        tile_vertices = np.empty((self.n_row, self.n_col, 4), dtype=int)
        for col in range(self.n_col):
            for row in range(self.n_row):
                center = np.asarray(point_grid[row, col], dtype=float)
                normal = self._normalize_vector(normal_grid[row, col])
                row_half = self._heatmap_tangent_vector(
                    self._heatmap_axis_half_vector(point_grid, row, col, axis=0),
                    normal,
                )
                col_half = self._heatmap_tangent_vector(
                    self._heatmap_axis_half_vector(point_grid, row, col, axis=1),
                    normal,
                )

                row_norm = float(np.linalg.norm(row_half))
                col_norm = float(np.linalg.norm(col_half))
                if row_norm <= 1e-9 and col_norm > 1e-9:
                    row_half = self._normalize_vector(
                        np.cross(normal, col_half)
                    ) * max(col_norm, fallback_half_size)
                elif col_norm <= 1e-9 and row_norm > 1e-9:
                    col_half = self._normalize_vector(
                        np.cross(row_half, normal)
                    ) * max(row_norm, fallback_half_size)
                elif row_norm <= 1e-9 and col_norm <= 1e-9:
                    reference = np.array([0.0, 0.0, 1.0], dtype=float)
                    if abs(float(np.dot(normal, reference))) > 0.9:
                        reference = np.array([0.0, 1.0, 0.0], dtype=float)
                    col_half = self._normalize_vector(
                        np.cross(reference, normal)
                    ) * fallback_half_size
                    row_half = self._normalize_vector(
                        np.cross(normal, col_half)
                    ) * fallback_half_size

                row_half *= tile_scale
                col_half *= tile_scale
                lifted_center = center + normal * surface_lift
                corners = (
                    lifted_center - row_half - col_half,
                    lifted_center - row_half + col_half,
                    lifted_center + row_half + col_half,
                    lifted_center + row_half - col_half,
                )
                first_vertex = len(tile_points)
                tile_points.extend(corners)
                vertex_ids = np.arange(first_vertex, first_vertex + 4, dtype=int)
                tile_vertices[row, col] = vertex_ids
                faces.extend([4, *vertex_ids.tolist()])

        poly = pv.PolyData(
            np.asarray(tile_points, dtype=float),
            np.asarray(faces, dtype=np.int_),
        )
        colors = np.full((len(tile_points), 4), 255, dtype=np.uint8)
        mask = self._cell_zero_mask_for_plotter()
        for col in range(self.n_col):
            for row in range(self.n_row):
                if mask[row, col]:
                    colors[tile_vertices[row, col], 3] = 0
        poly.point_data["heatmap_colors"] = colors
        poly.set_active_scalars("heatmap_colors")
        return poly, colors, tile_vertices

    def _rebuild_heatmap_visualization_actor(self, render=False):
        self.heatmapActor = self._remove_actor_safely(
            getattr(self, "heatmapActor", None)
        )
        self.heatmapGridActor = self._remove_actor_safely(
            getattr(self, "heatmapGridActor", None)
        )
        self.sensorSelectionActor = self._remove_actor_safely(
            getattr(self, "sensorSelectionActor", None)
        )
        self.sensorSelectionPoly = None
        self.heatmapPoly = None
        self.heatmapColors = None
        self.heatmapGridPoly = None
        self.heatmapGridDisplayPoly = None
        self._heatmap_logical_edges = None
        self._heatmap_tile_vertices = None

        poly, colors, tile_vertices = self._build_heatmap_polydata()
        if poly is None:
            return
        self.heatmapPoly = poly
        self.heatmapColors = colors
        self._heatmap_tile_vertices = tile_vertices
        try:
            self.heatmapActor = self.plotter.add_mesh(
                self.heatmapPoly,
                scalars="heatmap_colors",
                rgb=True,
                preference="point",
                show_edges=False,
                smooth_shading=False,
                lighting=False,
                ambient=1.0,
                name=self._sensor_actor_name("sensor_heatmap_3d"),
                render=False,
            )
        except Exception as exc:
            self.heatmapActor = None
            self.heatmapPoly = None
            self.heatmapColors = None
            self._heatmap_tile_vertices = None
            print(f"[SensorVisualization] Failed to build 3D heatmap: {exc}")
            return
        self._rebuild_heatmap_grid_actor(render=render)

    def _rebuild_heatmap_grid_actor(self, render=False):
        self.heatmapGridActor = self._remove_actor_safely(
            getattr(self, "heatmapGridActor", None)
        )
        self.heatmapGridPoly = self._build_visible_heatmap_grid_polydata()
        self.heatmapGridDisplayPoly = None
        if self.heatmapGridPoly.n_lines <= 0:
            return
        try:
            bounds = np.asarray(self.heatmapGridPoly.bounds, dtype=float)
            span = float(
                np.linalg.norm(
                    [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
                )
            )
            tube_radius = max(span * 0.0015, 1e-6)
            self.heatmapGridDisplayPoly = self.heatmapGridPoly.tube(
                radius=tube_radius,
                n_sides=10,
                capping=True,
            )
            self.heatmapGridActor = self.plotter.add_mesh(
                self.heatmapGridDisplayPoly,
                color="#495057",
                show_edges=False,
                smooth_shading=False,
                lighting=False,
                pickable=False,
                name=self._sensor_actor_name("sensor_heatmap_grid"),
                render=render,
            )
        except Exception as exc:
            self.heatmapGridActor = None
            self.heatmapGridPoly = None
            self.heatmapGridDisplayPoly = None
            print(f"[SensorVisualization] Failed to build heatmap grid: {exc}")

    def _ensure_heatmap_visualization_actor(self, render=False):
        if (
            getattr(self, "heatmapActor", None) is None
            or getattr(self, "heatmapPoly", None) is None
        ):
            self._rebuild_heatmap_visualization_actor(render=render)

    def _current_heatmap_sensor_matrix(self, fallback=None):
        data_obj = getattr(self, "_data", None)
        response_mode = normalize_heatmap_response_mode(
            getattr(self, "heatmap_response_mode", DEFAULT_HEATMAP_RESPONSE_MODE)
        )
        proximity_enhanced = (
            response_mode == HEATMAP_RESPONSE_PROXIMITY_ENHANCED
        )
        raw_attr = "rawData" if proximity_enhanced else "rawDataAve"
        raw_matrix = (
            getattr(data_obj, raw_attr, None) if data_obj is not None else None
        )
        calibration_override = getattr(
            self, "_heatmap_calibration_override", None
        )
        if raw_matrix is not None and calibration_override is not None:
            raw_values = np.asarray(raw_matrix, dtype=float)
            calibration_values = np.asarray(calibration_override, dtype=float)
            if not proximity_enhanced:
                # SensorDataBuffer applies this legacy display flip to all
                # averaged matrices, so align an external baseline with it.
                calibration_values = np.flipud(calibration_values)
            if raw_values.shape == calibration_values.shape:
                difference = raw_values - calibration_values
                if proximity_enhanced:
                    return np.flipud(difference)
                relative_difference = np.zeros_like(difference, dtype=float)
                np.divide(
                    100.0 * difference,
                    calibration_values,
                    out=relative_difference,
                    where=calibration_values != 0,
                )
                return relative_difference

        data_attr = "diffData" if proximity_enhanced else "diffPerDataAve"
        matrix = getattr(data_obj, data_attr, None) if data_obj is not None else None
        if proximity_enhanced and matrix is not None:
            return np.flipud(np.asarray(matrix))
        return fallback if matrix is None else matrix

    def get_heatmap_sensor_matrix(self):
        """Return the exact live value matrix currently coloured by the 3D heatmap."""
        matrix = self._current_heatmap_sensor_matrix()
        if matrix is None:
            return None
        return np.array(matrix, dtype=float, copy=True)

    def get_heatmap_recording_snapshot(self):
        """Return one self-contained tactile sample for experiment recording."""
        heatmap_values = self.get_heatmap_sensor_matrix()
        if heatmap_values is None:
            return None
        data_obj = getattr(self, "_data", None)
        raw_values = getattr(data_obj, "rawData", None)
        calibration_values = getattr(data_obj, "calData", None)
        return {
            "heatmap_values": np.array(heatmap_values, dtype=float, copy=True),
            "raw_values": (
                np.array(raw_values, dtype=float, copy=True)
                if raw_values is not None
                else None
            ),
            "calibration_values": (
                np.array(calibration_values, dtype=float, copy=True)
                if calibration_values is not None
                else None
            ),
        }

    def get_heatmap_recording_metadata(self):
        """Describe the scene and colour mapping needed for faithful replay."""
        return {
            "model": str(getattr(self, "current_model_name", "") or ""),
            "n_row": int(getattr(self, "n_row", 0) or 0),
            "n_col": int(getattr(self, "n_col", 0) or 0),
            "heatmap_settings": copy.deepcopy(self.get_heatmap_settings()),
            "geometry": copy.deepcopy(
                getattr(self, "current_sensor_geometry_config", None)
            ),
            "zero_mask": self.get_cell_zero_mask().astype(np.uint8).tolist(),
            "average_window": int(self.get_sensor_average_window_size()),
            "visualization_target_hz": float(self.get_visualization_target_hz()),
            "recorded_value": "exact_3d_heatmap_matrix",
        }

    def set_heatmap_playback_active(self, active):
        """Prevent live frames from replacing a recorded heatmap during replay."""
        self._heatmap_playback_active = bool(active)
        if self._heatmap_playback_active:
            return
        current_matrix = self.get_heatmap_sensor_matrix()
        if (
            current_matrix is not None
            and getattr(self, "heatmapPoly", None) is not None
            and self._is_heatmap_visualization_mode(
                getattr(self, "sensor_visualization_mode", "point_grid")
            )
        ):
            self._last_sensor_visualization_matrix = np.array(
                current_matrix, dtype=float, copy=True
            )
            self._update_heatmap_visualization(current_matrix)
            try:
                self.plotter.render()
            except Exception:
                pass

    def render_recorded_heatmap_frame(self, sensor_matrix):
        """Render one saved heatmap matrix through the live 3D heatmap actor."""
        values = np.asarray(sensor_matrix, dtype=float)
        expected_shape = (int(self.n_row), int(self.n_col))
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            return False
        self._ensure_heatmap_visualization_actor(render=False)
        if getattr(self, "heatmapPoly", None) is None:
            return False
        self._last_sensor_visualization_matrix = np.array(
            values, dtype=float, copy=True
        )
        self._update_heatmap_visualization(values)
        try:
            self.plotter.render()
        except Exception:
            return False
        return True

    def set_heatmap_calibration_baseline(self, values=None, n_row=None, n_col=None):
        """Share the signal viewer's heatmap baseline with the 3D heatmap."""
        if values is None:
            self._heatmap_calibration_override = None
            return True

        rows = int(n_row if n_row is not None else self.n_row)
        columns = int(n_col if n_col is not None else self.n_col)
        try:
            baseline = np.asarray(values, dtype=float)
            if baseline.ndim == 1:
                baseline = reshape_sensor_values_to_row_col_matrix(
                    baseline, rows, columns
                )
        except Exception:
            return False
        if baseline.shape != (rows, columns):
            return False

        self._heatmap_calibration_override = np.array(
            baseline, dtype=float, copy=True
        )
        current_matrix = self._current_heatmap_sensor_matrix()
        if current_matrix is not None:
            self._last_sensor_visualization_matrix = np.array(
                current_matrix, dtype=float, copy=True
            )
            if getattr(self, "heatmapPoly", None) is not None:
                self._update_heatmap_visualization(current_matrix)
                if self._is_heatmap_visualization_mode(
                    getattr(self, "sensor_visualization_mode", "point_grid")
                ):
                    try:
                        self.plotter.render()
                    except Exception:
                        pass
        return True

    def _update_heatmap_visualization(self, sensor_matrix):
        self._ensure_heatmap_visualization_actor(render=False)
        if (
            getattr(self, "heatmapPoly", None) is None
            or getattr(self, "heatmapColors", None) is None
            or getattr(self, "_heatmap_tile_vertices", None) is None
        ):
            return

        values = np.asarray(sensor_matrix, dtype=float)
        expected_shape = (int(self.n_row), int(self.n_col))
        if values.shape != expected_shape:
            if values.shape == (expected_shape[1], expected_shape[0]):
                values = values.T
            else:
                return

        rgb = heatmap_3d_rgb(
            values,
            palette=getattr(
                self, "heatmap_3d_palette", DEFAULT_HEATMAP_3D_PALETTE
            ),
            response_mode=getattr(
                self, "heatmap_response_mode", DEFAULT_HEATMAP_RESPONSE_MODE
            ),
            saturation_pct=getattr(
                self, "heatmap_saturation_pct", DEFAULT_HEATMAP_SATURATION_PCT
            ),
            noise_floor_pct=getattr(
                self, "heatmap_noise_floor_pct", DEFAULT_HEATMAP_NOISE_FLOOR_PCT
            ),
            proximity_noise_floor=getattr(
                self,
                "heatmap_proximity_noise_floor",
                DEFAULT_PROXIMITY_NOISE_FLOOR,
            ),
            proximity_knee=getattr(
                self, "heatmap_proximity_knee", DEFAULT_PROXIMITY_KNEE
            ),
            proximity_saturation=getattr(
                self,
                "heatmap_proximity_saturation",
                DEFAULT_PROXIMITY_SATURATION,
            ),
            color_gain=getattr(
                self,
                "heatmap_3d_color_gain",
                DEFAULT_HEATMAP_3D_COLOR_GAIN,
            ),
            use_absolute_signal=getattr(
                self,
                "visualization_use_absolute_signal",
                True,
            ),
        )
        mask = self._cell_zero_mask_for_plotter()
        for col in range(self.n_col):
            for row in range(self.n_row):
                vertex_ids = self._heatmap_tile_vertices[row, col]
                self.heatmapColors[vertex_ids, :3] = rgb[row, col]
                self.heatmapColors[vertex_ids, 3] = 0 if mask[row, col] else 255

        self.heatmapPoly.point_data["heatmap_colors"] = self.heatmapColors
        self.heatmapPoly.set_active_scalars("heatmap_colors")
        try:
            self.heatmapPoly.Modified()
        except Exception:
            pass

    def _coarse_grid_vectors(self, vectors):
        if vectors is None or self.n_row <= 0 or self.n_col <= 0:
            return None
        arr = np.asarray(vectors, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < self.n_row * self.n_col or arr.shape[1] != 3:
            return None
        grid = np.zeros((self.n_row, self.n_col, 3), dtype=float)
        for row in range(self.n_row):
            for col in range(self.n_col):
                grid[row, col] = arr[_column_major_idx(self.n_row, col, row)]
        return grid

    @staticmethod
    def _bilinear_grid_sample(grid, dense_rows, dense_cols):
        grid = np.asarray(grid, dtype=float)
        rows, cols = grid.shape[:2]
        if rows <= 0 or cols <= 0:
            return np.zeros((0, 0) + grid.shape[2:], dtype=float)
        row_pos = np.linspace(0.0, max(rows - 1, 0), int(dense_rows))
        col_pos = np.linspace(0.0, max(cols - 1, 0), int(dense_cols))
        row0 = np.floor(row_pos).astype(int)
        col0 = np.floor(col_pos).astype(int)
        row1 = np.clip(row0 + 1, 0, rows - 1)
        col1 = np.clip(col0 + 1, 0, cols - 1)
        row_weight = (row_pos - row0)[:, None]
        col_weight = (col_pos - col0)[None, :]

        top = (
            grid[row0[:, None], col0[None, :]] * (1.0 - col_weight)[..., None]
            + grid[row0[:, None], col1[None, :]] * col_weight[..., None]
        )
        bottom = (
            grid[row1[:, None], col0[None, :]] * (1.0 - col_weight)[..., None]
            + grid[row1[:, None], col1[None, :]] * col_weight[..., None]
        )
        return top * (1.0 - row_weight)[..., None] + bottom * row_weight[..., None]

    def _dense_cell_zero_mask(self, dense_rows, dense_cols):
        """Expand the taxel zero mask to a dense visualization grid."""
        dense_rows = int(dense_rows)
        dense_cols = int(dense_cols)
        if dense_rows <= 0 or dense_cols <= 0:
            return np.zeros((0, 0), dtype=bool)
        mask = self._cell_zero_mask_for_plotter()
        if mask.shape != (int(self.n_row), int(self.n_col)) or mask.size == 0:
            return np.zeros((dense_rows, dense_cols), dtype=bool)
        row_indices = np.rint(
            np.linspace(0.0, max(int(self.n_row) - 1, 0), dense_rows)
        ).astype(int)
        col_indices = np.rint(
            np.linspace(0.0, max(int(self.n_col) - 1, 0), dense_cols)
        ).astype(int)
        return mask[row_indices[:, None], col_indices[None, :]]

    def _apply_zero_mask_to_matrix_colors(
        self,
        colors,
        dense_shape,
        base_indices=None,
        top_indices=None,
    ):
        if colors is None or np.asarray(colors).ndim != 2 or colors.shape[1] < 4:
            return
        dense_rows, dense_cols = dense_shape
        dense_mask = self._dense_cell_zero_mask(dense_rows, dense_cols).reshape(-1)
        alpha = np.where(dense_mask, 0, 255).astype(np.uint8)
        if base_indices is not None and top_indices is not None:
            colors[np.asarray(base_indices, dtype=int), 3] = alpha
            colors[np.asarray(top_indices, dtype=int), 3] = alpha
        elif colors.shape[0] == alpha.shape[0]:
            colors[:, 3] = alpha

    def _normal_visualization_dense_grid(self, dense_scale=4, dense_cap=96):
        point_grid = self._coarse_grid_vectors(getattr(self, "points_origin", None))
        normal_grid = self._coarse_grid_vectors(getattr(self, "normals", None))
        if point_grid is None or normal_grid is None:
            return None, None, (0, 0)

        dense_rows = min(max(int(self.n_row) * int(dense_scale), int(self.n_row), 2), int(dense_cap))
        dense_cols = min(max(int(self.n_col) * int(dense_scale), int(self.n_col), 2), int(dense_cap))
        dense_points = self._bilinear_grid_sample(point_grid, dense_rows, dense_cols)
        dense_normals = self._bilinear_grid_sample(normal_grid, dense_rows, dense_cols)
        normal_norm = np.linalg.norm(dense_normals, axis=2)
        valid_normals = normal_norm > 1e-9
        dense_normals[valid_normals] = (
            dense_normals[valid_normals] / normal_norm[valid_normals][:, None]
        )
        dense_normals[~valid_normals] = [0.0, 0.0, 1.0]
        return dense_points, dense_normals, (dense_rows, dense_cols)

    def _build_matrix_line_polydata(self, mode=None):
        mode = str(mode or getattr(self, "sensor_visualization_mode", "stereo_field"))
        dense_scale = 6 if mode == "stereo_field" else 4
        dense_cap = 120 if mode == "stereo_field" else 96
        dense_points, dense_normals, dense_shape = self._normal_visualization_dense_grid(
            dense_scale=dense_scale,
            dense_cap=dense_cap,
        )
        if dense_points is None or dense_normals is None:
            return None, None, (0, 0)

        dense_rows, dense_cols = dense_shape

        lift = max(0.0008, abs(float(getattr(self, "sensor_visual_offset_scale", 0.0))) * 2.5)
        base_points = dense_points + dense_normals * lift

        if mode == "stereo_field":
            flat_base_points = base_points.reshape((-1, 3))
            flat_normals = dense_normals.reshape((-1, 3))
            finite = flat_base_points[np.all(np.isfinite(flat_base_points), axis=1)]
            if finite.size:
                span = float(np.linalg.norm(np.max(finite, axis=0) - np.min(finite, axis=0)))
            else:
                span = 0.05
            length_scale = float(getattr(self, "stereo_field_length_scale", 0.35) or 0.35)
            field_height = max(0.015, span * max(0.05, length_scale))
            height_variation = np.ones(flat_base_points.shape[0], dtype=float)

            points = np.empty((flat_base_points.shape[0] * 2, 3), dtype=float)
            base_indices = np.arange(flat_base_points.shape[0]) * 2
            top_indices = base_indices + 1
            points[base_indices] = flat_base_points
            points[top_indices] = flat_base_points + flat_normals * (
                field_height * height_variation[:, None]
            )

            lines = []
            for point_idx in range(flat_base_points.shape[0]):
                lines.append([2, int(base_indices[point_idx]), int(top_indices[point_idx])])

            poly = pv.PolyData(points)
            poly.lines = np.asarray(lines, dtype=np.int_)
            colors = np.empty((points.shape[0], 4), dtype=np.uint8)
            colors[base_indices] = [8, 70, 55, 255]
            colors[top_indices] = [55, 255, 185, 255]
            self._apply_zero_mask_to_matrix_colors(
                colors,
                dense_shape,
                base_indices=base_indices,
                top_indices=top_indices,
            )
            poly.point_data["matrix_colors"] = colors
            poly.set_active_scalars("matrix_colors")

            self._matrix_line_base_points = flat_base_points
            self._matrix_line_normals = flat_normals
            self._matrix_line_base_indices = base_indices
            self._matrix_line_top_indices = top_indices
            self._matrix_line_field_height = field_height
            self._matrix_line_height_variation = height_variation
            return poly, colors, dense_shape

        points = base_points.reshape((-1, 3))

        lines = []
        for row in range(dense_rows):
            row_start = row * dense_cols
            for col in range(dense_cols - 1):
                lines.append([2, row_start + col, row_start + col + 1])
        for col in range(dense_cols):
            for row in range(dense_rows - 1):
                lines.append([2, row * dense_cols + col, (row + 1) * dense_cols + col])

        poly = pv.PolyData(points)
        poly.lines = np.asarray(lines, dtype=np.int_)
        colors = np.empty((points.shape[0], 4), dtype=np.uint8)
        colors[:, 0] = 30
        colors[:, 1] = 235
        colors[:, 2] = 160
        colors[:, 3] = 255
        self._apply_zero_mask_to_matrix_colors(colors, dense_shape)
        poly.point_data["matrix_colors"] = colors
        poly.set_active_scalars("matrix_colors")
        self._matrix_line_base_points = None
        self._matrix_line_normals = None
        self._matrix_line_base_indices = None
        self._matrix_line_top_indices = None
        self._matrix_line_field_height = 0.0
        self._matrix_line_height_variation = None
        self._stereo_field_smoothed_visibility = None
        self._stereo_field_smoothed_color_response = None
        return poly, colors, (dense_rows, dense_cols)

    def _rebuild_matrix_visualization_actor(self, render=False):
        self.matrixLineActor = self._remove_actor_safely(
            getattr(self, "matrixLineActor", None)
        )
        self.matrixLinePoly = None
        self.matrixLineColors = None
        self._matrix_visual_actor_mode = None
        self._matrix_line_dense_shape = (0, 0)
        self._matrix_line_base_points = None
        self._matrix_line_normals = None
        self._matrix_line_base_indices = None
        self._matrix_line_top_indices = None
        self._matrix_line_field_height = 0.0
        self._matrix_line_height_variation = None
        self._stereo_field_smoothed_visibility = None
        self._stereo_field_smoothed_color_response = None

        mode = str(getattr(self, "sensor_visualization_mode", "stereo_field"))
        if not self._is_matrix_visualization_mode(mode):
            mode = "stereo_field"

        poly, colors, dense_shape = self._build_matrix_line_polydata(mode=mode)
        if poly is None:
            return
        self.matrixLinePoly = poly
        self.matrixLineColors = colors
        self._matrix_visual_actor_mode = mode
        self._matrix_line_dense_shape = dense_shape
        try:
            self.matrixLineActor = self.plotter.add_mesh(
                self.matrixLinePoly,
                scalars="matrix_colors",
                line_width=1 if mode == "stereo_field" else 3,
                rgb=True,
                render_lines_as_tubes=False if mode == "stereo_field" else True,
                lighting=False,
                ambient=1.0,
                name=self._sensor_actor_name(f"sensor_{mode}"),
                render=render,
            )
        except Exception as exc:
            self.matrixLineActor = None
            self.matrixLinePoly = None
            self.matrixLineColors = None
            self._matrix_visual_actor_mode = None
            self._matrix_line_dense_shape = (0, 0)
            self._stereo_field_smoothed_visibility = None
            self._stereo_field_smoothed_color_response = None
            print(f"[SensorVisualization] Failed to build matrix silhouette: {exc}")

    def _update_matrix_silhouette_visualization(self, sensor_matrix):
        mode = str(getattr(self, "sensor_visualization_mode", "stereo_field"))
        if (
            self.matrixLinePoly is None
            or self.matrixLineColors is None
            or getattr(self, "_matrix_visual_actor_mode", None) != mode
        ):
            self._rebuild_matrix_visualization_actor(render=False)
        if self.matrixLinePoly is None or self.matrixLineColors is None:
            return

        dense_rows, dense_cols = getattr(self, "_matrix_line_dense_shape", (0, 0))
        if dense_rows <= 0 or dense_cols <= 0:
            return
        values = np.asarray(sensor_matrix, dtype=float)
        if values.ndim != 2:
            return
        signed_values = np.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        )
        pressure = np.abs(signed_values)
        dense_signed = self._bilinear_grid_sample(
            signed_values[..., None], dense_rows, dense_cols
        )[..., 0]
        dense_pressure = self._bilinear_grid_sample(pressure[..., None], dense_rows, dense_cols)[..., 0]
        if mode == "stereo_field":
            ignore_noise = bool(getattr(self, "stereo_field_ignore_noise_enabled", True))
            deadband = max(0.0, float(getattr(self, "stereo_field_deadband_pct", 0.35)))
            response_scale = max(0.05, float(getattr(self, "stereo_field_response_scale_pct", 2.0)))
            if ignore_noise:
                contact = np.clip((dense_pressure - deadband) / response_scale, 0.0, 1.0)
                color_contact = np.clip(
                    (dense_pressure - deadband * 0.35) / (response_scale * 1.25),
                    0.0,
                    1.0,
                )
            else:
                contact = np.clip(dense_pressure * 150.0 / 255.0, 0.0, 1.0)
                color_contact = contact
            erased = np.power(contact, 0.65 if ignore_noise else 0.45)
            raw_visibility = np.clip(1.0 - erased, 0.08, 1.0).reshape(-1)
            raw_color_response = np.clip(np.power(color_contact, 0.72), 0.0, 1.0).reshape(-1)
            previous_visibility = getattr(self, "_stereo_field_smoothed_visibility", None)
            previous_color_response = getattr(self, "_stereo_field_smoothed_color_response", None)
            alpha = np.clip(
                float(getattr(self, "stereo_field_smoothing_alpha", 0.82)) if ignore_noise else 0.0,
                0.0,
                0.98,
            )
            if (
                previous_visibility is None
                or np.asarray(previous_visibility).shape != raw_visibility.shape
            ):
                visibility = raw_visibility
            else:
                visibility = alpha * np.asarray(previous_visibility) + (1.0 - alpha) * raw_visibility
            if (
                previous_color_response is None
                or np.asarray(previous_color_response).shape != raw_color_response.shape
            ):
                color_response = raw_color_response
            else:
                color_response = (
                    alpha * np.asarray(previous_color_response)
                    + (1.0 - alpha) * raw_color_response
                )
            self._stereo_field_smoothed_visibility = np.array(visibility, dtype=float, copy=True)
            self._stereo_field_smoothed_color_response = np.array(
                color_response,
                dtype=float,
                copy=True,
            )
        else:
            contact = np.clip(dense_pressure * 150.0 / 255.0, 0.0, 1.0)
            erased = np.power(contact, 0.45)
            visibility = np.clip(1.0 - erased, 0.02, 1.0).reshape(-1)
            color_response = 1.0 - visibility

        colors = self.matrixLineColors
        if mode == "stereo_field":
            base_points = getattr(self, "_matrix_line_base_points", None)
            normals = getattr(self, "_matrix_line_normals", None)
            top_indices = getattr(self, "_matrix_line_top_indices", None)
            base_indices = getattr(self, "_matrix_line_base_indices", None)
            field_height = float(getattr(self, "_matrix_line_field_height", 0.0) or 0.0)
            height_variation = getattr(self, "_matrix_line_height_variation", None)
            if (
                base_points is not None
                and normals is not None
                and top_indices is not None
                and base_indices is not None
                and height_variation is not None
                and field_height > 0.0
            ):
                height_variation = np.asarray(height_variation, dtype=float).reshape(-1)
                if height_variation.shape[0] != visibility.shape[0]:
                    height_variation = np.ones_like(visibility)
                height_scale = np.clip(0.12 + 0.88 * visibility, 0.12, 1.0)
                points = np.array(self.matrixLinePoly.points, dtype=float, copy=True)
                points[top_indices] = base_points + normals * (
                    field_height * height_variation[:, None] * height_scale[:, None]
                )
                self.matrixLinePoly.points = points
                colors[base_indices, 0] = np.clip(4 + visibility * 12, 0, 255).astype(np.uint8)
                colors[base_indices, 1] = np.clip(25 + visibility * 70, 0, 255).astype(np.uint8)
                colors[base_indices, 2] = np.clip(20 + visibility * 55, 0, 255).astype(np.uint8)
                response = np.clip(np.asarray(color_response, dtype=float), 0.0, 1.0)
                if bool(
                    getattr(self, "visualization_use_absolute_signal", True)
                ):
                    cool = np.array([25.0, 185.0, 255.0])
                    warm = np.array([255.0, 96.0, 42.0])
                    top_rgb = (
                        cool[None, :] * (1.0 - response[:, None])
                        + warm[None, :] * response[:, None]
                    )
                else:
                    neutral = np.array([245.0, 245.0, 245.0])
                    positive = np.array([220.0, 45.0, 45.0])
                    negative = np.array([35.0, 80.0, 205.0])
                    sign = np.sign(dense_signed).reshape(-1)
                    target = np.where(
                        (sign >= 0.0)[:, None],
                        positive[None, :],
                        negative[None, :],
                    )
                    top_rgb = (
                        neutral[None, :] * (1.0 - response[:, None])
                        + target * response[:, None]
                    )
                brightness = np.clip(0.62 + 0.38 * visibility, 0.35, 1.0)
                top_rgb = np.clip(top_rgb * brightness[:, None], 0, 255)
                base_rgb = np.clip(top_rgb * 0.32, 0, 255)
                colors[top_indices, :3] = top_rgb.astype(np.uint8)
                colors[base_indices, :3] = base_rgb.astype(np.uint8)
                self._apply_zero_mask_to_matrix_colors(
                    colors,
                    (dense_rows, dense_cols),
                    base_indices=base_indices,
                    top_indices=top_indices,
                )
                self.matrixLinePoly.point_data["matrix_colors"] = colors
                self.matrixLinePoly.set_active_scalars("matrix_colors")
                try:
                    self.matrixLinePoly.Modified()
                except Exception:
                    pass
                return

        colors[:, 0] = np.clip(5 + visibility * 25, 0, 255).astype(np.uint8)
        colors[:, 1] = np.clip(5 + visibility * 230, 0, 255).astype(np.uint8)
        colors[:, 2] = np.clip(5 + visibility * 155, 0, 255).astype(np.uint8)
        self._apply_zero_mask_to_matrix_colors(
            colors,
            (dense_rows, dense_cols),
        )
        self.matrixLinePoly.point_data["matrix_colors"] = colors
        self.matrixLinePoly.set_active_scalars("matrix_colors")
        try:
            self.matrixLinePoly.Modified()
        except Exception:
            pass

    def read_runtime_hz_report(self):
        direct_helper = getattr(self, "direct_finger_motion_class", None)
        direct_hz = 0.0
        direct_running = False
        if direct_helper is not None:
            direct_hz = float(getattr(direct_helper, "loop_hz", 0.0) or 0.0)
            direct_running = bool(getattr(direct_helper, "is_running", False))

        data_obj = getattr(self, "_data", None)
        window_size = (
            int(getattr(data_obj, "windowSize", self.sensor_average_window_size))
            if data_obj is not None
            else int(self.sensor_average_window_size)
        )
        hand_report = ""
        mesh_functions = getattr(getattr(self, "parent", None), "mesh_functions", None)
        if mesh_functions is not None and hasattr(mesh_functions, "hand_tactile_runtime_report"):
            try:
                hand_report = "\n" + mesh_functions.hand_tactile_runtime_report()
            except Exception as exc:
                hand_report = f"\nhand_3d_runtime_report_error: {exc}"

        goodix_report = ""
        goodix_client = getattr(self, "_goodix_client", None)
        if self.is_goodix_usb_transport() and goodix_client is not None:
            goodix_report = (
                "\ngoodix_usb_backend: "
                f"{goodix_client.backend_status}"
            )

        return (
            f"sensor_update_hz: {self._sensor_update_hz:.2f}\n"
            f"direct_finger_motion_loop_hz: {direct_hz:.2f}\n"
            f"direct_finger_motion_running: {direct_running}\n"
            f"visualization_actual_hz: {self._visualization_hz:.2f}\n"
            f"sensor_average_window_size: {window_size}\n"
            f"visualization_target_hz: {self.visualization_target_hz:.2f}"
            f"{goodix_report}"
            f"{hand_report}"
        )

    def toggle_ai_direct_finger_motion_execution(self, model_checkpoint_path=None, dry_run_predictions_only=None):
        helper = getattr(self, "ai_direct_finger_motion_execution_class", None)
        if helper is None:
            raise RuntimeError("AI_DirectFingerMotion_execution helper is not initialized.")
        if dry_run_predictions_only is not None and hasattr(helper, "set_dry_run_predictions_only"):
            helper.set_dry_run_predictions_only(dry_run_predictions_only)
        return helper.toggle_ai_direct_finger_motion_execution(
            model_checkpoint_path=model_checkpoint_path
        )

    def _ensure_sensor_api_backend(self):
        """Load the real serial API without opening a competing port handle."""
        ensure_api = getattr(self.parent, "ensure_sensor_api", None)
        if not callable(ensure_api):
            self._set_sensor_stream_error("The sensor serial API is unavailable.")
            return False

        try:
            ready = bool(ensure_api(connect_immediately=False))
        except TypeError:
            ready = bool(ensure_api())
        except Exception as exc:
            self._set_sensor_stream_error(
                f"Could not initialize the sensor serial API: {exc}"
            )
            return False

        if not ready:
            self._set_sensor_stream_error("The sensor serial API could not be loaded.")
            return False
        return True

    def buildScene(self, show_main_visualization: bool = True):
        self._set_sensor_stream_error("")
        selected_sources = self._get_selected_port_names()
        if not selected_sources:
            message = (
                "No sensor source selected. Highlight a sensor source first."
            )
            self._set_sensor_stream_error(message)
            print(message)
            return
        port_profiles = self._resolve_sensor_port_profiles(selected_sources)
        primary_profile = port_profiles[
            self._sensor_port_key(selected_sources[0])
        ]
        selected_model_name = self.SENSOR_MODEL_NAMES_BY_INDEX.get(
            int(self.parent.sensor_choice.currentRow())
        )
        if selected_model_name != "2d" and len(selected_sources) > 1:
            model_config = self.PREDEFINED_SENSOR_MODELS.get(
                str(selected_model_name), {}
            )
            expected_shape = (
                int(model_config.get("n_row", 0) or 0),
                int(model_config.get("n_col", 0) or 0),
            )
            mismatched = [
                source
                for source in selected_sources
                if (
                    port_profiles[self._sensor_port_key(source)]["n_row"],
                    port_profiles[self._sensor_port_key(source)]["n_col"],
                )
                != expected_shape
            ]
            if mismatched:
                message = (
                    "Different sensor grid sizes in Multiple Ports mode are "
                    "supported with the 2D sensor model. Select 2D, or make "
                    f"every port match {expected_shape[0]}x{expected_shape[1]}."
                )
                self._set_sensor_stream_error(message)
                print(message)
                return
        if len(selected_sources) > 1 and any(
            is_goodix_usb_source(source) for source in selected_sources
        ):
            message = (
                "Goodix USB cannot be combined with serial sensors in "
                "Multiple Ports mode. Select serial ports only."
            )
            self._set_sensor_stream_error(message)
            print(message)
            return
        use_goodix_usb = is_goodix_usb_source(selected_sources[0])
        if not use_goodix_usb and not self._ensure_sensor_api_backend():
            print(self.get_last_sensor_stream_error())
            return
        self.main_visualization_enabled = bool(show_main_visualization)
        self._reset_scene_build_state()
        self._set_sensor_update_button_enabled(False)
        if not self._close_serial_ports():
            message = (
                "Could not rebuild the sensor scene because the previous "
                "reader is still stopping."
            )
            self._set_sensor_stream_error(message)
            self._set_sensor_update_button_enabled(True)
            print(message)
            return
        self._clear_scene_actors()
        self._close_goodix_client()
        self._close_standalone_sensor_api()
        self.is_connected = False
        self._sensor_profiles_by_port = port_profiles

        if use_goodix_usb:
            self.sensor_transport = "goodix_usb"
            self._goodix_client = GoodixUsbSensorClient(
                rows=GOODIX_USB_ROWS,
                columns=GOODIX_USB_COLUMNS,
            )
            if not self._goodix_client.is_available:
                message = (
                    "Goodix USB sensor support is unavailable. Confirm the sensor "
                    "is connected and libusb or UsbTouchCore is available."
                )
                self._set_sensor_stream_error(message)
                print(message)
                self._close_goodix_client()
                return
            self.ser_list = []
        else:
            self.sensor_transport = "serial"
            self.ser_list = self._open_serial_ports(selected_sources)
            if not self.ser_list:
                if not self.get_last_sensor_stream_error():
                    self._set_sensor_stream_error(
                        "Could not open any of the selected sensor ports."
                    )
                print("Could not open any of the selected ports.")
                return

        self.clearParameters()
        if not self._initialize_selected_sensor_model(
            self.parent.sensor_choice.currentRow(),
            n_row=primary_profile["n_row"],
            n_col=primary_profile["n_col"],
        ):
            self._close_serial_ports()
            self._close_goodix_client()
            print("Unsupported sensor selection.")
            return
        self._configure_sensor_port_views()
        self.initialize_ai_helpers()
        self.update_ui_elements()
        if not bool(show_main_visualization):
            self.set_main_visualization_enabled(False, render=True)

    def _initialize_from_factory(self, model: SensorModelFactory):
        """Helper to assign all model attributes from the factory to the MySensor instance."""
        self.n_row, self.n_col, self.n_node = model.n_row, model.n_col, model.n_node
        self._data = model._data
        self.points, self.points_origin, self.normals = model.points, model.points_origin, model.normals
        self.edges, self.colors_3d = model.edges, model.colors_3d
        self._2D_map, self.array_positions, self.colors, self.line_poly = \
            model._2D_map, model.array_positions, model.colors, model.line_poly
        self.sensor_visual_offset_scale = float(getattr(model, "offset_scale", 0.0) or 0.0)
        self._capture_sensor_geometry_base()
        self.set_sensor_geometry_config(
            self.get_saved_sensor_geometry_config(
                self.current_model_name,
                n_row=self.n_row,
                n_col=self.n_col,
            ),
            save_current_sensor=False,
            render=False,
        )

        # Reset the zero-mask for the new layout, then try to restore a previously
        # saved one for this exact sensor key (model + grid size).
        self.cell_zero_mask = np.zeros((self.n_row, self.n_col), dtype=bool)
        self.load_cell_zero_mask_from_disk()

        # Reset connection flags for the new model
        self.is_connected = False
        self.show_2D = False
        self.show_PC = False
        self.show_FittedMesh = False

        # Add meshes only when the main sensor plotter is the active consumer.
        # Robot-dialog-only streaming still builds all geometry/data mappings,
        # but avoids a second OpenGL render path.
        if bool(getattr(self, "main_visualization_enabled", True)):
            self._ensure_main_sensor_visualization_actors()
            self._refresh_sensor_visualization_mode_actors()
            self.restore_saved_sensor_perspective(render=True)

    def init_2d_model(self, n_row=None, n_col=None):
        if n_row is None or n_col is None:
            n_row, n_col = self._get_2d_grid_shape()

        self.current_model_name = "2d"
        self._build_factory_model(
            n_row=n_row,
            n_col=n_col,
            offset_scale=0.0005,
        )

    def init_elbow_model(self):
        """Initializes the curved elbow sensor model."""
        self._init_predefined_model("elbow")

    def init_kuka_model(self):
        """Initializes the Kuka sensor model with row-major to flipped column-major reordering."""
        self._init_predefined_model("kuka")

    def init_double_curve_model(self):
        """Initializes the double curve sensor model."""
        self._init_predefined_model("double_curve")

    def init_half_cylinder_surface_model(self):
        """Initializes the half-cylinder model with vertical flip reordering."""
        self._init_predefined_model("half_cylinder_surface")

    def clearParameters(self):
        # 0) completely clear out anything left over from the last model
        self.points = None  # wipe old geometry buffers
        self.edges = None
        self.colors = None
        self.colors_3d = None
        self.normals = None
        self.colors_faces = None
        self.line_poly = None
        self.actionMesh = None
        self.points_origin = None
        self.array_positions = None
        self.n_node = None
        self._2D_map = None
        self._sensor_geometry_base_points_origin = None
        self._sensor_geometry_base_normals = None
        self._sensor_geometry_base_fine_points = None
        self.current_sensor_geometry_config = None
        self.sensor_visual_offset_scale = 0.0
        self._last_sensor_visualization_matrix = None
        self._clear_contact_normal_actor()
        self._clear_sensor_point_label_actor()

    def update_ui_elements(self):
        self.parent.buildScene.setText("Scene Built")
        self.parent.buildScene.setStyleSheet("""
            color: #3498db;
            font-weight: bold;
        """)
        self._set_sensor_update_button_enabled(True)

    def _reset_scene_build_state(self):
        self.parent.buildScene.setText("Build Scene")
        self.parent.buildScene.setStyleSheet("color: white; font-weight: normal;")

    def _set_sensor_update_button_enabled(self, enabled: bool):
        try:
            self.parent.sensor_update.setEnabled(bool(enabled))
        except Exception:
            pass

    def _remove_actor_safely(self, actor):
        if actor is None:
            return None
        try:
            self.plotter.remove_actor(actor, reset_camera=False)
        except Exception:
            self.plotter.clear()
        return None

    def _clear_scene_actors(self):
        self._clear_multi_port_views()
        self.objActor = self._remove_actor_safely(self.objActor)
        self.actionMesh = self._remove_actor_safely(self.actionMesh)
        self.matrixLineActor = self._remove_actor_safely(
            getattr(self, "matrixLineActor", None)
        )
        self.heatmapActor = self._remove_actor_safely(
            getattr(self, "heatmapActor", None)
        )
        self.heatmapGridActor = self._remove_actor_safely(
            getattr(self, "heatmapGridActor", None)
        )
        self.heatmapPoly = None
        self.heatmapColors = None
        self.heatmapGridPoly = None
        self.heatmapGridDisplayPoly = None
        self._heatmap_tile_vertices = None
        self._last_sensor_visualization_matrix = None
        self.matrixLinePoly = None
        self.matrixLineColors = None
        self._matrix_visual_actor_mode = None
        self._matrix_line_dense_shape = (0, 0)
        self._matrix_line_base_points = None
        self._matrix_line_normals = None
        self._matrix_line_base_indices = None
        self._matrix_line_top_indices = None
        self._matrix_line_field_height = 0.0
        self._matrix_line_height_variation = None
        self._stereo_field_smoothed_visibility = None
        self._stereo_field_smoothed_color_response = None
        self._clear_contact_normal_actor()
        self._clear_sensor_point_label_actor()

    def _get_selected_port_names(self):
        selected = []
        for item in self.parent.serial_channel.selectedItems():
            try:
                port_name = item.data(Qt.UserRole)
            except (AttributeError, TypeError):
                port_name = None
            selected.append(str(port_name or item.text()))
        return selected

    def _multi_port_display_enabled(self):
        combo = getattr(self.parent, "sensor_source_mode_combo", None)
        if combo is None:
            return False
        try:
            return combo.currentData() == "multiple"
        except Exception:
            return False

    def is_goodix_usb_transport(self):
        """Return whether the current scene uses the Goodix USB source."""
        return getattr(self, "sensor_transport", "serial") == "goodix_usb"

    def _close_standalone_sensor_api(self):
        sensor_api = getattr(self.parent, "sensor_api", None)
        close = getattr(sensor_api, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _sensor_port_key(port_name):
        text = str(port_name or "").strip()
        if not text:
            return ""
        if is_goodix_usb_source(text):
            return GOODIX_USB_SOURCE_ID
        if not os.path.isabs(text):
            text = os.path.join("/dev", text)
        try:
            return os.path.realpath(text)
        except Exception:
            return text

    def _normalize_sensor_port_profile(self, profile, fallback=None):
        fallback = dict(fallback or {})
        profile = dict(profile or {})
        try:
            n_row = int(profile.get("n_row", fallback.get("n_row", self.n_row)))
        except Exception:
            n_row = int(fallback.get("n_row", self.n_row))
        try:
            n_col = int(profile.get("n_col", fallback.get("n_col", self.n_col)))
        except Exception:
            n_col = int(fallback.get("n_col", self.n_col))
        return {
            "n_row": max(1, n_row),
            "n_col": max(1, n_col),
            "has_extra_column": bool(
                profile.get(
                    "has_extra_column",
                    fallback.get("has_extra_column", True),
                )
            ),
        }

    def _resolve_sensor_port_profiles(self, selected_sources):
        getter = getattr(self.parent, "get_sensor_port_profile", None)
        fallback = {
            "n_row": int(
                getattr(self.parent.grid_rows_spin, "value", lambda: 10)()
            ),
            "n_col": int(
                getattr(self.parent.grid_cols_spin, "value", lambda: 10)()
            ),
            "has_extra_column": self._raw_packet_has_extra_column(),
        }
        profiles = {}
        for source in selected_sources:
            raw_profile = getter(source) if callable(getter) else fallback
            profiles[self._sensor_port_key(source)] = (
                self._normalize_sensor_port_profile(
                    raw_profile,
                    fallback=fallback,
                )
            )
        return profiles

    def _sensor_profile_for_port(self, port_name):
        key = self._sensor_port_key(port_name)
        profile = getattr(self, "_sensor_profiles_by_port", {}).get(key)
        if profile is not None:
            return dict(profile)
        return {
            "n_row": int(self.n_row),
            "n_col": int(self.n_col),
            "has_extra_column": self._raw_packet_has_extra_column(),
        }

    @staticmethod
    def _safe_actor_suffix(port_name):
        return "".join(
            character if character.isalnum() else "_"
            for character in str(port_name or "sensor")
        ).strip("_") or "sensor"

    def _clear_multi_port_views(self):
        for view in getattr(self, "_multi_port_sensor_views", {}).values():
            view.close()
        self._multi_port_sensor_views = {}

        for actor in getattr(self, "_multi_port_label_actors", []):
            try:
                self.plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass
        self._multi_port_label_actors = []
        self._pending_multi_port_visualization_matrices = {}
        self._sensor_data_by_port = {}
        self._primary_sensor_port = None
        self._calibrated_sensor_ports = set()

    def _multi_port_scene_spacing(self):
        point_sets = [
            getattr(self, "points_origin", None),
            getattr(getattr(self, "_2D_map", None), "points", None),
        ]
        finite_sets = []
        for points in point_sets:
            try:
                array = np.asarray(points, dtype=float)
            except (TypeError, ValueError):
                continue
            if array.ndim == 2 and array.shape[1] == 3:
                array = array[np.all(np.isfinite(array), axis=1)]
                if array.size:
                    finite_sets.append(array)
        if not finite_sets:
            return 1.0
        combined = np.vstack(finite_sets)
        span = np.ptp(combined, axis=0)
        return max(float(np.max(span)) * 1.35, 0.1)

    def _add_multi_port_label(self, port_name, offset):
        points = np.asarray(self.points_origin, dtype=float) + np.asarray(
            offset, dtype=float
        )
        finite = points[np.all(np.isfinite(points), axis=1)]
        if not finite.size:
            return
        span = max(float(np.max(np.ptp(finite, axis=0))), 0.05)
        label_position = np.array(
            [
                float(np.mean(finite[:, 0])),
                float(np.max(finite[:, 1]) + span * 0.12),
                float(np.max(finite[:, 2]) + span * 0.06),
            ]
        )
        try:
            actor = self.plotter.add_point_labels(
                pv.PolyData(label_position.reshape(1, 3)),
                [os.path.basename(str(port_name))],
                font_size=12,
                text_color="#ffffff",
                show_points=False,
                shape="rounded_rect",
                shape_color="#263238",
                shape_opacity=0.9,
                margin=4,
                always_visible=True,
                name=self._sensor_actor_name(
                    f"multi_port_label_{self._safe_actor_suffix(port_name)}"
                ),
                render=False,
            )
        except Exception as exc:
            print(f"[MultiSensor] Could not label {port_name}: {exc}")
            return
        self._multi_port_label_actors.append(actor)

    def _build_multi_port_2d_model(self, profile):
        n_row = int(profile["n_row"])
        n_col = int(profile["n_col"])
        mode = self.get_saved_sensor_reorder_mode(
            "2d",
            n_row=n_row,
            n_col=n_col,
        )
        reorder_logic = self._reorder_mode_to_logic("2d", mode)
        model_kwargs = {
            "n_row": n_row,
            "n_col": n_col,
            "offset_scale": 0.0005,
            "window_size": self.sensor_average_window_size,
        }
        if reorder_logic is not None:
            model_kwargs["reorder_logic"] = reorder_logic
        return SensorModelFactory(**model_kwargs).build()

    def _configure_sensor_port_views(self):
        self._clear_multi_port_views()
        if self.is_goodix_usb_transport():
            self._primary_sensor_port = GOODIX_USB_SOURCE_ID
            self._sensor_data_by_port[self._primary_sensor_port] = self._data
            return

        port_names = [
            self._sensor_port_key(getattr(ser, "port", ""))
            for ser in self.ser_list
        ]
        port_names = [name for name in port_names if name]
        if not port_names:
            return

        self._primary_sensor_port = port_names[0]
        self._sensor_data_by_port[self._primary_sensor_port] = self._data
        if not self._multi_port_display_enabled() or len(port_names) <= 1:
            return

        spacing = self._multi_port_scene_spacing()
        grid_columns = max(1, int(np.ceil(np.sqrt(len(port_names)))))
        for display_index, port_name in enumerate(port_names):
            display_row, display_col = divmod(display_index, grid_columns)
            offset = np.array(
                [display_col * spacing, -display_row * spacing, 0.0],
                dtype=float,
            )
            self._add_multi_port_label(port_name, offset)
            if display_index == 0:
                continue

            profile = self._sensor_profile_for_port(port_name)
            replica_model = None
            if (
                int(profile["n_row"]) != int(self.n_row)
                or int(profile["n_col"]) != int(self.n_col)
            ):
                replica_model = self._build_multi_port_2d_model(profile)
                data_obj = replica_model._data
            else:
                data_obj = SensorDataBuffer(
                    int(profile["n_row"]),
                    int(profile["n_col"]),
                    window_size=self.sensor_average_window_size,
                )
            self._sensor_data_by_port[port_name] = data_obj
            self._multi_port_sensor_views[port_name] = (
                _MultiPortSensorReplica.from_owner(
                    self,
                    data_obj=data_obj,
                    port_name=port_name,
                    offset=offset,
                    model=replica_model,
                )
            )

        try:
            self.plotter.reset_camera()
            self.saveCameraPara()
            self.plotter.render()
        except Exception:
            pass

    def _sensor_data_for_port(self, port_name):
        key = self._sensor_port_key(port_name)
        return self._sensor_data_by_port.get(key)

    def get_live_sensor_port_profiles(self):
        """Return active sensor dimensions keyed by normalized source name."""
        profiles = {}
        for port_name, data_obj in getattr(
            self, "_sensor_data_by_port", {}
        ).items():
            profile = self._sensor_profile_for_port(port_name)
            profiles[self._sensor_port_key(port_name)] = {
                "n_row": int(getattr(data_obj, "n_row", profile["n_row"])),
                "n_col": int(getattr(data_obj, "n_col", profile["n_col"])),
                "has_extra_column": bool(
                    profile["has_extra_column"]
                ),
                "is_primary": bool(
                    self.is_primary_sensor_port(port_name)
                ),
            }
        return profiles

    def is_primary_sensor_port(self, port_name):
        primary = self._sensor_port_key(
            getattr(self, "_primary_sensor_port", "")
        )
        if not primary:
            return True
        candidate = self._sensor_port_key(port_name)
        if candidate == primary:
            return True
        try:
            return os.path.realpath(candidate) == os.path.realpath(primary)
        except Exception:
            return False

    def _close_goodix_client(self):
        client = getattr(self, "_goodix_client", None)
        if client is not None:
            client.close()
        self._goodix_client = None

    def _close_serial_ports(self):
        if not self._stop_sensor_reader_worker():
            print("Serial ports left open because the sensor reader is still stopping.")
            return False
        for ser in self.ser_list:
            try:
                ser.close()
            except Exception:
                pass
        self.ser_list = []
        return True

    def _open_serial_ports(self, port_names):
        serial_ports = []
        for port_name in port_names:
            try:
                ser = serial.Serial(
                    port=f"/dev/{port_name}",
                    baudrate=DEFAULT_SENSOR_BAUD_RATE,
                    timeout=DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC,
                )
                serial_ports.append(ser)
                print(f"Opened port: /dev/{port_name}")
            except Exception as exc:
                self._set_sensor_stream_error(
                    f"Could not open /dev/{port_name}: {exc}"
                )
                print(f"Failed to open /dev/{port_name}: {exc}")
        return serial_ports

    def _sensor_reader_is_running(self):
        thread = getattr(self, "_sensor_read_thread", None)
        return bool(thread is not None and thread.isRunning())

    def _start_sensor_reader_worker(self):
        if self._sensor_reader_is_running():
            return

        if self.is_goodix_usb_transport():
            if self._goodix_client is None:
                self._set_sensor_stream_error(
                    "The Goodix USB reader is not initialized. "
                    "Build the scene again."
                )
                return
        elif not self.ser_list:
            return

        self._latest_sensor_payloads = {}
        self._last_sensor_api_payloads = {}
        self._sensor_reader_generation += 1
        generation = self._sensor_reader_generation
        thread_parent = self.parent if isinstance(self.parent, QObject) else None
        thread = QThread(thread_parent)
        if self.is_goodix_usb_transport():
            worker = GoodixUsbReadWorker(
                self._goodix_client,
                generation=generation,
            )
        else:
            expected_by_port = {}
            for serial_port in self.ser_list:
                profile = self._sensor_profile_for_port(
                    getattr(serial_port, "port", "")
                )
                expected_by_port[str(serial_port.port)] = int(
                    profile["n_row"]
                ) * (
                    int(profile["n_col"])
                    + (1 if profile["has_extra_column"] else 0)
                )
            worker = _SensorReadWorker(
                self.ser_list,
                generation=generation,
                expected_payload_values=max(
                    expected_by_port.values(),
                    default=int(self.n_row) * (int(self.n_col) + 1),
                ),
                expected_payload_values_by_port=expected_by_port,
            )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        # Queued connection into the GUI thread: stores the payload AND
        # processes it immediately (event-driven, no polling delay).
        worker.raw_payload_ready.connect(self._payload_bridge.deliver)
        worker.error.connect(self._on_sensor_reader_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda thread=thread, worker=worker: self._on_sensor_reader_thread_finished(thread, worker)
        )

        self._sensor_read_thread = thread
        self._sensor_read_worker = worker
        thread.start()

    def _stop_sensor_reader_worker(self, wait_timeout_ms=5000):
        self._sensor_reader_generation += 1
        worker = getattr(self, "_sensor_read_worker", None)
        thread = getattr(self, "_sensor_read_thread", None)

        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass

        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(max(0, int(wait_timeout_ms))):
                print("[SensorReader] Stop timed out; keeping reader thread reference alive.")
                return False

        self._sensor_read_worker = None
        self._sensor_read_thread = None
        self._latest_sensor_payloads = {}
        return True

    def _on_sensor_reader_thread_finished(self, thread=None, worker=None):
        if thread is not None and self._sensor_read_thread is not thread:
            return
        if worker is not None and self._sensor_read_worker is not worker:
            return
        self._sensor_read_worker = None
        self._sensor_read_thread = None

    def _on_sensor_reader_payload(self, generation, port_name, data_list):
        if int(generation) != int(self._sensor_reader_generation):
            return
        if not self.is_connected:
            return
        self._latest_sensor_payloads[str(port_name)] = list(data_list or [])
        self._last_sensor_api_payloads[str(port_name)] = list(data_list or [])

    def get_last_sensor_api_payload(self, port_path=None):
        """Return the last API-level payload received by the live reader."""
        payloads = getattr(self, "_last_sensor_api_payloads", {})
        if not payloads:
            return None
        if not port_path:
            return list(next(iter(payloads.values())))

        requested_source = str(port_path)
        if requested_source in payloads:
            return list(payloads[requested_source])
        requested_path = os.path.realpath(str(port_path))
        for saved_path, payload in payloads.items():
            if os.path.realpath(str(saved_path)) == requested_path:
                return list(payload)
        return None

    def _on_sensor_reader_error(self, message):
        now = time.perf_counter()
        if (now - self._last_sensor_reader_error_log_time) < 2.0:
            return
        self._last_sensor_reader_error_log_time = now
        print(f"[SensorReader] {message}")

    def _take_latest_sensor_payloads(self):
        payloads = dict(self._latest_sensor_payloads)
        self._latest_sensor_payloads.clear()
        return payloads

    def shutdown(self):
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        for timer_name in ("timer", "_visualization_timer"):
            timer = getattr(self, timer_name, None)
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass
        self._sensor_calibration_stop_event.set()
        calibration_thread = self._sensor_calibration_thread
        if calibration_thread is not None and calibration_thread.is_alive():
            calibration_thread.join(timeout=5.0)
        self._stop_sensor_reader_worker()
        if calibration_thread is not None and calibration_thread.is_alive():
            print(
                "[SensorCalibration] Shutdown timed out; serial ports are left "
                "open until calibration exits to avoid a close/read race."
            )
            return
        self._sensor_calibration_thread = None
        for attr_name in self.AI_HELPER_ATTRS:
            helper = self.__dict__.get(attr_name)
            if helper is None or isinstance(helper, _LazyFeatureProxy):
                continue
            shutdown = getattr(helper, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    print(f"[{attr_name}] Shutdown failed: {exc}")
        self._close_serial_ports()
        self._close_goodix_client()
        self._close_standalone_sensor_api()

    def _get_2d_grid_shape(self):
        default_n_row, default_n_col = self.DEFAULT_2D_GRID_SHAPE
        try:
            return self.parent.grid_rows_spin.value(), self.parent.grid_cols_spin.value()
        except Exception:
            return default_n_row, default_n_col

    def _build_factory_model(self, **model_kwargs):
        model_kwargs.setdefault("window_size", self.sensor_average_window_size)
        model_kwargs = self._apply_saved_reorder_logic(model_kwargs)
        model = SensorModelFactory(**model_kwargs).build()
        self._initialize_from_factory(model)

    def _init_predefined_model(self, model_name):
        self.current_model_name = model_name
        self._build_factory_model(**self.PREDEFINED_SENSOR_MODELS[model_name])

    def _initialize_selected_sensor_model(
        self,
        sensor_index,
        n_row=None,
        n_col=None,
    ):
        model_initializers = {
            "elbow": self.init_elbow_model,
            "kuka": self.init_kuka_model,
            "double_curve": self.init_double_curve_model,
            "2d": self.init_2d_model,
            "half_cylinder_surface": self.init_half_cylinder_surface_model,
        }
        model_name = self.SENSOR_MODEL_NAMES_BY_INDEX.get(sensor_index)
        initializer = model_initializers.get(model_name)
        if initializer is None:
            return False
        if model_name == "2d":
            initializer(n_row=n_row, n_col=n_col)
        else:
            initializer()
        return True

    # ------------------------------------------------------------------
    # Sensor reorder logic: per-sensor persistent geometry/channel remapping.
    # ------------------------------------------------------------------
    def get_reorder_logic_options(self):
        return list(REORDER_LOGIC_OPTIONS)

    def get_sensor_model_choices(self):
        return [
            (model_name, self.SENSOR_MODEL_LABELS.get(model_name, model_name))
            for _, model_name in sorted(self.SENSOR_MODEL_NAMES_BY_INDEX.items())
        ]

    def get_sensor_model_name_for_index(self, sensor_index):
        return self.SENSOR_MODEL_NAMES_BY_INDEX.get(int(sensor_index), "sensor")

    def _sensor_shape_for_model(self, model_name):
        if model_name == "2d":
            return self._get_2d_grid_shape()
        config = self.PREDEFINED_SENSOR_MODELS.get(str(model_name), {})
        return int(config.get("n_row", 0) or 0), int(config.get("n_col", 0) or 0)

    def get_sensor_reorder_key(self, model_name=None, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        if n_row is None or n_col is None:
            n_row, n_col = self._sensor_shape_for_model(model)
        if self.is_goodix_usb_transport():
            model = f"goodix_usb_{model}"
        return f"{model}_{int(n_row)}x{int(n_col)}"

    def _default_reorder_logic_for_model(self, model_name):
        if model_name == "2d":
            return None
        config = self.PREDEFINED_SENSOR_MODELS.get(str(model_name), {})
        return config.get("reorder_logic", None)

    def _normalize_reorder_mode(self, mode):
        if mode is None:
            return REORDER_FACTORY_DEFAULT
        text = str(mode).strip()
        if text == "":
            return REORDER_FACTORY_DEFAULT
        if text not in REORDER_LOGIC_OPTIONS:
            return REORDER_FACTORY_DEFAULT
        return text

    def _reorder_mode_to_logic(self, model_name, mode):
        mode = self._normalize_reorder_mode(mode)
        if mode == REORDER_FACTORY_DEFAULT:
            return self._default_reorder_logic_for_model(model_name)
        if mode == REORDER_NONE:
            return None
        return mode

    def _read_reorder_logic_file(self):
        try:
            with open(SENSOR_REORDER_LOGIC_FILE, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                payload.setdefault("version", 1)
                settings = payload.get("settings")
                if not isinstance(settings, dict):
                    payload["settings"] = {}
                return payload
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[SensorReorder] Failed to read {SENSOR_REORDER_LOGIC_FILE}: {exc}")
        return {"version": 1, "settings": {}}

    def _write_reorder_logic_file(self, payload):
        try:
            os.makedirs(os.path.dirname(SENSOR_REORDER_LOGIC_FILE), exist_ok=True)
            with open(SENSOR_REORDER_LOGIC_FILE, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            return True
        except Exception as exc:
            print(f"[SensorReorder] Failed to write {SENSOR_REORDER_LOGIC_FILE}: {exc}")
            return False

    def get_saved_sensor_reorder_mode(self, model_name=None, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        item = payload.get("settings", {}).get(key, {})
        if isinstance(item, dict):
            return self._normalize_reorder_mode(item.get("reorder_mode"))
        return self._normalize_reorder_mode(item)

    def set_saved_sensor_reorder_mode(self, model_name, mode, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        mode = self._normalize_reorder_mode(mode)
        if n_row is None or n_col is None:
            n_row, n_col = self._sensor_shape_for_model(model)
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        settings = payload.setdefault("settings", {})
        item = settings.get(key, {})
        if not isinstance(item, dict):
            item = {}
        item.update({
            "model": model,
            "n_row": int(n_row),
            "n_col": int(n_col),
            "reorder_mode": mode,
        })
        settings[key] = item
        return self._write_reorder_logic_file(payload)

    def get_saved_sensor_point_labels_enabled(self, model_name=None, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        item = payload.get("settings", {}).get(key, {})
        if isinstance(item, dict):
            return bool(item.get("point_labels_enabled", False))
        return False

    def set_saved_sensor_point_labels_enabled(self, model_name, enabled, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        if n_row is None or n_col is None:
            n_row, n_col = self._sensor_shape_for_model(model)
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        settings = payload.setdefault("settings", {})
        item = settings.get(key, {})
        if not isinstance(item, dict):
            item = {}
        item.update({
            "model": model,
            "n_row": int(n_row),
            "n_col": int(n_col),
            "point_labels_enabled": bool(enabled),
        })
        settings[key] = item
        return self._write_reorder_logic_file(payload)

    def get_saved_sensor_background_reference_enabled(self, model_name=None, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        item = payload.get("settings", {}).get(key, {})
        if not isinstance(item, dict):
            return True
        return bool(item.get("background_reference_enabled", True))

    def set_saved_sensor_background_reference_enabled(self, model_name, enabled, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        if n_row is None or n_col is None:
            n_row, n_col = self._sensor_shape_for_model(model)
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        settings = payload.setdefault("settings", {})
        item = settings.get(key, {})
        if not isinstance(item, dict):
            item = {}
        item.update({
            "model": model,
            "n_row": int(n_row),
            "n_col": int(n_col),
            "background_reference_enabled": bool(enabled),
        })
        settings[key] = item
        return self._write_reorder_logic_file(payload)

    def get_saved_sensor_contact_force_scale(self, model_name=None, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        item = payload.get("settings", {}).get(key, {})
        if not isinstance(item, dict):
            return 0.0
        try:
            return max(0.0, float(item.get("force_scale_n_per_signal", 0.0)))
        except Exception:
            return 0.0

    def set_saved_sensor_contact_force_scale(self, model_name, scale, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        if n_row is None or n_col is None:
            n_row, n_col = self._sensor_shape_for_model(model)
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        settings = payload.setdefault("settings", {})
        item = settings.get(key, {})
        if not isinstance(item, dict):
            item = {}
        try:
            scale_value = max(0.0, float(scale))
        except Exception:
            scale_value = 0.0
        item.update({
            "model": model,
            "n_row": int(n_row),
            "n_col": int(n_col),
            "force_scale_n_per_signal": scale_value,
        })
        settings[key] = item
        return self._write_reorder_logic_file(payload)

    def get_saved_sensor_camera_config(
        self, model_name=None, n_row=None, n_col=None
    ):
        model = str(model_name or self.current_model_name or "sensor")
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        item = payload.get("settings", {}).get(key, {})
        config = item.get("camera", {}) if isinstance(item, dict) else {}
        return self._normalize_sensor_camera_config(config)

    def set_saved_sensor_camera_config(
        self, model_name, config, n_row=None, n_col=None
    ):
        model = str(model_name or self.current_model_name or "sensor")
        if n_row is None or n_col is None:
            n_row, n_col = self._sensor_shape_for_model(model)
        camera_config = self._normalize_sensor_camera_config(config)
        if camera_config is None:
            return False
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        settings = payload.setdefault("settings", {})
        item = settings.get(key, {})
        if not isinstance(item, dict):
            item = {}
        item.update({
            "model": model,
            "n_row": int(n_row),
            "n_col": int(n_col),
            "camera": camera_config,
        })
        settings[key] = item
        return self._write_reorder_logic_file(payload)

    def save_current_sensor_perspective(self):
        if not self.current_model_name or self.n_row <= 0 or self.n_col <= 0:
            return False
        camera_config = self.capture_sensor_camera_config()
        if camera_config is None:
            return False
        return self.set_saved_sensor_camera_config(
            self.current_model_name,
            camera_config,
            n_row=self.n_row,
            n_col=self.n_col,
        )

    def restore_saved_sensor_perspective(self, render=True):
        if not self.current_model_name or self.n_row <= 0 or self.n_col <= 0:
            return False
        camera_config = self.get_saved_sensor_camera_config(
            self.current_model_name,
            n_row=self.n_row,
            n_col=self.n_col,
        )
        if camera_config is None:
            return False
        return self.apply_sensor_camera_config(camera_config, render=render)

    @staticmethod
    def _default_sensor_geometry_config():
        return {
            "use_selected_shape": False,
            "shape": "flat",
            "bend_axis": "columns",
            "arc_deg": 0.0,
            "normal_flip": False,
            "rotation_deg": [0.0, 0.0, 0.0],
            "custom_points": [],
            "use_custom_heatmap_shape": False,
            "custom_heatmap_points": [],
            "custom_heatmap_corners": [],
            "use_curved_heatmap_edges": False,
            "custom_heatmap_edge_offsets": [],
        }

    @staticmethod
    def _normalize_sensor_rotation_degrees(value):
        if isinstance(value, dict):
            raw_values = [
                value.get("x", value.get("rx", 0.0)),
                value.get("y", value.get("ry", 0.0)),
                value.get("z", value.get("rz", 0.0)),
            ]
        elif isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
            raw_values = value[:3]
        else:
            raw_values = [0.0, 0.0, 0.0]

        rotation_deg = []
        for raw_value in raw_values:
            try:
                rotation_deg.append(float(np.clip(float(raw_value), -180.0, 180.0)))
            except Exception:
                rotation_deg.append(0.0)
        return rotation_deg

    def _normalize_sensor_geometry_config(self, config):
        default = self._default_sensor_geometry_config()
        if not isinstance(config, dict):
            return dict(default)

        shape = str(config.get("shape", default["shape"]) or default["shape"]).strip().lower()
        if shape not in ("flat", "cylinder", "custom"):
            shape = "flat"

        bend_axis = str(config.get("bend_axis", default["bend_axis"]) or default["bend_axis"]).strip().lower()
        if bend_axis not in ("columns", "rows"):
            bend_axis = "columns"

        try:
            arc_deg = float(config.get("arc_deg", default["arc_deg"]))
        except Exception:
            arc_deg = 0.0
        arc_deg = float(np.clip(arc_deg, -180.0, 180.0))

        if "use_selected_shape" in config:
            use_selected_shape = bool(config.get("use_selected_shape"))
        else:
            use_selected_shape = shape != "flat" or abs(arc_deg) > 1e-6

        if "rotation_deg" in config:
            rotation_deg = self._normalize_sensor_rotation_degrees(config.get("rotation_deg"))
        else:
            rotation_deg = self._normalize_sensor_rotation_degrees([
                config.get("rotation_x_deg", default["rotation_deg"][0]),
                config.get("rotation_y_deg", default["rotation_deg"][1]),
                config.get("rotation_z_deg", default["rotation_deg"][2]),
            ])

        custom_points = normalize_point_array(config.get("custom_points", []))
        custom_heatmap_points = normalize_point_array(
            config.get("custom_heatmap_points", [])
        )
        custom_heatmap_corners = normalize_point_array(
            config.get("custom_heatmap_corners", [])
        )
        custom_heatmap_edge_offsets = normalize_point_array(
            config.get("custom_heatmap_edge_offsets", [])
        )

        return {
            "use_selected_shape": use_selected_shape,
            "shape": shape,
            "bend_axis": bend_axis,
            "arc_deg": arc_deg,
            "normal_flip": bool(config.get("normal_flip", default["normal_flip"])),
            "rotation_deg": rotation_deg,
            "custom_points": (
                custom_points.tolist() if custom_points is not None else []
            ),
            "use_custom_heatmap_shape": bool(
                config.get(
                    "use_custom_heatmap_shape",
                    default["use_custom_heatmap_shape"],
                )
            ),
            "custom_heatmap_points": (
                custom_heatmap_points.tolist()
                if custom_heatmap_points is not None
                else []
            ),
            "custom_heatmap_corners": (
                custom_heatmap_corners.tolist()
                if custom_heatmap_corners is not None
                else []
            ),
            "use_curved_heatmap_edges": bool(
                config.get(
                    "use_curved_heatmap_edges",
                    default["use_curved_heatmap_edges"],
                )
            ),
            "custom_heatmap_edge_offsets": (
                custom_heatmap_edge_offsets.tolist()
                if custom_heatmap_edge_offsets is not None
                else []
            ),
        }

    def _effective_sensor_geometry_config(self, config):
        geometry = self._normalize_sensor_geometry_config(config)
        if not bool(geometry.get("use_selected_shape", False)):
            effective = dict(geometry)
            effective["shape"] = "flat"
            effective["arc_deg"] = 0.0
            effective["normal_flip"] = False
            return effective
        return geometry

    def get_saved_sensor_geometry_config(self, model_name=None, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        item = payload.get("settings", {}).get(key, {})
        geometry = item.get("geometry", {}) if isinstance(item, dict) else {}
        return self._normalize_sensor_geometry_config(geometry)

    def set_saved_sensor_geometry_config(self, model_name, config, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        if n_row is None or n_col is None:
            n_row, n_col = self._sensor_shape_for_model(model)
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        settings = payload.setdefault("settings", {})
        item = settings.get(key, {})
        if not isinstance(item, dict):
            item = {}
        item.update({
            "model": model,
            "n_row": int(n_row),
            "n_col": int(n_col),
            "geometry": self._normalize_sensor_geometry_config(config),
        })
        settings[key] = item
        return self._write_reorder_logic_file(payload)

    @staticmethod
    def _default_stereo_field_config():
        return {
            "ignore_noise_enabled": True,
            "deadband_pct": 0.35,
            "response_scale_pct": 2.0,
            "length_scale": 0.35,
        }

    def _normalize_stereo_field_config(self, config):
        default = self._default_stereo_field_config()
        if not isinstance(config, dict):
            return dict(default)

        try:
            deadband_pct = float(config.get("deadband_pct", default["deadband_pct"]))
        except Exception:
            deadband_pct = default["deadband_pct"]
        try:
            response_scale_pct = float(config.get("response_scale_pct", default["response_scale_pct"]))
        except Exception:
            response_scale_pct = default["response_scale_pct"]
        try:
            length_scale = float(config.get("length_scale", default["length_scale"]))
        except Exception:
            length_scale = default["length_scale"]

        return {
            "ignore_noise_enabled": bool(
                config.get("ignore_noise_enabled", default["ignore_noise_enabled"])
            ),
            "deadband_pct": float(np.clip(deadband_pct, 0.0, 20.0)),
            "response_scale_pct": float(np.clip(response_scale_pct, 0.05, 50.0)),
            "length_scale": float(np.clip(length_scale, 0.05, 2.0)),
        }

    def get_saved_sensor_stereo_field_config(self, model_name=None, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        item = payload.get("settings", {}).get(key, {})
        config = item.get("stereo_field", {}) if isinstance(item, dict) else {}
        return self._normalize_stereo_field_config(config)

    def set_saved_sensor_stereo_field_config(self, model_name, config, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        if n_row is None or n_col is None:
            n_row, n_col = self._sensor_shape_for_model(model)
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        settings = payload.setdefault("settings", {})
        item = settings.get(key, {})
        if not isinstance(item, dict):
            item = {}
        item.update({
            "model": model,
            "n_row": int(n_row),
            "n_col": int(n_col),
            "stereo_field": self._normalize_stereo_field_config(config),
        })
        settings[key] = item
        return self._write_reorder_logic_file(payload)

    def get_stereo_field_settings(self):
        return self._normalize_stereo_field_config({
            "ignore_noise_enabled": getattr(self, "stereo_field_ignore_noise_enabled", True),
            "deadband_pct": getattr(self, "stereo_field_deadband_pct", 0.35),
            "response_scale_pct": getattr(self, "stereo_field_response_scale_pct", 2.0),
            "length_scale": getattr(self, "stereo_field_length_scale", 0.35),
        })

    def set_stereo_field_settings(self, config, save_current_sensor: bool = False):
        settings = self._normalize_stereo_field_config(config)
        self.stereo_field_ignore_noise_enabled = bool(settings["ignore_noise_enabled"])
        self.stereo_field_deadband_pct = float(settings["deadband_pct"])
        self.stereo_field_response_scale_pct = float(settings["response_scale_pct"])
        previous_length_scale = float(getattr(self, "stereo_field_length_scale", 0.35) or 0.35)
        self.stereo_field_length_scale = float(settings["length_scale"])
        self._stereo_field_smoothed_visibility = None
        self._stereo_field_smoothed_color_response = None
        if abs(previous_length_scale - self.stereo_field_length_scale) > 1e-9:
            self._rebuild_matrix_visualization_actor(render=False)
            self._refresh_sensor_visualization_mode_actors()
        if save_current_sensor and self.current_model_name:
            self.set_saved_sensor_stereo_field_config(
                self.current_model_name,
                settings,
                n_row=self.n_row,
                n_col=self.n_col,
            )
        return True

    @staticmethod
    def _default_heatmap_config():
        return {
            "palette_3d": DEFAULT_HEATMAP_3D_PALETTE,
            "response_mode": DEFAULT_HEATMAP_RESPONSE_MODE,
            "saturation_pct": DEFAULT_HEATMAP_SATURATION_PCT,
            "noise_floor_pct": DEFAULT_HEATMAP_NOISE_FLOOR_PCT,
            "proximity_noise_floor": DEFAULT_PROXIMITY_NOISE_FLOOR,
            "proximity_knee": DEFAULT_PROXIMITY_KNEE,
            "proximity_saturation": DEFAULT_PROXIMITY_SATURATION,
            "color_gain_3d": DEFAULT_HEATMAP_3D_COLOR_GAIN,
            "use_absolute_signal": True,
            "point_grid_response_mode": POINT_GRID_RESPONSE_ZERO_CENTERED,
        }

    def _normalize_heatmap_config(self, config):
        default = self._default_heatmap_config()
        if not isinstance(config, dict):
            return dict(default)
        try:
            saturation_pct = float(
                config.get("saturation_pct", default["saturation_pct"])
            )
        except Exception:
            saturation_pct = default["saturation_pct"]
        try:
            noise_floor_pct = float(
                config.get("noise_floor_pct", default["noise_floor_pct"])
            )
        except Exception:
            noise_floor_pct = default["noise_floor_pct"]
        try:
            proximity_noise_floor = float(
                config.get(
                    "proximity_noise_floor", default["proximity_noise_floor"]
                )
            )
        except Exception:
            proximity_noise_floor = default["proximity_noise_floor"]
        try:
            proximity_knee = float(
                config.get("proximity_knee", default["proximity_knee"])
            )
        except Exception:
            proximity_knee = default["proximity_knee"]
        try:
            proximity_saturation = float(
                config.get(
                    "proximity_saturation", default["proximity_saturation"]
                )
            )
        except Exception:
            proximity_saturation = default["proximity_saturation"]
        try:
            color_gain_3d = float(
                config.get("color_gain_3d", default["color_gain_3d"])
            )
        except Exception:
            color_gain_3d = default["color_gain_3d"]
        proximity_noise_floor = float(np.clip(proximity_noise_floor, 0.0, 1e9))
        proximity_knee = float(
            np.clip(proximity_knee, proximity_noise_floor + 0.1, 1e9)
        )
        proximity_saturation = float(
            np.clip(proximity_saturation, proximity_knee + 0.1, 1e9)
        )
        point_grid_response_mode = str(
            config.get(
                "point_grid_response_mode",
                default["point_grid_response_mode"],
            )
        )
        if point_grid_response_mode not in (
            POINT_GRID_RESPONSE_ZERO_CENTERED,
            POINT_GRID_RESPONSE_LEGACY_OFFSET,
        ):
            point_grid_response_mode = default["point_grid_response_mode"]
        return {
            "palette_3d": normalize_heatmap_3d_palette(
                config.get("palette_3d", default["palette_3d"])
            ),
            "response_mode": normalize_heatmap_response_mode(
                config.get("response_mode", default["response_mode"])
            ),
            "saturation_pct": float(np.clip(saturation_pct, 0.1, 50.0)),
            "noise_floor_pct": float(np.clip(noise_floor_pct, 0.0, 20.0)),
            "proximity_noise_floor": proximity_noise_floor,
            "proximity_knee": proximity_knee,
            "proximity_saturation": proximity_saturation,
            "color_gain_3d": float(np.clip(color_gain_3d, 1.0, 3.0)),
            "use_absolute_signal": bool(
                config.get(
                    "use_absolute_signal",
                    default["use_absolute_signal"],
                )
            ),
            "point_grid_response_mode": point_grid_response_mode,
        }

    def get_saved_sensor_heatmap_config(self, model_name=None, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        item = payload.get("settings", {}).get(key, {})
        config = item.get("heatmap", {}) if isinstance(item, dict) else {}
        return self._normalize_heatmap_config(config)

    def set_saved_sensor_heatmap_config(self, model_name, config, n_row=None, n_col=None):
        model = str(model_name or self.current_model_name or "sensor")
        if n_row is None or n_col is None:
            n_row, n_col = self._sensor_shape_for_model(model)
        key = self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col)
        payload = self._read_reorder_logic_file()
        settings = payload.setdefault("settings", {})
        item = settings.get(key, {})
        if not isinstance(item, dict):
            item = {}
        item.update({
            "model": model,
            "n_row": int(n_row),
            "n_col": int(n_col),
            "heatmap": self._normalize_heatmap_config(config),
        })
        settings[key] = item
        return self._write_reorder_logic_file(payload)

    def get_heatmap_settings(self):
        return self._normalize_heatmap_config({
            "palette_3d": getattr(
                self, "heatmap_3d_palette", DEFAULT_HEATMAP_3D_PALETTE
            ),
            "response_mode": getattr(
                self, "heatmap_response_mode", DEFAULT_HEATMAP_RESPONSE_MODE
            ),
            "saturation_pct": getattr(
                self, "heatmap_saturation_pct", DEFAULT_HEATMAP_SATURATION_PCT
            ),
            "noise_floor_pct": getattr(
                self, "heatmap_noise_floor_pct", DEFAULT_HEATMAP_NOISE_FLOOR_PCT
            ),
            "proximity_noise_floor": getattr(
                self,
                "heatmap_proximity_noise_floor",
                DEFAULT_PROXIMITY_NOISE_FLOOR,
            ),
            "proximity_knee": getattr(
                self, "heatmap_proximity_knee", DEFAULT_PROXIMITY_KNEE
            ),
            "proximity_saturation": getattr(
                self,
                "heatmap_proximity_saturation",
                DEFAULT_PROXIMITY_SATURATION,
            ),
            "color_gain_3d": getattr(
                self,
                "heatmap_3d_color_gain",
                DEFAULT_HEATMAP_3D_COLOR_GAIN,
            ),
            "use_absolute_signal": getattr(
                self,
                "visualization_use_absolute_signal",
                True,
            ),
            "point_grid_response_mode": getattr(
                self,
                "point_grid_response_mode",
                POINT_GRID_RESPONSE_ZERO_CENTERED,
            ),
        })

    def set_heatmap_settings(self, config, save_current_sensor: bool = False):
        config = dict(config) if isinstance(config, dict) else {}
        config.setdefault(
            "use_absolute_signal",
            getattr(self, "visualization_use_absolute_signal", True),
        )
        config.setdefault(
            "point_grid_response_mode",
            getattr(
                self,
                "point_grid_response_mode",
                POINT_GRID_RESPONSE_ZERO_CENTERED,
            ),
        )
        settings = self._normalize_heatmap_config(config)
        previous_settings = self.get_heatmap_settings()
        settings_changed = any(
            previous_settings.get(key) != settings.get(key)
            for key in settings
        )
        self.heatmap_3d_palette = str(settings["palette_3d"])
        self.heatmap_response_mode = str(settings["response_mode"])
        self.heatmap_saturation_pct = float(settings["saturation_pct"])
        self.heatmap_noise_floor_pct = float(settings["noise_floor_pct"])
        self.heatmap_proximity_noise_floor = float(
            settings["proximity_noise_floor"]
        )
        self.heatmap_proximity_knee = float(settings["proximity_knee"])
        self.heatmap_proximity_saturation = float(
            settings["proximity_saturation"]
        )
        self.heatmap_3d_color_gain = float(settings["color_gain_3d"])
        self.visualization_use_absolute_signal = bool(
            settings["use_absolute_signal"]
        )
        self.point_grid_response_mode = str(
            settings["point_grid_response_mode"]
        )
        if settings_changed:
            current_matrix = self._current_heatmap_sensor_matrix(
                getattr(self, "_last_sensor_visualization_matrix", None)
            )
            if current_matrix is not None:
                self._last_sensor_visualization_matrix = np.array(
                    current_matrix, dtype=float, copy=True
                )
            if (
                current_matrix is not None
                and getattr(self, "heatmapPoly", None) is not None
            ):
                self._update_heatmap_visualization(current_matrix)
                if self._is_heatmap_visualization_mode(
                    getattr(self, "sensor_visualization_mode", "point_grid")
                ):
                    try:
                        self.plotter.render()
                    except Exception:
                        pass
        if save_current_sensor and self.current_model_name:
            self.set_saved_sensor_heatmap_config(
                self.current_model_name,
                settings,
                n_row=self.n_row,
                n_col=self.n_col,
            )
        return True

    def get_sensor_reorder_context(self, model_name=None):
        model = str(model_name or self.current_model_name or "sensor")
        n_row, n_col = self._sensor_shape_for_model(model)
        saved_mode = self.get_saved_sensor_reorder_mode(model, n_row=n_row, n_col=n_col)
        default_logic = self._default_reorder_logic_for_model(model)
        effective_logic = self._reorder_mode_to_logic(model, saved_mode)
        point_labels_enabled = self.get_saved_sensor_point_labels_enabled(
            model,
            n_row=n_row,
            n_col=n_col,
        )
        background_reference_enabled = self.get_saved_sensor_background_reference_enabled(
            model,
            n_row=n_row,
            n_col=n_col,
        )
        force_scale = self.get_saved_sensor_contact_force_scale(
            model,
            n_row=n_row,
            n_col=n_col,
        )
        geometry = self.get_saved_sensor_geometry_config(
            model,
            n_row=n_row,
            n_col=n_col,
        )
        stereo_field = self.get_saved_sensor_stereo_field_config(
            model,
            n_row=n_row,
            n_col=n_col,
        )
        heatmap = self.get_saved_sensor_heatmap_config(
            model,
            n_row=n_row,
            n_col=n_col,
        )
        camera = self.get_saved_sensor_camera_config(
            model,
            n_row=n_row,
            n_col=n_col,
        )
        return {
            "model": model,
            "label": self.SENSOR_MODEL_LABELS.get(model, model),
            "n_row": int(n_row),
            "n_col": int(n_col),
            "key": self.get_sensor_reorder_key(model, n_row=n_row, n_col=n_col),
            "saved_mode": saved_mode,
            "default_logic": default_logic,
            "effective_logic": effective_logic,
            "point_labels_enabled": point_labels_enabled,
            "background_reference_enabled": background_reference_enabled,
            "force_scale_n_per_signal": force_scale,
            "geometry": geometry,
            "stereo_field": stereo_field,
            "heatmap": heatmap,
            "camera": camera,
        }

    def _apply_saved_reorder_logic(self, model_kwargs):
        model = str(self.current_model_name or "sensor")
        n_row = int(model_kwargs.get("n_row", 0) or 0)
        n_col = int(model_kwargs.get("n_col", 0) or 0)
        mode = self.get_saved_sensor_reorder_mode(model, n_row=n_row, n_col=n_col)
        effective_logic = self._reorder_mode_to_logic(model, mode)
        updated = dict(model_kwargs)
        if effective_logic is None:
            updated.pop("reorder_logic", None)
        else:
            updated["reorder_logic"] = effective_logic
        self.current_reorder_mode = mode
        self.current_effective_reorder_logic = effective_logic
        self.current_sensor_reorder_key = self.get_sensor_reorder_key(
            model,
            n_row=n_row,
            n_col=n_col,
        )
        self.show_sensor_point_labels = self.get_saved_sensor_point_labels_enabled(
            model,
            n_row=n_row,
            n_col=n_col,
        )
        self.set_sensor_background_reference_enabled(
            self.get_saved_sensor_background_reference_enabled(
                model,
                n_row=n_row,
                n_col=n_col,
            ),
            save_current_sensor=False,
            render=False,
        )
        self.contact_force_scale_n_per_signal = self.get_saved_sensor_contact_force_scale(
            model,
            n_row=n_row,
            n_col=n_col,
        )
        self.set_stereo_field_settings(
            self.get_saved_sensor_stereo_field_config(
                model,
                n_row=n_row,
                n_col=n_col,
            ),
            save_current_sensor=False,
        )
        self.set_heatmap_settings(
            self.get_saved_sensor_heatmap_config(
                model,
                n_row=n_row,
                n_col=n_col,
            ),
            save_current_sensor=False,
        )
        print(
            "[SensorReorder] "
            f"{self.current_sensor_reorder_key} "
            f"mode={mode}, effective={effective_logic or 'none'}"
        )
        return updated

    def set_sensor_point_labels_enabled(self, enabled: bool, save_current_sensor: bool = False):
        self.show_sensor_point_labels = bool(enabled)
        if save_current_sensor and self.current_model_name:
            self.set_saved_sensor_point_labels_enabled(
                self.current_model_name,
                self.show_sensor_point_labels,
                n_row=self.n_row,
                n_col=self.n_col,
            )
        self._refresh_sensor_point_label_actor()
        try:
            self.plotter.render()
        except Exception:
            pass

    @staticmethod
    def _set_actor_visible(actor, visible):
        if actor is None:
            return
        try:
            actor.SetVisibility(bool(visible))
            return
        except Exception:
            pass
        try:
            actor.visibility = bool(visible)
        except Exception:
            pass

    def _apply_sensor_background_reference_visibility(self, render: bool = True):
        visible = bool(getattr(self, "show_sensor_background_reference", True))
        for actor in getattr(self, "referenceAxisActors", []) or []:
            self._set_actor_visible(actor, visible)
        self._set_actor_visible(getattr(self, "actorPlaneXY", None), visible)
        mesh_functions = getattr(getattr(self, "parent", None), "mesh_functions", None)
        if mesh_functions is not None and hasattr(
            mesh_functions,
            "set_secondary_background_reference_enabled",
        ):
            try:
                mesh_functions.set_secondary_background_reference_enabled(visible, render=False)
            except Exception:
                pass
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def set_sensor_background_reference_enabled(
        self,
        enabled: bool,
        save_current_sensor: bool = False,
        render: bool = True,
    ):
        self.show_sensor_background_reference = bool(enabled)
        if save_current_sensor and self.current_model_name:
            self.set_saved_sensor_background_reference_enabled(
                self.current_model_name,
                self.show_sensor_background_reference,
                n_row=self.n_row,
                n_col=self.n_col,
            )
        self._apply_sensor_background_reference_visibility(render=render)

    def set_sensor_contact_force_scale(self, scale, save_current_sensor: bool = False):
        try:
            self.contact_force_scale_n_per_signal = max(0.0, float(scale))
        except Exception:
            self.contact_force_scale_n_per_signal = 0.0
        if save_current_sensor and self.current_model_name:
            self.set_saved_sensor_contact_force_scale(
                self.current_model_name,
                self.contact_force_scale_n_per_signal,
                n_row=self.n_row,
                n_col=self.n_col,
            )

    def _capture_sensor_geometry_base(self):
        self._sensor_geometry_base_points_origin = (
            np.array(self.points_origin, dtype=float, copy=True)
            if self.points_origin is not None
            else None
        )
        self._sensor_geometry_base_normals = (
            np.array(self.normals, dtype=float, copy=True)
            if self.normals is not None
            else None
        )
        self._sensor_geometry_base_fine_points = (
            np.array(self._2D_map.points, dtype=float, copy=True)
            if self._2D_map is not None and getattr(self._2D_map, "n_points", 0) > 0
            else None
        )

    def _sensor_local_geometry(self, config):
        base_points = getattr(self, "_sensor_geometry_base_points_origin", None)
        base_normals = getattr(self, "_sensor_geometry_base_normals", None)
        if base_points is None or base_normals is None:
            return None, None

        expected_count = int(self.n_row) * int(self.n_col)
        if str(config.get("shape", "flat")) == "custom":
            custom_points = normalize_point_array(
                config.get("custom_points", []), expected_count
            )
            if custom_points is not None:
                custom_normals = compute_grid_normals(
                    custom_points,
                    self.n_row,
                    self.n_col,
                    normal_flip=bool(config.get("normal_flip", False)),
                )
                return custom_points, custom_normals

        return self._bend_points_to_cylinder(
            base_points,
            config,
            return_normals=True,
            base_normals=base_normals,
        )

    def get_sensor_geometry_editor_data(self, config=None):
        """Return editable local geometry for the currently built 2D sensor."""
        if str(self.current_model_name or "") != "2d":
            return None
        saved_geometry = self._normalize_sensor_geometry_config(
            config if config is not None else self.current_sensor_geometry_config
        )
        geometry = self._effective_sensor_geometry_config(saved_geometry)
        points, normals = self._sensor_local_geometry(geometry)
        flat_points = normalize_point_array(
            getattr(self, "_sensor_geometry_base_points_origin", None),
            int(self.n_row) * int(self.n_col),
        )
        if points is None or normals is None or flat_points is None:
            return None
        expected_count = int(self.n_row) * int(self.n_col)
        heatmap_points = normalize_point_array(
            saved_geometry.get("custom_heatmap_points", []),
            expected_count,
        )
        if heatmap_points is None:
            heatmap_points = np.array(points, dtype=float, copy=True)
        heatmap_normals = compute_grid_normals(
            heatmap_points,
            self.n_row,
            self.n_col,
            normal_flip=bool(geometry.get("normal_flip", False)),
        )
        default_heatmap_corners = heatmap_corner_points_from_centres(
            points,
            normals,
            self.n_row,
            self.n_col,
        )
        custom_heatmap_corners = normalize_point_array(
            saved_geometry.get("custom_heatmap_corners", []),
            (int(self.n_row) + 1) * (int(self.n_col) + 1),
        )
        if custom_heatmap_corners is None:
            custom_heatmap_corners = heatmap_corner_points_from_centres(
                heatmap_points,
                heatmap_normals,
                self.n_row,
                self.n_col,
            )
        heatmap_edge_pairs = structured_grid_edge_pairs(
            int(self.n_row) + 1,
            int(self.n_col) + 1,
        )
        heatmap_edge_offsets = normalize_point_array(
            saved_geometry.get("custom_heatmap_edge_offsets", []),
            len(heatmap_edge_pairs),
        )
        if heatmap_edge_offsets is None:
            heatmap_edge_offsets = np.zeros(
                (len(heatmap_edge_pairs), 3), dtype=float
            )
        return {
            "points": np.array(points, dtype=float, copy=True),
            "normals": np.array(normals, dtype=float, copy=True),
            "flat_points": flat_points,
            "heatmap_points": np.array(heatmap_points, dtype=float, copy=True),
            "heatmap_default_points": np.array(points, dtype=float, copy=True),
            "heatmap_corners": np.array(
                custom_heatmap_corners, dtype=float, copy=True
            ),
            "heatmap_default_corners": np.array(
                default_heatmap_corners, dtype=float, copy=True
            ),
            "heatmap_edges": structured_grid_edges(
                int(self.n_row) + 1,
                int(self.n_col) + 1,
            ),
            "heatmap_n_row": int(self.n_row) + 1,
            "heatmap_n_col": int(self.n_col) + 1,
            "heatmap_edge_pairs": heatmap_edge_pairs,
            "heatmap_edge_offsets": heatmap_edge_offsets,
            "edges": np.array(self.edges, dtype=int, copy=True),
            "n_row": int(self.n_row),
            "n_col": int(self.n_col),
            "geometry": saved_geometry,
        }

    @staticmethod
    def _bend_points_to_cylinder(points, config, return_normals=False, base_normals=None):
        points_np = np.array(points, dtype=float, copy=True)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            if return_normals:
                return points_np, base_normals
            return points_np

        normalized = {
            "use_selected_shape": bool(config.get("use_selected_shape", True)),
            "shape": str(config.get("shape", "flat")),
            "bend_axis": str(config.get("bend_axis", "columns")),
            "arc_deg": float(config.get("arc_deg", 0.0) or 0.0),
            "normal_flip": bool(config.get("normal_flip", False)),
        }

        normals = None
        if return_normals:
            if base_normals is None:
                normals = np.tile([0.0, 0.0, 1.0], (len(points_np), 1))
            else:
                normals = np.array(base_normals, dtype=float, copy=True)

        if (
            not normalized["use_selected_shape"]
            or normalized["shape"] != "cylinder"
            or abs(normalized["arc_deg"]) < 1e-6
        ):
            if normals is not None and normalized["normal_flip"]:
                normals *= -1.0
            return (points_np, normals) if return_normals else points_np

        axis_idx = 0 if normalized["bend_axis"] == "columns" else 1
        values = points_np[:, axis_idx]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            if normals is not None and normalized["normal_flip"]:
                normals *= -1.0
            return (points_np, normals) if return_normals else points_np

        width = float(np.max(finite) - np.min(finite))
        if width <= 1e-9:
            if normals is not None and normalized["normal_flip"]:
                normals *= -1.0
            return (points_np, normals) if return_normals else points_np

        center = float((np.max(finite) + np.min(finite)) * 0.5)
        arc_rad = float(np.radians(abs(normalized["arc_deg"])))
        if arc_rad <= 1e-9:
            if normals is not None and normalized["normal_flip"]:
                normals *= -1.0
            return (points_np, normals) if return_normals else points_np

        curve_sign = 1.0 if normalized["arc_deg"] >= 0.0 else -1.0
        radius = width / arc_rad
        u = values - center
        theta = u / max(radius, 1e-9)
        points_np[:, axis_idx] = center + radius * np.sin(theta)
        points_np[:, 2] = points_np[:, 2] + curve_sign * radius * (np.cos(theta) - 1.0)

        if normals is not None:
            normals = np.zeros_like(points_np)
            normals[:, axis_idx] = curve_sign * np.sin(theta)
            normals[:, 2] = np.cos(theta)
            norm = np.linalg.norm(normals, axis=1)
            valid = norm > 1e-9
            normals[valid] = normals[valid] / norm[valid, None]
            normals[~valid] = [0.0, 0.0, 1.0]
            if normalized["normal_flip"]:
                normals *= -1.0
            return points_np, normals

        return points_np

    @classmethod
    def _sensor_geometry_rotation_matrix_deg(cls, rotation_deg):
        rx_deg, ry_deg, rz_deg = cls._normalize_sensor_rotation_degrees(rotation_deg)
        rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        cos_z, sin_z = np.cos(rz), np.sin(rz)

        rot_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_x, -sin_x],
            [0.0, sin_x, cos_x],
        ])
        rot_y = np.array([
            [cos_y, 0.0, sin_y],
            [0.0, 1.0, 0.0],
            [-sin_y, 0.0, cos_y],
        ])
        rot_z = np.array([
            [cos_z, -sin_z, 0.0],
            [sin_z, cos_z, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return rot_z @ rot_y @ rot_x

    @staticmethod
    def _sensor_geometry_rotation_pivot(points):
        points_np = np.array(points, dtype=float, copy=False)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            return np.zeros(3, dtype=float)
        finite_rows = np.all(np.isfinite(points_np), axis=1)
        if not np.any(finite_rows):
            return np.zeros(3, dtype=float)
        return np.mean(points_np[finite_rows], axis=0)

    @classmethod
    def _rotate_sensor_geometry(cls, points, rotation_deg, normals=None, pivot=None):
        points_np = np.array(points, dtype=float, copy=True)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            if normals is None:
                return points_np
            return points_np, np.array(normals, dtype=float, copy=True)

        rotation = cls._sensor_geometry_rotation_matrix_deg(rotation_deg)
        if pivot is None:
            pivot_np = cls._sensor_geometry_rotation_pivot(points_np)
        else:
            pivot_np = np.array(pivot, dtype=float, copy=False)
            if pivot_np.shape != (3,) or not np.all(np.isfinite(pivot_np)):
                pivot_np = cls._sensor_geometry_rotation_pivot(points_np)

        finite_rows = np.all(np.isfinite(points_np), axis=1)
        if np.any(finite_rows):
            points_np[finite_rows] = (points_np[finite_rows] - pivot_np) @ rotation.T + pivot_np

        if normals is None:
            return points_np

        normals_np = np.array(normals, dtype=float, copy=True)
        if normals_np.ndim != 2 or normals_np.shape[1] != 3:
            return points_np, normals_np
        normals_np = normals_np @ rotation.T
        normal_norms = np.linalg.norm(normals_np, axis=1)
        valid_norms = np.isfinite(normal_norms) & (normal_norms > 1e-9)
        normals_np[valid_norms] = normals_np[valid_norms] / normal_norms[valid_norms, None]
        normals_np[~valid_norms] = [0.0, 0.0, 1.0]
        return points_np, normals_np

    def set_sensor_geometry_config(self, config, save_current_sensor: bool = False, render: bool = True):
        geometry = self._normalize_sensor_geometry_config(config)
        self.current_sensor_geometry_config = geometry
        effective_geometry = self._effective_sensor_geometry_config(geometry)

        if save_current_sensor and self.current_model_name:
            self.set_saved_sensor_geometry_config(
                self.current_model_name,
                geometry,
                n_row=self.n_row,
                n_col=self.n_col,
            )

        if str(self.current_model_name or "") != "2d":
            return False

        base_points = getattr(self, "_sensor_geometry_base_points_origin", None)
        base_normals = getattr(self, "_sensor_geometry_base_normals", None)
        if base_points is None or base_normals is None:
            return False

        bent_points, bent_normals = self._sensor_local_geometry(
            effective_geometry
        )
        if bent_points is None or bent_normals is None:
            return False
        rotation_deg = effective_geometry.get("rotation_deg", [0.0, 0.0, 0.0])
        rotation_pivot = self._sensor_geometry_rotation_pivot(bent_points)
        self.points_origin, self.normals = self._rotate_sensor_geometry(
            bent_points,
            rotation_deg,
            normals=bent_normals,
            pivot=rotation_pivot,
        )
        self.points = self.points_origin + self.normals * float(
            getattr(self, "sensor_visual_offset_scale", 0.0)
        )

        base_fine_points = getattr(self, "_sensor_geometry_base_fine_points", None)
        if self._2D_map is not None and base_fine_points is not None:
            if str(effective_geometry.get("shape", "flat")) == "custom":
                fine_points = interpolate_coarse_deformation(
                    base_points,
                    bent_points,
                    base_fine_points,
                )
            else:
                fine_points = self._bend_points_to_cylinder(
                    base_fine_points,
                    effective_geometry,
                    return_normals=False,
                )
            self._2D_map.points = self._rotate_sensor_geometry(
                fine_points,
                rotation_deg,
                pivot=rotation_pivot,
            )
            try:
                self._2D_map.Modified()
            except Exception:
                pass

        if self.line_poly is not None:
            self.line_poly.points = self._sensor_points_for_plotter(self.points)
            try:
                self.line_poly.Modified()
            except Exception:
                pass

        if self.actionMesh is not None or self.matrixLineActor is not None:
            self._rebuild_matrix_visualization_actor(render=False)
        if (
            getattr(self, "heatmapActor", None) is not None
            or getattr(self, "heatmapPoly", None) is not None
            or self._is_heatmap_visualization_mode(
                getattr(self, "sensor_visualization_mode", "point_grid")
            )
        ):
            self._rebuild_heatmap_visualization_actor(render=False)
            current_matrix = getattr(self, "_last_sensor_visualization_matrix", None)
            if current_matrix is not None:
                self._update_heatmap_visualization(current_matrix)
        self._refresh_sensor_visualization_mode_actors()
        self._clear_contact_normal_actor()
        self._refresh_sensor_point_label_actor()
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass
        return True

    def get_cell_point_label(self, row, col):
        """Return the user-facing top-down label used by mask and plotter UIs."""
        row = int(row)
        col = int(col)
        point_id = _column_major_idx(int(self.n_row), col, row)
        return f"P{point_id} r{row} c{col}"

    def _selected_sensor_plotter_location(self):
        """Map a Signal Viewer cell directly onto the same 3D grid cell."""
        selected = getattr(self, "_selected_sensor_cell", None)
        if selected is None:
            return None
        try:
            table_row, col = (int(selected[0]), int(selected[1]))
        except (TypeError, ValueError, IndexError):
            return None
        n_row = int(getattr(self, "n_row", 0) or 0)
        n_col = int(getattr(self, "n_col", 0) or 0)
        if not (0 <= table_row < n_row and 0 <= col < n_col):
            return None
        point_index = _column_major_idx(n_row, col, table_row)
        return table_row, col, point_index

    def _selected_sensor_marker_position(self):
        location = self._selected_sensor_plotter_location()
        if location is None:
            return None
        plotter_row, col, point_index = location

        position = None
        if self._is_heatmap_visualization_mode(
            getattr(self, "sensor_visualization_mode", "point_grid")
        ):
            heatmap_poly = getattr(self, "heatmapPoly", None)
            tile_vertices = getattr(self, "_heatmap_tile_vertices", None)
            if heatmap_poly is not None and tile_vertices is not None:
                try:
                    vertex_ids = tile_vertices[plotter_row, col]
                    position = np.mean(
                        np.asarray(heatmap_poly.points, dtype=float)[vertex_ids],
                        axis=0,
                    )
                except (IndexError, TypeError, ValueError):
                    position = None

        points = getattr(self, "points", None)
        if position is None and points is not None:
            try:
                position = np.asarray(points, dtype=float)[point_index].copy()
            except (IndexError, TypeError, ValueError):
                return None
        if position is None or not np.all(np.isfinite(position)):
            return None

        normals = getattr(self, "normals", None)
        try:
            normal = self._normalize_vector(
                np.asarray(normals, dtype=float)[point_index]
            )
        except (IndexError, TypeError, ValueError):
            normal = np.array([0.0, 0.0, 1.0], dtype=float)

        origin = getattr(self, "points_origin", None)
        try:
            finite = np.asarray(origin, dtype=float)
            finite = finite[np.all(np.isfinite(finite), axis=1)]
            span = float(
                np.linalg.norm(np.max(finite, axis=0) - np.min(finite, axis=0))
            )
        except (TypeError, ValueError):
            span = 0.0
        surface_lift = max(span * 0.01, 0.0002)
        return position + normal * surface_lift

    def _refresh_sensor_selection_actor(self, render=True):
        position = self._selected_sensor_marker_position()
        enabled = bool(getattr(self, "main_visualization_enabled", True))
        if position is None:
            self._set_actor_visible(
                getattr(self, "sensorSelectionActor", None), False
            )
        else:
            marker_poly = getattr(self, "sensorSelectionPoly", None)
            if marker_poly is None:
                marker_poly = pv.PolyData(np.asarray([position], dtype=float))
                self.sensorSelectionPoly = marker_poly
            else:
                marker_poly.points = np.asarray([position], dtype=float)
                try:
                    marker_poly.Modified()
                except Exception:
                    pass
            if getattr(self, "sensorSelectionActor", None) is None:
                try:
                    self.sensorSelectionActor = self.plotter.add_mesh(
                        marker_poly,
                        color="#00e676",
                        point_size=22,
                        render_points_as_spheres=True,
                        lighting=False,
                        pickable=False,
                        name=self._sensor_actor_name("sensor_selected_cell"),
                        render=False,
                    )
                except Exception as exc:
                    self.sensorSelectionActor = None
                    print(
                        "[SensorSelection] Failed to show selected cell: "
                        f"{exc}"
                    )
            self._set_actor_visible(
                getattr(self, "sensorSelectionActor", None), enabled
            )
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass

    def set_selected_sensor_cell(self, row=None, col=None, render=True):
        """Highlight one top-down Signal Viewer cell in the 3D plotter."""
        if row is None or col is None:
            self._selected_sensor_cell = None
            self._refresh_sensor_selection_actor(render=render)
            return True
        try:
            row = int(row)
            col = int(col)
        except (TypeError, ValueError):
            return False
        if not (0 <= row < int(self.n_row) and 0 <= col < int(self.n_col)):
            return False
        self._selected_sensor_cell = (row, col)
        self._refresh_sensor_selection_actor(render=render)
        return True

    def _sensor_point_labels(self):
        labels = []
        for col in range(int(self.n_col)):
            for row in range(int(self.n_row)):
                labels.append(self.get_cell_point_label(row, col))
        return labels

    def _clear_sensor_point_label_actor(self):
        self.sensorPointLabelActor = self._remove_actor_safely(
            getattr(self, "sensorPointLabelActor", None)
        )

    def _refresh_sensor_point_label_actor(self):
        self._clear_sensor_point_label_actor()
        if not bool(getattr(self, "show_sensor_point_labels", False)):
            return
        if self.line_poly is None or self.n_row <= 0 or self.n_col <= 0:
            return
        mask = self._cell_zero_mask_for_plotter()
        visible_indices = []
        labels = []
        all_labels = self._sensor_point_labels()
        for col in range(int(self.n_col)):
            for row in range(int(self.n_row)):
                if mask[row, col]:
                    continue
                idx = _column_major_idx(self.n_row, col, row)
                visible_indices.append(idx)
                labels.append(all_labels[idx])
        if not visible_indices:
            return
        label_points = pv.PolyData(
            np.asarray(self.points, dtype=float)[visible_indices]
        )
        try:
            self.sensorPointLabelActor = self.plotter.add_point_labels(
                label_points,
                labels,
                font_size=12,
                text_color="#111111",
                point_color="#00e5ff",
                point_size=7,
                show_points=True,
                shape="rounded_rect",
                shape_color="#ffd34d",
                shape_opacity=0.92,
                margin=4,
                always_visible=True,
                name=self._sensor_actor_name("sensor_point_labels"),
                render=False,
            )
        except Exception as exc:
            self.sensorPointLabelActor = None
            print(f"[SensorPointLabels] Failed to show point labels: {exc}")

    def _bind_sensor_api_to_port(self, ser, profile=None):
        """Reuse the shared sensor API object while switching its active serial port."""
        sensor_api = self.parent.sensor_api
        current_ser = getattr(sensor_api, "ser", None)
        if (
            current_ser is not None
            and current_ser is not ser
            and current_ser not in (self.ser_list or [])
        ):
            try:
                current_ser.close()
            except Exception:
                pass
        sensor_api.ser = ser
        sensor_api.serial_port = str(getattr(ser, "port", sensor_api.serial_port))
        sensor_api.baud_rate = int(
            getattr(ser, "baudrate", DEFAULT_SENSOR_BAUD_RATE)
        )
        profile = (
            self._sensor_profile_for_port(ser.port)
            if profile is None
            else self._normalize_sensor_port_profile(profile)
        )
        sensor_api.expected_payload_values = int(profile["n_row"]) * (
            int(profile["n_col"])
            + (1 if profile["has_extra_column"] else 0)
        )
        return sensor_api

    def _raw_packet_has_extra_column(self, port_name=None):
        if port_name is not None:
            key = self._sensor_port_key(port_name)
            profile = getattr(
                self, "_sensor_profiles_by_port", {}
            ).get(key)
            if profile is not None:
                return bool(profile.get("has_extra_column", True))
        checkbox = getattr(self.parent, "sensor_extra_column_checkbox", None)
        if checkbox is None:
            return True
        try:
            return bool(checkbox.isChecked())
        except Exception:
            return True

    def _trim_sensor_payload(
        self, data_list, n_row, n_col, expected_length=None
    ):
        """Apply model-specific payload fixes before validating the sample size."""
        if expected_length is None:
            expected_length = n_row * (n_col + 1)
        if (
            n_row == 10
            and n_col == 9
            and len(data_list) == int(expected_length) + n_row
        ):
            return data_list[:-10]
        return data_list

    def _extract_sensor_values(self, data_list, n_row, n_col, port_name):
        if data_list is None:
            message = f"No sensor response received from {port_name}."
            self._set_sensor_stream_error(message)
            print(f"Error on port {port_name}: No sensor payload received.")
            return None
        has_extra_column = self._raw_packet_has_extra_column(port_name)
        packet_columns = n_col + (1 if has_extra_column else 0)
        expected_length = n_row * packet_columns
        data_list = self._trim_sensor_payload(
            data_list,
            n_row,
            n_col,
            expected_length=expected_length,
        )
        if len(data_list) != expected_length:
            packet_format = (
                f"{n_row}x({n_col}+1)"
                if has_extra_column
                else f"{n_row}x{n_col}"
            )
            self._set_sensor_stream_error(
                f"Invalid sensor response from {port_name}: received "
                f"{len(data_list)} values, expected {expected_length} "
                f"for {packet_format} packet format."
            )
            print(
                f"Error on port {port_name}: Data length is {len(data_list)}, "
                f"expected {expected_length} ({packet_format})"
            )
            return None
        if has_extra_column:
            return data_list[:-n_row]
        return data_list

    def _reshape_sensor_values(self, values, n_row, n_col):
        return reshape_sensor_values_to_row_col_matrix(values, n_row, n_col)

    def _update_data_window(self, data_obj, raw_values, n_row, n_col, window_index):
        raw_matrix = self._reshape_sensor_values(raw_values, n_row, n_col)
        # Force user-selected cells to read exactly the calibration value. That makes
        # diff/diffPer collapse to 0 for those cells everywhere downstream
        # (visualization, AI, control logic) without touching each consumer.
        self._apply_cell_zero_mask_to_raw(raw_matrix, data_obj)
        data_obj.getRaw(raw_matrix)
        data_obj.calDiff()
        data_obj.calDiffPer()
        data_obj.getWin(window_index)

    def _apply_cell_zero_mask_to_raw(self, raw_matrix, data_obj):
        mask = getattr(self, "cell_zero_mask", None)
        if mask is None or mask.size == 0 or not bool(mask.any()):
            return
        if raw_matrix.shape != mask.shape:
            return
        cal = getattr(data_obj, "calData", None)
        if cal is None or cal.shape != raw_matrix.shape:
            return
        raw_matrix[mask] = cal[mask]

    def _apply_live_raw_overrides(self, raw_values, data_obj=None):
        """Patch known bad channels for specific live sensor layouts."""
        data_obj = self._data if data_obj is None else data_obj
        if (
            not self.is_goodix_usb_transport()
            and int(getattr(data_obj, "n_row", 0)) == 10
            and int(getattr(data_obj, "n_col", 0)) == 8
        ):
            flat_cal_data = _flatten_column_major_view(data_obj.calData)
            raw_values[0] = flat_cal_data[0]
            raw_values[1] = flat_cal_data[1]
            raw_values[8] = flat_cal_data[8]
            raw_values[9] = flat_cal_data[9]
            raw_values[69] = flat_cal_data[69]
            raw_values[70] = flat_cal_data[70]
            raw_values[71] = flat_cal_data[71]
            raw_values[72] = flat_cal_data[72]
            raw_values[77] = flat_cal_data[77]
            raw_values[78] = flat_cal_data[78]
            raw_values[79] = flat_cal_data[79]

    # ------------------------------------------------------------------
    # Cell zero-mask: force a chosen subset of cells to always read 0.
    # ------------------------------------------------------------------
    def get_zero_mask_key(self):
        """Stable identifier used as the JSON key for the current sensor layout."""
        model = self.current_model_name or "sensor"
        if self.is_goodix_usb_transport():
            model = f"goodix_usb_{model}"
        n_row = int(getattr(self, "n_row", 0) or 0)
        n_col = int(getattr(self, "n_col", 0) or 0)
        return f"{model}_{n_row}x{n_col}"

    def get_cell_zero_mask(self):
        n_row = int(getattr(self, "n_row", 0) or 0)
        n_col = int(getattr(self, "n_col", 0) or 0)
        mask = getattr(self, "cell_zero_mask", None)
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != bool
            or mask.shape != (n_row, n_col)
        ):
            mask = np.zeros((n_row, n_col), dtype=bool)
            self.cell_zero_mask = mask
        return mask

    def _cell_zero_mask_for_plotter(self):
        """Use the same row and column coordinates in the mask and 3D views."""
        return np.array(self.get_cell_zero_mask(), dtype=bool, copy=True)

    def _sensor_points_for_plotter(self, points=None):
        """Return display points with masked taxels made non-renderable."""
        source = self.points if points is None else points
        try:
            display_points = np.array(source, dtype=float, copy=True)
        except Exception:
            return source
        expected_count = int(getattr(self, "n_row", 0)) * int(
            getattr(self, "n_col", 0)
        )
        if display_points.ndim != 2 or display_points.shape != (expected_count, 3):
            return display_points
        mask = self._cell_zero_mask_for_plotter()
        for col in range(int(self.n_col)):
            for row in range(int(self.n_row)):
                if mask[row, col]:
                    display_points[_column_major_idx(self.n_row, col, row)] = np.nan
        return display_points

    def _apply_cell_zero_mask_to_visual_colors(self):
        """Set masked coarse and dense point alpha to fully transparent."""
        mask = self._cell_zero_mask_for_plotter()
        colors_3d = getattr(self, "colors_3d", None)
        dense_colors = getattr(self, "colors", None)
        array_positions = getattr(self, "array_positions", None)
        if (
            isinstance(colors_3d, np.ndarray)
            and colors_3d.ndim == 2
            and colors_3d.shape[1] >= 4
        ):
            colors_3d[:, 3] = 1.0
        if (
            isinstance(dense_colors, np.ndarray)
            and dense_colors.ndim == 2
            and dense_colors.shape[1] >= 4
        ):
            dense_colors[:, 3] = 1.0
        for col in range(int(self.n_col)):
            for row in range(int(self.n_row)):
                idx = _column_major_idx(self.n_row, col, row)
                hidden = bool(mask[row, col])
                if not hidden:
                    continue
                if (
                    isinstance(colors_3d, np.ndarray)
                    and colors_3d.ndim == 2
                    and idx < colors_3d.shape[0]
                    and colors_3d.shape[1] >= 4
                ):
                    colors_3d[idx, 3] = 0.0
                if (
                    isinstance(dense_colors, np.ndarray)
                    and dense_colors.ndim == 2
                    and dense_colors.shape[1] >= 4
                    and array_positions is not None
                    and idx < len(array_positions)
                ):
                    dense_indices = np.asarray(array_positions[idx], dtype=int)
                    valid = dense_indices[
                        (dense_indices >= 0) & (dense_indices < dense_colors.shape[0])
                    ]
                    dense_colors[valid, 3] = 0.0

    def _refresh_cell_zero_mask_visualization(self, render=True):
        """Apply the current mask to every active sensor visualization actor."""
        self._apply_cell_zero_mask_to_visual_colors()

        line_poly = getattr(self, "line_poly", None)
        points = getattr(self, "points", None)
        if line_poly is not None and points is not None:
            line_poly.points = self._sensor_points_for_plotter(points)
            try:
                line_poly.Modified()
            except Exception:
                pass

        fine_poly = getattr(self, "_2D_map", None)
        if fine_poly is not None:
            try:
                fine_poly.Modified()
            except Exception:
                pass

        matrix_colors = getattr(self, "matrixLineColors", None)
        matrix_poly = getattr(self, "matrixLinePoly", None)
        dense_shape = getattr(self, "_matrix_line_dense_shape", (0, 0))
        if matrix_colors is not None and matrix_poly is not None:
            self._apply_zero_mask_to_matrix_colors(
                matrix_colors,
                dense_shape,
                base_indices=getattr(self, "_matrix_line_base_indices", None),
                top_indices=getattr(self, "_matrix_line_top_indices", None),
            )
            matrix_poly.point_data["matrix_colors"] = matrix_colors
            try:
                matrix_poly.Modified()
            except Exception:
                pass

        heatmap_colors = getattr(self, "heatmapColors", None)
        heatmap_poly = getattr(self, "heatmapPoly", None)
        tile_vertices = getattr(self, "_heatmap_tile_vertices", None)
        if (
            heatmap_colors is not None
            and heatmap_poly is not None
            and tile_vertices is not None
        ):
            visual_mask = self._cell_zero_mask_for_plotter()
            for col in range(int(self.n_col)):
                for row in range(int(self.n_row)):
                    heatmap_colors[tile_vertices[row, col], 3] = (
                        0 if visual_mask[row, col] else 255
                    )
            heatmap_poly.point_data["heatmap_colors"] = heatmap_colors
            try:
                heatmap_poly.Modified()
            except Exception:
                pass
            self._rebuild_heatmap_grid_actor(render=False)
            heatmap_mode = (
                bool(getattr(self, "main_visualization_enabled", True))
                and self._is_heatmap_visualization_mode(
                    getattr(self, "sensor_visualization_mode", "point_grid")
                )
            )
            self._set_actor_visible(
                getattr(self, "heatmapGridActor", None), heatmap_mode
            )

        if hasattr(self, "sensorPointLabelActor"):
            self._refresh_sensor_point_label_actor()
        if render and hasattr(self, "plotter"):
            try:
                self.plotter.render()
            except Exception:
                pass

    def set_cell_zero_mask(self, mask):
        n_row = int(getattr(self, "n_row", 0) or 0)
        n_col = int(getattr(self, "n_col", 0) or 0)
        if n_row <= 0 or n_col <= 0:
            return False
        try:
            arr = np.asarray(mask, dtype=bool)
        except Exception:
            return False
        if arr.shape != (n_row, n_col):
            return False
        self.cell_zero_mask = arr.copy()
        self._refresh_cell_zero_mask_visualization(render=True)
        return True

    def clear_cell_zero_mask(self):
        n_row = int(getattr(self, "n_row", 0) or 0)
        n_col = int(getattr(self, "n_col", 0) or 0)
        self.cell_zero_mask = np.zeros((n_row, n_col), dtype=bool)
        self._refresh_cell_zero_mask_visualization(render=True)

    def _read_zero_mask_file(self):
        try:
            with open(SENSOR_ZERO_MASK_FILE, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                payload.setdefault("version", 1)
                masks = payload.get("masks")
                if not isinstance(masks, dict):
                    payload["masks"] = {}
                return payload
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[ZeroMask] Failed to read {SENSOR_ZERO_MASK_FILE}: {exc}")
        return {"version": 1, "masks": {}}

    def _write_zero_mask_file(self, payload):
        try:
            os.makedirs(os.path.dirname(SENSOR_ZERO_MASK_FILE), exist_ok=True)
            with open(SENSOR_ZERO_MASK_FILE, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            return True
        except Exception as exc:
            print(f"[ZeroMask] Failed to write {SENSOR_ZERO_MASK_FILE}: {exc}")
            return False

    def save_cell_zero_mask_to_disk(self):
        n_row = int(getattr(self, "n_row", 0) or 0)
        n_col = int(getattr(self, "n_col", 0) or 0)
        if n_row <= 0 or n_col <= 0:
            return False
        payload = self._read_zero_mask_file()
        masks = payload.setdefault("masks", {})
        masks[self.get_zero_mask_key()] = self.get_cell_zero_mask().astype(int).tolist()
        return self._write_zero_mask_file(payload)

    def load_cell_zero_mask_from_disk(self):
        n_row = int(getattr(self, "n_row", 0) or 0)
        n_col = int(getattr(self, "n_col", 0) or 0)
        if n_row <= 0 or n_col <= 0:
            return False
        payload = self._read_zero_mask_file()
        masks = payload.get("masks", {}) if isinstance(payload, dict) else {}
        raw = masks.get(self.get_zero_mask_key())
        if raw is None:
            return False
        try:
            arr = np.asarray(raw, dtype=bool)
        except Exception:
            return False
        if arr.shape != (n_row, n_col):
            return False
        self.cell_zero_mask = arr.copy()
        self._refresh_cell_zero_mask_visualization(render=True)
        return True

    def _read_port_sensor_values(
        self, ser, read_operation, error_prefix, max_attempts=3
    ):
        profile = self._sensor_profile_for_port(ser.port)
        sensor_api = self._bind_sensor_api_to_port(
            ser,
            profile=profile,
        )
        max_attempts = max(1, int(max_attempts))
        sensor_values = None
        for attempt in range(1, max_attempts + 1):
            try:
                data_list = read_operation(sensor_api)
            except Exception as exc:
                self._set_sensor_stream_error(
                    f"{error_prefix} on {ser.port}: {exc}"
                )
                print(f"{error_prefix} on port {ser.port}: {exc}")
                data_list = None

            sensor_values = self._extract_sensor_values(
                data_list,
                profile["n_row"],
                profile["n_col"],
                ser.port,
            )
            if sensor_values is not None:
                return sensor_api, sensor_values

            if attempt < max_attempts:
                print(
                    f"{error_prefix} on port {ser.port}: retrying "
                    f"({attempt + 1}/{max_attempts})..."
                )

        return sensor_api, None

    def _warm_sensor_window(self, sensor_api, ser, data_obj=None):
        data_obj = self._data if data_obj is None else data_obj
        n_row = int(getattr(data_obj, "n_row", self.n_row))
        n_col = int(getattr(data_obj, "n_col", self.n_col))
        data_obj.clearData()
        for window_index in range(1, data_obj.windowSize + 1):
            sensor_api, raw_values = self._read_port_sensor_values(
                ser,
                lambda api: api.read_raw(),
                "Error during sensor warm-up",
            )
            if raw_values is None:
                return False

            self._update_data_window(
                data_obj,
                raw_values,
                n_row,
                n_col,
                window_index,
            )
        return True

    def read_goodix_raw_frame(self):
        """Read and validate one frame from the selected Goodix USB sensor."""
        client = getattr(self, "_goodix_client", None)
        if client is None:
            raise GoodixUsbError("Build the Goodix USB sensor scene first.")
        values = client.read_raw()
        expected = int(self.n_row) * int(self.n_col)
        if len(values) != expected:
            raise GoodixUsbError(
                f"Goodix USB returned {len(values)} values; expected {expected}."
            )
        return list(values)

    def measure_goodix_raw_hz(self, duration_sec=1.0):
        """Measure complete Goodix frame acquisition through the active backend."""
        client = getattr(self, "_goodix_client", None)
        if client is None:
            raise GoodixUsbError("Build the Goodix USB sensor scene first.")
        return client.measure_read_raw_hz(duration_sec=duration_sec)

    def _calibrate_goodix_usb(self):
        """Capture a local baseline because this transport has no updateCal command."""
        baseline_count = max(8, int(self._data.windowSize) * 2)
        baseline_frames = []
        for _ in range(baseline_count):
            if self._sensor_calibration_stop_event.is_set():
                return False
            try:
                baseline_frames.append(self.read_goodix_raw_frame())
            except GoodixUsbError as exc:
                self._set_sensor_stream_error(
                    f"Goodix USB calibration failed: {exc}"
                )
                return False

        baseline_values = np.mean(
            np.asarray(baseline_frames, dtype=float),
            axis=0,
        ).tolist()
        self.cal_data = list(baseline_values)
        self._data.getCal(
            self._reshape_sensor_values(
                baseline_values,
                self.n_row,
                self.n_col,
            )
        )
        self._data.clearData()

        for window_index in range(1, self._data.windowSize + 1):
            if self._sensor_calibration_stop_event.is_set():
                return False
            try:
                raw_values = self.read_goodix_raw_frame()
            except GoodixUsbError as exc:
                self._set_sensor_stream_error(
                    f"Goodix USB warm-up failed: {exc}"
                )
                return False
            self._update_data_window(
                self._data,
                raw_values,
                self.n_row,
                self.n_col,
                window_index,
            )
        return True

    def updateCal(self):
        """Start a sensor recalibration without blocking the GUI.

        The serial round-trips (stop reader, update_cal per port, warm the
        sliding window) can take seconds, so they run on a background thread;
        `_finish_update_cal` restarts the reader on the GUI thread when done.
        Returns immediately; repeated calls while running are ignored.
        """
        if self._is_shutting_down:
            return
        if self._sensor_calibration_in_progress:
            print("Sensor calibration is already running.")
            return

        self._set_sensor_stream_error("")
        self._sensor_calibration_in_progress = True
        self._sensor_calibration_stop_event.clear()
        self._set_sensor_update_button_enabled(False)
        self.is_connected = False
        self._sensor_calibration_thread = threading.Thread(
            target=self._run_calibration_sequence,
            name="sensor-calibration",
            daemon=True,
        )
        self._sensor_calibration_thread.start()

    def _run_calibration_sequence(self):
        calibration_succeeded = False
        try:
            if self._sensor_calibration_stop_event.is_set():
                return
            if not self._stop_sensor_reader_worker():
                message = "Sensor calibration aborted: live reader did not stop in time."
                self._set_sensor_stream_error(message)
                print(message)
                return

            if self.is_goodix_usb_transport():
                calibration_succeeded = self._calibrate_goodix_usb()
                if calibration_succeeded:
                    self._calibrated_sensor_ports = {
                        self._sensor_port_key(self._primary_sensor_port)
                    }
                return

            primary_port = self._sensor_port_key(self._primary_sensor_port)
            successful_ports = set()
            for ser in self.ser_list:
                if self._sensor_calibration_stop_event.is_set():
                    return
                port_name = self._sensor_port_key(ser.port)
                data_obj = self._sensor_data_for_port(port_name)
                if data_obj is None:
                    continue
                sensor_api, cal_data_list = self._read_port_sensor_values(
                    ser,
                    lambda api: api.update_cal(),
                    "Error during calibration",
                )
                if cal_data_list is None:
                    continue

                if port_name == primary_port:
                    self.cal_data = cal_data_list
                port_rows = int(
                    getattr(data_obj, "n_row", self.n_row)
                )
                port_columns = int(
                    getattr(data_obj, "n_col", self.n_col)
                )
                data_obj.getCal(
                    self._reshape_sensor_values(
                        cal_data_list,
                        port_rows,
                        port_columns,
                    )
                )
                if self._sensor_calibration_stop_event.is_set():
                    return
                if self._warm_sensor_window(
                    sensor_api, ser, data_obj=data_obj
                ):
                    successful_ports.add(port_name)
            self._calibrated_sensor_ports = successful_ports
            calibration_succeeded = primary_port in successful_ports
        except Exception as exc:
            self._set_sensor_stream_error(f"Sensor calibration failed: {exc}")
            print(f"Sensor calibration failed: {exc}")
            calibration_succeeded = False
        finally:
            shutting_down = self._is_shutting_down or self._sensor_calibration_stop_event.is_set()
            if (
                not shutting_down
                and not calibration_succeeded
                and not self.get_last_sensor_stream_error()
            ):
                self._set_sensor_stream_error(
                    "Sensor calibration failed because no valid frame was received."
                )
            if not shutting_down:
                self._calibration_bridge.finished.emit(calibration_succeeded)

    def _finish_update_cal(self, calibration_succeeded):
        self._sensor_calibration_thread = None
        if self._is_shutting_down:
            return
        self.is_connected = bool(calibration_succeeded)
        if calibration_succeeded:
            self._set_sensor_stream_error("")
            try:
                self._start_sensor_reader_worker()
            except Exception as exc:
                self.is_connected = False
                self._set_sensor_stream_error(
                    f"Failed to start the live sensor reader: {exc}"
                )
                print(f"Failed to restart sensor reader after calibration: {exc}")
        elif not self.get_last_sensor_stream_error():
            self._set_sensor_stream_error(
                "Sensor calibration failed because no valid frame was received."
            )
        self._sensor_calibration_in_progress = False
        self._set_sensor_update_button_enabled(True)

    def update_animation(self):
        if not self.is_connected:
            return

        for port_name, data_list in self._take_latest_sensor_payloads().items():
            port_name = self._sensor_port_key(port_name)
            calibrated_ports = getattr(self, "_calibrated_sensor_ports", set())
            if calibrated_ports and port_name not in calibrated_ports:
                continue
            data_obj = self._sensor_data_for_port(port_name)
            if data_obj is None:
                continue
            port_rows = int(getattr(data_obj, "n_row", self.n_row))
            port_columns = int(
                getattr(data_obj, "n_col", self.n_col)
            )
            raw_data_list = self._extract_sensor_values(
                data_list,
                port_rows,
                port_columns,
                port_name,
            )
            if raw_data_list is None:
                continue

            try:
                self._apply_live_raw_overrides(
                    raw_data_list, data_obj=data_obj
                )
                self._update_data_window(
                    data_obj,
                    raw_data_list,
                    port_rows,
                    port_columns,
                    data_obj.windowSize,
                )
                self._record_sensor_update_tick()
            except Exception as exc:
                print(f"Error processing data from port {port_name}: {exc}")
                continue

            if bool(getattr(self, "main_visualization_enabled", True)):
                matrix = np.array(
                    data_obj.diffPerDataAve,
                    dtype=float,
                    copy=True,
                )
                if self.is_primary_sensor_port(port_name):
                    self._pending_sensor_visualization_matrix = matrix
                elif port_name in self._multi_port_sensor_views:
                    self._pending_multi_port_visualization_matrices[
                        port_name
                    ] = matrix

    @staticmethod
    def _point_grid_signal_response(
        sensor_value,
        *,
        use_absolute_signal,
        noise_floor_pct,
        full_scale_pct,
    ):
        """Return a bounded point-grid response in the range [-1, 1]."""
        try:
            value = float(sensor_value)
        except Exception:
            return 0.0
        if not np.isfinite(value):
            return 0.0

        noise_floor = max(0.0, float(noise_floor_pct))
        full_scale = max(noise_floor + 1e-6, float(full_scale_pct))
        magnitude = max(0.0, abs(value) - noise_floor)
        response = float(
            np.clip(magnitude / (full_scale - noise_floor), 0.0, 1.0)
        )
        if bool(use_absolute_signal) or value >= 0.0:
            return response
        return -response

    @classmethod
    def _point_grid_visual_state(
        cls,
        sensor_value,
        *,
        use_absolute_signal,
        response_mode,
        sensitivity_scale,
        noise_floor_pct,
        full_scale_pct,
    ):
        """Return displacement and RGBA colour for one point-grid taxel."""
        try:
            value = float(sensor_value)
        except Exception:
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        sensitivity = max(0.0, float(sensitivity_scale))

        if str(response_mode) == POINT_GRID_RESPONSE_LEGACY_OFFSET:
            display_value = abs(value) if use_absolute_signal else value
            displacement = (3.0 - display_value) * sensitivity
            intensity = float(np.clip(1.0 - abs(value) * 150.0 / 255.0, 0.0, 1.0))
            if bool(use_absolute_signal) or value >= 0.0:
                color = [1.0, intensity, intensity, 1.0]
            else:
                color = [intensity, intensity, 1.0, 1.0]
            return displacement, color

        response = cls._point_grid_signal_response(
            value,
            use_absolute_signal=use_absolute_signal,
            noise_floor_pct=noise_floor_pct,
            full_scale_pct=full_scale_pct,
        )
        displacement = response * sensitivity
        intensity = 1.0 - abs(response)
        if response >= 0.0:
            color = [1.0, intensity, intensity, 1.0]
        else:
            color = [intensity, intensity, 1.0, 1.0]
        return displacement, color

    def _sensor_matrix_for_plotter(self, sensor_matrix):
        """Use the Signal Viewer row and column layout in the 3D plotter."""
        try:
            values = np.asarray(sensor_matrix, dtype=float)
        except (TypeError, ValueError):
            return None
        expected_shape = (int(self.n_row), int(self.n_col))
        if values.shape != expected_shape:
            return None
        return np.array(values, dtype=float, copy=True)

    def update_visualization(self, sensor_matrix, render=True):
        self._record_visualization_tick()
        latest_heatmap_matrix = self._current_heatmap_sensor_matrix(sensor_matrix)
        self._last_sensor_visualization_matrix = np.array(
            latest_heatmap_matrix, dtype=float, copy=True
        )
        plotter_sensor_matrix = self._sensor_matrix_for_plotter(sensor_matrix)
        if plotter_sensor_matrix is None:
            return
        for col in range(self.n_col):
            for row in range(self.n_row):
                idx = _column_major_idx(self.n_row, col, row)
                sensor_value = plotter_sensor_matrix[row][col]
                use_absolute = bool(
                    getattr(self, "visualization_use_absolute_signal", True)
                )
                displacement, color = self._point_grid_visual_state(
                    sensor_value,
                    use_absolute_signal=use_absolute,
                    response_mode=getattr(
                        self,
                        "point_grid_response_mode",
                        POINT_GRID_RESPONSE_ZERO_CENTERED,
                    ),
                    sensitivity_scale=self.touch_sensitivity_scale,
                    noise_floor_pct=getattr(
                        self,
                        "heatmap_noise_floor_pct",
                        DEFAULT_HEATMAP_NOISE_FLOOR_PCT,
                    ),
                    full_scale_pct=getattr(
                        self,
                        "heatmap_saturation_pct",
                        DEFAULT_HEATMAP_SATURATION_PCT,
                    ),
                )

                target_position = self.points_origin[idx] + self.normals[idx] * displacement
                self.points[idx] += (target_position - self.points[idx]) * 0.3

                self.colors_3d[idx] = color
                for k in self.array_positions[idx]:
                    self.colors[k] = color

        self._apply_cell_zero_mask_to_visual_colors()
        self.line_poly.points = self._sensor_points_for_plotter(self.points)
        self.line_poly.point_data.set_scalars(self.colors_3d)
        try:
            self._2D_map.Modified()
            self.line_poly.Modified()
        except Exception:
            pass
        if self._is_matrix_visualization_mode(
            getattr(self, "sensor_visualization_mode", "point_grid")
        ):
            self._update_matrix_silhouette_visualization(plotter_sensor_matrix)
        elif self._is_heatmap_visualization_mode(
            getattr(self, "sensor_visualization_mode", "point_grid")
        ):
            self._update_heatmap_visualization(latest_heatmap_matrix)
        self._refresh_sensor_selection_actor(render=False)
        self._update_contact_force_status(sensor_matrix)
        self._update_contact_normal_visualization(plotter_sensor_matrix)
        if render:
            self.plotter.render()

    @staticmethod
    def _normalize_vector(vector, fallback=None):
        arr = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-9:
            return arr / norm
        if fallback is None:
            fallback = (0.0, 0.0, 1.0)
        fallback_arr = np.asarray(fallback, dtype=float)
        fallback_norm = float(np.linalg.norm(fallback_arr))
        if fallback_norm > 1e-9:
            return fallback_arr / fallback_norm
        return np.array([0.0, 0.0, 1.0], dtype=float)

    def _set_contact_normal_status(self, text):
        label = getattr(self.parent, "contact_normal_status_label", None)
        if label is not None:
            try:
                label.setText(str(text))
            except Exception:
                pass

    def _set_contact_force_status(self, text):
        label = getattr(self.parent, "contact_force_status_label", None)
        if label is not None:
            try:
                label.setText(str(text))
            except Exception:
                pass

    def set_contact_normal_visualization_enabled(self, enabled: bool):
        self.show_contact_normal_vector = bool(enabled)
        if not self.show_contact_normal_vector:
            self._clear_contact_normal_actor()
            self._set_contact_normal_status("Normal vector: off")
            try:
                self.plotter.render()
            except Exception:
                pass
        else:
            self._set_contact_normal_status("Normal vector: waiting for contact")

    def get_contact_normal_estimator_modes(self):
        return [
            ("motion_direction_v3", "Motion Direction (V3)"),
            ("touch_anchor_v4", "Touch Anchor Direction (V4)"),
        ]

    def set_contact_normal_estimator_mode(self, mode):
        valid_modes = {key for key, _label in self.get_contact_normal_estimator_modes()}
        mode = str(mode or "touch_anchor_v4")
        if mode not in valid_modes:
            mode = "touch_anchor_v4"
        self.contact_normal_estimator_mode = mode
        if mode == "motion_direction_v3":
            self._set_contact_normal_status("Contact vector mode: Motion Direction (V3)")
        else:
            self._set_contact_normal_status("Contact vector mode: Touch Anchor Direction (V4)")
        self._reset_contact_motion_tracker()
        self._contact_normal_smoothed_start = None
        self._contact_normal_smoothed_direction = None

    def _reset_contact_motion_tracker(self):
        self._contact_motion_previous_center = None
        self._contact_anchor_center = None
        self._contact_motion_smoothed_delta = None

    def _clear_contact_normal_actor(self):
        self.contactNormalActor = self._remove_actor_safely(
            getattr(self, "contactNormalActor", None)
        )
        self.contactNormalMesh = None
        self._contact_normal_smoothed_start = None
        self._contact_normal_smoothed_direction = None
        self._reset_contact_motion_tracker()

    def _sensor_scene_span(self):
        points = np.asarray(getattr(self, "points_origin", None), dtype=float)
        if points.size == 0:
            return 1.0
        finite = points[np.all(np.isfinite(points), axis=1)]
        if finite.size == 0:
            return 1.0
        span = np.ptp(finite, axis=0)
        return max(float(np.linalg.norm(span)), 1e-3)

    def _grid_point_index(self, col, row):
        col = int(np.clip(int(col), 0, max(0, self.n_col - 1)))
        row = int(np.clip(int(row), 0, max(0, self.n_row - 1)))
        return _column_major_idx(self.n_row, col, row)

    def _point_at_cell(self, col, row):
        return np.asarray(self.points_origin[self._grid_point_index(col, row)], dtype=float)

    def _local_contact_tangent_axes(self, peak_col, peak_row, surface_normal):
        normal = self._normalize_vector(surface_normal)

        left_col = max(0, int(peak_col) - 1)
        right_col = min(self.n_col - 1, int(peak_col) + 1)
        down_row = max(0, int(peak_row) - 1)
        up_row = min(self.n_row - 1, int(peak_row) + 1)

        tangent_col = self._point_at_cell(right_col, peak_row) - self._point_at_cell(left_col, peak_row)
        tangent_row = self._point_at_cell(peak_col, up_row) - self._point_at_cell(peak_col, down_row)

        tangent_col = tangent_col - normal * float(np.dot(tangent_col, normal))
        tangent_col = self._normalize_vector(tangent_col, fallback=np.cross(normal, [0.0, 0.0, 1.0]))
        if float(np.linalg.norm(tangent_col)) <= 1e-9:
            tangent_col = self._normalize_vector(np.cross(normal, [1.0, 0.0, 0.0]))

        tangent_row = tangent_row - normal * float(np.dot(tangent_row, normal))
        tangent_row = tangent_row - tangent_col * float(np.dot(tangent_row, tangent_col))
        tangent_row = self._normalize_vector(tangent_row, fallback=np.cross(normal, tangent_col))
        return tangent_col, tangent_row

    def _contact_cluster_mask(self, pressure, peak_row, peak_col, threshold):
        touched = np.asarray(pressure >= float(threshold), dtype=bool)
        if touched.size == 0 or not bool(touched[peak_row, peak_col]):
            return np.zeros_like(touched, dtype=bool)

        cluster = np.zeros_like(touched, dtype=bool)
        stack = [(int(peak_row), int(peak_col))]
        cluster[peak_row, peak_col] = True
        while stack:
            row, col = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = row + dr
                    nc = col + dc
                    if not (0 <= nr < self.n_row and 0 <= nc < self.n_col):
                        continue
                    if cluster[nr, nc] or not touched[nr, nc]:
                        continue
                    cluster[nr, nc] = True
                    stack.append((nr, nc))
        return cluster

    def _estimate_contact_force_signal(self, sensor_matrix, peak_threshold=None):
        values = np.asarray(sensor_matrix, dtype=float)
        if values.shape != (self.n_row, self.n_col):
            return None

        pressure = np.abs(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
        if peak_threshold is None:
            peak_threshold = getattr(self, "contact_normal_threshold_pct", 3.0)
        peak_threshold = max(0.0, float(peak_threshold))
        peak_flat = int(np.argmax(pressure))
        peak_row, peak_col = np.unravel_index(peak_flat, pressure.shape)
        peak_pressure = float(pressure[peak_row, peak_col])
        if peak_pressure < peak_threshold:
            return None

        cluster_floor = max(0.0, float(getattr(self, "contact_normal_cluster_floor_pct", 0.8)))
        cluster_threshold = min(
            peak_threshold,
            max(cluster_floor, peak_pressure * 0.08),
        )
        cluster = self._contact_cluster_mask(pressure, peak_row, peak_col, cluster_threshold)
        if not bool(cluster.any()):
            return None

        weights = np.where(cluster, np.maximum(pressure - cluster_threshold, 0.0), 0.0)
        signal_sum = float(np.sum(weights))
        if signal_sum <= 1e-9:
            weights = np.where(cluster, pressure, 0.0)
            signal_sum = float(np.sum(weights))
        if signal_sum <= 1e-9:
            return None

        rows, cols = np.indices((self.n_row, self.n_col))
        center_row = float(np.sum(rows * weights) / signal_sum)
        center_col = float(np.sum(cols * weights) / signal_sum)
        scale = max(0.0, float(getattr(self, "contact_force_scale_n_per_signal", 0.0)))
        force_n = signal_sum * scale if scale > 0.0 else None
        return {
            "signal_sum": signal_sum,
            "force_n": force_n,
            "scale": scale,
            "active_nodes": int(np.count_nonzero(cluster)),
            "center_row": center_row,
            "center_col": center_col,
            "peak_pressure": peak_pressure,
        }

    def _update_contact_force_status(self, sensor_matrix):
        estimate = self._estimate_contact_force_signal(sensor_matrix)
        if estimate is None:
            self._set_contact_force_status("Contact force: no contact")
            return

        force_n = estimate.get("force_n")
        if force_n is None:
            self._set_contact_force_status(
                "Contact force: uncalibrated | "
                f"signal {estimate['signal_sum']:.2f}, "
                f"nodes {estimate['active_nodes']}, "
                f"center r{estimate['center_row']:.2f} c{estimate['center_col']:.2f}"
            )
        else:
            self._set_contact_force_status(
                f"Contact force: {force_n:.3f} N | "
                f"signal {estimate['signal_sum']:.2f}, "
                f"scale {estimate['scale']:.6f} N/signal"
            )

    @staticmethod
    def _bilinear_sample_matrix(matrix, row, col):
        data = np.asarray(matrix, dtype=float)
        n_row, n_col = data.shape
        if row < 0.0 or col < 0.0 or row > (n_row - 1) or col > (n_col - 1):
            return None

        r0 = int(np.floor(row))
        c0 = int(np.floor(col))
        r1 = min(r0 + 1, n_row - 1)
        c1 = min(c0 + 1, n_col - 1)
        fr = float(row - r0)
        fc = float(col - c0)

        return float(
            data[r0, c0] * (1.0 - fr) * (1.0 - fc)
            + data[r1, c0] * fr * (1.0 - fc)
            + data[r0, c1] * (1.0 - fr) * fc
            + data[r1, c1] * fr * fc
        )

    def _pressure_residual_tilt_components(self, weights, center_row, center_col):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.n_row, self.n_col):
            return 0.0, 0.0, 0.0

        rows, cols = np.indices((self.n_row, self.n_col))
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-9:
            return 0.0, 0.0, 0.0

        d_rows = rows - float(center_row)
        d_cols = cols - float(center_col)
        spread = float(np.sqrt(np.sum(weights * (d_rows ** 2 + d_cols ** 2)) / weight_sum))
        spread = max(spread, 0.5)

        residual_row = 0.0
        residual_col = 0.0
        residual_abs = 0.0
        sample_count = 0

        active_indices = np.argwhere(weights > 0.0)
        for row, col in active_indices:
            mirror_row = 2.0 * float(center_row) - float(row)
            mirror_col = 2.0 * float(center_col) - float(col)
            mirror_value = self._bilinear_sample_matrix(weights, mirror_row, mirror_col)
            if mirror_value is None:
                continue

            residual = float(weights[row, col] - mirror_value)
            residual_row += residual * float(row - center_row)
            residual_col += residual * float(col - center_col)
            residual_abs += abs(residual)
            sample_count += 1

        if sample_count <= 0:
            return 0.0, 0.0, 0.0

        norm = max(weight_sum * spread, 1e-9)
        tilt_row = residual_row / norm
        tilt_col = residual_col / norm
        residual_strength = residual_abs / max(weight_sum, 1e-9)

        deadband = max(0.0, float(getattr(self, "contact_normal_residual_deadband", 0.06)))
        magnitude = float(np.hypot(tilt_row, tilt_col))
        if magnitude <= deadband:
            return 0.0, 0.0, residual_strength

        scale = (magnitude - deadband) / magnitude
        gain = float(getattr(self, "contact_normal_residual_gain", 1.2))
        return tilt_row * scale * gain, tilt_col * scale * gain, residual_strength

    def _contact_motion_delta(self, center_row, center_col):
        current_center = np.array([float(center_row), float(center_col)], dtype=float)
        previous_center = getattr(self, "_contact_motion_previous_center", None)
        self._contact_motion_previous_center = current_center

        if previous_center is None:
            raw_delta = np.zeros(2, dtype=float)
        else:
            previous_center = np.asarray(previous_center, dtype=float)
            if previous_center.shape != (2,) or not np.all(np.isfinite(previous_center)):
                raw_delta = np.zeros(2, dtype=float)
            else:
                raw_delta = current_center - previous_center

        alpha = float(getattr(self, "contact_motion_smoothing_alpha", 0.55))
        alpha = max(0.0, min(1.0, alpha))
        previous_delta = getattr(self, "_contact_motion_smoothed_delta", None)
        if previous_delta is None:
            smoothed_delta = raw_delta
        else:
            previous_delta = np.asarray(previous_delta, dtype=float)
            if previous_delta.shape != (2,) or not np.all(np.isfinite(previous_delta)):
                smoothed_delta = raw_delta
            else:
                smoothed_delta = alpha * raw_delta + (1.0 - alpha) * previous_delta

        self._contact_motion_smoothed_delta = smoothed_delta
        return raw_delta, smoothed_delta

    def _contact_anchor_delta(self, center_row, center_col):
        current_center = np.array([float(center_row), float(center_col)], dtype=float)
        anchor_center = getattr(self, "_contact_anchor_center", None)
        if anchor_center is None:
            self._contact_anchor_center = current_center
            raw_delta = np.zeros(2, dtype=float)
        else:
            anchor_center = np.asarray(anchor_center, dtype=float)
            if anchor_center.shape != (2,) or not np.all(np.isfinite(anchor_center)):
                self._contact_anchor_center = current_center
                raw_delta = np.zeros(2, dtype=float)
            else:
                raw_delta = current_center - anchor_center

        alpha = float(getattr(self, "contact_motion_smoothing_alpha", 0.55))
        alpha = max(0.0, min(1.0, alpha))
        previous_delta = getattr(self, "_contact_motion_smoothed_delta", None)
        if previous_delta is None:
            smoothed_delta = raw_delta
        else:
            previous_delta = np.asarray(previous_delta, dtype=float)
            if previous_delta.shape != (2,) or not np.all(np.isfinite(previous_delta)):
                smoothed_delta = raw_delta
            else:
                smoothed_delta = alpha * raw_delta + (1.0 - alpha) * previous_delta

        self._contact_motion_smoothed_delta = smoothed_delta
        return raw_delta, smoothed_delta

    @staticmethod
    def _angle_between_vectors_deg(vector_a, vector_b):
        a = np.asarray(vector_a, dtype=float)
        b = np.asarray(vector_b, dtype=float)
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        if a_norm <= 1e-9 or b_norm <= 1e-9:
            return 0.0
        return float(
            np.degrees(
                np.arccos(
                    np.clip(float(np.dot(a / a_norm, b / b_norm)), -1.0, 1.0)
                )
            )
        )

    def _smooth_contact_normal_pose(self, start, direction, preserve_direction_sign=False):
        alpha = float(getattr(self, "contact_normal_smoothing_alpha", 0.65))
        alpha = max(0.0, min(1.0, alpha))
        start = np.asarray(start, dtype=float)
        direction = self._normalize_vector(direction)

        previous_start = getattr(self, "_contact_normal_smoothed_start", None)
        previous_direction = getattr(self, "_contact_normal_smoothed_direction", None)
        if previous_start is None or previous_direction is None:
            self._contact_normal_smoothed_start = start
            self._contact_normal_smoothed_direction = direction
            return start, direction

        previous_start = np.asarray(previous_start, dtype=float)
        previous_direction = self._normalize_vector(previous_direction, fallback=direction)
        if not preserve_direction_sign and float(np.dot(previous_direction, direction)) < 0.0:
            previous_direction = -previous_direction

        smoothed_start = alpha * start + (1.0 - alpha) * previous_start
        smoothed_direction = self._normalize_vector(
            alpha * direction + (1.0 - alpha) * previous_direction,
            fallback=direction,
        )

        self._contact_normal_smoothed_start = smoothed_start
        self._contact_normal_smoothed_direction = smoothed_direction
        return smoothed_start, smoothed_direction

    @staticmethod
    def _make_contact_normal_arrow_mesh(start, direction, arrow_length):
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            direction = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            direction = direction / norm
        length = max(float(arrow_length), 1e-6)
        return pv.Arrow(
            start=np.asarray(start, dtype=float),
            direction=direction,
            scale=length,
            tip_length=0.28,
            tip_radius=0.065,
            shaft_radius=0.022,
            tip_resolution=24,
            shaft_resolution=24,
        )

    def _estimate_contact_normal_vector(self, sensor_matrix):
        if self.points_origin is None or self.normals is None:
            return None

        values = np.asarray(sensor_matrix, dtype=float)
        if values.shape != (self.n_row, self.n_col):
            return None

        pressure = np.abs(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
        peak_threshold = max(0.0, float(getattr(self, "contact_normal_threshold_pct", 3.0)))
        peak_flat = int(np.argmax(pressure))
        peak_row, peak_col = np.unravel_index(peak_flat, pressure.shape)
        peak_pressure = float(pressure[peak_row, peak_col])
        if peak_pressure < peak_threshold:
            return None

        cluster_floor = max(0.0, float(getattr(self, "contact_normal_cluster_floor_pct", 0.8)))
        cluster_threshold = min(
            peak_threshold,
            max(cluster_floor, peak_pressure * 0.08),
        )

        cluster = self._contact_cluster_mask(pressure, peak_row, peak_col, cluster_threshold)
        if not bool(cluster.any()):
            return None

        weights = np.where(cluster, np.maximum(pressure - cluster_threshold, 0.0), 0.0)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-9:
            weights = np.where(cluster, pressure, 0.0)
            weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-9:
            return None

        rows, cols = np.indices((self.n_row, self.n_col))
        center_row = float(np.sum(rows * weights) / weight_sum)
        center_col = float(np.sum(cols * weights) / weight_sum)

        weighted_point = np.zeros(3, dtype=float)
        weighted_surface_normal = np.zeros(3, dtype=float)
        for col in range(self.n_col):
            for row in range(self.n_row):
                weight = float(weights[row, col])
                if weight <= 0.0:
                    continue
                idx = _column_major_idx(self.n_row, col, row)
                weighted_point += np.asarray(self.points_origin[idx], dtype=float) * weight
                weighted_surface_normal += np.asarray(self.normals[idx], dtype=float) * weight

        contact_point = weighted_point / weight_sum
        peak_idx = _column_major_idx(self.n_row, int(peak_col), int(peak_row))
        surface_normal = self._normalize_vector(
            weighted_surface_normal / weight_sum,
            fallback=self.normals[peak_idx],
        )

        delta_col = float(center_col - peak_col)
        delta_row = float(center_row - peak_row)
        mode = str(getattr(self, "contact_normal_estimator_mode", "touch_anchor_v4"))
        residual_strength = 0.0
        if mode in ("motion_direction_v3", "touch_anchor_v4"):
            anchor_col = int(np.clip(round(center_col), 0, max(0, self.n_col - 1)))
            anchor_row = int(np.clip(round(center_row), 0, max(0, self.n_row - 1)))
            tangent_col, tangent_row = self._local_contact_tangent_axes(
                anchor_col,
                anchor_row,
                surface_normal,
            )
            if mode == "touch_anchor_v4":
                raw_delta, smoothed_delta = self._contact_anchor_delta(center_row, center_col)
            else:
                raw_delta, smoothed_delta = self._contact_motion_delta(center_row, center_col)
            motion_cells = float(np.linalg.norm(smoothed_delta))
            deadband_cells = max(0.0, float(getattr(self, "contact_motion_deadband_cells", 0.025)))
            motion_vector = tangent_row * float(smoothed_delta[0]) + tangent_col * float(smoothed_delta[1])
            motion_vector_norm = float(np.linalg.norm(motion_vector))
            motion_over_deadband = max(0.0, motion_cells - deadband_cells)
            has_motion = motion_over_deadband > 0.0 and motion_vector_norm > 1e-9
            if has_motion:
                motion_direction = self._normalize_vector(motion_vector, fallback=tangent_col)
                max_tilt_deg = max(
                    0.0,
                    min(90.0, float(getattr(self, "contact_motion_max_tilt_deg", 90.0))),
                )
                motion_strength = 1.0 - float(
                    np.exp(
                        -motion_over_deadband
                        * max(0.0, float(getattr(self, "contact_motion_tilt_gain", 6.0)))
                    )
                )
                tilt_rad = np.radians(max_tilt_deg * np.clip(motion_strength, 0.0, 1.0))
                display_direction = self._normalize_vector(
                    (-surface_normal * float(np.cos(tilt_rad)))
                    + (motion_direction * float(np.sin(tilt_rad))),
                    fallback=-surface_normal,
                )
                vector_type = "motion tilt"
            else:
                display_direction = self._normalize_vector(-surface_normal, fallback=(0.0, 0.0, -1.0))
                vector_type = "perpendicular"

            normal = self._normalize_vector(-display_direction, fallback=surface_normal)
            tilt_deg = self._angle_between_vectors_deg(-surface_normal, display_direction)
            return {
                "point": contact_point,
                "surface_normal": surface_normal,
                "normal": normal,
                "display_direction": display_direction,
                "center_row": center_row,
                "center_col": center_col,
                "peak_row": int(peak_row),
                "peak_col": int(peak_col),
                "peak_pressure": peak_pressure,
                "tilt_deg": tilt_deg,
                "mode": mode,
                "mode_label": (
                    "Touch Anchor Direction (V4)"
                    if mode == "touch_anchor_v4"
                    else "Motion Direction (V3)"
                ),
                "residual_strength": residual_strength,
                "motion_cells": motion_cells,
                "anchor_row": (
                    float(self._contact_anchor_center[0])
                    if mode == "touch_anchor_v4"
                    and getattr(self, "_contact_anchor_center", None) is not None
                    else None
                ),
                "anchor_col": (
                    float(self._contact_anchor_center[1])
                    if mode == "touch_anchor_v4"
                    and getattr(self, "_contact_anchor_center", None) is not None
                    else None
                ),
                "raw_motion_row": float(raw_delta[0]),
                "raw_motion_col": float(raw_delta[1]),
                "smoothed_motion_row": float(smoothed_delta[0]),
                "smoothed_motion_col": float(smoothed_delta[1]),
                "vector_type": vector_type,
            }

        if mode == "peak_offset_v1":
            tangent_col, tangent_row = self._local_contact_tangent_axes(
                int(peak_col),
                int(peak_row),
                surface_normal,
            )
            tilt = tangent_col * delta_col + tangent_row * delta_row
            mode_label = "Peak Offset (V1)"
        else:
            anchor_col = int(np.clip(round(center_col), 0, max(0, self.n_col - 1)))
            anchor_row = int(np.clip(round(center_row), 0, max(0, self.n_row - 1)))
            tangent_col, tangent_row = self._local_contact_tangent_axes(
                anchor_col,
                anchor_row,
                surface_normal,
            )
            tilt_row, tilt_col, residual_strength = self._pressure_residual_tilt_components(
                weights,
                center_row,
                center_col,
            )
            tilt = tangent_col * tilt_col + tangent_row * tilt_row
            mode_label = "Balanced Residual (V2)"

        normal = self._normalize_vector(
            surface_normal + float(self.contact_normal_tilt_gain) * tilt,
            fallback=surface_normal,
        )
        tilt_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(float(np.dot(surface_normal, normal)), -1.0, 1.0)
                )
            )
        )
        return {
            "point": contact_point,
            "surface_normal": surface_normal,
            "normal": normal,
            "center_row": center_row,
            "center_col": center_col,
            "peak_row": int(peak_row),
            "peak_col": int(peak_col),
            "peak_pressure": peak_pressure,
            "tilt_deg": tilt_deg,
            "mode": mode,
            "mode_label": mode_label,
            "residual_strength": residual_strength,
        }

    def _update_contact_normal_visualization(self, sensor_matrix):
        if not bool(getattr(self, "show_contact_normal_vector", False)):
            return

        estimate = self._estimate_contact_normal_vector(sensor_matrix)
        if estimate is None:
            if str(getattr(self, "contact_normal_estimator_mode", "")) in (
                "motion_direction_v3",
                "touch_anchor_v4",
            ):
                self._reset_contact_motion_tracker()
            self._contact_normal_missing_frames += 1
            grace = max(0, int(getattr(self, "contact_normal_missing_grace_frames", 4)))
            if self.contactNormalActor is not None and self._contact_normal_missing_frames <= grace:
                self._set_contact_normal_status("Normal vector: holding last estimate")
                return
            self._clear_contact_normal_actor()
            self._set_contact_normal_status("Normal vector: no contact")
            return
        self._contact_normal_missing_frames = 0

        span = self._sensor_scene_span()
        arrow_length = max(span * 0.22, 0.03)
        surface_offset = max(span * 0.015, 0.003)
        display_surface_normal = -estimate["surface_normal"]
        direction = estimate.get("display_direction")
        if direction is None:
            direction = -estimate["normal"]
        direction = self._normalize_vector(direction, fallback=-estimate["surface_normal"])
        start = estimate["point"] + display_surface_normal * surface_offset
        preserve_direction_sign = estimate.get("mode") in (
            "motion_direction_v3",
            "touch_anchor_v4",
        )
        start, direction = self._smooth_contact_normal_pose(
            start,
            direction,
            preserve_direction_sign=preserve_direction_sign,
        )

        try:
            new_arrow = self._make_contact_normal_arrow_mesh(start, direction, arrow_length)
            if self.contactNormalMesh is None or self.contactNormalActor is None:
                self.contactNormalMesh = new_arrow
                self.contactNormalActor = self.plotter.add_mesh(
                    self.contactNormalMesh,
                    color="#ffd34d",
                    smooth_shading=True,
                    reset_camera=False,
                    name="contact_normal_vector",
                    render=False,
                )
            else:
                self.contactNormalMesh.copy_from(new_arrow)
                self.contactNormalMesh.Modified()
        except Exception as exc:
            self._clear_contact_normal_actor()
            self._set_contact_normal_status(f"Normal vector error: {exc}")
            return

        if estimate.get("mode") in ("motion_direction_v3", "touch_anchor_v4"):
            if estimate.get("mode") == "touch_anchor_v4":
                anchor_text = (
                    f"anchor r{estimate.get('anchor_row', 0.0):.2f} "
                    f"c{estimate.get('anchor_col', 0.0):.2f}, "
                )
                distance_label = "anchor displacement"
            else:
                anchor_text = ""
                distance_label = "frame motion"
            self._set_contact_normal_status(
                "Contact vector: "
                f"{estimate.get('mode_label', 'Motion Direction (V3)')}, "
                f"{estimate.get('vector_type', 'motion')}, "
                f"{anchor_text}"
                f"row {estimate['center_row']:.2f}, col {estimate['center_col']:.2f}, "
                f"{distance_label} {estimate.get('motion_cells', 0.0):.3f} cells, "
                f"tilt {estimate.get('tilt_deg', 0.0):.1f} deg, "
                f"v=({direction[0]:+.2f}, {direction[1]:+.2f}, {direction[2]:+.2f})"
            )
        else:
            self._set_contact_normal_status(
                "Normal vector: "
                f"{estimate.get('mode_label', 'Balanced Residual (V2)')}, "
                f"row {estimate['center_row']:.2f}, col {estimate['center_col']:.2f}, "
                f"tilt {estimate['tilt_deg']:.1f} deg, "
                f"residual {estimate.get('residual_strength', 0.0):.2f}, "
                f"n=({direction[0]:+.2f}, {direction[1]:+.2f}, {direction[2]:+.2f})"
            )

    def set_touch_sensitivity(self, new_value: float):
        """
        Public method to safely update the visualization's touch sensitivity.
        This will be called by the UI slider.
        """
        self.touch_sensitivity_scale = new_value

    def loadMesh(self, file_paths):
        for file_path in file_paths:
            mesh = pv.read(file_path)
            self.plotter.add_mesh(mesh)

    def changeVisibility(self, item):
        if item.level:
            parent = item.parent
            if item._type == 0:
                print("vertices would be changed")
            elif item._type == 1:
                print("edges would be changed")
            elif item._type == 2:
                print("faces would be changed")
            elif item._type == 3:
                print("N_vertics would be changed")
            elif item._type == 4:
                print("N_edges would be changed")
            elif item._type == 5:
                print("N_faces would be changed")
        else:
            print("everything for mesh changes")

    def _format_data_for_display(self, data_array, fmt_spec=None):
        if fmt_spec:
            # Format each number using the specified format string
            formatted_parts = [f"{n:{fmt_spec}}" for n in data_array]
        else:
            # Original logic: convert whole numbers to int, otherwise keep as is
            formatted_parts = [str(int(n)) if n % 1 == 0 else str(n) for n in data_array]

        # Join the formatted parts into a single string AND wrap it in brackets
        return f"[{', '.join(formatted_parts)}]"

    def _format_matrix_for_display(self, matrix, fmt_spec=None):
        row_strings = [
            self._format_data_for_display(row_values, fmt_spec=fmt_spec)
            for row_values in np.asarray(matrix)
        ]
        return "[\n  " + ",\n  ".join(row_strings) + "\n]"

    def read_sensor_diff_data(self):
        """
        Reads the processed sensor difference data and formats it to 3 significant figures.
        """
        column_major_flat_diff = _flatten_column_major_view(self._data.diffPerDataAve)
        # Call the helper with a format specifier for 3 significant figures
        return self._format_data_for_display(column_major_flat_diff, fmt_spec=".3g")

    def read_sensor_diff_debug_views(self):
        sensor_matrix = self._data.diffPerDataAve
        column_major_flat_diff = _flatten_column_major_view(sensor_matrix)
        return (
            "sensor_matrix (row-major semantic, access as matrix[row][col]):\n"
            f"{self._format_matrix_for_display(sensor_matrix, fmt_spec='.3g')}\n"
            "column_major_flat_view:\n"
            f"{self._format_data_for_display(column_major_flat_diff, fmt_spec='.3g')}"
        )

    def read_sensor_raw_data(self):
        raw_data_flat = _flatten_column_major_view(self._data.rawData)
        return self._format_data_for_display(raw_data_flat)

    def read_sensor_raw_ave_data(self):
        column_major_flat_raw_ave = _flatten_column_major_view(self._data.rawDataAve)
        return self._format_data_for_display(column_major_flat_raw_ave)

    def read_raw_all_ports(self):
        restart_reader = bool(self.is_connected)
        self._stop_sensor_reader_worker()
        try:
            for ser in self.ser_list:
                _, raw_data_list = self._read_port_sensor_values(
                    ser,
                    lambda api: api.read_raw(),
                    "Error reading raw data",
                )
                if raw_data_list is not None:
                    print(f"Port {ser.port} raw data: {raw_data_list}")
        finally:
            if restart_reader:
                self._start_sensor_reader_worker()

    # Geneva demo support has been archived to `backup/func_sensor_geneva_archive.py`.


class _MultiPortSensorReplica(MySensor):
    """Render one additional serial sensor without duplicating control state."""

    _SYNCED_VISUAL_ATTRIBUTES = (
        "touch_sensitivity_scale",
        "point_grid_response_mode",
        "visualization_use_absolute_signal",
        "heatmap_saturation_pct",
        "heatmap_noise_floor_pct",
        "heatmap_response_mode",
        "heatmap_proximity_noise_floor",
        "heatmap_proximity_knee",
        "heatmap_proximity_saturation",
        "heatmap_3d_color_gain",
        "heatmap_3d_palette",
        "stereo_field_ignore_noise_enabled",
        "stereo_field_deadband_pct",
        "stereo_field_response_scale_pct",
        "stereo_field_length_scale",
        "stereo_field_smoothing_alpha",
    )

    @classmethod
    def from_owner(
        cls,
        owner,
        data_obj,
        port_name,
        offset,
        model=None,
    ):
        replica = cls.__new__(cls)
        replica.__dict__ = owner.__dict__.copy()
        source = owner if model is None else model
        replica._multi_port_owner = owner
        replica._data = data_obj
        replica.n_row = int(source.n_row)
        replica.n_col = int(source.n_col)
        replica.n_node = int(source.n_node)
        replica.sensor_visual_offset_scale = float(
            (
                getattr(owner, "sensor_visual_offset_scale", 0.0005)
                if model is None
                else getattr(model, "offset_scale", 0.0005)
            )
            or 0.0005
        )
        replica._actor_name_prefix = owner._safe_actor_suffix(port_name)
        replica.main_visualization_enabled = True
        replica._heatmap_playback_active = False
        replica._heatmap_calibration_override = None
        replica._last_sensor_visualization_matrix = None
        replica._pending_sensor_visualization_matrix = None
        replica._pending_multi_port_visualization_matrices = {}
        replica._multi_port_sensor_views = {}
        replica._multi_port_label_actors = []
        replica._selected_sensor_cell = None
        replica.show_contact_normal_vector = False
        replica.show_sensor_point_labels = False
        replica.referenceAxisActors = []
        replica.actorPlaneXY = None
        replica.contactNormalActor = None
        replica.contactNormalMesh = None
        replica.sensorPointLabelActor = None
        replica.sensorSelectionActor = None
        replica.sensorSelectionPoly = None

        offset = np.asarray(offset, dtype=float).reshape(3)
        replica.points = np.asarray(source.points, dtype=float).copy() + offset
        replica.points_origin = (
            np.asarray(source.points_origin, dtype=float).copy() + offset
        )
        replica.normals = np.asarray(source.normals, dtype=float).copy()
        replica.edges = np.asarray(source.edges, dtype=int).copy()
        replica.colors_3d = np.asarray(source.colors_3d).copy()
        replica.colors = np.asarray(source.colors).copy()
        replica.array_positions = copy.deepcopy(source.array_positions)

        replica.line_poly = pv.PolyData(replica.points)
        replica.line_poly.lines = replica.edges
        replica.line_poly.point_data.set_scalars(replica.colors_3d)
        replica._2D_map = source._2D_map.copy(deep=True)
        replica._2D_map.points = (
            np.asarray(replica._2D_map.points, dtype=float) + offset
        )
        replica._2D_map.point_data.set_scalars(replica.colors)

        if model is None:
            base_points = getattr(
                owner, "_sensor_geometry_base_points_origin", None
            )
            base_fine_points = getattr(
                owner, "_sensor_geometry_base_fine_points", None
            )
            base_normals = getattr(
                owner, "_sensor_geometry_base_normals", None
            )
        else:
            base_points = np.asarray(model.points_origin, dtype=float)
            base_fine_points = np.asarray(
                model._2D_map.points, dtype=float
            )
            base_normals = np.asarray(model.normals, dtype=float)
        replica._sensor_geometry_base_points_origin = (
            None
            if base_points is None
            else np.asarray(base_points, dtype=float).copy() + offset
        )
        replica._sensor_geometry_base_fine_points = (
            None
            if base_fine_points is None
            else np.asarray(base_fine_points, dtype=float).copy() + offset
        )
        replica._sensor_geometry_base_normals = (
            None
            if base_normals is None
            else np.asarray(base_normals, dtype=float).copy()
        )

        replica.objActor = None
        replica.actionMesh = None
        replica.matrixLineActor = None
        replica.matrixLinePoly = None
        replica.matrixLineColors = None
        replica._matrix_visual_actor_mode = None
        replica._matrix_line_dense_shape = (0, 0)
        replica._matrix_line_base_points = None
        replica._matrix_line_normals = None
        replica._matrix_line_base_indices = None
        replica._matrix_line_top_indices = None
        replica._matrix_line_field_height = 0.0
        replica._matrix_line_height_variation = None
        replica._stereo_field_smoothed_visibility = None
        replica._stereo_field_smoothed_color_response = None
        replica.heatmapActor = None
        replica.heatmapPoly = None
        replica.heatmapColors = None
        replica.heatmapGridActor = None
        replica.heatmapGridPoly = None
        replica.heatmapGridDisplayPoly = None
        replica._heatmap_logical_edges = None
        replica._heatmap_tile_vertices = None
        owner_mask = np.asarray(owner.cell_zero_mask, dtype=bool)
        if owner_mask.shape == (replica.n_row, replica.n_col):
            replica.cell_zero_mask = owner_mask.copy()
        else:
            replica.cell_zero_mask = np.zeros(
                (replica.n_row, replica.n_col),
                dtype=bool,
            )

        replica._ensure_main_sensor_visualization_actors()
        replica._refresh_sensor_visualization_mode_actors()
        return replica

    def _sync_visual_settings(self):
        owner = self._multi_port_owner
        for attr_name in self._SYNCED_VISUAL_ATTRIBUTES:
            setattr(self, attr_name, getattr(owner, attr_name))
        owner_mask = np.asarray(owner.cell_zero_mask, dtype=bool)
        if owner_mask.shape == (self.n_row, self.n_col):
            self.cell_zero_mask = owner_mask.copy()

    def set_replica_visualization_mode(self, mode):
        self._sync_visual_settings()
        valid_modes = {
            key for key, _label in self.SENSOR_VISUALIZATION_MODE_OPTIONS
        }
        self.sensor_visualization_mode = (
            mode if mode in valid_modes else "point_grid"
        )
        if self._is_matrix_visualization_mode(self.sensor_visualization_mode):
            if self.matrixLineActor is None:
                self._rebuild_matrix_visualization_actor(render=False)
        if self._is_heatmap_visualization_mode(self.sensor_visualization_mode):
            self._ensure_heatmap_visualization_actor(render=False)
        self._refresh_sensor_visualization_mode_actors()

    def set_replica_visibility(self, enabled):
        self.main_visualization_enabled = bool(enabled)
        self._refresh_sensor_visualization_mode_actors()

    def update_visualization(self, sensor_matrix, render=False):
        self._sync_visual_settings()
        super().update_visualization(sensor_matrix, render=render)

    def close(self):
        for actor_name in (
            "objActor",
            "actionMesh",
            "matrixLineActor",
            "heatmapActor",
            "heatmapGridActor",
            "contactNormalActor",
            "sensorPointLabelActor",
            "sensorSelectionActor",
        ):
            actor = getattr(self, actor_name, None)
            if actor is None:
                continue
            try:
                self.plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass
            setattr(self, actor_name, None)

    def _record_visualization_tick(self):
        return None

    def _update_contact_force_status(self, sensor_matrix):
        return None

    def _update_contact_normal_visualization(self, sensor_matrix):
        return None

    def _refresh_sensor_selection_actor(self, render=True):
        return None

    def saveCameraPara(self):
        return None

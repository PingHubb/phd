import json
import os
from importlib import import_module

import pyvista as pv
import numpy as np
import serial
import serial.tools.list_ports
import time
from pyvistaqt import QtInteractor
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QListWidgetItem
from tqdm import tqdm
from phd.dependence.paths import resource_path, sensor_resource_path
from phd.dependence.sensor_layout import (
    column_major_idx as _column_major_idx,
    flatten_column_major_view as _flatten_column_major_view,
    reshape_sensor_values_to_row_col_matrix,
    row_major_idx as _row_major_idx,
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


class _SensorReadWorker(QObject):
    """Continuously read raw sensor payloads without blocking Qt rendering."""

    raw_payload_ready = pyqtSignal(int, str, list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, serial_ports, generation=0, response_timeout=0.5, idle_sleep_sec=0.002):
        super().__init__()
        self.serial_ports = list(serial_ports or [])
        self.generation = int(generation)
        self.response_timeout = max(0.05, float(response_timeout))
        self.idle_sleep_sec = max(0.0, float(idle_sleep_sec))
        self._running = False

    def stop(self):
        self._running = False

    @staticmethod
    def _parse_ints(text):
        values = []
        for token in str(text).split():
            try:
                values.append(int(token))
            except ValueError:
                continue
        return values

    def _read_raw_from_port(self, ser):
        try:
            ser.write(b"readRaw\n")
        except Exception as exc:
            self.error.emit(f"Sensor write failed on {getattr(ser, 'port', 'unknown')}: {exc}")
            return None

        deadline = time.time() + self.response_timeout
        while self._running and time.time() < deadline:
            try:
                waiting = int(getattr(ser, "in_waiting", 0))
            except Exception as exc:
                self.error.emit(f"Sensor read failed on {getattr(ser, 'port', 'unknown')}: {exc}")
                return None

            if waiting <= 0:
                time.sleep(self.idle_sleep_sec)
                continue

            try:
                line = ser.readline().decode("utf-8", errors="ignore").rstrip()
            except Exception as exc:
                self.error.emit(f"Sensor line read failed on {getattr(ser, 'port', 'unknown')}: {exc}")
                return None

            if not line:
                continue

            data_list = self._parse_ints(line)
            if data_list:
                return data_list[2:-2] if len(data_list) >= 4 else data_list

        return None

    def run(self):
        self._running = True
        try:
            while self._running:
                emitted = False
                for ser in self.serial_ports:
                    if not self._running:
                        break
                    data_list = self._read_raw_from_port(ser)
                    if data_list is None:
                        continue
                    self.raw_payload_ready.emit(
                        self.generation,
                        str(getattr(ser, "port", "sensor")),
                        list(data_list),
                    )
                    emitted = True
                if not emitted:
                    time.sleep(self.idle_sleep_sec)
        finally:
            self.finished.emit()


class data:
    def __init__(self, n_row, n_col, window_size=3):
        self.n_row = n_row
        self.n_col = n_col
        self.windowSize = max(1, int(window_size))
        self.calData = np.zeros((self.n_row, self.n_col))
        self.rawData = np.zeros((self.n_row, self.n_col))
        self.diffData = np.zeros((self.n_row, self.n_col))
        self.diffPerData = np.zeros((self.n_row, self.n_col))
        self.rawDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffPerDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.rawDataAve = np.zeros((self.n_row, self.n_col))
        self.diffDataAve = np.zeros((self.n_row, self.n_col))
        self.diffPerDataAve = np.zeros((self.n_row, self.n_col))
        self.frame_sequence = 0

    def _recompute_averages(self):
        self.rawDataAve = np.flipud(np.mean(self.rawDataWin, axis=0))
        self.diffDataAve = np.flipud(np.mean(self.diffDataWin, axis=0))
        self.diffPerDataAve = np.flipud(np.mean(self.diffPerDataWin, axis=0))

    def setWindowSize(self, window_size):
        self.windowSize = max(1, int(window_size))
        self.rawDataWin = np.repeat(self.rawData[None, ...], self.windowSize, axis=0)
        self.diffDataWin = np.repeat(self.diffData[None, ...], self.windowSize, axis=0)
        self.diffPerDataWin = np.repeat(self.diffPerData[None, ...], self.windowSize, axis=0)
        self._recompute_averages()

    def getRaw(self, rawData):
        self.rawData = rawData

    def getCal(self, calData):
        self.calData = calData

    def calDiff(self):
        self.diffData = self.rawData - self.calData

    def calDiffPer(self):
        non_zero_mask = self.calData != 0
        self.diffPerData = np.zeros_like(self.calData, dtype=float)
        np.divide(
            100 * self.diffData,
            self.calData,
            out=self.diffPerData,
            where=non_zero_mask
        )

    def clearData(self):
        self.rawDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffPerDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.rawDataAve = np.zeros((self.n_row, self.n_col))
        self.diffDataAve = np.zeros((self.n_row, self.n_col))
        self.diffPerDataAve = np.zeros((self.n_row, self.n_col))
        self.frame_sequence = 0

    def getWin(self, i):
        # This index logic seems to be 1-based. Using i-1 to be safe.
        # It also looks like you want to handle a circular buffer.
        # Let's adjust the index to be robust using modulo.
        idx = (i - 1) % self.windowSize

        # The original code shifts the window only when i == windowSize.
        # A more common approach is to always shift, which might be what's intended.
        # However, sticking to the original logic:
        if i == self.windowSize:
            self.rawDataWin[:-1] = self.rawDataWin[1:]
            self.diffDataWin[:-1] = self.diffDataWin[1:]
            self.diffPerDataWin[:-1] = self.diffPerDataWin[1:]
            # The last element will be overwritten below.

        # Place the new data using the normalized index.
        self.rawDataWin[idx] = self.rawData
        self.diffDataWin[idx] = self.diffData
        self.diffPerDataWin[idx] = self.diffPerData

        if i >= self.windowSize:  # Calculate average once the window is full
            self._recompute_averages()
            self.frame_sequence += 1


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
    )

    def __init__(self, parent) -> None:
        self.parent = parent
        self.plotter: QtInteractor = self.parent.plotter_2
        self.actionMesh = None  # no mesh yet
        self.objActor = None
        self.matrixLineActor = None
        self.matrixLinePoly = None
        self.matrixLineColors = None
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
        self._sensor_geometry_base_points_origin = None
        self._sensor_geometry_base_normals = None
        self._sensor_geometry_base_fine_points = None
        self.current_sensor_geometry_config = None
        self.sensor_visual_offset_scale = 0.0
        self._contact_normal_missing_frames = 0
        self.sensorPointLabelActor = None
        self.n_col = 0
        self.n_row = 0
        self.touch_sensitivity_scale = 0.05
        self.sensor_visualization_mode = "point_grid"
        self.show_contact_normal_vector = True
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
        self.sensor_average_window_size = self.SENSOR_AVERAGE_WINDOW_SIZE
        self.visualization_target_hz = self.VISUALIZATION_TARGET_HZ
        self.creatPlaneXY()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(0)
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
        self._sensor_reader_generation = 0
        self._last_sensor_reader_error_log_time = 0.0
        self._sensor_calibration_in_progress = False
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

    def creatPlaneXY(self):
        self.plotter.camera.position = (1, -1, 1)
        self.saveCameraPara()

        # X-axis line
        line_x = pv.Line((-50, 0, 0), (50, 0, 0))
        self.plotter.add_mesh(line_x, color='r', line_width=2, label='X Axis')

        # Y-axis line
        line_y = pv.Line((0, -50, 0), (0, 50, 0))
        self.plotter.add_mesh(line_y, color='g', line_width=2, label='Y Axis')

        # Z-axis line
        # line_z = pv.Line((0, 0, -50), (0, 0, 50))
        # self.plotter.add_mesh(line_z, color='b', line_width=2, label='Z Axis')

        planeXY = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=100,
            j_size=100,
            i_resolution=100,
            j_resolution=100,
        )
        self.actorPlaneXY = self.plotter.add_mesh(planeXY, color='gray', style='wireframe')

    def initChannel(self):
        self.com_options = []
        ports = serial.tools.list_ports.comports()
        self.ser = None
        self.parent.serial_channel.clear()
        if ports:
            # Sort ports to have 'ttyACM' first, then others.
            ports = sorted(ports, key=lambda port: (0, int(port.name.replace('ttyACM', ''))) if port.name.startswith(
                'ttyACM') else (1, port.name))

            for port in ports:
                self.com_options.append(port.name)
                # Create a standard, selectable item (no checkbox).
                # Selection mode is already handled in ui_ping.py.
                item = QListWidgetItem(port.name)
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
        data_obj = getattr(self, "_data", None)
        if data_obj is not None:
            data_obj.setWindowSize(self.sensor_average_window_size)

    def get_visualization_target_hz(self):
        return float(self.visualization_target_hz)

    def set_visualization_target_hz(self, hz):
        self.visualization_target_hz = max(1.0, float(hz))
        self.visualization_min_interval_sec = 1.0 / self.visualization_target_hz

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
        self._refresh_sensor_visualization_mode_actors()
        try:
            self.plotter.render()
        except Exception:
            pass

    @staticmethod
    def _is_matrix_visualization_mode(mode):
        return str(mode) == "stereo_field"

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
        matrix_mode = self._is_matrix_visualization_mode(
            getattr(self, "sensor_visualization_mode", "point_grid")
        )
        self._set_actor_visible(getattr(self, "objActor", None), not matrix_mode)
        self._set_actor_visible(getattr(self, "actionMesh", None), not matrix_mode)
        self._set_actor_visible(getattr(self, "matrixLineActor", None), matrix_mode)

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
            colors = np.empty((points.shape[0], 3), dtype=np.uint8)
            colors[base_indices] = [8, 70, 55]
            colors[top_indices] = [55, 255, 185]
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
        colors = np.empty((points.shape[0], 3), dtype=np.uint8)
        colors[:, 0] = 30
        colors[:, 1] = 235
        colors[:, 2] = 160
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
                name=f"sensor_{mode}",
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
        pressure = np.abs(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
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
                cool = np.array([25.0, 185.0, 255.0])
                warm = np.array([255.0, 96.0, 42.0])
                top_rgb = cool[None, :] * (1.0 - response[:, None]) + warm[None, :] * response[:, None]
                brightness = np.clip(0.62 + 0.38 * visibility, 0.35, 1.0)
                top_rgb = np.clip(top_rgb * brightness[:, None], 0, 255)
                base_rgb = np.clip(top_rgb * 0.32, 0, 255)
                colors[base_indices] = base_rgb.astype(np.uint8)
                colors[top_indices] = top_rgb.astype(np.uint8)
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

        return (
            f"sensor_update_hz: {self._sensor_update_hz:.2f}\n"
            f"direct_finger_motion_loop_hz: {direct_hz:.2f}\n"
            f"direct_finger_motion_running: {direct_running}\n"
            f"visualization_actual_hz: {self._visualization_hz:.2f}\n"
            f"sensor_average_window_size: {window_size}\n"
            f"visualization_target_hz: {self.visualization_target_hz:.2f}"
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

    def buildScene(self):
        self._reset_scene_build_state()
        self._set_sensor_update_button_enabled(False)
        self._clear_scene_actors()
        self._close_serial_ports()
        self.is_connected = False

        selected_ports = self._get_selected_port_names()
        if not selected_ports:
            print("No serial port selected. Please select one or more ports by highlighting them.")
            return

        self.ser_list = self._open_serial_ports(selected_ports)
        if not self.ser_list:
            print("Could not open any of the selected ports.")
            return

        self.clearParameters()
        if not self._initialize_selected_sensor_model(self.parent.sensor_choice.currentRow()):
            self._close_serial_ports()
            print("Unsupported sensor selection.")
            return
        self.initialize_ai_helpers()
        self.update_ui_elements()

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

        # Add newly created meshes to the plotter
        if self._2D_map and self._2D_map.n_points > 0:
            self.objActor = self.plotter.add_mesh(
                self._2D_map, show_edges=True, scalars=self.colors, rgb=True
            )

        self.actionMesh = self.plotter.add_mesh(
            self.line_poly, scalars=self.colors_3d, point_size=10, line_width=3,
            render_points_as_spheres=True, rgb=True
        )
        if self.actionMesh is not None or self.matrixLineActor is not None:
            self._rebuild_matrix_visualization_actor(render=False)
            self._refresh_sensor_visualization_mode_actors()
        self._refresh_sensor_point_label_actor()

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
        self.objActor = self._remove_actor_safely(self.objActor)
        self.actionMesh = self._remove_actor_safely(self.actionMesh)
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
        self._clear_contact_normal_actor()
        self._clear_sensor_point_label_actor()

    def _get_selected_port_names(self):
        return [item.text() for item in self.parent.serial_channel.selectedItems()]

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
                ser = serial.Serial(port=f"/dev/{port_name}", baudrate=9600, timeout=0.1)
                serial_ports.append(ser)
                print(f"Opened port: /dev/{port_name}")
            except Exception as exc:
                print(f"Failed to open /dev/{port_name}: {exc}")
        return serial_ports

    def _sensor_reader_is_running(self):
        thread = getattr(self, "_sensor_read_thread", None)
        return bool(thread is not None and thread.isRunning())

    def _start_sensor_reader_worker(self):
        if not self.ser_list or self._sensor_reader_is_running():
            return

        self._latest_sensor_payloads = {}
        self._sensor_reader_generation += 1
        generation = self._sensor_reader_generation
        thread_parent = self.parent if isinstance(self.parent, QObject) else None
        thread = QThread(thread_parent)
        worker = _SensorReadWorker(self.ser_list, generation=generation)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.raw_payload_ready.connect(self._on_sensor_reader_payload)
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
        self._stop_sensor_reader_worker()
        self._close_serial_ports()

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

    def _initialize_selected_sensor_model(self, sensor_index):
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

    @staticmethod
    def _default_sensor_geometry_config():
        return {
            "use_selected_shape": False,
            "shape": "flat",
            "bend_axis": "columns",
            "arc_deg": 0.0,
            "normal_flip": False,
        }

    def _normalize_sensor_geometry_config(self, config):
        default = self._default_sensor_geometry_config()
        if not isinstance(config, dict):
            return dict(default)

        shape = str(config.get("shape", default["shape"]) or default["shape"]).strip().lower()
        if shape not in ("flat", "cylinder"):
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

        return {
            "use_selected_shape": use_selected_shape,
            "shape": shape,
            "bend_axis": bend_axis,
            "arc_deg": arc_deg,
            "normal_flip": bool(config.get("normal_flip", default["normal_flip"])),
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
            "force_scale_n_per_signal": force_scale,
            "geometry": geometry,
            "stereo_field": stereo_field,
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

        self.points_origin, self.normals = self._bend_points_to_cylinder(
            base_points,
            effective_geometry,
            return_normals=True,
            base_normals=base_normals,
        )
        self.points = self.points_origin + self.normals * float(
            getattr(self, "sensor_visual_offset_scale", 0.0)
        )

        base_fine_points = getattr(self, "_sensor_geometry_base_fine_points", None)
        if self._2D_map is not None and base_fine_points is not None:
            self._2D_map.points = self._bend_points_to_cylinder(
                base_fine_points,
                effective_geometry,
                return_normals=False,
            )
            try:
                self._2D_map.Modified()
            except Exception:
                pass

        if self.line_poly is not None:
            self.line_poly.points = self.points
            try:
                self.line_poly.Modified()
            except Exception:
                pass

        if self.actionMesh is not None or self.matrixLineActor is not None:
            self._rebuild_matrix_visualization_actor(render=False)
            self._refresh_sensor_visualization_mode_actors()
        self._clear_contact_normal_actor()
        self._refresh_sensor_point_label_actor()
        if render:
            try:
                self.plotter.render()
            except Exception:
                pass
        return True

    def _sensor_point_labels(self):
        labels = []
        for col in range(int(self.n_col)):
            for row in range(int(self.n_row)):
                idx = _column_major_idx(self.n_row, col, row)
                labels.append(f"P{idx} r{row} c{col}")
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
        try:
            self.sensorPointLabelActor = self.plotter.add_point_labels(
                self.line_poly,
                self._sensor_point_labels(),
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
                name="sensor_point_labels",
                render=False,
            )
        except Exception as exc:
            self.sensorPointLabelActor = None
            print(f"[SensorPointLabels] Failed to show point labels: {exc}")

    def _bind_sensor_api_to_port(self, ser):
        """Reuse the shared sensor API object while switching its active serial port."""
        self.parent.sensor_api.ser = ser
        return self.parent.sensor_api

    def _trim_sensor_payload(self, data_list, n_row, n_col):
        """Apply model-specific payload fixes before validating the sample size."""
        if n_row == 10 and n_col == 9:
            return data_list[:-10]
        return data_list

    def _extract_sensor_values(self, data_list, n_row, n_col, port_name):
        if data_list is None:
            print(f"Error on port {port_name}: No sensor payload received.")
            return None
        data_list = self._trim_sensor_payload(data_list, n_row, n_col)
        expected_length = n_row * (n_col + 1)
        if len(data_list) != expected_length:
            print(f"Error on port {port_name}: Data length is {len(data_list)}, expected {expected_length}")
            return None
        return data_list[0:-n_row]

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

    def _apply_live_raw_overrides(self, raw_values):
        """Patch known bad channels for specific live sensor layouts."""
        if self.n_row == 10 and self.n_col == 8:
            flat_cal_data = _flatten_column_major_view(self._data.calData)
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
        return True

    def clear_cell_zero_mask(self):
        n_row = int(getattr(self, "n_row", 0) or 0)
        n_col = int(getattr(self, "n_col", 0) or 0)
        self.cell_zero_mask = np.zeros((n_row, n_col), dtype=bool)

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
        return True

    def _read_port_sensor_values(self, ser, read_operation, error_prefix):
        try:
            sensor_api = self._bind_sensor_api_to_port(ser)
            data_list = read_operation(sensor_api)
        except Exception as exc:
            print(f"{error_prefix} on port {ser.port}: {exc}")
            return None, None

        sensor_values = self._extract_sensor_values(data_list, self.n_row, self.n_col, ser.port)
        return sensor_api, sensor_values

    def _warm_sensor_window(self, sensor_api, ser):
        self._data.clearData()
        for window_index in range(1, self._data.windowSize + 1):
            try:
                data_list = sensor_api.read_raw()
            except Exception as exc:
                print(f"Error during raw data processing on port {ser.port}: {exc}")
                return False

            raw_values = self._extract_sensor_values(data_list, self.n_row, self.n_col, ser.port)
            if raw_values is None:
                return False

            self._update_data_window(self._data, raw_values, self.n_row, self.n_col, window_index)
        return True

    def updateCal(self):
        if self._sensor_calibration_in_progress:
            print("Sensor calibration is already running.")
            return

        self._sensor_calibration_in_progress = True
        self._set_sensor_update_button_enabled(False)
        self.is_connected = False
        calibration_succeeded = False
        try:
            if not self._stop_sensor_reader_worker():
                print("Sensor calibration aborted: live reader did not stop in time.")
                return

            for ser in self.ser_list:
                sensor_api, cal_data_list = self._read_port_sensor_values(
                    ser,
                    lambda api: api.update_cal(),
                    "Error during calibration",
                )
                if cal_data_list is None:
                    continue

                self.cal_data = cal_data_list
                self._data.getCal(self._reshape_sensor_values(cal_data_list, self.n_row, self.n_col))
                if self._warm_sensor_window(sensor_api, ser):
                    calibration_succeeded = True
        finally:
            self.is_connected = calibration_succeeded
            if calibration_succeeded:
                try:
                    self._start_sensor_reader_worker()
                except Exception as exc:
                    self.is_connected = False
                    print(f"Failed to restart sensor reader after calibration: {exc}")
            self._sensor_calibration_in_progress = False
            self._set_sensor_update_button_enabled(True)

    def update_animation(self):
        if not self.is_connected:
            return

        for port_name, data_list in self._take_latest_sensor_payloads().items():
            raw_data_list = self._extract_sensor_values(
                data_list,
                self.n_row,
                self.n_col,
                port_name,
            )
            if raw_data_list is None:
                continue

            try:
                self._apply_live_raw_overrides(raw_data_list)
                self._update_data_window(
                    self._data, raw_data_list, self.n_row, self.n_col, self._data.windowSize
                )
                self._record_sensor_update_tick()
            except Exception as exc:
                print(f"Error processing data from port {port_name}: {exc}")
                continue

            if self._should_refresh_visualization():
                self.saveCameraPara()
                self.update_visualization(self._data.diffPerDataAve)

    def update_visualization(self, sensor_matrix):
        self._record_visualization_tick()
        for col in range(self.n_col):
            for row in range(self.n_row):
                idx = _column_major_idx(self.n_row, col, row)
                sensor_value = sensor_matrix[row][col]
                # It uses the up-to-date value of self.touch_sensitivity_scale
                displacement = (3 - abs(sensor_value)) * self.touch_sensitivity_scale

                target_position = self.points_origin[idx] + self.normals[idx] * displacement
                self.points[idx] += (target_position - self.points[idx]) * 0.3

                intensity = np.clip(1 - abs(sensor_value) * 150 / 255, 0, 1)
                self.colors_3d[idx] = [1, intensity, intensity, 1]
                for k in self.array_positions[idx]:
                    self.colors[k] = [1, intensity, intensity, 1]

        self.line_poly.points = self.points
        self.line_poly.point_data.set_scalars(self.colors_3d)
        if self._is_matrix_visualization_mode(
            getattr(self, "sensor_visualization_mode", "point_grid")
        ):
            self._update_matrix_silhouette_visualization(sensor_matrix)
        self._update_contact_force_status(sensor_matrix)
        self._update_contact_normal_visualization(sensor_matrix)
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

    def _estimate_contact_force_signal(self, sensor_matrix):
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
        if not bool(getattr(self, "show_contact_normal_vector", True)):
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

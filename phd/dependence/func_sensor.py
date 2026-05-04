import pyvista as pv
import numpy as np
import serial
import serial.tools.list_ports
import time
from pyvistaqt import QtInteractor
from PyQt5.QtCore import QTimer
from tqdm import tqdm
from phd.dependence.gesture_logic_direct_finger_motion import (
    DirectFingerMotion,
    DirectFingerMotionV2,
    ConsoleControl,
    AI_DirectFingerMotion,
    AI_DirectFingerMotion_execution,
)
from phd.dependence.gesture_logic_proximity_control import ProximityControl
from phd.dependence.gesture_logic_recording import RecordGesture
from phd.dependence.sensor_layout import (
    column_major_idx as _column_major_idx,
    flatten_column_major_view as _flatten_column_major_view,
    reshape_sensor_values_to_row_col_matrix,
    row_major_idx as _row_major_idx,
)
from phd.dependence.gesture_logic_three_level import ThreeLevelTransformer

from PyQt5.QtWidgets import QListWidgetItem


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
        is_row_major_input = self.reorder_logic in ['row_to_col', 'row_to_col_flipped']
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


class MySensor:
    DEFAULT_2D_GRID_SHAPE = (10, 10)
    VISUALIZATION_TARGET_HZ = 30.0
    SENSOR_AVERAGE_WINDOW_SIZE = 3
    AI_HELPER_ATTRS = {
        "record_gesture_class": "RecordGesture",
        "threelevel_hierarchical_transformer_class": "ThreeLevelTransformer",
        "proximity_control_class": "ProximityControl",
        "direct_finger_motion_class": "DirectFingerMotion",
        "direct_finger_motion_v2_class": "DirectFingerMotionV2",
        "console_control_class": "ConsoleControl",
        "ai_direct_finger_motion_class": "AI_DirectFingerMotion",
        "ai_direct_finger_motion_execution_class": "AI_DirectFingerMotion_execution",
    }
    PREDEFINED_SENSOR_MODELS = {
        "elbow": {
            "n_row": 10,
            "n_col": 13,
            "mesh_file": "/home/ping2/ros2_ws/src/phd/phd/resource/sensor/joint_1/mesh.obj",
            "signal_file": "/home/ping2/ros2_ws/src/phd/phd/resource/sensor/joint_1/signal.txt",
            "offset_scale": 0.02,
        },
        "kuka": {
            "n_row": 10,
            "n_col": 8,
            "mesh_file": "/home/ping2/ros2_ws/src/phd/phd/resource/sensor/kuka/knitting_mesh_raw.obj",
            "signal_file": "/home/ping2/ros2_ws/src/phd/phd/resource/sensor/kuka/signal.txt",
            "reorder_logic": "row_to_col_flipped",
            "offset_scale": 0.2,
        },
        "double_curve": {
            "n_row": 10,
            "n_col": 10,
            "mesh_file": "/home/ping2/ros2_ws/src/phd/phd/resource/sensor/dualC/mesh.obj",
            "signal_file": "/home/ping2/ros2_ws/src/phd/phd/resource/sensor/dualC/signal.txt",
            "offset_scale": 0.2,
        },
        "half_cylinder_surface": {
            "n_row": 10,
            "n_col": 9,
            "mesh_file": "/home/ping2/ros2_ws/src/phd/phd/resource/sensor/half_cylinder_surface/half_cylinder_2.obj",
            "signal_file": "/home/ping2/ros2_ws/src/phd/phd/resource/sensor/half_cylinder_surface/vertex_groups_2.txt",
            "reorder_logic": "flip_and_rotate",
            "offset_scale": 0.2,
        },
    }

    def __init__(self, parent) -> None:
        self.parent = parent
        self.plotter: QtInteractor = self.parent.plotter_2
        self.actionMesh = None  # no mesh yet
        self.objActor = None
        self.n_col = 0
        self.n_row = 0
        self.touch_sensitivity_scale = 0.05
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
        else:
            print("No serial ports found. Please connect the device and retry.")
            return

    def reset_ai_helpers(self, reason="Build the scene first to initialize AI features."):
        for attr_name, feature_name in self.AI_HELPER_ATTRS.items():
            if attr_name == "console_control_class":
                continue
            setattr(self, attr_name, _FeatureDisabledProxy(feature_name, reason))
        self.console_control_class = self._safe_create_helper(
            "ConsoleControl",
            lambda: ConsoleControl(self.parent, self),
        )

    def _safe_create_helper(self, feature_name, factory):
        try:
            return factory()
        except Exception as exc:
            print(f"[{feature_name}] Failed to initialize: {exc}")
            return _FeatureDisabledProxy(feature_name, f"Unavailable: {exc}")

    def _ai_helper_factories(self):
        return {
            "record_gesture_class": lambda: RecordGesture(self),
            "threelevel_hierarchical_transformer_class": lambda: ThreeLevelTransformer(
                self.parent, self, self.n_row, self.n_col
            ),
            "proximity_control_class": lambda: ProximityControl(self.parent, self),
            "direct_finger_motion_class": lambda: DirectFingerMotion(self.parent, self),
            "direct_finger_motion_v2_class": lambda: DirectFingerMotionV2(self.parent, self),
            "console_control_class": lambda: ConsoleControl(self.parent, self),
            "ai_direct_finger_motion_class": lambda: AI_DirectFingerMotion(self.parent, self),
            "ai_direct_finger_motion_execution_class": lambda: AI_DirectFingerMotion_execution(self.parent, self),
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
                self._safe_create_helper(feature_name, factories[attr_name]),
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

        return (
            f"sensor_update_hz: {self._sensor_update_hz:.2f}\n"
            f"direct_finger_motion_loop_hz: {direct_hz:.2f}\n"
            f"direct_finger_motion_running: {direct_running}\n"
            f"visualization_actual_hz: {self._visualization_hz:.2f}\n"
            f"sensor_average_window_size: {window_size}\n"
            f"visualization_target_hz: {self.visualization_target_hz:.2f}"
        )

    def toggle_ai_direct_finger_motion_execution(self, model_checkpoint_path=None):
        helper = getattr(self, "ai_direct_finger_motion_execution_class", None)
        if helper is None:
            raise RuntimeError("AI_DirectFingerMotion_execution helper is not initialized.")
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

    def init_2d_model(self, n_row=None, n_col=None):
        if n_row is None or n_col is None:
            n_row, n_col = self._get_2d_grid_shape()

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

    def _get_selected_port_names(self):
        return [item.text() for item in self.parent.serial_channel.selectedItems()]

    def _close_serial_ports(self):
        for ser in self.ser_list:
            try:
                ser.close()
            except Exception:
                pass
        self.ser_list = []

    def _open_serial_ports(self, port_names):
        serial_ports = []
        for port_name in port_names:
            try:
                ser = serial.Serial(port=f"/dev/{port_name}", baudrate=9600, timeout=1)
                serial_ports.append(ser)
                print(f"Opened port: /dev/{port_name}")
            except Exception as exc:
                print(f"Failed to open /dev/{port_name}: {exc}")
        return serial_ports

    def _get_2d_grid_shape(self):
        default_n_row, default_n_col = self.DEFAULT_2D_GRID_SHAPE
        try:
            return self.parent.grid_rows_spin.value(), self.parent.grid_cols_spin.value()
        except Exception:
            return default_n_row, default_n_col

    def _build_factory_model(self, **model_kwargs):
        model_kwargs.setdefault("window_size", self.sensor_average_window_size)
        model = SensorModelFactory(**model_kwargs).build()
        self._initialize_from_factory(model)

    def _init_predefined_model(self, model_name):
        self._build_factory_model(**self.PREDEFINED_SENSOR_MODELS[model_name])

    def _initialize_selected_sensor_model(self, sensor_index):
        model_initializers = {
            0: self.init_elbow_model,
            1: self.init_kuka_model,
            2: self.init_double_curve_model,
            3: self.init_2d_model,
            4: self.init_half_cylinder_surface_model,
        }
        initializer = model_initializers.get(sensor_index)
        if initializer is None:
            return False
        initializer()
        return True

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
        data_obj.getRaw(self._reshape_sensor_values(raw_values, n_row, n_col))
        data_obj.calDiff()
        data_obj.calDiffPer()
        data_obj.getWin(window_index)

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
        calibration_succeeded = False
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

        self.is_connected = calibration_succeeded
        self._set_sensor_update_button_enabled(True)

    def update_animation(self):
        if self.is_connected:
            for ser in self.ser_list:
                _, raw_data_list = self._read_port_sensor_values(
                    ser,
                    lambda api: api.read_raw(),
                    "Error reading from",
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
                    print(f"Error processing data from port {ser.port}: {exc}")
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
        self.plotter.render()

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
        for ser in self.ser_list:
            _, raw_data_list = self._read_port_sensor_values(
                ser,
                lambda api: api.read_raw(),
                "Error reading raw data",
            )
            if raw_data_list is not None:
                print(f"Port {ser.port} raw data: {raw_data_list}")

    # Geneva demo support has been archived to `backup/func_sensor_geneva_archive.py`.
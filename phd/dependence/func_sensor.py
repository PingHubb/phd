import pyvista as pv
import numpy as np
import serial
import serial.tools.list_ports
import time
import random
import os
import re
import math
from pyvistaqt import QtInteractor
from PyQt5.QtCore import QTimer
from tqdm import tqdm
from phd.dependence.cnn_lstm import GestureCNNLSTM
from phd.dependence.mini_robot_grok import MyCobotKinematics, MyCobotAPI
from phd.dependence.gesture_logic import RuleBased, LSTM, RecordGesture, HierarchicalTransformer, ThreeLevelTransformer
from torch import nn
from scipy.spatial.transform import Rotation as R
import torch
import threading
from PyQt5 import QtCore
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon, QColor, QCursor
from PyQt5.QtCore import pyqtSignal, Qt, QRect, QEvent, QTimer
from PyQt5.QtWidgets import (QSplitter, QWidget, QGridLayout, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
                             QTreeWidget, QTreeWidgetItem, QTabWidget, QDialog, QLineEdit, QTextEdit, QSlider,
                             QGroupBox, QComboBox, QScrollArea, QListWidget, QListWidgetItem)


class data:
    def __init__(self, n_row, n_col):
        self.n_row = n_row
        self.n_col = n_col
        self.windowSize = 10
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

    def getRaw(self, rawData):
        self.rawData = rawData.T

    def getCal(self, calData):
        self.calData = calData.T

    def calDiff(self):
        self.diffData = self.rawData - self.calData

    def calDiffPer(self):
        # --- MODIFIED METHOD ---
        # This safely handles division by zero.

        # 1. Create a boolean mask to identify where the denominator (calData) is not zero.
        non_zero_mask = self.calData != 0

        # 2. Initialize the result array with zeros. This handles the case where calData is 0.
        self.diffPerData = np.zeros_like(self.calData, dtype=float)

        # 3. Use np.divide to perform the calculation only on the elements
        #    where the mask is True. The results are placed into self.diffPerData.
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

        # Place the new data. Using i-1 to match the original logic.
        self.rawDataWin[i - 1] = self.rawData
        self.diffDataWin[i - 1] = self.diffData
        self.diffPerDataWin[i - 1] = self.diffPerData

        if i >= self.windowSize:  # Calculate average once the window is full
            self.rawDataAve = np.mean(self.rawDataWin, axis=0)
            self.diffDataAve = np.mean(self.diffDataWin, axis=0)
            self.diffPerDataAve = np.mean(self.diffPerDataWin, axis=0)

            # The flip might be specific to your sensor layout, so we'll keep it.
            self.rawDataAve = np.flipud(self.rawDataAve)
            self.diffDataAve = np.flipud(self.diffDataAve)
            self.diffPerDataAve = np.flipud(self.diffPerDataAve)


class SensorModelFactory:
    """
    A factory class to handle the initialization and data structures of a sensor model.
    It encapsulates the logic for creating the grid, loading mesh/signal files,
    calculating points and normals, applying reordering, and preparing for visualization.
    """

    def __init__(self, n_row, n_col, mesh_file=None, signal_file=None, reorder_logic=None, offset_scale=0.2):
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
        """
        # Store configuration
        self.n_row = n_row
        self.n_col = n_col
        self.n_node = self.n_row * self.n_col
        self.mesh_file = mesh_file
        self.signal_file = signal_file
        self.reorder_logic = reorder_logic
        self.offset_scale = offset_scale

        # Initialize core data structures
        self._data = data(self.n_row, self.n_col)
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
            for i in range(self.n_col):
                for j in range(self.n_row - 1):
                    self.edges[edge_idx] = [2, i * self.n_row + j, i * self.n_row + j + 1]
                    edge_idx += 1
            # Horizontal edges
            for i in range(self.n_col - 1):
                for j in range(self.n_row):
                    self.edges[edge_idx] = [2, i * self.n_row + j, (i + 1) * self.n_row + j]
                    edge_idx += 1
        elif major_order == 'row':
            # Horizontal edges
            for j in range(self.n_row):
                for i in range(self.n_col - 1):
                    self.edges[edge_idx] = [2, j * self.n_col + i, j * self.n_col + i + 1]
                    edge_idx += 1
            # Vertical edges
            for j in range(self.n_row - 1):
                for i in range(self.n_col):
                    self.edges[edge_idx] = [2, j * self.n_col + i, (j + 1) * self.n_col + i]
                    edge_idx += 1

    def _generate_2d_grid(self):
        """Generates a flat 2D grid procedurally."""
        print("Done: Initiate the 2D grid construction.")
        fine_scale = 5
        size = 0.05
        fine_row, fine_col = self.n_row * fine_scale, self.n_col * fine_scale
        fine_points = np.zeros((fine_row * fine_col, 3))
        for i in range(fine_col):
            for j in range(fine_row):
                fine_points[i * fine_row + j] = [(i + 0.5) * size / fine_scale, (j + 0.5) * size / fine_scale, 0]

        fine_points -= np.mean(fine_points, axis=0)
        self._2D_map = pv.PolyData(fine_points)
        fine_normals = np.tile([0, 0, 1.0], (self._2D_map.n_points, 1))

        self.array_positions = [[] for _ in range(self.n_node)]
        for i in range(self.n_col):
            for j in range(self.n_row):
                coarse_idx = i * self.n_row + j
                for fi in range(fine_scale):
                    for fj in range(fine_scale):
                        self.array_positions[coarse_idx].append(
                            (i * fine_scale + fi) * fine_row + (j * fine_scale + fj))
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
        mapping = np.zeros(self.n_node, dtype=int)

        if self.reorder_logic == 'row_to_col':
            for old_idx in range(self.n_node):
                r, c = old_idx // self.n_col, old_idx % self.n_col
                mapping[old_idx] = c * self.n_row + r

        elif self.reorder_logic == 'row_to_col_flipped':
            for old_idx in range(self.n_node):
                r, c = old_idx // self.n_col, old_idx % self.n_col
                r_flipped, c_flipped = (self.n_row - 1) - r, (self.n_col - 1) - c
                mapping[old_idx] = c_flipped * self.n_row + r_flipped

        elif self.reorder_logic == 'row_to_col_c_flip_only':
            for old_idx in range(self.n_node):
                r, c = old_idx // self.n_col, old_idx % self.n_col
                c_flipped = (self.n_col - 1) - c
                r_keep = r
                mapping[old_idx] = c_flipped * self.n_row + r_keep

        elif self.reorder_logic == 'row_to_col_r_flip_only':
            for old_idx in range(self.n_node):
                r, c = old_idx // self.n_col, old_idx % self.n_col
                r_flipped = (self.n_row - 1) - r
                c_keep = c
                mapping[old_idx] = c_keep * self.n_row + r_flipped

        elif self.reorder_logic == 'col_to_row':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                mapping[old_idx] = r * self.n_col + c

        elif self.reorder_logic == 'col_to_row_flipped':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                r_flipped = (self.n_row - 1) - r
                c_flipped = (self.n_col - 1) - c
                mapping[old_idx] = r_flipped * self.n_col + c_flipped

        elif self.reorder_logic == 'col_to_row_c_flip_only':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                c_flipped = (self.n_col - 1) - c
                r_keep = r
                mapping[old_idx] = r_keep * self.n_col + c_flipped

        elif self.reorder_logic == 'col_to_row_r_flip_only':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                r_flipped = (self.n_row - 1) - r
                c_keep = c
                mapping[old_idx] = r_flipped * self.n_col + c_keep

        elif self.reorder_logic == 'vertical_flip':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                mapping[old_idx] = c * self.n_row + (self.n_row - 1) - r

        elif self.reorder_logic == 'horizontal_flip':
            # This logic assumes the input is column-major and flips it left-to-right.
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                mapping[old_idx] = ((self.n_col - 1) - c) * self.n_row + r

        elif self.reorder_logic == 'flip_and_rotate':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                new_c = (self.n_col - 1) - c
                mapping[old_idx] = new_c * self.n_row + r

        elif self.reorder_logic == 'rotate_180':
            for old_idx in range(self.n_node):
                c, r = old_idx // self.n_row, old_idx % self.n_row
                c_flipped = (self.n_col - 1) - c
                r_flipped = (self.n_row - 1) - r
                mapping[old_idx] = c_flipped * self.n_row + r_flipped

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
            new_idx = mapping[old_idx]
            new_points[new_idx], new_normals[new_idx] = self.points[old_idx], self.normals[old_idx]
            new_points_origin[new_idx] = self.points_origin[old_idx]
            new_array_positions[new_idx] = self.array_positions[old_idx]
            new_colors_3d[new_idx] = self.colors_3d[old_idx]

        self.points, self.normals, self.points_origin, self.array_positions, self.colors_3d = \
            new_points, new_normals, new_points_origin, new_array_positions, new_colors_3d

        # Update edges by remapping endpoint indices
        new_edges = np.copy(self.edges)
        for k in range(new_edges.shape[0]):
            new_edges[k, 1] = mapping[int(self.edges[k, 1])]
            new_edges[k, 2] = mapping[int(self.edges[k, 2])]
        self.edges = new_edges

    def _setup_visualization_data(self, major_order='column'):
        """Sets up colors and creates the final PolyData object for visualization."""
        if self._2D_map:
            self.colors = np.ones((self._2D_map.n_points, 4)) * 0.5

        is_2d_procedural_model = not self.mesh_file

        # Color the coarse and fine meshes
        for i in range(self.n_col):
            for j in range(self.n_row):
                idx = i * self.n_row + j

                if is_2d_procedural_model:
                    # --- THIS IS THE MODIFIED LINE ---
                    color = [0.3, 0.3, 0.3, 1.0]  # A dimmer, darker gray
                else:
                    color = [i / self.n_col, j / self.n_row, 0.5, 1]

                self.colors_3d[idx] = color
                if self.colors is not None and idx < len(self.array_positions):
                    for k in self.array_positions[idx]:
                        self.colors[k] = color

        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges


class MySensor:
    def __init__(self, parent) -> None:
        self.parent = parent
        self.plotter: QtInteractor = self.parent.plotter_2
        self.actionMesh = None  # no mesh yet
        self.objActor = None
        self.n_col = 0
        self.n_row = 0
        self.touch_sensitivity_scale = 0.05
        self.cal_data = []
        self.cal_data_1 = []
        self.cal_data_2 = []
        self.creatPlaneXY()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(0)
        # self.timer_geneva = QTimer()
        # self.timer_geneva.timeout.connect(self.update_animation_geneva)
        # self.timer_geneva.start(0)
        self.is_connected = False
        self.initChannel()

        # Initialize other classes instance
        self.lstm_class = LSTM(self.parent, self)
        self.rule_based_class = RuleBased(self.parent, self)
        self.record_gesture_class = RecordGesture(self)
        self.hierarchical_transformer_class = HierarchicalTransformer(self.parent, self, self.n_col, self.n_row)
        self.threelevel_hierarchical_transformer_class = ThreeLevelTransformer(self.parent, self, self.n_col,
                                                                               self.n_row)

        self.kinematics_class = MyCobotKinematics()
        self.joint_angles = [0, 0, 0, 0, 0, 0]
        self.initial_guess = np.deg2rad([0, 0, 0, 0, 0, 0])
        self.is_connected_geneva = False
        self.touch_counter_small_skin = 0
        self.touch_counter_large_skin = 0

    def saveCameraPara(self):
        self.camera_pos = self.plotter.camera.position
        self.camera_focal = self.plotter.camera.focal_point
        self.camera_view_angle = self.plotter.camera.view_angle

    def loadCameraPare(self):
        self.plotter.camera.position = self.camera_pos
        self.plotter.camera.focal_point = self.camera_focal
        self.plotter.camera.view_angle = self.camera_view_angle

    def loadCameraPare(self, camera_pos, camera_focal, camera_view_angle):
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

        planeXY = pv.Plane((0, 0, 0), (0, 0, 1), 100, 100, 100, 100)
        self.actorPlaneXY = self.plotter.add_mesh(planeXY, color='gray', style='wireframe')

    def initChannel(self):
        # --- MODIFIED METHOD ---
        self.com_options = []
        ports = serial.tools.list_ports.comports()
        self.ser = None
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

    def buildScene(self):
        # --- MODIFIED METHOD ---
        # 0) reset button appearance
        self.parent.buildScene.setText("Build Scene")
        self.parent.buildScene.setStyleSheet("color: white; font-weight: normal;")

        # 1) remove existing OBJ mesh
        if self.objActor is not None:
            try:
                self.plotter.remove_actor(self.objActor, reset_camera=False)
            except Exception:
                self.plotter.clear()
            self.objActor = None

        # 2) remove the line mesh
        if self.actionMesh is not None:
            try:
                self.plotter.remove_actor(self.actionMesh, reset_camera=False)
            except Exception:
                self.plotter.clear()
            self.actionMesh = None

        # Collect selected (highlighted) ports from the QListWidget
        selected_items = self.parent.serial_channel.selectedItems()
        if not selected_items:
            print("No serial port selected. Please select one or more ports by highlighting them.")
            return

        selected_ports = [item.text() for item in selected_items]

        # For each selected port, open a serial connection
        self.ser_list = []
        for port in selected_ports:
            try:
                ser = serial.Serial(port=f'/dev/{port}', baudrate=9600, timeout=1)
                self.ser_list.append(ser)
                print(f"Opened port: /dev/{port}")
            except Exception as e:
                print(f"Failed to open /dev/{port}: {e}")

        if not self.ser_list:
            print("Could not open any of the selected ports.")
            return

        # Reset the parameters for the new model
        self.clearParameters()

        # Process the sensor choice once
        sensor_index = self.parent.sensor_choice.currentRow()

        if sensor_index == 0:
            self.init_elbow_model()
        elif sensor_index == 1:
            self.init_kuka_model()
        elif sensor_index == 2:
            self.init_double_curve_model()
        elif sensor_index == 3:
            # Try to read from UI; fall back to defaults if controls not present
            try:
                n_row = self.parent.grid_rows_spin.value()
                n_col = self.parent.grid_cols_spin.value()
            except Exception:
                n_row, n_col = 10, 10
            self.init_2d_model(n_row=n_row, n_col=n_col)
        elif sensor_index == 4:
            self.init_half_cylinder_surface_model()
        elif sensor_index == 5:
            self.init_mini_robot_large_skin()
        elif sensor_index == 6:
            self.init_mini_robot_small_skin()
        elif sensor_index == 7:
            self.init_geneva_demo()

        self.lstm_class = LSTM(self.parent, self)
        self.rule_based_class = RuleBased(self.parent, self)
        self.record_gesture_class = RecordGesture(self)
        self.hierarchical_transformer_class = HierarchicalTransformer(self.parent, self, self.n_col, self.n_row)
        self.threelevel_hierarchical_transformer_class = ThreeLevelTransformer(self.parent, self, self.n_col,
                                                                               self.n_row)

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

        # # --- NEW CODE TO DISPLAY POINT INDICES ---
        # labels = []
        # for idx in range(self.n_node):
        #     # 1. Determine the Visual Grid coordinates (i=col, j=row)
        #     # This matches the loop logic in your update_visualization function
        #     i = idx // self.n_row
        #     j = idx % self.n_row
        #
        #     # 2. Reverse the Data Transformations to find the original Raw Index
        #     #
        #     # The data flow is:
        #     # Raw[Raw_Row] -> Reshape -> Transpose -> FlipUD -> Visual[j][i]
        #     #
        #     # To get the label, we reverse the logic:
        #     # We need to calculate what the index was BEFORE the FlipUD.
        #     # FlipUD flips the row index (j).
        #
        #     j_unflipped = (self.n_row - 1) - j
        #
        #     # Now we calculate the original raw index based on the reshape logic.
        #     # Based on how your code reshapes (np.array(rawDataList).reshape(n_col, n_row)),
        #     # the raw index is calculated as:
        #     raw_index = (i * self.n_row) + j_unflipped
        #
        #     labels.append(str(raw_index))
        #
        # # Add the labels to the plotter
        # self.plotter.add_point_labels(
        #     self.points,
        #     labels,
        #     font_size=12,
        #     text_color='white',
        #     shape_color='black',
        #     shape_opacity=0.6,
        #     point_size=0
        # )
        # # --- END OF NEW CODE ---

    def init_2d_model(self, n_row=None, n_col=None):
        # Fallback to defaults if nothing provided
        if n_row is None: n_row = 10
        if n_col is None: n_col = 10

        model = SensorModelFactory(
            n_row=n_row,
            n_col=n_col,
            offset_scale=0.0005,
            # reorder_logic= 'vertical_flip'
        ).build()

        self._initialize_from_factory(model)

    def init_elbow_model(self):
        """Initializes the curved elbow sensor model."""
        model = SensorModelFactory(
            n_row=10,
            n_col=13,
            mesh_file='/home/ping2/ros2_ws/src/phd/phd/resource/sensor/joint_1/mesh.obj',
            signal_file='/home/ping2/ros2_ws/src/phd/phd/resource/sensor/joint_1/signal.txt',
            offset_scale=0.02
        ).build()
        self._initialize_from_factory(model)

    def init_kuka_model(self):
        """Initializes the Kuka sensor model with row-major to flipped column-major reordering."""
        model = SensorModelFactory(
            n_row=10,
            n_col=8,
            mesh_file='/home/ping2/ros2_ws/src/phd/phd/resource/sensor/kuka/knitting_mesh_raw.obj',
            signal_file='/home/ping2/ros2_ws/src/phd/phd/resource/sensor/kuka/signal.txt',
            reorder_logic='row_to_col_flipped',
            offset_scale=0.2
        ).build()
        self._initialize_from_factory(model)

    def init_double_curve_model(self):
        """Initializes the double curve sensor model."""
        model = SensorModelFactory(
            n_row=10,
            n_col=10,
            mesh_file='/home/ping2/ros2_ws/src/phd/phd/resource/sensor/dualC/mesh.obj',
            signal_file='/home/ping2/ros2_ws/src/phd/phd/resource/sensor/dualC/signal.txt',
            offset_scale=0.2
        ).build()
        self._initialize_from_factory(model)

    def init_half_cylinder_surface_model(self):
        """Initializes the half-cylinder model with vertical flip reordering."""
        model = SensorModelFactory(
            n_row=10,
            n_col=9,
            mesh_file='/home/ping2/ros2_ws/src/phd/phd/resource/sensor/half_cylinder_surface/half_cylinder_2.obj',
            signal_file='/home/ping2/ros2_ws/src/phd/phd/resource/sensor/half_cylinder_surface/vertex_groups_2.txt',
            reorder_logic='flip_and_rotate',
            offset_scale=0.2
        ).build()
        self._initialize_from_factory(model)

    def init_mini_robot_large_skin(self):
        """Initializes the large Geneva sensor model with row-to-column reordering."""
        model = SensorModelFactory(
            n_row=5,
            n_col=16,
            mesh_file='/home/ping2/ros2_ws/src/phd/phd/resource/geneva/large/knitting_mesh_raw.obj',
            signal_file='/home/ping2/ros2_ws/src/phd/phd/resource/geneva/large/signal.txt',
            reorder_logic='row_to_col',
            offset_scale=0.2
        ).build()
        self._initialize_from_factory(model)

    def init_mini_robot_small_skin(self):
        """Initializes the small Geneva sensor model with row-to-column reordering."""
        model = SensorModelFactory(
            n_row=5,
            n_col=14,
            mesh_file='/home/ping2/ros2_ws/src/phd/phd/resource/geneva/small/knitting_mesh_raw.obj',
            signal_file='/home/ping2/ros2_ws/src/phd/phd/resource/geneva/small/signal.txt',
            reorder_logic='row_to_col',
            offset_scale=0.2
        ).build()
        self._initialize_from_factory(model)

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

    def updateCal(self):

        for ser in self.ser_list:
            try:
                # Point the single API object to the correct serial port for this iteration
                self.parent.sensor_api.ser = ser
                data_list = self.parent.sensor_api.update_cal()
            except Exception as e:
                print(f"Error during calibration on port {ser.port}: {e}")
                continue

            if self.n_row == 10 and self.n_col == 9:  # For Cylinder, make it from 10x10 to 10x9
                data_list = data_list[:-10]

            expected_length = self.n_row * (self.n_col + 1)
            if len(data_list) != expected_length:
                print(f"Error on port {ser.port}: Data length is {len(data_list)}, expected {expected_length}")
                continue

            calDataList = data_list[0:- self.n_row]

            self.cal_data = calDataList
            self._data.getCal(np.array(calDataList).reshape(self.n_col, self.n_row))
            self._data.clearData()

            for j in range(self._data.windowSize):
                try:
                    # The API object still points to the correct 'ser' from the outer loop
                    data_list = self.parent.sensor_api.read_raw()
                    rawDataList = data_list[0:- self.n_row]
                    self._data.getRaw(np.array(rawDataList).reshape(self.n_col, self.n_row))
                    self._data.calDiff()
                    self._data.calDiffPer()
                    self._data.getWin(j)
                except Exception as e:
                    print(f"Error during raw data processing on port {ser.port}: {e}")
                    break

        self.is_connected = True
        self.parent.sensor_update.setDisabled(False)

    def update_animation(self):
        self.saveCameraPara()

        if self.is_connected:
            for ser in self.ser_list:
                try:
                    # Point the API to the correct serial port for this iteration
                    self.parent.sensor_api.ser = ser
                    data_list = self.parent.sensor_api.read_raw()

                    if self.n_row == 10 and self.n_col == 9:  # For Cylinder, make it from 10x10 to 10x9
                        data_list = data_list[:-10]

                    rawDataList = data_list[0: - self.n_row]

                    if self.n_row == 10 and self.n_col == 10:  # For 2D, replace last 10 values with cal data, it is still 10x10
                        flat_cal_data = self._data.calData.T.flatten()
                        rawDataList[-10:] = flat_cal_data[-10:]

                    if self.n_row == 10 and self.n_col == 8:  # For KUKA, replace some values with cal data, it is still 10x8
                        flat_cal_data = self._data.calData.T.flatten()
                        rawDataList[0] = flat_cal_data[0]
                        rawDataList[1] = flat_cal_data[1]
                        rawDataList[8] = flat_cal_data[8]
                        rawDataList[9] = flat_cal_data[9]
                        rawDataList[69] = flat_cal_data[69]
                        rawDataList[70] = flat_cal_data[70]
                        rawDataList[71] = flat_cal_data[71]
                        rawDataList[72] = flat_cal_data[72]
                        rawDataList[77] = flat_cal_data[77]
                        rawDataList[78] = flat_cal_data[78]
                        rawDataList[79] = flat_cal_data[79]

                except Exception as e:
                    print(f"Error reading from port {ser.port}: {e}")
                    continue

                # The rest of the logic processes the data for this specific sensor
                try:
                    self._data.getRaw(np.array(rawDataList).reshape(self.n_col, self.n_row))
                    self._data.calDiff()
                    self._data.calDiffPer()
                    self._data.getWin(self._data.windowSize)
                except Exception as e:
                    print(f"Error processing data from port {ser.port}: {e}")
                    continue

                self.update_visualization(self._data.diffPerDataAve)

    def update_visualization(self, data):
        for i in range(self.n_col):
            for j in range(self.n_row):
                idx = i * self.n_row + j
                # It uses the up-to-date value of self.touch_sensitivity_scale
                displacement = (3 - abs(data[j][i])) * self.touch_sensitivity_scale

                target_position = self.points_origin[idx] + self.normals[idx] * displacement
                self.points[idx] += (target_position - self.points[idx]) * 0.3

                intensity = np.clip(1 - abs(data[j][i]) * 150 / 255, 0, 1)
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

    def read_sensor_diff_data(self):
        """
        Reads the processed sensor difference data and formats it to 3 significant figures.
        """
        diffPerDataAve_Reverse = self._data.diffPerDataAve.T.flatten()
        # Call the helper with a format specifier for 3 significant figures
        return self._format_data_for_display(diffPerDataAve_Reverse, fmt_spec=".3g")

    def read_sensor_raw_data(self):
        rawData = self._data.rawData.flatten()
        return self._format_data_for_display(rawData)

    def read_sensor_raw_ave_data(self):
        rawDataAve = self._data.rawDataAve.T.flatten()
        return self._format_data_for_display(rawDataAve)

    def read_raw_all_ports(self):
        # --- MODIFIED METHOD ---
        for ser in self.ser_list:
            try:
                print(f"Reading raw data from port: {ser.port}")
                # Point the API to the correct serial port for this iteration
                self.parent.sensor_api.ser = ser
                data_list = self.parent.sensor_api.read_raw()
                print(f"Port {ser.port} raw data: {data_list}")
            except Exception as e:
                print(f"Error reading raw data from {ser.port}: {e}")

    def init_geneva_demo(self):
        ######################################### GENEVA ARM MODEL ##################################################

        self.n_row = 7  # 7
        self.n_col = 8  # 8

        total_points = self.n_row * self.n_col
        total_edges = (self.n_col - 1) * (self.n_row - 1) * 2 + (self.n_col - 1) + (self.n_row - 1)

        self._data = data(self.n_row, self.n_col)
        size = 1

        # Temporary finer grid for averaging (simulate mesh averaging)
        fine_scale = 5
        fine_row, fine_col = self.n_row * fine_scale, self.n_col * fine_scale
        fine_points = np.zeros((fine_row * fine_col, 3))

        for i in range(fine_col):
            for j in range(fine_row):
                idx = i * fine_row + j
                fine_points[idx] = [(i + 0.5) * size / fine_scale, (j + 0.5) * size / fine_scale, 0]

        # Initialize arrays
        self.points = np.zeros((total_points, 3))
        self.points_origin = np.zeros((total_points, 3))
        self.normals = np.zeros((total_points, 3))
        self.array_positions = [[] for _ in range(total_points)]

        # Map fine points to coarse points (simulate averaging)
        for i in range(self.n_col):
            for j in range(self.n_row):
                idx = i * self.n_row + j
                # simulate a cluster of points (averaging)
                for fi in range(fine_scale):
                    for fj in range(fine_scale):
                        fine_idx = (i * fine_scale + fi) * fine_row + (j * fine_scale + fj)
                        self.points[idx] += fine_points[fine_idx]
                        self.normals[idx] += [0, 0, 1]  # flat normals (upward)
                self.points[idx] /= fine_scale ** 2
                self.normals[idx] /= fine_scale ** 2
                self.normals[idx] /= np.linalg.norm(self.normals[idx])

                self.points_origin[idx] = self.points[idx]
                self.points[idx] += self.normals[idx] * 0.02  # exactly like jointConstruction

        # Define edges (same as before)
        self.edges = (np.ones((total_edges, 3)) * 2).astype(int)
        edge_idx = 0
        for i in range(self.n_col - 1):
            for j in range(self.n_row - 1):
                self.edges[edge_idx] = [2, i * self.n_row + j, i * self.n_row + j + 1]
                edge_idx += 1
                self.edges[edge_idx] = [2, i * self.n_row + j, (i + 1) * self.n_row + j]
                edge_idx += 1
        for i in range(self.n_row - 1):
            self.edges[edge_idx] = [2, (self.n_col - 1) * self.n_row + i, (self.n_col - 1) * self.n_row + i + 1]
            edge_idx += 1
        for i in range(self.n_col - 1):
            self.edges[edge_idx] = [2, ((i + 1) * self.n_row) - 1, ((i + 1) * self.n_row) - 1 + self.n_row]
            edge_idx += 1

        # Initialize colors
        self.colors_face = np.ones((total_points, 4)) * 0.5
        self.colors_3d = np.ones((total_points, 4)) * 0.5
        self.colors = np.ones((total_points + total_edges, 4)) * 0.5

        # Map color indices
        for idx in range(total_points):
            self.array_positions[idx].append(idx)
        for edge_idx in range(total_edges):
            edge_points = self.edges[edge_idx][1:]
            color_idx = total_points + edge_idx
            for point in edge_points:
                self.array_positions[point].append(color_idx)

        # Create visualization mesh
        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(
            self.line_poly, scalars=self.colors_3d,
            point_size=10, line_width=3,
            render_points_as_spheres=True,
            rgb=True, name='3d_1'
        )

        ######################################### GENEVA SMALL MODEL ##################################################

        # Set grid dimensions (originally in row-major order)
        self.n_row_2 = 5  # number of rows
        self.n_col_2 = 14  # number of columns
        self.n_node_2 = self.n_row_2 * self.n_col_2

        # Initialize data structures (constructed in row-major order)
        self._data_2 = data(self.n_row_2, self.n_col_2)
        self.points_2 = np.zeros((self.n_node_2, 3))
        total_edges = ((self.n_row_2 - 1) * (self.n_col_2 - 1) * 2 +
                       (self.n_row_2 - 1) + (self.n_col_2 - 1))
        self.edges_2 = (np.ones((total_edges, 3)) * 2).astype(int)
        self.colors_3d_2 = np.ones((self.n_node_2, 4)) * 0.5

        self.is_connected_2 = False
        self.show_2D_2 = False
        self.show_PC_2 = False
        self.show_FittedMesh_2 = False

        # Build edges from point indices using row-major order.
        # In row-major order, index = j * n_col + i.
        for j in range(self.n_row_2 - 1):
            for i in range(self.n_col_2 - 1):
                idx = j * self.n_col_2 + i
                base = 2 * (j * (self.n_col_2 - 1) + i)
                # Horizontal edge: from (j, i) to (j, i+1)
                self.edges_2[base] = [2, idx, idx + 1]
                # Vertical edge: from (j, i) to (j+1, i)
                self.edges_2[base + 1] = [2, idx, idx + self.n_col_2]
        # Last row: horizontal edges along the bottom row.
        for i in range(self.n_col_2 - 1):
            idx = (self.n_row_2 - 1) * self.n_col_2 + i
            edge_index = (self.n_row_2 - 1) * (self.n_col_2 - 1) * 2 + i
            self.edges_2[edge_index] = [2, idx, idx + 1]
        # Last column: vertical edges along the rightmost column.
        for j in range(self.n_row_2 - 1):
            idx = j * self.n_col_2 + (self.n_col_2 - 1)
            edge_index = (self.n_row_2 - 1) * (self.n_col_2 - 1) * 2 + (self.n_col_2 - 1) + j
            self.edges_2[edge_index] = [2, idx, idx + self.n_col_2]

        # Load the 2D mesh and signal data.
        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/geneva/small/knitting_mesh_raw.obj'
        self._2D_map_2 = pv.read(filename)
        self._2D_map_2.points *= 100

        translation_vector = [10.0, 15.0, 0.0]  # your desired translation along x, y, z
        self._2D_map_2.translate(translation_vector, inplace=True)

        # # Adjustable transformation parameters (modify as needed)
        # adjust_x = 42.0  # translation adjustment along x (units consistent with your scene)
        # adjust_y = 0.0  # translation adjustment along y
        # adjust_z = 10.0  # translation adjustment along z
        #
        #
        # adjust_rx = np.deg2rad(180)  # rotation about x-axis (roll, in radians)
        # adjust_ry = np.deg2rad(0)  # rotation about y-axis (pitch, in radians)
        # adjust_rz = np.deg2rad(270)  # rotation about z-axis (yaw, in radians)
        #
        #
        # # Compute rotation matrices for each axis
        # R_x = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(adjust_rx), -np.sin(adjust_rx)],
        #     [0, np.sin(adjust_rx), np.cos(adjust_rx)]
        # ])
        # R_y = np.array([
        #     [np.cos(adjust_ry), 0, np.sin(adjust_ry)],
        #     [0, 1, 0],
        #     [-np.sin(adjust_ry), 0, np.cos(adjust_ry)]
        # ])
        # R_z = np.array([
        #     [np.cos(adjust_rz), -np.sin(adjust_rz), 0],
        #     [np.sin(adjust_rz), np.cos(adjust_rz), 0],
        #     [0, 0, 1]
        # ])
        #
        # # Compose the rotations (assuming ZYX order: first Rx, then Ry, then Rz)
        # R_adj = R_z @ R_y @ R_x
        #
        # # Create the adjustment transformation matrix
        # T_adj = np.eye(4)
        # T_adj[0:3, 0:3] = R_adj
        # T_adj[0:3, 3] = [adjust_x, adjust_y, adjust_z]
        #
        # # Define target position and orientation
        # x, y, z = -100.2, -66.3, 300.9
        # rx, ry, rz = np.deg2rad(2.2), np.deg2rad(1.39), np.deg2rad(-89.76)  # Orientation in radians
        # R_target = self.kinematics_class.euler_to_rotation_matrix(rx, ry, rz)
        # T_target = np.eye(4)
        # T_target[0:3, 0:3] = R_target  # Rotation
        # T_target[0:3, 3] = np.array([x, y, z])  # Position
        #
        # joint_angles_solution = self.kinematics_class.solve_ik(T_target, self.initial_guess)
        #
        # # Compute the transformation up to joint 2 (T02)
        # T_03 = self.kinematics_class.T03(joint_angles_solution)
        #
        # # Combine the adjustments with T_03; T_adj is applied first
        # T_final = T_03 @ T_adj
        #
        # # Apply the final transformation to the mesh to attach it to joint 3
        # self._2D_map_2.transform(T_final)
        #
        # # Adjustable transformation parameters (modify as needed)
        # adjust_x = 42.0  # translation adjustment along x (units consistent with your scene)
        # adjust_y = 0.0  # translation adjustment along y
        # adjust_z = 10.0  # translation adjustment along z
        #
        #
        # adjust_rx = np.deg2rad(180)  # rotation about x-axis (roll, in radians)
        # adjust_ry = np.deg2rad(0)  # rotation about y-axis (pitch, in radians)
        # adjust_rz = np.deg2rad(270)  # rotation about z-axis (yaw, in radians)
        #
        #
        # # Compute rotation matrices for each axis
        # R_x = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(adjust_rx), -np.sin(adjust_rx)],
        #     [0, np.sin(adjust_rx), np.cos(adjust_rx)]
        # ])
        # R_y = np.array([
        #     [np.cos(adjust_ry), 0, np.sin(adjust_ry)],
        #     [0, 1, 0],
        #     [-np.sin(adjust_ry), 0, np.cos(adjust_ry)]
        # ])
        # R_z = np.array([
        #     [np.cos(adjust_rz), -np.sin(adjust_rz), 0],
        #     [np.sin(adjust_rz), np.cos(adjust_rz), 0],
        #     [0, 0, 1]
        # ])
        #
        # # Compose the rotations (assuming ZYX order: first Rx, then Ry, then Rz)
        # R_adj = R_z @ R_y @ R_x
        #
        # # Create the adjustment transformation matrix
        # T_adj = np.eye(4)
        # T_adj[0:3, 0:3] = R_adj
        # T_adj[0:3, 3] = [adjust_x, adjust_y, adjust_z]
        #
        # # Define target position and orientation
        # x, y, z = -100.2, -66.3, 300.9
        # rx, ry, rz = np.deg2rad(2.2), np.deg2rad(1.39), np.deg2rad(-89.76)  # Orientation in radians
        # R_target = self.kinematics_class.euler_to_rotation_matrix(rx, ry, rz)
        # T_target = np.eye(4)
        # T_target[0:3, 0:3] = R_target  # Rotation
        # T_target[0:3, 3] = np.array([x, y, z])  # Position
        #
        # joint_angles_solution = self.kinematics_class.solve_ik(T_target, self.initial_guess)
        #
        # # Compute the transformation up to joint 2 (T02)
        # T_03 = self.kinematics_class.T03(joint_angles_solution)
        #
        # # Combine the adjustments with T_03; T_adj is applied first
        # T_final = T_03 @ T_adj
        #
        # # Apply the final transformation to the mesh to attach it to joint 3
        # self._2D_map_2.transform(T_final)

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/geneva/small/signal.txt'
        with open(filename, 'r') as file:
            lines = file.readlines()
            numbers = [int(line.strip()) for line in lines]
            self.array_positions_2 = []
            self.normals_2 = np.zeros((self.n_node_2, 3))
            for idx in range(self.n_node_2):
                self.array_positions_2.append([])
            for idx, num in enumerate(numbers):
                if num != -1:
                    self.array_positions_2[num].append(idx)

        # Set colors for the 2D mesh.
        self.colors_2 = np.ones((self._2D_map_2.n_points, 4)) * 0.5
        # Use row-major indexing for colors.
        for j in range(self.n_row_2):
            for i in range(self.n_col_2):
                idx = j * self.n_col_2 + i
                self.colors_3d_2[idx] = [i / self.n_col_2, j / self.n_row_2, 0, 1]
                for k in self.array_positions_2[idx]:
                    self.colors_2[k] = [i / self.n_col_2, j / self.n_row_2, 0, 1]
        self.plotter.add_mesh(self._2D_map_2, show_edges=True, scalars=self.colors_2, rgb=True)

        # Extract surface normals and compute average point locations.
        a = self._2D_map_2.extract_surface()
        b = a.point_normals
        self.points_origin_2 = np.zeros((self.n_node_2, 3))
        for idx in tqdm(range(self.n_node_2)):
            for jdx in self.array_positions_2[idx]:
                self.normals_2[idx] += b[jdx]
                self.points_2[idx] += self._2D_map_2.GetPoint(jdx)
            self.normals_2[idx] = self.normals_2[idx] / len(self.array_positions_2[idx])
            self.normals_2[idx] = self.normals_2[idx] / np.linalg.norm(self.normals_2[idx])
            self.points_2[idx] = self.points_2[idx] / len(self.array_positions_2[idx])
            self.points_origin_2[idx] = self.points_2[idx]
            self.points_2[idx] += self.normals_2[idx] * 0.2

        # --- Apply reordering: Convert from row-major to column-major ordering.
        # In the original row-major order, old_index = j * n_col + i.
        # For column-major ordering, we want new_index = i * n_row + j.
        mapping = np.zeros(self.n_node_2, dtype=int)
        for old_idx in range(self.n_node_2):
            r = old_idx // self.n_col_2  # original row
            c = old_idx % self.n_col_2  # original column
            new_idx = c * self.n_row_2 + r
            mapping[old_idx] = new_idx

        # Reorder point-based arrays according to the mapping.
        new_points = np.zeros_like(self.points_2)
        new_normals = np.zeros_like(self.normals_2)
        new_points_origin = np.zeros_like(self.points_origin_2)
        new_array_positions = [None] * self.n_node_2
        new_colors_3d = np.zeros_like(self.colors_3d_2)
        for old_idx in range(self.n_node_2):
            new_idx = mapping[old_idx]
            new_points[new_idx] = self.points_2[old_idx]
            new_normals[new_idx] = self.normals_2[old_idx]
            new_points_origin[new_idx] = self.points_origin_2[old_idx]
            new_array_positions[new_idx] = self.array_positions_2[old_idx]
            new_colors_3d[new_idx] = self.colors_3d_2[old_idx]
        self.points_2 = new_points
        self.normals_2 = new_normals
        self.points_origin_2 = new_points_origin
        self.array_positions_2 = new_array_positions
        self.colors_3d_2 = new_colors_3d

        # Update edges: remap each endpoint using the new mapping.
        new_edges = np.copy(self.edges_2)
        for k in range(new_edges.shape[0]):
            new_edges[k, 1] = mapping[int(new_edges[k, 1])]
            new_edges[k, 2] = mapping[int(new_edges[k, 2])]
        self.edges_2 = new_edges

        # --- Visualization ---
        # Create a PolyData object for the edge mesh.
        self.line_poly_2 = pv.PolyData(self.points_2)
        self.line_poly_2.lines = self.edges_2
        self.actionMesh_2 = self.plotter.add_mesh(self.line_poly_2, scalars=self.colors_3d_2,
                                                  point_size=10, line_width=3,
                                                  render_points_as_spheres=True,
                                                  rgb=True, name='3d_2')

        ######################################### GENEVA LARGE MODEL ##################################################
        self.n_row_3 = 5  # number of rows
        self.n_col_3 = 16  # number of columns
        self.n_node_3 = self.n_row_3 * self.n_col_3

        # Initialize data structures (constructed in row-major order)
        self._data_3 = data(self.n_row_3, self.n_col_3)
        self.points_3 = np.zeros((self.n_node_3, 3))
        total_edges = ((self.n_row_3 - 1) * (self.n_col_3 - 1) * 2 +
                       (self.n_row_3 - 1) + (self.n_col_3 - 1))
        self.edges_3 = (np.ones((total_edges, 3)) * 2).astype(int)
        self.colors_3d_3 = np.ones((self.n_node_3, 4)) * 0.5

        self.is_connected_3 = False
        self.show2d_3 = False
        self.show_PC_3 = False
        self.show_FittedMesh_3 = False

        # --- Build edges using row-major ordering ---
        # In row-major order, point index = row * n_col + col.
        for r in range(self.n_row_3 - 1):
            for c in range(self.n_col_3 - 1):
                idx = r * self.n_col_3 + c
                base = 2 * (r * (self.n_col_3 - 1) + c)
                # Horizontal edge: from (r, c) to (r, c+1)
                self.edges_3[base] = [2, idx, idx + 1]
                # Vertical edge: from (r, c) to (r+1, c)
                self.edges_3[base + 1] = [2, idx, idx + self.n_col_3]
        # Last row: horizontal edges along the bottom row.
        for c in range(self.n_col_3 - 1):
            idx = (self.n_row_3 - 1) * self.n_col_3 + c
            edge_index = (self.n_row_3 - 1) * (self.n_col_3 - 1) * 2 + c
            self.edges_3[edge_index] = [2, idx, idx + 1]
        # Last column: vertical edges along the rightmost column.
        for r in range(self.n_row_3 - 1):
            idx = r * self.n_col_3 + (self.n_col_3 - 1)
            edge_index = (self.n_row_3 - 1) * (self.n_col_3 - 1) * 2 + (self.n_col_3 - 1) + r
            self.edges_3[edge_index] = [2, idx, idx + self.n_col_3]

        # Load the 2D mesh and signal data.
        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/geneva/large/knitting_mesh_raw.obj'
        self._2D_map_3 = pv.read(filename)
        self._2D_map_3.points *= 100

        translation_vector = [0.0, 15.0, 0.0]  # your desired translation along x, y, z
        self._2D_map_3.translate(translation_vector, inplace=True)

        # # Adjustable transformation parameters (modify as needed)
        # adjust_x = 50.0  # translation adjustment along x (units consistent with your scene)
        # adjust_y = 0.0  # translation adjustment along y
        # adjust_z = 50.0  # translation adjustment along z
        #
        #
        # adjust_rx = np.deg2rad(0)  # rotation about x-axis (roll, in radians)
        # adjust_ry = np.deg2rad(0)  # rotation about y-axis (pitch, in radians)
        # adjust_rz = np.deg2rad(90)  # rotation about z-axis (yaw, in radians)
        #
        #
        # # Compute rotation matrices for each axis
        # R_x = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(adjust_rx), -np.sin(adjust_rx)],
        #     [0, np.sin(adjust_rx), np.cos(adjust_rx)]
        # ])
        # R_y = np.array([
        #     [np.cos(adjust_ry), 0, np.sin(adjust_ry)],
        #     [0, 1, 0],
        #     [-np.sin(adjust_ry), 0, np.cos(adjust_ry)]
        # ])
        # R_z = np.array([
        #     [np.cos(adjust_rz), -np.sin(adjust_rz), 0],
        #     [np.sin(adjust_rz), np.cos(adjust_rz), 0],
        #     [0, 0, 1]
        # ])
        #
        # # Compose the rotations (assuming ZYX order: first Rx, then Ry, then Rz)
        # R_adj = R_z @ R_y @ R_x
        #
        # # Create the adjustment transformation matrix
        # T_adj = np.eye(4)
        # T_adj[0:3, 0:3] = R_adj
        # T_adj[0:3, 3] = [adjust_x, adjust_y, adjust_z]
        #
        # # Define target position and orientation
        # x, y, z = -100.2, -66.3, 300.9
        # rx, ry, rz = np.deg2rad(2.2), np.deg2rad(1.39), np.deg2rad(-89.76)  # Orientation in radians
        # R_target = self.kinematics_class.euler_to_rotation_matrix(rx, ry, rz)
        # T_target = np.eye(4)
        # T_target[0:3, 0:3] = R_target  # Rotation
        # T_target[0:3, 3] = np.array([x, y, z])  # Position
        #
        # joint_angles_solution = self.kinematics_class.solve_ik(T_target, self.initial_guess)
        #
        # # Compute the transformation up to joint 2 (T02)
        # T_02 = self.kinematics_class.T02(joint_angles_solution)
        #
        # # Combine the adjustments with T_03; T_adj is applied first
        # T_final = T_02 @ T_adj
        #
        # # Apply the final transformation to the mesh to attach it to joint 3
        # self._2D_map_3.transform(T_final)

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/geneva/large/signal.txt'
        with open(filename, 'r') as file:
            lines = file.readlines()
            numbers = [int(line.strip()) for line in lines]
            self.array_positions_3 = []
            self.normals_3 = np.zeros((self.n_node_3, 3))
            for idx in range(self.n_node_3):
                self.array_positions_3.append([])
            for idx, num in enumerate(numbers):
                if num != -1:
                    self.array_positions_3[num].append(idx)

        # Set colors for the 2D mesh.
        self.colors_3 = np.ones((self._2D_map_3.n_points, 4)) * 0.5
        # Use row-major ordering for colors.
        for r in range(self.n_row_3):
            for c in range(self.n_col_3):
                idx = r * self.n_col_3 + c
                self.colors_3d_3[idx] = [c / self.n_col_3, r / self.n_row_3, 0, 1]
                for k in self.array_positions_3[idx]:
                    self.colors_3[k] = [c / self.n_col_3, r / self.n_row_3, 0, 1]
        self.plotter.add_mesh(self._2D_map_3, show_edges=True, scalars=self.colors_3, rgb=True)

        # Extract surface normals and compute average point locations.
        a = self._2D_map_3.extract_surface()
        b = a.point_normals
        self.points_origin_3 = np.zeros((self.n_node_3, 3))
        for idx in tqdm(range(self.n_node_3)):
            for jdx in self.array_positions_3[idx]:
                self.normals_3[idx] += b[jdx]
                self.points_3[idx] += self._2D_map_3.GetPoint(jdx)
            self.normals_3[idx] = self.normals_3[idx] / len(self.array_positions_3[idx])
            self.normals_3[idx] = self.normals_3[idx] / np.linalg.norm(self.normals_3[idx])
            self.points_3[idx] = self.points_3[idx] / len(self.array_positions_3[idx])
            self.points_origin_3[idx] = self.points_3[idx]
            self.points_3[idx] += self.normals_3[idx] * 0.2

        # --- Apply reordering: convert from row-major to column-major ordering.
        # In the original row-major order, old_index = r * n_col + c.
        # We want new_index = c * n_row + r.
        mapping = np.zeros(self.n_node_3, dtype=int)
        for old_idx in range(self.n_node_3):
            r = old_idx // self.n_col_3  # original row
            c = old_idx % self.n_col_3  # original column
            new_idx = c * self.n_row_3 + r
            mapping[old_idx] = new_idx

        # Reorder point-based arrays according to the mapping.
        new_points = np.zeros_like(self.points_3)
        new_normals = np.zeros_like(self.normals_3)
        new_points_origin = np.zeros_like(self.points_origin_3)
        new_array_positions = [None] * self.n_node_3
        new_colors_3d = np.zeros_like(self.colors_3d_3)
        for old_idx in range(self.n_node_3):
            new_idx = mapping[old_idx]
            new_points[new_idx] = self.points_3[old_idx]
            new_normals[new_idx] = self.normals_3[old_idx]
            new_points_origin[new_idx] = self.points_origin_3[old_idx]
            new_array_positions[new_idx] = self.array_positions_3[old_idx]
            new_colors_3d[new_idx] = self.colors_3d_3[old_idx]
        self.points_3 = new_points
        self.normals_3 = new_normals
        self.points_origin_3 = new_points_origin
        self.array_positions_3 = new_array_positions
        self.colors_3d_3 = new_colors_3d

        # Update edges: remap each endpoint using the new mapping.
        new_edges = np.copy(self.edges_3)
        for k in range(new_edges.shape[0]):
            new_edges[k, 1] = mapping[int(new_edges[k, 1])]
            new_edges[k, 2] = mapping[int(new_edges[k, 2])]
        self.edges_3 = new_edges

        # --- Visualization ---
        # Create a PolyData object for the edge mesh.
        self.line_poly_3 = pv.PolyData(self.points_3)
        self.line_poly_3.lines = self.edges_3
        self.actionMesh = self.plotter.add_mesh(self.line_poly_3, scalars=self.colors_3d_3,
                                                point_size=10, line_width=3,
                                                render_points_as_spheres=True,
                                                rgb=True, name='3d_3')

    def startSensor_geneva(self):
        # --- MODIFIED METHOD ---
        for i, ser in enumerate(self.ser_list):
            # Point the API to the correct serial port for this iteration
            self.parent.sensor_api.ser = ser

            if i == 0:
                n_row, n_col, data_obj = self.n_row, self.n_col, self._data
            elif i == 1:
                n_row, n_col, data_obj = self.n_row_2, self.n_col_2, self._data_2
                continue
            elif i == 2:
                n_row, n_col, data_obj = self.n_row_3, self.n_col_3, self._data_3
            else:
                print(f"No sensor dimensions defined for sensor index {i}. Skipping port {ser.port}.")
                continue

            print(f"Processing sensor on port {ser.port} with dimensions (n_row: {n_row}, n_col: {n_col})")
            try:
                data_list = self.parent.sensor_api.update_cal()
            except Exception as e:
                print(f"Error during calibration on port {ser.port}: {e}")
                continue

            expected_length = n_row * (n_col + 1)
            if len(data_list) != expected_length:
                print(f"Error on port {ser.port}: Data length is {len(data_list)}, expected {expected_length}")
                continue

            calDataList = data_list[0: - n_row]
            if i == 1: self.cal_data_1 = calDataList
            if i == 2: self.cal_data_2 = calDataList

            data_obj.getCal(np.array(calDataList).reshape(n_col, n_row))
            data_obj.clearData()

            for j in range(data_obj.windowSize):
                try:
                    data_list = self.parent.sensor_api.read_raw()
                    rawDataList = data_list[0: - n_row]
                    data_obj.getRaw(np.array(rawDataList).reshape(n_col, n_row))
                    data_obj.calDiff()
                    data_obj.calDiffPer()
                    data_obj.getWin(j)
                except Exception as e:
                    print(f"Error during raw data processing on port {ser.port}: {e}")
                    break

            print(f"Sensor on port {ser.port} processed successfully.")

        self.is_connected_geneva = True
        self.parent.sensor_update.setDisabled(False)

    def update_animation_geneva(self):
        # --- MODIFIED METHOD ---
        self.saveCameraPara()

        if self.is_connected_geneva:
            for i, ser in enumerate(self.ser_list):
                # Point the API to the correct serial port
                self.parent.sensor_api.ser = ser

                if i == 0:
                    n_row, n_col, data_obj = self.n_row, self.n_col, self._data
                elif i == 1:
                    n_row, n_col, data_obj = self.n_row_2, self.n_col_2, self._data_2
                    continue
                elif i == 2:
                    n_row, n_col, data_obj = self.n_row_3, self.n_col_3, self._data_3
                else:
                    continue

                try:
                    data_list = self.parent.sensor_api.read_raw()
                    rawDataList = data_list[0: - n_row]
                except Exception as e:
                    print(f"Error reading from port {ser.port}: {e}")
                    continue

                # Logic for sensor-specific actions
                if i == 1:
                    # ... (your existing logic for small skin) ...
                    diffPerDataAve_Reverse = data_obj.diffPerDataAve.T.flatten()
                    transformed_data = np.where(diffPerDataAve_Reverse < -1, 1, 0)
                    if 1 in transformed_data:
                        self.touch_counter_small_skin += 1
                        print(f"Small skin detected hand approaching: {self.touch_counter_small_skin}")
                        self.parent.mini_robot.stop()
                if i == 2:
                    # ... (your existing logic for large skin) ...
                    diffPerDataAve_Reverse = data_obj.diffPerDataAve.T.flatten()
                    transformed_data = np.where(diffPerDataAve_Reverse < -2, 1, 0)
                    if 1 in transformed_data:
                        self.touch_counter_large_skin += 1
                        print(f"Large skin detected hand approaching: {self.touch_counter_large_skin}")
                        self.parent.mini_robot.stop()

                # Process raw data for the current sensor
                try:
                    data_obj.getRaw(np.array(rawDataList).reshape(n_col, n_row))
                    data_obj.calDiff()
                    data_obj.calDiffPer()
                    data_obj.getWin(data_obj.windowSize)
                except Exception as e:
                    print(f"Error processing data from port {ser.port}: {e}")
                    continue

                self.update_visualization_geneva(data_obj.diffPerDataAve, i)

    def update_visualization_geneva(self, data, sensor_index):
        """
        Update the visualization for a given sensor's data.

        Parameters:
          data: 2D array of processed sensor data.
          sensor_index: 0 for the first sensor (using self.n_row, self.n_col, etc.),
                        1 for the second sensor (using self.n_row_2, self.n_col_2, etc.),
                        2 for the third sensor (using self.n_row_3, self.n_col_3, etc.).
        """
        # Select parameters based on sensor_index
        if sensor_index == 0:
            n_row = self.n_row
            n_col = self.n_col
            points = self.points
            points_origin = self.points_origin
            normals = self.normals
            colors_3d = self.colors_3d
            array_positions = self.array_positions
            line_poly = self.line_poly
            plotter = self.plotter
        elif sensor_index == 1:
            n_row = self.n_row_2
            n_col = self.n_col_2
            points = self.points_2
            points_origin = self.points_origin_2
            normals = self.normals_2
            colors_3d = self.colors_3d_2
            array_positions = self.array_positions_2
            line_poly = self.line_poly_2
            plotter = self.plotter
        elif sensor_index == 2:
            n_row = self.n_row_3
            n_col = self.n_col_3
            points = self.points_3
            points_origin = self.points_origin_3
            normals = self.normals_3
            colors_3d = self.colors_3d_3
            array_positions = self.array_positions_3
            line_poly = self.line_poly_3
            plotter = self.plotter
        else:
            print("Invalid sensor index for visualization update.")
            return

        smoothing_factor = 0.05  # Adjust for smoothness

        for i in range(n_col):
            for j in range(n_row):
                idx = i * n_row + j
                # Calculate displacement based on data value at [j][i]
                displacement = (1.5 - abs(data[j][i])) * smoothing_factor
                # Determine target position by displacing the original point along its normal
                target_position = points_origin[idx] + normals[idx] * displacement
                # Interpolate for smooth transitions
                points[idx] += (target_position - points[idx]) * 0.3

                # Calculate intensity for coloring based on the data value
                intensity = np.clip(1 - abs(data[j][i]) * 150 / 255, 0, 1)
                colors_3d[idx] = [1, intensity, intensity, 1]
                # If you have a separate overall colors array associated with each point,
                # update it as well using the indices in array_positions.
                for k in array_positions[idx]:
                    # For example, if you have self.colors for sensor 0, self.colors_2 for sensor 1, etc.
                    # you could update it here. For now, this is a placeholder.
                    pass

        # Update the polydata and render the plotter
        line_poly.points = points
        line_poly.point_data.set_scalars(colors_3d)
        plotter.render()

    def updateCal_geneva(self):
        self.startSensor_geneva()
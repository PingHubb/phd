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
        self.diffPerData = 100 * self.diffData / self.calData

    def clearData(self):
        self.rawDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffPerDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.rawDataAve = np.zeros((self.n_row, self.n_col))
        self.diffDataAve = np.zeros((self.n_row, self.n_col))
        self.diffPerDataAve = np.zeros((self.n_row, self.n_col))

    def getWin(self, i):
        if i == self.windowSize:
            self.rawDataWin[:-1] = self.rawDataWin[1:]
            self.diffDataWin[:-1] = self.diffDataWin[1:]
            self.diffPerDataWin[:-1] = self.diffPerDataWin[1:]
        self.rawDataWin[i - 1] = self.rawData
        self.diffDataWin[i - 1] = self.diffData
        self.diffPerDataWin[i - 1] = self.diffPerData
        if i == self.windowSize:
            self.rawDataAve = np.mean(self.rawDataWin, axis=0)
            self.diffDataAve = np.mean(self.diffDataWin, axis=0)
            self.diffPerDataAve = np.mean(self.diffPerDataWin, axis=0)
            self.rawDataAve = np.flipud(self.rawDataAve)
            self.diffDataAve = np.flipud(self.diffDataAve)
            self.diffPerDataAve = np.flipud(self.diffPerDataAve)


class data_2:
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
        self.diffPerData = 100 * self.diffData / self.calData

    def clearData(self):
        self.rawDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffPerDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.rawDataAve = np.zeros((self.n_row, self.n_col))
        self.diffDataAve = np.zeros((self.n_row, self.n_col))
        self.diffPerDataAve = np.zeros((self.n_row, self.n_col))

    def getWin(self, i):
        if i == self.windowSize:
            self.rawDataWin[:-1] = self.rawDataWin[1:]
            self.diffDataWin[:-1] = self.diffDataWin[1:]
            self.diffPerDataWin[:-1] = self.diffPerDataWin[1:]
        self.rawDataWin[i - 1] = self.rawData
        self.diffDataWin[i - 1] = self.diffData
        self.diffPerDataWin[i - 1] = self.diffPerData
        if i == self.windowSize:
            self.rawDataAve = np.mean(self.rawDataWin, axis=0)
            self.diffDataAve = np.mean(self.diffDataWin, axis=0)
            self.diffPerDataAve = np.mean(self.diffPerDataWin, axis=0)
            self.rawDataAve = np.flipud(self.rawDataAve)
            self.diffDataAve = np.flipud(self.diffDataAve)
            self.diffPerDataAve = np.flipud(self.diffPerDataAve)


class data_3:
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
        self.diffPerData = 100 * self.diffData / self.calData

    def clearData(self):
        self.rawDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.diffPerDataWin = np.zeros((self.windowSize, self.n_row, self.n_col))
        self.rawDataAve = np.zeros((self.n_row, self.n_col))
        self.diffDataAve = np.zeros((self.n_row, self.n_col))
        self.diffPerDataAve = np.zeros((self.n_row, self.n_col))

    def getWin(self, i):
        if i == self.windowSize:
            self.rawDataWin[:-1] = self.rawDataWin[1:]
            self.diffDataWin[:-1] = self.diffDataWin[1:]
            self.diffPerDataWin[:-1] = self.diffPerDataWin[1:]
        self.rawDataWin[i - 1] = self.rawData
        self.diffDataWin[i - 1] = self.diffData
        self.diffPerDataWin[i - 1] = self.diffPerData
        if i == self.windowSize:
            self.rawDataAve = np.mean(self.rawDataWin, axis=0)
            self.diffDataAve = np.mean(self.diffDataWin, axis=0)
            self.diffPerDataAve = np.mean(self.diffPerDataWin, axis=0)
            self.rawDataAve = np.flipud(self.rawDataAve)
            self.diffDataAve = np.flipud(self.diffDataAve)
            self.diffPerDataAve = np.flipud(self.diffPerDataAve)


class MySensor:
    def __init__(self, parent) -> None:
        self.parent = parent
        self.plotter: QtInteractor = self.parent.plotter_2
        self.listModel = []
        self.listActor = []
        self.test = []
        self.test_1 = []
        self.creatPlaneXY()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(0)
        self.timer_geneva = QTimer()
        self.timer_geneva.timeout.connect(self.update_animation_geneva)
        self.timer_geneva.start(0)
        self.frame_count = 0
        self.last_time = time.time()
        self.is_connected = False
        self.initChannel()
        # Initialize other classes instance
        self.lstm_class = LSTM(self.parent, self)
        self.rule_based_class = RuleBased(self.parent, self)
        self.record_gesture_class = RecordGesture(self)

        self.kinematics_class = MyCobotKinematics()
        self.joint_angles = [0, 0, 0, 0, 0, 0]
        self.initial_guess = np.deg2rad([0, 0, 0, 0, 0, 0])
        self.large_skin_activated = False
        self.small_skin_activated = False
        self.elbow_skin_active = False
        self.is_connected_geneva = False
        self.touch_counter_small_skin = 0
        self.touch_counter_large_skin = 0

        # Path tracking variables
        self.is_tracking_path = False
        self.tracked_path = []

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
        self.com_options = []
        ports = serial.tools.list_ports.comports()
        self.ser = None
        if ports:
            ports = sorted(ports, key=lambda port: (0, int(port.name.replace('ttyACM', ''))) if port.name.startswith(
                'ttyACM') else (1, port.name))
            for port in ports:
                self.com_options.append(port.name)
                # Create a checkable item for each port:
                item = QListWidgetItem(port.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.parent.serial_channel.addItem(item)
        else:
            print("No serial ports found. Please connect the device and retry.")
            return

    def buildScene(self):
        # Collect selected (checked) ports from the QListWidget
        selected_ports = []
        for index in range(self.parent.serial_channel.count()):
            item = self.parent.serial_channel.item(index)
            if item.checkState() == Qt.Checked:
                selected_ports.append(item.text())

        # Check if any port is selected
        if not selected_ports:
            print("No serial port selected. Please select at least one port.")
            return

        # For each selected port, open a serial connection (or do what you need)
        self.ser_list = []
        for port in selected_ports:
            try:
                ser = serial.Serial(port=f'/dev/{port}', baudrate=9600, timeout=1)
                self.ser_list.append(ser)
                print(f"Opened port: /dev/{port}")
            except Exception as e:
                print(f"Failed to open /dev/{port}: {e}")

        # Now, process the sensor_choice. This is done once, outside the loop.
        sensor_index = self.parent.sensor_choice.currentIndex()
        if sensor_index == 0:
            self.init_elbow_model()
        elif sensor_index == 2:
            self.init_double_curve_model()
        elif sensor_index == 3:
            self.init_2d_model()
        elif sensor_index == 4:
            self.init_half_cylinder_surface_model()
        elif sensor_index == 5:
            self.init_geneva_large()
        elif sensor_index == 6:
            self.init_geneva_small()
        elif sensor_index == 1:
            self.init_robot()

    def init_2d_model(self):
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
            rgb=True, name='3d'
        )

        # # === Add labels for visualization ===
        # # Label each point with its index.
        # point_labels = [str(i) for i in range(total_points)]
        # self.plotter.add_point_labels(
        #     self.points, point_labels,
        #     point_size=10, font_size=12,
        #     text_color='black', name='point_labels'
        # )
        #
        # # For each edge, compute the midpoint and create a label showing its endpoints.
        # edge_midpoints = []
        # edge_labels = []
        # for idx, edge in enumerate(self.edges):
        #     p0 = self.points[int(edge[1])]
        #     p1 = self.points[int(edge[2])]
        #     midpoint = (p0 + p1) / 2.0
        #     edge_midpoints.append(midpoint)
        #     edge_labels.append(f"{int(edge[1])}-{int(edge[2])}")
        # edge_midpoints = np.array(edge_midpoints)
        # self.plotter.add_point_labels(
        #     edge_midpoints, edge_labels,
        #     point_size=10, font_size=12,
        #     text_color='blue', name='edge_labels'
        # )

        # Update UI elements
        self.parent.sensor_choice.setDisabled(True)
        self.parent.serial_channel.setDisabled(True)
        self.parent.buildScene.setText("Scene Built")
        self.parent.buildScene.setDisabled(True)
        self.parent.sensor_start.setDisabled(False)

    def init_elbow_model(self):
        self.n_row = 10
        self.n_col = 13
        self.n_node = self.n_row * self.n_col

        self._data = data(self.n_row, self.n_col)
        self.points = np.zeros((self.n_node, 3))
        self.edges = (np.ones(
            ((self.n_col - 1) * (self.n_row - 1) * 2 + (self.n_col - 1) + (self.n_row - 1), 3)) * 2).astype(int)

        self.colors_3d = np.ones((self.n_node, 4)) * 0.5

        self.is_connected = False
        self.show_2D = False
        self.show_PC = False
        self.show_FittedMesh = False

        print("Done: Initiate the joint construction.")

        for i in range(self.n_col - 1):
            for j in range(self.n_row - 1):
                self.edges[2 * (i * (self.n_row - 1) + j)] = [2, i * self.n_row + j, i * self.n_row + j + 1]
                self.edges[2 * (i * (self.n_row - 1) + j) + 1] = [2, i * self.n_row + j, (i + 1) * self.n_row + j]
        for i in range(self.n_row - 1):
            self.edges[(self.n_row - 1) * (self.n_col - 1) * 2 + i] = [2, (self.n_col - 1) * self.n_row + i,
                                                                       (self.n_col - 1) * self.n_row + i + 1]
        for i in range(self.n_col - 1):
            self.edges[(self.n_row - 1) * (self.n_col - 1) * 2 + (self.n_row - 1) + i] = [2, ((i + 1) * self.n_row) - 1,
                                                                                          ((
                                                                                                       i + 1) * self.n_row) - 1 + self.n_row]

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/joint_1/mesh.obj'
        self._2D_map = pv.read(filename)

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/joint_1/signal.txt'
        with open(filename, 'r') as file:
            lines = file.readlines()
            numbers = [int(line.strip()) for line in lines]
            self.array_positions = []
            self.normals = np.zeros((self.n_node, 3))

            for i in range(156):
                self.array_positions.append([])
            for idx, num in enumerate(numbers):
                if num != -1:
                    self.array_positions[num].append(idx)

        print("Done: Load the mesh and signal data.")

        for i in range((self.n_col) - 1, -1, -1):
            del self.array_positions[i * 12 + 11]
            del self.array_positions[i * 12]

        self.colors = np.ones((self._2D_map.n_points, 4)) * 0.5
        for i in range(self.n_col):
            for j in range(self.n_row):
                self.colors_3d[i * self.n_row + j] = [i / self.n_col, j / self.n_row, 0, 1]
                for k in self.array_positions[i * self.n_row + j]:
                    self.colors[k] = [i / self.n_col, j / self.n_row, 0, 1]
        self.plotter.add_mesh(self._2D_map, show_edges=True, scalars=self.colors, rgb=True)
        # TreeWidgetItem(self.parent.widget_tree,"_2D_map",0,0)

        a = self._2D_map.extract_surface()
        b = a.point_normals

        self.points_origin = np.zeros((self.n_node, 3))
        for i in tqdm(range(self.n_node)):
            for j in self.array_positions[i]:
                self.normals[i] += b[j]
                self.points[i] += self._2D_map.GetPoint(j)
            self.normals[i] = self.normals[i] / len(self.array_positions[i])
            self.normals[i] = self.normals[i] / np.linalg.norm(self.normals[i])
            self.points[i] = self.points[i] / len(self.array_positions[i])
            self.points_origin[i] = self.points[i]
            self.points[i] += self.normals[i] * 0.02

        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(self.line_poly, scalars=self.colors_3d, point_size=10, line_width=3,
                                                render_points_as_spheres=True, rgb=True, name='3d')

        # # --- Add labels for each point index ---
        # point_labels = [str(i) for i in range(self.n_node)]
        # self.plotter.add_point_labels(
        #     self.points,  # array of point coords, shape (n_node, 3)
        #     point_labels,  # list of strings '0', '1', ..., 'n_node-1'
        #     point_size=10,  # size of the point “dot” rendered
        #     font_size=12,  # label text size
        #     text_color='black',  # color of the index text
        #     name='point_labels',  # a unique name so you can toggle it later
        #     shape_opacity=0.0  # transparent background for the label
        # )

        self.elbow_skin_active = True

        self.parent.sensor_choice.setDisabled(True)
        self.parent.serial_channel.setDisabled(True)
        self.parent.buildScene.setText("Sence Builded")
        self.parent.buildScene.setDisabled(True)
        self.parent.sensor_start.setDisabled(False)

    def init_double_curve_model(self):
        self.n_row = 10
        self.n_col = 10
        self.n_node = self.n_row * self.n_col

        self._data = data(self.n_row, self.n_col)
        self.points = np.zeros((self.n_node, 3))
        self.edges = (np.ones(
            ((self.n_col - 1) * (self.n_row - 1) * 2 + (self.n_col - 1) + (self.n_row - 1), 3)) * 2).astype(int)

        self.colors_3d = np.ones((self.n_node, 4)) * 0.5

        self.is_connected = False
        self.show_2D = False
        self.show_PC = False
        self.show_FittedMesh = False

        for i in range(self.n_col - 1):
            for j in range(self.n_row - 1):
                self.edges[2 * (i * (self.n_row - 1) + j)] = [2, i * self.n_row + j, i * self.n_row + j + 1]
                self.edges[2 * (i * (self.n_row - 1) + j) + 1] = [2, i * self.n_row + j, (i + 1) * self.n_row + j]
        for i in range(self.n_row - 1):
            self.edges[(self.n_row - 1) * (self.n_col - 1) * 2 + i] = [2, (self.n_col - 1) * self.n_row + i,
                                                                       (self.n_col - 1) * self.n_row + i + 1]
        for i in range(self.n_col - 1):
            self.edges[(self.n_row - 1) * (self.n_col - 1) * 2 + (self.n_row - 1) + i] = [2, ((i + 1) * self.n_row) - 1,
                                                                                          ((
                                                                                                       i + 1) * self.n_row) - 1 + self.n_row]

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/dualC/mesh.obj'
        self._2D_map = pv.read(filename)

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/dualC/signal.txt'
        with open(filename, 'r') as file:
            lines = file.readlines()
            numbers = [int(line.strip()) for line in lines]
            self.array_positions = []
            self.normals = np.zeros((self.n_node, 3))

            for i in range(self.n_node):
                self.array_positions.append([])
            for idx, num in enumerate(numbers):
                if num != -1:
                    self.array_positions[num].append(idx)

        self.colors = np.ones((self._2D_map.n_points, 4)) * 0.5
        for i in range(self.n_col):
            for j in range(self.n_row):
                self.colors_3d[i * self.n_row + j] = [i / self.n_col, j / self.n_row, 0, 1]
                for k in self.array_positions[i * self.n_row + j]:
                    self.colors[k] = [i / self.n_col, j / self.n_row, 0, 1]
        self.plotter.add_mesh(self._2D_map, show_edges=True, scalars=self.colors, rgb=True)
        # TreeWidgetItem(self.parent.widget_tree,"_2D_map",0,0)

        a = self._2D_map.extract_surface()
        b = a.point_normals
        self.points_origin = np.zeros((self.n_node, 3))
        for i in tqdm(range(self.n_node)):
            for j in self.array_positions[i]:
                self.normals[i] += b[j]
                self.points[i] += self._2D_map.GetPoint(j)
            self.normals[i] = self.normals[i] / len(self.array_positions[i])
            self.normals[i] = self.normals[i] / np.linalg.norm(self.normals[i])
            self.points[i] = self.points[i] / len(self.array_positions[i])
            self.points_origin[i] = self.points[i]
            self.points[i] += self.normals[i] * 0.2

        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(self.line_poly, scalars=self.colors_3d, point_size=10, line_width=3,
                                                render_points_as_spheres=True, rgb=True, name='3d')
        # TreeWidgetItem(self.parent.widget_tree,"_3D_mesh",0,0)

        self.parent.sensor_choice.setDisabled(True)
        self.parent.serial_channel.setDisabled(True)
        self.parent.buildScene.setText("Scene Built")
        self.parent.buildScene.setDisabled(True)
        self.parent.sensor_start.setDisabled(False)

    def init_half_cylinder_surface_model(self):
        self.n_row = 10
        self.n_col = 10
        self.n_node = self.n_row * self.n_col

        self._data = data(self.n_row, self.n_col)
        self.points = np.zeros((self.n_node, 3))
        self.edges = (np.ones(
            ((self.n_col - 1) * (self.n_row - 1) * 2 + (self.n_col - 1) + (self.n_row - 1), 3)) * 2).astype(int)

        self.colors_3d = np.ones((self.n_node, 4)) * 0.5

        self.is_connected = False
        self.show_2D = False
        self.show_PC = False
        self.show_FittedMesh = False

        for i in range(self.n_col - 1):
            for j in range(self.n_row - 1):
                self.edges[2 * (i * (self.n_row - 1) + j)] = [2, i * self.n_row + j, i * self.n_row + j + 1]
                self.edges[2 * (i * (self.n_row - 1) + j) + 1] = [2, i * self.n_row + j, (i + 1) * self.n_row + j]
        for i in range(self.n_row - 1):
            self.edges[(self.n_row - 1) * (self.n_col - 1) * 2 + i] = [2, (self.n_col - 1) * self.n_row + i,
                                                                       (self.n_col - 1) * self.n_row + i + 1]
        for i in range(self.n_col - 1):
            self.edges[(self.n_row - 1) * (self.n_col - 1) * 2 + (self.n_row - 1) + i] = [2, ((i + 1) * self.n_row) - 1,
                                                                                          ((
                                                                                                       i + 1) * self.n_row) - 1 + self.n_row]
        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/half_cylinder_surface/half_cylinder.obj'
        self._2D_map = pv.read(filename)

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/half_cylinder_surface/vertex_groups.txt'
        with open(filename, 'r') as file:
            lines = file.readlines()
            numbers = [int(line.strip()) for line in lines]
            self.array_positions = []
            self.normals = np.zeros((self.n_node, 3))

            for i in range(self.n_node):
                self.array_positions.append([])
            for idx, num in enumerate(numbers):
                if num != -1:
                    self.array_positions[num].append(idx)

        self.colors = np.ones((self._2D_map.n_points, 4)) * 0.5
        for i in range(self.n_col):
            for j in range(self.n_row):
                self.colors_3d[i * self.n_row + j] = [i / self.n_col, j / self.n_row, 0, 1]
                for k in self.array_positions[i * self.n_row + j]:
                    self.colors[k] = [i / self.n_col, j / self.n_row, 0, 1]
        self.plotter.add_mesh(self._2D_map, show_edges=True, scalars=self.colors, rgb=True)
        # TreeWidgetItem(self.parent.widget_tree,"_2D_map",0,0)

        a = self._2D_map.extract_surface()
        b = a.point_normals
        self.points_origin = np.zeros((self.n_node, 3))
        for i in tqdm(range(self.n_node)):
            for j in self.array_positions[i]:
                self.normals[i] += b[j]
                self.points[i] += self._2D_map.GetPoint(j)
            self.normals[i] = self.normals[i] / len(self.array_positions[i])
            self.normals[i] = self.normals[i] / np.linalg.norm(self.normals[i])
            self.points[i] = self.points[i] / len(self.array_positions[i])
            self.points_origin[i] = self.points[i]
            self.points[i] += self.normals[i] * 0.2

        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(self.line_poly, scalars=self.colors_3d, point_size=10, line_width=3,
                                                render_points_as_spheres=True, rgb=True, name='3d')

        self.parent.sensor_choice.setDisabled(True)
        self.parent.serial_channel.setDisabled(True)
        self.parent.buildScene.setText("Scene Built")
        self.parent.buildScene.setDisabled(True)
        self.parent.sensor_start.setDisabled(False)

    def init_geneva_large(self):
        # Set grid dimensions (assumed built originally in row-major order)
        self.n_row = 5  # number of rows
        self.n_col = 16  # number of columns
        self.n_node = self.n_row * self.n_col

        # Initialize data structures (constructed in row-major order)
        self._data = data(self.n_row, self.n_col)
        self.points = np.zeros((self.n_node, 3))
        total_edges = ((self.n_row - 1) * (self.n_col - 1) * 2 +
                       (self.n_row - 1) + (self.n_col - 1))
        self.edges = (np.ones((total_edges, 3)) * 2).astype(int)
        self.colors_3d = np.ones((self.n_node, 4)) * 0.5

        self.is_connected = False
        self.show_2D = False
        self.show_PC = False
        self.show_FittedMesh = False

        # --- Build edges using row-major ordering ---
        # In row-major order, point index = row * n_col + col.
        for r in range(self.n_row - 1):
            for c in range(self.n_col - 1):
                idx = r * self.n_col + c
                base = 2 * (r * (self.n_col - 1) + c)
                # Horizontal edge: from (r, c) to (r, c+1)
                self.edges[base] = [2, idx, idx + 1]
                # Vertical edge: from (r, c) to (r+1, c)
                self.edges[base + 1] = [2, idx, idx + self.n_col]
        # Last row: horizontal edges along the bottom row.
        for c in range(self.n_col - 1):
            idx = (self.n_row - 1) * self.n_col + c
            edge_index = (self.n_row - 1) * (self.n_col - 1) * 2 + c
            self.edges[edge_index] = [2, idx, idx + 1]
        # Last column: vertical edges along the rightmost column.
        for r in range(self.n_row - 1):
            idx = r * self.n_col + (self.n_col - 1)
            edge_index = (self.n_row - 1) * (self.n_col - 1) * 2 + (self.n_col - 1) + r
            self.edges[edge_index] = [2, idx, idx + self.n_col]

        # Load the 2D mesh and signal data.
        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/geneva/large/knitting_mesh_raw.obj'
        self._2D_map = pv.read(filename)
        # Optionally scale: self._2D_map.points *= 1000

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/geneva/large/signal.txt'
        with open(filename, 'r') as file:
            lines = file.readlines()
            numbers = [int(line.strip()) for line in lines]
            self.array_positions = []
            self.normals = np.zeros((self.n_node, 3))
            for idx in range(self.n_node):
                self.array_positions.append([])
            for idx, num in enumerate(numbers):
                if num != -1:
                    self.array_positions[num].append(idx)

        # Set colors for the 2D mesh.
        self.colors = np.ones((self._2D_map.n_points, 4)) * 0.5
        # Use row-major ordering for colors.
        for r in range(self.n_row):
            for c in range(self.n_col):
                idx = r * self.n_col + c
                self.colors_3d[idx] = [c / self.n_col, r / self.n_row, 0, 1]
                for k in self.array_positions[idx]:
                    self.colors[k] = [c / self.n_col, r / self.n_row, 0, 1]
        self.plotter.add_mesh(self._2D_map, show_edges=True, scalars=self.colors, rgb=True)

        # Extract surface normals and compute average point locations.
        a = self._2D_map.extract_surface()
        b = a.point_normals
        self.points_origin = np.zeros((self.n_node, 3))
        for idx in tqdm(range(self.n_node)):
            for jdx in self.array_positions[idx]:
                self.normals[idx] += b[jdx]
                self.points[idx] += self._2D_map.GetPoint(jdx)
            self.normals[idx] = self.normals[idx] / len(self.array_positions[idx])
            self.normals[idx] = self.normals[idx] / np.linalg.norm(self.normals[idx])
            self.points[idx] = self.points[idx] / len(self.array_positions[idx])
            self.points_origin[idx] = self.points[idx]
            self.points[idx] += self.normals[idx] * 0.2

        # --- Apply reordering: convert from row-major to column-major ordering.
        # In the original row-major order, old_index = r * n_col + c.
        # We want new_index = c * n_row + r.
        mapping = np.zeros(self.n_node, dtype=int)
        for old_idx in range(self.n_node):
            r = old_idx // self.n_col  # original row
            c = old_idx % self.n_col  # original column
            new_idx = c * self.n_row + r
            mapping[old_idx] = new_idx

        # Reorder point-based arrays according to the mapping.
        new_points = np.zeros_like(self.points)
        new_normals = np.zeros_like(self.normals)
        new_points_origin = np.zeros_like(self.points_origin)
        new_array_positions = [None] * self.n_node
        new_colors_3d = np.zeros_like(self.colors_3d)
        for old_idx in range(self.n_node):
            new_idx = mapping[old_idx]
            new_points[new_idx] = self.points[old_idx]
            new_normals[new_idx] = self.normals[old_idx]
            new_points_origin[new_idx] = self.points_origin[old_idx]
            new_array_positions[new_idx] = self.array_positions[old_idx]
            new_colors_3d[new_idx] = self.colors_3d[old_idx]
        self.points = new_points
        self.normals = new_normals
        self.points_origin = new_points_origin
        self.array_positions = new_array_positions
        self.colors_3d = new_colors_3d

        # Update edges: remap each endpoint using the new mapping.
        new_edges = np.copy(self.edges)
        for k in range(new_edges.shape[0]):
            new_edges[k, 1] = mapping[int(new_edges[k, 1])]
            new_edges[k, 2] = mapping[int(new_edges[k, 2])]
        self.edges = new_edges

        # --- Visualization ---
        # Create a PolyData object for the edge mesh.
        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(self.line_poly, scalars=self.colors_3d,
                                                point_size=10, line_width=3,
                                                render_points_as_spheres=True,
                                                rgb=True, name='3d')

        # # --- Add labels for points ---
        # point_labels = [str(i) for i in range(self.n_node)]
        # self.plotter.add_point_labels(self.points, point_labels,
        #                               point_size=10, font_size=12,
        #                               text_color='black', name='point_labels')
        #
        # # --- Compute edge midpoints and add labels for edges ---
        # edge_midpoints = []
        # edge_labels = []
        # for idx, edge in enumerate(self.edges):
        #     p0 = self.points[int(edge[1])]
        #     p1 = self.points[int(edge[2])]
        #     midpoint = (p0 + p1) / 2.0
        #     edge_midpoints.append(midpoint)
        #     edge_labels.append(f"{int(edge[1])}-{int(edge[2])}")
        # edge_midpoints = np.array(edge_midpoints)
        # self.plotter.add_point_labels(edge_midpoints, edge_labels,
        #                               point_size=10, font_size=12,
        #                               text_color='blue', name='edge_labels')

        self.large_skin_activated = True

        self.parent.sensor_choice.setDisabled(True)
        self.parent.serial_channel.setDisabled(True)
        self.parent.buildScene.setText("Scene Built")
        self.parent.buildScene.setDisabled(True)
        self.parent.sensor_start.setDisabled(False)

    def init_geneva_small(self):
        # Set grid dimensions (originally in row-major order)
        self.n_row = 5  # number of rows
        self.n_col = 14  # number of columns
        self.n_node = self.n_row * self.n_col

        # Initialize data structures (constructed in row-major order)
        self._data = data(self.n_row, self.n_col)
        self.points = np.zeros((self.n_node, 3))
        total_edges = ((self.n_row - 1) * (self.n_col - 1) * 2 +
                       (self.n_row - 1) + (self.n_col - 1))
        self.edges = (np.ones((total_edges, 3)) * 2).astype(int)
        self.colors_3d = np.ones((self.n_node, 4)) * 0.5

        self.is_connected = False
        self.show_2D = False
        self.show_PC = False
        self.show_FittedMesh = False

        # Build edges from point indices using row-major order.
        # In row-major order, index = j * n_col + i.
        for j in range(self.n_row - 1):
            for i in range(self.n_col - 1):
                idx = j * self.n_col + i
                base = 2 * (j * (self.n_col - 1) + i)
                # Horizontal edge: from (j, i) to (j, i+1)
                self.edges[base] = [2, idx, idx + 1]
                # Vertical edge: from (j, i) to (j+1, i)
                self.edges[base + 1] = [2, idx, idx + self.n_col]
        # Last row: horizontal edges along the bottom row.
        for i in range(self.n_col - 1):
            idx = (self.n_row - 1) * self.n_col + i
            edge_index = (self.n_row - 1) * (self.n_col - 1) * 2 + i
            self.edges[edge_index] = [2, idx, idx + 1]
        # Last column: vertical edges along the rightmost column.
        for j in range(self.n_row - 1):
            idx = j * self.n_col + (self.n_col - 1)
            edge_index = (self.n_row - 1) * (self.n_col - 1) * 2 + (self.n_col - 1) + j
            self.edges[edge_index] = [2, idx, idx + self.n_col]

        # Load the 2D mesh and signal data.
        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/geneva/small/knitting_mesh_raw.obj'
        self._2D_map = pv.read(filename)
        # Optionally scale: self._2D_map.points *= 1000

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/geneva/small/signal.txt'
        with open(filename, 'r') as file:
            lines = file.readlines()
            numbers = [int(line.strip()) for line in lines]
            self.array_positions = []
            self.normals = np.zeros((self.n_node, 3))
            for idx in range(self.n_node):
                self.array_positions.append([])
            for idx, num in enumerate(numbers):
                if num != -1:
                    self.array_positions[num].append(idx)

        # Set colors for the 2D mesh.
        self.colors = np.ones((self._2D_map.n_points, 4)) * 0.5
        # Use row-major indexing for colors.
        for j in range(self.n_row):
            for i in range(self.n_col):
                idx = j * self.n_col + i
                self.colors_3d[idx] = [i / self.n_col, j / self.n_row, 0, 1]
                for k in self.array_positions[idx]:
                    self.colors[k] = [i / self.n_col, j / self.n_row, 0, 1]
        self.plotter.add_mesh(self._2D_map, show_edges=True, scalars=self.colors, rgb=True)

        # Extract surface normals and compute average point locations.
        a = self._2D_map.extract_surface()
        b = a.point_normals
        self.points_origin = np.zeros((self.n_node, 3))
        for idx in tqdm(range(self.n_node)):
            for jdx in self.array_positions[idx]:
                self.normals[idx] += b[jdx]
                self.points[idx] += self._2D_map.GetPoint(jdx)
            self.normals[idx] = self.normals[idx] / len(self.array_positions[idx])
            self.normals[idx] = self.normals[idx] / np.linalg.norm(self.normals[idx])
            self.points[idx] = self.points[idx] / len(self.array_positions[idx])
            self.points_origin[idx] = self.points[idx]
            self.points[idx] += self.normals[idx] * 0.2

        # --- Apply reordering: Convert from row-major to column-major ordering.
        # In the original row-major order, old_index = j * n_col + i.
        # For column-major ordering, we want new_index = i * n_row + j.
        mapping = np.zeros(self.n_node, dtype=int)
        for old_idx in range(self.n_node):
            r = old_idx // self.n_col  # original row
            c = old_idx % self.n_col  # original column
            new_idx = c * self.n_row + r
            mapping[old_idx] = new_idx

        # Reorder point-based arrays according to the mapping.
        new_points = np.zeros_like(self.points)
        new_normals = np.zeros_like(self.normals)
        new_points_origin = np.zeros_like(self.points_origin)
        new_array_positions = [None] * self.n_node
        new_colors_3d = np.zeros_like(self.colors_3d)
        for old_idx in range(self.n_node):
            new_idx = mapping[old_idx]
            new_points[new_idx] = self.points[old_idx]
            new_normals[new_idx] = self.normals[old_idx]
            new_points_origin[new_idx] = self.points_origin[old_idx]
            new_array_positions[new_idx] = self.array_positions[old_idx]
            new_colors_3d[new_idx] = self.colors_3d[old_idx]
        self.points = new_points
        self.normals = new_normals
        self.points_origin = new_points_origin
        self.array_positions = new_array_positions
        self.colors_3d = new_colors_3d

        # Update edges: remap each endpoint using the new mapping.
        new_edges = np.copy(self.edges)
        for k in range(new_edges.shape[0]):
            new_edges[k, 1] = mapping[int(new_edges[k, 1])]
            new_edges[k, 2] = mapping[int(new_edges[k, 2])]
        self.edges = new_edges

        # --- Visualization ---
        # Create a PolyData object for the edge mesh.
        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(self.line_poly, scalars=self.colors_3d,
                                                point_size=10, line_width=3,
                                                render_points_as_spheres=True,
                                                rgb=True, name='3d')

        # # --- Add labels for points ---
        # point_labels = [str(i) for i in range(self.n_node)]
        # self.plotter.add_point_labels(self.points, point_labels,
        #                               point_size=10, font_size=12,
        #                               text_color='black', name='point_labels')
        #
        # # --- Compute edge midpoints and add labels for edges ---
        # edge_midpoints = []
        # edge_labels = []
        # for idx, edge in enumerate(self.edges):
        #     p0 = self.points[int(edge[1])]
        #     p1 = self.points[int(edge[2])]
        #     midpoint = (p0 + p1) / 2.0
        #     edge_midpoints.append(midpoint)
        #     edge_labels.append(f"{int(edge[1])}-{int(edge[2])}")
        # edge_midpoints = np.array(edge_midpoints)
        # self.plotter.add_point_labels(edge_midpoints, edge_labels,
        #                               point_size=10, font_size=12,
        #                               text_color='blue', name='edge_labels')

        self.small_skin_activated = True

        # Update UI elements.
        self.parent.sensor_choice.setDisabled(True)
        self.parent.serial_channel.setDisabled(True)
        self.parent.buildScene.setText("Scene Built")
        self.parent.buildScene.setDisabled(True)
        self.parent.sensor_start.setDisabled(False)

    def init_robot(self):

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
        self._data_2 = data_2(self.n_row_2, self.n_col_2)
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

        self.parent.sensor_choice.setDisabled(True)
        self.parent.serial_channel.setDisabled(True)
        self.parent.buildScene.setText("Scene Built")
        self.parent.buildScene.setDisabled(True)
        self.parent.sensor_start.setDisabled(False)

    def startSensor(self):

        for i, ser in enumerate(self.ser_list):
            try:
                # Calibration phase:
                ser.write(b'updateCal')
                # time.sleep(1)
                ser.write(b'readCal')
                response = ser.readline().decode('utf-8').rstrip()
                data_list = [int(value) for value in response.split() if value.isdigit()]
            except Exception as e:
                print(f"Error during calibration on port {ser.port}: {e}")
                continue

            # Expected data length: n_row * (n_col + 1) + 4
            expected_length = self.n_row * (self.n_col + 1) + 4
            if len(data_list) != expected_length:
                print(f"Error on port {ser.port}: Data length is {len(data_list)}, expected {expected_length}")
                continue

            # Extract calibration data (using slicing as in your original code)
            calDataList = data_list[2:-2 - self.n_row]
            # Optionally store or print calibration data; here we assign to self.test
            self.test = calDataList
            # Process calibration data (reshape as (n_col, n_row) to match your original code)
            self._data.getCal(np.array(calDataList).reshape(self.n_col, self.n_row))

            # Clear previous data
            self._data.clearData()

            # Raw data phase:
            for j in range(self._data.windowSize):
                try:
                    ser.write(b'readRaw')
                    response = ser.readline().decode('utf-8').rstrip()
                    data_list = [int(value) for value in response.split() if value.isdigit()]
                    # Extract raw data using the same slicing logic
                    rawDataList = data_list[2:-2 - self.n_row]
                    # Process raw data: reshape to (n_col, n_row)
                    self._data.getRaw(np.array(rawDataList).reshape(self.n_col, self.n_row))
                    self._data.calDiff()
                    self._data.calDiffPer()
                    self._data.getWin(j)
                except Exception as e:
                    print(f"Error during raw data processing on port {ser.port}: {e}")
                    break

            print(f"Sensor on port {ser.port} processed successfully.")

        # Once at least one sensor has been processed, mark as connected
        self.is_connected = True
        self.is_connected_2 = True
        self.is_connected_3 = True

        self.parent.sensor_update.setDisabled(False)

    def startSensor_geneva(self):
        """
        Process sensor calibration and raw readings for each serial connection in self.ser_list.
        For sensor port at index 0, use self.n_row and self.n_col with self._data;
        for index 1, use self.n_row_2 and self.n_col_2 with self._data_2;
        for index 2, use self.n_row_3 and self.n_col_3 with self._data_3.
        """
        for i, ser in enumerate(self.ser_list):
            # Choose sensor dimensions and corresponding data object based on sensor index
            if i == 0:
                n_row = self.n_row
                n_col = self.n_col
                data_obj = self._data
            elif i == 1:
                n_row = self.n_row_2
                n_col = self.n_col_2
                data_obj = self._data_2
                continue
            elif i == 2:
                n_row = self.n_row_3
                n_col = self.n_col_3
                data_obj = self._data_3
            else:
                print(f"No sensor dimensions defined for sensor index {i}. Skipping port {ser.port}.")
                continue

            print(f"Processing sensor on port {ser.port} with dimensions (n_row: {n_row}, n_col: {n_col})")
            try:
                # Calibration phase:
                ser.write(b'updateCal')
                # time.sleep(1)
                ser.write(b'readCal')
                response = ser.readline().decode('utf-8').rstrip()
                data_list = [int(value) for value in response.split() if value.isdigit()]
            except Exception as e:
                print(f"Error during calibration on port {ser.port}: {e}")
                continue

            # Expected data length: n_row * (n_col + 1) + 4
            expected_length = n_row * (n_col + 1) + 4
            if len(data_list) != expected_length:
                print(f"Error on port {ser.port}: Data length is {len(data_list)}, expected {expected_length}")
                continue

            # Extract calibration data using slicing
            calDataList = data_list[2:-2 - n_row]
            # Optionally store calibration data for sensor index 1
            if i == 1:
                self.test_1 = calDataList

            if i == 2:
                self.test_2 = calDataList

            # Process calibration data (reshape as (n_col, n_row))
            data_obj.getCal(np.array(calDataList).reshape(n_col, n_row))

            # Clear previous data
            data_obj.clearData()

            # Raw data phase:
            for j in range(data_obj.windowSize):
                try:
                    ser.write(b'readRaw')
                    response = ser.readline().decode('utf-8').rstrip()
                    data_list = [int(value) for value in response.split() if value.isdigit()]
                    # Extract raw data using the same slicing logic
                    rawDataList = data_list[2:-2 - n_row]
                    # Process raw data: reshape to (n_col, n_row)
                    data_obj.getRaw(np.array(rawDataList).reshape(n_col, n_row))
                    data_obj.calDiff()
                    data_obj.calDiffPer()
                    data_obj.getWin(j)
                except Exception as e:
                    print(f"Error during raw data processing on port {ser.port}: {e}")
                    break

            print(f"Sensor on port {ser.port} processed successfully.")

        # Once at least one sensor has been processed, mark as connected
        self.is_connected_geneva = True
        self.parent.sensor_update.setDisabled(False)

    def update_animation(self):
        self.saveCameraPara()

        if self.is_connected:
            for i, ser in enumerate(self.ser_list):
                try:
                    ser.write(b'readRaw')
                    response = ser.readline().decode('utf-8').rstrip()
                    data_list = [int(value) for value in response.split() if value.isdigit()]
                    rawDataList = data_list[2:-2 - self.n_row]
                except Exception as e:
                    print(f"Error reading from port {ser.port}: {e}")
                    continue

                # # Apply modifications if elbow skin is active
                # if self.elbow_skin_active:
                #     elbow_indices = [0,10,20,30,40,50,60,70,80,90,100,110,120,9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119, 129]
                #     for idx in elbow_indices:
                #         if idx < len(rawDataList) and idx < len(self.test):
                #             rawDataList[idx] = self.test[idx]

                # Apply modifications if small skin is active
                if self.small_skin_activated:
                    small_skin_indices = [65, 66, 67, 68, 69]
                    for idx in small_skin_indices:
                        if idx < len(rawDataList) and idx < len(self.test):
                            rawDataList[idx] = self.test[idx]

                    diffPerDataAve_Reverse = self._data.diffPerDataAve.T.flatten()
                    transformed_data = np.where(diffPerDataAve_Reverse < -1, 1, 0)
                    if 1 in transformed_data:
                        print("Small skin detected hand approaching")

                # Process raw data: update _data with new raw values
                try:
                    self._data.getRaw(np.array(rawDataList).reshape(self.n_col, self.n_row))
                    self._data.calDiff()
                    self._data.calDiffPer()
                    self._data.getWin(self._data.windowSize)
                except Exception as e:
                    print(f"Error processing data from port {ser.port}: {e}")
                    continue

                # Update visualization for this sensor's reading
                self.update_visualization(self._data.diffPerDataAve)

                # If path tracking is active, record touched points
                if self.is_tracking_path:
                    diffPerDataAve_Reverse = self._data.diffPerDataAve.T.flatten()
                    transformed_data = np.where(diffPerDataAve_Reverse < -1, 1, 0)
                    touched_points = np.where(transformed_data == 1)[0]
                    for p in touched_points:
                        if p not in self.tracked_path:
                            self.tracked_path.append(p)

    def update_animation_geneva(self):
        self.saveCameraPara()

        if self.is_connected_geneva:
            for i, ser in enumerate(self.ser_list):
                # Select sensor dimensions and processing object based on sensor index
                if i == 0:
                    n_row = self.n_row
                    n_col = self.n_col
                    data_obj = self._data
                elif i == 1:
                    n_row = self.n_row_2
                    n_col = self.n_col_2
                    data_obj = self._data_2
                    continue
                elif i == 2:
                    n_row = self.n_row_3
                    n_col = self.n_col_3
                    data_obj = self._data_3
                else:
                    print(f"No sensor dimensions defined for sensor index {i}. Skipping port {ser.port}.")
                    continue

                try:
                    ser.write(b'readRaw')
                    response = ser.readline().decode('utf-8').rstrip()
                    data_list = [int(value) for value in response.split() if value.isdigit()]
                    # Adjust slicing based on the current sensor's n_row
                    rawDataList = data_list[2:-2 - n_row]
                except Exception as e:
                    print(f"Error reading from port {ser.port}: {e}")
                    continue

                # Apply modifications if small skin is active (only for sensor index 1 here)
                if i == 1:
                    small_skin_indices = [65, 66, 67, 68, 69]
                    for idx in small_skin_indices:
                        if idx < len(rawDataList) and idx < len(self.test_1):
                            rawDataList[idx] = self.test_1[idx]
                    diffPerDataAve_Reverse = data_obj.diffPerDataAve.T.flatten()
                    transformed_data = np.where(diffPerDataAve_Reverse < -1, 1, 0)
                    if 1 in transformed_data:
                        self.touch_counter_small_skin += 1
                        print(f"Small skin detected hand approaching: {self.touch_counter_small_skin}")
                        # self.parent.mini_robot.pause()
                        self.parent.mini_robot.stop()

                if i == 2:
                    large_skin_indices = [75, 76, 77, 78, 79]
                    for idx in large_skin_indices:
                        if idx < len(rawDataList) and idx < len(self.test_2):
                            rawDataList[idx] = self.test_2[idx]
                    diffPerDataAve_Reverse = data_obj.diffPerDataAve.T.flatten()
                    transformed_data = np.where(diffPerDataAve_Reverse < -2, 1, 0)
                    if 1 in transformed_data:
                        self.touch_counter_large_skin += 1
                        print(f"Large skin detected hand approaching: {self.touch_counter_large_skin}")
                        # self.parent.mini_robot.pause()
                        self.parent.mini_robot.stop()


                # Process raw data: update the corresponding data object with new raw values
                try:
                    data_obj.getRaw(np.array(rawDataList).reshape(n_col, n_row))
                    data_obj.calDiff()
                    data_obj.calDiffPer()
                    data_obj.getWin(data_obj.windowSize)
                except Exception as e:
                    print(f"Error processing data from port {ser.port}: {e}")
                    continue

                # Optionally update visualization (if you want per-sensor updates)
                self.update_visualization_geneva(data_obj.diffPerDataAve, i)

    def update_visualization(self, data):
        smoothing_factor = 0.05  # Adjust for smoothness
        for i in range(self.n_col):
            for j in range(self.n_row):
                idx = i * self.n_row + j
                displacement = (1.5 - abs(data[j][i])) * smoothing_factor
                # Smooth movement using normals
                target_position = self.points_origin[idx] + self.normals[idx] * displacement
                # Interpolate for smooth transitions
                self.points[idx] += (target_position - self.points[idx]) * 0.3

                # Update colors smoothly (optional)
                intensity = np.clip(1 - abs(data[j][i]) * 150 / 255, 0, 1)
                self.colors_3d[idx] = [1, intensity, intensity, 1]
                for k in self.array_positions[idx]:
                    self.colors[k] = [1, intensity, intensity, 1]

        self.line_poly.points = self.points
        self.line_poly.point_data.set_scalars(self.colors_3d)
        self.plotter.render()

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

    def updateCal(self):
        self.is_connected = False
        self.is_connected_2 = False
        self.is_connected_3 = False
        self.startSensor()

    def updateCal_geneva(self):
        self.is_connected = False
        self.is_connected_2 = False
        self.is_connected_3 = False
        self.startSensor_geneva()   # <------------

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

    def read_sensor_diff_data(self):
        diffPerDataAve_Reverse = self._data.diffPerDataAve.T.flatten()
        return diffPerDataAve_Reverse

    def read_sensor_raw_data(self):
        rawDataAve = self._data.rawDataAve.T.flatten()
        return rawDataAve

    def read_raw_all_ports(self):
        """
        Iterate through all serial connections in self.ser_list,
        send the b'readRaw' command, read the response, decode it,
        extract integer values, and print the results.
        """

        for ser in self.ser_list:
            try:
                print(f"Reading raw data from port: {ser.port}")
                ser.write(b'readRaw')
                # Read the response and process it
                response = ser.readline().decode('utf-8').rstrip()
                data_list = [int(value) for value in response.split() if value.isdigit()]
                print(f"Port {ser.port} raw data: {data_list}")
            except Exception as e:
                print(f"Error reading raw data from {ser.port}: {e}")


class RecordGesture:
    def __init__(self, my_sensor_instance):
        self.my_sensor = my_sensor_instance
        # Initialize additional attributes for recording
        self.timer_record_gesture = QTimer()
        self.timer_record_gesture.timeout.connect(self.record_gesture)
        self.is_recording = False
        self.current_gesture_data = []
        self.current_gesture_data_diff = []
        self.trial_number = None
        self.gesture_number = None
        self.start_record_pressed = False

        # For automatic noise recording
        self.auto_recording = False
        self.auto_timer = QTimer()
        self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self.auto_record_timeout)

        # New attribute to keep track of the trigger mode
        self.trigger_mode = None  # Can be 'trigger' or 'no_trigger'

    def set_trigger_mode(self, mode):
        if mode in ["trigger", "no_trigger"]:
            self.trigger_mode = mode
            print(f"Trigger mode set to '{self.trigger_mode}'.")
        else:
            print("Invalid trigger mode specified.")

    def start_record_gesture(self, gesture_number):
        if self.trigger_mode is None:
            print("Warning: Trigger mode not set. Please set the trigger mode before recording.")
            return  # Do nothing
        if gesture_number == "noise_auto":
            # Handle automatic noise recording
            if not self.auto_recording:
                # Start automatic recording
                self.auto_recording = True
                self.gesture_number = "noise"  # Use "noise" as the gesture_number
                self.trial_number = self.get_next_trial_number(self.gesture_number)
                self.current_gesture_data_diff = []
                self.start_auto_recording()
                print("Automatic noise recording started.")
            else:
                # Stop automatic recording
                self.auto_timer.stop()
                if self.is_recording:
                    self.timer_record_gesture.stop()
                    self.is_recording = False
                    self.save_gesture_data()
                    self.my_sensor.updateCal()
                    print(f"Recording stopped for gesture '{self.gesture_number}', trial {self.trial_number}. Data saved.")
                    self.trial_number += 1
                self.auto_recording = False
                print("Automatic noise recording stopped.")
        else:
            if not self.start_record_pressed:
                # Start the recording
                self.start_record_pressed = True
                # Sanitize the gesture_number
                gesture_number = self.sanitize_gesture_number(gesture_number)
                self.gesture_number = gesture_number
                self.trial_number = self.get_next_trial_number(self.gesture_number)
                self.current_gesture_data_diff = []
                self.is_recording = True  # Start recording immediately
                self.timer_record_gesture.start(0)
                print(f"Recording started for gesture '{self.gesture_number}', trial {self.trial_number}.")
            else:
                # Stop the recording
                self.timer_record_gesture.stop()
                self.start_record_pressed = False
                self.is_recording = False  # Stop recording
                self.save_gesture_data()
                self.my_sensor.updateCal()
                print(f"Recording stopped for gesture '{self.gesture_number}', trial {self.trial_number}. Data saved.")
                self.trial_number += 1  # Prepare for the next trial

    def start_auto_recording(self):
        # Generate a random duration between 5 and 10 seconds
        self.random_duration = random.randint(5, 10)
        print(f"Next recording will be for {self.random_duration} seconds.")
        # Start recording
        self.current_gesture_data_diff = []
        self.is_recording = True
        self.timer_record_gesture.start(0)  # Start the data collection timer
        # Set auto_timer to stop recording after random_duration seconds
        self.auto_timer.start(self.random_duration * 1000)  # Convert to milliseconds

    def auto_record_timeout(self):
        # Stop the current recording
        self.timer_record_gesture.stop()
        self.is_recording = False
        self.save_gesture_data()
        self.my_sensor.updateCal()
        print(f"Recording stopped for gesture '{self.gesture_number}', trial {self.trial_number}. Data saved.")
        self.trial_number += 1
        if self.auto_recording:
            # Start a new recording
            self.start_auto_recording()

    def _transform_data(self, data):
        """
        Transforms the input data based on defined conditions.
        """

        # Define conditions for each threshold
        conditions = [
            data > 2,       # Values greater than 2
            data < -1,
            data < -0.2,
            data >= -0.2   # adjust this to make sure it don't detect the rubbish signal when leave the sensor
        ]
        choices = [2, 1, 0.2, 0]
        transformed_data = np.select(conditions, choices)
        transformed_data = np.where(transformed_data == 0.0, 0, transformed_data)

        return transformed_data

    def sanitize_gesture_number(self, gesture_number):
        # Remove any characters that are not alphanumeric or underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', gesture_number)
        return sanitized

    def record_gesture(self):
        diffPerDataAve_Reverse = self.my_sensor._data.diffPerDataAve.T.flatten()

        if self.gesture_number in ["noise", "noise_auto"]:
            # For noise recording, record data continuously without transformation
            if self.is_recording:
                self.current_gesture_data_diff.append(diffPerDataAve_Reverse)
        else:
            if self.trigger_mode == "no_trigger":
                # Record data immediately without checking trigger conditions
                if self.is_recording:
                    self.current_gesture_data_diff.append(diffPerDataAve_Reverse)
            elif self.trigger_mode == "trigger":
                # Use the transformation method
                transformed_data = self._transform_data(diffPerDataAve_Reverse)

                # Check for the condition to start recording
                if not self.is_recording and np.any(transformed_data == 1):  # <----------This is the trigger condition
                    self.is_recording = True
                    self.current_gesture_data_diff = []
                    print(f"Started recording Gesture '{self.gesture_number}', Trial {self.trial_number}.")

                # Append data to the current recording if we are in a recording state
                if self.is_recording:
                    self.current_gesture_data_diff.append(diffPerDataAve_Reverse)

                # Check for the condition to stop recording
                if self.is_recording and np.all(transformed_data != 1):  # <------------This is the stop condition
                    self.is_recording = False
                    self.save_gesture_data()
                    self.my_sensor.updateCal()
                    print(f"Stopped recording Gesture '{self.gesture_number}', Trial {self.trial_number}. Data saved.")
                    self.trial_number += 1  # Prepare for the next trial
            else:
                # If trigger_mode is somehow invalid, print a warning
                print("Invalid trigger mode. Cannot record data.")
                return

    def get_next_trial_number(self, gesture_number):
        gesture_diff_dir = f"/home/ping2/ros2_ws/src/phd/phd/resource/ai/data/offset/gesture_{gesture_number}"

        if not os.path.exists(gesture_diff_dir):
            os.makedirs(gesture_diff_dir)
        files = os.listdir(gesture_diff_dir)
        return len(files) + 1

    def save_gesture_data(self):
        gesture_diff_dir = f"/home/ping2/ros2_ws/src/phd/phd/resource/ai/data/offset/gesture_{self.gesture_number}"

        filename = os.path.join(gesture_diff_dir, f"{self.trial_number}.txt")
        with open(filename, 'w') as file:
            for data_entry in self.current_gesture_data_diff:
                file.write(' '.join(map(str, data_entry)) + "\n")

        self.current_gesture_data_diff = []


class RuleBased:
    def __init__(self, ros_splitter_instance, my_sensor_instance):
        self.ros_splitter = ros_splitter_instance
        self.my_sensor = my_sensor_instance
        # Initialize attributes that were previously in MySensor
        self.reset_gesture_state()
        self.activate_rule_button = False
        self.initial_cells = []
        self.initial_cell = None
        self.four_fingers_detected = True
        self.last_large_detection_time = None
        self.found_finger_timer = None
        self.cell_cooldowns = {}
        self.saved_row = None
        self.saved_col = None
        self.saved_value = None
        self.consecutive_moves = 0
        self.movement_direction = None
        self.last_joint_1_movement = 0.0
        self.last_joint_3_movement = 0.0
        self.initial_message = ""
        # Initialize the timer
        self.timer_2 = QTimer()
        self.timer_2.timeout.connect(self.handle_timer)
        # Initialize the gesture recognition timer
        self.gesture_timer = QTimer()
        self.gesture_timer.timeout.connect(self.gesture_recognition)

    def activate_rule_based(self):
        self.activate_rule_button = not self.activate_rule_button
        if self.activate_rule_button:
            print("Activate Rule-Based")
            # Start the gesture recognition timer with an interval (e.g., 50 ms)
            self.gesture_timer.start(0)  # Adjust the interval as needed
        else:
            print("Deactivate Rule-Based")
            # Stop the gesture recognition timer
            self.gesture_timer.stop()

    def gesture_recognition(self):
        if self.activate_rule_button is True:
            diffPerDataAve_Reverse = self.my_sensor._data.diffPerDataAve.T
            detected_fingers = 0

            for i in range(self.my_sensor.n_col):
                for j in range(self.my_sensor.n_row):
                    if self.check_cooldown(i, j):
                        continue
                    current_value = diffPerDataAve_Reverse[i][j]
                    if current_value < -1:
                        detected_fingers += 1
                        if not self.timer_2.isActive() and self.four_fingers_detected:
                            self.add_initial_cell(i, j)
                            self.set_focus(i, j, current_value)
                            # print(f"No. of fingers detected: {detected_fingers}")

            if detected_fingers > 20:
                self.four_fingers_detected = False
                # print("10+ fingers are detected")
                # self.ros_splitter.robot_api.send_request("StopContinueVmode()")
                # self.ros_splitter.robot_api.send_request("StopAndClearBuffer()")
                # self.ros_splitter.robot_api.send_and_process_request([1.0, -0.49, 1.57, 0.48, 1.57, 0.0])
                self.last_large_detection_time = time.time()  # Set the timestamp when more than 20 fingers are detected

            # Check the time since last large detection
            if detected_fingers == 4:
                current_time = time.time()
                if (self.last_large_detection_time is None or
                        (current_time - self.last_large_detection_time) > 1):
                    self.four_fingers_detected = True
                    # print("4 fingers are detected")
                else:
                    print("Detection of 4 fingers ignored due to recent large detection")

    def check_cooldown(self, row, col):
        current_time = time.time()
        if (row, col) in self.cell_cooldowns and current_time - self.cell_cooldowns[(row, col)] < 0.5:
            return True
        return False

    def add_initial_cell(self, row, col):
        # Add cell to initial_cells and manage bounds
        for (r, c) in self.initial_cells:
            if abs(r - row) <= 1 and abs(c - col) <= 1:
                return  # Already within the initial touch area
        self.initial_cells.append((row, col))

    def set_focus(self, row, col, value):
        if not self.initial_cell:
            self.initial_cell = (row, col)
        self.saved_row, self.saved_col = row, col
        self.saved_value = value
        self.found_finger_timer = time.time()
        self.initial_message = f"Initial detection at ({row}, {col})."
        self.timer_2.start(0)  # This is handle_timer function

    def handle_timer(self):
        if self.found_finger_timer is None:
            print("Error: Timer fired but start time was never set.", flush=True)
            self.timer_2.stop()
            return

        current_time = time.time()
        elapsed_time = current_time - self.found_finger_timer
        current_value = self.my_sensor._data.diffPerDataAve.T[self.saved_row][self.saved_col]

        if current_value > -1:
            self.timer_2.stop()
            # print(f"\r{self.initial_message} Finger is removed.", flush=True)
            self.reset_gesture_state()
            return

        self.check_adjacent_cells()
        # print(
        #     f"\r{self.initial_message} Elapsed time since initial detection: {elapsed_time:.2f} seconds, Current value: {current_value:.2f}",
        #     flush=True, end="")

    def reset_gesture_state(self):
        self.initial_cells = []
        self.movement_direction = None
        self.consecutive_moves = 0
        self.last_joint_1_movement = 0.0
        self.last_joint_3_movement = 0.0
        # self.ros_splitter.robot_api.send_request("SuspendContinueVmode()")

    def check_adjacent_cells(self):
        # Calculate bounds of initial touch area
        min_row = min(self.initial_cells, key=lambda x: x[0])[0]
        max_row = max(self.initial_cells, key=lambda x: x[0])[0]
        min_col = min(self.initial_cells, key=lambda x: x[1])[1]
        max_col = max(self.initial_cells, key=lambda x: x[1])[1]

        # Check horizontally adjacent cells at the bounds
        for col in range(min_col - 1, max_col + 2):
            self.check_cell(min_row - 1, col, "up")  # Check row above the top
            self.check_cell(max_row + 1, col, "down")  # Check row below the bottom

        # Check vertically adjacent cells at the bounds
        for row in range(min_row, max_row + 1):
            self.check_cell(row, min_col - 1, "left")  # Check column left of the leftmost
            self.check_cell(row, max_col + 1, "right")  # Check column right of the rightmost

    def check_cell(self, row, col, direction):
        if 0 <= row < 13 and 0 <= col < 10:
            if self.check_cooldown(row, col):
                return  # Skip if the cell is cooling down
            adjacent_value = self.my_sensor._data.diffPerDataAve.T[row][col]
            if adjacent_value < -1:
                move_direction = 'right' if direction == "right" else 'left' if direction in ["left", "right"] else None
                if move_direction and (move_direction == self.movement_direction or not self.movement_direction):
                    self.consecutive_moves += 1
                    # print(
                    #     f"\r{self.consecutive_moves} {direction.capitalize()} adjacent cell at [{row}][{col}] is touched.",
                    #     flush=True)
                else:
                    self.consecutive_moves = 1
                    # print(
                    #     f"\rFirst {direction.capitalize()} adjacent cell at [{row}][{col}] is touched. Direction reset.",
                    #     flush=True)

                # self.switch_focus(row, col, adjacent_value)
                # if hasattr(self.ros_splitter, 'robot_api'):
                #     movement_factor = 0.05 * self.consecutive_moves
                #     if direction in ["left", "right"]:
                #         x_movement = -movement_factor if direction == "left" else movement_factor
                #         self.last_joint_1_movement = x_movement
                #     elif direction in ["up", "down"]:
                #         z_movement = -movement_factor if direction == "up" else movement_factor
                #         self.last_joint_3_movement = z_movement
                #     Send the combined movement command
                    # self.ros_splitter.robot_api.combined_end_effector_velocity(
                    #     [self.last_joint_1_movement, 0.0, self.last_joint_3_movement, 0.0, 0.0, 0.0])

    def switch_focus(self, row, col, value):
        self.cell_cooldowns[(self.saved_row, self.saved_col)] = time.time()
        self.saved_row, self.saved_col = row, col
        self.saved_value = value
        self.found_finger_timer = time.time()
        self.timer_2.start(0)


class LSTM:
    def __init__(self, ros_splitter_instance, my_sensor_instance):
        self.ros_splitter = ros_splitter_instance
        self.my_sensor = my_sensor_instance
        # Rest of your initialization code
        self.last_prediction = None
        self.movement_x = 0.0
        self.movement_y = 0.010
        self.movement_z = 0.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.prediction_counter = 0
        self.window_size = 50
        # self.minimum_sequence_length = 5
        self.minimum_sequence_length = 10

        self.triggerd_value = -1
        self.proximity_triggerd_value = -0.3
        self.is_recognizing_gesture = False
        self.recognition_timer = QTimer()
        self.recognition_timer.timeout.connect(self.start_gesture_recognition)
        self.current_predict_data = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.touching = False  # Add this line
        self.delay_active = False  # Flag to indicate delay state
        self.pred_2_counter = 0  # Counter for consecutive pred == 2 detections

        self.coords_list = [-170.2, -66.3, 240, 0.62, 0.78, -88.49]
        self.coords_x_counter = 0
        self.coords_y_counter = 0
        self.coords_z_counter = 0

        # Define model paths and parameters
        self.model_paths = {
            'model1': {
                # 'model_txt_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/old_chip/0_1_2_3_8_9_6_7_11/best_model_9.txt',
                # 'model_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/old_chip/0_1_2_3_8_9_6_7_11/best_model_9.pth',
                # 'model_txt_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/new_chip_for_elbow_0_1_2_3/best_model_15.txt',
                # 'model_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/new_chip_for_elbow_0_1_2_3/best_model_15.pth',
                'model_txt_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/new_chip_for_arm_0_1_2_3/best_model_38.txt',
                'model_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/new_chip_for_arm_0_1_2_3/best_model_38.pth'
            },
            'model2': {
                # 'model_txt_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/old_chip/15_16_17/best_model_21.txt',
                # 'model_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/old_chip/15_16_17/best_model_21.pth',
                'model_txt_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/new_chip_for_arm_0_1_2_3/best_model_7.txt',
                'model_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/new_chip_for_arm_0_1_2_3/best_model_7.pth'
            }
        }

        # Define transformation functions
        self.transformation_functions = {
            'model1': self.transformation_function_model1,
            'model2': self.transformation_function_model2,
        }

        self.current_model_name = 'model1'
        self.load_model(self.current_model_name)

    def load_model(self, model_name):
        try:
            paths = self.model_paths[model_name]
            model_txt_path = paths['model_txt_path']
            model_path = paths['model_path']
            parameters = self.parse_parameters_from_file(model_txt_path)
            self.hidden_dim = parameters.get('hidden_dim', 64)
            self.output_dim = parameters.get('output_dim', 4)
            print(f"Loaded {model_name} with {self.output_dim} classes.")
            self.epochs = parameters.get('epochs', 50)
            self.batch_size = parameters.get('batch_size', 16)
            self.k_folds = parameters.get('k_folds', 5)
            self.num_layers = parameters.get('num_layers', 1)
            self.learning_rate = parameters.get('learning_rate', 0.001)
            self.dropout_rate = parameters.get('dropout_rate', 0.3)
            self.input_size = parameters.get('input_size', 56)   # 56

            # Initialize the model
            self.model = GestureCNNLSTM(
                self.input_size, self.hidden_dim, self.output_dim,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.current_transformation = self.transformation_functions[model_name]
            self.current_model_name = model_name
            print(f"Model {model_name} loaded.")
        except Exception as e:
            print(f'Error loading gesture recognition model {model_name}: {e}')
            self.model = None

    def reset_movement_variables(self):
        self.movement_x = 0.0
        self.movement_y = 0.0
        self.movement_z = 0.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0

    def transformation_function_model1(self, diffPerDataAve_Reverse):
        transformed_data = np.where(diffPerDataAve_Reverse < self.triggerd_value, 1, 0)
        return transformed_data

    def transformation_function_model2(self, diffPerDataAve_Reverse):
        transformed_data = np.where(diffPerDataAve_Reverse > 2, 2,
                                    np.where(diffPerDataAve_Reverse < -1, 1,
                                             np.where(diffPerDataAve_Reverse < self.proximity_triggerd_value, 0.2, 0)))
        return transformed_data

    def toggle_gesture_recognition(self):
        """Toggle the real-time gesture recognition process."""
        if not self.is_recognizing_gesture:
            self.is_recognizing_gesture = True
            self.recognition_timer.start(0)
            print("Gesture recognition started.")
            self.current_predict_data = []  # Reset the data at the start
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.enable_end_effector_velocity_mode())
        else:
            self.recognition_timer.stop()
            self.is_recognizing_gesture = False
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_end_effector_velocity_mode())
            self.current_predict_data = []  # Reset after saving and prediction
            print("Gesture recognition stopped.")

    def toggle_model(self):
        """Toggle the currently loaded model."""
        if self.current_model_name == 'model1':
            print("Switching from model1 to model2")
            # self.load_model('model2')
            self.delay_active = True  # Activate the delay
            self.start_delay_timer('model2')
            self.current_predict_data = []
        else:
            print("Switching from model2 to model1")
            # self.load_model('model1')
            self.pred_2_counter = 0  # Reset counter
            self.delay_active = True
            self.start_delay_timer('model1')
            self.current_predict_data = []
            self.ros_splitter.robot_api.send_request(
                self.ros_splitter.robot_api.enable_end_effector_velocity_mode())

    def start_gesture_recognition(self):
        if self.delay_active:
            # Skip processing during delay
            return

        diffPerDataAve_Reverse = self.my_sensor._data.diffPerDataAve.T.flatten()
        transformed_data = self.current_transformation(diffPerDataAve_Reverse)

        touching_now = np.any(transformed_data > 0)

        if touching_now and not self.touching:
            # Touch has just started
            self.touching = True
            self.current_predict_data = []  # Reset data collected
            self.touch_start_time = time.time()  # Record touch start time if needed
            # Reset last prediction and counter on touch start
            self.last_prediction = None
        elif not touching_now and self.touching:
            # Touch has just ended
            self.touching = False
            if self.current_predict_data:
                # print("Finger lifted")

                # threading.Thread(target=self.ros_splitter.mini_robot.sync_send_coords,
                #                  args=([-170.2, -66.3, 294.9, 0.62, 0.78, -88.49], 1, 1)).start()

                # if len(self.current_predict_data) >= self.minimum_sequence_length:
                #     # For both models, make a final prediction after touch ends
                #     self.predict_gesture(self.current_predict_data)
                # else:
                #     print("Not enough data to make a final prediction.")

                # Reset movement variables
                self.reset_movement_variables()
                self.ros_splitter.robot_api.send_request(
                    self.ros_splitter.robot_api.suspend_end_effector_velocity_mode())

            # Reset variables
            self.current_predict_data = []
            self.prediction_counter = 0
            # Do not reset last_prediction here to keep track between touches
        elif touching_now:
            # Touch is continuing
            self.current_predict_data.append(transformed_data.tolist())

            # Ensure the data buffer does not exceed the window size
            if len(self.current_predict_data) > self.window_size:
                self.current_predict_data.pop(0)

            if self.current_model_name == 'model1':
                # For model1, only start predicting after minimum_sequence_length data frames collected
                if len(self.current_predict_data) >= self.minimum_sequence_length:
                    self.predict_gesture(self.current_predict_data)
            else:
                # For model2, do not predict yet during touch
                pass
        else:
            # Not touching, and touch has not just ended
            pass

    def predict_gesture(self, gesture_data):
        # Convert to numpy array
        gesture_data = np.array(gesture_data, dtype=np.float32)
        seq_length, input_size = gesture_data.shape

        # Ensure input_size matches model's expected input size
        if input_size != self.input_size:
            print(f"Input size mismatch: expected {self.input_size}, got {input_size}")
            return

        # Reshape data to match model input shape
        gesture_data = gesture_data.reshape(1, seq_length, input_size)

        # Convert to tensor
        data_tensor = torch.tensor(gesture_data, dtype=torch.float32).to(self.device)
        lengths = torch.tensor([seq_length], dtype=torch.long).to(self.device)

        # Prepare the data for the model
        packed_input = nn.utils.rnn.pack_padded_sequence(
            data_tensor, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Run the model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(packed_input)
            probabilities = torch.softmax(outputs.data, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            pred = predicted.item()
            confidence = max_prob.item()

        # Call the appropriate handler based on the current model
        if self.current_model_name == 'model1':
            self.handle_prediction_model1(pred, confidence)
        elif self.current_model_name == 'model2':
            self.handle_prediction_model2(pred, confidence)
        else:
            print(f"Unknown model {self.current_model_name}")

    def handle_prediction_model1(self, pred, confidence):
        # Set a confidence threshold
        confidence_threshold = 0.8
        if confidence < confidence_threshold:
            # print(f"Low confidence ({confidence:.2f}) for {pred}, prediction ignored.")
            return

        if pred == 8:
            if self.touching:
                print("Touching is True, ignoring pred == 8 during touch")
                return
            else:
                print("Touch has ended, handling pred == 8")
                print("Model1 Predicted Gesture (8): Switch to model2 after 1 second delay")
                self.delay_active = True  # Activate the delay
                self.start_delay_timer('model2')
                self.current_predict_data = []
                return  # Wait for delay to finish

        # For other predictions, act as before
        if self.touching:
            # During touch, check if prediction is same as last
            if pred == self.last_prediction:
                return
            else:
                self.last_prediction = pred

        # For other predictions, act as before
        # Reset movement variables
        self.reset_movement_variables()

        # Define actions for model1's gestures
        if pred == 0:
            print("Model1 Predicted Gesture (0): Left")
            self.movement_y = -0.1
            # self.new_coords_list = [self.coords_list[0], 200,
            #                         self.coords_list[2], self.coords_list[3], self.coords_list[4], self.coords_list[5]]
            # threading.Thread(target=self.ros_splitter.mini_robot.sync_send_coords,
            #                      args=(self.new_coords_list, 60, 1)).start()

        elif pred == 1:
            print("Model1 Predicted Gesture (1): Right")
            self.movement_y = 0.1
            # self.new_coords_list = [self.coords_list[0], -200,
            #                         self.coords_list[2], self.coords_list[3], self.coords_list[4], self.coords_list[5]]
            # threading.Thread(target=self.ros_splitter.mini_robot.sync_send_coords,
            #                      args=(self.new_coords_list, 60, 1)).start()
        elif pred == 2:
            print("Model1 Predicted Gesture (2): Down")
            self.movement_z = -0.1
            # self.new_coords_list = [self.coords_list[0], self.coords_list[1],
            #                         240, self.coords_list[3], self.coords_list[4], self.coords_list[5]]
            # threading.Thread(target=self.ros_splitter.mini_robot.sync_send_coords,
            #                      args=(self.new_coords_list, 60, 1)).start()
        elif pred == 3:
            print("Model1 Predicted Gesture (3): Up")
            self.movement_z = 0.1
            # self.new_coords_list = [self.coords_list[0], self.coords_list[1],
            #                         360 , self.coords_list[3], self.coords_list[4], self.coords_list[5]]
            # threading.Thread(target=self.ros_splitter.mini_robot.sync_send_coords,
            #                      args=(self.new_coords_list, 60, 1)).start()
        elif pred == 4:
            print("Model1 Predicted Gesture (4): Double Left")
            self.movement_x = -0.1
        elif pred == 5:
            print("Model1 Predicted Gesture (5): Double Right")
            self.movement_x = 0.1
        elif pred == 6:
            print("Model1 Predicted Gesture (6): Double Down")
            self.movement_z = -0.1
        elif pred == 7:
            print("Model1 Predicted Gesture (7): Double Up")
            self.movement_z = 0.1

        # Send the movement command
        self.ros_splitter.robot_api.send_request(
            self.ros_splitter.robot_api.set_end_effector_velocity([
                self.movement_x,
                self.movement_y,
                self.movement_z,
                self.rotation_x,
                self.rotation_y,
                self.rotation_z
            ])
        )

    def handle_prediction_model2(self, pred, confidence):
        # Set a confidence threshold
        confidence_threshold = 0.7
        if confidence < confidence_threshold:
            print(f"Low confidence ({confidence:.2f}) for {pred}, prediction ignored.")
            return

        # Handle pred == 2 detections
        if pred == 2:
            if not self.touching:
                # Touch has ended after pred == 2
                self.pred_2_counter += 1
                print(f"pred == 2 detected after touch end. Counter: {self.pred_2_counter}")
            else:
                # Touch is ongoing, do not increment counter
                print("Touching is True, pred == 2 detected during touch")
                return

            if self.pred_2_counter >= 2:
                print("Two consecutive pred == 2 detected after touch end. Switching back to model1")
                self.pred_2_counter = 0  # Reset counter
                self.delay_active = True
                self.start_delay_timer('model1')
                self.current_predict_data = []
                self.ros_splitter.robot_api.send_request(
                    self.ros_splitter.robot_api.enable_end_effector_velocity_mode())
                return  # Wait for delay to finish
        else:
            # If any other prediction is detected, reset the counter
            self.pred_2_counter = 0

        # Handle other gestures
        if pred == 0:
            print("Model2 Predicted Gesture (0): Swipe to Left")
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_end_effector_velocity_mode())
            self.ros_splitter.robot_api.send_and_process_request([-0.3, -0.7, 1.8, 0.5, 1.6, -1.3])
        elif pred == 1:
            print("Model2 Predicted Gesture (1): Swipe to Right")
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_end_effector_velocity_mode())
            self.ros_splitter.robot_api.send_and_process_request([1.0, 0.0, 1.57, 0.0, 1.57, 0.0])
        # ... [other gestures]

    def start_delay_timer(self, model_name):
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)  # Timer will fire only once
        self.delay_timer.timeout.connect(lambda: self.finish_delay(model_name))
        self.delay_timer.start(1000)  # Delay duration in milliseconds (1000ms = 1 second)
        print(f"Delay timer started for switching to {model_name}")

    def finish_delay(self, model_name):
        self.delay_active = False
        self.my_sensor.updateCal()
        self.load_model(model_name)
        print(f"Delay finished. Switched to {model_name}. Gesture recognition can start.")

    def parse_parameters_from_file(self, file_path):
        params = {}
        with open(file_path, 'r') as file:
            for line in file:
                if ': ' in line:
                    key, value = line.strip().split(': ')
                    params[key] = float(value) if '.' in value else int(value)
        return params






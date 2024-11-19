from networkx import is_connected
import pyvista as pv
import numpy as np
import serial
import serial.tools.list_ports
import time
import random
import os
import re
from pyvistaqt import QtInteractor, MainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox, QLabel, QSplitter
from phd.ui.ui_design import TreeWidgetItem
from tqdm import tqdm
from math import sin, cos
from phd.dependence.robot_api import RobotController
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
import torch
import pickle


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


class GestureCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, num_layers, dropout_rate):
        super(GestureCNNLSTM, self).__init__()
        # Adjust the height and width based on input_size
        self.height = 13
        self.width = 10
        if self.height * self.width != input_size:
            raise ValueError(f'Input size {input_size} does not match expected grid size {self.height}x{self.width}')
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        conv_output_size = self._get_conv_output_size()
        self.lstm = nn.LSTM(conv_output_size, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def _get_conv_output_size(self):
        dummy_input = torch.zeros(1, 1, self.height, self.width)
        output = self.conv(dummy_input)
        output_size = output.view(1, -1).size(1)
        return output_size

    def forward(self, x):
        # Unpack sequences
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        batch_size = x.size(0)
        seq_len = x.size(1)
        # Reshape to (batch_size * seq_len, 1, H, W)
        x = x.contiguous().view(-1, 1, self.height, self.width)
        x = self.conv(x)
        x = x.view(batch_size, seq_len, -1)
        # Pack sequences again
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(x)
        logits = self.classifier(hidden[-1])
        return logits


class MySensor:
    def __init__(self, parent) -> None:
        print("Initializing MySensor...")
        self.parent = parent
        self.plotter: QtInteractor = self.parent.plotter_2
        self.listModel = []
        self.listActor = []
        self.creatPlaneXY()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(0)
        self.frame_count = 0
        self.last_time = time.time()
        self.is_connected = False
        self.initChannel()
        # Initialize other classes instance
        self.lstm_class = LSTM(self.parent, self)
        self.rule_based_class = RuleBased(self.parent, self)
        self.record_gesture_class = RecordGesture(self)
        print("Finished Initializing all classes...")

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
        line = pv.Line((-50, 0, 0), (50, 0, 0))

        # 添加X轴线段，并设置为红色
        self.plotter.add_mesh(line, color='r', line_width=2, label='X Axis')
        line = pv.Line((0, -50, 0), (0, 50, 0))

        # 添加Y轴线段，并设置为绿色
        self.plotter.add_mesh(line, color='g', line_width=2, label='Y Axis')
        planeXY = pv.Plane((0, 0, 0), (0, 0, 1), 100, 100, 100, 100)

        self.actorPlaneXY = self.plotter.add_mesh(planeXY, color='gray', style='wireframe')

    def initChannel(self):
        self.com_options = []
        ports = serial.tools.list_ports.comports()
        self.ser = None
        if len(ports):
            ports = sorted(ports, key=lambda port: port.name != 'ttyACM0')
            for port in ports:
                self.com_options.append(port.name)
                self.parent.serial_channel.addItem(port.name)
                print(port.name)
        else:
            print("No serial ports found. Please connect the device and retry.")
            return

    def buildScene(self):
        if len(self.com_options) == None:
            return
        else:
            # 判断串口是否正确，还没做
            # 直接开始设置
            self.ser = serial.Serial(
                port=f'/dev/{self.parent.serial_channel.currentText()}',
                baudrate=9600,
                timeout=1
            )
            self.ser.write(b'ConnectionCheck')
            response = self.ser.readall()
            # if response == b'Connected\r\n':
            #     print("Connected")
            if self.parent.sensor_choice.currentIndex() == 0:
                self.init_jointConstruction()
            elif self.parent.sensor_choice.currentIndex() == 1:
                self.init_dualC()
            elif self.parent.sensor_choice.currentIndex() == 2:
                self.init_handConstruction()
            elif self.parent.sensor_choice.currentIndex() == 3:
                self.init_cylinder()
            # else:
            #     print("Error: The device is not connected.")
            #     self.is_connected = False
            #     self.ser.close()
            #     self.ser = None
            #     return

    def init_handConstruction(self):
        self.n_row = 10
        self.n_col = 19
        self._data = data(self.n_row, self.n_col)
        size = 1
        self.points = np.ones((190, 3)) * 3 * size
        self.edges = (np.ones(
            ((self.n_col - 1) * (self.n_row - 1) * 2 + (self.n_col - 1) + (self.n_row - 1), 3)) * 2).astype(int)
        self.colors_face = np.ones((190, 4)) * 0.5
        self.colors = np.ones((541, 4)) * 0.5
        self.is_connected = False
        self.show_2D = False
        self.show_PC = False
        self.show_FittedMesh = False

        # define the points
        for i in range(self.n_row):
            for j in range(self.n_col):
                self.points[i * (self.n_col) + j][0] = (j + 0.5) * size
                self.points[i * (self.n_col) + j][1] = (i + 0.5) * size

        # define the edges
        for i in range(self.n_row - 1):
            for j in range(self.n_col - 1):
                self.edges[2 * (i * (self.n_col - 1) + j)] = [2, i * self.n_col + j, i * self.n_col + j + 1]
                self.edges[2 * (i * (self.n_col - 1) + j) + 1] = [2, i * self.n_col + j, (i + 1) * self.n_col + j]
        for i in range(self.n_col - 1):
            self.edges[(self.n_col - 1) * (self.n_row - 1) * 2 + i] = [2, (self.n_row - 1) * self.n_col + i,
                                                                       (self.n_row - 1) * self.n_col + i + 1]
        for i in range(self.n_row - 1):
            self.edges[(self.n_col - 1) * (self.n_row - 1) * 2 + (self.n_col - 1) + i] = [2, ((i + 1) * self.n_col) - 1,
                                                                                          ((
                                                                                                       i + 1) * self.n_col) - 1 + self.n_col]

        self.ser.write(b'readCal')
        response = self.ser.readline().decode('utf-8').rstrip()
        data_list = [int(value) for value in response.split() if value.isdigit()]

        if len(data_list) != self.n_row * (self.n_col + 1) + 4:
            print("Error: The Data length is", len(data_list), ", while the required input should be",
                  int(self.n_row * (self.n_col + 1) + 4))
            return

        calDataList = data_list[2:-2 - self.n_row]
        self._data.getCal(np.array(calDataList).reshape(self.n_col, self.n_row))
        for i in range(self._data.windowSize):
            self.ser.write(b'readRaw')
            response = self.ser.readline().decode('utf-8').rstrip()
            data_list = [int(value) for value in response.split() if value.isdigit()]
            rawDataList = data_list[2:-2 - self.n_row]
            self._data.getRaw(np.array(rawDataList).reshape(self.n_col, self.n_row))
            self._data.calDiff()
            self._data.calDiffPer()
            self._data.getWin(i)
        for i in range(self.n_row):
            for j in range(self.n_col):
                self.colors_face[i * self.n_col + j] = [i * 0.1, j * 0.05, 0, 1]
                # self.points[i*self.n_col+j][2] = 3-abs(self._data.diffPerDataAve[i][j])*2
                self.colors[i * self.n_col + j] = [i * 0.1, j * 0.05, 0, 1]
        for i in range(len(self.colors)):
            if i < 190:
                continue
            else:
                self.colors[i] = (self.colors[self.edges[i - 190][1]] + self.colors[self.edges[i - 190][2]]) / 2

        # register 2D map
        self._2D_map = pv.Plane((9.5, 5, 0), (0, 0, 1), 19, 10, 19, 10)
        self.actionMap = self.plotter.add_mesh(self._2D_map, scalars=self.colors_face, show_edges=True, name='2d',
                                               rgb=True)
        # TreeWidgetItem(self.parent.widget_tree, "_2D_map", 0, 0)

        # register network
        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(self.line_poly, scalars=self.colors, point_size=10, line_width=3,
                                                render_points_as_spheres=True, rgb=True, name='3d')
        # TreeWidgetItem(self.parent.widget_tree, "_3D_mesh", 0, 0)

        # self.line_poly.points = self.points
        # self._2D_map.cell_data.set_scalars(self.colors_face)
        # self.plotter.render()

        self.parent.sensor_choice.setDisabled(True)
        self.parent.serial_channel.setDisabled(True)
        self.parent.buildScene.setText("Sence Builded")
        self.parent.buildScene.setDisabled(True)
        self.parent.sensor_start.setDisabled(False)

    def init_jointConstruction(self):
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

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/joint_1/singal.txt'
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
        # TreeWidgetItem(self.parent.widget_tree,"_3D_mesh",0,0)

        self.parent.sensor_choice.setDisabled(True)
        self.parent.serial_channel.setDisabled(True)
        self.parent.buildScene.setText("Sence Builded")
        self.parent.buildScene.setDisabled(True)
        self.parent.sensor_start.setDisabled(False)

        print("Done: Initiate the joint construction.")

    def init_dualC(self):
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

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/dualC/singal.txt'
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

    def init_cylinder(self):
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
        # TreeWidgetItem(self.parent.widget_tree,"_3D_mesh",0,0)

        self.parent.sensor_choice.setDisabled(True)
        self.parent.serial_channel.setDisabled(True)
        self.parent.buildScene.setText("Scene Built")
        self.parent.buildScene.setDisabled(True)
        self.parent.sensor_start.setDisabled(False)

    def startSensor(self):
        self.ser.write(b'updateCal')
        time.sleep(1)
        self.ser.write(b'readCal')
        response = self.ser.readline().decode('utf-8').rstrip()
        data_list = [int(value) for value in response.split() if value.isdigit()]
        if len(data_list) != self.n_row * (self.n_col + 1) + 4:
            print("Error: The Data length is", len(data_list), ", while the required input should be",
                  int(self.n_row * (self.n_col + 1) + 4))
            return
        calDataList = data_list[2:-2 - self.n_row]
        self._data.getCal(np.array(calDataList).reshape(self.n_col, self.n_row))
        self._data.clearData()
        for i in range(self._data.windowSize):
            self.ser.write(b'readRaw')
            response = self.ser.readline().decode('utf-8').rstrip()
            data_list = [int(value) for value in response.split() if value.isdigit()]
            rawDataList = data_list[2:-2 - self.n_row]
            self._data.getRaw(np.array(rawDataList).reshape(self.n_col, self.n_row))
            self._data.calDiff()
            self._data.calDiffPer()
            self._data.getWin(i)
        self.is_connected = True
        self.parent.sensor_update.setDisabled(False)

    def updateCal(self):
        self.is_connected = False
        self.startSensor()

    def update_animation(self):
        self.saveCameraPara()

        if self.is_connected:
            self.ser.write(b'readRaw')
            response = self.ser.readline().decode('utf-8').rstrip()
            data_list = [int(value) for value in response.split() if value.isdigit()]
            rawDataList = data_list[2:-2 - self.n_row]
            self._data.getRaw(np.array(rawDataList).reshape(self.n_col, self.n_row))
            self._data.calDiff()
            self._data.calDiffPer()
            self._data.getWin(self._data.windowSize)

            if self.parent.sensor_choice.currentIndex() == 0:
                for i in range(self.n_col):
                    for j in range(self.n_row):
                        self.points[i * self.n_row + j] = self.points_origin[i * self.n_row + j] + self.normals[
                            i * self.n_row + j] * (1.5 - abs(self._data.diffPerDataAve[j][i])) * 0.03
                        self.colors_3d[i * self.n_row + j] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255,
                                                              1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255, 1]
                        for k in self.array_positions[i * self.n_row + j]:
                            self.colors[k] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255,
                                              1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255, 1]

            elif self.parent.sensor_choice.currentIndex() == 1:
                self._data.diffPerDataAve = np.fliplr(self._data.diffPerDataAve)
                self._data.diffPerDataAve = np.flipud(self._data.diffPerDataAve)
                for i in range(self.n_col):
                    for j in range(self.n_row):
                        self.points[i * self.n_row + j] = self.points_origin[i * self.n_row + j] + self.normals[
                            i * self.n_row + j] * (1.5 - abs(self._data.diffPerDataAve[j][i])) * 0.2
                        self.colors_3d[i * self.n_row + j] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 200 / 255,
                                                              1 - abs(self._data.diffPerDataAve[j][i]) * 200 / 255, 1]
                        for k in self.array_positions[i * self.n_row + j]:
                            self.colors[k] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 200 / 255,
                                              1 - abs(self._data.diffPerDataAve[j][i]) * 200 / 255, 1]

            # Visualization updates
            elif self.parent.sensor_choice.currentIndex() == 2:
                for i in range(self.n_row):
                    for j in range(self.n_col):
                        self.colors_face[i * self.n_col + j] = (
                        1, 1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255,
                        1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255, 1)
                        self.points[i * self.n_col + j][2] = 3 - abs(self._data.diffPerDataAve[i][j]) * 2
                        self.colors[i * self.n_col + j] = [1,
                                                           1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255,
                                                           1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255,
                                                           1]

                for i in range(len(self.colors)):
                    if i < self.n_row * self.n_col:
                        continue
                    else:
                        self.colors[i] = (self.colors[self.edges[i - 190][1]] + self.colors[self.edges[i - 190][2]]) / 2

            elif self.parent.sensor_choice.currentIndex() == 3:
                for i in range(self.n_col):
                    for j in range(self.n_row):
                        self.points[i * self.n_row + j] = self.points_origin[i * self.n_row + j] + self.normals[
                            i * self.n_row + j] * (1.5 - abs(self._data.diffPerDataAve[j][i])) * 0.03
                        self.colors_3d[i * self.n_row + j] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255,
                                                              1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255, 1]
                        for k in self.array_positions[i * self.n_row + j]:
                            self.colors[k] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255,
                                              1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255, 1]

            self.line_poly.points = self.points
            self._2D_map.point_data.set_scalars(self.colors)
            self.plotter.render()

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
        self.triggerd_value = -1
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
        # Use the trigger value to control recording
        # transformed_data = np.where(data < self.triggerd_value, 1, 0)

        # Define conditions for each threshold
        conditions = [
            data > 2,       # Values greater than 2
            data < -1,
            data < -0.2,
            data >= -0.2    # Catch the rest (between -0.2 and 2)
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
                if not self.is_recording and np.any(transformed_data == 0.2):
                    self.is_recording = True
                    self.current_gesture_data_diff = []
                    print(f"Started recording Gesture '{self.gesture_number}', Trial {self.trial_number}.")

                # Append data to the current recording if we are in a recording state
                if self.is_recording:
                    self.current_gesture_data_diff.append(diffPerDataAve_Reverse)

                # Check for the condition to stop recording
                if self.is_recording and np.all(transformed_data == 0):
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
                self.ros_splitter.robot_api.send_request("StopContinueVmode()")
                self.ros_splitter.robot_api.send_request("StopAndClearBuffer()")
                self.ros_splitter.robot_api.send_and_process_request([1.0, -0.49, 1.57, 0.48, 1.57, 0.0])
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
        self.ros_splitter.robot_api.send_request("SuspendContinueVmode()")

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

                self.switch_focus(row, col, adjacent_value)
                if hasattr(self.ros_splitter, 'robot_api'):
                    movement_factor = 0.05 * self.consecutive_moves
                    if direction in ["left", "right"]:
                        x_movement = -movement_factor if direction == "left" else movement_factor
                        self.last_joint_1_movement = x_movement
                    elif direction in ["up", "down"]:
                        z_movement = -movement_factor if direction == "up" else movement_factor
                        self.last_joint_3_movement = z_movement
                    # Send the combined movement command
                    self.ros_splitter.robot_api.combined_end_effector_velocity(
                        [self.last_joint_1_movement, 0.0, self.last_joint_3_movement, 0.0, 0.0, 0.0])

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
        self.movement_y = 0.0
        self.movement_z = 0.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.prediction_counter = 0
        self.window_size = 50
        self.minimum_sequence_length = 5
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

        # Define model paths and parameters
        self.model_paths = {
            'model1': {
                'model_txt_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/0_1_2_3_8_9_6_7_11/best_model_9.txt',
                'model_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/0_1_2_3_8_9_6_7_11/best_model_9.pth'
            },
            'model2': {
                'model_txt_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/15_16_17/best_model_21.txt',
                'model_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/15_16_17/best_model_21.pth'
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
            self.hidden_dim = parameters.get('hidden_dim', 80)
            self.output_dim = parameters.get('output_dim', 8)
            print(f"Loaded {model_name} with {self.output_dim} classes.")
            self.epochs = parameters.get('epochs', 100)
            self.batch_size = parameters.get('batch_size', 32)
            self.k_folds = parameters.get('k_folds', 5)
            self.num_layers = parameters.get('num_layers', 1)
            self.learning_rate = parameters.get('learning_rate', 0.01)
            self.dropout_rate = parameters.get('dropout_rate', 0.5)
            self.input_size = parameters.get('input_size', 130)

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
                print("Finger lifted")

                if len(self.current_predict_data) >= self.minimum_sequence_length:
                    # For both models, make a final prediction after touch ends
                    self.predict_gesture(self.current_predict_data)
                else:
                    print("Not enough data to make a final prediction.")

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
        confidence_threshold = 0.7
        if confidence < confidence_threshold:
            print(f"Low confidence ({confidence:.2f}) for {pred}, prediction ignored.")
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
        elif pred == 1:
            print("Model1 Predicted Gesture (1): Right")
            self.movement_y = 0.1
        elif pred == 2:
            print("Model1 Predicted Gesture (2): Down")
            self.movement_z = -0.1
        elif pred == 3:
            print("Model1 Predicted Gesture (3): Up")
            self.movement_z = 0.1
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


class Denoise:
    pass


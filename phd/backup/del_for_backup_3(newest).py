from networkx import is_connected
import pyvista as pv
import numpy as np
import serial
import serial.tools.list_ports
import time
import os
from pyvistaqt import QtInteractor, MainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox, QLabel, QSplitter
from phd.ui.ui_design import TreeWidgetItem
from tqdm import tqdm
from math import sin, cos
from phd.dependence.robot_api import RobotController


# from phd.dependence.lstm_model import LSTMModel


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


class MySensor():
    def __init__(self, parent) -> None:
        self.parent = parent
        self.plotter: QtInteractor = self.parent.plotter_2
        self.listModel = []
        self.listActor = []
        self.creatPlaneXY()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(0)
        self.timer_2 = QTimer()
        self.timer_2.timeout.connect(self.handle_timer)
        self.timer_2.start(0)
        self.frame_count = 0
        self.last_time = time.time()
        self.is_connected = False
        self.initChannel()
        self.startSensor_pressed = False
        self.found_finger_timer = None
        self.cell_cooldowns = {}  # Stores cooldown times for cells
        self.counter = 0
        self.saved_row = None
        self.saved_col = None
        self.saved_value = None
        self.initial_message = ""
        self.reset_gesture_state()
        self.initial_cell = None
        self.four_fingers_detected = True
        self.last_large_detection_time = None  # Timestamp for last large detection (>20 fingers)
        self.timer_experiment = QTimer()
        self.timer_experiment.timeout.connect(self.experiment)
        self.flag = True
        self.timer_experiment_2 = QTimer()
        self.timer_experiment_2.timeout.connect(self.experiment_2)
        self.flag_2 = True
        # Initialize additional attributes for recording
        self.is_recording = False
        self.current_gesture_data = []
        self.gesture_number = None

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
            for port in ports:
                self.com_options.append(port.name)
                self.parent.serial_channel.addItem(port.name)
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
            if response == b'Connected\r\n':
                if self.parent.sensor_choice.currentIndex() == 0:
                    self.init_handConstruction()
                elif self.parent.sensor_choice.currentIndex() == 1:
                    self.init_jointConstruction()
                elif self.parent.sensor_choice.currentIndex() == 2:
                    self.init_dualC()
            # elif self.parent.sensor_choice.currentIndex() == 3:
            #     self.cylinder()
            else:
                self.is_connected = False
                self.ser.close()
                self.ser = None
                return

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
        TreeWidgetItem(self.parent.widget_tree, "_2D_map", 0, 0)

        # register network
        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(self.line_poly, scalars=self.colors, point_size=10, line_width=3,
                                                render_points_as_spheres=True, rgb=True, name='3d')
        TreeWidgetItem(self.parent.widget_tree, "_3D_mesh", 0, 0)

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
        self.startSensor_pressed = True

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

            # Visualization updates
            if self.parent.sensor_choice.currentIndex() == 0:
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

            elif self.parent.sensor_choice.currentIndex() == 1:
                for i in range(self.n_col):
                    for j in range(self.n_row):
                        self.points[i * self.n_row + j] = self.points_origin[i * self.n_row + j] + self.normals[
                            i * self.n_row + j] * (1.5 - abs(self._data.diffPerDataAve[j][i])) * 0.03
                        self.colors_3d[i * self.n_row + j] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255,
                                                              1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255, 1]
                        for k in self.array_positions[i * self.n_row + j]:
                            self.colors[k] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255,
                                              1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255, 1]

            elif self.parent.sensor_choice.currentIndex() == 2:
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

            self.line_poly.points = self.points
            self._2D_map.point_data.set_scalars(self.colors)
            self.plotter.render()

        # 计算帧率
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_time >= 0.1:
            fps = self.frame_count / (current_time - self.last_time)
            # self.message(f"FPS: {fps:.2f}")
            self.last_time = current_time
            self.frame_count = 0

        # # My function to get the gesture
        if self.startSensor_pressed is True:
            # self.gesture_recognition()

            diffPerDataAve_Reverse = self._data.diffPerDataAve.T

            # Start or stop recording based on data conditions
            if not self.is_recording and np.any(diffPerDataAve_Reverse < -1):
                self.is_recording = True
                self.current_gesture_data = []  # Start fresh for new recording
                print(f"Started recording gesture {self.gesture_number}.")

            elif self.is_recording and np.all(diffPerDataAve_Reverse > -1):
                self.is_recording = False
                self.save_gesture_data()
                print(f"Stopped recording gesture {self.gesture_number} and data saved.")

            if self.is_recording:
                # Append current data frame to the recording list
                self.current_gesture_data.append(diffPerDataAve_Reverse.flatten())

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

    def gesture_recognition(self):
        diffPerDataAve_Reverse = self._data.diffPerDataAve.T
        detected_fingers = 0

        for i in range(self.n_col):
            for j in range(self.n_row):
                if self.check_cooldown(i, j):
                    continue
                current_value = diffPerDataAve_Reverse[i][j]
                if current_value < -1:
                    detected_fingers += 1
                    if not self.timer_2.isActive() and self.four_fingers_detected:
                        self.add_initial_cell(i, j)
                        self.set_focus(i, j, current_value)
                        print(f"No. of fingers detected: {detected_fingers}")

        if detected_fingers > 20:
            self.four_fingers_detected = False
            print("10+ fingers are detected")
            self.parent.robot_api.send_request("StopContinueVmode()")
            self.parent.robot_api.send_request("StopAndClearBuffer()")
            self.parent.robot_api.send_and_process_request([1.0, -0.49, 1.57, 0.48, 1.57, 0.0])
            self.last_large_detection_time = time.time()  # Set the timestamp when more than 20 fingers are detected

        # Check the time since last large detection
        if detected_fingers == 4:
            current_time = time.time()
            if (self.last_large_detection_time is None or
                    (current_time - self.last_large_detection_time) > 1):
                self.four_fingers_detected = True
                print("4 fingers are detected")
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
        current_value = self._data.diffPerDataAve.T[self.saved_row][self.saved_col]

        if current_value > -1:
            self.timer_2.stop()
            # print(f"\r{self.initial_message} Finger is removed.", flush=True)
            self.reset_gesture_state()
            return

        self.check_adjacent_cells()
        print(
            f"\r{self.initial_message} Elapsed time since initial detection: {elapsed_time:.2f} seconds, Current value: {current_value:.2f}",
            flush=True, end="")

    def reset_gesture_state(self):
        self.initial_cells = []
        self.movement_direction = None
        self.consecutive_moves = 0
        self.last_joint_1_movement = 0.0
        self.last_joint_3_movement = 0.0
        self.parent.robot_api.send_request("SuspendContinueVmode()")

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
            adjacent_value = self._data.diffPerDataAve.T[row][col]
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
                if hasattr(self.parent, 'robot_api'):
                    movement_factor = 0.05 * self.consecutive_moves
                    if direction in ["left", "right"]:
                        x_movement = -movement_factor if direction == "left" else movement_factor
                        self.last_joint_1_movement = x_movement
                    elif direction in ["up", "down"]:
                        z_movement = -movement_factor if direction == "up" else movement_factor
                        self.last_joint_3_movement = z_movement
                    # Send the combined movement command
                    self.parent.robot_api.combined_end_effector_velocity(
                        [self.last_joint_1_movement, 0.0, self.last_joint_3_movement, 0.0, 0.0, 0.0])

    def switch_focus(self, row, col, value):
        self.cell_cooldowns[(self.saved_row, self.saved_col)] = time.time()
        self.saved_row, self.saved_col = row, col
        self.saved_value = value
        self.found_finger_timer = time.time()
        self.timer_2.start(0)

    def experiment(self):
        if self.flag is True:
            self.flag = False
            self.parent.robot_api.send_positions_tool_position([-0.300, 0.750, 0.203], [0.0, 1.0, 0.0, 0.0])

        self.timer_experiment.start(20)
        # print(self._data.diffPerDataAve)
        # print(self._data.rawDataAve)
        self.counter += 1
        print(self.counter)

        diffPerDataAve_string = ' '.join(map(str, self._data.diffPerDataAve.flatten()))
        rawDataAve_string = ' '.join(map(str, self._data.rawDataAve.flatten()))

        with open("/home/ping2/ros2_ws/src/phd/phd/resource/singa_request/diffPerDataAve_output.txt", "a") as file:
            file.write(diffPerDataAve_string + "\n")  # Each new entry on a new line, but data remains on one line

        with open("/home/ping2/ros2_ws/src/phd/phd/resource/singa_request/rawDataAve_output.txt", "a") as file:
            file.write(rawDataAve_string + "\n")  # Each new entry on a new line, but data remains on one line

        if self.counter == 199:
            self.timer_experiment.stop()
            self.parent.robot_api.send_positions_tool_position([-0.300, 0.750, 0.25], [0.0, 1.0, 0.0, 0.0])
            self.flag = True
            self.counter = 0

    def experiment_2(self):
        if self.flag_2 is True:
            self.flag_2 = False
            self.parent.robot_api.send_positions_tool_position([-0.285, 0.700, 0.207], [0.0, 1.0, 0.0, 0.0])

        self.timer_experiment_2.start(20)
        # print(self._data.diffPerDataAve)
        # print(self._data.rawDataAve)
        self.counter += 1
        print(self.counter)

        diffPerDataAve_string = ' '.join(map(str, self._data.diffPerDataAve.flatten()))
        rawDataAve_string = ' '.join(map(str, self._data.rawDataAve.flatten()))

        with open("/home/ping2/ros2_ws/src/phd/phd/resource/singa_request/dualC/diffPerDataAve_output.txt",
                  "a") as file:
            file.write(diffPerDataAve_string + "\n")  # Each new entry on a new line, but data remains on one line

        with open("/home/ping2/ros2_ws/src/phd/phd/resource/singa_request/dualC/rawDataAve_output.txt", "a") as file:
            file.write(rawDataAve_string + "\n")  # Each new entry on a new line, but data remains on one line

        if self.counter == 199:
            self.timer_experiment_2.stop()
            self.parent.robot_api.send_positions_tool_position([-0.285, 0.700, 0.254], [0.0, 1.0, 0.0, 0.0])
            self.flag_2 = True
            self.counter = 0

    def record_gesture(self, gesture_number):
        if not self.is_recording:
            # Start recording
            self.gesture_number = gesture_number
            self.trial_number = self.get_last_trial_number(gesture_number) + 1
            self.current_gesture_data = []  # Initialize the data list for a new recording
            print(f"Started recording Gesture {gesture_number}, Trial {self.trial_number}.")
            self.is_recording = True  # Explicitly start recording
        else:
            # Stop recording
            self.save_gesture_data()
            print(f"Stopped recording Gesture {gesture_number}, Trial {self.trial_number}. Data saved.")
            self.trial_number += 1  # Increment the trial number only after stopping
            self.is_recording = False  # Explicitly stop recording

    def get_last_trial_number(self, gesture_number):
        filename = f"/home/ping2/ros2_ws/src/phd/phd/resource/ai/gesture_{gesture_number}_data.txt"
        try:
            with open(filename, "r") as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1]
                    last_trial_number = int(last_line.split()[0])
                    print(f"Read last trial number: {last_trial_number} from {filename}")
                    return last_trial_number
        except FileNotFoundError:
            print(f"File {filename} not found. Starting from trial number 0.")
            return 0
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
        return 0

    def save_gesture_data(self):
        filename = f"/home/ping2/ros2_ws/src/phd/phd/resource/ai/gesture_{self.gesture_number}_data.txt"
        with open(filename, "a") as file:
            for data_entry in self.current_gesture_data:
                file.write(f"{self.trial_number} " + ' '.join(map(str, data_entry)) + "\n")
        print(f"Data for gesture {self.gesture_number} trial {self.trial_number} has been saved to {filename}.")



















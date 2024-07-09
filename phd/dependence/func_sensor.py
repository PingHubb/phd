from networkx import is_connected
import pyvista as pv
import numpy as np
import serial
import serial.tools.list_ports
import time
import os
from pyvistaqt import QtInteractor, MainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox, QLabel,QSplitter
from phd.ui.ui_design import TreeWidgetItem
from tqdm import tqdm
from math import sin, cos


class data:
    def __init__(self, n_row, n_col):
        self.n_row = n_row
        self.n_col = n_col
        self.windowSize = 10
        self.calData = np.zeros((self.n_row, self.n_col))
        self.rawData = np.zeros((self.n_row, self.n_col))
        self.diffData = np.zeros((self.n_row, self.n_col))
        self.diffPerData = np.zeros((self.n_row, self.n_col))
        self.rawDataWin = np.zeros((self.windowSize,self.n_row, self.n_col))
        self.diffDataWin = np.zeros((self.windowSize,self.n_row, self.n_col))
        self.diffPerDataWin = np.zeros((self.windowSize,self.n_row, self.n_col))
        self.rawDataAve = np.zeros((self.n_row, self.n_col))
        self.diffDataAve = np.zeros((self.n_row, self.n_col))
        self.diffPerDataAve = np.zeros((self.n_row, self.n_col))

    def getRaw(self,rawData):
        self.rawData = rawData.T

    def getCal(self,calData):
        self.calData = calData.T

    def calDiff(self):
        self.diffData = self.rawData - self.calData

    def calDiffPer(self):
        self.diffPerData = 100 * self.diffData / self.calData

    def clearData(self):
        self.rawDataWin = np.zeros((self.windowSize,self.n_row, self.n_col))
        self.diffDataWin = np.zeros((self.windowSize,self.n_row, self.n_col))
        self.diffPerDataWin = np.zeros((self.windowSize,self.n_row, self.n_col))
        self.rawDataAve = np.zeros((self.n_row, self.n_col))
        self.diffDataAve = np.zeros((self.n_row, self.n_col))
        self.diffPerDataAve = np.zeros((self.n_row, self.n_col))

    def getWin(self,i):
        if i == self.windowSize:
            self.rawDataWin[:-1] = self.rawDataWin[1:]
            self.diffDataWin[:-1] = self.diffDataWin[1:]
            self.diffPerDataWin[:-1] = self.diffPerDataWin[1:]
        self.rawDataWin[i-1] = self.rawData
        self.diffDataWin[i-1] = self.diffData
        self.diffPerDataWin[i-1] = self.diffPerData
        if i == self.windowSize:
            self.rawDataAve = np.mean(self.rawDataWin, axis=0)
            self.diffDataAve = np.mean(self.diffDataWin, axis=0)
            self.diffPerDataAve = np.mean(self.diffPerDataWin, axis=0)
            self.rawDataAve = np.flipud( self.rawDataAve)            
            self.diffDataAve = np.flipud( self.diffDataAve)
            self.diffPerDataAve = np.flipud( self.diffPerDataAve)


class MySensor():
    def __init__(self, parent) -> None:
        self.parent = parent
        self.plotter: QtInteractor = self.parent.plotter
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
        self.startSensor_pressed = False

    def saveCameraPara(self):
        self.camera_pos = self.plotter.camera.position
        self.camera_focal = self.plotter.camera.focal_point
        self.camera_view_angle = self.plotter.camera.view_angle
    
    def loadCameraPare(self):
        self.plotter.camera.position = self.camera_pos
        self.plotter.camera.focal_point = self.camera_focal
        self.plotter.camera.view_angle = self.camera_view_angle

    def loadCameraPare(self,camera_pos,camera_focal,camera_view_angle):
        self.plotter.camera.position = camera_pos
        self.plotter.camera.focal_point = camera_focal
        self.plotter.camera.view_angle = camera_view_angle

    def creatPlaneXY(self):
        self.plotter.camera.position = (1,-1,1)
        self.saveCameraPara()
        line = pv.Line((-50, 0, 0), (50, 0,0 ))
        
        # 添加X轴线段，并设置为红色
        self.plotter.add_mesh(line, color='r', line_width=2, label='X Axis')
        line = pv.Line((0, -50, 0), (0,50, 0))

        # 添加Y轴线段，并设置为绿色
        self.plotter.add_mesh(line, color='g', line_width=2, label='Y Axis')
        planeXY = pv.Plane((0,0,0),(0,0,1),100,100,100,100)

        self.actorPlaneXY = self.plotter.add_mesh(planeXY, color='gray',style='wireframe')

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
            else:
                self.is_connected = False
                self.ser.close()
                self.ser = None
                return

    def init_handConstruction(self):
        self.n_row = 10
        self.n_col = 19
        self._data = data(self.n_row,self.n_col)
        size = 1
        self.points = np.ones((190,3)) * 3 * size
        self.edges = (np.ones(((self.n_col-1)*(self.n_row-1)*2+(self.n_col-1)+(self.n_row-1),3)) * 2).astype(int)
        self.colors_face = np.ones((190,4))*0.5
        self.colors = np.ones((541,4))*0.5
        self.is_connected = False
        self.show_2D = False
        self.show_PC = False
        self.show_FittedMesh = False

        # define the points
        for i in range(self.n_row):
            for j in range(self.n_col):
                self.points[i*(self.n_col)+j][0] = (j + 0.5) * size
                self.points[i*(self.n_col)+j][1] = (i + 0.5) * size

        # define the edges
        for i in range(self.n_row-1):
            for j in range(self.n_col-1):
                self.edges[2 * (i * (self.n_col - 1) + j)] = [2,i * self.n_col + j, i * self.n_col + j + 1]
                self.edges[2 * (i * (self.n_col - 1) + j) + 1] = [2,i * self.n_col + j, (i + 1) * self.n_col + j]
        for i in range(self.n_col - 1):
            self.edges[(self.n_col - 1) * (self.n_row - 1) * 2 + i] = [2, (self.n_row - 1) * self.n_col + i,(self.n_row - 1) * self.n_col + i + 1]
        for i in range(self.n_row - 1):
            self.edges[(self.n_col - 1) * (self.n_row - 1) * 2 + (self.n_col - 1) + i] = [2, ((i + 1) * self.n_col) - 1,((i + 1) * self.n_col) - 1 + self.n_col]

        self.ser.write(b'readCal')
        response = self.ser.readline().decode('utf-8').rstrip()
        data_list = [int(value) for value in response.split() if value.isdigit()]

        if len(data_list) !=self.n_row*(self.n_col+1) + 4:
            print("Error: The Data length is", len(data_list), ", while the required input should be", int(self.n_row*(self.n_col+1)+4))
            return

        calDataList = data_list[2:-2-self.n_row]
        self._data.getCal(np.array(calDataList).reshape(self.n_col,self.n_row))
        for i in range(self._data.windowSize):
            self.ser.write(b'readRaw')
            response = self.ser.readline().decode('utf-8').rstrip()
            data_list = [int(value) for value in response.split() if value.isdigit()]
            rawDataList = data_list[2:-2-self.n_row]
            self._data.getRaw(np.array(rawDataList).reshape(self.n_col,self.n_row))
            self._data.calDiff()
            self._data.calDiffPer()
            self._data.getWin(i)
        for i in range(self.n_row):
            for j in range(self.n_col):
                self.colors_face[i*self.n_col+j] = [i*0.1,j*0.05,0,1]
                # self.points[i*self.n_col+j][2] = 3-abs(self._data.diffPerDataAve[i][j])*2
                self.colors[i*self.n_col+j] = [i*0.1,j*0.05,0,1]
        for i in range(len(self.colors)):
            if i < 190:
                continue
            else:
                self.colors[i] = (self.colors[self.edges[i-190][1]]+self.colors[self.edges[i-190][2]])/2

        # register 2D map
        self._2D_map = pv.Plane((9.5,5,0),(0,0,1),19,10,19,10)
        self.actionMap = self.plotter.add_mesh(self._2D_map,scalars=self.colors_face, show_edges=True,name = '2d',rgb = True)
        TreeWidgetItem(self.parent.widget_tree,"_2D_map",0,0)

        # register network
        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(self.line_poly,scalars=self.colors, point_size=10,line_width = 3,render_points_as_spheres=True,rgb = True,name = '3d')
        TreeWidgetItem(self.parent.widget_tree,"_3D_mesh",0,0)
        
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

        self._data = data(self.n_row,self.n_col)
        self.points = np.zeros((self.n_node,3))
        self.edges = (np.ones(((self.n_col-1)*(self.n_row-1)*2+(self.n_col-1)+(self.n_row-1),3)) * 2).astype(int)

        self.colors_3d = np.ones((self.n_node,4))*0.5

        self.is_connected = False
        self.show_2D = False
        self.show_PC = False
        self.show_FittedMesh = False
            
        for i in range(self.n_col-1):
            for j in range(self.n_row-1):
                self.edges[2 * (i * (self.n_row - 1) + j)] = [2,i * self.n_row + j, i * self.n_row + j + 1]
                self.edges[2 * (i * (self.n_row - 1) + j) + 1] = [2,i * self.n_row + j, (i + 1) * self.n_row + j]
        for i in range(self.n_row - 1):
            self.edges[(self.n_row - 1) * (self.n_col - 1) * 2 + i] = [2, (self.n_col - 1) * self.n_row + i,(self.n_col - 1) * self.n_row + i + 1]
        for i in range(self.n_col - 1):
            self.edges[(self.n_row - 1) * (self.n_col - 1) * 2 + (self.n_row - 1) + i] = [2, ((i + 1) * self.n_row) - 1,((i + 1) * self.n_row) - 1 + self.n_row]

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/joint_1/mesh.obj'
        self._2D_map = pv.read(filename)
        
        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/joint_1/singal.txt'
        with open(filename,'r') as file:
            lines = file.readlines()
            numbers = [int(line.strip()) for line in lines]
            self.array_positions = []
            self.normals = np.zeros((self.n_node,3))

            for i in range(156):
                self.array_positions.append([])
            for idx, num in enumerate(numbers):
                if num != -1:
                    self.array_positions[num].append(idx)

        for i in range((self.n_col)-1,-1,-1):
            del self.array_positions[i*12+11]
            del self.array_positions[i*12]
        

        self.colors = np.ones((self._2D_map.n_points,4))*0.5
        for i in range(self.n_col):
            for j in range(self.n_row):
                self.colors_3d[i*self.n_row+j] = [i/self.n_col,j/self.n_row,0,1]
                for k in self.array_positions[i*self.n_row+j]:
                    self.colors[k] = [i/self.n_col,j/self.n_row,0,1]
        self.plotter.add_mesh(self._2D_map,show_edges = True,scalars=self.colors,rgb = True)
        # TreeWidgetItem(self.parent.widget_tree,"_2D_map",0,0)

        a = self._2D_map.extract_surface()
        b = a.point_normals
        self.points_origin = np.zeros((self.n_node,3))
        for i in tqdm(range(self.n_node)):
            for j in self.array_positions[i]:
                self.normals[i] += b[j]
                self.points[i] += self._2D_map.GetPoint(j)
            self.normals[i] = self.normals[i]/len(self.array_positions[i])
            self.normals[i] = self.normals[i]/np.linalg.norm(self.normals[i])
            self.points[i] = self.points[i]/len(self.array_positions[i])
            self.points_origin[i] = self.points[i]
            self.points[i] += self.normals[i]*0.02

        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(self.line_poly,scalars=self.colors_3d, point_size=10,line_width = 3,render_points_as_spheres=True,rgb = True,name = '3d')
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

        self._data = data(self.n_row,self.n_col)
        self.points = np.zeros((self.n_node,3))
        self.edges = (np.ones(((self.n_col-1)*(self.n_row-1)*2+(self.n_col-1)+(self.n_row-1),3)) * 2).astype(int)

        self.colors_3d = np.ones((self.n_node,4))*0.5

        self.is_connected = False
        self.show_2D = False
        self.show_PC = False
        self.show_FittedMesh = False

        for i in range(self.n_col-1):
            for j in range(self.n_row-1):
                self.edges[2 * (i * (self.n_row - 1) + j)] = [2,i * self.n_row + j, i * self.n_row + j + 1]
                self.edges[2 * (i * (self.n_row - 1) + j) + 1] = [2,i * self.n_row + j, (i + 1) * self.n_row + j]
        for i in range(self.n_row - 1):
            self.edges[(self.n_row - 1) * (self.n_col - 1) * 2 + i] = [2, (self.n_col - 1) * self.n_row + i,(self.n_col - 1) * self.n_row + i + 1]
        for i in range(self.n_col - 1):
            self.edges[(self.n_row - 1) * (self.n_col - 1) * 2 + (self.n_row - 1) + i] = [2, ((i + 1) * self.n_row) - 1,((i + 1) * self.n_row) - 1 + self.n_row]

        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/dualC/mesh.obj'
        self._2D_map = pv.read(filename)
        
        filename = '/home/ping2/ros2_ws/src/phd/phd/resource/sensor/dualC/singal.txt'
        with open(filename, 'r') as file:
            lines = file.readlines()
            numbers = [int(line.strip()) for line in lines]
            self.array_positions = []
            self.normals = np.zeros((self.n_node,3))

            for i in range(self.n_node):
                self.array_positions.append([])
            for idx, num in enumerate(numbers):
                if num != -1:
                    self.array_positions[num].append(idx)

        self.colors = np.ones((self._2D_map.n_points,4))*0.5
        for i in range(self.n_col):
            for j in range(self.n_row):
                self.colors_3d[i*self.n_row+j] = [i/self.n_col,j/self.n_row,0,1]
                for k in self.array_positions[i*self.n_row+j]:
                    self.colors[k] = [i/self.n_col,j/self.n_row,0,1]
        self.plotter.add_mesh(self._2D_map,show_edges = True,scalars=self.colors,rgb = True)
        TreeWidgetItem(self.parent.widget_tree,"_2D_map",0,0)

        a = self._2D_map.extract_surface()
        b = a.point_normals
        self.points_origin = np.zeros((self.n_node,3))
        for i in tqdm(range(self.n_node)):
            for j in self.array_positions[i]:
                self.normals[i] += b[j]
                self.points[i] += self._2D_map.GetPoint(j)
            self.normals[i] = self.normals[i]/len(self.array_positions[i])
            self.normals[i] = self.normals[i]/np.linalg.norm(self.normals[i])
            self.points[i] = self.points[i]/len(self.array_positions[i])
            self.points_origin[i] = self.points[i]
            self.points[i] += self.normals[i]*0.2

        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMesh = self.plotter.add_mesh(self.line_poly,scalars=self.colors_3d, point_size=10,line_width = 3,render_points_as_spheres=True,rgb = True,name = '3d')
        TreeWidgetItem(self.parent.widget_tree,"_3D_mesh",0,0)

        
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
        if len(data_list) !=self.n_row*(self.n_col+1) + 4:
            print("Error: The Data length is", len(data_list), ", while the required input should be", int(self.n_row*(self.n_col+1)+4))
            return
        calDataList = data_list[2:-2-self.n_row]
        self._data.getCal(np.array(calDataList).reshape(self.n_col,self.n_row))
        self._data.clearData()
        for i in range(self._data.windowSize):
            self.ser.write(b'readRaw')
            response = self.ser.readline().decode('utf-8').rstrip()
            data_list = [int(value) for value in response.split() if value.isdigit()]
            rawDataList = data_list[2:-2-self.n_row]
            self._data.getRaw(np.array(rawDataList).reshape(self.n_col,self.n_row))
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
            rawDataList = data_list[2:-2-self.n_row]
            self._data.getRaw(np.array(rawDataList).reshape(self.n_col,self.n_row))
            self._data.calDiff()
            self._data.calDiffPer()
            self._data.getWin(self._data.windowSize)
            if self.parent.sensor_choice.currentIndex() == 0:
                for i in range(self.n_row):
                    for j in range(self.n_col):
                        self.colors_face[i * self.n_col + j] = (1, 1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255, 1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255,1)
                        self.points[i*self.n_col+j][2] = 3-abs(self._data.diffPerDataAve[i][j])*2
                        self.colors[i*self.n_col+j] = [1, 1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255, 1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255,1]
                for i in range(len(self.colors)):
                    if i < self.n_row*self.n_col:
                        continue
                    else:
                        self.colors[i] = (self.colors[self.edges[i-190][1]]+self.colors[self.edges[i-190][2]])/2
            elif self.parent.sensor_choice.currentIndex() == 1:
                for i in range(self.n_col):
                    for j in range(self.n_row):
                        self.points[i * self.n_row+j] = self.points_origin[i * self.n_row+j] + self.normals[i * self.n_row+j] * (1.5 - abs(self._data.diffPerDataAve[j][i])) * 0.03
                        self.colors_3d[i*self.n_row+j] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255, 1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255,1]
                        for k in self.array_positions[i*self.n_row+j]:
                            self.colors[k] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255, 1 - abs(self._data.diffPerDataAve[j][i]) * 150 / 255,1]
            elif self.parent.sensor_choice.currentIndex() == 2:
                self._data.diffPerDataAve = np.fliplr(self._data.diffPerDataAve)
                self._data.diffPerDataAve = np.flipud(self._data.diffPerDataAve)
                for i in range(self.n_col):
                    for j in range(self.n_row):
                        self.points[i * self.n_row+j] = self.points_origin[i * self.n_row+j] + self.normals[i * self.n_row+j] * (1.5 - abs(self._data.diffPerDataAve[j][i])) * 0.2
                        self.colors_3d[i*self.n_row+j] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 200 / 255, 1 - abs(self._data.diffPerDataAve[j][i]) * 200 / 255,1]
                        for k in self.array_positions[i*self.n_row+j]:
                            self.colors[k] = [1, 1 - abs(self._data.diffPerDataAve[j][i]) * 200 / 255, 1 - abs(self._data.diffPerDataAve[j][i]) * 200 / 255,1]
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

        # My function to get the gesture
        if self.startSensor_pressed is True:
            self.gesture_recognition()

    def loadMesh(self,file_paths):
        for file_path in file_paths:
            mesh = pv.read(file_path)
            self.plotter.add_mesh(mesh)

    def changeVisibility(self,item):
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
        rawDataAve_Reverse = self._data.rawDataAve.T
        # print(f"{rawDataAve_Reverse}")
        return rawDataAve_Reverse

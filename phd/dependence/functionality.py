import pyvista as pv
import numpy as np
from pyvistaqt import QtInteractor, MainWindow
import os
from PyQt5.QtCore import QTimer
import time
from PyQt5.QtWidgets import QMessageBox, QLabel
from ui.ui_design import TreeWidgetItem

import serial
import serial.tools.list_ports
from math import sin, cos

class data:
    def __init__(self,n_row,n_col):
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


class MyFunction():
    def __init__(self,window:MainWindow) -> None:
        self.window: MainWindow = window
        # self.plotter: QtInteractor = window.plotter

        self.listModel = []
        self.listActor = []
        # self.creatPlaneXY()
        self.timer = QTimer()
        # self.timer.timeout.connect(self.update_animation)
        self.timer.start(0)
        self.frame_count = 0
        self.last_time = time.time()
        self.is_connected = False

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

    def add_sphere(self,showEdge):
        self.saveCameraPara()
        sphere = pv.Sphere()
        TreeWidgetItem(self.window.widget_tree,"Sphere",0,0)
        sphere.compute_normals(inplace=True)
        # vectors = np.vstack(
        #                         (
        #                             np.sin(sphere.points[:, 0]),
        #                             np.cos(sphere.points[:, 1]),
        #                             np.cos(sphere.points[:, 2]),
        #                         )
        #                     ).T
        # add and scale
        arrows = sphere['Normals']
        centers = sphere.cell_centers().points
        # sphere["vectors"] = vectors * 0.3
        # sphere.set_active_vectors("vectors")

        # self.window.plotter.add_mesh(sphere.arrows, show_edges=False)
        self.window.plotter.add_arrows(centers, arrows*0.05, color='white')
        self.window.plotter.add_mesh(sphere, show_edges=True)

    def addRobot(self):
        self.saveCameraPara()
        self.joints = [0,0,0,0,0,0,0]
        
        folder_path  = './resource/iiwa7/'
        stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]
        for stl_file in stl_files:
            stl_path = os.path.join(folder_path, stl_file)
            mesh = pv.read(stl_path)
            
            num_faces = mesh.n_points
            colors = np.ones((num_faces, 3)) * 0.5
            self.listModel.append(mesh)
            self.listActor.append(self.plotter.add_mesh(mesh,show_edges = False, scalars=colors, rgb=True))
            TreeWidgetItem(self.window.widget_tree,stl_path,0,0)
        self.T01 = np.array([[          cos(self.joints[0]),         -sin(self.joints[0]),               0,        0],
                                           [          sin(self.joints[0]),          cos(self.joints[0]),               0,        0],
                                           [                            0,                            0,               1,   0.1575],
                                           [                            0,                            0,               0,        1]])
        self.T12 = np.array([[    cos(np.pi+self.joints[1]),   -sin(np.pi+self.joints[1]),               0,        0],
                                           [                            0,                            0,               1,        0],
                                           [   -sin(np.pi+self.joints[1]),   -cos(np.pi+self.joints[1]),               0,    0.183],
                                           [                            0,                            0,               0,        1]])
        self.T23 = np.array([[    cos(np.pi+self.joints[2]),   -sin(np.pi+self.joints[2]),               0,        0],
                                           [                            0,                            0,               1,    0.184],
                                           [   -sin(np.pi+self.joints[2]),   -cos(np.pi+self.joints[2]),               0,        0],
                                           [                            0,                            0,               0,        1]])
        self.T34 = np.array([[          cos(self.joints[3]),         -sin(self.joints[3]),               0,        0],
                                           [                            0,                            0,              -1,        0],
                                           [          sin(self.joints[3]),          cos(self.joints[3]),               0,    0.216],
                                           [                            0,                            0,               0,        1]])
        self.T45 = np.array([[         -cos(self.joints[4]),          sin(self.joints[4]),               0,        0],
                                           [                            0,                            0,               1,    0.184],
                                           [          sin(self.joints[4]),          cos(self.joints[4]),               0,        0],
                                           [                            0,                            0,               0,        1]])
        self.T56 = np.array([[          cos(self.joints[5]),         -sin(self.joints[5]),               0,        0],
                                           [                            0,                            0,              -1,     0.06],
                                           [          sin(self.joints[5]),          cos(self.joints[5]),               0,    0.216],
                                           [                            0,                            0,               0,        1]])
        self.T67 = np.array([[         -cos(self.joints[6]),          sin(self.joints[6]),               0,        0],
                                           [                            0,                            0,               1,    0.081],
                                           [          sin(self.joints[6]),          cos(self.joints[6]),               0,     0.06],
                                           [                            0,                            0,               0,        1]])
        self.listModel[1].transform(self.T01)
        self.listModel[2].transform(self.T01@self.T12)
        self.listModel[3].transform(self.T01@self.T12@self.T23)
        self.listModel[4].transform(self.T01@self.T12@self.T23@self.T34)
        self.listModel[5].transform(self.T01@self.T12@self.T23@self.T34@self.T45)
        self.listModel[6].transform(self.T01@self.T12@self.T23@self.T34@self.T45@self.T56)
        self.listModel[7].transform(self.T01@self.T12@self.T23@self.T34@self.T45@self.T56@self.T67)

    def Init_handConstruction(self):
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
        self.com_options = []
        ports = serial.tools.list_ports.comports()
        self.ser = None
        for port in ports:
            self.com_options.append(port.name)
            self.window.serial_channel.addItem(port.name)
        for i in range(self.n_row):
            for j in range(self.n_col):
                self.points[i*(self.n_col)+j][0] = (j + 0.5) * size
                self.points[i*(self.n_col)+j][1] = (i + 0.5) * size
        
        for i in range(self.n_row-1):
            for j in range(self.n_col-1):
                self.edges[2 * (i * (self.n_col - 1) + j)] = [2,i * self.n_col + j, i * self.n_col + j + 1]
                self.edges[2 * (i * (self.n_col - 1) + j) + 1] = [2,i * self.n_col + j, (i + 1) * self.n_col + j]
        for i in range(self.n_col - 1):
            self.edges[(self.n_col - 1) * (self.n_row - 1) * 2 + i] = [2, (self.n_row - 1) * self.n_col + i,(self.n_row - 1) * self.n_col + i + 1]
        for i in range(self.n_row - 1):
            self.edges[(self.n_col - 1) * (self.n_row - 1) * 2 + (self.n_col - 1) + i] = [2, ((i + 1) * self.n_col) - 1,((i + 1) * self.n_col) - 1 + self.n_col]
        self._2D_map = pv.Plane((9.5,5,0),(0,0,1),19,10,19,10)
        self.line_poly = pv.PolyData(self.points)
        self.line_poly.lines = self.edges
        self.actionMap = self.plotter.add_mesh(self._2D_map,scalars=self.colors_face, show_edges=True,name = '2d',rgb = True)
        self.actionMesh = self.plotter.add_mesh(self.line_poly,scalars=self.colors, point_size=10,line_width = 3,render_points_as_spheres=True,rgb = True,name = '3d')
        TreeWidgetItem(self.window.widget_tree,"_2D_map",0,0)
        TreeWidgetItem(self.window.widget_tree,"_3D_mesh",0,0)

    def serialConnection(self):
        if self.ser:
                self.message("Already Connected")
        else:
            # 打开串口连接
            self.ser = serial.Serial(
                port = self.window.serial_channel.currentText(),
                baudrate=9600,
                timeout=1
            )
            # 向设备发送数据
            self.ser.write(b'ConnectionCheck')
            # 读取设备响应
            response = self.ser.readall()
            if response == b'Connected\r\n':
                self.message("The Serial Device Is Connected!")
                self.is_connected = True
                self.ser.write(b'readCal')
                response = self.ser.readline().decode('utf-8').rstrip()
                data_list = [int(value) for value in response.split() if value.isdigit()]
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
                        self.colors[i*self.n_col+j] = [i*0.1,j*0.05,0,1]
                for i in range(len(self.colors)):
                    if i < 190:
                        continue
                    else:
                        self.colors[i] = (self.colors[self.edges[i-190][1]]+self.colors[self.edges[i-190][2]])/2
                self.plotter.remove_actor(self.actionMap)
                self.actionMap = self.plotter.add_mesh(self._2D_map,scalars=self.colors_face, show_edges=True,name = '2d',rgb = True)
                self.plotter.remove_actor(self.actionMesh)
                self.actionMesh = self.plotter.add_mesh(self.line_poly,scalars=self.colors, point_size=10,line_width = 3,render_points_as_spheres=True,rgb = True,name = '3d')
            else:
                self.message("The Serial Device Is Not Responding!")
                self.is_connected = False
                self.ser.close()
                self.ser = None
    
        


    def update_animation(self):
        self.saveCameraPara()
        for i in range(len(self.listModel)):
            self.listModel[i].rotate_y(0.01, inplace=True)
            colors = np.random.rand(self.listModel[i].n_points, 3)
            self.listModel[i].point_data.set_scalars(colors)
        self.plotter.render()

        if self.is_connected:
            self.ser.write(b'readRaw')
            response = self.ser.readline().decode('utf-8').rstrip()
            data_list = [int(value) for value in response.split() if value.isdigit()]
            rawDataList = data_list[2:-2-self.n_row]
            self._data.getRaw(np.array(rawDataList).reshape(self.n_col,self.n_row))
            self._data.calDiff()
            self._data.calDiffPer()
            self._data.getWin(self._data.windowSize)
            for i in range(self.n_row):
                for j in range(self.n_col):
                    self.colors_face[i * self.n_col + j] = (1, 1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255, 1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255,1)
                    self.points[i*self.n_col+j][2] = 3-abs(self._data.diffPerDataAve[i][j])*2
                    self.colors[i*self.n_col+j] = [1, 1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255, 1 - abs(self._data.diffPerDataAve[i][j]) * 200 * 1.5 / 255,1]
            for i in range(len(self.colors)):
                if i < 190:
                    continue
                else:
                    self.colors[i] = (self.colors[self.edges[i-190][1]]+self.colors[self.edges[i-190][2]])/2
            self.line_poly.points = self.points
            self._2D_map.cell_data.set_scalars(self.colors_face)
            self.plotter.render()
        
        # 计算帧率
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_time >= 0.1:
            fps = self.frame_count / (current_time - self.last_time)
            self.message(f"FPS: {fps:.2f}")
            self.last_time = current_time
            self.frame_count = 0

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
            

    def message(self,text):
        self.window.label_info.setText(text)

    def message_about(self):
        QMessageBox.information(self.window, 'SingaLab', 'Hello, my friend.\nThis is Singa, hope everything goes well at your side.\nI creat SingaLab for research development with the Sponsorship from CPII. ')


    
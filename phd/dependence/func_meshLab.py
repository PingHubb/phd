import pyvista as pv
import numpy as np
from pyvistaqt import QtInteractor
import os
from PyQt5.QtCore import QTimer
import time
from phd.ui.ui_design import TreeWidgetItem
import math
from math import sin, cos


class MyMeshLab():
    def __init__(self, parent) -> None:
        self.parent = parent
        self.plotter: QtInteractor = self.parent.plotter
        self.plotter_2: QtInteractor = self.parent.plotter_2
        self.robotModel = []
        self.origin_list = []
        self.robotActor = []
        self.reT = []
        self.creatPlaneXY()
        self.timer = QTimer()
        # self.timer.timeout.connect(self.update_animation)
        self.timer.start(0)
        self.frame_count = 0
        self.last_time = time.time()
        self.is_connected = False
        self.counter = 0

        self.joints = [0.0] * 6  # Initialize joint angles

    def saveCameraPara(self):
        self.camera_pos = self.plotter.camera.position
        self.camera_focal = self.plotter.camera.focal_point
        self.camera_view_angle = self.plotter.camera.view_angle
    
    def loadCameraPare(self):
        self.plotter.camera.position = self.camera_pos
        self.plotter.camera.focal_point = self.camera_focal
        self.plotter.camera.view_angle = self.camera_view_angle

        self.plotter_2.camera.position = self.camera_pos
        self.plotter_2.camera.focal_point = self.camera_focal
        self.plotter_2.camera.view_angle = self.camera_view_angle

    def loadCameraPare(self, camera_pos, camera_focal, camera_view_angle):
        self.plotter.camera.position = camera_pos
        self.plotter.camera.focal_point = camera_focal
        self.plotter.camera.view_angle = camera_view_angle

        self.plotter_2.camera.position = self.camera_pos
        self.plotter_2.camera.focal_point = self.camera_focal
        self.plotter_2.camera.view_angle = self.camera_view_angle

    def creatPlaneXY(self):
        self.plotter.camera.position = (1,-1,1)
        self.plotter_2.camera.position = (1,-1,1)

        self.saveCameraPara()
        line = pv.Line((-50, 0, 0), (50, 0,0 ))
        
        # 添加X轴线段，并设置为红色
        self.plotter.add_mesh(line, color='r', line_width=2, label='X Axis')
        self.plotter_2.add_mesh(line, color='r', line_width=2, label='X Axis')

        line = pv.Line((0, -50, 0), (0,50, 0))

        # 添加Y轴线段，并设置为绿色
        self.plotter.add_mesh(line, color='g', line_width=2, label='Y Axis')
        self.plotter_2.add_mesh(line, color='g', line_width=2, label='Y Axis')

        planeXY = pv.Plane((0,0,0),(0,0,1),100,100,100,100)

        self.actorPlaneXY = self.plotter.add_mesh(planeXY, color='gray',style='wireframe')
        self.actorPlaneXY_2 = self.plotter_2.add_mesh(planeXY, color='gray',style='wireframe')


    def add_sphere(self, showEdge):
        self.saveCameraPara()
        sphere = pv.Sphere()
        TreeWidgetItem(self.parent.widget_tree,"Sphere",0,0)
        sphere.compute_normals(inplace=True)
        arrows = sphere['Normals']
        centers = sphere.cell_centers().points
        self.plotter.add_arrows(centers, arrows*0.05, color='white')
        self.plotter.add_mesh(sphere, show_edges=True)

    def addRobot(self):
        self.saveCameraPara()
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # self.joints = [math.radians(-180), math.radians(45), math.radians(45), math.radians(0), math.radians(90), math.radians(0)]
        folder_path = ('/home/ping2/ros2_ws/src/phd/phd/resource/robot/')

        try:
            obj_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.obj')])
            for obj_file in obj_files:
                obj_path = os.path.join(folder_path, obj_file)
                mesh = pv.read(obj_path)
                
                # Add the transformed mesh to the plotter
                self.robotModel.append(mesh)
                self.robotActor.append(self.plotter.add_mesh(mesh, show_edges=False))
                print(f"Loaded and centered {obj_path}")

        except Exception as e:
            print(f"Error reading or processing STL files: {e}")

        # for i in range(len(self.robotModel)):
        #     a = []
        #     for k in self.robotModel[i].points:
        #         a.append(k)
        #     self.origin_list.append(a)
        #     print(len(self.origin_list[i]))

        for i in range(7):
            self.reT.append(np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]))

        self.T01 = np.array([[cos(self.joints[0]), -sin(self.joints[0]), 0, 0],
                             [sin(self.joints[0]), cos(self.joints[0]), 0, 0],
                             [0, 0, 1, 0.1452],
                             [0, 0, 0, 1]])
        self.T12 = np.array([[sin(self.joints[1]), cos(self.joints[1]), 0, 0],
                             [0, 0, 1, 0],
                             [cos(self.joints[1]), -sin(self.joints[1]), 0, 0],
                             [0, 0, 0, 1]])
        self.T23 = np.array([[cos(self.joints[2]), -sin(self.joints[2]), 0, 0.429],
                             [sin(self.joints[2]), cos(self.joints[2]), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        self.T34 = np.array([[cos(np.pi / 2 + self.joints[3]), -sin(np.pi / 2 + self.joints[3]), 0, 0.4115],
                             [sin(np.pi / 2 + self.joints[3]), cos(np.pi / 2 + self.joints[3]), 0, 0],
                             [0, 0, 1, -0.1223],
                             [0, 0, 0, 1]])
        self.T45 = np.array([[cos(self.joints[4]), -sin(self.joints[4]), 0, 0],
                             [0, cos(np.pi / 2), -sin(np.pi / 2), -0.106],
                             [sin(self.joints[4]), cos(self.joints[4]), 0, 0],
                             [0, 0, 0, 1]])
        self.T56 = np.array([[cos(self.joints[5]), -sin(self.joints[5]), 0, 0],
                             [0, cos(np.pi / 2), -sin(np.pi / 2), -0.11315],
                             [sin(self.joints[5]), cos(self.joints[5]), 0, 0],
                             [0, 0, 0, 1]])

        self.robotModel[1].transform(self.T01 @ self.reT[0])
        self.robotModel[2].transform(self.T01 @ self.T12 @ self.reT[1])
        self.robotModel[3].transform(self.T01 @ self.T12 @ self.T23 @ self.reT[2])
        self.robotModel[4].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.reT[3])
        self.robotModel[5].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.reT[4])
        self.robotModel[6].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56 @ self.reT[5])
        self.robotModel[7].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56 @ self.reT[6])

        self.reT[0] = np.linalg.inv(self.T01)
        self.reT[1] = np.linalg.inv(self.T01 @ self.T12)
        self.reT[2] = np.linalg.inv(self.T01 @ self.T12 @ self.T23)
        self.reT[3] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34)
        self.reT[4] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45)
        self.reT[5] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56)
        self.reT[6] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56)

    def update_robot_joints(self, new_angles):
        if len(new_angles) != 6:
            print("Error: Expected 6 joint angles.")
            return

        # self.parent.log_display.append(f"Angles: {new_angles}")
        self.joints = new_angles
        self.update_robot_model()  # Call to update the robot's visual model

    def update_robot_model(self):
        # for idx, i in enumerate(self.robotModel):
        #     i.points = self.origin_list[idx]

        self.T01 = np.array([[cos(self.joints[0]), -sin(self.joints[0]), 0, 0],
                             [sin(self.joints[0]), cos(self.joints[0]), 0, 0],
                             [0, 0, 1, 0.1452],
                             [0, 0, 0, 1]])
        self.T12 = np.array([[sin(self.joints[1]), cos(self.joints[1]), 0, 0],
                             [0, 0, 1, 0],
                             [cos(self.joints[1]), -sin(self.joints[1]), 0, 0],
                             [0, 0, 0, 1]])
        self.T23 = np.array([[cos(self.joints[2]), -sin(self.joints[2]), 0, 0.429],
                             [sin(self.joints[2]), cos(self.joints[2]), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        self.T34 = np.array([[cos(np.pi / 2 + self.joints[3]), -sin(np.pi / 2 + self.joints[3]), 0, 0.4115],
                             [sin(np.pi / 2 + self.joints[3]), cos(np.pi / 2 + self.joints[3]), 0, 0],
                             [0, 0, 1, -0.1223],
                             [0, 0, 0, 1]])
        self.T45 = np.array([[cos(self.joints[4]), -sin(self.joints[4]), 0, 0],
                             [0, cos(np.pi / 2), -sin(np.pi / 2), -0.106],
                             [sin(self.joints[4]), cos(self.joints[4]), 0, 0],
                             [0, 0, 0, 1]])
        self.T56 = np.array([[cos(self.joints[5]), -sin(self.joints[5]), 0, 0],
                             [0, cos(np.pi / 2), -sin(np.pi / 2), -0.11315],
                             [sin(self.joints[5]), cos(self.joints[5]), 0, 0],
                             [0, 0, 0, 1]])

        self.robotModel[1].transform(self.T01 @ self.reT[0])
        self.robotModel[2].transform(self.T01 @ self.T12 @ self.reT[1])
        self.robotModel[3].transform(self.T01 @ self.T12 @ self.T23 @ self.reT[2])
        self.robotModel[4].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.reT[3])
        self.robotModel[5].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.reT[4])
        self.robotModel[6].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56 @ self.reT[5])
        self.robotModel[7].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56 @ self.reT[6])

        self.reT[0] = np.linalg.inv(self.T01)
        self.reT[1] = np.linalg.inv(self.T01 @ self.T12)
        self.reT[2] = np.linalg.inv(self.T01 @ self.T12 @ self.T23)
        self.reT[3] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34)
        self.reT[4] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45)
        self.reT[5] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56)
        self.reT[6] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56)

        # self.plotter.render()

    def serialConnection(self):
        print("Not implemented yet")
    
    def update_animation(self):
        self.saveCameraPara()
        for i in range(len(self.robotModel)):
            self.robotModel[i].rotate_y(0.01, inplace=True)
            colors = np.random.rand(self.robotModel[i].n_points, 3)
            self.robotModel[i].point_data.set_scalars(colors)
        self.plotter.render()

        # 计算帧率
        current_time = time.time()
        self.frame_count += 1
        if current_time - self.last_time >= 0.1:
            fps = self.frame_count / (current_time - self.last_time)
            # self.message(f"FPS: {fps:.2f}")
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

    
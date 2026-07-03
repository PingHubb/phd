import pyvista as pv
import numpy as np
from pyvistaqt import QtInteractor
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
)
import time
from phd.dependence.paths import resource_path, robot_resource_path
import math
from math import sin, cos


class MyMeshLab():
    _STEP_POINT_RE = re.compile(
        r"CARTESIAN_POINT\s*\(\s*''\s*,\s*\(\s*"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*\)\s*\)"
    )
    _RH56F1_FINGER_SENSOR_LINKS = (
        ("little", "little_force_sensor"),
        ("ring", "ring_force_sensor"),
        ("middle", "middle_force_sensor"),
        ("index", "index_force_sensor"),
        ("thumb", "thumb_force_sensor"),
    )
    _RH56F1_SENSOR_LINKS = {link for _, link in _RH56F1_FINGER_SENSOR_LINKS}
    _RH56F1_SENSOR_LINKS.add("plam_force_sensor")
    # Palm data arrives as three groups of (normal force, tangential force,
    # tangential direction). Hardware testing showed the visual left/right
    # order is mirrored from the manual register wording, so map the incoming
    # groups to left -> middle -> right.
    _RH56F1_PALM_DATA_REGIONS = ("palm_left", "palm_middle", "palm_right")
    # Full-scale raw force for colour mapping: manual states raw 1024 = 10.24 N.
    _RH56F1_FORCE_FULL_SCALE = 1024.0
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
        # Keep this timer stopped unless update_animation is explicitly wired.
        # A 0-ms timer with no connected slot still adds avoidable Qt traffic.
        self._hand_tactile_update_count = 0
        self._hand_tactile_render_count = 0
        self._hand_tactile_render_skip_count = 0
        self._hand_tactile_render_total_sec = 0.0
        self._hand_tactile_last_render_sec = 0.0
        self._hand_tactile_report_started_at = time.perf_counter()
        self.frame_count = 0
        self.last_time = time.time()
        self.is_connected = False
        self.counter = 0

        self.joints = [0.0] * 6  # Initialize joint angles

    def saveCameraPara(self):
        self.camera_pos = self.plotter.camera.position
        self.camera_focal = self.plotter.camera.focal_point
        self.camera_view_angle = self.plotter.camera.view_angle

    def _apply_camera_parameters(self, plotter, camera_pos, camera_focal, camera_view_angle):
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = camera_focal
        plotter.camera.view_angle = camera_view_angle

    def loadCameraPare(self, camera_pos=None, camera_focal=None, camera_view_angle=None):
        if camera_pos is None:
            camera_pos = getattr(self, 'camera_pos', None)
        if camera_focal is None:
            camera_focal = getattr(self, 'camera_focal', None)
        if camera_view_angle is None:
            camera_view_angle = getattr(self, 'camera_view_angle', None)

        if camera_pos is None or camera_focal is None or camera_view_angle is None:
            return

        self._apply_camera_parameters(self.plotter, camera_pos, camera_focal, camera_view_angle)
        self._apply_camera_parameters(self.plotter_2, camera_pos, camera_focal, camera_view_angle)

    def creatPlaneXY(self):
        self.plotter.camera.position = (1, -1, 1)
        self.plotter_2.camera.position = (1, -1, 1)

        self.saveCameraPara()
        line = pv.Line((-50, 0, 0), (50, 0, 0))

        # 添加X轴线段，并设置为红色
        self.plotter.add_mesh(line, color='r', line_width=2, label='X Axis')
        self.plotter_2.add_mesh(line, color='r', line_width=2, label='X Axis')

        line = pv.Line((0, -50, 0), (0, 50, 0))

        # 添加Y轴线段，并设置为绿色
        self.plotter.add_mesh(line, color='g', line_width=2, label='Y Axis')
        self.plotter_2.add_mesh(line, color='g', line_width=2, label='Y Axis')

        planeXY = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size=100,
            j_size=100,
            i_resolution=100,
            j_resolution=100,
        )

        self.actorPlaneXY = self.plotter.add_mesh(planeXY, color='gray', style='wireframe')
        self.actorPlaneXY_2 = self.plotter_2.add_mesh(planeXY, color='gray', style='wireframe')

    def add_sphere(self, showEdge):
        self.saveCameraPara()
        sphere = pv.Sphere()
        sphere.compute_normals(inplace=True)
        arrows = sphere['Normals']
        centers = sphere.cell_centers().points
        self.plotter.add_arrows(centers, arrows * 0.05, color='white')
        self.plotter.add_mesh(sphere, show_edges=True)

    def addRobot(self):
        self.saveCameraPara()
        self.joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # self.joints = [math.radians(-180), math.radians(45), math.radians(45), math.radians(0), math.radians(90), math.radians(0)]
        folder_path = robot_resource_path()

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

        self.robotModel[1].transform(self.T01 @ self.reT[0], inplace=True)
        self.robotModel[2].transform(self.T01 @ self.T12 @ self.reT[1], inplace=True)
        self.robotModel[3].transform(self.T01 @ self.T12 @ self.T23 @ self.reT[2], inplace=True)
        self.robotModel[4].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.reT[3], inplace=True)
        self.robotModel[5].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.reT[4], inplace=True)
        self.robotModel[6].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56 @ self.reT[5], inplace=True)
        self.robotModel[7].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56 @ self.reT[6], inplace=True)

        self.reT[0] = np.linalg.inv(self.T01)
        self.reT[1] = np.linalg.inv(self.T01 @ self.T12)
        self.reT[2] = np.linalg.inv(self.T01 @ self.T12 @ self.T23)
        self.reT[3] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34)
        self.reT[4] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45)
        self.reT[5] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56)
        self.reT[6] = np.linalg.inv(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56)

        # The robot meshes are added BEFORE the kinematic transforms, so their
        # bounds in the plotter (and any current camera framing) ignore the robot
        # entirely. After all transforms are applied, reset the camera so the
        # newly-imported robot is actually visible, and force a redraw in case
        # PyVista did not invalidate the actor bounds automatically.
        try:
            self.plotter.reset_camera()
            self.plotter.render()
        except Exception as exc:
            print(f"[addRobot] camera reset failed: {exc}")

    # ------------------------------------------------------------------
    # Pop-up robot viewer (separate from the sensor plotter)
    # ------------------------------------------------------------------
    def _poke_main_plotters(self):
        """Force a Render() on the main sensor plotter(s).

        Creating / interacting with a second ``QtInteractor`` in another
        window can leave the primary plotter without a refresh, which shows
        up as a black or frozen render area. Calling ``Render()`` on its
        VTK render window (and on the secondary ``plotter_2``) recovers it.
        Safe to call repeatedly and at any time.
        """
        for attr in ("plotter", "plotter_2"):
            p = getattr(self, attr, None)
            if p is None:
                continue
            # Try the highest-level API first; fall back to VTK if needed.
            try:
                p.render()
                continue
            except Exception:
                pass
            try:
                interactor = getattr(p, "interactor", None)
                if interactor is not None:
                    rw = interactor.GetRenderWindow()
                    if rw is not None:
                        rw.Render()
                    interactor.update()
            except Exception:
                pass

    @staticmethod
    def _compute_kinematic_chain(joints):
        """Return cumulative homogeneous transforms for the 7 visible links.

        Output is a list of length 8 aligned with the 8 OBJ meshes
        ``[base, L1, L2, L3, L4, L5, L6, tool]``. Index 0 is ``None`` because
        the base mesh stays at the world origin. Indices 6 and 7 share the
        same transform (wrist + tool flange).
        """
        T01 = np.array([[cos(joints[0]), -sin(joints[0]), 0, 0],
                        [sin(joints[0]),  cos(joints[0]), 0, 0],
                        [0, 0, 1, 0.1452],
                        [0, 0, 0, 1]])
        T12 = np.array([[sin(joints[1]),  cos(joints[1]), 0, 0],
                        [0, 0, 1, 0],
                        [cos(joints[1]), -sin(joints[1]), 0, 0],
                        [0, 0, 0, 1]])
        T23 = np.array([[cos(joints[2]), -sin(joints[2]), 0, 0.429],
                        [sin(joints[2]),  cos(joints[2]), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        T34 = np.array([[cos(np.pi / 2 + joints[3]), -sin(np.pi / 2 + joints[3]), 0, 0.4115],
                        [sin(np.pi / 2 + joints[3]),  cos(np.pi / 2 + joints[3]), 0, 0],
                        [0, 0, 1, -0.1223],
                        [0, 0, 0, 1]])
        T45 = np.array([[cos(joints[4]), -sin(joints[4]), 0, 0],
                        [0, cos(np.pi / 2), -sin(np.pi / 2), -0.106],
                        [sin(joints[4]),  cos(joints[4]), 0, 0],
                        [0, 0, 0, 1]])
        T56 = np.array([[cos(joints[5]), -sin(joints[5]), 0, 0],
                        [0, cos(np.pi / 2), -sin(np.pi / 2), -0.11315],
                        [sin(joints[5]),  cos(joints[5]), 0, 0],
                        [0, 0, 0, 1]])
        C1 = T01
        C2 = C1 @ T12
        C3 = C2 @ T23
        C4 = C3 @ T34
        C5 = C4 @ T45
        C6 = C5 @ T56
        return [None, C1, C2, C3, C4, C5, C6, C6]

    def addRobotInDialog(self):
        """Open the imported 3D robot in a stand-alone dialog window.

        Builds a fresh ``QtInteractor`` plotter so the robot meshes never enter
        the main sensor plotter. Re-pressing the button just raises the existing
        dialog instead of creating a duplicate (and re-loading hundreds of MB
        of OBJ data). The window starts at joints = [0]*6 and exposes a
        "Real-time Live" toggle that polls ``robot_api.get_current_positions()``
        periodically and re-poses the displayed links.
        """
        # Re-use existing window if it's still alive AND its VTK resources
        # haven't been torn down. After a previous close the dialog widget
        # may still be Python-referenced briefly while Qt deletes it; that's
        # not a valid window to re-show, so we sanity-check before reusing.
        existing = getattr(self, "_robot_dialog", None)
        existing_plotter = getattr(self, "_robot_dialog_plotter", None)
        if existing is not None and existing_plotter is not None:
            try:
                if existing.isVisible() or not existing.testAttribute(Qt.WA_DeleteOnClose):
                    existing.show()
                    existing.raise_()
                    existing.activateWindow()
                    self._poke_main_plotters()
                    return
            except RuntimeError:
                # Qt widget has been deleted underneath us — drop the stale
                # reference and fall through to build a fresh dialog.
                pass
            except Exception:
                pass
        # Anything left over from a previous (now-dead) dialog: drop it.
        self._robot_dialog = None
        self._robot_dialog_plotter = None
        self._robot_dialog_meshes = None
        self._robot_dialog_applied = None
        self._robot_dialog_live_timer = None
        self._robot_dialog_drag_timer = None
        self._robot_dialog_main_keepalive_timer = None

        parent_widget = getattr(self, "parent", None)

        # Flush any pending paint events for the sensor plotter BEFORE we
        # spin up a second QtInteractor. Creating another VTK render window
        # in the same Qt app can otherwise steal the GL context from the
        # existing plotter, leaving it black/frozen until something forces
        # a Render() on it again.
        try:
            QApplication.processEvents()
        except Exception:
            pass
        self._poke_main_plotters()
        dialog = QDialog(parent_widget)
        dialog.setWindowTitle("3D Robot Model")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.Window)
        # Destroy the underlying Qt widget on close so the QtInteractor's VTK
        # render window is properly finalized and stops competing for the
        # OpenGL context. Without this the sensor plotter can stay frozen.
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.resize(900, 760)
        v = QVBoxLayout(dialog)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ------------------------------------------------------------------
        # Top toolbar: live follow toggle + reset + status label
        # ------------------------------------------------------------------
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 6, 8, 6)
        toolbar.setSpacing(8)

        btn_live = QPushButton("● Real-time Live")
        btn_live.setCheckable(True)
        btn_live.setToolTip(
            "Toggle live follow: poll the robot's current joint angles and "
            "update the visualization in real time."
        )
        btn_live.setStyleSheet(
            "QPushButton{padding:4px 12px;}"
            "QPushButton:checked{background-color:#c84040;color:white;font-weight:bold;}"
        )
        toolbar.addWidget(btn_live)

        btn_reset = QPushButton("Reset View")
        btn_reset.setToolTip(
            "Re-sync the model to the robot's current joint angles and "
            "re-frame the camera."
        )
        toolbar.addWidget(btn_reset)

        btn_home = QPushButton("Home Pose")
        btn_home.setToolTip("Pose the model at joints = 0 (does NOT move the real robot).")
        toolbar.addWidget(btn_home)

        btn_drag_vel = QPushButton("✥ Drag (Velocity)")
        btn_drag_vel.setCheckable(True)
        btn_drag_vel.setToolTip(
            "Velocity drag: stream a base-frame linear velocity toward the dragged "
            "sphere via a 30 Hz P-controller (uses ContinueVLine velocity mode).\n"
            "Best for smooth, fast, continuous tracking."
        )
        btn_drag_vel.setStyleSheet(
            "QPushButton{padding:4px 12px;}"
            "QPushButton:checked{background-color:#d68e1d;color:white;font-weight:bold;}"
        )
        toolbar.addWidget(btn_drag_vel)

        btn_drag_ptp = QPushButton("✥ Drag (PTP)")
        btn_drag_ptp.setCheckable(True)
        btn_drag_ptp.setToolTip(
            "PTP drag: throttled tool-space PTPs as the sphere moves (motion_type=2 "
            "PTP via SetPositions).\n"
            "Best for discrete, precise hops to a teach target."
        )
        btn_drag_ptp.setStyleSheet(
            "QPushButton{padding:4px 12px;}"
            "QPushButton:checked{background-color:#1d8ed6;color:white;font-weight:bold;}"
        )
        toolbar.addWidget(btn_drag_ptp)

        toolbar.addStretch(1)

        status_label = QLabel("Idle")
        status_label.setStyleSheet("color:#aaaaaa;")
        toolbar.addWidget(status_label)

        v.addLayout(toolbar)

        # ------------------------------------------------------------------
        # Plotter
        # ------------------------------------------------------------------
        plotter = QtInteractor(parent=dialog)
        v.addWidget(plotter.interactor)
        # Match the main sensor plotter look-and-feel: dark charcoal bg, faint
        # wireframe ground plane and bright RGB axes, so the imported robot
        # reads the same as inside the sensor view.
        try:
            plotter.background_color = '#202020'
        except Exception:
            try:
                plotter.set_background('#202020')
            except Exception:
                pass

        try:
            ground = pv.Plane(
                center=(0, 0, 0),
                direction=(0, 0, 1),
                i_size=2.0,
                j_size=2.0,
                i_resolution=20,
                j_resolution=20,
            )
            plotter.add_mesh(ground, color='gray', style='wireframe', opacity=0.4)
        except Exception:
            pass

        plotter.add_mesh(pv.Line((-1.5, 0, 0), (1.5, 0, 0)), color='#ff5050', line_width=2)
        plotter.add_mesh(pv.Line((0, -1.5, 0), (0, 1.5, 0)), color='#50ff50', line_width=2)
        plotter.add_mesh(pv.Line((0, 0, 0), (0, 0, 1.5)), color='#5080ff', line_width=2)
        try:
            plotter.add_axes(interactive=False)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Load OBJ links + pose at joints = 0
        # ------------------------------------------------------------------
        folder_path = robot_resource_path()
        meshes = []
        try:
            obj_files = sorted(
                f for f in os.listdir(folder_path) if f.endswith('.obj')
            )
            for obj_file in obj_files:
                obj_path = os.path.join(folder_path, obj_file)
                meshes.append(pv.read(obj_path))
                print(f"[RobotDialog] Loaded {obj_path}")
        except Exception as exc:
            print(f"[RobotDialog] Error reading OBJ files: {exc}")

        zero_joints = [0.0] * 6
        applied = self._compute_kinematic_chain(zero_joints)
        for idx, M in enumerate(applied):
            if M is None or idx >= len(meshes):
                continue
            try:
                meshes[idx].transform(M, inplace=True)
            except Exception as exc:
                print(f"[RobotDialog] Transform link {idx} failed: {exc}")

        for mesh in meshes:
            plotter.add_mesh(
                mesh,
                show_edges=False,
                color='#d8d8dc',
                specular=0.5,
                specular_power=15,
                smooth_shading=True,
            )

        try:
            plotter.reset_camera()
            plotter.render()
        except Exception as exc:
            print(f"[RobotDialog] camera reset failed: {exc}")

        # ------------------------------------------------------------------
        # Live-follow plumbing
        # ------------------------------------------------------------------
        self._robot_dialog = dialog
        self._robot_dialog_plotter = plotter
        self._robot_dialog_meshes = meshes
        self._robot_dialog_applied = applied            # what's currently shown
        self._robot_dialog_current_joints = list(zero_joints)
        self._robot_dialog_status = status_label
        self._robot_dialog_live_btn = btn_live
        self._robot_dialog_drag_vel_btn = btn_drag_vel
        self._robot_dialog_drag_ptp_btn = btn_drag_ptp
        self._robot_dialog_drag_active = False
        self._robot_dialog_drag_mode = None  # 'velocity' | 'ptp' | None
        self._robot_dialog_drag_anchor = None
        self._robot_dialog_drag_quat = None
        self._robot_dialog_drag_confirmed = False
        self._robot_dialog_drag_target = None
        self._robot_dialog_drag_velocity_mode_on = False
        self._robot_dialog_drag_ptp_last_t = 0.0
        self._robot_dialog_drag_ptp_last_target = None

        # Closed-loop velocity timer: while drag mode is on this fires at
        # ~30 Hz, reads the real TCP, and streams a tool-velocity command
        # toward the dragged sphere position. Speed is responsive yet bounded.
        drag_timer = QTimer(dialog)
        drag_timer.setInterval(33)
        drag_timer.timeout.connect(self._robot_dialog_drag_velocity_tick)
        self._robot_dialog_drag_timer = drag_timer

        live_timer = QTimer(dialog)
        live_timer.setInterval(100)  # 10 Hz live follow
        live_timer.timeout.connect(self._robot_dialog_live_tick)
        self._robot_dialog_live_timer = live_timer

        def _on_live_toggled(checked: bool):
            if checked:
                api = getattr(self.parent, "robot_api", None)
                if api is None or not hasattr(api, "get_current_positions"):
                    status_label.setText("Robot API not available")
                    btn_live.setChecked(False)
                    return
                status_label.setText("Live: following robot…")
                # Fire immediately, then start the periodic poll.
                self._robot_dialog_live_tick()
                live_timer.start()
            else:
                live_timer.stop()
                status_label.setText("Idle")

        btn_live.toggled.connect(_on_live_toggled)

        def _on_reset():
            # Re-sync the model to the robot's current joint angles (one-shot
            # snapshot), then re-frame the camera. This is more useful than a
            # camera-only reset: it lets the user snap the simulation back to
            # the real robot's pose without needing to enable live follow.
            api = getattr(self.parent, "robot_api", None)
            joints = None
            if api is not None and hasattr(api, "get_current_positions"):
                try:
                    joints = api.get_current_positions()
                except Exception:
                    joints = None
            if joints is not None and len(joints) >= 6:
                self._apply_robot_dialog_joints(joints)
                deg = [j * 180.0 / np.pi for j in joints[:6]]
                status_label.setText(
                    "Reset: J=[{0:+6.1f}, {1:+6.1f}, {2:+6.1f}, {3:+6.1f}, "
                    "{4:+6.1f}, {5:+6.1f}]°".format(*deg)
                )
            else:
                status_label.setText("Reset: no joint feedback (camera only)")
            try:
                plotter.reset_camera()
                plotter.render()
            except Exception:
                pass

        btn_reset.clicked.connect(_on_reset)

        def _on_home():
            # Stop live follow and snap the visualization back to joints = 0.
            if btn_live.isChecked():
                btn_live.setChecked(False)
            self._apply_robot_dialog_joints([0.0] * 6)
            status_label.setText("Pose: home (0,0,0,0,0,0)")

        btn_home.clicked.connect(_on_home)

        def _on_drag_vel_toggled(checked: bool):
            if checked and btn_drag_ptp.isChecked():
                btn_drag_ptp.blockSignals(True)
                btn_drag_ptp.setChecked(False)
                btn_drag_ptp.blockSignals(False)
                # Tear down the other mode cleanly before switching.
                self._teardown_robot_dialog_drag()
            self._toggle_robot_dialog_drag(checked, mode='velocity')

        def _on_drag_ptp_toggled(checked: bool):
            if checked and btn_drag_vel.isChecked():
                btn_drag_vel.blockSignals(True)
                btn_drag_vel.setChecked(False)
                btn_drag_vel.blockSignals(False)
                self._teardown_robot_dialog_drag()
            self._toggle_robot_dialog_drag(checked, mode='ptp')

        btn_drag_vel.toggled.connect(_on_drag_vel_toggled)
        btn_drag_ptp.toggled.connect(_on_drag_ptp_toggled)

        # Keep-alive renderer: every ~750 ms force a Render() on the main
        # sensor plotter while the robot dialog is open. This is the most
        # reliable workaround for multi-QtInteractor OpenGL context
        # contention (otherwise the main plotter can freeze/turn black).
        main_keepalive_timer = QTimer(dialog)
        main_keepalive_timer.setInterval(750)
        main_keepalive_timer.timeout.connect(self._poke_main_plotters)
        self._robot_dialog_main_keepalive_timer = main_keepalive_timer
        main_keepalive_timer.start()

        # Cleanly stop timers / drag widget when the user closes the dialog.
        def _on_finished(_result):
            try:
                live_timer.stop()
            except Exception:
                pass
            try:
                if getattr(self, "_robot_dialog_drag_active", False):
                    self._teardown_robot_dialog_drag()
            except Exception:
                pass
            try:
                drag_timer.stop()
            except Exception:
                pass
            try:
                main_keepalive_timer.stop()
            except Exception:
                pass

            # Explicitly tear down the dialog's QtInteractor so its VTK
            # render window is finalized and releases the OpenGL context
            # back to the system. Without this the main sensor plotter
            # often stays "frozen" until another GL event nudges it.
            dlg_plotter = getattr(self, "_robot_dialog_plotter", None)
            if dlg_plotter is not None:
                try:
                    dlg_plotter.close()
                except Exception:
                    pass
                try:
                    iren = getattr(dlg_plotter, "interactor", None)
                    if iren is not None:
                        rw = iren.GetRenderWindow()
                        if rw is not None:
                            rw.Finalize()
                        iren.TerminateApp()
                except Exception:
                    pass

            # Drop every reference so the next "Import 3D Robot Model" press
            # builds a fresh dialog and doesn't try to revive dead VTK
            # resources.
            self._robot_dialog = None
            self._robot_dialog_plotter = None
            self._robot_dialog_meshes = None
            self._robot_dialog_applied = None
            self._robot_dialog_current_joints = None
            self._robot_dialog_status = None
            self._robot_dialog_live_btn = None
            self._robot_dialog_drag_vel_btn = None
            self._robot_dialog_drag_ptp_btn = None
            self._robot_dialog_drag_active = False
            self._robot_dialog_drag_mode = None
            self._robot_dialog_drag_target = None
            self._robot_dialog_drag_widget = None
            self._robot_dialog_live_timer = None
            self._robot_dialog_drag_timer = None
            self._robot_dialog_main_keepalive_timer = None

            # Spam the recovery render a few times: GL context handoff can
            # take a moment after the window destroys, so single Render()
            # isn't always enough.
            self._poke_main_plotters()
            for ms in (50, 200, 500, 900, 1500):
                QTimer.singleShot(ms, self._poke_main_plotters)

        dialog.finished.connect(_on_finished)

        dialog.show()

        # After showing the new dialog, give VTK a moment to settle and then
        # force a render on the main plotter so it re-paints itself.
        for ms in (50, 150, 350, 700):
            QTimer.singleShot(ms, self._poke_main_plotters)

    # ------------------------------------------------------------------
    # Drag-to-PTP support
    # ------------------------------------------------------------------
    def _toggle_robot_dialog_drag(self, enabled: bool, mode: str = 'velocity'):
        """Enable/disable the drag-the-TCP sphere widget inside the robot dialog.

        Two modes are supported:

        * ``'velocity'`` — places a yellow sphere widget at the current TCP
          and runs a 30 Hz closed-loop P-controller that streams base-frame
          velocity (``ContinueVLine``) toward the dragged sphere. Smooth and
          fast; the robot starts moving as soon as the sphere moves.

        * ``'ptp'`` — places the same sphere widget but on each VTK interaction
          event sends a throttled tool-space PTP (``SetPositions`` with
          ``motion_type=2``). Good for discrete precise hops to a teach target.
        """
        plotter = getattr(self, "_robot_dialog_plotter", None)
        status = getattr(self, "_robot_dialog_status", None)
        if mode == 'ptp':
            btn = getattr(self, "_robot_dialog_drag_ptp_btn", None)
        else:
            btn = getattr(self, "_robot_dialog_drag_vel_btn", None)
        dialog = getattr(self, "_robot_dialog", None)
        if plotter is None:
            return

        if not enabled:
            self._teardown_robot_dialog_drag()
            return

        # Optional one-time warning shared by both modes: dragging moves the
        # real robot.
        if not getattr(self, "_robot_dialog_drag_confirmed", False):
            box = QMessageBox(dialog)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Drag Tool: real robot will move")
            box.setText(
                "Drag modes (Velocity / PTP) command the REAL robot while you "
                "drag the yellow sphere.\n\n"
                "Make sure the surroundings are clear before continuing."
            )
            box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Cancel)
            if box.exec_() != QMessageBox.Ok:
                if btn is not None:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
                if status is not None:
                    status.setText("Drag: cancelled.")
                return
            self._robot_dialog_drag_confirmed = True

        api = getattr(self.parent, "robot_api", None)
        if mode == 'velocity':
            required = (
                api is not None
                and hasattr(api, "get_current_tool_position")
                and hasattr(api, "send_request")
                and (
                    hasattr(api, "send_end_effector_velocity")
                    or (
                        hasattr(api, "enable_end_effector_velocity_mode")
                        and hasattr(api, "stop_end_effector_velocity_mode")
                        and hasattr(api, "set_end_effector_velocity")
                    )
                )
            )
            need_msg = "Drag: robot velocity API not available."
        else:  # 'ptp'
            required = (
                api is not None
                and hasattr(api, "get_current_tool_position")
                and hasattr(api, "send_positions_tool_position")
            )
            need_msg = "Drag: robot PTP API not available."

        if not required:
            if status is not None:
                status.setText(need_msg)
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
            return

        pose = None
        try:
            pose = api.get_current_tool_position()
        except Exception as exc:
            if status is not None:
                status.setText(f"Drag: pose read failed: {exc}")
        if not pose or pose[0] is None:
            if status is not None:
                status.setText("Drag: waiting for tool pose feedback…")
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
            return
        pos, quat = pose  # quat = (w, x, y, z)

        # Auto-enable live follow when entering drag mode so the user can see
        # the simulated robot chase the dragged sphere in real time. The live
        # tick only updates the link meshes (not the sphere widget), so the
        # drag target the user is holding is unaffected.
        live_btn = getattr(self, "_robot_dialog_live_btn", None)
        if live_btn is not None and not live_btn.isChecked():
            live_btn.setChecked(True)

        self._robot_dialog_drag_anchor = list(pos)
        self._robot_dialog_drag_quat = list(quat)
        self._robot_dialog_drag_target = list(pos)
        # PTP-mode throttle state.
        self._robot_dialog_drag_ptp_last_t = 0.0
        self._robot_dialog_drag_ptp_last_target = list(pos)

        # Pick the callback for the current mode.
        if mode == 'ptp':
            cb = self._on_robot_dialog_drag_ptp_event
            sphere_color = '#3aa0ff'  # blue tint for PTP mode
        else:
            cb = self._on_robot_dialog_drag_target_changed
            sphere_color = 'yellow'

        try:
            widget = plotter.add_sphere_widget(
                callback=cb,
                center=tuple(pos),
                radius=0.025,
                color=sphere_color,
                selected_color='red',
                test_callback=False,
                interaction_event='always',
            )
            self._robot_dialog_drag_widget = widget
            self._robot_dialog_drag_active = True
            self._robot_dialog_drag_mode = mode
            try:
                plotter.render()
            except Exception:
                pass
        except Exception as exc:
            if status is not None:
                status.setText(f"Drag: widget failed: {exc}")
            self._robot_dialog_drag_active = False
            self._robot_dialog_drag_mode = None
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
            return

        if mode == 'velocity':
            # Enter velocity mode and start the 30 Hz closed-loop tracker.
            try:
                if hasattr(api, "enter_end_effector_velocity_mode"):
                    api.enter_end_effector_velocity_mode(suspend_existing=True)
                else:
                    api.send_request(api.enable_end_effector_velocity_mode())
                self._robot_dialog_drag_velocity_mode_on = True
            except Exception as exc:
                if status is not None:
                    status.setText(f"Drag: enable velocity failed: {exc}")
                self._robot_dialog_drag_velocity_mode_on = False
                try:
                    plotter.clear_sphere_widgets()
                except Exception:
                    pass
                self._robot_dialog_drag_active = False
                self._robot_dialog_drag_mode = None
                if btn is not None:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
                return

            timer = getattr(self, "_robot_dialog_drag_timer", None)
            if timer is not None and not timer.isActive():
                timer.start()

            if status is not None:
                status.setText(
                    "Drag (Velocity): move the yellow sphere; robot tracks live."
                )
        else:
            # PTP mode: no timer; throttled PTP sent from the widget callback.
            if status is not None:
                status.setText(
                    "Drag (PTP): hold and drag the blue sphere — robot hops to it."
                )

    def _teardown_robot_dialog_drag(self):
        """Common teardown logic used by both drag modes."""
        plotter = getattr(self, "_robot_dialog_plotter", None)
        status = getattr(self, "_robot_dialog_status", None)

        # Stop the velocity loop first so no more commands are emitted.
        timer = getattr(self, "_robot_dialog_drag_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

        api = getattr(self.parent, "robot_api", None)
        if api is not None and getattr(self, "_robot_dialog_drag_velocity_mode_on", False):
            try:
                if hasattr(api, "exit_end_effector_velocity_mode"):
                    api.exit_end_effector_velocity_mode(send_zero=True)
                else:
                    api.send_request(api.set_end_effector_velocity([0.0] * 6))
                    if hasattr(api, "suspend_end_effector_velocity_mode"):
                        api.send_request(api.suspend_end_effector_velocity_mode())
                    api.send_request(api.stop_end_effector_velocity_mode())
            except Exception:
                pass
        self._robot_dialog_drag_velocity_mode_on = False

        if plotter is not None:
            try:
                plotter.clear_sphere_widgets()
            except Exception:
                try:
                    plotter.clear_button_widgets()
                except Exception:
                    pass
            try:
                plotter.render()
            except Exception:
                pass

        self._robot_dialog_drag_active = False
        self._robot_dialog_drag_mode = None
        self._robot_dialog_drag_target = None
        if status is not None:
            status.setText("Drag: idle.")

    def _on_robot_dialog_drag_target_changed(self, *args):
        """Sphere widget moved (velocity mode) → just update the loop target.

        The actual command is emitted by ``_robot_dialog_drag_velocity_tick``
        on a QTimer so the ROS service isn't flooded by VTK's continuous
        interaction events.
        """
        if not args:
            return
        raw = args[0]
        try:
            self._robot_dialog_drag_target = [
                float(raw[0]), float(raw[1]), float(raw[2])
            ]
        except Exception:
            return

    def _on_robot_dialog_drag_ptp_event(self, *args):
        """Sphere widget moved (PTP mode) → throttled tool-space PTP send.

        Uses a small time + delta gate so we don't queue a flood of PTP
        commands while VTK is firing interaction events continuously.
        """
        if not args:
            return
        raw = args[0]
        try:
            target = [float(raw[0]), float(raw[1]), float(raw[2])]
        except Exception:
            return

        status = getattr(self, "_robot_dialog_status", None)
        api = getattr(self.parent, "robot_api", None)
        if api is None or not hasattr(api, "send_positions_tool_position"):
            if status is not None:
                status.setText("Drag (PTP): robot API not available.")
            return

        now = time.monotonic()
        min_dt = 0.10           # ≤ 10 Hz PTP rate
        min_step_m = 0.003      # 3 mm minimum delta to send a new hop

        last_t = getattr(self, "_robot_dialog_drag_ptp_last_t", 0.0)
        last_target = getattr(self, "_robot_dialog_drag_ptp_last_target", None)
        if (now - last_t) < min_dt:
            return
        if last_target is not None and len(last_target) == 3:
            dx = target[0] - float(last_target[0])
            dy = target[1] - float(last_target[1])
            dz = target[2] - float(last_target[2])
            if (dx * dx + dy * dy + dz * dz) ** 0.5 < min_step_m:
                return

        quat = getattr(self, "_robot_dialog_drag_quat", None)
        if quat is None or len(quat) != 4:
            quat = [1.0, 0.0, 0.0, 0.0]

        sent = False
        try:
            sent = api.send_positions_tool_position(
                target,
                list(quat),
                velocity=0.3,
                blend_percentage=100,
                fine_goal=False,
            )
        except Exception as exc:
            if status is not None:
                status.setText(f"Drag (PTP): send failed: {exc}")
            return

        self._robot_dialog_drag_ptp_last_t = now
        self._robot_dialog_drag_ptp_last_target = list(target)
        self._robot_dialog_drag_target = list(target)
        self._robot_dialog_drag_anchor = list(target)

        if status is not None:
            tag = "sent" if sent else "REJECTED"
            status.setText(
                "Drag (PTP) {0}: ({1:+.3f}, {2:+.3f}, {3:+.3f}) m".format(
                    tag, target[0], target[1], target[2]
                )
            )

    def _robot_dialog_drag_velocity_tick(self):
        """30 Hz: stream a base-frame linear velocity toward the dragged sphere."""
        if not getattr(self, "_robot_dialog_drag_active", False):
            return
        target = getattr(self, "_robot_dialog_drag_target", None)
        if target is None or len(target) != 3:
            return

        api = getattr(self.parent, "robot_api", None)
        if (
            api is None
            or not hasattr(api, "send_request")
            or not hasattr(api, "set_end_effector_velocity")
            or not hasattr(api, "get_current_tool_position")
        ):
            return

        pose = None
        try:
            pose = api.get_current_tool_position()
        except Exception:
            return
        if not pose or pose[0] is None:
            return
        pos, _quat = pose

        # P-controller in BASE frame with a safety velocity ceiling.
        K = 4.0           # 1/s; err 0.075 m → v ≈ 0.3 m/s, clamped below
        v_max = 0.30      # m/s
        deadband_m = 0.0015

        err = np.array(
            [
                float(target[0]) - float(pos[0]),
                float(target[1]) - float(pos[1]),
                float(target[2]) - float(pos[2]),
            ],
            dtype=float,
        )
        err_norm = float(np.linalg.norm(err))

        if err_norm < deadband_m:
            v = np.zeros(3, dtype=float)
        else:
            v = K * err
            v_norm = float(np.linalg.norm(v))
            if v_norm > v_max:
                v = v * (v_max / v_norm)

        v6 = [float(v[0]), float(v[1]), float(v[2]), 0.0, 0.0, 0.0]
        try:
            if hasattr(api, "send_end_effector_velocity"):
                api.send_end_effector_velocity(v6)
            else:
                api.send_request(api.set_end_effector_velocity(v6))
        except Exception:
            return

        status = getattr(self, "_robot_dialog_status", None)
        if status is not None:
            speed = float(np.linalg.norm(v))
            status.setText(
                "Drag: live → ({0:+.3f}, {1:+.3f}, {2:+.3f}) m  "
                "|v|={3:.2f} m/s  Δ={4:.1f} mm".format(
                    target[0], target[1], target[2], speed, err_norm * 1000.0
                )
            )

    def _apply_robot_dialog_joints(self, joints):
        """Re-pose the dialog meshes to ``joints`` (length-6, radians)."""
        meshes = getattr(self, "_robot_dialog_meshes", None)
        applied = getattr(self, "_robot_dialog_applied", None)
        plotter = getattr(self, "_robot_dialog_plotter", None)
        if not meshes or applied is None or plotter is None:
            return
        if len(joints) < 6:
            return

        target = self._compute_kinematic_chain(list(joints[:6]))
        for idx, M_new in enumerate(target):
            if M_new is None or idx >= len(meshes):
                continue
            M_prev = applied[idx]
            try:
                if M_prev is None:
                    delta = M_new
                else:
                    delta = M_new @ np.linalg.inv(M_prev)
                meshes[idx].transform(delta, inplace=True)
                applied[idx] = M_new
            except Exception as exc:
                print(f"[RobotDialog] live-tick link {idx} failed: {exc}")

        self._robot_dialog_applied = applied
        self._robot_dialog_current_joints = list(joints[:6])
        try:
            plotter.render()
        except Exception:
            pass

    def _robot_dialog_live_tick(self):
        """Timer slot: pull the latest joint angles and re-pose the meshes."""
        dialog = getattr(self, "_robot_dialog", None)
        if dialog is None or not dialog.isVisible():
            timer = getattr(self, "_robot_dialog_live_timer", None)
            if timer is not None:
                timer.stop()
            return

        api = getattr(self.parent, "robot_api", None)
        status_label = getattr(self, "_robot_dialog_status", None)
        if api is None or not hasattr(api, "get_current_positions"):
            if status_label is not None:
                status_label.setText("Robot API not available")
            btn = getattr(self, "_robot_dialog_live_btn", None)
            if btn is not None:
                btn.setChecked(False)
            return

        joints = None
        try:
            joints = api.get_current_positions()
        except Exception as exc:
            if status_label is not None:
                status_label.setText(f"Read failed: {exc}")
            return

        if joints is None or len(joints) < 6:
            if status_label is not None:
                status_label.setText("Waiting for joint feedback…")
            return

        self._apply_robot_dialog_joints(joints)
        if status_label is not None:
            deg = [j * 180.0 / np.pi for j in joints[:6]]
            status_label.setText(
                "Live: J=[{0:+6.1f}, {1:+6.1f}, {2:+6.1f}, {3:+6.1f}, {4:+6.1f}, {5:+6.1f}]°".format(*deg)
            )

    def update_robot_joints(self, new_angles):
        if len(new_angles) != 6:
            print("Error: Expected 6 joint angles.")
            return

        # self.parent.log_display.append(f"Angles: {new_angles}")
        self.joints = new_angles
        self.update_robot_model()  # Call to update the robot's visual models

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

        self.robotModel[1].transform(self.T01 @ self.reT[0], inplace=True)
        self.robotModel[2].transform(self.T01 @ self.T12 @ self.reT[1], inplace=True)
        self.robotModel[3].transform(self.T01 @ self.T12 @ self.T23 @ self.reT[2], inplace=True)
        self.robotModel[4].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.reT[3], inplace=True)
        self.robotModel[5].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.reT[4], inplace=True)
        self.robotModel[6].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56 @ self.reT[5], inplace=True)
        self.robotModel[7].transform(self.T01 @ self.T12 @ self.T23 @ self.T34 @ self.T45 @ self.T56 @ self.reT[6], inplace=True)

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

    def _step_mesh_cache_path(self, step_path):
        step_path = Path(step_path)
        cache_dir = step_path.parent / "generated"
        return cache_dir / f"{step_path.stem}_tessellated.stl"

    def _convert_step_to_mesh_cache(self, step_path, cache_path):
        try:
            import cadquery as cq  # noqa: F401
            from cadquery import exporters, importers
        except Exception as exc:
            raise RuntimeError(
                "CadQuery/OpenCascade is not available. Install cadquery or "
                "provide an STL/OBJ mesh in resource/dexterous_hand."
            ) from exc

        step_path = Path(step_path)
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        shape = importers.importStep(str(step_path))
        exporters.export(
            shape,
            str(cache_path),
            tolerance=0.2,
            angularTolerance=0.2,
        )
        return cache_path

    def _default_dexterous_hand_model_path(self):
        folder = Path(resource_path("dexterous_hand"))
        if not folder.exists():
            return None

        mesh_exts = (
            ".stl",
            ".obj",
            ".ply",
            ".vtp",
            ".vtk",
        )
        for ext in mesh_exts:
            matches = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ext)
            if matches:
                return matches[0]

        step_matches = []
        for ext in (".step", ".stp"):
            step_matches.extend(sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ext))
        if not step_matches:
            return None

        step_path = step_matches[0]
        cache_path = self._step_mesh_cache_path(step_path)
        try:
            if cache_path.exists() and cache_path.stat().st_mtime >= step_path.stat().st_mtime:
                return cache_path
        except Exception:
            pass

        try:
            return self._convert_step_to_mesh_cache(step_path, cache_path)
        except Exception as exc:
            print(f"[DexterousHandModel] STEP tessellation failed, using point preview: {exc}")
            return step_path

        return None

    def _default_rh56f1_urdf_path(self):
        urdf_path = Path(resource_path("dexterous_hand", "rh56f1_right", "urdf", "RH56F1_R.urdf"))
        return urdf_path if urdf_path.exists() else None

    @staticmethod
    def _hand_tactile_region_choices():
        return (
            ("thumb", "Thumb"),
            ("index", "Index"),
            ("middle", "Middle"),
            ("ring", "Ring"),
            ("little", "Little"),
            ("palm_left", "Palm left"),
            ("palm_middle", "Palm middle"),
            ("palm_right", "Palm right"),
        )

    @staticmethod
    def _hand_sensor_adjustment_config_path():
        return Path(resource_path("config", "dexterous_hand_sensor_adjustments.json"))

    @staticmethod
    def _default_hand_sensor_adjustment():
        return {
            "translation_mm": [0.0, 0.0, 0.0],
            "rotation_deg": [0.0, 0.0, 0.0],
        }

    def _load_hand_sensor_adjustments(self):
        path = self._hand_sensor_adjustment_config_path()
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        regions = payload.get("regions") if isinstance(payload, dict) else None
        return regions if isinstance(regions, dict) else {}

    def _save_hand_sensor_adjustments(self, adjustments):
        path = self._hand_sensor_adjustment_config_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "config_version": 1,
                "note": "RH56F1 tactile pad translation/rotation offsets in the centered 3D display frame.",
                "regions": adjustments,
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            return True
        except Exception as exc:
            print(f"[DexterousHandModel] Failed to save sensor adjustments: {exc}")
            return False

    def _coerce_hand_sensor_adjustment(self, value):
        result = self._default_hand_sensor_adjustment()
        if not isinstance(value, dict):
            return result
        for key in ("translation_mm", "rotation_deg"):
            raw_values = value.get(key, result[key])
            try:
                values = [float(v) for v in raw_values[:3]]
            except Exception:
                values = list(result[key])
            if len(values) != 3:
                values = list(result[key])
            result[key] = values
        return result

    def _get_hand_sensor_adjustment(self, region):
        adjustments = self._load_hand_sensor_adjustments()
        return self._coerce_hand_sensor_adjustment(adjustments.get(str(region), {}))

    def _apply_tactile_region_adjustment(self, mesh, region):
        adjustment = self._get_hand_sensor_adjustment(region)
        translation = np.asarray(adjustment["translation_mm"], dtype=float)
        rotation_deg = np.asarray(adjustment["rotation_deg"], dtype=float)
        if np.allclose(translation, 0.0) and np.allclose(rotation_deg, 0.0):
            return mesh.copy(deep=True)

        adjusted = mesh.copy(deep=True)
        points = np.asarray(adjusted.points, dtype=float)
        if points.size == 0:
            return adjusted

        center = points.mean(axis=0)
        rpy = [math.radians(float(v)) for v in rotation_deg]
        rotation = self._matrix_from_xyz_rpy([0.0, 0.0, 0.0], rpy)[:3, :3]
        adjusted.points = (points - center) @ rotation.T + center + translation
        return adjusted

    @staticmethod
    def _matrix_from_xyz_rpy(xyz, rpy):
        x, y, z = xyz
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cr, -sr],
                [0.0, sr, cr],
            ],
            dtype=float,
        )
        ry = np.array(
            [
                [cp, 0.0, sp],
                [0.0, 1.0, 0.0],
                [-sp, 0.0, cp],
            ],
            dtype=float,
        )
        rz = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        mat = np.eye(4, dtype=float)
        mat[:3, :3] = rz @ ry @ rx
        mat[:3, 3] = [x, y, z]
        return mat

    @staticmethod
    def _parse_urdf_origin(node):
        if node is None:
            return np.eye(4, dtype=float)
        try:
            xyz = [float(v) for v in node.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in node.attrib.get("rpy", "0 0 0").split()]
        except Exception:
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
        if len(xyz) != 3:
            xyz = [0.0, 0.0, 0.0]
        if len(rpy) != 3:
            rpy = [0.0, 0.0, 0.0]
        return MyMeshLab._matrix_from_xyz_rpy(xyz, rpy)

    @staticmethod
    def _resolve_rh56f1_mesh_uri(urdf_path, uri):
        mesh_name = str(uri or "").strip()
        if not mesh_name:
            return None
        if mesh_name.startswith("package://RH56F1_R/meshes/"):
            mesh_name = mesh_name.split("package://RH56F1_R/meshes/", 1)[1]
            return Path(urdf_path).parent.parent / "meshes" / mesh_name
        if mesh_name.startswith("package://"):
            mesh_name = mesh_name.rsplit("/", 1)[-1]
            return Path(urdf_path).parent.parent / "meshes" / mesh_name
        path = Path(mesh_name)
        if path.is_absolute():
            return path
        return Path(urdf_path).parent / path

    def _load_rh56f1_urdf_visual_meshes(self, urdf_path):
        root = ET.parse(str(urdf_path)).getroot()
        child_joints = {}
        for joint in root.findall("joint"):
            parent_node = joint.find("parent")
            child_node = joint.find("child")
            if parent_node is None or child_node is None:
                continue
            parent_name = parent_node.attrib.get("link", "")
            child_name = child_node.attrib.get("link", "")
            if not parent_name or not child_name:
                continue
            child_joints.setdefault(parent_name, []).append(
                (child_name, self._parse_urdf_origin(joint.find("origin")))
            )

        link_transforms = {"base_link": np.eye(4, dtype=float)}
        stack = ["base_link"]
        while stack:
            parent_name = stack.pop()
            parent_transform = link_transforms[parent_name]
            for child_name, joint_transform in child_joints.get(parent_name, []):
                link_transforms[child_name] = parent_transform @ joint_transform
                stack.append(child_name)

        visual_meshes = []
        bounds_min = []
        bounds_max = []
        for link in root.findall("link"):
            link_name = link.attrib.get("name", "")
            link_transform = link_transforms.get(link_name)
            if not link_name or link_transform is None:
                continue
            visual = link.find("visual")
            if visual is None:
                continue
            mesh_node = visual.find("./geometry/mesh")
            if mesh_node is None:
                continue
            mesh_path = self._resolve_rh56f1_mesh_uri(
                urdf_path,
                mesh_node.attrib.get("filename", ""),
            )
            if mesh_path is None or not mesh_path.exists():
                continue

            try:
                mesh = pv.read(str(mesh_path)).triangulate()
            except Exception as exc:
                print(f"[DexterousHandModel] Failed to load URDF mesh {mesh_path}: {exc}")
                continue
            if mesh.n_points <= 0:
                continue

            visual_transform = self._parse_urdf_origin(visual.find("origin"))
            transform = link_transform @ visual_transform
            points = np.asarray(mesh.points, dtype=float)
            homogeneous = np.c_[points, np.ones(mesh.n_points, dtype=float)]
            mesh.points = (transform @ homogeneous.T).T[:, :3] * 1000.0

            color = (0.86, 0.88, 0.91, 1.0)
            color_node = visual.find("./material/color")
            if color_node is not None:
                try:
                    rgba = [float(v) for v in color_node.attrib.get("rgba", "").split()]
                    if len(rgba) == 4:
                        color = tuple(rgba)
                except Exception:
                    pass

            bounds = mesh.bounds
            bounds_min.append([bounds[0], bounds[2], bounds[4]])
            bounds_max.append([bounds[1], bounds[3], bounds[5]])
            visual_meshes.append(
                {
                    "link": link_name,
                    "mesh": mesh,
                    "color": color,
                    "is_sensor": link_name in self._RH56F1_SENSOR_LINKS,
                }
            )

        if not visual_meshes:
            raise RuntimeError(f"No visual meshes could be loaded from {urdf_path}")

        bounds_min = np.asarray(bounds_min, dtype=float).min(axis=0)
        bounds_max = np.asarray(bounds_max, dtype=float).max(axis=0)
        center = (bounds_min + bounds_max) * 0.5
        for item in visual_meshes:
            item["mesh"].translate(-center, inplace=True)

        span = float(max(bounds_max - bounds_min))
        return visual_meshes, center, span

    @staticmethod
    def _blend_color(color_a, color_b, t):
        t = max(0.0, min(1.0, float(t)))
        a = np.asarray(color_a, dtype=float)
        b = np.asarray(color_b, dtype=float)
        return tuple((a + (b - a) * t).tolist())

    def _tactile_level_color(self, level):
        level = max(0.0, min(1.0, float(level)))
        idle = (0.10, 0.38, 0.95)
        mid = (1.00, 0.78, 0.18)
        hot = (1.00, 0.12, 0.05)
        if level < 0.5:
            return self._blend_color(idle, mid, level * 2.0)
        return self._blend_color(mid, hot, (level - 0.5) * 2.0)

    @staticmethod
    def _set_actor_visual(actor, color, opacity):
        if actor is None:
            return
        try:
            prop = actor.GetProperty()
            prop.SetColor(*color)
            prop.SetOpacity(float(opacity))
            return
        except Exception:
            pass
        try:
            actor.prop.color = color
            actor.prop.opacity = float(opacity)
        except Exception:
            pass

    @staticmethod
    def _valid_tactile_raw(value):
        try:
            raw = int(value)
        except Exception:
            return None
        if raw < 0 or raw in {65535, 0xFFFF}:
            return None
        return raw

    def _force_raw_to_level(self, raw):
        if raw is None:
            return 0.0
        return max(0.0, min(1.0, raw / self._RH56F1_FORCE_FULL_SCALE))

    @staticmethod
    def _direction_raw_to_degrees(value):
        try:
            raw = int(value)
        except Exception:
            return None
        if raw < 0 or raw in {65535, 0xFFFF}:
            return None
        return float(raw % 360)

    def _extract_tactile_readings(self, data):
        """Map tactile data to live values used by the hand-model overlay.

        Each fingertip reports one capacitive sensor (normal force, raw 1024 =
        10.24 N). The palm module reports three regions, each as a
        (normal, tangential, direction) triple.
        """
        readings = {}
        if not data:
            return readings

        finger_forces = list(data.get("finger_forces") or [])
        finger_tangentials = list(data.get("finger_tangentials") or [])
        finger_angles = list(data.get("finger_angles") or [])
        for index, (region, _link_name) in enumerate(self._RH56F1_FINGER_SENSOR_LINKS):
            if index < len(finger_forces):
                raw = self._valid_tactile_raw(finger_forces[index])
                tangential_raw = (
                    self._valid_tactile_raw(finger_tangentials[index])
                    if index < len(finger_tangentials)
                    else None
                )
                direction_deg = (
                    self._direction_raw_to_degrees(finger_angles[index])
                    if index < len(finger_angles)
                    else None
                )
                readings[region] = {
                    "level": self._force_raw_to_level(raw),
                    "force_n": (raw / 100.0) if raw is not None else None,
                    "tangential_raw": tangential_raw,
                    "tangential_level": self._force_raw_to_level(tangential_raw),
                    "tangential_n": (
                        tangential_raw / 100.0 if tangential_raw is not None else None
                    ),
                    "direction_deg": direction_deg,
                }

        palm_data = list(data.get("palm_data") or [])
        for index, region in enumerate(self._RH56F1_PALM_DATA_REGIONS):
            start = index * 3
            group = palm_data[start:start + 3]
            if not group:
                continue
            raw = self._valid_tactile_raw(group[0])  # normal force only
            tangential_raw = (
                self._valid_tactile_raw(group[1]) if len(group) > 1 else None
            )
            direction_deg = (
                self._direction_raw_to_degrees(group[2]) if len(group) > 2 else None
            )
            readings[region] = {
                "level": self._force_raw_to_level(raw),
                "force_n": (raw / 100.0) if raw is not None else None,
                "tangential_raw": tangential_raw,
                "tangential_level": self._force_raw_to_level(tangential_raw),
                "tangential_n": (
                    tangential_raw / 100.0 if tangential_raw is not None else None
                ),
                "direction_deg": direction_deg,
            }
        return readings

    def _split_palm_sensor_mesh(self, palm_mesh):
        """Split the palm sensor pad into its three reported regions.

        Verified against the official URDF (RH56F1_R): in the display frame the
        palm pad's long axis is Y (span ~61 mm) and the fingertip sensors sit
        at thumb y=+0.092 ... little y=-0.034, i.e. +Y is the thumb side.
        Looking at the right palm from the sensor (palm) side, the thumb side
        is the viewer's right, so +Y -> palm_right, -Y -> palm_left.
        """
        if palm_mesh is None or palm_mesh.n_cells <= 0:
            return []
        try:
            centers = np.asarray(palm_mesh.cell_centers().points, dtype=float)
        except Exception:
            return [("palm_middle", palm_mesh)]
        if len(centers) != palm_mesh.n_cells:
            return [("palm_middle", palm_mesh)]

        y_values = centers[:, 1]
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or abs(y_max - y_min) < 1e-9:
            return [("palm_middle", palm_mesh)]

        cut_1 = y_min + (y_max - y_min) / 3.0
        cut_2 = y_min + (y_max - y_min) * 2.0 / 3.0
        masks = (
            ("palm_left", y_values < cut_1),
            ("palm_middle", (y_values >= cut_1) & (y_values < cut_2)),
            ("palm_right", y_values >= cut_2),
        )
        parts = []
        for region, mask in masks:
            cell_ids = np.flatnonzero(mask)
            if len(cell_ids) == 0:
                continue
            try:
                parts.append((region, palm_mesh.extract_cells(cell_ids)))
            except Exception:
                pass
        return parts or [("palm_middle", palm_mesh)]

    def _make_estimated_fingertip_pad_patch(self, region, sensor_mesh):
        """Create a smaller visual pad from the vendor fingertip sensor mesh.

        The RH56F1 URDF provides fingertip ``*_force_sensor`` meshes, but those
        meshes are effectively the full fingertip shell. For visualization we
        display only a trimmed front patch to represent the likely sensitive
        pad area. This is an estimate, not an official CAD boundary.
        """
        if sensor_mesh is None or sensor_mesh.n_cells <= 0:
            return sensor_mesh

        try:
            centers = np.asarray(sensor_mesh.cell_centers().points, dtype=float)
        except Exception:
            return sensor_mesh
        if len(centers) != sensor_mesh.n_cells:
            return sensor_mesh

        # In the RH56F1 display frame the palm/front contact side is the +X
        # side. Keep only that face, then trim the patch along the fingertip.
        x_values = centers[:, 0]
        front_cut = float(np.quantile(x_values, 0.58))
        front_mask = x_values >= front_cut

        spans = np.ptp(centers, axis=0)
        length_axis = 2 if spans[2] >= spans[1] else 1
        width_axis = 1 if length_axis == 2 else 2

        length_values = centers[:, length_axis]
        width_values = centers[:, width_axis]
        length_lo, length_hi = np.quantile(length_values, [0.22, 0.88])
        width_lo, width_hi = np.quantile(width_values, [0.18, 0.82])
        patch_mask = (
            front_mask
            & (length_values >= length_lo)
            & (length_values <= length_hi)
            & (width_values >= width_lo)
            & (width_values <= width_hi)
        )

        cell_ids = np.flatnonzero(patch_mask)
        min_cells = max(80, int(sensor_mesh.n_cells * 0.015))
        if len(cell_ids) < min_cells:
            # Relax the cut if a specific fingertip geometry is more curved
            # than expected.
            front_cut = float(np.quantile(x_values, 0.50))
            length_lo, length_hi = np.quantile(length_values, [0.16, 0.94])
            width_lo, width_hi = np.quantile(width_values, [0.12, 0.88])
            patch_mask = (
                (x_values >= front_cut)
                & (length_values >= length_lo)
                & (length_values <= length_hi)
                & (width_values >= width_lo)
                & (width_values <= width_hi)
            )
            cell_ids = np.flatnonzero(patch_mask)

        if len(cell_ids) < min_cells:
            print(
                f"[DexterousHandModel] Estimated {region} pad too small; "
                "using full fingertip sensor mesh."
            )
            return sensor_mesh

        try:
            patch = sensor_mesh.extract_cells(cell_ids).extract_surface().triangulate()
        except Exception:
            return sensor_mesh

        try:
            # Lift the patch slightly above the hand body to prevent z-fighting
            # with the fingertip shell mesh.
            patch.points = np.asarray(patch.points, dtype=float) + np.array([0.8, 0.0, 0.0])
        except Exception:
            pass
        return patch

    def _add_rh56f1_urdf_hand_to_plotter(self, plotter, urdf_path):
        visual_meshes, _center, span = self._load_rh56f1_urdf_visual_meshes(urdf_path)
        tactile_actors = {}
        body_count = 0

        for item in visual_meshes:
            link_name = item["link"]
            mesh = item["mesh"]
            if item["is_sensor"]:
                continue
            rgba = item["color"]
            color = tuple(rgba[:3])
            opacity = float(rgba[3]) if len(rgba) >= 4 else 1.0
            if link_name.endswith("_tip"):
                opacity = min(opacity, 0.32)
            try:
                plotter.add_mesh(
                    mesh,
                    color=color,
                    opacity=opacity,
                    show_edges=False,
                    specular=0.28,
                    specular_power=14,
                    smooth_shading=True,
                )
                body_count += 1
            except Exception as exc:
                print(f"[DexterousHandModel] Failed to display {link_name}: {exc}")

        idle_color = self._tactile_level_color(0.0)
        value_text_actors = {}
        pad_frames = {}
        base_meshes = {}
        pad_meshes = {}
        marker_meshes = {}
        for item in visual_meshes:
            link_name = item["link"]
            if not item["is_sensor"]:
                continue
            sensor_regions = []
            if link_name == "plam_force_sensor":
                sensor_regions = self._split_palm_sensor_mesh(item["mesh"])
            else:
                region = link_name.replace("_force_sensor", "")
                sensor_regions = [
                    (region, self._make_estimated_fingertip_pad_patch(region, item["mesh"]))
                ]

            for region, mesh in sensor_regions:
                base_mesh = mesh.copy(deep=True)
                mesh = self._apply_tactile_region_adjustment(base_mesh, region)
                try:
                    actor = plotter.add_mesh(
                        mesh,
                        color=idle_color,
                        opacity=0.32,
                        show_edges=False,
                        specular=0.55,
                        specular_power=20,
                        smooth_shading=True,
                    )
                    tactile_actors[region] = actor
                    base_meshes[region] = base_mesh
                    pad_meshes[region] = mesh
                except Exception as exc:
                    print(f"[DexterousHandModel] Failed to display tactile pad {region}: {exc}")
                    continue

                # Sensor centre marker + live value label, anchored on the
                # official URDF sensor pad geometry (mm display space).
                try:
                    points = np.asarray(mesh.points, dtype=float)
                    center = points.mean(axis=0)
                    normal = self._estimate_sensor_outward_normal(mesh, center)
                    pad_frames[region] = self._make_tactile_pad_frame(mesh, center, normal)
                    marker_mesh = pv.Sphere(radius=1.1, center=center)
                    plotter.add_mesh(
                        marker_mesh,
                        color="#ffffff",
                        opacity=0.9,
                        smooth_shading=True,
                    )
                    marker_meshes[region] = marker_mesh
                    text_actor = self._make_sensor_value_text_actor(
                        center + normal * 7.0
                    )
                    plotter.renderer.AddActor(text_actor)
                    value_text_actors[region] = text_actor
                except Exception as exc:
                    print(f"[DexterousHandModel] Failed to add sensor marker {region}: {exc}")

        self._hand_tactile_pad_actors = tactile_actors
        self._hand_tactile_value_text_actors = value_text_actors
        self._hand_tactile_pad_frames = pad_frames
        self._hand_tactile_base_meshes = base_meshes
        self._hand_tactile_pad_meshes = pad_meshes
        self._hand_tactile_marker_meshes = marker_meshes
        return body_count, len(tactile_actors), span

    @staticmethod
    def _estimate_sensor_outward_normal(mesh, center):
        """Average the pad's face normals to find its outward direction."""
        try:
            normals = np.asarray(
                mesh.compute_normals(
                    cell_normals=True,
                    point_normals=False,
                    auto_orient_normals=True,
                )["Normals"],
                dtype=float,
            )
            mean_normal = normals.mean(axis=0)
            norm = float(np.linalg.norm(mean_normal))
            if norm > 1e-9:
                direction = mean_normal / norm
                # Pads face away from the hand interior (origin side).
                if float(np.dot(direction, center)) < 0.0:
                    direction = -direction
                return direction
        except Exception:
            pass
        norm = float(np.linalg.norm(center))
        if norm > 1e-9:
            return np.asarray(center, dtype=float) / norm
        return np.array([0.0, 0.0, 1.0])

    @staticmethod
    def _normalize_vector(vector, fallback):
        arr = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(arr))
        if norm > 1e-9:
            return arr / norm
        return np.asarray(fallback, dtype=float)

    def _make_tactile_pad_frame(self, mesh, center, normal):
        center = np.asarray(center, dtype=float)
        normal = self._normalize_vector(normal, (0.0, 0.0, 1.0))
        fallback_axis = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(normal, fallback_axis))) > 0.92:
            fallback_axis = np.array([1.0, 0.0, 0.0])

        tangent_x = fallback_axis - normal * float(np.dot(fallback_axis, normal))
        try:
            points = np.asarray(mesh.points, dtype=float)
            offsets = points - center
            planar_offsets = offsets - np.outer(offsets @ normal, normal)
            if len(planar_offsets) >= 3 and float(np.max(np.linalg.norm(planar_offsets, axis=1))) > 1e-9:
                _u, _s, vh = np.linalg.svd(planar_offsets, full_matrices=False)
                candidate = vh[0]
                candidate = candidate - normal * float(np.dot(candidate, normal))
                if float(np.linalg.norm(candidate)) > 1e-9:
                    tangent_x = candidate
            spans = np.ptp(points, axis=0) if len(points) else np.array([8.0, 8.0, 8.0])
            pad_span = max(float(np.max(spans)), 8.0)
        except Exception:
            pad_span = 12.0

        tangent_x = self._normalize_vector(tangent_x, fallback_axis)
        dominant_axis = int(np.argmax(np.abs(tangent_x)))
        if tangent_x[dominant_axis] < 0:
            tangent_x = -tangent_x
        tangent_y = self._normalize_vector(np.cross(normal, tangent_x), (0.0, 1.0, 0.0))
        tangent_x = self._normalize_vector(np.cross(tangent_y, normal), tangent_x)

        return {
            "center": center,
            "normal": normal,
            "tangent_x": tangent_x,
            "tangent_y": tangent_y,
            "arrow_offset": max(5.0, min(14.0, pad_span * 0.14)),
            "arrow_scale": max(11.0, min(30.0, pad_span * 0.85)),
        }

    @staticmethod
    def _make_tactile_direction_arrow_mesh(start, direction, scale):
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / norm
        return pv.Arrow(
            start=tuple(np.asarray(start, dtype=float)),
            direction=tuple(direction),
            tip_length=0.34,
            tip_radius=0.145,
            shaft_radius=0.055,
            shaft_resolution=16,
            tip_resolution=24,
            scale=float(scale),
        )

    @staticmethod
    def _set_actor_visibility(actor, visible):
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

    def _get_hand_tactile_overlay_renderer(self, plotter):
        overlay = getattr(self, "_hand_tactile_overlay_renderer", None)
        if overlay is not None:
            return overlay
        if plotter is None:
            return None
        try:
            import vtk
            render_window = getattr(plotter, "ren_win", None) or getattr(plotter, "render_window", None)
            main_renderer = getattr(plotter, "renderer", None)
            if render_window is None or main_renderer is None:
                return None
            render_window.SetNumberOfLayers(max(int(render_window.GetNumberOfLayers()), 2))
            overlay = vtk.vtkRenderer()
            overlay.SetLayer(1)
            overlay.SetActiveCamera(main_renderer.GetActiveCamera())
            overlay.SetPreserveColorBuffer(True)
            overlay.SetPreserveDepthBuffer(False)
            overlay.SetInteractive(False)
            try:
                overlay.SetBackgroundAlpha(0.0)
            except Exception:
                pass
            render_window.AddRenderer(overlay)
            self._hand_tactile_overlay_renderer = overlay
            return overlay
        except Exception:
            return None

    @staticmethod
    def _style_hand_tactile_direction_actor(actor):
        if actor is None:
            return
        try:
            actor.UseBoundsOff()
        except Exception:
            pass
        try:
            actor.ForceOpaqueOn()
        except Exception:
            pass
        try:
            prop = actor.GetProperty()
            prop.SetColor(0.125, 0.843, 1.0)
            prop.SetOpacity(1.0)
            prop.SetAmbient(0.75)
            prop.SetDiffuse(0.7)
            prop.SetSpecular(0.25)
            prop.SetSpecularPower(10)
        except Exception:
            pass
        try:
            mapper = actor.GetMapper()
            mapper.SetResolveCoincidentTopologyToPolygonOffset()
            mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-4.0, -66000.0)
        except Exception:
            pass

    def _add_hand_tactile_direction_actor(self, plotter, mesh):
        try:
            import vtk
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self._style_hand_tactile_direction_actor(actor)
            overlay = self._get_hand_tactile_overlay_renderer(plotter)
            if overlay is not None:
                overlay.AddActor(actor)
                return actor
        except Exception:
            pass

        actor = plotter.add_mesh(
            mesh,
            color="#20d7ff",
            opacity=1.0,
            show_edges=False,
            specular=0.35,
            specular_power=16,
            smooth_shading=True,
            reset_camera=False,
        )
        self._style_hand_tactile_direction_actor(actor)
        return actor

    def _remove_hand_tactile_direction_actor(self, plotter, actor):
        if actor is None:
            return
        overlay = getattr(self, "_hand_tactile_overlay_renderer", None)
        if overlay is not None:
            try:
                overlay.RemoveActor(actor)
                return
            except Exception:
                pass
        if plotter is not None:
            try:
                plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass

    def _update_hand_tactile_direction_vector(self, plotter, region, reading):
        if plotter is None:
            return
        frames = getattr(self, "_hand_tactile_pad_frames", None) or {}
        frame = frames.get(region)
        if frame is None:
            return

        direction_deg = reading.get("direction_deg")
        tangential_n = reading.get("tangential_n")
        force_n = reading.get("force_n")
        tangential_level = float(reading.get("tangential_level") or 0.0)
        arrow_actors = getattr(self, "_hand_tactile_direction_actors", None)
        if arrow_actors is None:
            arrow_actors = {}
            self._hand_tactile_direction_actors = arrow_actors
        arrow_meshes = getattr(self, "_hand_tactile_direction_meshes", None)
        if arrow_meshes is None:
            arrow_meshes = {}
            self._hand_tactile_direction_meshes = arrow_meshes

        actor = arrow_actors.get(region)
        has_contact = force_n is not None and float(force_n) > 0.0
        has_shear = tangential_n is not None and float(tangential_n) > 0.0
        if direction_deg is None or not (has_contact or has_shear):
            self._set_actor_visibility(actor, False)
            return

        angle_rad = math.radians(float(direction_deg))
        direction = (
            math.cos(angle_rad) * np.asarray(frame["tangent_x"], dtype=float)
            + math.sin(angle_rad) * np.asarray(frame["tangent_y"], dtype=float)
        )
        start = (
            np.asarray(frame["center"], dtype=float)
            + np.asarray(frame["normal"], dtype=float) * float(frame["arrow_offset"])
        )
        arrow_scale = float(frame["arrow_scale"]) * (0.45 + 0.75 * min(1.0, tangential_level))
        new_mesh = self._make_tactile_direction_arrow_mesh(start, direction, arrow_scale)

        existing_mesh = arrow_meshes.get(region)
        if existing_mesh is not None and actor is not None:
            try:
                existing_mesh.copy_from(new_mesh)
                self._style_hand_tactile_direction_actor(actor)
                self._set_actor_visibility(actor, True)
                return
            except Exception:
                self._remove_hand_tactile_direction_actor(plotter, actor)
                arrow_actors.pop(region, None)
                arrow_meshes.pop(region, None)

        try:
            arrow_meshes[region] = new_mesh
            actor = self._add_hand_tactile_direction_actor(plotter, new_mesh)
            arrow_actors[region] = actor
        except Exception as exc:
            print(f"[DexterousHandModel] Failed to display direction vector {region}: {exc}")

    def _refresh_hand_tactile_region_adjustment(self, plotter, region):
        if plotter is None:
            return False
        base_meshes = getattr(self, "_hand_tactile_base_meshes", None) or {}
        pad_meshes = getattr(self, "_hand_tactile_pad_meshes", None) or {}
        marker_meshes = getattr(self, "_hand_tactile_marker_meshes", None) or {}
        actors = getattr(self, "_hand_tactile_pad_actors", None) or {}
        region = str(region)
        base_mesh = base_meshes.get(region)
        if base_mesh is None:
            return False

        adjusted_mesh = self._apply_tactile_region_adjustment(base_mesh, region)
        displayed_mesh = pad_meshes.get(region)
        if displayed_mesh is not None:
            try:
                displayed_mesh.copy_from(adjusted_mesh)
            except Exception:
                old_actor = actors.get(region)
                if old_actor is not None:
                    try:
                        plotter.remove_actor(old_actor, reset_camera=False)
                    except Exception:
                        pass
                try:
                    actors[region] = plotter.add_mesh(
                        adjusted_mesh,
                        color=self._tactile_level_color(0.0),
                        opacity=0.32,
                        show_edges=False,
                        specular=0.55,
                        specular_power=20,
                        smooth_shading=True,
                        reset_camera=False,
                    )
                    pad_meshes[region] = adjusted_mesh
                    displayed_mesh = adjusted_mesh
                except Exception:
                    return False
        else:
            pad_meshes[region] = adjusted_mesh
            displayed_mesh = adjusted_mesh

        try:
            points = np.asarray(displayed_mesh.points, dtype=float)
            center = points.mean(axis=0)
            normal = self._estimate_sensor_outward_normal(displayed_mesh, center)
            self._hand_tactile_pad_frames[region] = self._make_tactile_pad_frame(
                displayed_mesh,
                center,
                normal,
            )

            marker_mesh = marker_meshes.get(region)
            if marker_mesh is not None:
                marker_mesh.copy_from(pv.Sphere(radius=1.1, center=center))

            text_actor = (getattr(self, "_hand_tactile_value_text_actors", None) or {}).get(region)
            if text_actor is not None:
                text_actor.SetPosition(*[float(v) for v in center + normal * 7.0])
        except Exception:
            pass

        arrow_actors = getattr(self, "_hand_tactile_direction_actors", None) or {}
        arrow_meshes = getattr(self, "_hand_tactile_direction_meshes", None) or {}
        arrow_actor = arrow_actors.pop(region, None)
        arrow_meshes.pop(region, None)
        if arrow_actor is not None:
            self._remove_hand_tactile_direction_actor(plotter, arrow_actor)

        try:
            plotter.render()
        except Exception:
            pass
        self._apply_latest_dexterous_hand_tactile()
        return True

    @staticmethod
    def _make_sensor_value_text_actor(position):
        import vtk

        actor = vtk.vtkBillboardTextActor3D()
        actor.SetInput("--")
        actor.SetPosition(*[float(v) for v in position])
        prop = actor.GetTextProperty()
        prop.SetFontSize(15)
        prop.SetColor(1.0, 1.0, 1.0)
        prop.SetBackgroundColor(0.08, 0.08, 0.10)
        prop.SetBackgroundOpacity(0.55)
        prop.SetJustificationToCentered()
        return actor

    def _direct_finger_motion_active(self):
        parent = getattr(self, "parent", None)
        if parent is not None and bool(getattr(parent, "_direct_finger_active", False)):
            return True

        sensor_functions = getattr(parent, "sensor_functions", None)
        direct_helper = getattr(sensor_functions, "direct_finger_motion_class", None)
        return bool(getattr(direct_helper, "is_running", False))

    def _render_hand_tactile_plotter(self, plotter):
        if plotter is None:
            return

        self._hand_tactile_update_count = int(getattr(self, "_hand_tactile_update_count", 0)) + 1

        # DFM v1 is driven by a Qt timer on the same process. VTK renders from
        # the hand model window can steal enough event-loop time to make robot
        # control feel slow/stuttery, so render the tactile overlay less often
        # while DFM is active.
        min_interval = 0.35 if self._direct_finger_motion_active() else 0.08
        now = time.perf_counter()
        last_render = float(getattr(self, "_hand_tactile_last_render_at", 0.0))
        if now - last_render < min_interval:
            self._hand_tactile_render_skip_count = (
                int(getattr(self, "_hand_tactile_render_skip_count", 0)) + 1
            )
            return

        self._hand_tactile_last_render_at = now
        started = time.perf_counter()
        try:
            plotter.render()
        except Exception:
            pass
        elapsed = time.perf_counter() - started
        self._hand_tactile_render_count = int(getattr(self, "_hand_tactile_render_count", 0)) + 1
        self._hand_tactile_last_render_sec = float(elapsed)
        self._hand_tactile_render_total_sec = (
            float(getattr(self, "_hand_tactile_render_total_sec", 0.0)) + float(elapsed)
        )

    def hand_tactile_runtime_report(self):
        plotter_open = getattr(self, "_hand_model_dialog_plotter", None) is not None
        actors = getattr(self, "_hand_tactile_pad_actors", None) or {}
        updates = int(getattr(self, "_hand_tactile_update_count", 0))
        renders = int(getattr(self, "_hand_tactile_render_count", 0))
        skips = int(getattr(self, "_hand_tactile_render_skip_count", 0))
        total_render_sec = float(getattr(self, "_hand_tactile_render_total_sec", 0.0))
        last_render_ms = float(getattr(self, "_hand_tactile_last_render_sec", 0.0)) * 1000.0
        started_at = float(getattr(self, "_hand_tactile_report_started_at", time.perf_counter()))
        elapsed = max(1e-6, time.perf_counter() - started_at)
        avg_render_ms = (total_render_sec / renders * 1000.0) if renders else 0.0
        return (
            f"hand_3d_open: {plotter_open}\n"
            f"hand_3d_tactile_regions: {len(actors)}\n"
            f"hand_3d_tactile_update_hz: {updates / elapsed:.2f}\n"
            f"hand_3d_render_hz: {renders / elapsed:.2f}\n"
            f"hand_3d_render_skipped: {skips}\n"
            f"hand_3d_last_render_ms: {last_render_ms:.2f}\n"
            f"hand_3d_avg_render_ms: {avg_render_ms:.2f}"
        )

    def updateDexterousHandTactile(self, data=None):
        actors = getattr(self, "_hand_tactile_pad_actors", None)
        if not actors:
            return False

        readings = self._extract_tactile_readings(data)
        text_actors = getattr(self, "_hand_tactile_value_text_actors", None) or {}
        plotter = getattr(self, "_hand_model_dialog_plotter", None)
        max_force = None
        max_tangential = None
        max_level = 0.0
        for region, actor in actors.items():
            reading = readings.get(region) or {}
            level = float(reading.get("level") or 0.0)
            force_n = reading.get("force_n")
            tangential_n = reading.get("tangential_n")
            max_level = max(max_level, level)
            if force_n is not None and (max_force is None or force_n > max_force):
                max_force = force_n
            if (
                tangential_n is not None
                and (max_tangential is None or tangential_n > max_tangential)
            ):
                max_tangential = tangential_n
            color = self._tactile_level_color(level)
            opacity = 0.24 + 0.34 * max(0.0, min(1.0, level))
            self._set_actor_visual(actor, color, opacity)
            self._update_hand_tactile_direction_vector(plotter, region, reading)

            text_actor = text_actors.get(region)
            if text_actor is not None:
                try:
                    if force_n is None:
                        text_actor.SetInput("--")
                    else:
                        text_actor.SetInput(f"{force_n:.2f}N")
                except Exception:
                    pass

        label = getattr(self, "_hand_tactile_overlay_label", None)
        if label is not None:
            if data and max_force is not None:
                if max_tangential is not None and max_tangential > 0.0:
                    label.setText(
                        f"Tactile overlay: peak {max_force:.2f} N | shear {max_tangential:.2f} N"
                    )
                else:
                    label.setText(f"Tactile overlay: peak {max_force:.2f} N")
            elif data:
                label.setText("Tactile overlay: no contact")
            else:
                label.setText("Tactile overlay: idle")

        self._render_hand_tactile_plotter(plotter)
        return True

    def _apply_latest_dexterous_hand_tactile(self):
        api = getattr(getattr(self, "parent", None), "robot_api", None)
        if api is None or not hasattr(api, "get_latest_hand_tactile"):
            return self.updateDexterousHandTactile(None)
        try:
            data = api.get_latest_hand_tactile()
        except Exception:
            data = None
        return self.updateDexterousHandTactile(data)

    def _load_step_point_cloud(self, step_path, max_points=220000):
        """Create a visual preview from STEP CARTESIAN_POINT entries.

        This is a fallback for systems without a CAD kernel. It does not
        reconstruct B-rep faces, but it gives a useful geometry preview from
        the points already stored in the STEP file.
        """
        points = []
        with open(step_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if "CARTESIAN_POINT" not in line:
                    continue
                match = self._STEP_POINT_RE.search(line)
                if not match:
                    continue
                try:
                    points.append([float(match.group(i)) for i in range(1, 4)])
                except Exception:
                    continue

        if not points:
            raise RuntimeError(f"No CARTESIAN_POINT entries found in {step_path}")

        original_count = len(points)
        point_array = np.asarray(points, dtype=float)
        if len(point_array) > int(max_points):
            sample_idx = np.linspace(
                0,
                len(point_array) - 1,
                int(max_points),
                dtype=int,
            )
            point_array = point_array[sample_idx]

        center = (point_array.min(axis=0) + point_array.max(axis=0)) * 0.5
        point_array = point_array - center
        cloud = pv.PolyData(point_array)
        return cloud, original_count, len(point_array)

    def _add_hand_dialog_axes(self, plotter, length):
        return None

    @staticmethod
    def _reset_hand_dialog_camera(plotter, zoom=1.65):
        try:
            plotter.reset_camera()
        except Exception:
            return
        try:
            plotter.camera.Zoom(float(zoom))
        except Exception:
            try:
                plotter.camera.zoom(float(zoom))
            except Exception:
                pass
        try:
            plotter.reset_camera_clipping_range()
        except Exception:
            pass

    def addDexterousHandInDialog(self):
        existing = getattr(self, "_hand_model_dialog", None)
        existing_plotter = getattr(self, "_hand_model_dialog_plotter", None)
        if existing is not None and existing_plotter is not None:
            try:
                if existing.isVisible() or not existing.testAttribute(Qt.WA_DeleteOnClose):
                    existing.show()
                    existing.raise_()
                    existing.activateWindow()
                    self._apply_latest_dexterous_hand_tactile()
                    self._poke_main_plotters()
                    return True
            except Exception:
                pass

        urdf_path = self._default_rh56f1_urdf_path()
        model_path = None if urdf_path is not None else self._default_dexterous_hand_model_path()
        if urdf_path is None and model_path is None:
            QMessageBox.warning(
                self.parent,
                "Dexterous Hand Model",
                "No STL/OBJ/STEP hand model was found in resource/dexterous_hand.",
            )
            return False

        try:
            QApplication.processEvents()
        except Exception:
            pass
        self._poke_main_plotters()

        dialog = QDialog(getattr(self, "parent", None))
        source_name = urdf_path.name if urdf_path is not None else model_path.name
        dialog.setWindowTitle(f"Dexterous Hand Model - {source_name}")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.Window)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.resize(900, 760)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 6, 8, 6)
        status_label = QLabel("Loading hand model...")
        toolbar.addWidget(status_label, 1)

        tactile_status_label = QLabel("Tactile overlay: idle")
        tactile_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        toolbar.addWidget(tactile_status_label)

        reset_button = QPushButton("Reset View")
        toolbar.addWidget(reset_button)
        layout.addLayout(toolbar)

        adjust_layout = QHBoxLayout()
        adjust_layout.setContentsMargins(8, 0, 8, 6)
        adjust_layout.setSpacing(6)
        adjust_layout.addWidget(QLabel("Adjust"))

        adjust_region_combo = QComboBox()
        for region_key, region_label in self._hand_tactile_region_choices():
            adjust_region_combo.addItem(region_label, region_key)
        adjust_layout.addWidget(adjust_region_combo)

        def _make_adjust_spin(suffix, decimals, step, minimum, maximum):
            spin = QDoubleSpinBox()
            spin.setRange(float(minimum), float(maximum))
            spin.setDecimals(int(decimals))
            spin.setSingleStep(float(step))
            spin.setSuffix(suffix)
            spin.setFixedWidth(90)
            return spin

        adjust_layout.addWidget(QLabel("X"))
        adjust_x_spin = _make_adjust_spin(" mm", 2, 0.5, -100.0, 100.0)
        adjust_layout.addWidget(adjust_x_spin)
        adjust_layout.addWidget(QLabel("Y"))
        adjust_y_spin = _make_adjust_spin(" mm", 2, 0.5, -100.0, 100.0)
        adjust_layout.addWidget(adjust_y_spin)
        adjust_layout.addWidget(QLabel("Z"))
        adjust_z_spin = _make_adjust_spin(" mm", 2, 0.5, -100.0, 100.0)
        adjust_layout.addWidget(adjust_z_spin)

        adjust_layout.addWidget(QLabel("RX"))
        adjust_rx_spin = _make_adjust_spin(" deg", 1, 1.0, -180.0, 180.0)
        adjust_layout.addWidget(adjust_rx_spin)
        adjust_layout.addWidget(QLabel("RY"))
        adjust_ry_spin = _make_adjust_spin(" deg", 1, 1.0, -180.0, 180.0)
        adjust_layout.addWidget(adjust_ry_spin)
        adjust_layout.addWidget(QLabel("RZ"))
        adjust_rz_spin = _make_adjust_spin(" deg", 1, 1.0, -180.0, 180.0)
        adjust_layout.addWidget(adjust_rz_spin)

        apply_adjust_button = QPushButton("Apply")
        reset_adjust_button = QPushButton("Reset")
        adjust_layout.addWidget(apply_adjust_button)
        adjust_layout.addWidget(reset_adjust_button)
        adjust_layout.addStretch(1)
        layout.addLayout(adjust_layout)

        plotter = QtInteractor(dialog)
        layout.addWidget(plotter.interactor)
        try:
            plotter.set_background("#202020")
        except Exception:
            pass

        loaded_actor = None
        try:
            self._hand_tactile_pad_actors = {}
            self._hand_tactile_value_text_actors = {}
            self._hand_tactile_pad_frames = {}
            self._hand_tactile_direction_actors = {}
            self._hand_tactile_direction_meshes = {}
            self._hand_tactile_base_meshes = {}
            self._hand_tactile_pad_meshes = {}
            self._hand_tactile_marker_meshes = {}
            self._hand_tactile_overlay_renderer = None
            self._hand_tactile_overlay_label = tactile_status_label

            if urdf_path is not None:
                body_count, pad_count, span = self._add_rh56f1_urdf_hand_to_plotter(
                    plotter,
                    urdf_path,
                )
                self._add_hand_dialog_axes(plotter, span * 0.6)
                status_label.setText(
                    f"{urdf_path.name} | {body_count} hand parts | {pad_count} tactile pads"
                )
                loaded_actor = True
            elif model_path.suffix.lower() in {".step", ".stp"}:
                cloud, original_count, shown_count = self._load_step_point_cloud(model_path)
                loaded_actor = plotter.add_mesh(
                    cloud,
                    color="#d8dee9",
                    style="points",
                    point_size=2.2,
                    render_points_as_spheres=True,
                    opacity=0.95,
                )
                bounds = cloud.bounds
                span = max(
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4],
                )
                self._add_hand_dialog_axes(plotter, span * 0.6)
                status_label.setText(
                    f"{model_path.name} | STEP point preview | {shown_count:,}/{original_count:,} points"
                )
            else:
                mesh = pv.read(str(model_path))
                center = np.asarray(mesh.center, dtype=float)
                mesh.translate(-center, inplace=True)
                loaded_actor = plotter.add_mesh(
                    mesh,
                    color="#d8dee9",
                    show_edges=False,
                    specular=0.45,
                    specular_power=15,
                    smooth_shading=True,
                )
                bounds = mesh.bounds
                span = max(
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4],
                )
                self._add_hand_dialog_axes(plotter, span * 0.6)
                status_label.setText(f"{model_path.name} | mesh loaded")

            self._reset_hand_dialog_camera(plotter)
            plotter.render()
        except Exception as exc:
            QMessageBox.warning(
                self.parent,
                "Dexterous Hand Model",
                f"Failed to load {source_name}:\n{exc}",
            )
            try:
                dialog.close()
            except Exception:
                pass
            return False

        def _reset_view():
            try:
                self._reset_hand_dialog_camera(plotter)
                plotter.render()
            except Exception:
                pass

        reset_button.clicked.connect(_reset_view)

        adjust_spins = (
            adjust_x_spin,
            adjust_y_spin,
            adjust_z_spin,
            adjust_rx_spin,
            adjust_ry_spin,
            adjust_rz_spin,
        )

        def _selected_adjust_region():
            return str(adjust_region_combo.currentData() or "thumb")

        def _set_adjust_spins(region):
            adjustment = self._get_hand_sensor_adjustment(region)
            values = list(adjustment["translation_mm"]) + list(adjustment["rotation_deg"])
            for spin, value in zip(adjust_spins, values):
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)

        def _current_adjustment_from_spins():
            return {
                "translation_mm": [
                    float(adjust_x_spin.value()),
                    float(adjust_y_spin.value()),
                    float(adjust_z_spin.value()),
                ],
                "rotation_deg": [
                    float(adjust_rx_spin.value()),
                    float(adjust_ry_spin.value()),
                    float(adjust_rz_spin.value()),
                ],
            }

        def _write_adjustment(region, adjustment):
            adjustments = self._load_hand_sensor_adjustments()
            translation = np.asarray(adjustment["translation_mm"], dtype=float)
            rotation = np.asarray(adjustment["rotation_deg"], dtype=float)
            if np.allclose(translation, 0.0) and np.allclose(rotation, 0.0):
                adjustments.pop(region, None)
            else:
                adjustments[region] = adjustment
            return self._save_hand_sensor_adjustments(adjustments)

        def _apply_region_adjustment():
            region = _selected_adjust_region()
            adjustment = _current_adjustment_from_spins()
            if not _write_adjustment(region, adjustment):
                QMessageBox.warning(self.parent, "Sensor Adjustment", "Failed to save adjustment.")
                return
            if self._refresh_hand_tactile_region_adjustment(plotter, region):
                status_label.setText(f"Applied {region} sensor adjustment")
            else:
                status_label.setText(f"{region} sensor adjustment saved")

        def _reset_region_adjustment():
            region = _selected_adjust_region()
            default_adjustment = self._default_hand_sensor_adjustment()
            _write_adjustment(region, default_adjustment)
            _set_adjust_spins(region)
            if self._refresh_hand_tactile_region_adjustment(plotter, region):
                status_label.setText(f"Reset {region} sensor adjustment")
            else:
                status_label.setText(f"{region} sensor adjustment reset")

        adjust_region_combo.currentIndexChanged.connect(
            lambda _index: _set_adjust_spins(_selected_adjust_region())
        )
        apply_adjust_button.clicked.connect(_apply_region_adjustment)
        reset_adjust_button.clicked.connect(_reset_region_adjustment)
        _set_adjust_spins(_selected_adjust_region())

        def _on_finished(_result):
            dlg_plotter = getattr(self, "_hand_model_dialog_plotter", None)
            if dlg_plotter is not None:
                try:
                    dlg_plotter.close()
                except Exception:
                    pass
                try:
                    iren = getattr(dlg_plotter, "interactor", None)
                    if iren is not None:
                        rw = iren.GetRenderWindow()
                        if rw is not None:
                            rw.Finalize()
                        iren.TerminateApp()
                except Exception:
                    pass
            self._hand_model_dialog = None
            self._hand_model_dialog_plotter = None
            self._hand_model_actor = None
            self._hand_tactile_pad_actors = {}
            self._hand_tactile_value_text_actors = {}
            self._hand_tactile_pad_frames = {}
            self._hand_tactile_direction_actors = {}
            self._hand_tactile_direction_meshes = {}
            self._hand_tactile_base_meshes = {}
            self._hand_tactile_pad_meshes = {}
            self._hand_tactile_marker_meshes = {}
            self._hand_tactile_overlay_renderer = None
            self._hand_tactile_overlay_label = None
            self._poke_main_plotters()

        dialog.finished.connect(_on_finished)

        self._hand_model_dialog = dialog
        self._hand_model_dialog_plotter = plotter
        self._hand_model_actor = loaded_actor
        self._hand_tactile_overlay_label = tactile_status_label
        dialog.show()
        self._apply_latest_dexterous_hand_tactile()
        try:
            plotter.render()
        except Exception:
            pass
        return True

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

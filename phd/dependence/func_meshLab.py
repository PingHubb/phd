import pyvista as pv
import numpy as np
from pyvistaqt import QtInteractor
import os
import re
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
        return None

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
        length = max(float(length), 1.0)
        try:
            plotter.add_mesh(pv.Line((-length, 0, 0), (length, 0, 0)), color="#ff5050", line_width=2)
            plotter.add_mesh(pv.Line((0, -length, 0), (0, length, 0)), color="#50ff50", line_width=2)
            plotter.add_mesh(pv.Line((0, 0, -length), (0, 0, length)), color="#5080ff", line_width=2)
            plotter.add_axes(interactive=False)
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
                    self._poke_main_plotters()
                    return True
            except Exception:
                pass

        model_path = self._default_dexterous_hand_model_path()
        if model_path is None:
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
        dialog.setWindowTitle(f"Dexterous Hand Model - {model_path.name}")
        dialog.setWindowFlags(dialog.windowFlags() | Qt.Window)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.resize(900, 760)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 6, 8, 6)
        status_label = QLabel("Loading hand model...")
        toolbar.addWidget(status_label)
        toolbar.addStretch(1)

        reset_button = QPushButton("Reset View")
        toolbar.addWidget(reset_button)
        layout.addLayout(toolbar)

        plotter = QtInteractor(dialog)
        layout.addWidget(plotter.interactor)
        try:
            plotter.set_background("#202020")
        except Exception:
            pass

        loaded_actor = None
        suffix = model_path.suffix.lower()
        try:
            if suffix in {".step", ".stp"}:
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

            plotter.reset_camera()
            plotter.render()
        except Exception as exc:
            QMessageBox.warning(
                self.parent,
                "Dexterous Hand Model",
                f"Failed to load {model_path}:\n{exc}",
            )
            try:
                dialog.close()
            except Exception:
                pass
            return False

        def _reset_view():
            try:
                plotter.reset_camera()
                plotter.render()
            except Exception:
                pass

        reset_button.clicked.connect(_reset_view)

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
            self._poke_main_plotters()

        dialog.finished.connect(_on_finished)

        self._hand_model_dialog = dialog
        self._hand_model_dialog_plotter = plotter
        self._hand_model_actor = loaded_actor
        dialog.show()
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

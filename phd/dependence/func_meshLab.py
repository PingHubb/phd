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
    QFormLayout,
    QCheckBox,
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
        self.referenceAxisActors = []
        self.referenceAxisActors_2 = []
        self.actorPlaneXY = None
        self.actorPlaneXY_2 = None
        self.show_secondary_background_reference = True
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
        self._robot_dialog_admittance_active = False
        self._robot_dialog_admittance_confirmed = False
        self._robot_dialog_admittance_velocity_mode_on = False
        self._robot_dialog_admittance_filtered_velocity = np.zeros(3, dtype=float)
        self._admittance_source = None
        self._admittance_pending_start = False
        self._admittance_mapping_config = None
        self._robot_dialog_admittance_timer = QTimer(self.parent)
        self._robot_dialog_admittance_timer.setInterval(33)
        self._robot_dialog_admittance_timer.timeout.connect(
            self._robot_dialog_admittance_tick
        )

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
        self.referenceAxisActors = []
        self.referenceAxisActors_2 = []
        line = pv.Line((-50, 0, 0), (50, 0, 0))

        # 添加X轴线段，并设置为红色
        self.referenceAxisActors.append(
            self.plotter.add_mesh(line, color='r', line_width=2, label='X Axis')
        )
        self.referenceAxisActors_2.append(
            self.plotter_2.add_mesh(line, color='r', line_width=2, label='X Axis')
        )

        line = pv.Line((0, -50, 0), (0, 50, 0))

        # 添加Y轴线段，并设置为绿色
        self.referenceAxisActors.append(
            self.plotter.add_mesh(line, color='g', line_width=2, label='Y Axis')
        )
        self.referenceAxisActors_2.append(
            self.plotter_2.add_mesh(line, color='g', line_width=2, label='Y Axis')
        )

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
        self._apply_secondary_background_reference_visibility(render=False)

    @staticmethod
    def _set_actor_visible(actor, visible):
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

    def _apply_secondary_background_reference_visibility(self, render=True):
        visible = bool(getattr(self, "show_secondary_background_reference", True))
        for actor in getattr(self, "referenceAxisActors_2", []) or []:
            self._set_actor_visible(actor, visible)
        self._set_actor_visible(getattr(self, "actorPlaneXY_2", None), visible)
        if render:
            try:
                self.plotter_2.render()
            except Exception:
                pass

    def set_secondary_background_reference_enabled(self, enabled, render=True):
        self.show_secondary_background_reference = bool(enabled)
        self._apply_secondary_background_reference_visibility(render=render)

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
        periodically and re-poses the displayed links. It starts from the
        latest available joint feedback and enables this read-only live follow
        automatically, so the model and Link 5 frame represent the real pose.
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

        btn_robot_sensor_stream = QPushButton("Start Sensor")
        btn_robot_sensor_stream.setCheckable(True)
        btn_robot_sensor_stream.setToolTip(
            "Build, calibrate and read the selected sensor for this robot viewer only.\n"
            "The same single serial reader is used; main-UI sensor rendering is disabled."
        )
        btn_robot_sensor_stream.setStyleSheet(
            "QPushButton{padding:4px 12px;}"
            "QPushButton:checked{background-color:#356aa0;color:white;font-weight:bold;}"
        )
        toolbar.addWidget(btn_robot_sensor_stream)

        btn_map_sensor = QPushButton("Map Sensor")
        btn_map_sensor.setCheckable(True)
        btn_map_sensor.setToolTip(
            "Overlay the current built sensor, or preview the selected sensor model, on a robot link.\n"
            "Use 'Sensor Mount...' to tune and save the link-to-sensor transform."
        )
        btn_map_sensor.setStyleSheet(
            "QPushButton{padding:4px 12px;}"
            "QPushButton:checked{background-color:#2f8f72;color:white;font-weight:bold;}"
        )
        toolbar.addWidget(btn_map_sensor)

        btn_sensor_mount = QPushButton("Sensor Mount…")
        btn_sensor_mount.setToolTip(
            "Choose the parent link and adjust sensor XYZ / RPY / scale / opacity."
        )
        toolbar.addWidget(btn_sensor_mount)

        btn_sensor_signal = QPushButton("Live Signal")
        btn_sensor_signal.setCheckable(True)
        btn_sensor_signal.setChecked(True)
        btn_sensor_signal.setToolTip(
            "Colour the mapped sensor from live diffPerDataAve values. "
            "Turn off to restore the row/column orientation colours."
        )
        btn_sensor_signal.setStyleSheet(
            "QPushButton{padding:4px 12px;}"
            "QPushButton:checked{background-color:#a34b3f;color:white;font-weight:bold;}"
        )
        toolbar.addWidget(btn_sensor_signal)

        v.addLayout(toolbar)

        motion_toolbar = QHBoxLayout()
        motion_toolbar.setContentsMargins(8, 0, 8, 6)
        motion_toolbar.setSpacing(8)

        btn_admittance = QPushButton("Pressure Admittance")
        btn_admittance.setCheckable(True)
        btn_admittance.setToolTip(
            "Yield along the mapped sensor normal using pressure-controlled "
            "base-frame TCP velocity. This commands the real robot."
        )
        btn_admittance.setStyleSheet(
            "QPushButton{padding:4px 12px;}"
            "QPushButton:checked{background-color:#b76a20;color:white;font-weight:bold;}"
        )
        btn_admittance.setChecked(
            bool(getattr(self, "_robot_dialog_admittance_active", False))
        )
        motion_toolbar.addWidget(btn_admittance)

        btn_link5_frame = QPushButton("Link 5 Frame")
        btn_link5_frame.setCheckable(True)
        btn_link5_frame.setChecked(True)
        btn_link5_frame.setToolTip(
            "Show the exact link-5 kinematic frame origin and its local XYZ axes."
        )
        btn_link5_frame.setStyleSheet(
            "QPushButton{padding:4px 12px;}"
            "QPushButton:checked{background-color:#3977a8;color:white;font-weight:bold;}"
        )
        motion_toolbar.addWidget(btn_link5_frame)

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
        motion_toolbar.addWidget(btn_drag_vel)

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
        motion_toolbar.addWidget(btn_drag_ptp)

        motion_toolbar.addStretch(1)

        status_label = QLabel("Idle")
        status_label.setStyleSheet("color:#aaaaaa;")
        status_label.setMaximumWidth(300)
        motion_toolbar.addWidget(status_label)

        v.addLayout(motion_toolbar)

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

        initial_joints = [0.0] * 6
        api = getattr(self.parent, "robot_api", None)
        if api is not None and hasattr(api, "get_current_positions"):
            try:
                feedback = api.get_current_positions()
                if feedback is not None and len(feedback) >= 6:
                    candidate = [float(value) for value in feedback[:6]]
                    if np.all(np.isfinite(candidate)):
                        initial_joints = candidate
            except Exception:
                pass

        applied = self._compute_kinematic_chain(initial_joints)
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
                opacity=0.34,
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
        self._robot_dialog_current_joints = list(initial_joints)
        self._robot_dialog_status = status_label
        self._robot_dialog_live_btn = btn_live
        self._robot_dialog_sensor_stream_btn = btn_robot_sensor_stream
        self._robot_dialog_map_sensor_btn = btn_map_sensor
        self._robot_dialog_sensor_mount_btn = btn_sensor_mount
        self._robot_dialog_sensor_signal_btn = btn_sensor_signal
        self._robot_dialog_admittance_btn = btn_admittance
        self._robot_dialog_link5_frame_btn = btn_link5_frame
        self._robot_dialog_drag_vel_btn = btn_drag_vel
        self._robot_dialog_drag_ptp_btn = btn_drag_ptp
        self._robot_dialog_sensor_actor = None
        self._robot_dialog_sensor_mesh = None
        self._robot_dialog_sensor_local_mesh = None
        self._robot_dialog_sensor_preview_model = None
        self._robot_dialog_sensor_preview_key = None
        self._robot_dialog_sensor_last_frame = None
        self._robot_dialog_sensor_stream_owned = False
        self._robot_dialog_sensor_stream_state = "idle"
        self._robot_dialog_previous_main_visualization_enabled = None
        self._robot_dialog_sensor_mapping_config = None
        self._robot_dialog_sensor_mapping_dialog = None
        self._robot_dialog_link5_frame_visible = False
        self._robot_dialog_link5_origin_mesh = None
        self._robot_dialog_link5_origin_actor = None
        self._robot_dialog_link5_axis_meshes = []
        self._robot_dialog_link5_axis_actors = []
        self._robot_dialog_link5_label_points = None
        self._robot_dialog_link5_label_actor = None
        self._robot_dialog_control_center_mesh = None
        self._robot_dialog_control_center_actor = None
        self._robot_dialog_control_center_label_points = None
        self._robot_dialog_control_center_label_actor = None
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

        sensor_signal_timer = QTimer(dialog)
        sensor_signal_timer.setInterval(50)  # Up to 20 Hz; frame-gated below.
        sensor_signal_timer.timeout.connect(self._robot_dialog_sensor_signal_tick)
        self._robot_dialog_sensor_signal_timer = sensor_signal_timer

        admittance_timer = self._ensure_pressure_admittance_timer()

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

        btn_robot_sensor_stream.toggled.connect(
            self._toggle_robot_dialog_sensor_stream
        )

        def _on_map_sensor_toggled(checked: bool):
            if self._toggle_robot_dialog_sensor_mapping(checked):
                return
            if checked:
                btn_map_sensor.blockSignals(True)
                btn_map_sensor.setChecked(False)
                btn_map_sensor.blockSignals(False)

        btn_map_sensor.toggled.connect(_on_map_sensor_toggled)
        btn_sensor_mount.clicked.connect(self._open_robot_dialog_sensor_mapping_dialog)
        btn_sensor_signal.toggled.connect(
            self._set_robot_dialog_sensor_signal_enabled
        )
        btn_admittance.toggled.connect(self._toggle_robot_dialog_admittance)
        btn_link5_frame.toggled.connect(
            self._set_robot_dialog_link5_frame_visible
        )

        def _on_drag_vel_toggled(checked: bool):
            if checked and getattr(self, "_robot_dialog_admittance_active", False):
                self._teardown_robot_dialog_admittance(
                    status_text="Pressure admittance stopped for velocity drag."
                )
            if checked and btn_drag_ptp.isChecked():
                btn_drag_ptp.blockSignals(True)
                btn_drag_ptp.setChecked(False)
                btn_drag_ptp.blockSignals(False)
                # Tear down the other mode cleanly before switching.
                self._teardown_robot_dialog_drag()
            self._toggle_robot_dialog_drag(checked, mode='velocity')

        def _on_drag_ptp_toggled(checked: bool):
            if checked and getattr(self, "_robot_dialog_admittance_active", False):
                self._teardown_robot_dialog_admittance(
                    status_text="Pressure admittance stopped for PTP drag."
                )
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
                sensor_signal_timer.stop()
            except Exception:
                pass
            try:
                if getattr(self, "_admittance_source", None) == "robot_dialog":
                    self._teardown_robot_dialog_admittance(update_status=False)
            except Exception:
                pass
            try:
                if getattr(self, "_admittance_source", None) == "robot_dialog":
                    admittance_timer.stop()
            except Exception:
                pass
            try:
                stopped = self._stop_robot_dialog_sensor_stream(restore_main=True)
                if not stopped:
                    sensor = getattr(self.parent, "sensor_functions", None)
                    previous_main = getattr(
                        self,
                        "_robot_dialog_previous_main_visualization_enabled",
                        True,
                    )
                    self._stop_sensor_stream_after_calibration(
                        sensor, bool(previous_main)
                    )
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
            mapping_dialog = getattr(self, "_robot_dialog_sensor_mapping_dialog", None)
            if mapping_dialog is not None:
                try:
                    mapping_dialog.close()
                except Exception:
                    pass

            try:
                self._hide_robot_dialog_link5_frame(render=False)
            except Exception:
                pass
            try:
                self._hide_robot_dialog_control_center(render=False)
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
            self._robot_dialog_sensor_stream_btn = None
            self._robot_dialog_map_sensor_btn = None
            self._robot_dialog_sensor_mount_btn = None
            self._robot_dialog_sensor_signal_btn = None
            self._robot_dialog_admittance_btn = None
            self._robot_dialog_link5_frame_btn = None
            self._robot_dialog_drag_vel_btn = None
            self._robot_dialog_drag_ptp_btn = None
            self._robot_dialog_sensor_actor = None
            self._robot_dialog_sensor_mesh = None
            self._robot_dialog_sensor_local_mesh = None
            self._robot_dialog_sensor_preview_model = None
            self._robot_dialog_sensor_preview_key = None
            self._robot_dialog_sensor_last_frame = None
            self._robot_dialog_sensor_stream_owned = False
            self._robot_dialog_sensor_stream_state = "idle"
            self._robot_dialog_previous_main_visualization_enabled = None
            self._robot_dialog_sensor_mapping_config = None
            self._robot_dialog_sensor_mapping_dialog = None
            self._robot_dialog_sensor_signal_timer = None
            self._robot_dialog_link5_frame_visible = False
            self._robot_dialog_link5_origin_mesh = None
            self._robot_dialog_link5_origin_actor = None
            self._robot_dialog_link5_axis_meshes = []
            self._robot_dialog_link5_axis_actors = []
            self._robot_dialog_link5_label_points = None
            self._robot_dialog_link5_label_actor = None
            self._robot_dialog_control_center_mesh = None
            self._robot_dialog_control_center_actor = None
            self._robot_dialog_control_center_label_points = None
            self._robot_dialog_control_center_label_actor = None
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
        self._set_robot_dialog_link5_frame_visible(True)
        # This only reads joint feedback; it never commands the robot. Keep
        # the model, mounted sensor and Link 5 frame synchronized by default.
        btn_live.setChecked(True)

        # After showing the new dialog, give VTK a moment to settle and then
        # force a render on the main plotter so it re-paints itself.
        for ms in (50, 150, 350, 700):
            QTimer.singleShot(ms, self._poke_main_plotters)

    # ------------------------------------------------------------------
    # Robot-dialog link-frame overlay
    # ------------------------------------------------------------------
    @staticmethod
    def _link_frame_origin_and_axes(transform):
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return None, None
        origin = np.array(matrix[:3, 3], dtype=float, copy=True)
        axes = np.array(matrix[:3, :3], dtype=float, copy=True)
        for index in range(3):
            norm = float(np.linalg.norm(axes[:, index]))
            if norm <= 1e-9:
                return None, None
            axes[:, index] /= norm
        return origin, axes

    def _robot_dialog_link5_transform(self):
        applied = getattr(self, "_robot_dialog_applied", None)
        if not isinstance(applied, (list, tuple)) or len(applied) <= 5:
            return None
        return applied[5]

    def _set_robot_dialog_link5_frame_visible(self, enabled):
        button = getattr(self, "_robot_dialog_link5_frame_btn", None)
        self._set_checked_without_signal(button, bool(enabled))
        if not bool(enabled):
            self._hide_robot_dialog_link5_frame(render=True)
            status = getattr(self, "_robot_dialog_status", None)
            if status is not None:
                status.setText("Link 5 frame hidden")
            return

        self._robot_dialog_link5_frame_visible = True
        if not self._update_robot_dialog_link5_frame(render=True):
            self._robot_dialog_link5_frame_visible = False
            self._set_checked_without_signal(button, False)
            return

        origin, _axes = self._link_frame_origin_and_axes(
            self._robot_dialog_link5_transform()
        )
        status = getattr(self, "_robot_dialog_status", None)
        if status is not None and origin is not None:
            status.setText(
                "Link 5 origin: ({0:+.3f}, {1:+.3f}, {2:+.3f}) m".format(
                    float(origin[0]), float(origin[1]), float(origin[2])
                )
            )

    def _update_robot_dialog_link5_frame(self, render=False):
        if not getattr(self, "_robot_dialog_link5_frame_visible", False):
            return False
        plotter = getattr(self, "_robot_dialog_plotter", None)
        transform = self._robot_dialog_link5_transform()
        origin, axes = self._link_frame_origin_and_axes(transform)
        if plotter is None or origin is None or axes is None:
            return False

        axis_length = 0.11
        axis_colours = ("#ff4040", "#45d96b", "#4385ff")
        origin_mesh = getattr(self, "_robot_dialog_link5_origin_mesh", None)
        axis_meshes = list(
            getattr(self, "_robot_dialog_link5_axis_meshes", None) or []
        )
        if origin_mesh is None or len(axis_meshes) != 3:
            origin_mesh = pv.Sphere(radius=0.014, center=origin)
            origin_actor = plotter.add_mesh(
                origin_mesh,
                color="#ffd447",
                smooth_shading=True,
                lighting=False,
                pickable=False,
            )
            axis_meshes = []
            axis_actors = []
            for index, colour in enumerate(axis_colours):
                axis_mesh = pv.Arrow(
                    start=origin,
                    direction=axes[:, index],
                    scale=axis_length,
                    tip_length=0.22,
                    tip_radius=0.11,
                    shaft_radius=0.035,
                )
                axis_meshes.append(axis_mesh)
                axis_actors.append(
                    plotter.add_mesh(
                        axis_mesh,
                        color=colour,
                        lighting=False,
                        pickable=False,
                    )
                )

            label_points = pv.PolyData(np.asarray([origin], dtype=float))
            label_actor = plotter.add_point_labels(
                label_points,
                ["Link 5 frame origin"],
                font_size=15,
                text_color="#ffffff",
                show_points=True,
                point_color="#ffd447",
                point_size=16,
                shape="rect",
                shape_color="#20242a",
                shape_opacity=0.82,
                render_points_as_spheres=True,
                always_visible=True,
                reset_camera=False,
                render=False,
            )
            self._robot_dialog_link5_origin_mesh = origin_mesh
            self._robot_dialog_link5_origin_actor = origin_actor
            self._robot_dialog_link5_axis_meshes = axis_meshes
            self._robot_dialog_link5_axis_actors = axis_actors
            self._robot_dialog_link5_label_points = label_points
            self._robot_dialog_link5_label_actor = label_actor
        else:
            origin_mesh.copy_from(pv.Sphere(radius=0.014, center=origin))
            for index, axis_mesh in enumerate(axis_meshes):
                axis_mesh.copy_from(
                    pv.Arrow(
                        start=origin,
                        direction=axes[:, index],
                        scale=axis_length,
                        tip_length=0.22,
                        tip_radius=0.11,
                        shaft_radius=0.035,
                    )
                )
            label_points = getattr(
                self, "_robot_dialog_link5_label_points", None
            )
            if label_points is not None:
                label_points.points = np.asarray([origin], dtype=float)
                label_points.Modified()

        if render:
            try:
                plotter.render()
            except Exception:
                pass
        return True

    def _hide_robot_dialog_link5_frame(self, render=True):
        plotter = getattr(self, "_robot_dialog_plotter", None)
        actors = [
            getattr(self, "_robot_dialog_link5_origin_actor", None),
            *(getattr(self, "_robot_dialog_link5_axis_actors", None) or []),
            getattr(self, "_robot_dialog_link5_label_actor", None),
        ]
        if plotter is not None:
            for actor in actors:
                if actor is None:
                    continue
                try:
                    plotter.remove_actor(actor, reset_camera=False, render=False)
                except Exception:
                    pass
        self._robot_dialog_link5_frame_visible = False
        self._robot_dialog_link5_origin_mesh = None
        self._robot_dialog_link5_origin_actor = None
        self._robot_dialog_link5_axis_meshes = []
        self._robot_dialog_link5_axis_actors = []
        self._robot_dialog_link5_label_points = None
        self._robot_dialog_link5_label_actor = None
        if render and plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Robot-dialog sensor mounting / mapping preview
    # ------------------------------------------------------------------
    @staticmethod
    def _set_checked_without_signal(button, checked):
        if button is None:
            return
        button.blockSignals(True)
        button.setChecked(bool(checked))
        button.blockSignals(False)

    def _update_robot_dialog_sensor_stream_state(self):
        button = getattr(self, "_robot_dialog_sensor_stream_btn", None)
        sensor = getattr(self.parent, "sensor_functions", None)
        if button is None or sensor is None or not button.isChecked():
            return

        calibrating = bool(
            getattr(sensor, "_sensor_calibration_in_progress", False)
        )
        running = bool(
            getattr(sensor, "_sensor_reader_is_running", lambda: False)()
            and getattr(sensor, "is_connected", False)
        )
        status = getattr(self, "_robot_dialog_status", None)
        previous = getattr(self, "_robot_dialog_sensor_stream_state", "idle")
        error_detail = ""
        if calibrating:
            state = "calibrating"
            button.setText("Calibrating…")
        elif running:
            state = "running"
            button.setText("Stop Sensor")
        else:
            state = "failed"
            if hasattr(sensor, "get_last_sensor_stream_error"):
                error_detail = sensor.get_last_sensor_stream_error()
            if previous in {"starting", "calibrating"}:
                self._stop_robot_dialog_sensor_stream(restore_main=True)
            button.setText("Start Sensor")
            self._set_checked_without_signal(button, False)

        if state != previous and status is not None:
            if state == "calibrating":
                status.setText("Sensor: calibrating for robot viewer…")
                status.setToolTip("")
            elif state == "running":
                status.setText("Sensor: robot-viewer stream active")
                status.setToolTip("")
            else:
                status.setText("Sensor failed (hover for details)")
                status.setToolTip(
                    error_detail or "No valid sensor frame was received."
                )
        if state == "failed" and error_detail:
            button.setToolTip(error_detail)
        self._robot_dialog_sensor_stream_state = state

    def _toggle_robot_dialog_sensor_stream(self, enabled):
        button = getattr(self, "_robot_dialog_sensor_stream_btn", None)
        sensor = getattr(self.parent, "sensor_functions", None)
        status = getattr(self, "_robot_dialog_status", None)
        if sensor is None or not hasattr(
            sensor, "start_external_visualization_stream"
        ):
            self._set_checked_without_signal(button, False)
            QMessageBox.warning(
                getattr(self, "_robot_dialog", None),
                "Robot Sensor Stream",
                "Sensor functions are unavailable.",
            )
            return

        if not bool(enabled):
            self._stop_robot_dialog_sensor_stream(restore_main=True)
            return

        self._robot_dialog_previous_main_visualization_enabled = bool(
            getattr(sensor, "main_visualization_enabled", True)
        )
        already_running = bool(
            getattr(sensor, "_sensor_reader_is_running", lambda: False)()
            and getattr(sensor, "is_connected", False)
        )
        if already_running:
            self._robot_dialog_sensor_stream_owned = False
            sensor.set_main_visualization_enabled(False, render=True)
            started = True
        else:
            started = bool(sensor.start_external_visualization_stream())
            self._robot_dialog_sensor_stream_owned = started

        if not started:
            error_detail = ""
            if hasattr(sensor, "get_last_sensor_stream_error"):
                error_detail = sensor.get_last_sensor_stream_error()
            self._set_checked_without_signal(button, False)
            if button is not None:
                button.setText("Start Sensor")
                if error_detail:
                    button.setToolTip(error_detail)
            if status is not None:
                status.setText("Sensor could not start (hover for details)")
                status.setToolTip(
                    error_detail or "Select a valid sensor serial port."
                )
            QMessageBox.information(
                getattr(self, "_robot_dialog", None),
                "Robot Sensor Stream",
                "The sensor stream could not start.\n\n"
                + (error_detail or "Select a valid sensor serial port.")
                + "\n\nSelect the sensor model and serial port in the Sensor "
                "tab, then press Start Sensor here.",
            )
            return

        self._robot_dialog_sensor_stream_state = "starting"
        if button is not None:
            button.setText("Calibrating…" if not already_running else "Stop Sensor")

        # Rebuild the overlay from the actual selected model, then enable live
        # signal rendering. No second serial/API reader is created.
        self._robot_dialog_sensor_preview_model = None
        self._robot_dialog_sensor_preview_key = None
        self._robot_dialog_sensor_local_mesh = None
        map_button = getattr(self, "_robot_dialog_map_sensor_btn", None)
        if map_button is not None and not map_button.isChecked():
            map_button.setChecked(True)
        else:
            self._refresh_robot_dialog_sensor_overlay(render=True)
        signal_button = getattr(self, "_robot_dialog_sensor_signal_btn", None)
        if signal_button is not None and not signal_button.isChecked():
            signal_button.setChecked(True)
        else:
            self._set_robot_dialog_sensor_signal_enabled(True)
        self._update_robot_dialog_sensor_stream_state()

    def _stop_robot_dialog_sensor_stream(self, restore_main=True):
        button = getattr(self, "_robot_dialog_sensor_stream_btn", None)
        sensor = getattr(self.parent, "sensor_functions", None)
        status = getattr(self, "_robot_dialog_status", None)
        owned = bool(getattr(self, "_robot_dialog_sensor_stream_owned", False))

        if (
            getattr(self, "_robot_dialog_admittance_active", False)
            and getattr(self, "_admittance_source", None) == "robot_dialog"
        ):
            self._teardown_robot_dialog_admittance(update_status=False)

        stopped = True
        if sensor is not None and owned:
            stopped = bool(sensor.stop_external_visualization_stream())
        if not stopped:
            if status is not None:
                status.setText("Sensor: wait for calibration before stopping")
            self._set_checked_without_signal(button, True)
            return False

        previous = getattr(
            self, "_robot_dialog_previous_main_visualization_enabled", None
        )
        if sensor is not None and bool(restore_main) and previous is not None:
            sensor.set_main_visualization_enabled(bool(previous), render=True)
        self._robot_dialog_sensor_stream_owned = False
        self._robot_dialog_previous_main_visualization_enabled = None
        self._robot_dialog_sensor_stream_state = "idle"
        self._set_checked_without_signal(button, False)
        if button is not None:
            button.setText("Start Sensor")
        if status is not None:
            status.setText("Sensor: stopped")
        return True

    def _stop_sensor_stream_after_calibration(
        self, sensor, restore_main_enabled, attempts=0
    ):
        """Finish closing a robot-owned stream if its calibration was in flight."""
        if sensor is None:
            return
        if bool(getattr(sensor, "_sensor_calibration_in_progress", False)):
            if int(attempts) < 80:
                QTimer.singleShot(
                    250,
                    lambda: self._stop_sensor_stream_after_calibration(
                        sensor, restore_main_enabled, int(attempts) + 1
                    ),
                )
            return
        try:
            sensor.stop_external_visualization_stream()
        except Exception:
            pass
        try:
            sensor.set_main_visualization_enabled(
                bool(restore_main_enabled), render=True
            )
        except Exception:
            pass

    @staticmethod
    def _robot_sensor_mapping_config_path():
        return Path(resource_path("config", "robot_sensor_mapping.json"))

    @staticmethod
    def _default_robot_sensor_mapping_config():
        return {
            "link_index": 5,
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": 1.0,
            "horizontal_scale": 1.0,
            "vertical_scale": 1.0,
            "opacity": 0.82,
            "admittance_max_speed_mps": 0.03,
            "admittance_contact_threshold_pct": 3.0,
            "admittance_full_scale_pct": 12.0,
            "admittance_smoothing_alpha": 0.35,
            "admittance_reverse_direction": False,
            "admittance_direction_mode": "surface_normal",
            "admittance_control_center_m": [0.0, 0.0, 0.0],
        }

    @classmethod
    def _normalize_robot_sensor_mapping_config(cls, config):
        default = cls._default_robot_sensor_mapping_config()
        if not isinstance(config, dict):
            config = {}

        try:
            link_index = int(config.get("link_index", default["link_index"]))
        except Exception:
            link_index = default["link_index"]
        link_index = int(np.clip(link_index, 0, 7))

        def _vector3(name, fallback, minimum, maximum):
            value = config.get(name, fallback)
            if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
                value = fallback
            result = []
            for index in range(3):
                try:
                    result.append(float(np.clip(float(value[index]), minimum, maximum)))
                except Exception:
                    result.append(float(fallback[index]))
            return result

        translation_m = _vector3(
            "translation_m", default["translation_m"], -2.0, 2.0
        )
        rotation_deg = _vector3(
            "rotation_deg", default["rotation_deg"], -180.0, 180.0
        )
        try:
            scale = float(np.clip(float(config.get("scale", default["scale"])), 0.001, 100.0))
        except Exception:
            scale = default["scale"]
        try:
            horizontal_scale = float(
                np.clip(
                    float(
                        config.get(
                            "horizontal_scale", default["horizontal_scale"]
                        )
                    ),
                    0.001,
                    100.0,
                )
            )
        except Exception:
            horizontal_scale = default["horizontal_scale"]
        try:
            vertical_scale = float(
                np.clip(
                    float(
                        config.get("vertical_scale", default["vertical_scale"])
                    ),
                    0.001,
                    100.0,
                )
            )
        except Exception:
            vertical_scale = default["vertical_scale"]
        try:
            opacity = float(np.clip(float(config.get("opacity", default["opacity"])), 0.05, 1.0))
        except Exception:
            opacity = default["opacity"]

        def _number(name, minimum, maximum):
            try:
                return float(
                    np.clip(float(config.get(name, default[name])), minimum, maximum)
                )
            except Exception:
                return float(default[name])

        admittance_max_speed_mps = _number(
            "admittance_max_speed_mps", 0.001, 0.1
        )
        admittance_contact_threshold_pct = _number(
            "admittance_contact_threshold_pct", 0.1, 99.0
        )
        admittance_full_scale_pct = _number(
            "admittance_full_scale_pct", 0.2, 100.0
        )
        admittance_full_scale_pct = max(
            admittance_contact_threshold_pct + 0.1,
            admittance_full_scale_pct,
        )
        admittance_smoothing_alpha = _number(
            "admittance_smoothing_alpha", 0.01, 1.0
        )
        admittance_reverse_direction = bool(
            config.get(
                "admittance_reverse_direction",
                default["admittance_reverse_direction"],
            )
        )
        admittance_direction_mode = str(
            config.get(
                "admittance_direction_mode",
                default["admittance_direction_mode"],
            )
        ).strip().lower()
        if admittance_direction_mode not in ("surface_normal", "centre_directed"):
            admittance_direction_mode = default["admittance_direction_mode"]
        admittance_control_center_m = _vector3(
            "admittance_control_center_m",
            default["admittance_control_center_m"],
            -2.0,
            2.0,
        )

        return {
            "link_index": link_index,
            "translation_m": translation_m,
            "rotation_deg": rotation_deg,
            "scale": scale,
            "horizontal_scale": horizontal_scale,
            "vertical_scale": vertical_scale,
            "opacity": opacity,
            "admittance_max_speed_mps": admittance_max_speed_mps,
            "admittance_contact_threshold_pct": admittance_contact_threshold_pct,
            "admittance_full_scale_pct": admittance_full_scale_pct,
            "admittance_smoothing_alpha": admittance_smoothing_alpha,
            "admittance_reverse_direction": admittance_reverse_direction,
            "admittance_direction_mode": admittance_direction_mode,
            "admittance_control_center_m": admittance_control_center_m,
        }

    def _robot_sensor_mapping_key(self):
        sensor = getattr(self.parent, "sensor_functions", None)
        if sensor is None:
            return "sensor"
        model = str(getattr(sensor, "current_model_name", "") or "")
        rows = int(getattr(sensor, "n_row", 0) or 0)
        cols = int(getattr(sensor, "n_col", 0) or 0)
        if not model:
            try:
                selected_index = int(self.parent.sensor_choice.currentRow())
                model = str(sensor.get_sensor_model_name_for_index(selected_index))
                rows, cols = sensor._sensor_shape_for_model(model)
            except Exception:
                model = "sensor"
        return f"{model}_{rows}x{cols}"

    def _load_robot_sensor_mapping_config(self):
        default = self._default_robot_sensor_mapping_config()
        path = self._robot_sensor_mapping_config_path()
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            mappings = payload.get("mappings", {}) if isinstance(payload, dict) else {}
            return self._normalize_robot_sensor_mapping_config(
                mappings.get(self._robot_sensor_mapping_key(), default)
            )
        except FileNotFoundError:
            return default
        except Exception as exc:
            print(f"[RobotSensorMapping] Failed to load {path}: {exc}")
            return default

    def _save_robot_sensor_mapping_config(self, config):
        path = self._robot_sensor_mapping_config_path()
        payload = {"version": 1, "mappings": {}}
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    payload.update(loaded)
                if not isinstance(payload.get("mappings"), dict):
                    payload["mappings"] = {}
            normalized = self._normalize_robot_sensor_mapping_config(config)
            payload["mappings"][self._robot_sensor_mapping_key()] = normalized
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            return True
        except Exception as exc:
            print(f"[RobotSensorMapping] Failed to save {path}: {exc}")
            return False

    def _build_selected_sensor_preview_model(self):
        sensor = getattr(self.parent, "sensor_functions", None)
        if sensor is None:
            return None
        try:
            selected_index = int(self.parent.sensor_choice.currentRow())
            model_name = str(sensor.get_sensor_model_name_for_index(selected_index))
            n_row, n_col = sensor._sensor_shape_for_model(model_name)
        except Exception as exc:
            print(f"[RobotSensorMapping] Cannot resolve selected sensor: {exc}")
            return None

        preview_key = f"{model_name}_{int(n_row)}x{int(n_col)}"
        cached_key = getattr(self, "_robot_dialog_sensor_preview_key", None)
        cached_model = getattr(self, "_robot_dialog_sensor_preview_model", None)
        if cached_model is not None and cached_key == preview_key:
            return cached_model

        try:
            # Local import avoids coupling the mesh module to sensor startup.
            from phd.dependence.func_sensor import SensorModelFactory

            if model_name == "2d":
                model = SensorModelFactory(
                    n_row=int(n_row),
                    n_col=int(n_col),
                    window_size=int(
                        getattr(sensor, "sensor_average_window_size", 3) or 3
                    ),
                    offset_scale=0.0005,
                ).build()

                geometry = sensor.get_saved_sensor_geometry_config(
                    model_name, n_row=n_row, n_col=n_col
                )
                effective = sensor._effective_sensor_geometry_config(geometry)
                bent_coarse, _bent_normals = sensor._bend_points_to_cylinder(
                    model.points_origin,
                    effective,
                    return_normals=True,
                    base_normals=model.normals,
                )
                rotation_deg = effective.get("rotation_deg", [0.0, 0.0, 0.0])
                pivot = sensor._sensor_geometry_rotation_pivot(bent_coarse)
                fine_points = sensor._bend_points_to_cylinder(
                    np.asarray(model._2D_map.points, dtype=float),
                    effective,
                    return_normals=False,
                )
                model._2D_map.points = sensor._rotate_sensor_geometry(
                    fine_points, rotation_deg, pivot=pivot
                )
            else:
                model_kwargs = dict(sensor.PREDEFINED_SENSOR_MODELS[model_name])
                model_kwargs["window_size"] = int(
                    getattr(sensor, "sensor_average_window_size", 3) or 3
                )
                model_kwargs = sensor._apply_saved_reorder_logic(model_kwargs)
                model = SensorModelFactory(**model_kwargs).build()

            model.current_model_name = model_name
            self._robot_dialog_sensor_preview_key = preview_key
            self._robot_dialog_sensor_preview_model = model
            return model
        except Exception as exc:
            print(
                f"[RobotSensorMapping] Failed to build {preview_key} preview: {exc}"
            )
            return None

    def _current_built_sensor_geometry(self):
        sensor = getattr(self.parent, "sensor_functions", None)
        source = getattr(sensor, "_2D_map", None) if sensor is not None else None
        if source is not None and int(getattr(source, "n_points", 0) or 0) > 0:
            return sensor, source

        preview = self._build_selected_sensor_preview_model()
        source = getattr(preview, "_2D_map", None) if preview is not None else None
        if source is None or int(getattr(source, "n_points", 0) or 0) <= 0:
            return None, None
        return preview, source

    def _build_robot_dialog_sensor_local_mesh(self):
        sensor, source = self._current_built_sensor_geometry()
        if sensor is None or source is None:
            return None
        try:
            mesh = source.copy(deep=True)
            points = np.asarray(mesh.points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
                return None
            finite = np.all(np.isfinite(points), axis=1)
            if not np.any(finite):
                return None
            center = np.mean(points[finite], axis=0)
            points = np.array(points, dtype=float, copy=True)
            points[finite] -= center
            mesh.points = points

            # Mapping-preview colours encode logical taxel orientation rather
            # than live signal amplitude: red increases with column, green
            # increases with row. This makes flips / rotations visible even
            # for the procedural 2D sensor whose normal display is one colour.
            mapping_colors = np.tile([0.2, 0.2, 0.72, 1.0], (mesh.n_points, 1))
            array_positions = list(getattr(sensor, "array_positions", None) or [])
            n_row = int(getattr(sensor, "n_row", 0) or 0)
            n_col = int(getattr(sensor, "n_col", 0) or 0)
            if len(array_positions) == n_row * n_col and n_row > 0 and n_col > 0:
                for taxel_index, fine_indices in enumerate(array_positions):
                    row = taxel_index % n_row
                    col = taxel_index // n_row
                    colour = [
                        0.2 + 0.8 * col / max(1, n_col - 1),
                        0.2 + 0.8 * row / max(1, n_row - 1),
                        0.72,
                        1.0,
                    ]
                    valid_indices = [
                        int(index)
                        for index in fine_indices
                        if 0 <= int(index) < mesh.n_points
                    ]
                    if valid_indices:
                        mapping_colors[valid_indices] = colour
            else:
                colors = getattr(sensor, "colors", None)
                if colors is not None:
                    colors = np.asarray(colors)
                    if len(colors) == mesh.n_points:
                        mapping_colors = np.array(colors, copy=True)
            mesh.point_data["sensor_mapping_colors"] = mapping_colors
            return mesh
        except Exception as exc:
            print(f"[RobotSensorMapping] Failed to copy sensor geometry: {exc}")
            return None

    @staticmethod
    def _robot_sensor_local_transform(config):
        config = MyMeshLab._normalize_robot_sensor_mapping_config(config)
        rx, ry, rz = np.radians(config["rotation_deg"])
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])

        overall_scale = float(config["scale"])
        local_scale = np.diag(
            [
                overall_scale * float(config["horizontal_scale"]),
                overall_scale * float(config["vertical_scale"]),
                overall_scale,
            ]
        )

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = (rot_z @ rot_y @ rot_x) @ local_scale
        transform[:3, 3] = np.asarray(config["translation_m"], dtype=float)
        return transform

    @staticmethod
    def _robot_sensor_mapping_link_label(link_index):
        labels = {
            0: "Base",
            1: "Link 1",
            2: "Link 2",
            3: "Link 3",
            4: "Link 4",
            5: "Link 5",
            6: "Link 6 / flange",
            7: "Tool visual",
        }
        return labels.get(int(link_index), f"Link {int(link_index)}")

    @staticmethod
    def _robot_sensor_mount_rotation(config):
        linear = MyMeshLab._robot_sensor_local_transform(config)[:3, :3]
        rotation = np.array(linear, dtype=float, copy=True)
        for index in range(3):
            norm = float(np.linalg.norm(rotation[:, index]))
            if norm <= 1e-9:
                return None
            rotation[:, index] /= norm
        return rotation

    def _robot_dialog_control_center_base_point(self, config):
        config = self._normalize_robot_sensor_mapping_config(config)
        center_sensor = np.asarray(
            config["admittance_control_center_m"], dtype=float
        )
        mount_rotation = self._robot_sensor_mount_rotation(config)
        if center_sensor.shape != (3,) or mount_rotation is None:
            return None

        center_link = (
            mount_rotation @ center_sensor
            + np.asarray(config["translation_m"], dtype=float)
        )
        applied = getattr(self, "_robot_dialog_applied", None)
        link_index = int(config["link_index"])
        link_transform = None
        if isinstance(applied, (list, tuple)) and link_index < len(applied):
            link_transform = applied[link_index]
        if link_transform is None:
            link_transform = np.eye(4, dtype=float)
        link_transform = np.asarray(link_transform, dtype=float)
        return link_transform[:3, :3] @ center_link + link_transform[:3, 3]

    def _update_robot_dialog_control_center(self, config=None, render=False):
        plotter = getattr(self, "_robot_dialog_plotter", None)
        sensor_actor = getattr(self, "_robot_dialog_sensor_actor", None)
        config = self._normalize_robot_sensor_mapping_config(
            config
            or getattr(self, "_robot_dialog_sensor_mapping_config", None)
            or self._load_robot_sensor_mapping_config()
        )
        if (
            plotter is None
            or sensor_actor is None
            or config["admittance_direction_mode"] != "centre_directed"
        ):
            self._hide_robot_dialog_control_center(render=render)
            return False

        center = self._robot_dialog_control_center_base_point(config)
        if center is None or not np.all(np.isfinite(center)):
            self._hide_robot_dialog_control_center(render=render)
            return False

        center_mesh = getattr(self, "_robot_dialog_control_center_mesh", None)
        if center_mesh is None:
            center_mesh = pv.Sphere(radius=0.010, center=center)
            center_actor = plotter.add_mesh(
                center_mesh,
                color="#25e6dc",
                smooth_shading=True,
                lighting=False,
                pickable=False,
            )
            label_points = pv.PolyData(np.asarray([center], dtype=float))
            label_actor = plotter.add_point_labels(
                label_points,
                ["Admittance control centre"],
                font_size=14,
                text_color="#ffffff",
                show_points=True,
                point_color="#25e6dc",
                point_size=14,
                shape="rect",
                shape_color="#164d50",
                shape_opacity=0.84,
                render_points_as_spheres=True,
                always_visible=True,
                reset_camera=False,
                render=False,
            )
            self._robot_dialog_control_center_mesh = center_mesh
            self._robot_dialog_control_center_actor = center_actor
            self._robot_dialog_control_center_label_points = label_points
            self._robot_dialog_control_center_label_actor = label_actor
        else:
            center_mesh.copy_from(pv.Sphere(radius=0.010, center=center))
            label_points = getattr(
                self, "_robot_dialog_control_center_label_points", None
            )
            if label_points is not None:
                label_points.points = np.asarray([center], dtype=float)
                label_points.Modified()

        if render:
            try:
                plotter.render()
            except Exception:
                pass
        return True

    def _hide_robot_dialog_control_center(self, render=True):
        plotter = getattr(self, "_robot_dialog_plotter", None)
        for actor in (
            getattr(self, "_robot_dialog_control_center_actor", None),
            getattr(self, "_robot_dialog_control_center_label_actor", None),
        ):
            if plotter is None or actor is None:
                continue
            try:
                plotter.remove_actor(actor, reset_camera=False, render=False)
            except Exception:
                pass
        self._robot_dialog_control_center_mesh = None
        self._robot_dialog_control_center_actor = None
        self._robot_dialog_control_center_label_points = None
        self._robot_dialog_control_center_label_actor = None
        if render and plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass

    def _mapped_robot_dialog_sensor_mesh(self, config):
        local_mesh = getattr(self, "_robot_dialog_sensor_local_mesh", None)
        if local_mesh is None:
            local_mesh = self._build_robot_dialog_sensor_local_mesh()
            self._robot_dialog_sensor_local_mesh = local_mesh
        if local_mesh is None:
            return None

        config = self._normalize_robot_sensor_mapping_config(config)
        mapped = local_mesh.copy(deep=True)
        mapped.transform(self._robot_sensor_local_transform(config), inplace=True)

        applied = getattr(self, "_robot_dialog_applied", None)
        link_index = int(config["link_index"])
        link_transform = None
        if isinstance(applied, (list, tuple)) and link_index < len(applied):
            link_transform = applied[link_index]
        if link_transform is None:
            link_transform = np.eye(4, dtype=float)
        mapped.transform(np.asarray(link_transform, dtype=float), inplace=True)
        return mapped

    def _refresh_robot_dialog_sensor_overlay(self, config=None, render=True):
        plotter = getattr(self, "_robot_dialog_plotter", None)
        if plotter is None:
            return False
        if config is None:
            config = getattr(self, "_robot_dialog_sensor_mapping_config", None)
        config = self._normalize_robot_sensor_mapping_config(
            config or self._load_robot_sensor_mapping_config()
        )
        mapped = self._mapped_robot_dialog_sensor_mesh(config)
        if mapped is None:
            return False

        actor = getattr(self, "_robot_dialog_sensor_actor", None)
        display_mesh = getattr(self, "_robot_dialog_sensor_mesh", None)
        opacity = float(config["opacity"])
        if actor is None or display_mesh is None:
            display_mesh = mapped
            kwargs = {
                "opacity": opacity,
                "show_edges": True,
                "edge_color": "#111111",
                "line_width": 1,
                "lighting": False,
                "label": "Mapped sensor",
            }
            if "sensor_mapping_colors" in display_mesh.point_data:
                kwargs.update({"scalars": "sensor_mapping_colors", "rgb": True})
            else:
                kwargs.update({"color": "#2fd0b5"})
            try:
                if int(getattr(display_mesh, "n_faces_strict", 0) or 0) <= 0:
                    kwargs.update(
                        {
                            "style": "points",
                            "point_size": 7,
                            "render_points_as_spheres": True,
                            "show_edges": False,
                        }
                    )
            except Exception:
                pass
            actor = plotter.add_mesh(display_mesh, **kwargs)
            self._robot_dialog_sensor_actor = actor
            self._robot_dialog_sensor_mesh = display_mesh
        else:
            preserved_colors = None
            signal_button = getattr(self, "_robot_dialog_sensor_signal_btn", None)
            if signal_button is not None and signal_button.isChecked():
                try:
                    preserved_colors = np.array(
                        display_mesh.point_data["sensor_mapping_colors"],
                        copy=True,
                    )
                except Exception:
                    preserved_colors = None
            try:
                display_mesh.copy_from(mapped)
            except Exception:
                display_mesh.points = np.asarray(mapped.points, dtype=float)
                try:
                    display_mesh.Modified()
                except Exception:
                    pass
            if (
                preserved_colors is not None
                and len(preserved_colors) == display_mesh.n_points
            ):
                display_mesh.point_data["sensor_mapping_colors"] = preserved_colors
            try:
                actor.GetProperty().SetOpacity(opacity)
            except Exception:
                pass

        self._robot_dialog_sensor_mapping_config = config
        self._update_robot_dialog_control_center(config, render=False)
        if render:
            try:
                plotter.render()
            except Exception:
                pass
        return True

    def _restore_robot_dialog_sensor_orientation_colours(self, render=True):
        local_mesh = getattr(self, "_robot_dialog_sensor_local_mesh", None)
        if local_mesh is None:
            return False
        colors = local_mesh.point_data.get("sensor_mapping_colors")
        if colors is None:
            return False
        if not self._apply_robot_dialog_sensor_colours(colors):
            return False
        display_mesh = getattr(self, "_robot_dialog_sensor_mesh", None)
        try:
            display_mesh.Modified()
        except Exception:
            pass
        if render:
            plotter = getattr(self, "_robot_dialog_plotter", None)
            if plotter is not None:
                try:
                    plotter.render()
                except Exception:
                    pass
        return True

    def _apply_robot_dialog_sensor_colours(self, colors):
        """Update the mapped mesh's existing RGBA buffer in place."""
        display_mesh = getattr(self, "_robot_dialog_sensor_mesh", None)
        if display_mesh is None:
            return False
        colors = np.asarray(colors, dtype=float)
        if colors.shape != (display_mesh.n_points, 4):
            return False

        try:
            current = display_mesh.point_data.get("sensor_mapping_colors")
            if current is not None and np.shape(current) == colors.shape:
                current[:] = colors
                vtk_colors = display_mesh.GetPointData().GetArray(
                    "sensor_mapping_colors"
                )
                if vtk_colors is not None:
                    vtk_colors.Modified()
            else:
                display_mesh.point_data["sensor_mapping_colors"] = np.array(
                    colors, copy=True
                )
            display_mesh.set_active_scalars("sensor_mapping_colors")
            display_mesh.Modified()

            actor = getattr(self, "_robot_dialog_sensor_actor", None)
            mapper = actor.GetMapper() if actor is not None else None
            if mapper is not None:
                mapper.SelectColorArray("sensor_mapping_colors")
                mapper.Modified()
                mapper.Update()
            return True
        except Exception as exc:
            print(f"[RobotSensorMapping] Failed to update live colours: {exc}")
            return False

    def _robot_dialog_live_sensor_colours(self):
        sensor = getattr(self.parent, "sensor_functions", None)
        data_obj = getattr(sensor, "_data", None) if sensor is not None else None
        matrix = (
            getattr(data_obj, "diffPerDataAve", None)
            if data_obj is not None
            else None
        )
        if matrix is None:
            return None, None

        n_row = int(getattr(sensor, "n_row", 0) or 0)
        n_col = int(getattr(sensor, "n_col", 0) or 0)
        values = np.asarray(matrix, dtype=float)
        if values.shape != (n_row, n_col) or n_row <= 0 or n_col <= 0:
            return None, None

        display_mesh = getattr(self, "_robot_dialog_sensor_mesh", None)
        if display_mesh is None:
            return None, None
        array_positions = list(getattr(sensor, "array_positions", None) or [])
        if len(array_positions) != n_row * n_col:
            return None, None

        colors = np.tile([0.3, 0.3, 0.3, 1.0], (display_mesh.n_points, 1))
        safe_values = np.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        )
        for col in range(n_col):
            for row in range(n_row):
                taxel_index = col * n_row + row
                sensor_value = float(safe_values[row, col])
                intensity = float(
                    np.clip(1.0 - abs(sensor_value) * 150.0 / 255.0, 0.0, 1.0)
                )
                colour = [1.0, intensity, intensity, 1.0]
                valid_indices = [
                    int(index)
                    for index in array_positions[taxel_index]
                    if 0 <= int(index) < display_mesh.n_points
                ]
                if valid_indices:
                    colors[valid_indices] = colour
        return colors, getattr(data_obj, "frame_sequence", None)

    def _robot_dialog_sensor_signal_tick(self):
        self._update_robot_dialog_sensor_stream_state()
        dialog = getattr(self, "_robot_dialog", None)
        actor = getattr(self, "_robot_dialog_sensor_actor", None)
        button = getattr(self, "_robot_dialog_sensor_signal_btn", None)
        if (
            dialog is None
            or not dialog.isVisible()
            or actor is None
            or button is None
            or not button.isChecked()
        ):
            return

        colors, frame_sequence = self._robot_dialog_live_sensor_colours()
        if colors is None:
            return
        if (
            frame_sequence is not None
            and frame_sequence == getattr(
                self, "_robot_dialog_sensor_last_frame", None
            )
        ):
            return

        display_mesh = getattr(self, "_robot_dialog_sensor_mesh", None)
        if display_mesh is None or len(colors) != display_mesh.n_points:
            return
        if not self._apply_robot_dialog_sensor_colours(colors):
            return
        self._robot_dialog_sensor_last_frame = frame_sequence
        plotter = getattr(self, "_robot_dialog_plotter", None)
        if plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass

    def _set_robot_dialog_sensor_signal_enabled(self, enabled):
        timer = getattr(self, "_robot_dialog_sensor_signal_timer", None)
        actor = getattr(self, "_robot_dialog_sensor_actor", None)
        self._robot_dialog_sensor_last_frame = None
        if bool(enabled) and actor is not None:
            if timer is not None:
                timer.start()
            self._robot_dialog_sensor_signal_tick()
            return
        if timer is not None:
            timer.stop()
        self._restore_robot_dialog_sensor_orientation_colours(render=True)

    def _hide_robot_dialog_sensor_mapping(self):
        if (
            getattr(self, "_robot_dialog_admittance_active", False)
            and getattr(self, "_admittance_source", None) == "robot_dialog"
        ):
            self._teardown_robot_dialog_admittance(update_status=False)
        plotter = getattr(self, "_robot_dialog_plotter", None)
        actor = getattr(self, "_robot_dialog_sensor_actor", None)
        timer = getattr(self, "_robot_dialog_sensor_signal_timer", None)
        if timer is not None:
            timer.stop()
        self._hide_robot_dialog_control_center(render=False)
        if plotter is not None and actor is not None:
            try:
                plotter.remove_actor(actor, reset_camera=False)
            except Exception:
                pass
        self._robot_dialog_sensor_actor = None
        self._robot_dialog_sensor_mesh = None
        self._robot_dialog_sensor_local_mesh = None
        self._robot_dialog_sensor_last_frame = None
        if plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass

    def _toggle_robot_dialog_sensor_mapping(self, enabled):
        if not bool(enabled):
            self._hide_robot_dialog_sensor_mapping()
            status = getattr(self, "_robot_dialog_status", None)
            if status is not None:
                status.setText("Sensor mapping hidden")
            return True

        sensor, _source = self._current_built_sensor_geometry()
        if sensor is None:
            QMessageBox.information(
                getattr(self, "_robot_dialog", None),
                "Map Sensor",
                "The selected sensor geometry could not be created.\n\n"
                "Check the selected sensor model and its mesh/signal resources.",
            )
            return False

        self._robot_dialog_sensor_local_mesh = None
        config = self._load_robot_sensor_mapping_config()
        if not self._refresh_robot_dialog_sensor_overlay(config=config, render=True):
            QMessageBox.warning(
                getattr(self, "_robot_dialog", None),
                "Map Sensor",
                "The current sensor geometry could not be mapped.",
            )
            return False

        status = getattr(self, "_robot_dialog_status", None)
        if status is not None:
            status.setText(
                f"Sensor mapped to {self._robot_sensor_mapping_link_label(config['link_index'])}"
            )
        signal_button = getattr(self, "_robot_dialog_sensor_signal_btn", None)
        if signal_button is not None:
            self._set_robot_dialog_sensor_signal_enabled(
                signal_button.isChecked()
            )
        return True

    def _open_robot_dialog_sensor_mapping_dialog(self):
        parent_dialog = getattr(self, "_robot_dialog", None)
        if parent_dialog is None:
            return
        sensor, _source = self._current_built_sensor_geometry()
        if sensor is None:
            QMessageBox.information(
                parent_dialog,
                "Sensor Mount",
                "The selected sensor geometry could not be created.",
            )
            return

        existing = getattr(self, "_robot_dialog_sensor_mapping_dialog", None)
        if existing is not None:
            try:
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                self._robot_dialog_sensor_mapping_dialog = None

        config = self._normalize_robot_sensor_mapping_config(
            getattr(self, "_robot_dialog_sensor_mapping_config", None)
            or self._load_robot_sensor_mapping_config()
        )
        dialog = QDialog(parent_dialog)
        dialog.setWindowTitle(
            f"Sensor Mount Mapping — {self._robot_sensor_mapping_key()}"
        )
        dialog.setModal(False)
        dialog.resize(520, 760)
        layout = QVBoxLayout(dialog)
        description = QLabel(
            "Tune the transform from the selected robot link to the center of the "
            "currently built sensor. Units: metres and degrees. In the preview, "
            "red increases by sensor column and green increases by sensor row."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        link_combo = QComboBox(dialog)
        for link_index in range(8):
            link_combo.addItem(
                self._robot_sensor_mapping_link_label(link_index), link_index
            )
        selected = link_combo.findData(int(config["link_index"]))
        link_combo.setCurrentIndex(max(0, selected))
        form.addRow("Parent link:", link_combo)

        def _spin(value, minimum, maximum, step, decimals):
            widget = QDoubleSpinBox(dialog)
            widget.setRange(float(minimum), float(maximum))
            widget.setDecimals(int(decimals))
            widget.setSingleStep(float(step))
            widget.setValue(float(value))
            return widget

        tx, ty, tz = config["translation_m"]
        rx, ry, rz = config["rotation_deg"]
        tx_spin = _spin(tx, -2.0, 2.0, 0.005, 4)
        ty_spin = _spin(ty, -2.0, 2.0, 0.005, 4)
        tz_spin = _spin(tz, -2.0, 2.0, 0.005, 4)
        rx_spin = _spin(rx, -180.0, 180.0, 1.0, 2)
        ry_spin = _spin(ry, -180.0, 180.0, 1.0, 2)
        rz_spin = _spin(rz, -180.0, 180.0, 1.0, 2)
        scale_spin = _spin(config["scale"], 0.001, 100.0, 0.05, 4)
        horizontal_scale_spin = _spin(
            config["horizontal_scale"], 0.001, 100.0, 0.05, 4
        )
        vertical_scale_spin = _spin(
            config["vertical_scale"], 0.001, 100.0, 0.05, 4
        )
        horizontal_scale_spin.setToolTip(
            "Stretch the sensor along its local horizontal X direction."
        )
        vertical_scale_spin.setToolTip(
            "Stretch the sensor along its local vertical Y direction."
        )
        opacity_spin = _spin(config["opacity"], 0.05, 1.0, 0.05, 2)
        admittance_speed_spin = _spin(
            config["admittance_max_speed_mps"], 0.001, 0.1, 0.005, 3
        )
        admittance_threshold_spin = _spin(
            config["admittance_contact_threshold_pct"], 0.1, 99.0, 0.5, 2
        )
        admittance_full_scale_spin = _spin(
            config["admittance_full_scale_pct"], 0.2, 100.0, 0.5, 2
        )
        admittance_smoothing_spin = _spin(
            config["admittance_smoothing_alpha"], 0.01, 1.0, 0.05, 2
        )
        admittance_mode_combo = QComboBox(dialog)
        admittance_mode_combo.addItem("Surface Normal", "surface_normal")
        admittance_mode_combo.addItem("Centre-Directed", "centre_directed")
        selected_mode = admittance_mode_combo.findData(
            config["admittance_direction_mode"]
        )
        admittance_mode_combo.setCurrentIndex(max(0, selected_mode))
        center_x, center_y, center_z = config["admittance_control_center_m"]
        center_x_spin = _spin(center_x, -2.0, 2.0, 0.005, 4)
        center_y_spin = _spin(center_y, -2.0, 2.0, 0.005, 4)
        center_z_spin = _spin(center_z, -2.0, 2.0, 0.005, 4)
        btn_geometry_center = QPushButton("Use Geometry Centre", dialog)
        admittance_reverse_check = QCheckBox("Reverse motion direction", dialog)
        admittance_reverse_check.setChecked(
            bool(config["admittance_reverse_direction"])
        )
        admittance_speed_spin.setToolTip(
            "Safety-limited maximum linear TCP speed. Hard limit: 0.1 m/s."
        )
        admittance_threshold_spin.setToolTip(
            "Peak sensor pressure below this value is treated as no contact."
        )
        admittance_full_scale_spin.setToolTip(
            "Peak pressure that reaches the configured maximum speed."
        )
        admittance_smoothing_spin.setToolTip(
            "New-command weight from 0.01 (smooth) to 1.0 (immediate)."
        )
        admittance_mode_combo.setToolTip(
            "Surface Normal follows the contacted taxel normal. Centre-Directed "
            "moves from the pressure centroid toward the configured control centre."
        )
        for center_spin, axis in zip(
            (center_x_spin, center_y_spin, center_z_spin), "XYZ"
        ):
            center_spin.setToolTip(
                f"Control-centre {axis} coordinate in mapped sensor-local metres, "
                "after sensor scaling and before mount rotation."
            )
        btn_geometry_center.setToolTip(
            "Estimate the cylinder/surface centre from the current sensor points "
            "and normals. The value remains editable before Apply or Save."
        )
        admittance_reverse_check.setToolTip(
            "Default motion follows an inward push (-surface normal). "
            "Enable this if the mapped sensor normal points the other way."
        )

        form.addRow("X (m):", tx_spin)
        form.addRow("Y (m):", ty_spin)
        form.addRow("Z (m):", tz_spin)
        form.addRow("Roll / Rx (deg):", rx_spin)
        form.addRow("Pitch / Ry (deg):", ry_spin)
        form.addRow("Yaw / Rz (deg):", rz_spin)
        form.addRow("Overall scale:", scale_spin)
        form.addRow("Horizontal stretch (X):", horizontal_scale_spin)
        form.addRow("Vertical stretch (Y):", vertical_scale_spin)
        form.addRow("Opacity:", opacity_spin)
        form.addRow(QLabel("Pressure Admittance"))
        form.addRow("Direction mode:", admittance_mode_combo)
        form.addRow("Control centre X (m):", center_x_spin)
        form.addRow("Control centre Y (m):", center_y_spin)
        form.addRow("Control centre Z (m):", center_z_spin)
        form.addRow("Centre helper:", btn_geometry_center)
        form.addRow("Maximum speed (m/s):", admittance_speed_spin)
        form.addRow("Contact threshold (%):", admittance_threshold_spin)
        form.addRow("Full-speed pressure (%):", admittance_full_scale_spin)
        form.addRow("Velocity smoothing:", admittance_smoothing_spin)
        form.addRow("Direction:", admittance_reverse_check)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        btn_apply = QPushButton("Apply", dialog)
        btn_save = QPushButton("Save", dialog)
        btn_reset_mapping = QPushButton("Reset", dialog)
        btn_close = QPushButton("Close", dialog)
        buttons.addWidget(btn_apply)
        buttons.addWidget(btn_save)
        buttons.addWidget(btn_reset_mapping)
        buttons.addStretch(1)
        buttons.addWidget(btn_close)
        layout.addLayout(buttons)

        def _widget_config():
            return self._normalize_robot_sensor_mapping_config(
                {
                    "link_index": link_combo.currentData(),
                    "translation_m": [
                        tx_spin.value(),
                        ty_spin.value(),
                        tz_spin.value(),
                    ],
                    "rotation_deg": [
                        rx_spin.value(),
                        ry_spin.value(),
                        rz_spin.value(),
                    ],
                    "scale": scale_spin.value(),
                    "horizontal_scale": horizontal_scale_spin.value(),
                    "vertical_scale": vertical_scale_spin.value(),
                    "opacity": opacity_spin.value(),
                    "admittance_max_speed_mps": admittance_speed_spin.value(),
                    "admittance_contact_threshold_pct": admittance_threshold_spin.value(),
                    "admittance_full_scale_pct": admittance_full_scale_spin.value(),
                    "admittance_smoothing_alpha": admittance_smoothing_spin.value(),
                    "admittance_reverse_direction": admittance_reverse_check.isChecked(),
                    "admittance_direction_mode": admittance_mode_combo.currentData(),
                    "admittance_control_center_m": [
                        center_x_spin.value(),
                        center_y_spin.value(),
                        center_z_spin.value(),
                    ],
                }
            )

        def _use_geometry_center():
            suggestion_config = _widget_config()
            suggested = self._suggest_robot_sensor_control_center_m(
                sensor, suggestion_config
            )
            for widget, value in zip(
                (center_x_spin, center_y_spin, center_z_spin), suggested
            ):
                widget.setValue(float(value))

        def _on_mode_changed(_index):
            center_values = np.array(
                [
                    center_x_spin.value(),
                    center_y_spin.value(),
                    center_z_spin.value(),
                ],
                dtype=float,
            )
            if (
                admittance_mode_combo.currentData() == "centre_directed"
                and float(np.linalg.norm(center_values)) <= 1e-9
            ):
                _use_geometry_center()

        def _ensure_button_checked():
            button = getattr(self, "_robot_dialog_map_sensor_btn", None)
            if button is not None and not button.isChecked():
                button.blockSignals(True)
                button.setChecked(True)
                button.blockSignals(False)

        def _apply():
            new_config = _widget_config()
            if getattr(self, "_robot_dialog_admittance_active", False):
                self._teardown_robot_dialog_admittance(
                    status_text="Pressure admittance stopped: mapping changed"
                )
            self._robot_dialog_sensor_mapping_config = new_config
            self._robot_dialog_sensor_local_mesh = None
            if self._refresh_robot_dialog_sensor_overlay(new_config, render=True):
                _ensure_button_checked()
                signal_button = getattr(
                    self, "_robot_dialog_sensor_signal_btn", None
                )
                if signal_button is not None:
                    self._set_robot_dialog_sensor_signal_enabled(
                        signal_button.isChecked()
                    )
                status = getattr(self, "_robot_dialog_status", None)
                if status is not None:
                    status.setText(
                        "Sensor preview: "
                        + self._robot_sensor_mapping_link_label(
                            new_config["link_index"]
                        )
                    )

        def _save():
            _apply()
            new_config = _widget_config()
            if self._save_robot_sensor_mapping_config(new_config):
                status = getattr(self, "_robot_dialog_status", None)
                if status is not None:
                    status.setText(
                        f"Sensor mapping saved: {self._robot_sensor_mapping_key()}"
                    )
            else:
                QMessageBox.warning(dialog, "Sensor Mount", "Failed to save mapping.")

        def _reset():
            reset_config = self._default_robot_sensor_mapping_config()
            link_combo.setCurrentIndex(link_combo.findData(reset_config["link_index"]))
            for widget, value in zip(
                (tx_spin, ty_spin, tz_spin), reset_config["translation_m"]
            ):
                widget.setValue(value)
            for widget, value in zip(
                (rx_spin, ry_spin, rz_spin), reset_config["rotation_deg"]
            ):
                widget.setValue(value)
            scale_spin.setValue(reset_config["scale"])
            horizontal_scale_spin.setValue(reset_config["horizontal_scale"])
            vertical_scale_spin.setValue(reset_config["vertical_scale"])
            opacity_spin.setValue(reset_config["opacity"])
            admittance_speed_spin.setValue(
                reset_config["admittance_max_speed_mps"]
            )
            admittance_threshold_spin.setValue(
                reset_config["admittance_contact_threshold_pct"]
            )
            admittance_full_scale_spin.setValue(
                reset_config["admittance_full_scale_pct"]
            )
            admittance_smoothing_spin.setValue(
                reset_config["admittance_smoothing_alpha"]
            )
            admittance_reverse_check.setChecked(
                reset_config["admittance_reverse_direction"]
            )
            admittance_mode_combo.setCurrentIndex(
                admittance_mode_combo.findData(
                    reset_config["admittance_direction_mode"]
                )
            )
            for widget, value in zip(
                (center_x_spin, center_y_spin, center_z_spin),
                reset_config["admittance_control_center_m"],
            ):
                widget.setValue(value)
            _apply()

        btn_geometry_center.clicked.connect(_use_geometry_center)
        admittance_mode_combo.currentIndexChanged.connect(_on_mode_changed)
        _on_mode_changed(admittance_mode_combo.currentIndex())
        btn_apply.clicked.connect(_apply)
        btn_save.clicked.connect(_save)
        btn_reset_mapping.clicked.connect(_reset)
        btn_close.clicked.connect(dialog.close)
        dialog.finished.connect(
            lambda _result: setattr(
                self, "_robot_dialog_sensor_mapping_dialog", None
            )
        )

        self._robot_dialog_sensor_mapping_dialog = dialog
        dialog.show()
        _apply()

    # ------------------------------------------------------------------
    # Pressure-based normal admittance
    # ------------------------------------------------------------------
    def _ensure_pressure_admittance_timer(self):
        timer = getattr(self, "_robot_dialog_admittance_timer", None)
        if timer is not None:
            try:
                timer.isActive()
                return timer
            except RuntimeError:
                pass
        timer = QTimer(self.parent)
        timer.setInterval(33)
        timer.timeout.connect(self._robot_dialog_admittance_tick)
        self._robot_dialog_admittance_timer = timer
        return timer

    def _set_pressure_admittance_status(self, text):
        message = str(text or "")
        robot_status = getattr(self, "_robot_dialog_status", None)
        if robot_status is not None:
            try:
                robot_status.setText(message)
            except RuntimeError:
                pass
        ai_status = getattr(self.parent, "admittance_control_status_label", None)
        if ai_status is not None:
            ai_status.setText(message)

    def _sync_pressure_admittance_buttons(self, active, pending=False):
        checked = bool(active or pending)
        self.parent._admittance_control_active = bool(active)
        robot_button = getattr(self, "_robot_dialog_admittance_btn", None)
        self._set_checked_without_signal(robot_button, checked)

        ai_button = getattr(self.parent, "admittance_control_button", None)
        self._set_checked_without_signal(ai_button, checked)
        if ai_button is not None:
            ai_button.setText(
                "Starting Admittance Control..."
                if pending
                else (
                    "Stop Admittance Control"
                    if active
                    else "Start Admittance Control"
                )
            )
            set_active = getattr(self.parent, "_set_button_active", None)
            if callable(set_active):
                set_active(ai_button, checked)

    def is_pressure_admittance_active(self):
        return bool(getattr(self, "_robot_dialog_admittance_active", False))

    @staticmethod
    def _admittance_speed_from_pressure(
        peak_pressure, contact_threshold, full_scale_pressure, max_speed
    ):
        """Map peak sensor pressure to a bounded linear speed."""
        try:
            peak = max(0.0, float(peak_pressure))
            threshold = max(0.0, float(contact_threshold))
            full_scale = max(threshold + 1e-6, float(full_scale_pressure))
            speed_limit = float(np.clip(float(max_speed), 0.0, 0.1))
        except Exception:
            return 0.0
        ratio = float(np.clip((peak - threshold) / (full_scale - threshold), 0.0, 1.0))
        return speed_limit * ratio

    def _robot_dialog_other_velocity_control_name(self):
        """Return the name of another live robot-motion source, if any."""
        if getattr(self, "_robot_dialog_drag_active", False):
            return "3D robot drag control"

        parent = getattr(self, "parent", None)
        if parent is not None and bool(getattr(parent, "_direct_finger_active", False)):
            return "Direct Finger Motion"
        if parent is not None and bool(
            getattr(parent, "_ai_direct_finger_robot_active", False)
        ):
            return "AI DFM recording with robot motion"

        main_window = None
        if parent is not None:
            try:
                main_window = parent.window()
            except Exception:
                main_window = None
        if main_window is not None and bool(
            getattr(main_window, "_keyboard_tool_velocity_enabled", False)
        ):
            return "keyboard tool velocity"

        sensor = getattr(parent, "sensor_functions", None)
        helper_specs = (
            ("direct_finger_motion_class", "Direct Finger Motion", "is_running"),
            ("console_control_class", "console sensor control", "is_running"),
            ("proximity_control_class", "proximity control", "is_running"),
            (
                "threelevel_hierarchical_transformer_class",
                "three-level robot control",
                "is_recognizing_gesture",
            ),
        )
        for attr_name, label, state_name in helper_specs:
            helper = getattr(sensor, attr_name, None) if sensor is not None else None
            if helper is None or helper.__class__.__name__.startswith("_"):
                continue
            if bool(getattr(helper, state_name, False)):
                return label

        ai_record = (
            getattr(sensor, "ai_direct_finger_motion_class", None)
            if sensor is not None
            else None
        )
        if (
            ai_record is not None
            and not ai_record.__class__.__name__.startswith("_")
            and bool(getattr(ai_record, "is_running", False))
            and bool(getattr(ai_record, "send_robot_commands", False))
        ):
            return "AI DFM recording with robot motion"

        ai_execute = (
            getattr(sensor, "ai_direct_finger_motion_execution_class", None)
            if sensor is not None
            else None
        )
        if (
            ai_execute is not None
            and not ai_execute.__class__.__name__.startswith("_")
            and bool(getattr(ai_execute, "is_running", False))
            and not bool(getattr(ai_execute, "dry_run_predictions_only", True))
        ):
            return "AI Direct Finger Motion execution"
        return None

    @staticmethod
    def _interpolate_sensor_grid_vector(vectors, n_row, n_col, estimate):
        values = np.asarray(vectors, dtype=float)
        if values.shape != (int(n_row) * int(n_col), 3):
            return None
        try:
            center_row = float(
                np.clip(float(estimate["center_row"]), 0.0, int(n_row) - 1)
            )
            center_col = float(
                np.clip(float(estimate["center_col"]), 0.0, int(n_col) - 1)
            )
        except Exception:
            return None

        row0 = int(np.floor(center_row))
        col0 = int(np.floor(center_col))
        row1 = min(row0 + 1, int(n_row) - 1)
        col1 = min(col0 + 1, int(n_col) - 1)
        row_fraction = center_row - row0
        col_fraction = center_col - col0

        def _at(row, col):
            return values[col * int(n_row) + row]

        result = (
            _at(row0, col0)
            * (1.0 - row_fraction)
            * (1.0 - col_fraction)
            + _at(row1, col0) * row_fraction * (1.0 - col_fraction)
            + _at(row0, col1) * (1.0 - row_fraction) * col_fraction
            + _at(row1, col1) * row_fraction * col_fraction
        )
        if not np.all(np.isfinite(result)):
            return None
        return np.asarray(result, dtype=float)

    @staticmethod
    def _sensor_geometry_visual_center(sensor):
        source = getattr(sensor, "_2D_map", None)
        points = np.asarray(getattr(source, "points", None), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            points = np.asarray(getattr(sensor, "points_origin", None), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            return None
        finite = points[np.all(np.isfinite(points), axis=1)]
        if len(finite) == 0:
            return None
        return np.mean(finite, axis=0)

    def _robot_dialog_contact_point_sensor_m(self, sensor, estimate, config):
        n_row = int(getattr(sensor, "n_row", 0) or 0)
        n_col = int(getattr(sensor, "n_col", 0) or 0)
        point = self._interpolate_sensor_grid_vector(
            getattr(sensor, "points_origin", None), n_row, n_col, estimate
        )
        visual_center = self._sensor_geometry_visual_center(sensor)
        if point is None or visual_center is None:
            return None

        config = self._normalize_robot_sensor_mapping_config(config)
        scale = float(config["scale"])
        local_scale = np.diag(
            [
                scale * float(config["horizontal_scale"]),
                scale * float(config["vertical_scale"]),
                scale,
            ]
        )
        return local_scale @ (point - visual_center)

    def _suggest_robot_sensor_control_center_m(self, sensor, config):
        points = np.asarray(getattr(sensor, "points_origin", None), dtype=float)
        normals = np.asarray(getattr(sensor, "normals", None), dtype=float)
        visual_center = self._sensor_geometry_visual_center(sensor)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or normals.shape != points.shape
            or visual_center is None
        ):
            return np.zeros(3, dtype=float)

        normal_norms = np.linalg.norm(normals, axis=1)
        valid = (
            np.all(np.isfinite(points), axis=1)
            & np.all(np.isfinite(normals), axis=1)
            & np.isfinite(normal_norms)
            & (normal_norms > 1e-9)
        )
        if not np.any(valid):
            return np.zeros(3, dtype=float)

        valid_points = points[valid]
        valid_normals = normals[valid] / normal_norms[valid, None]
        matrix = np.zeros((3, 3), dtype=float)
        vector = np.zeros(3, dtype=float)
        identity = np.eye(3, dtype=float)
        for point, normal in zip(valid_points, valid_normals):
            projection = identity - np.outer(normal, normal)
            matrix += projection
            vector += projection @ point

        if np.linalg.matrix_rank(matrix, tol=1e-8) >= 3:
            raw_center = np.linalg.lstsq(matrix, vector, rcond=None)[0]
        else:
            mean_normal = np.mean(valid_normals, axis=0)
            mean_norm = float(np.linalg.norm(mean_normal))
            if mean_norm <= 1e-9:
                mean_normal = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                mean_normal /= mean_norm
            spans = np.ptp(valid_points, axis=0)
            positive_spans = spans[spans > 1e-6]
            depth = (
                0.5 * float(np.min(positive_spans))
                if len(positive_spans)
                else 0.05
            )
            raw_center = visual_center - mean_normal * depth

        config = self._normalize_robot_sensor_mapping_config(config)
        scale = float(config["scale"])
        local_scale = np.diag(
            [
                scale * float(config["horizontal_scale"]),
                scale * float(config["vertical_scale"]),
                scale,
            ]
        )
        suggested = local_scale @ (raw_center - visual_center)
        if not np.all(np.isfinite(suggested)):
            return np.zeros(3, dtype=float)
        return suggested

    def _robot_dialog_centre_direction_base(self, sensor, estimate, joints, config):
        config = self._normalize_robot_sensor_mapping_config(config)
        contact_sensor = self._robot_dialog_contact_point_sensor_m(
            sensor, estimate, config
        )
        if contact_sensor is None:
            return None
        center_sensor = np.asarray(
            config["admittance_control_center_m"], dtype=float
        )
        direction_sensor = center_sensor - contact_sensor
        direction_norm = float(np.linalg.norm(direction_sensor))
        if direction_norm <= 1e-9 or not np.all(np.isfinite(direction_sensor)):
            return None
        direction_sensor /= direction_norm

        joints_array = np.asarray(list(joints)[:6], dtype=float)
        if joints_array.shape != (6,) or not np.all(np.isfinite(joints_array)):
            return None
        chain = self._compute_kinematic_chain(joints_array)
        link_transform = chain[int(config["link_index"])]
        if link_transform is None:
            link_transform = np.eye(4, dtype=float)
        mount_rotation = self._robot_sensor_mount_rotation(config)
        if mount_rotation is None:
            return None
        base_direction = (
            np.asarray(link_transform, dtype=float)[:3, :3]
            @ mount_rotation
            @ direction_sensor
        )
        base_norm = float(np.linalg.norm(base_direction))
        if base_norm <= 1e-9 or not np.all(np.isfinite(base_direction)):
            return None
        return base_direction / base_norm

    def _robot_dialog_admittance_direction_base(
        self, sensor, estimate, joints, config
    ):
        config = self._normalize_robot_sensor_mapping_config(config)
        mode = config["admittance_direction_mode"]
        if mode == "centre_directed":
            direction = self._robot_dialog_centre_direction_base(
                sensor, estimate, joints, config
            )
        else:
            surface_normal = self._robot_dialog_contact_surface_normal_base(
                sensor, estimate, joints, config
            )
            direction = None if surface_normal is None else -surface_normal
        if direction is None:
            return None
        if config["admittance_reverse_direction"]:
            direction = -direction
        return direction

    def _robot_dialog_contact_surface_normal_base(self, sensor, estimate, joints, config):
        """Transform the contacted taxel's surface normal into the base frame."""
        n_row = int(getattr(sensor, "n_row", 0) or 0)
        n_col = int(getattr(sensor, "n_col", 0) or 0)
        normals = np.asarray(getattr(sensor, "normals", None), dtype=float)
        if n_row <= 0 or n_col <= 0 or normals.shape != (n_row * n_col, 3):
            return None

        try:
            center_row = float(
                np.clip(float(estimate["center_row"]), 0.0, n_row - 1)
            )
            center_col = float(
                np.clip(float(estimate["center_col"]), 0.0, n_col - 1)
            )
            row0 = int(np.floor(center_row))
            col0 = int(np.floor(center_col))
            row1 = min(row0 + 1, n_row - 1)
            col1 = min(col0 + 1, n_col - 1)
            row_fraction = center_row - row0
            col_fraction = center_col - col0

            def _normal_at(row, col):
                return normals[col * n_row + row]

            local_normal = (
                _normal_at(row0, col0)
                * (1.0 - row_fraction)
                * (1.0 - col_fraction)
                + _normal_at(row1, col0)
                * row_fraction
                * (1.0 - col_fraction)
                + _normal_at(row0, col1)
                * (1.0 - row_fraction)
                * col_fraction
                + _normal_at(row1, col1) * row_fraction * col_fraction
            )
            if not np.all(np.isfinite(local_normal)):
                return None
            local_norm = float(np.linalg.norm(local_normal))
            if local_norm <= 1e-9:
                return None
            local_normal /= local_norm

            joints_array = np.asarray(list(joints)[:6], dtype=float)
            if joints_array.shape != (6,) or not np.all(np.isfinite(joints_array)):
                return None
            chain = self._compute_kinematic_chain(joints_array)
            config = self._normalize_robot_sensor_mapping_config(config)
            link_index = int(config["link_index"])
            link_transform = chain[link_index]
            if link_transform is None:
                link_transform = np.eye(4, dtype=float)
            mount_linear = self._robot_sensor_local_transform(config)[:3, :3]
            combined_linear = np.asarray(link_transform, dtype=float)[:3, :3] @ mount_linear
            base_normal = np.linalg.inv(combined_linear).T @ local_normal
            base_norm = float(np.linalg.norm(base_normal))
            if base_norm <= 1e-9 or not np.all(np.isfinite(base_normal)):
                return None
            return base_normal / base_norm
        except Exception:
            return None

    def _send_robot_dialog_admittance_velocity(self, velocity):
        api = getattr(self.parent, "robot_api", None)
        if api is None:
            return False
        linear = [float(value) for value in np.asarray(velocity, dtype=float)[:3]]
        try:
            if hasattr(api, "send_end_effector_velocity_in_frame"):
                return bool(
                    api.send_end_effector_velocity_in_frame(
                        linear,
                        (0.0, 0.0, 0.0),
                        frame="base",
                        ensure_mode=False,
                    )
                )
            command = linear + [0.0, 0.0, 0.0]
            if hasattr(api, "send_end_effector_velocity"):
                return bool(api.send_end_effector_velocity(command, ensure_mode=False))
            if hasattr(api, "send_request") and hasattr(
                api, "set_end_effector_velocity"
            ):
                return bool(api.send_request(api.set_end_effector_velocity(command)))
        except Exception as exc:
            print(f"[PressureAdmittance] Velocity send failed: {exc}")
        return False

    def _toggle_robot_dialog_admittance(
        self,
        enabled,
        require_visual_mapping=True,
        source="robot_dialog",
        dialog_parent=None,
    ):
        button = (
            getattr(self, "_robot_dialog_admittance_btn", None)
            if source == "robot_dialog"
            else getattr(self.parent, "admittance_control_button", None)
        )
        dialog = dialog_parent or getattr(self, "_robot_dialog", None) or self.parent
        if not bool(enabled):
            self._teardown_robot_dialog_admittance()
            return

        sensor = getattr(self.parent, "sensor_functions", None)
        sensor_running = bool(
            sensor is not None
            and getattr(sensor, "_sensor_reader_is_running", lambda: False)()
            and getattr(sensor, "is_connected", False)
        )
        map_button = getattr(self, "_robot_dialog_map_sensor_btn", None)
        if not sensor_running:
            self._set_checked_without_signal(button, False)
            self._sync_pressure_admittance_buttons(False)
            self._set_pressure_admittance_status(
                "Pressure admittance: sensor is not calibrated"
            )
            QMessageBox.information(
                dialog,
                "Pressure Admittance",
                "Start and calibrate the selected sensor before enabling admittance control.",
            )
            return
        if (
            require_visual_mapping
            and (
                map_button is None
                or not map_button.isChecked()
                or getattr(self, "_robot_dialog_sensor_actor", None) is None
            )
        ):
            self._set_checked_without_signal(button, False)
            self._sync_pressure_admittance_buttons(False)
            self._set_pressure_admittance_status(
                "Pressure admittance: sensor mapping is not active"
            )
            QMessageBox.information(
                dialog,
                "Pressure Admittance",
                "Map the sensor to the correct robot link before enabling motion.",
            )
            return

        conflict = self._robot_dialog_other_velocity_control_name()
        if conflict:
            self._set_checked_without_signal(button, False)
            self._sync_pressure_admittance_buttons(False)
            self._set_pressure_admittance_status(
                f"Pressure admittance: stop {conflict} first"
            )
            QMessageBox.warning(
                dialog,
                "Pressure Admittance",
                f"Stop {conflict} before enabling pressure admittance.",
            )
            return

        api = getattr(self.parent, "robot_api", None)
        required = bool(
            api is not None
            and hasattr(api, "get_current_positions")
            and hasattr(api, "enter_end_effector_velocity_mode")
            and hasattr(api, "exit_end_effector_velocity_mode")
            and (
                hasattr(api, "send_end_effector_velocity_in_frame")
                or hasattr(api, "send_end_effector_velocity")
            )
        )
        if not required:
            self._set_checked_without_signal(button, False)
            self._sync_pressure_admittance_buttons(False)
            self._set_pressure_admittance_status(
                "Pressure admittance: robot velocity API unavailable"
            )
            return

        if not getattr(self, "_robot_dialog_admittance_confirmed", False):
            config = self._normalize_robot_sensor_mapping_config(
                (
                    getattr(self, "_robot_dialog_sensor_mapping_config", None)
                    if source == "robot_dialog"
                    else None
                )
                or self._load_robot_sensor_mapping_config()
            )
            box = QMessageBox(dialog)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Pressure Admittance: real robot will move")
            direction_label = (
                "Centre-Directed"
                if config["admittance_direction_mode"] == "centre_directed"
                else "Surface Normal"
            )
            box.setText(
                "Touching the mapped sensor will command the REAL robot in velocity mode.\n\n"
                f"Direction mode: {direction_label}\n"
                f"Maximum speed: {config['admittance_max_speed_mps']:.3f} m/s\n"
                "No contact sends zero velocity immediately. Keep an emergency stop ready."
            )
            box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Cancel)
            if box.exec_() != QMessageBox.Ok:
                self._set_checked_without_signal(button, False)
                self._sync_pressure_admittance_buttons(False)
                self._set_pressure_admittance_status(
                    "Pressure admittance: cancelled"
                )
                return
            self._robot_dialog_admittance_confirmed = True

        try:
            entered = api.enter_end_effector_velocity_mode(suspend_existing=True)
        except Exception as exc:
            entered = False
            print(f"[PressureAdmittance] Could not enter velocity mode: {exc}")
        if not entered:
            self._set_checked_without_signal(button, False)
            self._sync_pressure_admittance_buttons(False)
            self._set_pressure_admittance_status(
                "Pressure admittance: velocity mode failed"
            )
            return

        self._admittance_mapping_config = self._normalize_robot_sensor_mapping_config(
            (
                getattr(self, "_robot_dialog_sensor_mapping_config", None)
                if source == "robot_dialog"
                else None
            )
            or self._load_robot_sensor_mapping_config()
        )
        self._admittance_source = str(source)
        self._admittance_pending_start = False
        self._robot_dialog_admittance_velocity_mode_on = True
        self._robot_dialog_admittance_active = True
        self._robot_dialog_admittance_filtered_velocity = np.zeros(3, dtype=float)
        self._ensure_pressure_admittance_timer().start()
        if source == "robot_dialog":
            live_button = getattr(self, "_robot_dialog_live_btn", None)
            if live_button is not None and not live_button.isChecked():
                live_button.setChecked(True)
        self._sync_pressure_admittance_buttons(True)
        self._set_pressure_admittance_status(
            "Pressure admittance: active, waiting for contact"
        )
        self._robot_dialog_admittance_tick()

    def set_ai_admittance_control_enabled(self, enabled):
        """Start/stop admittance from the AI tab without opening a 3D view."""
        if not bool(enabled):
            self._admittance_pending_start = False
            self._teardown_robot_dialog_admittance(
                status_text="Pressure admittance: idle"
            )
            return

        if self.is_pressure_admittance_active():
            self._sync_pressure_admittance_buttons(True)
            return

        sensor = getattr(self.parent, "sensor_functions", None)
        if sensor is None:
            self._sync_pressure_admittance_buttons(False)
            self._set_pressure_admittance_status(
                "Pressure admittance: sensor functions unavailable"
            )
            return

        running = bool(
            getattr(sensor, "_sensor_reader_is_running", lambda: False)()
            and getattr(sensor, "is_connected", False)
        )
        if running:
            self._toggle_robot_dialog_admittance(
                True,
                require_visual_mapping=False,
                source="ai_tab",
                dialog_parent=self.parent,
            )
            return

        if not hasattr(sensor, "start_external_visualization_stream"):
            self._sync_pressure_admittance_buttons(False)
            self._set_pressure_admittance_status(
                "Pressure admittance: sensor auto-start unavailable"
            )
            return

        self._admittance_pending_start = True
        self._sync_pressure_admittance_buttons(False, pending=True)
        self._set_pressure_admittance_status(
            "Pressure admittance: starting and calibrating sensor..."
        )
        try:
            started = bool(sensor.start_external_visualization_stream())
            if started and hasattr(sensor, "set_main_visualization_enabled"):
                sensor.set_main_visualization_enabled(True, render=True)
        except Exception as exc:
            started = False
            print(f"[PressureAdmittance] Sensor auto-start failed: {exc}")
        if not started:
            self._admittance_pending_start = False
            self._sync_pressure_admittance_buttons(False)
            detail = (
                sensor.get_last_sensor_stream_error()
                if hasattr(sensor, "get_last_sensor_stream_error")
                else ""
            )
            self._set_pressure_admittance_status(
                detail or "Pressure admittance: sensor could not start"
            )
            return
        self._finish_ai_admittance_sensor_start(attempt=0)

    def _finish_ai_admittance_sensor_start(self, attempt=0):
        if not getattr(self, "_admittance_pending_start", False):
            return
        sensor = getattr(self.parent, "sensor_functions", None)
        running = bool(
            sensor is not None
            and getattr(sensor, "_sensor_reader_is_running", lambda: False)()
            and getattr(sensor, "is_connected", False)
        )
        if running:
            self._admittance_pending_start = False
            self._toggle_robot_dialog_admittance(
                True,
                require_visual_mapping=False,
                source="ai_tab",
                dialog_parent=self.parent,
            )
            return

        calibrating = bool(
            sensor is not None
            and getattr(sensor, "_sensor_calibration_in_progress", False)
        )
        if (calibrating and int(attempt) < 300) or int(attempt) < 5:
            QTimer.singleShot(
                100,
                lambda: self._finish_ai_admittance_sensor_start(int(attempt) + 1),
            )
            return

        self._admittance_pending_start = False
        self._sync_pressure_admittance_buttons(False)
        detail = (
            sensor.get_last_sensor_stream_error()
            if sensor is not None and hasattr(sensor, "get_last_sensor_stream_error")
            else ""
        )
        self._set_pressure_admittance_status(
            detail or "Pressure admittance: sensor calibration failed"
        )

    def _teardown_robot_dialog_admittance(self, status_text=None, update_status=True):
        self._admittance_pending_start = False
        timer = getattr(self, "_robot_dialog_admittance_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

        was_active = bool(
            getattr(self, "_robot_dialog_admittance_active", False)
            or getattr(self, "_robot_dialog_admittance_velocity_mode_on", False)
        )
        self._robot_dialog_admittance_active = False
        api = getattr(self.parent, "robot_api", None)
        if api is not None and getattr(
            self, "_robot_dialog_admittance_velocity_mode_on", False
        ):
            try:
                api.exit_end_effector_velocity_mode(send_zero=True)
            except Exception as exc:
                print(f"[PressureAdmittance] Velocity teardown failed: {exc}")
        self._robot_dialog_admittance_velocity_mode_on = False
        self._robot_dialog_admittance_filtered_velocity = np.zeros(3, dtype=float)
        self._admittance_source = None
        self._admittance_mapping_config = None
        self._sync_pressure_admittance_buttons(False)
        if update_status and (was_active or status_text):
            self._set_pressure_admittance_status(
                status_text or "Pressure admittance: stopped"
            )

    def _robot_dialog_admittance_tick(self):
        if not getattr(self, "_robot_dialog_admittance_active", False):
            return

        conflict = self._robot_dialog_other_velocity_control_name()
        if conflict:
            self._teardown_robot_dialog_admittance(
                status_text=f"Pressure admittance stopped: {conflict} became active"
            )
            return

        sensor = getattr(self.parent, "sensor_functions", None)
        running = bool(
            sensor is not None
            and getattr(sensor, "_sensor_reader_is_running", lambda: False)()
            and getattr(sensor, "is_connected", False)
        )
        if not running:
            self._teardown_robot_dialog_admittance(
                status_text="Pressure admittance stopped: sensor stream lost"
            )
            return

        config = self._normalize_robot_sensor_mapping_config(
            getattr(self, "_admittance_mapping_config", None)
            or getattr(self, "_robot_dialog_sensor_mapping_config", None)
            or self._load_robot_sensor_mapping_config()
        )
        data_obj = getattr(sensor, "_data", None)
        raw_matrix = getattr(data_obj, "diffPerData", None)
        averaged_matrix = getattr(data_obj, "diffPerDataAve", None)
        try:
            raw_estimate = sensor._estimate_contact_force_signal(
                raw_matrix,
                peak_threshold=config["admittance_contact_threshold_pct"],
            )
            estimate = (
                sensor._estimate_contact_force_signal(
                    averaged_matrix,
                    peak_threshold=config["admittance_contact_threshold_pct"],
                )
                if raw_estimate is not None
                else None
            )
            if estimate is None:
                estimate = raw_estimate
        except Exception:
            estimate = None

        if estimate is None:
            self._robot_dialog_admittance_filtered_velocity = np.zeros(3, dtype=float)
            if not self._send_robot_dialog_admittance_velocity(np.zeros(3, dtype=float)):
                self._teardown_robot_dialog_admittance(
                    status_text="Pressure admittance stopped: velocity send failed"
                )
                return
            self._set_pressure_admittance_status(
                "Pressure admittance: active, no contact"
            )
            return

        api = getattr(self.parent, "robot_api", None)
        try:
            joints = api.get_current_positions()
        except Exception:
            joints = None
        if joints is None or len(joints) < 6:
            self._robot_dialog_admittance_filtered_velocity = np.zeros(3, dtype=float)
            self._send_robot_dialog_admittance_velocity(np.zeros(3, dtype=float))
            self._set_pressure_admittance_status(
                "Pressure admittance: waiting for joint feedback"
            )
            return

        motion_direction = self._robot_dialog_admittance_direction_base(
            sensor, estimate, joints, config
        )
        if motion_direction is None:
            self._robot_dialog_admittance_filtered_velocity = np.zeros(3, dtype=float)
            self._send_robot_dialog_admittance_velocity(np.zeros(3, dtype=float))
            self._set_pressure_admittance_status(
                "Pressure admittance: invalid mapped direction"
            )
            return

        speed = self._admittance_speed_from_pressure(
            estimate["peak_pressure"],
            config["admittance_contact_threshold_pct"],
            config["admittance_full_scale_pct"],
            config["admittance_max_speed_mps"],
        )
        target_velocity = speed * motion_direction
        alpha = float(config["admittance_smoothing_alpha"])
        previous = np.asarray(
            getattr(
                self,
                "_robot_dialog_admittance_filtered_velocity",
                np.zeros(3, dtype=float),
            ),
            dtype=float,
        )
        filtered = previous + alpha * (target_velocity - previous)
        filtered_norm = float(np.linalg.norm(filtered))
        max_speed = float(config["admittance_max_speed_mps"])
        if filtered_norm > max_speed:
            filtered *= max_speed / filtered_norm
        self._robot_dialog_admittance_filtered_velocity = filtered

        if not self._send_robot_dialog_admittance_velocity(filtered):
            self._teardown_robot_dialog_admittance(
                status_text="Pressure admittance stopped: velocity send failed"
            )
            return
        mode_label = (
            "Centre"
            if config["admittance_direction_mode"] == "centre_directed"
            else "Normal"
        )
        self._set_pressure_admittance_status(
            "Admittance {0}: p={1:.1f}% |v|={2:.3f} m/s "
            "d=({3:+.2f},{4:+.2f},{5:+.2f})".format(
                mode_label,
                float(estimate["peak_pressure"]),
                float(np.linalg.norm(filtered)),
                float(motion_direction[0]),
                float(motion_direction[1]),
                float(motion_direction[2]),
            )
        )

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
        if getattr(self, "_robot_dialog_sensor_actor", None) is not None:
            self._refresh_robot_dialog_sensor_overlay(render=False)
        if getattr(self, "_robot_dialog_link5_frame_visible", False):
            self._update_robot_dialog_link5_frame(render=False)
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
            origin, _axes = self._link_frame_origin_and_axes(
                self._robot_dialog_link5_transform()
            )
            if origin is not None:
                status_label.setText(
                    "Live | L5=({0:+.3f}, {1:+.3f}, {2:+.3f}) m".format(
                        float(origin[0]), float(origin[1]), float(origin[2])
                    )
                )
            else:
                status_label.setText("Live: following robot")

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

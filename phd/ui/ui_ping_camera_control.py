from __future__ import annotations

import os
import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class CameraControlMixin:
    def _set_auto_center_ui_state(self, active: bool):
        self.centering_active = active
        self.auto_center_button.blockSignals(True)
        self.auto_center_button.setChecked(active)
        self.auto_center_button.blockSignals(False)
        self.auto_center_button.setText("Stop Auto-Centering" if active else "Auto-Center on object")
        self._set_button_active(self.auto_center_button, active)

    def _stop_auto_centering(self):
        if self.centering_active or self.auto_center_button.isChecked():
            self.send_velocity_command(0.0, 0.0, 0.0)
        self._set_auto_center_ui_state(False)

    def _set_live_camera_ui_state(self, running: bool):
        self.live_yolo_button.setText(
            "Stop Live Object Detection" if running else "Start Live Object Detection"
        )
        self._set_button_active(self.live_yolo_button, running)
        self.auto_center_button.setEnabled(running and self.features.get("robot_ready", False))
        if not running:
            self._set_auto_center_ui_state(False)

    def _ensure_camera_window(self):
        if self.cam_window is not None:
            return

        self.cam_window = QWidget()
        self.cam_window.setWindowTitle("Live Object Detection")
        self.cam_window.resize(640, 480)

        self.cam_label = QLabel(self.cam_window)
        self.cam_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.cam_label)
        self.cam_window.setLayout(layout)

    def _close_camera_window(self):
        if self.cam_window is None:
            return

        self.cam_window.close()
        self.cam_window.deleteLater()
        self.cam_window = None
        self.cam_label = None

    def _stop_yolo_worker(self):
        worker = self.yolo_worker
        self.yolo_worker = None
        if worker is None:
            return

        try:
            worker.stop()
        except Exception as exc:
            self._startup_log(f"⚠️ Failed to stop camera worker cleanly: {exc}")

    def _shutdown_toggle_helper(self, attr_name, state_attr, stop_method_name, feature_label):
        helper = self._get_sensor_helper(attr_name)
        if helper is None or not bool(getattr(helper, state_attr, False)):
            return

        try:
            getattr(helper, stop_method_name)()
        except Exception as exc:
            self._startup_log(f"⚠️ Failed to stop {feature_label} cleanly: {exc}")

    def _shutdown_recording_helper(self):
        helper = self._get_sensor_helper("record_gesture_class")
        if helper is None or not bool(getattr(helper, "is_recording", False)):
            return

        try:
            helper.is_recording = False
            timer = getattr(helper, "timer_record_gesture", None)
            if timer is not None:
                timer.stop()
            auto_timer = getattr(helper, "auto_timer", None)
            if auto_timer is not None:
                auto_timer.stop()
        except Exception as exc:
            self._startup_log(f"⚠️ Failed to stop gesture recording cleanly: {exc}")

    def shutdown(self):
        if self._is_shutting_down:
            return

        self._is_shutting_down = True
        self.manual_watchdog_timer.stop()
        self.manual_mode_active = False
        self.is_lifting = False
        self.grab_triggered = False

        self._stop_auto_centering()
        self._stop_yolo_worker()
        self._close_camera_window()
        self._shutdown_toggle_helper(
            "threelevel_hierarchical_transformer_class",
            "is_recognizing_gesture",
            "toggle_gesture_recognition",
            "3-Level recognition",
        )
        self._shutdown_toggle_helper(
            "direct_finger_motion_class",
            "is_running",
            "toggle_direct_finger_motion",
            "Direct Finger Motion",
        )
        self._shutdown_toggle_helper(
            "ai_direct_finger_motion_class",
            "is_running",
            "toggle_ai_direct_finger_motion",
            "AI Direct Finger Motion recording",
        )
        self._shutdown_toggle_helper(
            "ai_direct_finger_motion_execution_class",
            "is_running",
            "toggle_ai_direct_finger_motion_execution",
            "AI Direct Finger Motion execution",
        )
        self._shutdown_recording_helper()

        dialog = getattr(self, "direct_finger_motion_settings_dialog", None)
        if dialog is not None:
            dialog.close()

    def toggle_yolo_camera(self):
        if self._is_shutting_down:
            return

        if not self.features.get("camera_ready", False):
            self.log_display.append("⚠️ Camera feature is unavailable.")
            self._set_live_camera_ui_state(False)
            return

        camera_device = "/dev/video8"
        if not self.yolo_worker and not os.path.exists(camera_device):
            self.log_display.append(f"⚠️ Camera device not found: {camera_device}")
            self._set_live_camera_ui_state(False)
            return

        if self.yolo_worker and self.yolo_worker.isRunning():
            self._stop_auto_centering()
            self._stop_yolo_worker()
            self._close_camera_window()
            self._set_live_camera_ui_state(False)
        else:
            self._ensure_camera_window()
            worker_class = getattr(self, "_yolo_worker_class", None)

            try:
                if worker_class is None:
                    raise RuntimeError("Camera worker is unavailable.")

                self.yolo_worker = worker_class(dev_source=camera_device)
                self.yolo_worker.image_signal.connect(self.update_cam_window)
                self.yolo_worker.data_signal.connect(self.process_yolo_data)
                self.yolo_worker.finished_signal.connect(self.reset_yolo_button_state)
                self.yolo_worker.start()
                self.cam_window.show()
            except Exception as exc:
                self._stop_yolo_worker()
                self._close_camera_window()
                self.log_display.append(f"❌ Failed to start camera: {exc}")
                self._set_live_camera_ui_state(False)
                return

            self._set_live_camera_ui_state(True)

    def toggle_centering_mode(self):
        if not self.features.get("robot_ready", False):
            self._set_auto_center_ui_state(False)
            self.log_display.append("⚠️ Robot API unavailable: auto-centering disabled.")
            return

        self.centering_active = self.auto_center_button.isChecked()

        if self.centering_active:
            self._set_auto_center_ui_state(True)
            print("Auto-Centering ACTIVATED. Enabling Robot Velocity Mode...")

            try:
                self.robot_api.send_request(self.robot_api.suspend_end_effector_velocity_mode())
                self.robot_api.send_request(self.robot_api.enable_end_effector_velocity_mode())
            except Exception as exc:
                print(f"Failed to enable velocity mode: {exc}")
        else:
            print("Auto-Centering STOPPED. Sending 0 velocity...")
            self._stop_auto_centering()

    def process_yolo_data(self, detections, frame_size):
        if self._is_shutting_down:
            return

        if not self.centering_active:
            self.grab_triggered = False
            return

        if self.manual_mode_active:
            return

        frame_w, frame_h = frame_size
        center_x = frame_w / 2
        center_y = frame_h / 2

        speed = 0.02
        tolerance = 40
        target_distance = 0.18

        cmd_x = 0.0
        cmd_y = 0.0
        cmd_z = 0.0

        if not hasattr(self, "grab_triggered"):
            self.grab_triggered = False

        if self.grab_triggered:
            if not self.is_lifting:
                self.send_velocity_command(0.0, 0.0, 0.0)
            return

        ready_to_grab = False

        for obj in detections:
            if obj["name"] != "mouse":
                continue

            x1, y1, x2, y2 = obj["box"]
            obj_cx = (x1 + x2) / 2
            obj_cy = (y1 + y2) / 2
            current_dist = obj["depth"]

            diff_x = obj_cx - center_x
            diff_y = obj_cy - center_y

            aligned_x = False
            if diff_x > tolerance:
                cmd_x = speed
            elif diff_x < -tolerance:
                cmd_x = -speed
            else:
                cmd_x = 0.0
                aligned_x = True

            aligned_y = False
            if diff_y > tolerance:
                cmd_y = speed
            elif diff_y < -tolerance:
                cmd_y = -speed
            else:
                cmd_y = 0.0
                aligned_y = True

            if aligned_x and aligned_y:
                if current_dist > 0 and current_dist > target_distance:
                    cmd_z = speed
                    print(f"Approaching... {current_dist:.3f}m")
                elif current_dist > 0 and current_dist <= target_distance:
                    cmd_z = 0.0
                    ready_to_grab = True
                    print(f"Target Reached! Dist: {current_dist:.3f}m")
                else:
                    cmd_z = 0.0
            else:
                cmd_z = 0.0

            break

        self.send_velocity_command(cmd_x, cmd_y, cmd_z)

        if ready_to_grab and not self.grab_triggered:
            print("\n>>> TARGET REACHED <<<")
            print(f"1. Position BEFORE closing: {self.gripper.get_pos_string()}")
            self.grab_triggered = True
            self.gripper.close(soft=True)
            QTimer.singleShot(2000, self.check_grip_result)

    def check_grip_result(self):
        if self._is_shutting_down:
            return

        action, self.grip_fail_count = self.gripper.evaluate_grip_attempt(self.grip_fail_count)

        if action == self.gripper.ACTION_LIFT:
            self.perform_lift_action()
        elif action == self.gripper.ACTION_RETRY:
            self.reset_grab_flag()
        elif action == self.gripper.ACTION_RETRY_OPEN:
            self.gripper.open()
            QTimer.singleShot(2000, self.reset_grab_flag)
        elif action == self.gripper.ACTION_MANUAL:
            self.gripper.open()
            self.start_manual_mode()

    def perform_lift_action(self):
        self.is_lifting = True
        lift_speed = -0.04

        print(f"🚀 Lifting... Speed: {lift_speed} m/s")
        self.send_velocity_command(0.0, 0.0, lift_speed)
        QTimer.singleShot(5000, self.stop_lift_action)

    def stop_lift_action(self):
        if self._is_shutting_down:
            self.is_lifting = False
            return

        print("✅ Lift Complete.")
        self.is_lifting = False
        self.send_velocity_command(0.0, 0.0, 0.0)

    def reset_grab_flag(self):
        if self._is_shutting_down:
            return

        print("[System] Ready to grab again.")
        self.grab_triggered = False

    def start_manual_mode(self):
        if self._is_shutting_down:
            return

        self.manual_mode_active = True
        self.send_velocity_command(0.0, 0.0, 0.0)

        three_level = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
        if three_level is None or not hasattr(three_level, "last_gesture_time"):
            print("[UI] 3-Level gesture helper is not available.")
            return

        three_level.last_gesture_time = time.time()

        if not three_level.is_recognizing_gesture:
            three_level.toggle_gesture_recognition()

        self.manual_watchdog_timer.start()

    def check_manual_timeout(self):
        if self._is_shutting_down:
            return

        three_level = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
        if three_level is None or not hasattr(three_level, "last_gesture_time"):
            print("[UI] 3-Level gesture helper is not available.")
            return

        elapsed = time.time() - three_level.last_gesture_time
        if elapsed > 5.0:
            print(f"⏰ No gestures for {elapsed:.1f}s. Returning to Auto Mode.")
            self.stop_manual_mode()

    def stop_manual_mode(self):
        self.grab_triggered = True
        self.manual_mode_active = False
        self.grip_fail_count = 0
        self.manual_watchdog_timer.stop()

        print(">>> TIMEOUT: SWITCHING TO IMMEDIATE GRIP <<<")

        three_level = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
        if three_level is not None and getattr(three_level, "is_recognizing_gesture", False):
            three_level.toggle_gesture_recognition()

        self.send_velocity_command(0.0, 0.0, 0.0)
        self.gripper.close(soft=True)
        QTimer.singleShot(2000, self.check_grip_result)

    def update_cam_window(self, qt_image):
        if self._is_shutting_down:
            return

        if self.cam_window is not None and self.cam_window.isVisible() and self.cam_label is not None:
            self.cam_label.setPixmap(QPixmap.fromImage(qt_image))
            self.cam_label.resize(qt_image.width(), qt_image.height())

    def reset_yolo_button_state(self):
        self._close_camera_window()
        self._set_live_camera_ui_state(False)
        if self.yolo_worker is not None and not self.yolo_worker.isRunning():
            self.yolo_worker = None

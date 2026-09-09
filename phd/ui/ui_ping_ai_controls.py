from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QFileDialog,
    QLineEdit,
    QShortcut,
    QTextEdit,
)

from phd.dependence.paths import ai_resource_path
from phd.dependence.sensor_layout import column_major_matrix_view
from phd.ui import theme


class AiControlsMixin:
    AI_PROXIMITY_MODEL_DIR = Path(
        ai_resource_path("models", "tactile_proximity")
    )

    def _set_ai_frame(self, frame: str):
        self.ai_selected_frame = frame
        if hasattr(self, "ai_frame_input") and self.ai_frame_input is not None:
            self.ai_frame_input.setText(frame)
        self._update_ai_frame_buttons()

    def _update_ai_frame_buttons(self):
        selected = getattr(self, "ai_selected_frame", "tool")
        for frame_key, btn in getattr(self, "ai_frame_buttons", {}).items():
            is_selected = frame_key == selected
            btn.setChecked(is_selected)
            self._set_button_active(btn, is_selected)

    def _init_ai_toggle_states(self):
        self._lstm_active = False
        self._hier_active = False
        self._three_active = False
        self._direct_finger_active = False
        self._console_control_active = False
        self._console_control_sensor_active = False
        self._console_control_sensor_v2_active = False
        self._ai_direct_finger_active = False
        self._ai_direct_finger_robot_active = False
        self._ai_direct_finger_execution_active = False
        self._ai_proximity_detection_active = False
        self._ai_proximity_detector = None
        self._ai_proximity_model_signature = None
        self._ai_proximity_model_path = None
        self._ai_proximity_manual_model_path = None
        self._ai_proximity_executor = None
        self._ai_proximity_pending = None
        self._ai_proximity_last_frame_sequence = None
        self._ai_proximity_previous_averaged_frame = None
        self._ai_proximity_result = None
        self._ai_proximity_state = None
        self._ai_proximity_frames_submitted = 0
        self._ai_proximity_robot_motion_active = False
        self._ai_proximity_motion_confirmed = False
        self._admittance_control_active = False
        self._hier_mode_is_continues = True

        self._set_button_active(self.predict_threelevel_hierarchical_transformer_gesture_button, False)
        self._set_button_active(self.direct_finger_motion_button, False)
        self._set_button_active(self.console_control_button, False)
        if hasattr(self, "console_control_sensor_button"):
            self._set_button_active(self.console_control_sensor_button, False)
        if hasattr(self, "console_control_sensor_v2_button"):
            self._set_button_active(self.console_control_sensor_v2_button, False)
        self._set_button_active(self.ai_direct_finger_motion_button, False)
        if hasattr(self, "ai_direct_finger_motion_robot_button"):
            self._set_button_active(self.ai_direct_finger_motion_robot_button, False)
        self._set_button_active(self.ai_direct_finger_motion_execution_button, False)
        if hasattr(self, "ai_proximity_detection_button"):
            self.ai_proximity_detection_button.setChecked(False)
            self._set_button_active(
                self.ai_proximity_detection_button,
                False,
            )
        if hasattr(self, "ai_proximity_admittance_button"):
            self.ai_proximity_admittance_button.setChecked(False)
            self.ai_proximity_admittance_button.setEnabled(False)
            self._set_button_active(
                self.ai_proximity_admittance_button,
                False,
            )
        if hasattr(self, "admittance_control_button"):
            self.admittance_control_button.setChecked(False)
            self._set_button_active(self.admittance_control_button, False)

        if all(
            hasattr(self, name)
            for name in (
                "send_position_PTP_J_button",
                "send_position_PTP_T_button",
                "send_position_PTP_T_toolframe_button",
                "send_script_button",
            )
        ):
            self._set_button_active(self.send_position_PTP_J_button, False)
            self._set_button_active(self.send_position_PTP_T_button, False)
            self._set_button_active(self.send_position_PTP_T_toolframe_button, False)
            self._set_button_active(self.send_script_button, False)

        three = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
        latch_on = bool(getattr(three, "latch_mode", False)) if three else False

        self._set_button_active(self.btn_toggle_3lvl_latch, latch_on)
        self.btn_toggle_3lvl_latch.setText(f"3-Level: Latch {'ON' if latch_on else 'OFF'}")
        self._update_anchor_button_label()
        self._refresh_ai_proximity_model_preview()

    def _install_keyboard_shortcuts(self):
        self.shortcut_toggle_ai_direct_finger_motion = QShortcut(QKeySequence("Space"), self)
        self.shortcut_toggle_ai_direct_finger_motion.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_toggle_ai_direct_finger_motion.activated.connect(
            self._shortcut_toggle_ai_direct_finger_motion
        )

        self.shortcut_stop_ai_direct_finger_motion = QShortcut(QKeySequence("Esc"), self)
        self.shortcut_stop_ai_direct_finger_motion.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_stop_ai_direct_finger_motion.activated.connect(
            self._shortcut_stop_ai_direct_finger_motion
        )

        if hasattr(self, "ai_direct_finger_motion_button"):
            self.ai_direct_finger_motion_button.setToolTip(
                "Start/stop AI Direct Finger Motion recording without sending robot commands. Shortcut: Space. Stop: Esc."
            )
        if hasattr(self, "ai_direct_finger_motion_robot_button"):
            self.ai_direct_finger_motion_robot_button.setToolTip(
                "Start/stop AI Direct Finger Motion recording while sending the rule-based robot commands."
            )
        if hasattr(self, "ai_direct_finger_motion_execution_button"):
            self.ai_direct_finger_motion_execution_button.setToolTip(
                "Start/stop AI Direct Finger Motion execution using a trained checkpoint."
            )
        if hasattr(self, "ai_proximity_detection_button"):
            self.ai_proximity_detection_button.setToolTip(
                "Compare the live tactile stream with the learned normal "
                "environment and report a persistent local proximity change. "
                "Robot motion remains off until Proximity Admittance Control "
                "is enabled."
            )

    def _ai_proximity_sensitivity(self):
        combo = getattr(self, "ai_proximity_sensitivity_combo", None)
        if combo is None:
            return 0.85
        value = combo.currentData()
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.85

    def _ai_proximity_detection_mode(self):
        combo = getattr(self, "ai_proximity_detection_mode_combo", None)
        if combo is None:
            return "hybrid"
        return str(combo.currentData() or "hybrid")

    def _on_ai_proximity_detection_mode_changed(self, _index=None):
        mode = self._ai_proximity_detection_mode()
        sensitivity_combo = getattr(
            self,
            "ai_proximity_sensitivity_combo",
            None,
        )
        if sensitivity_combo is not None:
            sensitivity_combo.setEnabled(mode != "cnn_gru")
        detector = getattr(self, "_ai_proximity_detector", None)
        if detector is not None:
            detector.set_detection_mode(mode)
        self._ai_proximity_state = None
        self._zero_ai_proximity_robot_motion()

    def _on_ai_proximity_sensitivity_changed(self, _index=None):
        detector = getattr(self, "_ai_proximity_detector", None)
        if detector is not None:
            detector.set_sensitivity(self._ai_proximity_sensitivity())

    @staticmethod
    def _checkpoint_signature(path):
        try:
            resolved = Path(path).expanduser().resolve()
            stat = resolved.stat()
        except OSError:
            return None
        return (
            str(resolved),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )

    @staticmethod
    def _ai_proximity_model_filename(sensor_rows, sensor_cols):
        return (
            "latest_proximity_cnn_gru_"
            f"{int(sensor_rows)}x{int(sensor_cols)}.pt"
        )

    def _ai_proximity_sensor_shape(self):
        sensor = getattr(self, "sensor_functions", None)
        if sensor is None:
            return None
        try:
            rows = int(getattr(sensor, "n_row", 0))
            cols = int(getattr(sensor, "n_col", 0))
        except (TypeError, ValueError):
            rows, cols = 0, 0
        if rows > 0 and cols > 0:
            return rows, cols

        data = getattr(sensor, "_data", None)
        frame = getattr(data, "diffPerDataAve", None)
        if frame is None:
            return None
        shape = np.asarray(frame).shape
        if len(shape) == 2 and shape[0] > 0 and shape[1] > 0:
            return int(shape[0]), int(shape[1])
        return None

    def _ai_proximity_model_path_for_shape(self, sensor_shape):
        rows, cols = (int(sensor_shape[0]), int(sensor_shape[1]))
        filename = self._ai_proximity_model_filename(rows, cols)
        checkpoint_path = Path(self.AI_PROXIMITY_MODEL_DIR) / filename
        if checkpoint_path.is_file():
            return checkpoint_path

        available = sorted(
            path.name
            for path in Path(self.AI_PROXIMITY_MODEL_DIR).glob(
                "latest_proximity_cnn_gru_*x*.pt"
            )
        )
        available_text = ", ".join(available) if available else "none"
        raise FileNotFoundError(
            f"No proximity model is trained for the {rows}x{cols} sensor. "
            f"Expected {filename}. Available size-specific models: "
            f"{available_text}."
        )

    def _set_ai_proximity_model_status(
        self,
        checkpoint_path=None,
        sensor_shape=None,
        *,
        error=None,
    ):
        label = getattr(self, "ai_proximity_model_status", None)
        if label is None:
            return
        if error:
            label.setText(f"Model: {error}")
            label.setStyleSheet(f"color: {theme.DANGER_HOVER};")
            self._update_ai_proximity_model_buttons()
            return
        if checkpoint_path is None:
            label.setText("Model: automatic selection from sensor size")
            label.setStyleSheet(theme.MUTED_LABEL_STYLE)
            self._update_ai_proximity_model_buttons()
            return
        shape_text = ""
        if sensor_shape is not None:
            shape_text = (
                f" | sensor {int(sensor_shape[0])}x"
                f"{int(sensor_shape[1])}"
            )
        manual_path = getattr(
            self,
            "_ai_proximity_manual_model_path",
            None,
        )
        selection_text = "Selected" if manual_path else "Auto"
        label.setText(
            f"Model: {selection_text}: "
            f"{Path(checkpoint_path).name}{shape_text}"
        )
        label.setToolTip(str(Path(checkpoint_path).expanduser()))
        label.setStyleSheet(
            f"color: {theme.SUCCESS_HOVER}; font-weight: 600;"
        )
        self._update_ai_proximity_model_buttons()

    def _update_ai_proximity_model_buttons(self):
        active = bool(
            getattr(self, "_ai_proximity_detection_active", False)
        )
        select_button = getattr(
            self,
            "ai_proximity_select_model_button",
            None,
        )
        if select_button is not None:
            select_button.setEnabled(not active)
        auto_button = getattr(
            self,
            "ai_proximity_use_auto_model_button",
            None,
        )
        if auto_button is not None:
            auto_button.setEnabled(
                not active
                and bool(
                    getattr(
                        self,
                        "_ai_proximity_manual_model_path",
                        None,
                    )
                )
            )

    def _refresh_ai_proximity_model_preview(self):
        """Show the model that will be loaded for the current sensor."""
        if bool(getattr(self, "_ai_proximity_detection_active", False)):
            return
        sensor_shape = self._ai_proximity_sensor_shape()
        manual_path = getattr(
            self,
            "_ai_proximity_manual_model_path",
            None,
        )
        if manual_path:
            checkpoint_path = Path(manual_path).expanduser()
            if checkpoint_path.is_file():
                self._set_ai_proximity_model_status(
                    checkpoint_path,
                    sensor_shape,
                )
            else:
                self._set_ai_proximity_model_status(
                    error=f"selected checkpoint not found: {checkpoint_path}"
                )
            self._update_ai_proximity_model_buttons()
            return
        if sensor_shape is None:
            self._set_ai_proximity_model_status()
            return
        try:
            checkpoint_path = self._ai_proximity_model_path_for_shape(
                sensor_shape
            )
        except FileNotFoundError as exc:
            self._set_ai_proximity_model_status(error=str(exc))
            return
        self._set_ai_proximity_model_status(
            checkpoint_path,
            sensor_shape,
        )

    def _on_select_ai_proximity_model(self):
        if bool(getattr(self, "_ai_proximity_detection_active", False)):
            return
        current_path = getattr(
            self,
            "_ai_proximity_manual_model_path",
            None,
        )
        start_directory = Path(self.AI_PROXIMITY_MODEL_DIR)
        if current_path:
            candidate_parent = Path(current_path).expanduser().parent
            if candidate_parent.is_dir():
                start_directory = candidate_parent
        dialog = QFileDialog(
            self,
            "Select AI Proximity Model",
            str(start_directory),
            "PyTorch Models (*.pt *.pth);;All Files (*)",
        )
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_() != QFileDialog.Accepted:
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        self._ai_proximity_manual_model_path = str(
            Path(selected[0]).expanduser().resolve()
        )
        self._ai_proximity_detector = None
        self._ai_proximity_model_signature = None
        self._ai_proximity_model_path = None
        self._refresh_ai_proximity_model_preview()

    def _on_use_auto_ai_proximity_model(self):
        if bool(getattr(self, "_ai_proximity_detection_active", False)):
            return
        self._ai_proximity_manual_model_path = None
        self._ai_proximity_detector = None
        self._ai_proximity_model_signature = None
        self._ai_proximity_model_path = None
        self._refresh_ai_proximity_model_preview()

    def _set_ai_proximity_status(self, text, state="idle"):
        label = getattr(self, "ai_proximity_detection_status", None)
        if label is None:
            return
        label.setText(str(text))
        if state == "detected":
            label.setStyleSheet(
                f"color: {theme.DANGER_HOVER}; font-weight: 600;"
            )
        elif state == "normal":
            label.setStyleSheet(
                f"color: {theme.SUCCESS_HOVER}; font-weight: 600;"
            )
        elif state == "error":
            label.setStyleSheet(f"color: {theme.DANGER_HOVER};")
        else:
            label.setStyleSheet(theme.MUTED_LABEL_STYLE)

    def _ensure_ai_proximity_timer(self):
        timer = getattr(self, "_ai_proximity_timer", None)
        if timer is not None:
            return timer
        timer = QTimer(self)
        timer.setInterval(10)
        timer.timeout.connect(self._poll_ai_proximity_detection)
        self._ai_proximity_timer = timer
        return timer

    def _load_ai_proximity_detector(self):
        sensor_shape = self._ai_proximity_sensor_shape()
        if sensor_shape is None:
            raise ValueError(
                "Sensor dimensions are unavailable. Build and update the "
                "sensor before loading proximity AI."
            )
        manual_path = getattr(
            self,
            "_ai_proximity_manual_model_path",
            None,
        )
        if manual_path:
            checkpoint_path = Path(manual_path).expanduser()
            if not checkpoint_path.is_file():
                message = (
                    f"Selected proximity model was not found: "
                    f"{checkpoint_path}"
                )
                self._set_ai_proximity_model_status(error=message)
                raise FileNotFoundError(message)
        else:
            try:
                checkpoint_path = self._ai_proximity_model_path_for_shape(
                    sensor_shape
                )
            except Exception as exc:
                self._set_ai_proximity_model_status(error=str(exc))
                raise

        signature = self._checkpoint_signature(checkpoint_path)
        if (
            self._ai_proximity_detector is None
            or signature != self._ai_proximity_model_signature
        ):
            # Keep this import lazy so opening phd_ui does not load PyTorch.
            from phd.dependence.tactile_proximity import (
                TactileProximityDetector,
            )

            self._ai_proximity_detector = TactileProximityDetector(
                checkpoint_path,
                device="cpu",
                consecutive_required=2,
                sensitivity=self._ai_proximity_sensitivity(),
                detection_mode=self._ai_proximity_detection_mode(),
                localized_warmup_frames=60,
            )
            self._ai_proximity_model_signature = signature
            self._ai_proximity_model_path = checkpoint_path

        data = getattr(self.sensor_functions, "_data", None)
        live_model_shape = column_major_matrix_view(
            data.diffPerDataAve
        ).shape
        expected_shape = (
            self._ai_proximity_detector.rows,
            self._ai_proximity_detector.cols,
        )
        if live_model_shape != expected_shape:
            self._set_ai_proximity_model_status(
                error=(
                    f"{checkpoint_path.name} expects {expected_shape[0]}x"
                    f"{expected_shape[1]} model input, but the live sensor "
                    f"provides {live_model_shape[0]}x{live_model_shape[1]}"
                )
            )
            raise ValueError(
                f"Proximity model {checkpoint_path.name} is incompatible "
                f"with the {sensor_shape[0]}x{sensor_shape[1]} sensor."
            )
        self._set_ai_proximity_model_status(
            checkpoint_path,
            sensor_shape,
        )
        return checkpoint_path

    def _ai_proximity_admittance_active(self):
        return bool(
            getattr(self, "_ai_proximity_robot_motion_active", False)
        )

    def _set_ai_proximity_admittance_button_state(
        self,
        *,
        checked,
        enabled,
    ):
        button = getattr(self, "ai_proximity_admittance_button", None)
        if button is None:
            return
        button.blockSignals(True)
        button.setChecked(bool(checked))
        button.setEnabled(bool(enabled))
        button.blockSignals(False)
        self._set_button_active(button, bool(checked))

    def _zero_ai_proximity_robot_motion(self):
        if not bool(
            getattr(self, "_ai_proximity_robot_motion_active", False)
        ):
            return
        helper = getattr(self, "mesh_functions", None)
        if helper is None or not hasattr(
            helper,
            "update_ai_proximity_motion",
        ):
            return
        helper.update_ai_proximity_motion(
            None,
            None,
            0.0,
            detected=False,
            dry_run=False,
        )

    def _stop_ai_proximity_robot_motion(self):
        helper = getattr(self, "mesh_functions", None)
        if helper is not None and hasattr(
            helper,
            "stop_ai_proximity_motion",
        ):
            try:
                helper.stop_ai_proximity_motion()
            except Exception as exc:
                print(
                    "[AI Proximity] Robot motion teardown failed: "
                    f"{exc}"
                )
        self._ai_proximity_robot_motion_active = False

    def _prepare_ai_proximity_robot_motion(self):
        self._ai_proximity_motion_confirmed = True

        helper = getattr(self, "mesh_functions", None)
        if helper is None or not hasattr(
            helper,
            "start_ai_proximity_motion",
        ):
            raise RuntimeError("Robot sensor mapping is unavailable")
        started, message = helper.start_ai_proximity_motion()
        if not started:
            raise RuntimeError(str(message))
        self._ai_proximity_robot_motion_active = True
        return True

    def _on_toggle_ai_proximity_admittance_control(self, enabled):
        enabled = bool(enabled)
        if not enabled:
            self._stop_ai_proximity_robot_motion()
            self._set_ai_proximity_admittance_button_state(
                checked=False,
                enabled=bool(
                    getattr(
                        self,
                        "_ai_proximity_detection_active",
                        False,
                    )
                ),
            )
            if bool(
                getattr(self, "_ai_proximity_detection_active", False)
            ):
                self._set_ai_proximity_status(
                    "Proximity AI: monitoring only",
                    "waiting",
                )
            return

        if not bool(
            getattr(self, "_ai_proximity_detection_active", False)
        ):
            self._set_ai_proximity_admittance_button_state(
                checked=False,
                enabled=False,
            )
            self._set_ai_proximity_status(
                "Proximity AI: start detection before admittance control",
                "error",
            )
            return

        try:
            if not self._prepare_ai_proximity_robot_motion():
                self._set_ai_proximity_admittance_button_state(
                    checked=False,
                    enabled=True,
                )
                return
        except Exception as exc:
            self._stop_ai_proximity_robot_motion()
            self._set_ai_proximity_admittance_button_state(
                checked=False,
                enabled=True,
            )
            self._set_ai_proximity_status(
                f"Proximity admittance error: {exc}",
                "error",
            )
            print(f"[AI Proximity] Admittance start failed: {exc}")
            return

        self._set_ai_proximity_admittance_button_state(
            checked=True,
            enabled=True,
        )
        self._set_ai_proximity_status(
            "Proximity AI: admittance armed, waiting for detection",
            "waiting",
        )
        print("[AI Proximity] ADMITTANCE CONTROL ENABLED")

    def _apply_ai_proximity_motion_result(self, result):
        admittance_active = self._ai_proximity_admittance_active()
        sensor_row = result.center_col
        sensor_col = result.center_row
        if not admittance_active:
            if (
                result.detected
                and sensor_row is not None
                and sensor_col is not None
            ):
                return (
                    "Monitoring only | "
                    f"grid r{float(sensor_row):.1f} "
                    f"c{float(sensor_col):.1f}"
                )
            return "Monitoring only"

        helper = getattr(self, "mesh_functions", None)
        if helper is None or not hasattr(
            helper,
            "update_ai_proximity_motion",
        ):
            raise RuntimeError("Robot sensor mapping is unavailable")

        # Model recordings use the historic transposed matrix view. Convert
        # its row/column coordinates back to the sensor geometry convention.
        motion = helper.update_ai_proximity_motion(
            sensor_row,
            sensor_col,
            result.anomaly_score,
            detected=result.detected,
            dry_run=False,
        )
        if not bool(motion.get("ok", False)):
            error = str(motion.get("error") or "motion update failed")
            raise RuntimeError(error)

        if not result.detected:
            position_error = float(motion.get("position_error_m", 0.0))
            if bool(motion.get("returning", False)):
                return (
                    "Returning to start | "
                    f"error={position_error * 1000.0:.1f} mm | "
                    f"v={float(motion.get('speed_mps', 0.0)):.3f} m/s"
                )
            return (
                "At start position | "
                f"error={position_error * 1000.0:.1f} mm"
            )
        if result.anomaly_score <= 1.0:
            return "Robot holding"
        if sensor_row is None or sensor_col is None:
            return "Robot stopped | location unavailable"

        direction = motion.get("direction")
        direction_text = ""
        if isinstance(direction, (list, tuple)) and len(direction) >= 3:
            direction_text = (
                " | d=("
                f"{float(direction[0]):+.2f},"
                f"{float(direction[1]):+.2f},"
                f"{float(direction[2]):+.2f})"
            )
        return (
            f"Robot moving | grid r{float(sensor_row):.1f} "
            f"c{float(sensor_col):.1f} | "
            f"v={float(motion.get('speed_mps', 0.0)):.3f} m/s"
            f"{direction_text}"
        )

    def _on_toggle_ai_proximity_detection(self):
        if bool(getattr(self, "_ai_proximity_detection_active", False)):
            self._stop_ai_proximity_detection()
            return

        sensor = getattr(self, "sensor_functions", None)
        data = getattr(sensor, "_data", None) if sensor is not None else None
        if sensor is None or data is None or not bool(
            getattr(sensor, "is_connected", False)
        ):
            message = "Build and update the sensor before starting proximity AI."
            print(f"[AI Proximity] {message}")
            self._set_ai_proximity_status(
                f"Proximity AI: {message}",
                "error",
            )
            button = getattr(self, "ai_proximity_detection_button", None)
            if button is not None:
                button.setChecked(False)
            return

        try:
            checkpoint_path = self._load_ai_proximity_detector()
            detector = self._ai_proximity_detector
            current_shape = column_major_matrix_view(
                data.diffPerDataAve
            ).shape
            expected_shape = (detector.rows, detector.cols)
            if current_shape != expected_shape:
                raise ValueError(
                    f"Live sensor shape is {current_shape}, but the trained "
                    f"model expects {expected_shape}."
                )

            detector.reset()
            self._ai_proximity_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="tactile-proximity",
            )
            self._ai_proximity_pending = None
            self._ai_proximity_last_frame_sequence = None
            self._ai_proximity_previous_averaged_frame = None
            self._ai_proximity_result = None
            self._ai_proximity_state = None
            self._ai_proximity_frames_submitted = 0
            self._ai_proximity_detection_active = True
            self._update_ai_proximity_model_buttons()
            self._set_ai_proximity_admittance_button_state(
                checked=False,
                enabled=True,
            )
            button = self.ai_proximity_detection_button
            button.setChecked(True)
            self._set_button_active(button, True)
            self._set_ai_proximity_status(
                "Proximity AI: warming up",
                "waiting",
            )
            self._ensure_ai_proximity_timer().start()
            print(
                "[AI Proximity] STARTED | "
                f"model={checkpoint_path} | "
                f"mode={detector.detection_mode} | "
                f"sensitivity={detector.sensitivity:.2f} | "
                "admittance=False"
            )
        except Exception as exc:
            self._stop_ai_proximity_robot_motion()
            self._ai_proximity_detection_active = False
            self._update_ai_proximity_model_buttons()
            self.ai_proximity_detection_button.setChecked(False)
            self._set_ai_proximity_admittance_button_state(
                checked=False,
                enabled=False,
            )
            self._set_button_active(
                self.ai_proximity_detection_button,
                False,
            )
            self._set_ai_proximity_status(
                f"Proximity AI error: {exc}",
                "error",
            )
            print(f"[AI Proximity] Failed to start: {exc}")

    def _stop_ai_proximity_detection(self):
        # Stop the robot before waiting for background inference shutdown.
        self._stop_ai_proximity_robot_motion()
        timer = getattr(self, "_ai_proximity_timer", None)
        if timer is not None:
            timer.stop()

        pending = getattr(self, "_ai_proximity_pending", None)
        if pending is not None:
            try:
                pending.cancel()
            except Exception:
                pass
        executor = getattr(self, "_ai_proximity_executor", None)
        self._ai_proximity_executor = None
        self._ai_proximity_pending = None
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=True)

        was_active = bool(
            getattr(self, "_ai_proximity_detection_active", False)
        )
        self._ai_proximity_detection_active = False
        self._update_ai_proximity_model_buttons()
        self._ai_proximity_last_frame_sequence = None
        self._ai_proximity_previous_averaged_frame = None
        self._ai_proximity_state = None
        button = getattr(self, "ai_proximity_detection_button", None)
        if button is not None:
            button.setChecked(False)
            self._set_button_active(button, False)
        self._set_ai_proximity_admittance_button_state(
            checked=False,
            enabled=False,
        )
        self._set_ai_proximity_status("Proximity AI: idle", "idle")
        if was_active:
            print("[AI Proximity] STOPPED")

    def _collect_ai_proximity_result(self):
        pending = getattr(self, "_ai_proximity_pending", None)
        if pending is None or not pending.done():
            return
        self._ai_proximity_pending = None
        try:
            result = pending.result()
        except Exception as exc:
            print(f"[AI Proximity] Inference failed: {exc}")
            self._set_ai_proximity_status(
                f"Proximity AI error: {exc}",
                "error",
            )
            self._stop_ai_proximity_detection()
            return

        self._ai_proximity_result = result
        if not result.ready:
            self._zero_ai_proximity_robot_motion()
            detector = self._ai_proximity_detector
            warmup_frames = getattr(
                detector,
                "active_warmup_frames",
                detector.sequence_length,
            )
            collected = min(
                self._ai_proximity_frames_submitted,
                warmup_frames,
            )
            self._set_ai_proximity_status(
                "Proximity AI: warming up "
                f"({collected}/{warmup_frames})",
                "waiting",
            )
            return

        state = "detected" if result.detected else "normal"
        if result.detection_mode == "cnn_gru":
            metrics_text = (
                f"Score: {result.anomaly_score:.2f} | "
                f"CNN-GRU threshold: {result.model_threshold:.2f}"
            )
        elif result.detection_mode == "localized":
            metrics_text = (
                f"Score: {result.anomaly_score:.2f} | "
                "Localized threshold: "
                f"{result.localized_change_threshold:.2f}"
            )
        else:
            metrics_text = (
                f"Score: {result.anomaly_score:.2f} | "
                f"CNN-GRU threshold: {result.model_threshold:.2f} | "
                "Localized threshold: "
                f"{result.localized_change_threshold:.2f}"
            )
        try:
            motion_text = self._apply_ai_proximity_motion_result(result)
        except Exception as exc:
            message = f"Robot motion stopped: {exc}"
            print(f"[AI Proximity] {message}")
            self._stop_ai_proximity_detection()
            self._set_ai_proximity_status(
                f"Proximity AI error: {message}",
                "error",
            )
            return

        if result.detected:
            self._set_ai_proximity_status(
                "Proximity AI: PROXIMITY CHANGE DETECTED | "
                f"{metrics_text} | {motion_text}",
                "detected",
            )
        else:
            self._set_ai_proximity_status(
                "Proximity AI: environment normal | "
                f"{metrics_text} | {motion_text}",
                "normal",
            )

        previous_state = self._ai_proximity_state
        self._ai_proximity_state = state
        if state == "detected" and previous_state != state:
            print(
                "[AI Proximity] PROXIMITY CHANGE DETECTED | "
                f"score={result.anomaly_score:.4f} | "
                f"threshold={result.threshold:.4f} | "
                f"mode={result.detection_mode} | "
                f"model={result.model_anomaly_score:.4f} | "
                f"local={result.localized_change_score:.4f}"
            )
        elif state == "normal" and previous_state == "detected":
            print(
                "[AI Proximity] ENVIRONMENT NORMAL | "
                f"score={result.anomaly_score:.4f}"
            )

    def _poll_ai_proximity_detection(self):
        if not bool(getattr(self, "_ai_proximity_detection_active", False)):
            return
        self._collect_ai_proximity_result()
        if self._ai_proximity_pending is not None:
            return

        sensor = getattr(self, "sensor_functions", None)
        data = getattr(sensor, "_data", None) if sensor is not None else None
        if data is None or not bool(getattr(sensor, "is_connected", False)):
            if bool(
                getattr(self, "_ai_proximity_robot_motion_active", False)
            ):
                self._stop_ai_proximity_detection()
                self._set_ai_proximity_status(
                    "Proximity AI error: sensor stream lost; robot stopped",
                    "error",
                )
                return
            self._set_ai_proximity_status(
                "Proximity AI: waiting for sensor",
                "waiting",
            )
            return

        frame_sequence = getattr(data, "frame_sequence", None)
        if frame_sequence == self._ai_proximity_last_frame_sequence:
            return
        self._ai_proximity_last_frame_sequence = frame_sequence

        try:
            diff_frame = column_major_matrix_view(
                data.diffPerData,
                dtype=np.float32,
                copy=True,
            )
            averaged_diff_frame = column_major_matrix_view(
                data.diffPerDataAve,
                dtype=np.float32,
                copy=True,
            )
            from phd.dependence.tactile_proximity import (
                build_tactile_proximity_channel_frame,
            )

            channel_frame = build_tactile_proximity_channel_frame(
                self._ai_proximity_detector.channels,
                diff_frame,
                averaged_diff_frame,
                self._ai_proximity_previous_averaged_frame,
            )
            self._ai_proximity_previous_averaged_frame = (
                averaged_diff_frame.copy()
            )
            self._ai_proximity_frames_submitted += 1
            self._ai_proximity_pending = self._ai_proximity_executor.submit(
                self._ai_proximity_detector.update,
                channel_frame,
            )
        except Exception as exc:
            print(f"[AI Proximity] Live frame failed: {exc}")
            self._set_ai_proximity_status(
                f"Proximity AI error: {exc}",
                "error",
            )
            self._stop_ai_proximity_detection()

    def _focus_on_text_input(self):
        focused = self.focusWidget()
        return isinstance(focused, (QLineEdit, QTextEdit))

    def _is_data_training_subtab_active(self):
        return (
            hasattr(self, "ai_sub_tabs")
            and self.ai_sub_tabs.currentIndex()
            == int(getattr(self, "ai_data_training_tab_index", 2))
        )

    def _shortcut_toggle_ai_direct_finger_motion(self):
        if self._focus_on_text_input():
            return
        if not self._is_data_training_subtab_active():
            return
        if not self.ai_direct_finger_motion_button.isEnabled():
            return
        self._on_toggle_ai_direct_finger_motion()

    def _shortcut_stop_ai_direct_finger_motion(self):
        if self._focus_on_text_input():
            return
        if not (
            getattr(self, "_ai_direct_finger_active", False)
            or getattr(self, "_ai_direct_finger_robot_active", False)
        ):
            return
        if not self.ai_direct_finger_motion_button.isEnabled():
            return
        self._on_toggle_ai_direct_finger_motion()

    def on_toggle_threelevel_latch(self):
        try:
            three = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
            if not three:
                print("[UI] 3-Level instance not available")
                return
            three.toggle_latch_mode()
            latch = bool(getattr(three, "latch_mode", False))
            self.btn_toggle_3lvl_latch.setText(f"3-Level: Latch {'ON' if latch else 'OFF'}")
            self._set_button_active(self.btn_toggle_3lvl_latch, latch)
        except Exception as exc:
            print(f"[UI] Could not toggle 3-Level latch mode: {exc}")

    def _on_sensitivity_changed(self, value: int):
        sensitivity_float = value / 1000.0
        self.sensitivity_value_label.setText(f"{sensitivity_float:.3f}")
        self.sensor_functions.set_touch_sensitivity(sensitivity_float)

    def _on_toggle_threelevel_predict(self):
        try:
            three = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
            if three is None:
                raise AttributeError("threelevel_hierarchical_transformer_class is not available")
            previous_state = bool(getattr(three, "is_recognizing_gesture", False))
            three.toggle_gesture_recognition()
            current_state = bool(getattr(three, "is_recognizing_gesture", False))
            if current_state == previous_state:
                print("[UI] ThreeLevel state unchanged after toggle request.")
            self._three_active = current_state
            self._set_button_active(
                self.predict_threelevel_hierarchical_transformer_gesture_button,
                self._three_active,
            )
        except Exception as exc:
            print(f"[UI] ThreeLevel toggle failed: {exc}")
            self._three_active = bool(getattr(self, "_three_active", False))
            self._set_button_active(
                self.predict_threelevel_hierarchical_transformer_gesture_button,
                self._three_active,
            )
        self._update_anchor_button_label()

    def _update_anchor_button_label(self):
        if not hasattr(self, "btn_toggle_anchor_axes"):
            return

        anchor_available = (
            bool(getattr(self, "_three_active", False))
            and not bool(getattr(self, "_direct_finger_active", False))
        )

        self.btn_toggle_anchor_axes.setEnabled(anchor_available)

        if not anchor_available:
            self.btn_toggle_anchor_axes.setText("Axes: Anchored OFF")
            # setEnabled(False) above + the global QSS ":disabled" rule already
            # render the proper muted style; just clear any active override.
            self.btn_toggle_anchor_axes.setStyleSheet("")
            return

        three = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
        anchored = bool(getattr(three, "anchor_enabled", True)) if three else True
        self.btn_toggle_anchor_axes.setText(f"Axes: Anchored {'ON' if anchored else 'OFF'}")
        self._set_button_active(self.btn_toggle_anchor_axes, anchored)

    def _on_toggle_anchor_axes(self):
        if hasattr(self, "btn_toggle_anchor_axes") and not self.btn_toggle_anchor_axes.isEnabled():
            return

        three = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
        if not three:
            print("[UI] 3-Level instance not available yet.")
            return

        new_val = not bool(getattr(three, "anchor_enabled", True))
        setattr(three, "anchor_enabled", new_val)

        if new_val and hasattr(three, "_set_anchor_from_current_frame"):
            try:
                three._set_anchor_from_current_frame()
            except Exception as exc:
                print(f"[UI] Could not set anchor: {exc}")

        self._update_anchor_button_label()

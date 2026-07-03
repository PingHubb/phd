from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import QObject, pyqtSignal


class _HandAsyncSignals(QObject):
    result = pyqtSignal(str, str, object)
    error = pyqtSignal(str, str, str)


class UiInteractionsMixin:
    def _get_sensor_helper(self, attr_name: str, default=None):
        sensor_functions = getattr(self, "sensor_functions", None)
        if sensor_functions is None:
            return default
        return getattr(sensor_functions, attr_name, default)

    def connect_function(self):
        self.read_sensor_api_button.pressed.connect(self._on_sensor_api_read_raw)
        self.read_sensor_api_hz_button.pressed.connect(self._on_sensor_api_read_raw_hz)
        self.read_sensor_channel_button.pressed.connect(self._on_sensor_api_channel_check)
        self.read_sensor_raw_button.pressed.connect(self._on_read_sensor_raw)
        self.read_sensor_raw_ave_button.pressed.connect(self._on_read_sensor_raw_average)
        self.read_sensor_diff_button.pressed.connect(self._on_read_sensor_diff)
        self.read_sensor_diff_debug_button.pressed.connect(self._on_read_sensor_diff_debug_views)
        self.read_runtime_hz_button.pressed.connect(self._on_read_runtime_hz_report)

        self.read_joint_angle_button.pressed.connect(self._on_read_joint_angles)
        self.read_tool_position_button.pressed.connect(self._on_read_tool_position)
        self.send_position_PTP_J_button.pressed.connect(self.toggle_joint_angle_input)
        self.send_position_PTP_T_button.pressed.connect(self.toggle_tool_position_input)
        self.send_position_PTP_T_toolframe_button.pressed.connect(self.toggle_tool_frame_position_input)
        self.send_script_button.pressed.connect(self.toggle_robot_script_input)
        self.position_script_widget.transmit_script.connect(self._on_transmit_robot_script)
        self.show_robot_button.pressed.connect(self._on_show_robot)
        self.buildScene.pressed.connect(self._on_build_scene)
        self.sensor_update.pressed.connect(self._on_sensor_update)
        self.record_gesture_button.pressed.connect(self.start_record_gesture)
        self.set_no_trigger_button.pressed.connect(self._on_set_no_trigger_mode)
        self.set_no_trigger_auto_button.pressed.connect(self._on_set_no_trigger_auto_mode)
        self.set_no_trigger_no_updatecal_auto_button.pressed.connect(
            self._on_set_no_trigger_no_updatecal_auto_mode
        )
        self.set_trigger_button.pressed.connect(self._on_set_trigger_mode)
        self.predict_threelevel_hierarchical_transformer_gesture_button.pressed.connect(
            self._on_toggle_threelevel_predict
        )
        self.btn_toggle_3lvl_latch.pressed.connect(self.on_toggle_threelevel_latch)
        self.proximity_control_button.pressed.connect(self._on_toggle_proximity_control)
        self.proximity_record_button.pressed.connect(self._on_toggle_proximity_recording)
        self.apply_proximity_settings_button.clicked.connect(
            self._apply_proximity_settings_from_ui
        )
        self.reload_proximity_settings_button.clicked.connect(
            self._load_proximity_settings_into_ui
        )
        self.close_proximity_settings_button.clicked.connect(self.proximity_settings_dialog.close)
        self._load_proximity_settings_into_ui()
        self.direct_finger_motion_button.pressed.connect(self._on_toggle_direct_finger_motion)
        self.console_control_button.pressed.connect(self._on_toggle_console_control)
        self.console_control_sensor_button.pressed.connect(
            self._on_toggle_console_control_sensor_placeholder
        )
        self.console_control_sensor_v2_button.pressed.connect(
            self._on_toggle_console_control_sensor_v2
        )
        self.direct_finger_motion_tool_pose_record_menu_button.pressed.connect(
            self._toggle_direct_finger_motion_tool_pose_recording
        )
        self.load_tool_pose_path_button.clicked.connect(self._load_direct_finger_motion_tool_pose_path_from_dialog)
        self.clear_tool_pose_path_button.clicked.connect(self._clear_direct_finger_motion_tool_pose_path_plot)
        self.apply_direct_finger_motion_settings_button.clicked.connect(
            self._apply_direct_finger_motion_settings_from_ui
        )
        self.reload_direct_finger_motion_settings_button.clicked.connect(
            self._load_direct_finger_motion_settings_into_ui
        )
        self.apply_console_control_settings_button.clicked.connect(
            self._apply_console_control_settings_from_ui
        )
        self.reload_console_control_settings_button.clicked.connect(
            self._load_console_control_settings_into_ui
        )
        if hasattr(self, "ai_teaching_label_buttons"):
            for teaching_label, button in self.ai_teaching_label_buttons.items():
                if teaching_label == "auto":
                    button.clicked.connect(
                        lambda _checked=False, label=teaching_label: self._set_ai_teaching_label(label)
                    )
                else:
                    button.pressed.connect(
                        lambda label=teaching_label: self._set_ai_teaching_label(label)
                    )
                    button.released.connect(
                        lambda label="auto": self._set_ai_teaching_label(label)
                    )
            self._update_ai_teaching_label_ui()
        self.ai_direct_finger_motion_button.pressed.connect(
            lambda: self._on_toggle_ai_direct_finger_motion(send_robot_commands=False)
        )
        if hasattr(self, "ai_direct_finger_motion_robot_button"):
            self.ai_direct_finger_motion_robot_button.pressed.connect(
                lambda: self._on_toggle_ai_direct_finger_motion(send_robot_commands=True)
            )
        self.ai_direct_finger_motion_execution_button.pressed.connect(
            self._on_toggle_ai_direct_finger_motion_execution
        )
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        self.sensor_average_window_spin.valueChanged.connect(self._on_sensor_average_window_changed)
        self.visualization_target_hz_spin.valueChanged.connect(self._on_visualization_target_hz_changed)
        self.sensor_visualization_mode_combo.currentIndexChanged.connect(
            self._on_sensor_visualization_mode_changed
        )
        self.contact_normal_checkbox.toggled.connect(self._on_contact_normal_visibility_changed)
        self.contact_normal_estimator_combo.currentIndexChanged.connect(
            self._on_contact_normal_estimator_changed
        )
        self.btn_toggle_anchor_axes.pressed.connect(self._on_toggle_anchor_axes)
        self.hand_open_all_button.clicked.connect(self._on_hand_open_all)
        self.hand_close_all_button.clicked.connect(self._on_hand_close_all)
        self.hand_apply_speed_button.clicked.connect(self._on_hand_apply_speed)
        self.hand_apply_force_button.clicked.connect(self._on_hand_apply_force)
        self.hand_read_angles_button.clicked.connect(self._on_hand_read_angles)
        self.hand_thumb_left_button.clicked.connect(self._on_hand_thumb_left)
        self.hand_thumb_center_button.clicked.connect(self._on_hand_thumb_center)
        self.hand_thumb_right_button.clicked.connect(self._on_hand_thumb_right)
        self.hand_send_custom_angles_button.clicked.connect(self._on_hand_send_custom_angles)
        self.hand_model_show_button.clicked.connect(self._on_show_dexterous_hand_model)
        self.hand_sliders_load_open_button.clicked.connect(
            self._on_hand_sliders_load_open
        )
        self.hand_sliders_load_close_button.clicked.connect(
            self._on_hand_sliders_load_close
        )
        self.hand_sliders_sync_button.clicked.connect(self._on_hand_sliders_sync)
        self.hand_tactile_live_button.clicked.connect(self._on_hand_tactile_live_toggled)
        self.hand_tactile_refresh_button.clicked.connect(self._refresh_hand_tactile_display)
        # Per-slider hooks: value label updates + per-finger Send + optional
        # live streaming (throttled to ~20 Hz to avoid spamming the service).
        from PyQt5.QtCore import QTimer as _QTimer  # local import to avoid header churn
        if not hasattr(self, "_hand_live_pending"):
            self._hand_live_pending = {}
        if not hasattr(self, "_hand_live_timer"):
            self._hand_live_timer = _QTimer(self)
            self._hand_live_timer.setSingleShot(True)
            self._hand_live_timer.setInterval(50)
            self._hand_live_timer.timeout.connect(self._hand_flush_live_pending)
        for slider_idx, slider in enumerate(
            getattr(self, "hand_angle_sliders", []) or []
        ):
            slider.valueChanged.connect(
                lambda v, i=slider_idx: self._on_hand_slider_value_changed(i, int(v))
            )
        for btn_idx, btn in enumerate(
            getattr(self, "_hand_angle_send_buttons", []) or []
        ):
            btn.clicked.connect(
                lambda _checked=False, i=btn_idx: self._on_hand_send_single_slider(i)
            )

    def _on_read_sensor_raw(self):
        self.log_display.append(f"Raw data: {self.sensor_functions.read_sensor_raw_data()}")

    def _on_read_sensor_raw_average(self):
        self.log_display.append(f"Raw ave data: {self.sensor_functions.read_sensor_raw_ave_data()}")

    def _on_read_sensor_diff(self):
        self.log_display.append(f"Diff data: {self.sensor_functions.read_sensor_diff_data()}")

    def _on_read_sensor_diff_debug_views(self):
        self.log_display.append(self.sensor_functions.read_sensor_diff_debug_views())

    def _on_read_runtime_hz_report(self):
        self.log_display.append(self.sensor_functions.read_runtime_hz_report())

    def _on_read_joint_angles(self):
        self.log_display.append(f"Joint angles: {self.robot_api.get_current_positions()}")

    def _on_read_tool_position(self):
        self.log_display.append(f"Tool position: {self.robot_api.get_current_tool_position()}")

    def _on_show_robot(self):
        # Open the imported robot inside its own dedicated pop-up window so it
        # is never mixed with / hidden behind the sensor plotter.
        helper = self.mesh_functions
        if helper is None:
            return
        if hasattr(helper, "addRobotInDialog"):
            helper.addRobotInDialog()
        else:
            helper.addRobot()

    def _on_show_dexterous_hand_model(self):
        helper = getattr(self, "mesh_functions", None)
        status = getattr(self, "hand_model_status_label", None)
        if helper is None or not hasattr(helper, "addDexterousHandInDialog"):
            if status is not None:
                status.setText("Hand model viewer unavailable")
            return
        ok = bool(helper.addDexterousHandInDialog())
        if status is not None:
            status.setText("Hand model opened" if ok else "Hand model not loaded")

    def _on_build_scene(self):
        self.sensor_functions.buildScene()

    def _on_sensor_average_window_changed(self, value: int):
        self.sensor_functions.set_sensor_average_window_size(value)

    def _on_visualization_target_hz_changed(self, value: float):
        self.sensor_functions.set_visualization_target_hz(value)

    def _on_sensor_visualization_mode_changed(self, *_args):
        helper = getattr(self, "sensor_functions", None)
        combo = getattr(self, "sensor_visualization_mode_combo", None)
        if helper is None or combo is None:
            return
        mode = combo.currentData()
        if hasattr(helper, "set_sensor_visualization_mode"):
            helper.set_sensor_visualization_mode(mode)

    def _on_contact_normal_visibility_changed(self, checked: bool):
        helper = getattr(self, "sensor_functions", None)
        if helper is not None and hasattr(helper, "set_contact_normal_visualization_enabled"):
            helper.set_contact_normal_visualization_enabled(bool(checked))

    def _on_contact_normal_estimator_changed(self, *_args):
        helper = getattr(self, "sensor_functions", None)
        combo = getattr(self, "contact_normal_estimator_combo", None)
        if helper is None or combo is None:
            return
        mode = combo.currentData()
        if hasattr(helper, "set_contact_normal_estimator_mode"):
            helper.set_contact_normal_estimator_mode(mode)

    def _set_record_trigger_mode(self, mode: str):
        helper = self._get_sensor_helper("record_gesture_class")
        if helper is not None:
            helper.set_trigger_mode(mode)

    def _on_set_no_trigger_mode(self):
        self._set_record_trigger_mode("no_trigger")

    def _on_set_no_trigger_auto_mode(self):
        self._set_record_trigger_mode("no_trigger_auto")

    def _on_set_no_trigger_no_updatecal_auto_mode(self):
        self._set_record_trigger_mode("no_trigger_no_updatecal_auto")

    def _on_set_trigger_mode(self):
        self._set_record_trigger_mode("trigger")

    def _on_toggle_proximity_control(self):
        try:
            helper = self._get_sensor_helper("proximity_control_class")
            if helper is None:
                raise AttributeError("proximity_control_class is not available")
            helper.toggle_proximity_control()
            self._proximity_control_active = bool(getattr(helper, "is_running", False))
            if hasattr(self, "_set_button_active"):
                self._set_button_active(
                    self.proximity_control_button,
                    self._proximity_control_active,
                )
            self._sync_proximity_record_button()
        except Exception as exc:
            print(f"[UI] Proximity control toggle failed: {exc}")
            self._proximity_control_active = bool(
                getattr(self, "_proximity_control_active", False)
            )
            if hasattr(self, "_set_button_active"):
                self._set_button_active(
                    self.proximity_control_button,
                    self._proximity_control_active,
                )
            self._sync_proximity_record_button()

    def _on_toggle_proximity_recording(self):
        try:
            helper = self._get_sensor_helper("proximity_control_class")
            if helper is None:
                raise AttributeError("proximity_control_class is not available")
            helper.toggle_recording()
        except Exception as exc:
            print(f"[UI] Proximity recording toggle failed: {exc}")
        self._sync_proximity_record_button()

    def _sync_proximity_record_button(self):
        helper = self._get_sensor_helper("proximity_control_class")
        recording = bool(getattr(helper, "is_recording", False)) if helper is not None else False
        if hasattr(self, "_set_button_active"):
            self._set_button_active(self.proximity_record_button, recording)

    def _on_toggle_console_control(self):
        self._console_control_active = not getattr(self, "_console_control_active", False)
        if hasattr(self, "_set_button_active"):
            self._set_button_active(self.console_control_button, self._console_control_active)
        try:
            helper = self._get_sensor_helper("console_control_class")
            if helper is None or not hasattr(helper, "toggle_console_control"):
                raise AttributeError("console_control_class is not available")
            helper.toggle_console_control()
            is_running = getattr(helper, "is_running", self._console_control_active)
            source = str(getattr(helper, "console_input_source", "ps5"))
            self._console_control_active = bool(is_running and source == "ps5")
            self._console_control_sensor_active = bool(is_running and source == "sensor")
            self._console_control_sensor_v2_active = bool(is_running and source == "sensor_v2")
            if hasattr(self, "_set_button_active"):
                self._set_button_active(self.console_control_button, self._console_control_active)
                self._set_button_active(
                    self.console_control_sensor_button,
                    self._console_control_sensor_active,
                )
                self._set_button_active(
                    self.console_control_sensor_v2_button,
                    self._console_control_sensor_v2_active,
                )
        except Exception as exc:
            message = f"[UI] Console control toggle failed: {exc}"
            print(message)
            if hasattr(self, "log_display"):
                self.log_display.append(message)
            self._console_control_active = not self._console_control_active
            self._console_control_sensor_active = False
            self._console_control_sensor_v2_active = False
            if hasattr(self, "_set_button_active"):
                self._set_button_active(self.console_control_button, self._console_control_active)
                self._set_button_active(self.console_control_sensor_button, False)
                self._set_button_active(self.console_control_sensor_v2_button, False)

    def _on_toggle_console_control_sensor_placeholder(self):
        self._console_control_sensor_active = not getattr(self, "_console_control_sensor_active", False)
        if hasattr(self, "_set_button_active"):
            self._set_button_active(
                self.console_control_sensor_button,
                self._console_control_sensor_active,
            )
        try:
            helper = self._get_sensor_helper("console_control_class")
            if helper is None or not hasattr(helper, "toggle_console_control_sensor"):
                raise AttributeError("console_control_class is not available")
            helper.toggle_console_control_sensor()
            is_running = getattr(helper, "is_running", self._console_control_sensor_active)
            source = str(getattr(helper, "console_input_source", "ps5"))
            self._console_control_sensor_active = bool(is_running and source == "sensor")
            self._console_control_active = bool(is_running and source == "ps5")
            self._console_control_sensor_v2_active = bool(is_running and source == "sensor_v2")
            if hasattr(self, "_set_button_active"):
                self._set_button_active(
                    self.console_control_sensor_button,
                    self._console_control_sensor_active,
                )
                self._set_button_active(self.console_control_button, self._console_control_active)
                self._set_button_active(
                    self.console_control_sensor_v2_button,
                    self._console_control_sensor_v2_active,
                )
        except Exception as exc:
            message = f"[UI] Console control sensor toggle failed: {exc}"
            print(message)
            if hasattr(self, "log_display"):
                self.log_display.append(message)
            self._console_control_sensor_active = not self._console_control_sensor_active
            self._console_control_sensor_v2_active = False
            if hasattr(self, "_set_button_active"):
                self._set_button_active(
                    self.console_control_sensor_button,
                    self._console_control_sensor_active,
                )
                self._set_button_active(self.console_control_sensor_v2_button, False)

    def _on_toggle_console_control_sensor_v2(self):
        self._console_control_sensor_v2_active = not getattr(
            self, "_console_control_sensor_v2_active", False
        )
        if hasattr(self, "_set_button_active"):
            self._set_button_active(
                self.console_control_sensor_v2_button,
                self._console_control_sensor_v2_active,
            )
        try:
            helper = self._get_sensor_helper("console_control_class")
            if helper is None or not hasattr(helper, "toggle_console_control_sensor_v2"):
                raise AttributeError("console_control_class V2 is not available")
            helper.toggle_console_control_sensor_v2()
            is_running = getattr(helper, "is_running", self._console_control_sensor_v2_active)
            source = str(getattr(helper, "console_input_source", "ps5"))
            self._console_control_sensor_v2_active = bool(is_running and source == "sensor_v2")
            self._console_control_sensor_active = bool(is_running and source == "sensor")
            self._console_control_active = bool(is_running and source == "ps5")
            if hasattr(self, "_set_button_active"):
                self._set_button_active(
                    self.console_control_sensor_v2_button,
                    self._console_control_sensor_v2_active,
                )
                self._set_button_active(
                    self.console_control_sensor_button,
                    self._console_control_sensor_active,
                )
                self._set_button_active(self.console_control_button, self._console_control_active)
        except Exception as exc:
            message = f"[UI] Console control sensor V2 toggle failed: {exc}"
            print(message)
            if hasattr(self, "log_display"):
                self.log_display.append(message)
            self._console_control_sensor_v2_active = not self._console_control_sensor_v2_active
            if hasattr(self, "_set_button_active"):
                self._set_button_active(
                    self.console_control_sensor_v2_button,
                    self._console_control_sensor_v2_active,
                )

    def open_proximity_settings_dialog(self):
        dialog = getattr(self, "proximity_settings_dialog", None)
        if dialog is None:
            return
        self._load_proximity_settings_into_ui()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _collect_proximity_settings_from_ui(self):
        return {
            "frame_interval_ms": int(self.proximity_frame_interval_spin.value()),
            "lateral_speed": float(self.proximity_lateral_speed_spin.value()),
            "normal_speed": float(self.proximity_normal_speed_spin.value()),
            "centroid_deadband": float(self.proximity_centroid_deadband_spin.value()),
            "strength_deadband": float(self.proximity_strength_deadband_spin.value()),
            "max_linear_speed": float(self.proximity_max_linear_speed_spin.value()),
            "center_window_size": int(self.proximity_center_window_spin.value()),
            "smoothing_alpha": float(self.proximity_smoothing_alpha_spin.value()),
            "lost_signal_normal_recovery_frames": int(self.proximity_lost_signal_recovery_frames_spin.value()),
            "lost_signal_normal_speed_ratio": float(self.proximity_lost_signal_speed_ratio_spin.value()),
        }

    def _load_proximity_settings_into_ui(self):
        helper = self._get_sensor_helper("proximity_control_class")
        # ``_FeatureDisabledProxy`` (used before the scene is built) auto-creates a
        # no-op for every attribute, so ``hasattr`` alone always says True; gate
        # on ``bool(helper)`` (the proxy returns False) and confirm the returned
        # settings actually look like a real dict before iterating it.
        if not helper or not hasattr(helper, "get_settings"):
            return
        try:
            settings = helper.get_settings()
            if not isinstance(settings, dict):
                return
            mapping = {
                "frame_interval_ms": self.proximity_frame_interval_spin,
                "lateral_speed": self.proximity_lateral_speed_spin,
                "normal_speed": self.proximity_normal_speed_spin,
                "centroid_deadband": self.proximity_centroid_deadband_spin,
                "strength_deadband": self.proximity_strength_deadband_spin,
                "max_linear_speed": self.proximity_max_linear_speed_spin,
                "center_window_size": self.proximity_center_window_spin,
                "smoothing_alpha": self.proximity_smoothing_alpha_spin,
                "lost_signal_normal_recovery_frames": self.proximity_lost_signal_recovery_frames_spin,
                "lost_signal_normal_speed_ratio": self.proximity_lost_signal_speed_ratio_spin,
            }
            for name, widget in mapping.items():
                if name in settings:
                    widget.setValue(settings[name])
        except Exception as exc:
            print(f"[UI] Failed to load proximity settings into UI: {exc}")

    def _apply_proximity_settings_from_ui(self):
        helper = self._get_sensor_helper("proximity_control_class")
        if helper is None:
            return
        try:
            settings = self._collect_proximity_settings_from_ui()
            if hasattr(helper, "apply_settings"):
                helper.apply_settings(settings, save_to_file=True)
            else:
                helper.apply_runtime_params(**settings)
            print("[UI] Proximity parameters applied.")
        except Exception as exc:
            print(f"[UI] Failed to apply proximity settings: {exc}")

    def toggle_plotter_visibility(self):
        self.log_display.setVisible(not self.log_display.isVisible())
        self.adjust_splitter_sizes()

    def _toggle_robot_editor(self, target_widget):
        """Exclusive robot send panels: only one open; same button closes its panel.

        Opening a panel closes any other panel first. Re-clicking the active
        send-operation button only hides that panel (others stay closed).
        """
        widgets = [
            self.position_entry_widget,
            self.position_quaternion_widget,
            self.position_toolframe_widget,
            self.position_script_widget,
        ]

        if target_widget.isVisible():
            target_widget.toggle_visibility()
        else:
            for w in widgets:
                if w.isVisible():
                    w.toggle_visibility()
            target_widget.toggle_visibility()

        any_open = any(widget.isVisible() for widget in widgets)
        self.read_group_robot.setVisible(not any_open)
        self._sync_robot_send_button_highlights()

    def _sync_robot_send_button_highlights(self):
        """Green highlight on the send-operation button whose panel is open."""
        if not hasattr(self, "_set_button_active"):
            return
        if not all(
            hasattr(self, name)
            for name in (
                "send_position_PTP_J_button",
                "send_position_PTP_T_button",
                "send_position_PTP_T_toolframe_button",
                "send_script_button",
                "position_entry_widget",
                "position_quaternion_widget",
                "position_toolframe_widget",
                "position_script_widget",
            )
        ):
            return
        self._set_button_active(
            self.send_position_PTP_J_button,
            self.position_entry_widget.isVisible(),
        )
        self._set_button_active(
            self.send_position_PTP_T_button,
            self.position_quaternion_widget.isVisible(),
        )
        self._set_button_active(
            self.send_position_PTP_T_toolframe_button,
            self.position_toolframe_widget.isVisible(),
        )
        self._set_button_active(
            self.send_script_button,
            self.position_script_widget.isVisible(),
        )

    def _on_transmit_robot_script(self, script: str):
        if not self.features.get("robot_ready", False):
            self.log_display.append("⚠️ Robot is not ready (ROS / SendScript unavailable).")
            return
        if not script:
            self.log_display.append("⚠️ Script is empty. Enter a TM script, then press Transmit script.")
            return
        api = getattr(self, "robot_api", None)
        if api is None or not hasattr(api, "send_request"):
            self.log_display.append("⚠️ Robot API has no send_request.")
            return
        ok = bool(api.send_request(script))
        preview = script if len(script) <= 160 else script[:160] + "…"
        if ok:
            self.log_display.append(f"📤 SendScript queued: {preview}")
        else:
            self.log_display.append("❌ send_request returned false (check ROS / send_script service).")

    def toggle_robot_script_input(self):
        self._toggle_robot_editor(self.position_script_widget)

    def toggle_joint_angle_input(self):
        self._toggle_robot_editor(self.position_entry_widget)

    def toggle_tool_position_input(self):
        self._toggle_robot_editor(self.position_quaternion_widget)

    def toggle_tool_frame_position_input(self):
        self._toggle_robot_editor(self.position_toolframe_widget)

    def show_log_if_hidden(self):
        if not self.log_display.isVisible():
            self.log_display.setVisible(True)
            self.adjust_splitter_sizes()

    def _update_slider_label(self, value):
        self.gripper_label.setText(f"{value/100.0:.2f}")

    def _on_slider_released(self):
        if not self.features.get("gripper_ready", False):
            self.log_display.append("⚠️ Gripper is unavailable.")
            return

        val_int = self.gripper_slider.value()
        self.gripper.set_slider_pos(val_int)

    def set_gripper_manual(self, val_int):
        self.gripper_slider.setValue(val_int)
        self._update_slider_label(val_int)
        self._on_slider_released()

    # ------------------------------------------------------------------
    # Dexterous hand (RH56F1)
    # ------------------------------------------------------------------
    def _hand_log(self, message: str):
        if hasattr(self, "log_display"):
            self.log_display.append(message)
        if hasattr(self, "hand_state_label"):
            self.hand_state_label.setText(str(message))

    def _hand_api_available(self):
        api = getattr(self, "robot_api", None)
        return api is not None and hasattr(api, "hand_services_available") and api.hand_services_available()

    def _hand_tactile_api_available(self):
        api = getattr(self, "robot_api", None)
        return api is not None and hasattr(api, "hand_tactile_available") and api.hand_tactile_available()

    def _ensure_hand_async_worker(self):
        """Create the single-worker hand command queue lazily.

        RH56F1 service helpers wait for ROS futures internally. Running them on
        this worker keeps slider drags and button clicks from freezing Qt's main
        thread while preserving command order.
        """
        if getattr(self, "_hand_async_executor", None) is not None:
            return True
        self._hand_async_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rh56f1-hand",
        )
        self._hand_async_signals = _HandAsyncSignals()
        self._hand_async_signals.result.connect(self._on_hand_async_result)
        self._hand_async_signals.error.connect(self._on_hand_async_error)
        self._hand_live_inflight = False
        self._hand_live_deferred = None
        return True

    def _submit_hand_worker(self, kind: str, label: str, call_fn, *, live: bool = False):
        if not self._hand_api_available():
            self._hand_log(
                "RH56F1 command interface unavailable. Check /set_angle_data or /Setangle."
            )
            return
        self._ensure_hand_async_worker()

        if live and bool(getattr(self, "_hand_live_inflight", False)):
            # The hand service can take much longer than slider valueChanged
            # events. Keep only the newest live command so dragging does not
            # build a stale backlog.
            self._hand_live_deferred = (kind, label, call_fn)
            return

        if live:
            self._hand_live_inflight = True

        api = self.robot_api
        signals = self._hand_async_signals

        def job():
            return call_fn(api)

        future = self._hand_async_executor.submit(job)

        def done_callback(fut):
            try:
                result = fut.result()
            except Exception as exc:
                signals.error.emit(kind, label, str(exc))
                return
            signals.result.emit(kind, label, result)

        future.add_done_callback(done_callback)

    def _submit_hand_angles_async(self, angles, label="Set angles", *, live: bool = False):
        payload = [int(v) for v in list(angles or [])[:6]]
        if len(payload) < 6:
            payload += [-1] * (6 - len(payload))
        kind = "angles_live" if live else "angles"
        self._submit_hand_worker(
            kind,
            label,
            lambda api, a=payload: api.hand_set_angles(a),
            live=live,
        )

    def _on_hand_async_result(self, kind: str, label: str, result):
        if kind == "read_angles_sync":
            self._apply_hand_actual_angles_to_sliders(result)
            self._hand_log(f"{label}: {'OK' if result is not None else 'FAILED'}")
        elif kind == "read_angles":
            self._hand_log(
                f"{label}: {result}" if result is not None else f"{label}: FAILED"
            )
        else:
            self._hand_log(f"{label}: {'OK' if result is not None else 'FAILED'}")

        if kind == "angles_live":
            self._hand_live_inflight = False
            deferred = getattr(self, "_hand_live_deferred", None)
            self._hand_live_deferred = None
            live_check = getattr(self, "hand_sliders_live_check", None)
            if deferred is not None and live_check is not None and live_check.isChecked():
                d_kind, d_label, d_call_fn = deferred
                self._submit_hand_worker(d_kind, d_label, d_call_fn, live=True)

    def _on_hand_async_error(self, kind: str, label: str, message: str):
        self._hand_log(f"{label}: FAILED ({message})")
        if kind == "angles_live":
            self._hand_live_inflight = False
            self._hand_live_deferred = None

    def _shutdown_hand_async_worker(self):
        tactile_timer = getattr(self, "hand_tactile_timer", None)
        if tactile_timer is not None:
            tactile_timer.stop()
        tactile_button = getattr(self, "hand_tactile_live_button", None)
        if tactile_button is not None:
            tactile_button.blockSignals(True)
            tactile_button.setChecked(False)
            tactile_button.setText("Start Live Tactile")
            tactile_button.blockSignals(False)
        api = getattr(self, "robot_api", None)
        if api is not None and hasattr(api, "enable_hand_tactile_subscription"):
            try:
                api.enable_hand_tactile_subscription(False)
            except Exception:
                pass

        timer = getattr(self, "_hand_live_timer", None)
        if timer is not None:
            timer.stop()
        self._hand_live_pending = {}
        self._hand_live_deferred = None
        self._hand_live_inflight = False
        executor = getattr(self, "_hand_async_executor", None)
        self._hand_async_executor = None
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)

    def _send_hand_angles(self, angles, label="Set angles"):
        self._submit_hand_angles_async(angles, label=label)

    def _on_hand_tactile_live_toggled(self, checked: bool):
        button = getattr(self, "hand_tactile_live_button", None)
        timer = getattr(self, "hand_tactile_timer", None)
        api = getattr(self, "robot_api", None)

        if checked:
            if not self._hand_tactile_api_available() or api is None:
                if button is not None:
                    button.blockSignals(True)
                    button.setChecked(False)
                    button.setText("Start Live Tactile")
                    button.blockSignals(False)
                self._hand_log("RH56F1 tactile topic unavailable. Start the hand topic node and check /touch_data.")
                return

            ok = bool(api.enable_hand_tactile_subscription(True))
            if not ok:
                if button is not None:
                    button.blockSignals(True)
                    button.setChecked(False)
                    button.setText("Start Live Tactile")
                    button.blockSignals(False)
                self._hand_log("Failed to subscribe to /touch_data.")
                return

            if button is not None:
                button.setText("Stop Live Tactile")
                self._set_button_active(button, True)
            if timer is not None:
                timer.start()
            self._refresh_hand_tactile_display()
            return

        if timer is not None:
            timer.stop()
        if api is not None and hasattr(api, "enable_hand_tactile_subscription"):
            api.enable_hand_tactile_subscription(False)
        if button is not None:
            button.setText("Start Live Tactile")
            self._set_button_active(button, False)
        status = getattr(self, "hand_tactile_status_label", None)
        if status is not None:
            status.setText("Live tactile stopped")

    @staticmethod
    def _hand_force_text(raw_value):
        try:
            raw_int = int(raw_value)
        except Exception:
            return "--"
        return f"{raw_int / 100.0:.2f}"

    @staticmethod
    def _hand_direction_text(raw_value):
        try:
            raw_int = int(raw_value)
        except Exception:
            return "--"
        if raw_int in {65535, 0xFFFF}:
            return "invalid"
        return f"{raw_int} deg"

    @staticmethod
    def _hand_raw_text(raw_value):
        try:
            return str(int(raw_value))
        except Exception:
            return "--"

    def _set_hand_tactile_cell(self, row: int, col: int, text: str):
        table = getattr(self, "hand_tactile_table", None)
        if table is None:
            return
        item = table.item(row, col)
        if item is None:
            from PyQt5.QtWidgets import QTableWidgetItem
            item = QTableWidgetItem()
            table.setItem(row, col, item)
        item.setText(str(text))

    def _refresh_hand_tactile_display(self):
        api = getattr(self, "robot_api", None)
        status = getattr(self, "hand_tactile_status_label", None)
        if api is None or not hasattr(api, "get_latest_hand_tactile"):
            if status is not None:
                status.setText("Tactile unavailable")
            return

        data = api.get_latest_hand_tactile()
        if not data:
            if status is not None:
                publisher_count = 0
                if hasattr(api, "hand_tactile_publisher_count"):
                    publisher_count = api.hand_tactile_publisher_count()
                if publisher_count <= 0:
                    status.setText("No /touch_data publisher")
                else:
                    status.setText("Waiting for /touch_data")
            helper = getattr(self, "mesh_functions", None)
            if helper is not None and hasattr(helper, "updateDexterousHandTactile"):
                helper.updateDexterousHandTactile(None)
            return

        finger_forces = list(data.get("finger_forces") or [])
        finger_tangentials = list(data.get("finger_tangentials") or [])
        finger_angles = list(data.get("finger_angles") or [])
        finger_proximity = list(data.get("finger_proximity") or [])
        for row in range(5):
            self._set_hand_tactile_cell(
                row,
                1,
                self._hand_force_text(finger_forces[row]) if row < len(finger_forces) else "--",
            )
            self._set_hand_tactile_cell(
                row,
                2,
                self._hand_force_text(finger_tangentials[row]) if row < len(finger_tangentials) else "--",
            )
            self._set_hand_tactile_cell(
                row,
                3,
                self._hand_direction_text(finger_angles[row]) if row < len(finger_angles) else "--",
            )
            self._set_hand_tactile_cell(
                row,
                4,
                self._hand_raw_text(finger_proximity[row]) if row < len(finger_proximity) else "--",
            )

        palm_data = list(data.get("palm_data") or [])
        for palm_index in range(3):
            row = 5 + palm_index
            base = palm_index * 3
            self._set_hand_tactile_cell(
                row,
                1,
                self._hand_force_text(palm_data[base]) if base < len(palm_data) else "--",
            )
            self._set_hand_tactile_cell(
                row,
                2,
                self._hand_force_text(palm_data[base + 1]) if (base + 1) < len(palm_data) else "--",
            )
            self._set_hand_tactile_cell(
                row,
                3,
                self._hand_direction_text(palm_data[base + 2]) if (base + 2) < len(palm_data) else "--",
            )
            self._set_hand_tactile_cell(row, 4, "--")

        timestamp = data.get("timestamp")
        if status is not None:
            if timestamp:
                age = max(0.0, time.time() - float(timestamp))
                status.setText(f"/touch_data age {age:.2f}s")
            else:
                status.setText("/touch_data received")

        helper = getattr(self, "mesh_functions", None)
        if helper is not None and hasattr(helper, "updateDexterousHandTactile"):
            helper.updateDexterousHandTactile(data)

    def _on_hand_open_all(self):
        self._send_hand_angles([1720, 1720, 1720, 1720, 1350, -1], label="Open all")

    def _on_hand_close_all(self):
        self._send_hand_angles([900, 900, 900, 900, 1100, -1], label="Close all")

    def _on_hand_apply_speed(self):
        speed = int(self.hand_speed_spin.value())
        self._submit_hand_worker(
            "speed",
            f"Set speed={speed}",
            lambda api, s=speed: api.hand_set_speed_all(s),
        )

    def _on_hand_apply_force(self):
        force = int(self.hand_force_spin.value())
        self._submit_hand_worker(
            "force",
            f"Set force={force} (all)",
            lambda api, f=force: api.hand_set_force_all([f] * 6),
        )

    def _on_hand_read_angles(self):
        self._submit_hand_worker(
            "read_angles",
            "Actual angles",
            lambda api: api.hand_get_actual_angles(),
        )

    def _on_hand_thumb_left(self):
        self._send_hand_angles([-1, -1, -1, -1, -1, 600], label="Thumb left")

    def _on_hand_thumb_center(self):
        self._send_hand_angles([-1, -1, -1, -1, -1, 1000], label="Thumb center")

    def _on_hand_thumb_right(self):
        self._send_hand_angles([-1, -1, -1, -1, -1, 1800], label="Thumb right")

    def _on_hand_send_custom_angles(self):
        sliders = getattr(self, "hand_angle_sliders", []) or []
        angles = [int(s.value()) for s in sliders]
        if len(angles) != 6:
            self._hand_log("Custom angles not ready.")
            return
        self._send_hand_angles(angles, label=f"Custom angles {angles}")

    # ------------------------------------------------------------------
    # Per-slider UX (live update / single-finger send / quick presets)
    # ------------------------------------------------------------------
    def _on_hand_slider_value_changed(self, slider_idx: int, value: int):
        """Update the side label live; if the user has 'Live update' ticked
        also stream the change to the hand at ~20 Hz so the motion follows
        the drag in real time without flooding the ROS2 service."""
        labels = getattr(self, "_hand_angle_value_labels", []) or []
        if 0 <= slider_idx < len(labels):
            try:
                labels[slider_idx].setText(str(int(value)))
            except Exception:
                pass

        live_check = getattr(self, "hand_sliders_live_check", None)
        if live_check is None or not live_check.isChecked():
            return
        # Coalesce rapid changes: remember the latest value per slider, then
        # flush after a short delay so dragging fires at most ~20 Hz.
        if not hasattr(self, "_hand_live_pending"):
            self._hand_live_pending = {}
        self._hand_live_pending[int(slider_idx)] = int(value)
        timer = getattr(self, "_hand_live_timer", None)
        if timer is not None and not timer.isActive():
            timer.start()

    def _hand_flush_live_pending(self):
        pending = getattr(self, "_hand_live_pending", None)
        if not pending:
            return
        for slider_idx, value in list(pending.items()):
            angles = [-1, -1, -1, -1, -1, -1]
            if 0 <= slider_idx < 6:
                angles[slider_idx] = int(value)
                self._submit_hand_angles_async(
                    angles,
                    label=f"Live angle{slider_idx}={value}",
                    live=True,
                )
        self._hand_live_pending = {}

    def _on_hand_send_single_slider(self, slider_idx: int):
        sliders = getattr(self, "hand_angle_sliders", []) or []
        if not (0 <= slider_idx < len(sliders)):
            return
        value = int(sliders[slider_idx].value())
        angles = [-1, -1, -1, -1, -1, -1]
        angles[slider_idx] = value
        self._submit_hand_angles_async(angles, label=f"angle{slider_idx}={value}")

    def _on_hand_sliders_load_open(self):
        opens = getattr(self, "_hand_angle_open_values", None) or []
        sliders = getattr(self, "hand_angle_sliders", []) or []
        for i, s in enumerate(sliders):
            if i < len(opens):
                s.setValue(int(opens[i]))
        self._hand_log("Sliders loaded with OPEN preset (not yet sent).")

    def _on_hand_sliders_load_close(self):
        closes = getattr(self, "_hand_angle_close_values", None) or []
        sliders = getattr(self, "hand_angle_sliders", []) or []
        for i, s in enumerate(sliders):
            if i < len(closes):
                s.setValue(int(closes[i]))
        self._hand_log("Sliders loaded with CLOSE preset (not yet sent).")

    def _on_hand_sliders_sync(self):
        self._submit_hand_worker(
            "read_angles_sync",
            "Sync sliders from actual angles",
            lambda api: api.hand_get_actual_angles(),
        )

    def _apply_hand_actual_angles_to_sliders(self, result):
        if not result:
            return
        # ``result`` is expected to expose ``angle0..angle5`` (rh56f1 service).
        sliders = getattr(self, "hand_angle_sliders", []) or []
        labels = getattr(self, "_hand_angle_value_labels", []) or []
        values: list = []
        for i in range(6):
            attr = f"angle{i}"
            try:
                values.append(int(getattr(result, attr)))
            except Exception:
                # Fallback: try dict-like or list-like containers.
                try:
                    values.append(int(result[i]))
                except Exception:
                    values.append(None)
        for i, s in enumerate(sliders):
            if i < len(values) and values[i] is not None:
                v = max(s.minimum(), min(s.maximum(), int(values[i])))
                s.blockSignals(True)
                s.setValue(v)
                s.blockSignals(False)
                if i < len(labels):
                    try:
                        labels[i].setText(str(int(v)))
                    except Exception:
                        pass
        self._hand_log(f"Sliders synced from actual angles: {values}")

    def adjust_splitter_sizes(self):
        total_width = self.splitter_1.width()
        if self.log_display.isVisible():
            self.splitter_1.setSizes([int(total_width * 0.4), int(total_width * 0.4), int(total_width * 0.2)])
        else:
            self.splitter_1.setSizes([int(total_width * 0.5), int(total_width * 0.5), 0])

    def reLayout(self):
        self.setSizes([round(self.width() * 4), round(self.width())])
        self.splitter_1.setSizes([self.width(), self.width(), 0])

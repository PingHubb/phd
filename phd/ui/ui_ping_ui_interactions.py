from __future__ import annotations


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
        self.direct_finger_motion_v2_button.pressed.connect(self._on_toggle_direct_finger_motion_v2)
        self.console_control_button.pressed.connect(self._on_toggle_console_control)
        self.console_control_sensor_button.pressed.connect(
            self._on_toggle_console_control_sensor_placeholder
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
        self.apply_direct_finger_motion_v2_settings_button.clicked.connect(
            self._apply_direct_finger_motion_v2_settings_from_ui
        )
        self.reload_direct_finger_motion_v2_settings_button.clicked.connect(
            self._load_direct_finger_motion_v2_settings_into_ui
        )
        self.load_direct_finger_motion_v2_profile_button.clicked.connect(
            self._load_direct_finger_motion_v2_profile_from_ui
        )
        self.apply_console_control_settings_button.clicked.connect(
            self._apply_console_control_settings_from_ui
        )
        self.reload_console_control_settings_button.clicked.connect(
            self._load_console_control_settings_into_ui
        )
        self.ai_direct_finger_motion_button.pressed.connect(self._on_toggle_ai_direct_finger_motion)
        self.ai_direct_finger_motion_execution_button.pressed.connect(
            self._on_toggle_ai_direct_finger_motion_execution
        )
        self.update_sensor_button.pressed.connect(self._on_sensor_update)
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        self.sensor_average_window_spin.valueChanged.connect(self._on_sensor_average_window_changed)
        self.visualization_target_hz_spin.valueChanged.connect(self._on_visualization_target_hz_changed)
        self.btn_toggle_anchor_axes.pressed.connect(self._on_toggle_anchor_axes)

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
        self.mesh_functions.addRobot()

    def _on_build_scene(self):
        self.sensor_functions.buildScene()

    def _on_sensor_average_window_changed(self, value: int):
        self.sensor_functions.set_sensor_average_window_size(value)

    def _on_visualization_target_hz_changed(self, value: float):
        self.sensor_functions.set_visualization_target_hz(value)

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

    def _on_toggle_direct_finger_motion_v2(self):
        self._direct_finger_v2_active = not getattr(self, "_direct_finger_v2_active", False)
        if hasattr(self, "_set_button_active"):
            self._set_button_active(self.direct_finger_motion_v2_button, self._direct_finger_v2_active)
        try:
            helper = self._get_sensor_helper("direct_finger_motion_v2_class")
            if helper is None:
                raise AttributeError("direct_finger_motion_v2_class is not available")
            helper.toggle_direct_finger_motion_v2()
            self._direct_finger_v2_active = bool(getattr(helper, "is_running", self._direct_finger_v2_active))
            if hasattr(self, "_set_button_active"):
                self._set_button_active(self.direct_finger_motion_v2_button, self._direct_finger_v2_active)
        except Exception as exc:
            print(f"[UI] Direct finger motion v2 toggle failed: {exc}")
            self._direct_finger_v2_active = not self._direct_finger_v2_active
            if hasattr(self, "_set_button_active"):
                self._set_button_active(self.direct_finger_motion_v2_button, self._direct_finger_v2_active)

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
            if hasattr(self, "_set_button_active"):
                self._set_button_active(self.console_control_button, self._console_control_active)
                self._set_button_active(
                    self.console_control_sensor_button,
                    self._console_control_sensor_active,
                )
        except Exception as exc:
            message = f"[UI] Console control toggle failed: {exc}"
            print(message)
            if hasattr(self, "log_display"):
                self.log_display.append(message)
            self._console_control_active = not self._console_control_active
            self._console_control_sensor_active = False
            if hasattr(self, "_set_button_active"):
                self._set_button_active(self.console_control_button, self._console_control_active)
                self._set_button_active(self.console_control_sensor_button, False)

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
            if hasattr(self, "_set_button_active"):
                self._set_button_active(
                    self.console_control_sensor_button,
                    self._console_control_sensor_active,
                )
                self._set_button_active(self.console_control_button, self._console_control_active)
        except Exception as exc:
            message = f"[UI] Console control sensor toggle failed: {exc}"
            print(message)
            if hasattr(self, "log_display"):
                self.log_display.append(message)
            self._console_control_sensor_active = not self._console_control_sensor_active
            if hasattr(self, "_set_button_active"):
                self._set_button_active(
                    self.console_control_sensor_button,
                    self._console_control_sensor_active,
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
        if helper is None or not hasattr(helper, "get_settings"):
            return
        try:
            settings = helper.get_settings()
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

    def adjust_splitter_sizes(self):
        total_width = self.splitter_1.width()
        if self.log_display.isVisible():
            self.splitter_1.setSizes([int(total_width * 0.4), int(total_width * 0.4), int(total_width * 0.2)])
        else:
            self.splitter_1.setSizes([int(total_width * 0.5), int(total_width * 0.5), 0])

    def reLayout(self):
        self.setSizes([round(self.width() * 4), round(self.width())])
        self.splitter_1.setSizes([self.width(), self.width(), 0])

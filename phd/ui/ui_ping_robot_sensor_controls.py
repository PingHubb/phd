from __future__ import annotations

class RobotSensorControlsMixin:
    def set_robot_subtab_enabled(self, enabled: bool):
        self._set_widgets_enabled(
            [
                self.read_joint_angle_button,
                self.read_tool_position_button,
                self.send_position_PTP_J_button,
                self.send_position_PTP_T_button,
                self.send_position_PTP_T_toolframe_button,
                self.position_script_widget,
                self.send_script_button,
                self.show_robot_button,
                self.continuous_read_button,
            ],
            enabled,
        )
        self.robots_sub_tabs.setTabEnabled(0, enabled)

    def disable_robot_controls(self, disable: bool):
        return None

    def _on_sensor_api_read_raw(self):
        if self.ensure_sensor_api():
            self.log_display.append(f"API raw data: {self.sensor_api.read_raw()}")
        else:
            self.log_display.append("Sensor API is not ready yet.")

    def _on_sensor_api_read_raw_hz(self):
        if self.ensure_sensor_api():
            result = self.sensor_api.measure_read_raw_hz(duration_sec=1.0)
            if not result:
                self.log_display.append("Sensor API raw Hz measurement failed.")
                return
            self.log_display.append(
                "Sensor API raw Hz: "
                f"{result['hz']:.2f} | "
                f"success={result['success_count']} / attempts={result['total_attempts']} | "
                f"elapsed={result['elapsed_sec']:.2f}s"
            )
        else:
            self.log_display.append("Sensor API is not ready yet.")

    def _on_sensor_api_channel_check(self):
        if self.ensure_sensor_api():
            self.log_display.append(f"Sensor channel data: {self.sensor_api.channel_check()}")
        else:
            self.log_display.append("Sensor API is not ready yet.")

    def _on_sensor_update(self):
        if self.ensure_sensor_api():
            self.sensor_functions.updateCal()
        else:
            self.log_display.append("Cannot calibrate: Sensor API is not ready.")

    def start_record_gesture(self):
        gesture_number = self.gesture_number_input.text().strip()
        if not gesture_number:
            self.log_display.append("Please enter a gesture number or name.")
            return
        helper = self._get_sensor_helper("record_gesture_class")
        if helper is None:
            self.log_display.append("Record gesture helper is not ready yet.")
            return
        helper.start_record_gesture(gesture_number)

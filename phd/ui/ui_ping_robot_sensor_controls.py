from __future__ import annotations

import os
import threading
import time

from PyQt5 import QtCore

from phd.dependence.goodix_usb_sensor import (
    GOODIX_USB_COLUMNS,
    GOODIX_USB_ROWS,
    GOODIX_USB_SOURCE_ID,
    GoodixUsbError,
    is_goodix_usb_source,
)


class _SensorHzResultBridge(QtCore.QObject):
    """Delivers a background Hz measurement result back to the GUI thread."""

    finished = QtCore.pyqtSignal(object)


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
            ],
            enabled,
        )
        self.robots_sub_tabs.setTabEnabled(0, enabled)

    def disable_robot_controls(self, disable: bool):
        return None

    def _get_selected_sensor_port_path(self):
        port_list = getattr(self, "serial_channel", None)
        if port_list is None:
            return None

        selected_items = list(port_list.selectedItems() or [])
        if not selected_items:
            return None
        current_item = port_list.currentItem()
        item = current_item if current_item in selected_items else selected_items[0]
        try:
            stored_port = item.data(QtCore.Qt.UserRole)
        except (AttributeError, TypeError):
            stored_port = None
        port_name = str(stored_port or item.text() or "").strip()
        if not port_name:
            return None
        if is_goodix_usb_source(port_name):
            return GOODIX_USB_SOURCE_ID
        if os.path.isabs(port_name):
            return port_name
        return os.path.join("/dev", port_name)

    def _prepare_selected_sensor_api(self):
        port_path = self._get_selected_sensor_port_path()
        if not port_path:
            self.log_display.append(
                "Select a serial port in Sensor > Send Operation first."
            )
            return None
        if is_goodix_usb_source(port_path):
            return port_path
        if not self.ensure_sensor_api(serial_port=port_path):
            self.log_display.append(f"Could not open selected sensor port: {port_path}")
            return None
        return port_path

    def _on_sensor_api_read_raw(self):
        port_path = self._get_selected_sensor_port_path()
        if not port_path:
            self.log_display.append(
                "Select a serial port in Sensor > Send Operation first."
            )
            return

        sensor = getattr(self, "sensor_functions", None)
        reader_running = bool(
            sensor is not None
            and getattr(sensor, "_sensor_reader_is_running", lambda: False)()
        )
        if reader_running:
            getter = getattr(sensor, "get_last_sensor_api_payload", None)
            payload = getter(port_path) if callable(getter) else None
            if payload is None:
                self.log_display.append(
                    f"API raw data ({port_path}): waiting for a live sensor frame."
                )
            else:
                self.log_display.append(
                    f"API raw data ({port_path}, live): {payload}"
                )
            return

        if is_goodix_usb_source(port_path):
            try:
                payload = sensor.read_goodix_raw_frame()
            except (AttributeError, GoodixUsbError) as exc:
                self.log_display.append(f"Goodix USB raw read failed: {exc}")
            else:
                self.log_display.append(
                    f"API raw data ({GOODIX_USB_SOURCE_ID}): {payload}"
                )
            return

        port_path = self._prepare_selected_sensor_api()
        if port_path:
            self.log_display.append(
                f"API raw data ({port_path}): {self.sensor_api.read_raw()}"
            )

    def _on_sensor_api_read_raw_hz(self):
        if getattr(self, "_sensor_hz_measurement_running", False):
            self.log_display.append("Sensor Hz measurement is already in progress.")
            return

        port_path = self._get_selected_sensor_port_path()
        if not port_path:
            self.log_display.append(
                "Select a serial port in Sensor > Send Operation first."
            )
            return

        # While the live visualization is streaming, the serial port belongs to
        # the background reader thread. Poking the port from here would corrupt
        # both streams (and can close the shared port, freezing the display),
        # so measure the rate passively from the frames that already arrive.
        sensor = getattr(self, "sensor_functions", None)
        reader_is_running = getattr(
            sensor, "_sensor_reader_is_running", lambda: False
        )
        if sensor is not None and reader_is_running():
            self._measure_stream_hz(sensor)
            return

        if is_goodix_usb_source(port_path):
            client = getattr(sensor, "_goodix_client", None)
            if client is None:
                self.log_display.append(
                    "Build the Goodix USB sensor scene before measuring its rate."
                )
                return
            self._measure_direct_hz(api=client, label="Goodix USB raw Hz")
            return

        if not self.ensure_sensor_api(serial_port=port_path):
            self.log_display.append(f"Could not open selected sensor port: {port_path}")
            return
        self._measure_direct_hz()

    def _set_sensor_hz_measurement_running(self, running: bool):
        self._sensor_hz_measurement_running = bool(running)
        button = getattr(self, "read_sensor_api_hz_button", None)
        if button is not None:
            button.setEnabled(not running)

    def _measure_stream_hz(self, sensor, duration_ms=1000):
        data_obj = getattr(sensor, "_data", None)
        start_seq = getattr(data_obj, "frame_sequence", None)
        if start_seq is None:
            self.log_display.append("Sensor stream is running but its frame counter is unavailable.")
            return

        self._set_sensor_hz_measurement_running(True)
        self.log_display.append("Measuring live sensor stream rate over 1s...")
        started = time.perf_counter()
        start_seq = int(start_seq)

        def finish():
            self._set_sensor_hz_measurement_running(False)
            elapsed = max(time.perf_counter() - started, 1e-9)
            current = getattr(getattr(sensor, "_data", None), "frame_sequence", None)
            if current is None:
                self.log_display.append("Sensor stream stopped during the Hz measurement.")
                return
            frames = max(0, int(current) - start_seq)
            self.log_display.append(
                "Sensor stream Hz (live, non-intrusive): "
                f"{frames / elapsed:.2f} | frames={frames} | elapsed={elapsed:.2f}s"
            )

        QtCore.QTimer.singleShot(int(duration_ms), finish)

    def _measure_direct_hz(self, api=None, label="Sensor API raw Hz"):
        self._set_sensor_hz_measurement_running(True)
        self.log_display.append(f"Measuring {label} over 1s (background)...")

        bridge = _SensorHzResultBridge(self)
        bridge.finished.connect(self._on_sensor_hz_measured)
        api = api or self.sensor_api

        def worker():
            try:
                result = api.measure_read_raw_hz(duration_sec=1.0)
            except Exception as exc:
                result = exc
            # Queued signal: the handler runs back on the GUI thread.
            bridge.finished.emit(result)

        threading.Thread(target=worker, name="sensor-hz-measure", daemon=True).start()

    def _on_sensor_hz_measured(self, result):
        self._set_sensor_hz_measurement_running(False)
        bridge = self.sender()
        if bridge is not None:
            bridge.deleteLater()

        if isinstance(result, Exception):
            self.log_display.append(f"Sensor API raw Hz measurement failed: {result}")
            return
        if not result:
            self.log_display.append("Sensor API raw Hz measurement failed.")
            return
        backend = str(result.get("backend", "") or "").strip()
        backend_text = f" | backend={backend}" if backend else ""
        self.log_display.append(
            "Sensor API raw Hz: "
            f"{result['hz']:.2f} | "
            f"success={result['success_count']} / attempts={result['total_attempts']} | "
            f"elapsed={result['elapsed_sec']:.2f}s"
            f"{backend_text}"
        )

    def _on_sensor_api_channel_check(self):
        port_path = self._get_selected_sensor_port_path()
        if is_goodix_usb_source(port_path):
            sensor = getattr(self, "sensor_functions", None)
            client = getattr(sensor, "_goodix_client", None)
            backend = (
                f" Backend: {client.backend_status}."
                if client is not None
                else ""
            )
            self.log_display.append(
                "Goodix USB channels: "
                f"{GOODIX_USB_ROWS} rows x {GOODIX_USB_COLUMNS} columns = "
                f"{GOODIX_USB_ROWS * GOODIX_USB_COLUMNS} taxels "
                "(drv_num=8, sen_num=10)."
                f"{backend}"
            )
            return
        sensor = getattr(self, "sensor_functions", None)
        if sensor is not None and getattr(
            sensor, "_sensor_reader_is_running", lambda: False
        )():
            self.log_display.append(
                "Stop the live sensor stream before running Sensor API Channel."
            )
            return
        port_path = self._prepare_selected_sensor_api()
        if port_path:
            self.log_display.append(
                f"Sensor channel data ({port_path}): "
                f"{self.sensor_api.channel_check()}"
            )

    def _on_sensor_update(self):
        sensor = getattr(self, "sensor_functions", None)
        if sensor is not None and getattr(
            sensor,
            "is_goodix_usb_transport",
            lambda: False,
        )():
            sensor.updateCal()
            return
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

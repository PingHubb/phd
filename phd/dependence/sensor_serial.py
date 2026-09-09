"""Shared serial defaults and background reader for tactile sensors."""

import time

from PyQt5.QtCore import QObject, pyqtSignal

from phd.dependence.sensor_protocol import (
    DEFAULT_SENSOR_BAUD_RATE,
    DEFAULT_SENSOR_IDLE_SLEEP_SEC,
    DEFAULT_SENSOR_RESPONSE_TIMEOUT_SEC,
    SENSOR_READ_RAW_COMMAND,
    parse_serial_ints,
    read_complete_serial_line,
    sensor_response_timeout_for_values,
)


class SensorReadWorker(QObject):
    """Continuously read raw payloads without blocking the Qt event loop."""

    raw_payload_ready = pyqtSignal(int, str, list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        serial_ports,
        generation=0,
        response_timeout=DEFAULT_SENSOR_RESPONSE_TIMEOUT_SEC,
        idle_sleep_sec=DEFAULT_SENSOR_IDLE_SLEEP_SEC,
        expected_payload_values=None,
        expected_payload_values_by_port=None,
    ):
        super().__init__()
        self.serial_ports = list(serial_ports or [])
        self.generation = int(generation)
        self.response_timeout = max(0.05, float(response_timeout))
        self.idle_sleep_sec = max(0.0, float(idle_sleep_sec))
        self.expected_payload_values = (
            None
            if expected_payload_values is None
            else max(0, int(expected_payload_values))
        )
        self.expected_payload_values_by_port = {
            str(port): max(0, int(value))
            for port, value in dict(
                expected_payload_values_by_port or {}
            ).items()
        }
        self._running = False

    def stop(self):
        self._running = False

    def _request_raw_from_port(self, serial_port):
        port_name = str(getattr(serial_port, "port", "unknown"))
        try:
            serial_port.write(SENSOR_READ_RAW_COMMAND)
            return True
        except Exception as exc:
            self.error.emit(f"Sensor write failed on {port_name}: {exc}")
            return False

    def _read_raw_response_from_port(self, serial_port):
        port_name = str(getattr(serial_port, "port", "unknown"))
        baud_rate = getattr(
            serial_port, "baudrate", DEFAULT_SENSOR_BAUD_RATE
        )
        expected_values = self.expected_payload_values_by_port.get(
            port_name,
            self.expected_payload_values or 0,
        )
        response_timeout = sensor_response_timeout_for_values(
            expected_values,
            baud_rate=baud_rate,
            minimum=self.response_timeout,
        )
        deadline = time.perf_counter() + response_timeout
        while self._running and time.perf_counter() < deadline:
            try:
                line = read_complete_serial_line(
                    serial_port,
                    timeout=max(0.0, deadline - time.perf_counter()),
                    idle_sleep_sec=self.idle_sleep_sec,
                    should_continue=lambda: self._running,
                )
            except Exception as exc:
                self.error.emit(
                    f"Sensor line read failed on {port_name}: {exc}"
                )
                return None

            if not line:
                continue

            values = parse_serial_ints(line)
            if values:
                return values[2:-2] if len(values) >= 4 else values

        return None

    def _read_raw_from_port(self, serial_port):
        if not self._request_raw_from_port(serial_port):
            return None
        return self._read_raw_response_from_port(serial_port)

    def _read_cycle(self):
        emitted = False
        requested_ports = []
        for serial_port in self.serial_ports:
            if not self._running:
                break
            if self._request_raw_from_port(serial_port):
                requested_ports.append(serial_port)

        for serial_port in requested_ports:
            if not self._running:
                break
            values = self._read_raw_response_from_port(serial_port)
            if values is None:
                continue
            self.raw_payload_ready.emit(
                self.generation,
                str(getattr(serial_port, "port", "sensor")),
                list(values),
            )
            emitted = True
        return emitted

    def run(self):
        self._running = True
        try:
            while self._running:
                emitted = self._read_cycle()
                if not emitted:
                    time.sleep(self.idle_sleep_sec)
        finally:
            self.finished.emit()


class SensorCalibrationBridge(QObject):
    """Carry a background calibration result back to the GUI thread."""

    finished = pyqtSignal(bool)


class SensorPayloadBridge(QObject):
    """Deliver reader-thread payloads to a sensor object on the GUI thread."""

    frame_processed = pyqtSignal(float, int, object, object)
    port_frame_processed = pyqtSignal(
        str,
        float,
        int,
        object,
        object,
    )

    def __init__(self, sensor):
        super().__init__()
        self._sensor = sensor

    def deliver(self, generation, port_name, data_list):
        self._sensor._on_sensor_reader_payload(
            generation, port_name, data_list
        )
        self._sensor.update_animation()
        data_getter = getattr(
            self._sensor,
            "_sensor_data_for_port",
            None,
        )
        data_obj = (
            data_getter(port_name)
            if callable(data_getter)
            else getattr(self._sensor, "_data", None)
        )
        if data_obj is None:
            return
        raw = getattr(data_obj, "rawData", None)
        calibration = getattr(data_obj, "calData", None)
        if raw is None or calibration is None:
            return
        raw_copy = raw.copy() if hasattr(raw, "copy") else raw
        calibration_copy = (
            calibration.copy()
            if hasattr(calibration, "copy")
            else calibration
        )
        timestamp = time.perf_counter()
        frame_sequence = int(
            getattr(data_obj, "frame_sequence", 0)
        )
        self.port_frame_processed.emit(
            str(port_name),
            timestamp,
            frame_sequence,
            raw_copy,
            calibration_copy,
        )
        is_primary = getattr(self._sensor, "is_primary_sensor_port", None)
        if callable(is_primary) and not is_primary(port_name):
            return
        self.frame_processed.emit(
            timestamp,
            frame_sequence,
            raw_copy,
            calibration_copy,
        )

"""Shared serial defaults and background reader for tactile sensors."""

import time

from PyQt5.QtCore import QObject, pyqtSignal

from phd.dependence.sensor_protocol import (
    DEFAULT_SENSOR_IDLE_SLEEP_SEC,
    DEFAULT_SENSOR_RESPONSE_TIMEOUT_SEC,
    SENSOR_READ_RAW_COMMAND,
    parse_serial_ints,
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
    ):
        super().__init__()
        self.serial_ports = list(serial_ports or [])
        self.generation = int(generation)
        self.response_timeout = max(0.05, float(response_timeout))
        self.idle_sleep_sec = max(0.0, float(idle_sleep_sec))
        self._running = False

    def stop(self):
        self._running = False

    def _read_raw_from_port(self, serial_port):
        port_name = str(getattr(serial_port, "port", "unknown"))
        try:
            serial_port.write(SENSOR_READ_RAW_COMMAND)
        except Exception as exc:
            self.error.emit(f"Sensor write failed on {port_name}: {exc}")
            return None

        deadline = time.perf_counter() + self.response_timeout
        while self._running and time.perf_counter() < deadline:
            try:
                waiting = int(getattr(serial_port, "in_waiting", 0))
            except Exception as exc:
                self.error.emit(f"Sensor read failed on {port_name}: {exc}")
                return None

            if waiting <= 0:
                time.sleep(self.idle_sleep_sec)
                continue

            try:
                line = serial_port.readline().decode("utf-8", errors="ignore").rstrip()
            except Exception as exc:
                self.error.emit(f"Sensor line read failed on {port_name}: {exc}")
                return None

            if not line:
                continue

            values = parse_serial_ints(line)
            if values:
                return values[2:-2] if len(values) >= 4 else values

        return None

    def run(self):
        self._running = True
        try:
            while self._running:
                emitted = False
                for serial_port in self.serial_ports:
                    if not self._running:
                        break
                    values = self._read_raw_from_port(serial_port)
                    if values is None:
                        continue
                    self.raw_payload_ready.emit(
                        self.generation,
                        str(getattr(serial_port, "port", "sensor")),
                        list(values),
                    )
                    emitted = True
                if not emitted:
                    time.sleep(self.idle_sleep_sec)
        finally:
            self.finished.emit()


class SensorCalibrationBridge(QObject):
    """Carry a background calibration result back to the GUI thread."""

    finished = pyqtSignal(bool)


class SensorPayloadBridge(QObject):
    """Deliver reader-thread payloads to a sensor object on the GUI thread."""

    def __init__(self, sensor):
        super().__init__()
        self._sensor = sensor

    def deliver(self, generation, port_name, data_list):
        self._sensor._on_sensor_reader_payload(generation, port_name, data_list)
        self._sensor.update_animation()

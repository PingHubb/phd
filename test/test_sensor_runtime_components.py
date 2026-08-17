import numpy as np

from phd.dependence.sensor_api import ArduinoCommander
from phd.dependence.sensor_data import SensorDataBuffer
from phd.dependence.sensor_protocol import (
    DEFAULT_SENSOR_BAUD_RATE,
    DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC,
    SENSOR_READ_RAW_COMMAND,
    parse_serial_ints,
)
from phd.dependence.sensor_serial import SensorReadWorker


class _FakeSerial:
    port = "/dev/fake-sensor"

    def __init__(self, line):
        self._line = line
        self.writes = []

    @property
    def in_waiting(self):
        return len(self._line)

    def write(self, payload):
        self.writes.append(payload)

    def readline(self):
        line, self._line = self._line, b""
        return line


def test_serial_defaults_are_used_by_sensor_api():
    api = ArduinoCommander(connect_immediately=False)

    assert api.baud_rate == DEFAULT_SENSOR_BAUD_RATE
    assert api.timeout == DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC


def test_parse_serial_ints_ignores_non_numeric_debug_tokens():
    assert parse_serial_ints("debug 10 -2 invalid 35") == [10, -2, 35]


def test_sensor_worker_requests_and_unframes_one_raw_payload():
    serial_port = _FakeSerial(b"90 91 10 20 30 92 93\n")
    worker = SensorReadWorker([serial_port])
    worker._running = True

    assert worker._read_raw_from_port(serial_port) == [10, 20, 30]
    assert serial_port.writes == [SENSOR_READ_RAW_COMMAND]


def test_sensor_data_buffer_calculates_safe_percentage_difference():
    buffer = SensorDataBuffer(2, 2, window_size=1)
    buffer.getCal(np.array([[10.0, 0.0], [20.0, 40.0]]))
    buffer.getRaw(np.array([[15.0, 9.0], [10.0, 60.0]]))

    buffer.calDiff()
    buffer.calDiffPer()
    buffer.getWin(1)

    np.testing.assert_allclose(
        buffer.diffPerData,
        np.array([[50.0, 0.0], [-50.0, 50.0]]),
    )
    np.testing.assert_allclose(buffer.diffPerDataAve, np.flipud(buffer.diffPerData))
    assert buffer.frame_sequence == 1


def test_sensor_data_buffer_resize_preserves_current_frame_as_window_seed():
    buffer = SensorDataBuffer(2, 1, window_size=1)
    buffer.getRaw(np.array([[3.0], [7.0]]))
    buffer.setWindowSize(4)

    assert buffer.rawDataWin.shape == (4, 2, 1)
    np.testing.assert_allclose(buffer.rawDataWin[0], buffer.rawData)
    np.testing.assert_allclose(buffer.rawDataAve, np.flipud(buffer.rawData))

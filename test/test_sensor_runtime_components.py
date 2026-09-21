import time
from types import SimpleNamespace

import numpy as np

import phd.dependence.sensor_serial as sensor_serial_module
from phd.dependence.sensor_api import ArduinoCommander
from phd.dependence.sensor_data import SensorDataBuffer
from phd.dependence.func_sensor import (
    MySensor,
    SensorModelFactory,
    _MultiPortSensorReplica,
)
from phd.dependence.sensor_protocol import (
    DEFAULT_SENSOR_BAUD_RATE,
    DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC,
    SENSOR_READ_RAW_COMMAND,
    parse_serial_ints,
    read_complete_serial_line,
    sensor_response_timeout_for_values,
    take_complete_sensor_payload,
)
from phd.dependence.sensor_serial import SensorPayloadBridge, SensorReadWorker


class _FakeSerial:
    port = "/dev/fake-sensor"

    def __init__(self, line):
        self._line = line
        self.writes = []
        self.input_resets = 0

    @property
    def in_waiting(self):
        return len(self._line)

    def reset_input_buffer(self):
        self.input_resets += 1

    def write(self, payload):
        self.writes.append(payload)

    def readline(self):
        line, self._line = self._line, b""
        return line


class _FragmentedSerial:
    port = "/dev/fragmented-sensor"
    baudrate = 9600
    is_open = True

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.writes = []

    @property
    def in_waiting(self):
        return len(self._chunks[0]) if self._chunks else 0

    def write(self, payload):
        self.writes.append(payload)

    def read(self, _size):
        return self._chunks.pop(0) if self._chunks else b""


class _TrackedSerial(_FragmentedSerial):
    def __init__(self, port, chunks, events):
        super().__init__(chunks)
        self.port = port
        self.events = events

    def write(self, payload):
        self.events.append(("write", self.port))
        super().write(payload)

    def read(self, size):
        self.events.append(("read", self.port))
        return super().read(size)


class _RenderCounter:
    def __init__(self):
        self.calls = 0

    def render(self):
        self.calls += 1


def test_serial_defaults_are_used_by_sensor_api():
    api = ArduinoCommander(connect_immediately=False)

    assert api.baud_rate == DEFAULT_SENSOR_BAUD_RATE
    assert api.timeout == DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC


def test_disconnect_sensor_scene_releases_resources_and_resets_runtime_state():
    sensor = object.__new__(MySensor)
    events = []
    update_enabled = []
    scene_ready = []
    sensor._is_shutting_down = False
    sensor._sensor_calibration_in_progress = False
    sensor.is_connected = True
    sensor.current_model_name = "2d"
    sensor.sensor_transport = "goodix_usb"
    sensor._sensor_profiles_by_port = {"/dev/fake": {"n_row": 2, "n_col": 2}}
    sensor._latest_sensor_payloads = {"/dev/fake": [1, 2, 3, 4]}
    sensor._last_sensor_api_payloads = {"/dev/fake": [1, 2, 3, 4]}
    sensor._primary_sensor_port = "/dev/fake"
    sensor._calibrated_sensor_ports = {"/dev/fake"}
    sensor.cal_data = [1, 2, 3, 4]
    sensor._sensor_update_hz = 58.0
    sensor._sensor_update_tick_count = 10
    sensor._visualization_hz = 55.0
    sensor._visualization_tick_count = 10
    sensor.plotter = _RenderCounter()
    sensor._set_sensor_stream_error = lambda message: setattr(
        sensor, "_last_sensor_stream_error", str(message)
    )
    sensor._set_sensor_update_button_enabled = update_enabled.append
    sensor._set_sensor_scene_ready = scene_ready.append
    sensor._close_serial_ports = lambda: events.append("serial") or True
    sensor._close_goodix_client = lambda: events.append("goodix")
    sensor._close_standalone_sensor_api = lambda: events.append("api")
    sensor._clear_scene_actors = lambda: events.append("actors")
    sensor.clearParameters = lambda: events.append("parameters")
    sensor.reset_ai_helpers = lambda _reason: events.append("helpers")
    sensor._reset_scene_build_state = lambda: events.append("ui")

    disconnected = MySensor.disconnect_sensor_scene(sensor)

    assert disconnected
    assert events == [
        "serial",
        "goodix",
        "api",
        "actors",
        "parameters",
        "helpers",
        "ui",
    ]
    assert update_enabled == [False, True]
    assert scene_ready == [False]
    assert not sensor.is_connected
    assert sensor.current_model_name is None
    assert sensor.sensor_transport == "serial"
    assert sensor._sensor_profiles_by_port == {}
    assert sensor._last_sensor_api_payloads == {}
    assert sensor._sensor_update_hz == 0.0
    assert sensor.plotter.calls == 1
    assert sensor._last_sensor_stream_error == ""


def test_disconnect_sensor_scene_waits_for_active_calibration():
    sensor = object.__new__(MySensor)
    sensor._is_shutting_down = False
    sensor._sensor_calibration_in_progress = True
    sensor._set_sensor_stream_error = lambda message: setattr(
        sensor, "_last_sensor_stream_error", str(message)
    )

    assert not MySensor.disconnect_sensor_scene(sensor)
    assert "calibration to finish" in sensor._last_sensor_stream_error


def test_parse_serial_ints_ignores_non_numeric_debug_tokens():
    assert parse_serial_ints("debug 10 -2 invalid 35") == [10, -2, 35]


def test_serial_line_reader_aborts_quiet_partial_line_early():
    serial_port = _FragmentedSerial([b"55555 55555 1 2 3"])
    started = time.perf_counter()

    line = read_complete_serial_line(
        serial_port,
        timeout=0.5,
        idle_sleep_sec=0.001,
        inter_byte_timeout=0.02,
    )

    assert line == ""
    assert (time.perf_counter() - started) < 0.2


def test_sensor_api_extracts_only_complete_framed_payloads():
    values = [
        7,
        55555,
        55555,
        1,
        2,
        44444,
        44444,
        99,
        55555,
        55555,
        3,
        4,
        44444,
        44444,
    ]

    assert ArduinoCommander._extract_sensor_frame(values) == [3, 4]
    assert ArduinoCommander._extract_sensor_frame([17, 16]) is None
    assert ArduinoCommander._extract_sensor_frame(
        [55555, 55555, 1, 2]
    ) is None


def test_take_complete_sensor_payload_ignores_short_remnants():
    payload, remaining = take_complete_sensor_payload(
        [1, 2, 3, 4, 5, 6, 7],
        expected_payload_values=80,
    )
    assert payload is None
    assert remaining == [1, 2, 3, 4, 5, 6, 7]

    payload, remaining = take_complete_sensor_payload(
        [1, 2, 3, 4, 5, 6, 7, 55555, 55555, 10, 20, 30, 44444, 44444],
        expected_payload_values=3,
    )
    assert payload == [10, 20, 30]
    assert remaining == []


def test_take_complete_sensor_payload_resyncs_at_new_header():
    truncated = [55555, 55555, 91, 92, 93]
    complete = [
        55555,
        55555,
        *range(10),
        44444,
        44444,
    ]

    payload, remaining = take_complete_sensor_payload(
        truncated + complete,
        expected_payload_values=10,
    )

    assert payload == list(range(10))
    assert remaining == []


def test_take_complete_sensor_payload_unframed_only_on_fresh_line():
    headerless = [90, 91, 10, 20, 30, 92, 93]
    payload, remaining = take_complete_sensor_payload(
        headerless,
        expected_payload_values=3,
        allow_unframed=True,
    )
    assert payload == [10, 20, 30]
    assert remaining == []

    payload, remaining = take_complete_sensor_payload(
        [1, 2, 3, 4] + headerless,
        expected_payload_values=3,
        allow_unframed=False,
    )
    assert payload is None
    assert remaining == [1, 2, 3, 4] + headerless


def test_sensor_worker_requests_and_unframes_one_raw_payload():
    serial_port = _FakeSerial(b"55555 55555 10 20 30 44444 44444\n")
    worker = SensorReadWorker([serial_port], expected_payload_values=3)
    worker._running = True

    assert worker._read_raw_from_port(serial_port) == [10, 20, 30]
    assert serial_port.writes == [SENSOR_READ_RAW_COMMAND]
    assert serial_port.input_resets == 0


def test_sensor_worker_accepts_legacy_unframed_line_of_expected_length():
    serial_port = _FakeSerial(b"90 91 10 20 30 92 93\n")
    worker = SensorReadWorker([serial_port], expected_payload_values=3)
    worker._running = True

    assert worker._read_raw_from_port(serial_port) == [10, 20, 30]


def test_sensor_worker_skips_short_line_then_reads_complete_frame():
    frame = "55555 55555 " + " ".join(str(n) for n in range(10)) + " 44444 44444"
    serial_port = _FragmentedSerial(
        [f"1 2 3 4 5 6 7\n{frame}\n".encode("ascii")]
    )
    worker = SensorReadWorker(
        [serial_port],
        expected_payload_values=10,
    )
    worker._running = True

    assert worker._read_raw_from_port(serial_port) == list(range(10))


def test_sensor_worker_resyncs_after_unterminated_frame():
    frame = "55555 55555 " + " ".join(str(n) for n in range(10)) + " 44444 44444"
    serial_port = _FragmentedSerial(
        [f"55555 55555 91 92 93\n{frame}\n".encode("ascii")]
    )
    worker = SensorReadWorker(
        [serial_port],
        expected_payload_values=10,
    )
    worker._running = True

    assert worker._read_raw_from_port(serial_port) == list(range(10))


def test_sensor_worker_does_not_accept_a_short_integer_line():
    serial_port = _FakeSerial(b"1 2 3 4 5 6 7\n")
    worker = SensorReadWorker(
        [serial_port],
        expected_payload_values=10,
        response_timeout=0.05,
    )
    worker._running = True

    assert worker._read_raw_from_port(serial_port) is None


def test_sensor_worker_retries_immediately_after_wrong_length_frame(monkeypatch):
    calls = []

    def read_wrong_frame(*_args, **_kwargs):
        calls.append(True)
        return "55555 55555 1 2 44444 44444"

    monkeypatch.setattr(
        sensor_serial_module,
        "read_complete_serial_line",
        read_wrong_frame,
    )
    monkeypatch.setattr(
        sensor_serial_module,
        "wait_for_serial_input",
        lambda *_args, **_kwargs: True,
    )
    worker = SensorReadWorker(
        [_FakeSerial(b"")],
        expected_payload_values=10,
    )
    worker._running = True

    assert worker._read_raw_response_from_port(worker.serial_ports[0]) is None
    assert len(calls) == 1


def test_sensor_worker_retries_after_unterminated_frame_goes_quiet(monkeypatch):
    calls = []
    input_waits = iter((True, False))

    def read_unterminated_frame(*_args, **_kwargs):
        calls.append(True)
        return "55555 55555 1 2 3"

    monkeypatch.setattr(
        sensor_serial_module,
        "read_complete_serial_line",
        read_unterminated_frame,
    )
    monkeypatch.setattr(
        sensor_serial_module,
        "wait_for_serial_input",
        lambda *_args, **_kwargs: next(input_waits),
    )
    worker = SensorReadWorker(
        [_FakeSerial(b"")],
        expected_payload_values=10,
    )
    worker._running = True

    assert worker._read_raw_response_from_port(worker.serial_ports[0]) is None
    assert len(calls) == 1


def test_sensor_worker_does_not_wait_full_frame_timeout_without_first_byte(
    monkeypatch,
):
    read_calls = []
    monkeypatch.setattr(
        sensor_serial_module,
        "wait_for_serial_input",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        sensor_serial_module,
        "read_complete_serial_line",
        lambda *_args, **_kwargs: read_calls.append(True),
    )
    worker = SensorReadWorker(
        [_FakeSerial(b"")],
        expected_payload_values=272,
    )
    worker._running = True

    assert worker._read_raw_response_from_port(worker.serial_ports[0]) is None
    assert read_calls == []


def test_large_sensor_timeout_accounts_for_full_ascii_frame():
    timeout = sensor_response_timeout_for_values(255, baud_rate=9600)

    assert timeout > 2.0


def test_sensor_worker_accumulates_fragmented_15_by_16_packet():
    payload = list(range(255))
    framed = "55555 55555 " + " ".join(map(str, payload)) + " 44444 44444\n"
    encoded = framed.encode("ascii")
    serial_port = _FragmentedSerial(
        [encoded[index:index + 73] for index in range(0, len(encoded), 73)]
    )
    worker = SensorReadWorker(
        [serial_port],
        expected_payload_values=255,
    )
    worker._running = True

    assert worker._read_raw_from_port(serial_port) == payload
    assert serial_port.writes == [SENSOR_READ_RAW_COMMAND]


def test_sensor_api_skips_debug_line_before_fragmented_sensor_packet():
    payload = list(range(240))
    packet = "55555 55555 " + " ".join(map(str, payload)) + " 44444 44444\n"
    encoded = ("boot debug line\n" + packet).encode("ascii")
    serial_port = _FragmentedSerial(
        [encoded[index:index + 61] for index in range(0, len(encoded), 61)]
    )
    api = ArduinoCommander(connect_immediately=False)
    api.ser = serial_port
    api.expected_payload_values = 240

    assert api.read_response("readRaw") == payload


def test_multi_port_read_cycle_requests_all_ports_before_waiting_for_data():
    events = []
    packet = b"55555 55555 10 20 44444 44444\n"
    first = _TrackedSerial("/dev/ttyACM0", [packet], events)
    second = _TrackedSerial("/dev/ttyACM1", [packet], events)
    worker = SensorReadWorker([first, second], expected_payload_values=2)
    worker._running = True
    received = []
    worker.raw_payload_ready.connect(
        lambda _generation, port, values: received.append((port, values))
    )

    assert worker._read_cycle()
    assert events[:2] == [
        ("write", "/dev/ttyACM0"),
        ("write", "/dev/ttyACM1"),
    ]
    assert received == [
        ("/dev/ttyACM0", [10, 20]),
        ("/dev/ttyACM1", [10, 20]),
    ]


def test_multi_port_live_frames_use_independent_data_buffers():
    first_data = SensorDataBuffer(1, 2, window_size=1)
    second_data = SensorDataBuffer(1, 2, window_size=1)
    first_data.getCal(np.ones((1, 2)))
    second_data.getCal(np.ones((1, 2)))

    sensor = MySensor.__new__(MySensor)
    sensor.is_connected = True
    sensor.n_row = 1
    sensor.n_col = 2
    sensor.cell_zero_mask = np.zeros((1, 2), dtype=bool)
    sensor._primary_sensor_port = "/dev/ttyACM0"
    sensor._sensor_data_by_port = {
        "/dev/ttyACM0": first_data,
        "/dev/ttyACM1": second_data,
    }
    sensor._calibrated_sensor_ports = set(sensor._sensor_data_by_port)
    sensor._latest_sensor_payloads = {
        "/dev/ttyACM0": [11, 12],
        "/dev/ttyACM1": [21, 22],
    }
    sensor._multi_port_sensor_views = {}
    sensor.main_visualization_enabled = False
    sensor._extract_sensor_values = (
        lambda values, _rows, _columns, _port: list(values)
    )
    sensor._record_sensor_update_tick = lambda: None

    sensor.update_animation()

    np.testing.assert_allclose(first_data.rawData, [[11, 12]])
    np.testing.assert_allclose(second_data.rawData, [[21, 22]])


def test_multi_port_live_frames_support_different_shapes_and_extra_columns():
    first_data = SensorDataBuffer(1, 2, window_size=1)
    second_data = SensorDataBuffer(2, 1, window_size=1)
    first_data.getCal(np.ones((1, 2)))
    second_data.getCal(np.ones((2, 1)))

    sensor = MySensor.__new__(MySensor)
    sensor.parent = type("_Parent", (), {})()
    sensor.is_connected = True
    sensor.n_row = 1
    sensor.n_col = 2
    sensor.cell_zero_mask = np.zeros((1, 2), dtype=bool)
    sensor._primary_sensor_port = "/dev/ttyACM0"
    sensor._sensor_profiles_by_port = {
        "/dev/ttyACM0": {
            "n_row": 1,
            "n_col": 2,
            "has_extra_column": False,
        },
        "/dev/ttyACM1": {
            "n_row": 2,
            "n_col": 1,
            "has_extra_column": True,
        },
    }
    sensor._sensor_data_by_port = {
        "/dev/ttyACM0": first_data,
        "/dev/ttyACM1": second_data,
    }
    sensor._calibrated_sensor_ports = set(sensor._sensor_data_by_port)
    sensor._latest_sensor_payloads = {
        "/dev/ttyACM0": [11, 12],
        # 2x(1+1): the final two values are the extra packet column.
        "/dev/ttyACM1": [21, 22, 91, 92],
    }
    sensor._multi_port_sensor_views = {}
    sensor.main_visualization_enabled = False
    sensor._record_sensor_update_tick = lambda: None

    sensor.update_animation()

    np.testing.assert_allclose(first_data.rawData, [[11, 12]])
    np.testing.assert_allclose(second_data.rawData, [[21], [22]])


def test_multi_port_replica_uses_its_own_grid_geometry(monkeypatch):
    secondary_model = SensorModelFactory(
        n_row=2,
        n_col=4,
        offset_scale=0.0005,
        window_size=1,
    ).build()
    owner = MySensor.__new__(MySensor)
    owner.cell_zero_mask = np.zeros((3, 3), dtype=bool)
    saved_mask = np.zeros((2, 4), dtype=bool)
    saved_mask[0, 3] = True
    owner.get_saved_cell_zero_mask = (
        lambda model_name, n_row, n_col: saved_mask.copy()
    )
    owner._safe_actor_suffix = lambda _port: "ttyACM1"
    owner.sensor_visual_offset_scale = 0.0005
    monkeypatch.setattr(
        _MultiPortSensorReplica,
        "_ensure_main_sensor_visualization_actors",
        lambda _self: None,
    )
    monkeypatch.setattr(
        _MultiPortSensorReplica,
        "_refresh_sensor_visualization_mode_actors",
        lambda _self: None,
    )

    replica = _MultiPortSensorReplica.from_owner(
        owner,
        data_obj=secondary_model._data,
        port_name="/dev/ttyACM1",
        offset=np.array([1.0, 0.0, 0.0]),
        model=secondary_model,
    )

    assert (replica.n_row, replica.n_col, replica.n_node) == (2, 4, 8)
    assert replica._data.rawData.shape == (2, 4)
    assert replica.points.shape == (8, 3)
    assert replica.cell_zero_mask.shape == (2, 4)
    np.testing.assert_array_equal(replica.cell_zero_mask, saved_mask)
    np.testing.assert_allclose(
        replica.points[:, 0],
        secondary_model.points[:, 0] + 1.0,
    )


def test_secondary_sensor_raw_data_uses_its_replica_zero_mask():
    sensor = MySensor.__new__(MySensor)
    sensor._data = SensorDataBuffer(1, 1, window_size=1)
    sensor.cell_zero_mask = np.zeros((1, 1), dtype=bool)

    secondary_data = SensorDataBuffer(2, 2, window_size=1)
    secondary_data.getCal(np.array([[10.0, 20.0], [30.0, 40.0]]))
    secondary_mask = np.array([[False, True], [False, False]])
    sensor._multi_port_sensor_views = {
        "/dev/ttyACM1": SimpleNamespace(
            _data=secondary_data,
            cell_zero_mask=secondary_mask,
        )
    }

    raw = np.array([[11.0, 99.0], [31.0, 41.0]])
    sensor._apply_cell_zero_mask_to_raw(raw, secondary_data)

    assert raw[0, 1] == 20.0
    assert raw[0, 0] == 11.0


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
    np.testing.assert_allclose(
        buffer.diffPerDataAve, np.flipud(buffer.diffPerData)
    )
    assert buffer.frame_sequence == 1


def test_sensor_data_buffer_resize_preserves_current_frame_as_window_seed():
    buffer = SensorDataBuffer(2, 1, window_size=1)
    buffer.getRaw(np.array([[3.0], [7.0]]))
    buffer.setWindowSize(4)

    assert buffer.rawDataWin.shape == (4, 2, 1)
    np.testing.assert_allclose(buffer.rawDataWin[0], buffer.rawData)
    np.testing.assert_allclose(buffer.rawDataAve, np.flipud(buffer.rawData))


def test_payload_bridge_emits_every_processed_native_frame():
    class _Sensor:
        def __init__(self):
            self._data = SensorDataBuffer(1, 2, window_size=1)
            self._data.getCal(np.array([[10.0, 20.0]]))

        def _on_sensor_reader_payload(self, _generation, _port, values):
            self.values = list(values)

        def update_animation(self):
            self._data.getRaw(np.asarray([self.values], dtype=float))
            self._data.calDiff()
            self._data.calDiffPer()
            self._data.getWin(1)

    sensor = _Sensor()
    bridge = SensorPayloadBridge(sensor)
    received = []
    received_by_port = []
    bridge.frame_processed.connect(
        lambda timestamp, sequence, raw, calibration: received.append(
            (timestamp, sequence, raw, calibration)
        )
    )
    bridge.port_frame_processed.connect(
        lambda port, timestamp, sequence, raw, calibration: (
            received_by_port.append(
                (port, timestamp, sequence, raw, calibration)
            )
        )
    )

    bridge.deliver(1, "/dev/fake", [12.0, 17.0])

    assert len(received) == 1
    timestamp, sequence, raw, calibration = received[0]
    assert timestamp > 0.0
    assert sequence == 1
    np.testing.assert_allclose(raw, [[12.0, 17.0]])
    np.testing.assert_allclose(calibration, [[10.0, 20.0]])
    assert len(received_by_port) == 1
    assert received_by_port[0][0] == "/dev/fake"
    assert received_by_port[0][2] == 1

from types import SimpleNamespace

import numpy as np

from phd.dependence.func_sensor import (
    MySensor,
    format_live_sensor_source_caption,
)
from phd.dependence.sensor_heatmap import HEATMAP_RESPONSE_LINEAR_RELATIVE
from phd.dependence.sensor_signal_window import SensorSignalWindow


def test_format_live_sensor_source_caption_includes_port_model_and_grid():
    assert format_live_sensor_source_caption(
        "ttyACM0 - Left Hand",
        model_label="2D",
        n_row=10,
        n_col=10,
        is_primary=True,
    ) == "ttyACM0 - Left Hand · 2D · 10×10 · primary"
    assert format_live_sensor_source_caption(
        "ttyACM1",
        model_label="2D",
        n_row=8,
        n_col=10,
    ) == "ttyACM1 · 2D · 8×10"


def _bind_sensor_source_methods(sensor):
    sensor._sensor_port_key = MySensor._sensor_port_key
    sensor._sensor_data_for_port = MySensor._sensor_data_for_port.__get__(sensor)
    sensor._sensor_profile_for_port = MySensor._sensor_profile_for_port.__get__(
        sensor
    )
    sensor.is_primary_sensor_port = MySensor.is_primary_sensor_port.__get__(
        sensor
    )
    sensor._sensor_source_list_label = (
        MySensor._sensor_source_list_label.__get__(sensor)
    )
    sensor.describe_live_sensor_source = (
        MySensor.describe_live_sensor_source.__get__(sensor)
    )
    sensor.list_live_sensor_sources = (
        MySensor.list_live_sensor_sources.__get__(sensor)
    )
    sensor._current_heatmap_sensor_matrix = (
        MySensor._current_heatmap_sensor_matrix.__get__(sensor)
    )
    sensor.get_heatmap_sensor_matrix = (
        MySensor.get_heatmap_sensor_matrix.__get__(sensor)
    )
    return sensor


def test_list_live_sensor_sources_puts_primary_first_and_labels_grid():
    primary = SimpleNamespace(n_row=10, n_col=10)
    secondary = SimpleNamespace(n_row=8, n_col=10)
    sensor = SimpleNamespace(
        SENSOR_MODEL_LABELS=MySensor.SENSOR_MODEL_LABELS,
        _sensor_data_by_port={
            "/dev/ttyACM0": primary,
            "/dev/ttyACM1": secondary,
        },
        _sensor_profiles_by_port={},
        _primary_sensor_port="/dev/ttyACM0",
        _data=primary,
        current_model_name="2d",
        parent=SimpleNamespace(serial_channel=None),
        n_row=10,
        n_col=10,
        _raw_packet_has_extra_column=lambda: True,
    )
    _bind_sensor_source_methods(sensor)

    sources = sensor.list_live_sensor_sources()

    assert [item["key"] for item in sources] == [
        "/dev/ttyACM0",
        "/dev/ttyACM1",
    ]
    assert sources[0]["is_primary"] is True
    assert sources[0]["caption"] == "ttyACM0 · 2D · 10×10 · primary"
    assert sources[1]["is_primary"] is False
    assert sources[1]["n_row"] == 8
    assert sources[1]["n_col"] == 10
    assert sources[1]["caption"] == "ttyACM1 · 2D · 8×10"


def test_heatmap_matrix_can_read_a_named_secondary_port():
    primary = SimpleNamespace(
        rawDataAve=np.array([[1.0, 2.0]]),
        diffPerDataAve=np.array([[10.0, 20.0]]),
        rawData=None,
        diffData=None,
    )
    secondary = SimpleNamespace(
        rawDataAve=np.array([[3.0, 4.0, 5.0]]),
        diffPerDataAve=np.array([[30.0, 40.0, 50.0]]),
        rawData=None,
        diffData=None,
    )
    sensor = SimpleNamespace(
        _data=primary,
        _sensor_data_by_port={
            "/dev/ttyACM0": primary,
            "/dev/ttyACM1": secondary,
        },
        _primary_sensor_port="/dev/ttyACM0",
        heatmap_response_mode=HEATMAP_RESPONSE_LINEAR_RELATIVE,
        _heatmap_calibration_override=None,
    )
    _bind_sensor_source_methods(sensor)

    np.testing.assert_array_equal(
        sensor.get_heatmap_sensor_matrix(),
        [[10.0, 20.0]],
    )
    np.testing.assert_array_equal(
        sensor.get_heatmap_sensor_matrix(port_name="/dev/ttyACM1"),
        [[30.0, 40.0, 50.0]],
    )
    sensor._heatmap_calibration_override = np.array([[99.0, 99.0]])
    # A primary-only baseline must not leak onto the other port.
    np.testing.assert_array_equal(
        sensor.get_heatmap_sensor_matrix(port_name="/dev/ttyACM1"),
        [[30.0, 40.0, 50.0]],
    )


def test_signal_viewer_shows_and_switches_watched_source():
    sources = [
        {
            "key": "/dev/ttyACM0",
            "port_label": "ttyACM0",
            "model": "2d",
            "model_label": "2D",
            "n_row": 10,
            "n_col": 10,
            "is_primary": True,
            "caption": "ttyACM0 · 2D · 10×10 · primary",
        },
        {
            "key": "/dev/ttyACM1",
            "port_label": "ttyACM1 - Left Hand",
            "model": "2d",
            "model_label": "2D",
            "n_row": 8,
            "n_col": 10,
            "is_primary": False,
            "caption": "ttyACM1 - Left Hand · 2D · 8×10",
        },
    ]
    viewer = SensorSignalWindow.__new__(SensorSignalWindow)
    viewer._using_shared_sensor_data = True
    viewer._watched_source_key = None
    viewer._listed_source_keys = None
    viewer.table_rows = 1
    viewer.table_columns = 1
    viewer.selected_index = 4
    viewer._shared_calibration_overridden = True
    viewer.calibration_data = [1]
    viewer.initial_diffs = [1]
    viewer.threshold_max = {0: 1}
    viewer.source_combo = None
    viewer.source_caption = None
    viewer.info_label = None
    viewer._signal_tracker_window = None
    viewer._shared_sensor_functions_ref = SimpleNamespace(
        list_live_sensor_sources=lambda: sources,
        _data=object(),
        set_selected_sensor_cell=lambda *_args, **_kwargs: True,
        is_primary_sensor_port=lambda key: key == "/dev/ttyACM0",
        _sensor_data_for_port=lambda _key: None,
    )
    viewer._resolve_sensor_functions = (
        lambda: viewer._shared_sensor_functions_ref
    )
    viewer._title = ""
    viewer.setWindowTitle = lambda text: setattr(viewer, "_title", text)

    viewer._refresh_watched_sources(prefer_primary=True)

    assert viewer._watched_source_key == "/dev/ttyACM0"
    assert viewer.table_rows == 10
    assert viewer.table_columns == 10
    assert viewer._title == "Sensor Signal — ttyACM0 · 2D · 10×10 · primary"
    assert viewer._idle_info_text().startswith("Watching ttyACM0")

    viewer._watched_source_key = "/dev/ttyACM1"
    viewer._apply_watched_source(sources[1], source_changed=True)

    assert viewer.table_rows == 8
    assert viewer.table_columns == 10
    assert viewer.selected_index is None
    assert viewer._idle_info_text().startswith("Watching ttyACM1 - Left Hand")
    assert viewer._cell_info_text(3) == (
        "ttyACM1 - Left Hand · 2D · 8×10 · cell 3"
    )
    assert viewer._is_watching_primary_source() is False


def test_signal_viewer_ignores_frames_from_other_ports():
    viewer = SensorSignalWindow.__new__(SensorSignalWindow)
    viewer._watched_source_key = "/dev/ttyACM1"
    viewer._using_shared_sensor_data = True
    viewer._shared_sensor_functions_ref = SimpleNamespace(
        _data=object(),
        _sensor_port_key=MySensor._sensor_port_key,
    )
    viewer._resolve_sensor_functions = (
        lambda: viewer._shared_sensor_functions_ref
    )

    assert viewer._port_matches_watched_source("/dev/ttyACM1")
    assert not viewer._port_matches_watched_source("/dev/ttyACM0")

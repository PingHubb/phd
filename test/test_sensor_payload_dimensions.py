from phd.dependence.func_sensor import MySensor


class _CheckBox:
    def __init__(self, checked):
        self._checked = bool(checked)

    def isChecked(self):
        return self._checked


class _Parent:
    def __init__(self, has_extra_column):
        self.sensor_extra_column_checkbox = _CheckBox(has_extra_column)


def _sensor(has_extra_column):
    sensor = MySensor.__new__(MySensor)
    sensor.parent = _Parent(has_extra_column)
    sensor._last_sensor_stream_error = ""
    return sensor


def test_exact_packet_keeps_all_grid_values():
    sensor = _sensor(has_extra_column=False)
    payload = list(range(49))

    values = sensor._extract_sensor_values(payload, 7, 7, "/dev/test")

    assert values == payload
    assert sensor.get_last_sensor_stream_error() == ""


def test_plus_one_packet_removes_extra_column():
    sensor = _sensor(has_extra_column=True)
    payload = list(range(56))

    values = sensor._extract_sensor_values(payload, 7, 7, "/dev/test")

    assert values == list(range(49))
    assert sensor.get_last_sensor_stream_error() == ""


def test_wrong_packet_format_reports_selected_expected_size():
    sensor = _sensor(has_extra_column=False)

    values = sensor._extract_sensor_values(
        list(range(56)), 7, 7, "/dev/test"
    )

    assert values is None
    assert "expected 49 for 7x7 packet format" in sensor.get_last_sensor_stream_error()


def test_legacy_10x9_payload_only_trims_when_second_extra_block_exists():
    plus_one_sensor = _sensor(has_extra_column=True)
    exact_sensor = _sensor(has_extra_column=False)

    plus_one_values = plus_one_sensor._extract_sensor_values(
        list(range(110)), 10, 9, "/dev/test"
    )
    exact_values = exact_sensor._extract_sensor_values(
        list(range(90)), 10, 9, "/dev/test"
    )

    assert plus_one_values == list(range(90))
    assert exact_values == list(range(90))

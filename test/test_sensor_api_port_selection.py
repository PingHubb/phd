from phd.dependence.sensor_api import ArduinoCommander
from phd.ui.ui_ping import UI
from phd.ui.ui_ping_robot_sensor_controls import RobotSensorControlsMixin


class _Item:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _PortList:
    def __init__(self, selected, current=None):
        self._selected = list(selected)
        self._current = current

    def selectedItems(self):
        return list(self._selected)

    def currentItem(self):
        return self._current


class _Log:
    def __init__(self):
        self.messages = []

    def append(self, message):
        self.messages.append(str(message))


class _Api:
    def read_raw(self):
        return [10, 20, 30]


class _DirectSensor:
    @staticmethod
    def _sensor_reader_is_running():
        return False


class _LiveSensor:
    @staticmethod
    def _sensor_reader_is_running():
        return True

    @staticmethod
    def get_last_sensor_api_payload(port_path):
        assert port_path == "/dev/ttyACM1"
        return [40, 50, 60]


class _Harness(RobotSensorControlsMixin):
    def __init__(self, sensor):
        self.log_display = _Log()
        self.sensor_functions = sensor
        self.sensor_api = _Api()
        self.opened_port = None

    def ensure_sensor_api(self, connect_immediately=True, serial_port=None):
        self.opened_port = serial_port
        return True


class _FakeSerial:
    def __init__(self):
        self.is_open = True
        self.closed = False

    def close(self):
        self.closed = True
        self.is_open = False


class _ValueWidget:
    def __init__(self, value):
        self.value = value

    def setValue(self, value):
        self.value = value


class _CheckWidget:
    def __init__(self, checked):
        self.checked = checked

    def setChecked(self, checked):
        self.checked = bool(checked)


class _PortPresetHarness:
    _default_sensor_grid_shape_for_port = staticmethod(
        UI._default_sensor_grid_shape_for_port
    )
    _on_sensor_port_selection_changed = UI._on_sensor_port_selection_changed

    def __init__(self):
        self.grid_rows_spin = _ValueWidget(10)
        self.grid_cols_spin = _ValueWidget(10)
        self.sensor_extra_column_checkbox = _CheckWidget(True)


def test_ttyacm1_uses_a_7_by_7_visualization_grid_default():
    assert UI._default_sensor_grid_shape_for_port("ttyACM1") == (7, 7)
    assert UI._default_sensor_grid_shape_for_port("/dev/ttyACM1") == (7, 7)
    assert UI._default_sensor_grid_shape_for_port("ttyACM0") is None


def test_ttyacm1_disables_the_extra_raw_packet_column_by_default():
    ui = _PortPresetHarness()

    ui._on_sensor_port_selection_changed(_Item("ttyACM1"))

    assert ui.grid_rows_spin.value == 7
    assert ui.grid_cols_spin.value == 7
    assert ui.sensor_extra_column_checkbox.checked is False


def test_direct_raw_uses_current_selected_send_operation_port():
    tty0 = _Item("ttyACM0")
    tty1 = _Item("ttyACM1")
    ui = _Harness(_DirectSensor())
    ui.serial_channel = _PortList([tty0, tty1], current=tty1)

    ui._on_sensor_api_read_raw()

    assert ui.opened_port == "/dev/ttyACM1"
    assert ui.log_display.messages == [
        "API raw data (/dev/ttyACM1): [10, 20, 30]"
    ]


def test_live_raw_reuses_latest_payload_without_opening_second_reader():
    tty1 = _Item("ttyACM1")
    ui = _Harness(_LiveSensor())
    ui.serial_channel = _PortList([tty1], current=tty1)

    ui._on_sensor_api_read_raw()

    assert ui.opened_port is None
    assert ui.log_display.messages == [
        "API raw data (/dev/ttyACM1, live): [40, 50, 60]"
    ]


def test_arduino_commander_can_switch_ports_without_hard_coded_device():
    api = ArduinoCommander(serial_port="/dev/ttyACM0", connect_immediately=False)
    old_serial = _FakeSerial()
    api.ser = old_serial
    connected_ports = []

    def connect():
        connected_ports.append(api.serial_port)
        return True

    api._connect = connect

    assert api.set_serial_port("/dev/ttyACM1", reconnect=True)
    assert old_serial.closed
    assert api.serial_port == "/dev/ttyACM1"
    assert connected_ports == ["/dev/ttyACM1"]

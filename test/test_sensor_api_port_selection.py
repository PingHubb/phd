from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidget, QListWidgetItem

from phd.dependence import func_sensor
from phd.dependence.func_sensor import MySensor
from phd.dependence.sensor_api import ArduinoCommander
from phd.dependence.goodix_usb_sensor import (
    GOODIX_USB_SOURCE_ID,
    GOODIX_USB_SOURCE_LABEL,
)
from phd.ui import ui_ping
from phd.ui.ui_ping import UI
from phd.ui.ui_ping_robot_sensor_controls import RobotSensorControlsMixin


class _Item:
    def __init__(self, text):
        self._text = text
        self.selected = False

    def text(self):
        return self._text

    def setSelected(self, selected):
        self.selected = bool(selected)


class _PortList:
    def __init__(self, selected, current=None):
        self._selected = list(selected)
        self._current = current

    def selectedItems(self):
        return list(self._selected)

    def currentItem(self):
        return self._current


class _SourceList:
    def __init__(self):
        self.items = []
        self.current_row = -1

    def clear(self):
        self.items.clear()

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def item(self, index):
        return self.items[index]

    def setCurrentRow(self, index):
        self.current_row = int(index)


class _Port:
    def __init__(self, name):
        self.name = name


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
        self._value = value

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value

    def blockSignals(self, _blocked):
        return None


class _CheckWidget:
    def __init__(self, checked):
        self.checked = checked

    def setChecked(self, checked):
        self.checked = bool(checked)

    def isChecked(self):
        return self.checked

    def blockSignals(self, _blocked):
        return None


class _PortPresetHarness:
    _default_sensor_grid_shape_for_port = staticmethod(
        UI._default_sensor_grid_shape_for_port
    )
    _sensor_port_profile_key = staticmethod(UI._sensor_port_profile_key)
    _default_sensor_port_profile = UI._default_sensor_port_profile
    _save_sensor_port_profile_from_controls = (
        UI._save_sensor_port_profile_from_controls
    )
    get_sensor_port_profile = UI.get_sensor_port_profile
    _load_sensor_port_profile_into_controls = (
        UI._load_sensor_port_profile_into_controls
    )
    _on_sensor_port_selection_changed = UI._on_sensor_port_selection_changed

    def __init__(self):
        self.grid_rows_spin = _ValueWidget(10)
        self.grid_cols_spin = _ValueWidget(10)
        self.sensor_extra_column_checkbox = _CheckWidget(True)
        self._sensor_port_profiles = {}
        self._sensor_port_profile_updating = False


class _ModeCombo:
    def __init__(self, mode):
        self.mode = mode

    def currentData(self):
        return self.mode


class _ModePortList:
    def __init__(self, items, current_index=0):
        self.items = list(items)
        self.current_index = current_index
        self.selection_mode = None

    def setSelectionMode(self, mode):
        self.selection_mode = mode

    def currentItem(self):
        if not 0 <= self.current_index < len(self.items):
            return None
        return self.items[self.current_index]

    def selectedItems(self):
        return [item for item in self.items if item.selected]

    def count(self):
        return len(self.items)

    def item(self, index):
        return self.items[index]


class _SourceModeHarness:
    _on_sensor_source_mode_changed = UI._on_sensor_source_mode_changed

    def __init__(self, mode):
        items = [_Item("ttyACM0"), _Item("ttyACM1")]
        items[0].setSelected(True)
        items[1].setSelected(True)
        self.serial_channel = _ModePortList(items, current_index=1)
        self.sensor_source_mode_combo = _ModeCombo(mode)


def test_unrecognized_ttyacm1_uses_legacy_grid_default(monkeypatch):
    monkeypatch.setattr(
        ui_ping,
        "humanoid_sensor_grid_shape_for_device",
        lambda _port: None,
    )
    assert UI._default_sensor_grid_shape_for_port("ttyACM1") == (8, 10)
    assert UI._default_sensor_grid_shape_for_port("/dev/ttyACM1") == (8, 10)


def test_unrecognized_ttyacm0_uses_legacy_grid_and_packet_format(
    monkeypatch,
):
    monkeypatch.setattr(
        ui_ping,
        "humanoid_sensor_grid_shape_for_device",
        lambda _port: None,
    )
    assert UI._default_sensor_grid_shape_for_port("ttyACM0") == (10, 10)
    assert UI._default_sensor_grid_shape_for_port("/dev/ttyACM0") == (
        10,
        10,
    )

    ui = _PortPresetHarness()
    ui.sensor_extra_column_checkbox.setChecked(False)
    ui._on_sensor_port_selection_changed(_Item("ttyACM0"))

    assert ui.grid_rows_spin.value() == 10
    assert ui.grid_cols_spin.value() == 10
    assert ui.sensor_extra_column_checkbox.checked is True


def test_goodix_usb_uses_a_10_by_8_exact_grid_default():
    assert UI._default_sensor_grid_shape_for_port(GOODIX_USB_SOURCE_LABEL) == (
        10,
        8,
    )

    ui = _PortPresetHarness()
    ui._on_sensor_port_selection_changed(_Item(GOODIX_USB_SOURCE_LABEL))

    assert ui.grid_rows_spin.value() == 10
    assert ui.grid_cols_spin.value() == 8
    assert ui.sensor_extra_column_checkbox.checked is False


def test_goodix_is_after_acm_and_system_ttys_ports_are_hidden(monkeypatch):
    sources = _SourceList()
    sensor = object.__new__(MySensor)
    sensor.parent = type("_Parent", (), {"serial_channel": sources})()
    monkeypatch.setattr(
        func_sensor.serial.tools.list_ports,
        "comports",
        lambda: [_Port("ttyS0"), _Port("ttyUSB0"), _Port("ttyACM0"), _Port("ttyS31")],
    )
    monkeypatch.setattr(func_sensor, "goodix_usb_connected", lambda: True)

    sensor.initChannel()

    labels = [item.text() for item in sources.items]
    assert labels == ["ttyACM0", GOODIX_USB_SOURCE_LABEL, "ttyUSB0"]
    assert sensor.com_options == ["ttyACM0", GOODIX_USB_SOURCE_ID, "ttyUSB0"]
    assert sources.current_row == 0


def test_recognized_humanoid_sensor_name_keeps_raw_port_data(monkeypatch):
    sources = _SourceList()
    sensor = object.__new__(MySensor)
    sensor.parent = type("_Parent", (), {"serial_channel": sources})()
    monkeypatch.setattr(
        func_sensor.serial.tools.list_ports,
        "comports",
        lambda: [_Port("ttyACM1")],
    )
    monkeypatch.setattr(func_sensor, "goodix_usb_connected", lambda: False)
    monkeypatch.setattr(
        func_sensor,
        "load_humanoid_device_assignments",
        lambda: {"saved": "assignment"},
    )
    monkeypatch.setattr(
        func_sensor,
        "humanoid_sensor_annotation",
        lambda port, assignments: "Head (9x14)",
    )

    sensor.initChannel()

    item = sources.items[0]
    assert item.text() == "ttyACM1 - Head (9x14)"
    assert item.data(Qt.UserRole) == "ttyACM1"


def test_sensor_operations_use_raw_port_behind_friendly_label():
    item = QListWidgetItem("ttyACM1 - Head (9x14)")
    item.setData(Qt.UserRole, "ttyACM1")
    ui = _Harness(_DirectSensor())
    ui.serial_channel = _PortList([item], current=item)

    ui._on_sensor_api_read_raw()

    assert ui.opened_port == "/dev/ttyACM1"


def test_recognized_head_loads_its_grid_shape(monkeypatch):
    monkeypatch.setattr(
        ui_ping,
        "humanoid_sensor_grid_shape_for_device",
        lambda _port: (9, 14),
    )
    ui = _PortPresetHarness()

    ui._on_sensor_port_selection_changed(_Item("ttyACM1"))

    assert ui.grid_rows_spin.value() == 9
    assert ui.grid_cols_spin.value() == 14
    assert ui.sensor_extra_column_checkbox.checked is False


def test_recognized_end_effector_loads_shape_and_packet_format(
    monkeypatch,
):
    monkeypatch.setattr(
        ui_ping,
        "humanoid_sensor_grid_shape_for_device",
        lambda _port: (10, 10),
    )
    monkeypatch.setattr(
        ui_ping,
        "humanoid_sensor_extra_column_for_device",
        lambda _port: True,
    )
    ui = _PortPresetHarness()
    ui.sensor_extra_column_checkbox.setChecked(False)

    ui._on_sensor_port_selection_changed(_Item("ttyACM7"))

    assert ui.grid_rows_spin.value() == 10
    assert ui.grid_cols_spin.value() == 10
    assert ui.sensor_extra_column_checkbox.checked is True


def test_multiple_ports_remember_independent_grid_and_packet_profiles(
    monkeypatch,
):
    monkeypatch.setattr(
        ui_ping,
        "humanoid_sensor_grid_shape_for_device",
        lambda _port: None,
    )
    ui = _PortPresetHarness()
    acm0 = _Item("ttyACM0")
    acm1 = _Item("ttyACM1")

    ui._on_sensor_port_selection_changed(acm0)
    ui.grid_rows_spin.setValue(12)
    ui.grid_cols_spin.setValue(9)
    ui.sensor_extra_column_checkbox.setChecked(True)
    ui._on_sensor_port_selection_changed(acm1, acm0)

    assert ui.grid_rows_spin.value() == 8
    assert ui.grid_cols_spin.value() == 10
    assert ui.sensor_extra_column_checkbox.isChecked() is False

    ui.grid_rows_spin.setValue(7)
    ui.grid_cols_spin.setValue(6)
    ui.sensor_extra_column_checkbox.setChecked(False)
    ui._save_sensor_port_profile_from_controls("ttyACM1")
    ui._load_sensor_port_profile_into_controls("ttyACM0")

    assert ui.grid_rows_spin.value() == 12
    assert ui.grid_cols_spin.value() == 9
    assert ui.sensor_extra_column_checkbox.isChecked() is True
    assert ui._sensor_port_profiles["ttyacm1"] == {
        "n_row": 7,
        "n_col": 6,
        "has_extra_column": False,
    }


def test_multiple_sensor_mode_allows_toggle_selection():
    ui = _SourceModeHarness("multiple")

    ui._on_sensor_source_mode_changed()

    assert ui.serial_channel.selection_mode == QListWidget.MultiSelection


def test_returning_to_single_sensor_keeps_only_current_port_selected():
    ui = _SourceModeHarness("single")

    ui._on_sensor_source_mode_changed()

    assert ui.serial_channel.selection_mode == QListWidget.SingleSelection
    assert [item.selected for item in ui.serial_channel.items] == [False, True]


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

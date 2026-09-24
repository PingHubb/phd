"""Regression coverage for operator-facing runtime availability state."""

from types import SimpleNamespace

from phd.ui.ui_initial import MyMainWindow
from phd.ui.ui_ping import UI
from phd.ui.ui_ping_ui_interactions import UiInteractionsMixin


class _Widget:
    def __init__(self, text="", checked=False):
        self.enabled = True
        self.tooltip = ""
        self._text = text
        self._checked = checked

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setToolTip(self, tooltip):
        self.tooltip = str(tooltip)

    def text(self):
        return self._text

    def setText(self, text):
        self._text = str(text)

    def isChecked(self):
        return self._checked

    def toolTip(self):
        return self.tooltip


class _SessionStrip:
    def __init__(self):
        self.values = {}
        self.states = {}

    def set_value(self, key, value, *, tooltip="", state=None):
        self.values[key] = str(value)
        self.states[key] = state


class _StatusIndicator:
    def __init__(self):
        self.visible = False
        self.state = None
        self.text = ""
        self.tooltip = ""

    def set_state(self, state, text):
        self.state = state
        self.text = text

    def setToolTip(self, tooltip):
        self.tooltip = str(tooltip)

    def setVisible(self, visible):
        self.visible = bool(visible)


def test_force_meter_port_identity_uses_stable_chip_symlink(tmp_path):
    tty = tmp_path / "ttyUSB7"
    tty.touch()
    by_id = tmp_path / "by-id"
    by_id.mkdir()
    stable = by_id / "usb-FTDI_FT232R_USB_UART_AG0K5VFJ-if00-port0"
    stable.symlink_to(tty)

    identity = UiInteractionsMixin._stable_serial_port_identity(
        str(tty),
        str(by_id),
    )

    assert identity == str(stable)
    assert "ttyUSB7" not in identity


def test_sensor_update_requires_a_ready_scene_and_idle_backend():
    window = type("_Window", (), {})()
    window._sensor_scene_ready = False
    window._sensor_update_backend_enabled = True
    window.action_update_sensor = _Widget()
    window.toolbar_update_sensor_button = _Widget()
    window.action_disconnect_sensor = _Widget()
    window.toolbar_disconnect_sensor_button = _Widget()
    window._refresh_sensor_update_control = (
        MyMainWindow._refresh_sensor_update_control.__get__(window)
    )

    MyMainWindow.set_sensor_scene_ready(window, False)
    assert not window.action_update_sensor.enabled
    assert "Connect a sensor scene" in window.action_update_sensor.tooltip

    MyMainWindow.set_sensor_scene_ready(window, True)
    assert window.action_update_sensor.enabled
    assert window.action_disconnect_sensor.enabled

    MyMainWindow.set_sensor_update_enabled(window, False)
    assert not window.action_update_sensor.enabled
    assert not window.action_disconnect_sensor.enabled
    assert "current sensor operation" in window.action_disconnect_sensor.tooltip

    MyMainWindow.set_sensor_scene_ready(window, False)
    MyMainWindow.set_sensor_update_enabled(window, True)
    assert not window.toolbar_update_sensor_button.enabled
    assert not window.toolbar_disconnect_sensor_button.enabled
    assert "No built sensor scene" in window.action_disconnect_sensor.tooltip


def test_sensor_summary_names_offline_and_streaming_states():
    item = _Widget("ttyACM1 - Highlighted Sensor")
    ports = type("_Ports", (), {"currentItem": lambda self: item})()
    rows = type("_Spin", (), {"value": lambda self: 8})()
    cols = type("_Spin", (), {"value": lambda self: 10})()
    live_sensor = SimpleNamespace(
        describe_live_sensor_source=lambda: {
            "port_label": "ttyACM0 - Streaming Sensor",
            "n_row": 10,
            "n_col": 10,
        }
    )
    ui = SimpleNamespace(
        serial_channel=ports,
        grid_rows_spin=rows,
        grid_cols_spin=cols,
        sensor_functions=live_sensor,
    )

    offline, _tooltip = MyMainWindow._selected_sensor_summary(ui, False, 60.0)
    live, _tooltip = MyMainWindow._selected_sensor_summary(ui, True, 59.6)

    assert offline.startswith("Offline · ttyACM1 - Highlighted Sensor · 8x10")
    assert "Hz" not in offline
    assert live.startswith("Streaming · ttyACM0 - Streaming Sensor · 10x10")
    assert "ttyACM1" not in live
    assert live.endswith("60 Hz")


def test_session_summary_colors_active_and_inactive_services():
    strip = _SessionStrip()
    ui = SimpleNamespace(
        session_strip=strip,
        serial_channel=None,
        grid_rows_spin=None,
        grid_cols_spin=None,
        sensor_functions=SimpleNamespace(
            describe_live_sensor_source=lambda: {
                "port_label": "ttyACM0",
                "n_row": 10,
                "n_col": 10,
            }
        ),
        ai_direct_execution_model_status=_Widget("Model: tactile.pt"),
        force_meter_value_label=_Widget("+1.2500 N"),
        force_meter_connect_button=_Widget(checked=False),
    )
    window = SimpleNamespace(
        _selected_sensor_summary=MyMainWindow._selected_sensor_summary
    )

    MyMainWindow._refresh_session_strip(window, ui, "Idle", False, 0.0)
    assert strip.states == {
        "sensor": "idle",
        "control": "idle",
        "model": "idle",
        "force": "idle",
    }

    ui.force_meter_connect_button._checked = True
    MyMainWindow._refresh_session_strip(
        window,
        ui,
        "AI DFM (exec)",
        True,
        60.0,
    )
    assert strip.states == {
        "sensor": "active",
        "control": "active",
        "model": "active",
        "force": "active",
    }

    MyMainWindow._refresh_session_strip(
        window,
        ui,
        "AI DFM (record)",
        True,
        60.0,
    )
    assert strip.states["control"] == "active"
    assert strip.states["model"] == "idle"


def test_running_experiment_keeps_status_and_stop_visible_in_main_window():
    indicator = _StatusIndicator()
    stop_button = _Widget()
    stop_button.visible = False
    stop_button.setVisible = lambda visible: setattr(
        stop_button,
        "visible",
        bool(visible),
    )
    window = SimpleNamespace(
        status_experiment=indicator,
        status_experiment_stop_button=stop_button,
        _sidebar_active_task=SimpleNamespace(
            id="sensor_capture",
            label="Sensor Capture",
        ),
    )

    MyMainWindow._refresh_experiment_status(window)
    assert indicator.visible
    assert indicator.state == "active"
    assert indicator.text == "Experiment: Sensor Capture"
    assert stop_button.visible
    assert stop_button.enabled

    window._sidebar_active_task = None
    MyMainWindow._refresh_experiment_status(window)
    assert not indicator.visible
    assert not stop_button.visible
    assert not stop_button.enabled


def test_hand_workspace_gates_motion_and_tactile_independently():
    command = _Widget()
    tactile = _Widget()
    connection = _Widget()
    ui = type("_Ui", (), {})()
    ui.hand_open_all_button = command
    ui.hand_tactile_live_button = tactile
    ui.hand_connection_label = connection
    ui._hand_angle_send_buttons = []
    ui._set_widgets_enabled = lambda widgets, enabled: [
        widget.setEnabled(enabled) for widget in widgets
    ]

    UI._set_hand_controls_available(ui, False, True)
    assert not command.enabled
    assert tactile.enabled
    assert connection.text() == "Connected · tactile only"

    UI._set_hand_controls_available(ui, True, False)
    assert command.enabled
    assert not tactile.enabled
    assert connection.text() == "Connected · motion only"

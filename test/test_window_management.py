"""Operator window actions reopen, focus, and dismiss auxiliary windows."""

from types import SimpleNamespace

from phd.ui.ui_initial import MyMainWindow


class _Window:
    def __init__(self):
        self.visible = False
        self.closed = False
        self.geometry_saved = False

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def close(self):
        self.closed = True
        self.visible = False

    def remember_geometry(self):
        self.geometry_saved = True


def test_window_log_action_opens_focuses_and_marks_log_read():
    console = _Window()
    calls = []
    ui = SimpleNamespace(
        log_console_window=console,
        _clear_log_unread_count=lambda: calls.append("cleared"),
    )
    window = SimpleNamespace(
        ui_ros=ui,
        _require_ui_ros=lambda _message: True,
        _focus_window=lambda target: calls.append(target),
    )

    MyMainWindow.open_application_log_window(window)

    assert console.visible
    assert calls == [console, "cleared"]


def test_close_auxiliary_windows_hides_all_three_routine_windows():
    sensor = _Window()
    console = _Window()
    log_display = SimpleNamespace(setVisible=lambda visible: None)
    calls = []
    window = SimpleNamespace(
        sensor_window=sensor,
        ui_ros=SimpleNamespace(
            log_console_window=console,
            log_display=log_display,
        ),
        _set_sidebar_visible=lambda visible: calls.append(("experiments", visible)),
    )

    MyMainWindow.close_auxiliary_windows(window)

    assert calls[0] == ("experiments", False)
    assert sensor.closed
    assert console.geometry_saved
    assert not console.visible

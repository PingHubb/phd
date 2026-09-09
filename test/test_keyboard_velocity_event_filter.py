from PyQt5.QtCore import QEvent, Qt

from phd.ui.ui_initial import MyMainWindow


class _KeyEvent:
    def __init__(self, event_type, key, auto_repeat=False):
        self._event_type = event_type
        self._key = key
        self._auto_repeat = bool(auto_repeat)
        self.accepted = False

    def type(self):
        return self._event_type

    def key(self):
        return self._key

    def isAutoRepeat(self):
        return self._auto_repeat

    def accept(self):
        self.accepted = True


class _KeyboardHarness:
    def __init__(self):
        self._keyboard_tool_velocity_enabled = True
        self._keyboard_vel_active_tokens = set()
        self.apply_count = 0

    @staticmethod
    def _keyboard_velocity_should_capture():
        return True

    @staticmethod
    def _keyboard_vel_key_token(key):
        return {
            Qt.Key_W: "w",
            Qt.Key_P: "p",
            Qt.Key_R: "r",
        }.get(key)

    def _keyboard_vel_apply_active_tokens(self):
        self.apply_count += 1


def test_keyboard_velocity_consumes_auto_repeat_before_vtk_shortcuts():
    harness = _KeyboardHarness()
    harness._keyboard_vel_active_tokens.add("p")
    event = _KeyEvent(QEvent.KeyPress, Qt.Key_P, auto_repeat=True)

    consumed = MyMainWindow.eventFilter(harness, object(), event)

    assert consumed is True
    assert harness._keyboard_vel_active_tokens == {"p"}
    assert harness.apply_count == 0


def test_keyboard_velocity_updates_tokens_only_for_real_press_and_release():
    harness = _KeyboardHarness()

    press = _KeyEvent(QEvent.KeyPress, Qt.Key_W)
    assert MyMainWindow.eventFilter(harness, object(), press) is True
    assert harness._keyboard_vel_active_tokens == {"w"}
    assert harness.apply_count == 1

    repeated_release = _KeyEvent(QEvent.KeyRelease, Qt.Key_W, auto_repeat=True)
    assert MyMainWindow.eventFilter(harness, object(), repeated_release) is True
    assert harness._keyboard_vel_active_tokens == {"w"}
    assert harness.apply_count == 1

    release = _KeyEvent(QEvent.KeyRelease, Qt.Key_W)
    assert MyMainWindow.eventFilter(harness, object(), release) is True
    assert harness._keyboard_vel_active_tokens == set()
    assert harness.apply_count == 2


def test_keyboard_velocity_accepts_shortcut_override_for_auto_repeat():
    harness = _KeyboardHarness()
    event = _KeyEvent(QEvent.ShortcutOverride, Qt.Key_R, auto_repeat=True)

    consumed = MyMainWindow.eventFilter(harness, object(), event)

    assert consumed is True
    assert event.accepted is True

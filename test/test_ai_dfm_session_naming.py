from phd.ui.ui_ping_direct_finger_motion import DirectFingerMotionMixin


class _Spin:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _Input:
    def __init__(self, text=""):
        self._text = text
        self.placeholder = ""

    def text(self):
        return self._text

    def setPlaceholderText(self, text):
        self.placeholder = str(text)


def _controls(rows=10, cols=10, custom=""):
    controls = DirectFingerMotionMixin()
    controls.grid_rows_spin = _Spin(rows)
    controls.grid_cols_spin = _Spin(cols)
    controls.ai_dfm_session_input = _Input(custom)
    return controls


def test_ai_dfm_session_name_defaults_to_sensor_size():
    controls = _controls(rows=8, cols=10)

    assert controls._current_ai_dfm_session_tag() == "ai_dfm_8x10_v1"


def test_ai_dfm_custom_session_overrides_automatic_name():
    controls = _controls(custom="push_focus_v2")

    assert controls._current_ai_dfm_session_tag() == "push_focus_v2"


def test_ai_dfm_session_placeholder_tracks_grid_dimensions():
    controls = _controls(rows=10, cols=10)

    controls._update_ai_dfm_session_placeholder()

    assert controls.ai_dfm_session_input.placeholder == (
        "Auto: ai_dfm_10x10_v1"
    )

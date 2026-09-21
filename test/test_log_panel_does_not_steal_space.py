"""The log utility must never open itself or resize the workspace.

It used to be wired so that *any* appended line forced the panel visible and
handed it a fifth of the visualization width. Background chatter (motion-ratio
logging, per-second experiment ticks) therefore kept re-opening a panel the
user had deliberately closed, shrinking the 3D view during robot motion.
"""

from phd.ui.ui_ping_ui_interactions import UiInteractionsMixin


class _LogDisplay:
    def __init__(self, visible=False):
        self._visible = bool(visible)
        self.lines = []

    def isVisible(self):  # noqa: N802 - Qt API shape
        return self._visible

    def setVisible(self, visible):  # noqa: N802 - Qt API shape
        self._visible = bool(visible)

    def append(self, line):
        self.lines.append(line)

    def document(self):
        blocks = []
        for line in self.lines:
            blocks.extend(str(line).split("\n"))
        return _Document(blocks)


class _Block:
    def __init__(self, blocks, index):
        self._blocks = blocks
        self._index = index

    def isValid(self):  # noqa: N802 - Qt API shape
        return 0 <= self._index < len(self._blocks)

    def text(self):
        return self._blocks[self._index] if self.isValid() else ""

    def previous(self):
        return _Block(self._blocks, self._index - 1)


class _Document:
    def __init__(self, blocks):
        self._blocks = blocks

    def lastBlock(self):  # noqa: N802 - Qt API shape
        return _Block(self._blocks, len(self._blocks) - 1)


class _Window:
    """Stands in for MyMainWindow, which owns the Log action."""

    def __init__(self):
        self.unread = None
        self.activity = None

    def set_log_unread_count(self, count):
        self.unread = int(count)

    def set_activity_message(self, message):
        self.activity = str(message)


class _LogConsole:
    def __init__(self, visible=False):
        self._visible = bool(visible)
        self.geometry_saves = 0

    def isVisible(self):  # noqa: N802 - Qt API shape
        return self._visible

    def show(self):
        self._visible = True

    def hide(self):
        self._visible = False

    def raise_(self):
        return None

    def activateWindow(self):  # noqa: N802 - Qt API shape
        return None

    def remember_geometry(self):
        self.geometry_saves += 1


class _Harness:
    """Minimal UI surface: a log display and its modeless utility window."""

    def __init__(self, visible=False):
        self.log_display = _LogDisplay(visible)
        self.log_console_window = _LogConsole(visible)
        self._window = _Window()
        self._log_unread_count = 0
        self.splitter_resizes = 0

    def window(self):
        return self._window

    def adjust_splitter_sizes(self):
        self.splitter_resizes += 1

    # Methods under test, bound to this harness.
    def note_log_activity(self):
        UiInteractionsMixin.note_log_activity(self)

    def toggle_plotter_visibility(self):
        UiInteractionsMixin.toggle_plotter_visibility(self)

    def publish_latest_log_message(self):
        UiInteractionsMixin.publish_latest_log_message(self)

    def _clear_log_unread_count(self):
        UiInteractionsMixin._clear_log_unread_count(self)

    def _publish_log_unread_count(self):
        UiInteractionsMixin._publish_log_unread_count(self)


def test_logging_while_hidden_does_not_open_the_panel():
    harness = _Harness(visible=False)

    for _ in range(50):
        harness.log_display.append("motion ratio 0.42")
        harness.note_log_activity()

    assert harness.log_console_window.isVisible() is False
    assert harness.splitter_resizes == 0, "the workspace must not be resized"
    assert harness._window.unread == 50


def test_logging_while_visible_does_not_accumulate_unread():
    harness = _Harness(visible=True)

    harness.log_display.append("hello")
    harness.note_log_activity()

    assert harness._log_unread_count == 0
    assert harness._window.unread is None, "no badge update needed when open"
    assert harness.splitter_resizes == 0


def test_opening_the_console_clears_the_unread_badge():
    harness = _Harness(visible=False)
    harness.note_log_activity()
    assert harness._window.unread == 1

    harness.toggle_plotter_visibility()

    assert harness.log_console_window.isVisible() is True
    assert harness._log_unread_count == 0
    assert harness._window.unread == 0
    assert harness.splitter_resizes == 0, (
        "the utility window must not resize the workspace"
    )


def test_closing_the_console_remembers_geometry_without_resizing():
    harness = _Harness(visible=True)

    harness.toggle_plotter_visibility()

    assert harness.log_console_window.isVisible() is False
    assert harness._window.unread is None
    assert harness.splitter_resizes == 0
    assert harness.log_console_window.geometry_saves == 1


def test_unread_count_survives_a_window_without_the_setter():
    """A window that predates the badge must not break logging."""
    harness = _Harness(visible=False)
    harness._window = object()

    harness.note_log_activity()

    assert harness._log_unread_count == 1


def test_latest_non_empty_log_line_is_mirrored_to_activity_status():
    harness = _Harness(visible=False)
    harness.log_display.append("Sensor update started")
    harness.log_display.append("Raw response details\nSensor update complete\n")

    harness.publish_latest_log_message()

    assert harness._window.activity == "Sensor update complete"

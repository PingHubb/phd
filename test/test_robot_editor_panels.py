"""The robot send editors are exclusive, and live inside the Robots page.

They used to sit above the tab widget, so opening one left it hanging over
every other workspace and pushed the whole control panel down. Moving them
into the page means `isVisible()` is False whenever another workspace is
showing, so the exclusivity logic and the button highlights have to ask
`is_visible_to_parent` instead -- otherwise an open editor reads as closed and
a second one opens on top of it.
"""

from phd.ui import components
from phd.ui.ui_ping_ui_interactions import UiInteractionsMixin


class _Editor:
    """Mimics a send editor: its own flag, plus a possibly-hidden parent."""

    def __init__(self, parent_visible=True):
        self._shown = False
        self._parent_visible = parent_visible

    # -- Qt surface used by the code under test ----------------------
    def parentWidget(self):  # noqa: N802 - Qt API shape
        return self

    def isVisibleTo(self, _parent):  # noqa: N802 - Qt API shape
        return self._shown

    def isVisible(self):  # noqa: N802 - Qt API shape
        return self._shown and self._parent_visible

    def toggle_visibility(self):
        self._shown = not self._shown


class _Section:
    def __init__(self):
        self._shown = True

    def parentWidget(self):  # noqa: N802 - Qt API shape
        return self

    def isVisibleTo(self, _parent):  # noqa: N802 - Qt API shape
        return self._shown

    def isVisible(self):  # noqa: N802 - Qt API shape
        return self._shown

    def setVisible(self, visible):  # noqa: N802 - Qt API shape
        self._shown = bool(visible)


class _Harness:
    def __init__(self, parent_visible=True):
        self.position_entry_widget = _Editor(parent_visible)
        self.position_quaternion_widget = _Editor(parent_visible)
        self.position_toolframe_widget = _Editor(parent_visible)
        self.position_script_widget = _Editor(parent_visible)
        self.read_group_robot = _Section()

    def editors(self):
        return [
            self.position_entry_widget,
            self.position_quaternion_widget,
            self.position_toolframe_widget,
            self.position_script_widget,
        ]

    def _sync_robot_send_button_highlights(self):
        # Button wiring is covered by the live checks; not needed here.
        pass

    def toggle(self, editor):
        UiInteractionsMixin._toggle_robot_editor(self, editor)


def _open_count(harness):
    return sum(components.is_visible_to_parent(e) for e in harness.editors())


def test_opening_one_editor_opens_exactly_one():
    harness = _Harness()
    harness.toggle(harness.position_entry_widget)

    assert _open_count(harness) == 1
    assert components.is_visible_to_parent(harness.position_entry_widget)


def test_opening_a_second_editor_closes_the_first():
    harness = _Harness()
    harness.toggle(harness.position_entry_widget)
    harness.toggle(harness.position_toolframe_widget)

    assert _open_count(harness) == 1
    assert components.is_visible_to_parent(harness.position_toolframe_widget)
    assert not components.is_visible_to_parent(harness.position_entry_widget)


def test_reclicking_closes_without_opening_another():
    harness = _Harness()
    harness.toggle(harness.position_script_widget)
    harness.toggle(harness.position_script_widget)

    assert _open_count(harness) == 0


def test_read_section_hides_only_while_an_editor_is_open():
    harness = _Harness()
    assert harness.read_group_robot.isVisible() is True

    harness.toggle(harness.position_quaternion_widget)
    assert harness.read_group_robot.isVisible() is False

    harness.toggle(harness.position_quaternion_widget)
    assert harness.read_group_robot.isVisible() is True


def test_exclusivity_holds_while_the_workspace_is_hidden():
    """Regression: with the page hidden, isVisible() reports every editor closed.

    Keying off it would let a second editor open on top of the first, and the
    Read section would reappear underneath an open editor.
    """
    harness = _Harness(parent_visible=False)
    harness.toggle(harness.position_entry_widget)

    # Qt would report this editor invisible, but it *is* open.
    assert harness.position_entry_widget.isVisible() is False
    assert components.is_visible_to_parent(harness.position_entry_widget)

    harness.toggle(harness.position_toolframe_widget)

    assert _open_count(harness) == 1, "a second editor opened on top of the first"
    assert harness.read_group_robot.isVisible() is False


def test_is_visible_to_parent_handles_a_missing_widget():
    assert components.is_visible_to_parent(None) is False

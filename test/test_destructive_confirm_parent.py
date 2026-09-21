"""Confirmation dialogs must be parented to the window, not to Qt's method.

`UiInteractionsMixin` is mixed into a QSplitter, so `self.parent` is Qt's bound
*method* rather than a widget. Passing it to QMessageBox raises TypeError,
which would have crashed "Clear Tool Pose Path" whenever a path was actually
plotted -- the `has_path` guard is the only reason it went unnoticed.
"""

from phd.ui import components
from phd.ui.ui_ping_ui_interactions import UiInteractionsMixin


class _Harness:
    """Mimics the mixin's real host: a widget that also has Qt's `parent`."""

    def __init__(self, *, has_path=True):
        self.top_level = object()
        self.cleared = False
        if has_path:
            self._direct_finger_motion_tool_pose_plot_actors = ["actor"]

    def window(self):
        return self.top_level

    def parent(self):
        """Stand-in for QWidget.parent -- a method, not a widget."""
        return None

    def _clear_direct_finger_motion_tool_pose_path_plot(self):
        self.cleared = True

    def confirm_clear(self):
        UiInteractionsMixin._confirm_clear_direct_finger_motion_tool_pose_path(self)


def test_confirm_is_parented_to_the_window(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        components, "confirm", lambda parent, *a, **k: seen.setdefault("parent", parent) or True
    )

    harness = _Harness()
    harness.confirm_clear()

    assert seen["parent"] is harness.top_level
    assert seen["parent"] is not harness.parent, "must not pass Qt's bound method"
    assert harness.cleared is True


def test_declining_the_confirmation_keeps_the_path(monkeypatch):
    monkeypatch.setattr(components, "confirm", lambda *a, **k: False)

    harness = _Harness()
    harness.confirm_clear()

    assert harness.cleared is False


def test_no_plotted_path_clears_without_asking(monkeypatch):
    def _refuse(*_args, **_kwargs):
        raise AssertionError("should not prompt when there is nothing to discard")

    monkeypatch.setattr(components, "confirm", _refuse)

    harness = _Harness(has_path=False)
    harness.confirm_clear()

    assert harness.cleared is True

"""The workspace nav bar must not impose its full width on the control panel.

A QTabBar's minimum becomes a floor on the whole panel it lives in. When the
nav bar reported each entry's full width as its minimum, the five entries
together held the control panel at 545 px, which pushed the 3D viewport below
its own minimum on a 960x540 window -- the app's smallest supported size.
"""

from phd.ui import nav_tabs


class _Bar:
    """Exercises the size hints against a known set of labels and icons."""

    def __init__(self, labels, *, with_icons=True):
        self._labels = list(labels)
        self._with_icons = with_icons
        self._widths = {}

    # -- the bits of QTabBar the hints rely on ------------------------
    def count(self):
        return len(self._labels)

    def tabText(self, index):  # noqa: N802 - Qt API shape
        return self._labels[index]

    def tabData(self, index):  # noqa: N802 - Qt API shape
        return "icon" if self._with_icons else None

    def _tab_width(self, index):
        # Stand in for font measurement: 8 px per character.
        return (
            nav_tabs.PAD_OUTER * 2
            + nav_tabs.PAD_INNER * 2
            + (nav_tabs.ICON_SIZE + nav_tabs.PAD_ICON_GAP if self._with_icons else 0)
            + 8 * len(self._labels[index])
        )

    # -- methods under test ------------------------------------------
    def tabSizeHint(self, index):  # noqa: N802 - Qt API shape
        return nav_tabs.NavigationTabBar.tabSizeHint(self, index)

    def minimumTabSizeHint(self, index):  # noqa: N802 - Qt API shape
        return nav_tabs.NavigationTabBar.minimumTabSizeHint(self, index)


WORKSPACES = ["Sensor", "Robots", "Control", "AI", "Hand", "Tools"]


def test_minimum_is_narrower_than_preferred_for_long_labels():
    bar = _Bar(WORKSPACES)
    index = WORKSPACES.index("Control")

    assert (
        bar.minimumTabSizeHint(index).width() < bar.tabSizeHint(index).width()
    ), "a long entry must be able to compress"


def test_minimum_never_exceeds_preferred_for_short_labels():
    """A short entry is already narrow; its minimum must not inflate it."""
    bar = _Bar(WORKSPACES)
    index = WORKSPACES.index("AI")

    assert (
        bar.minimumTabSizeHint(index).width() <= bar.tabSizeHint(index).width()
    )


def test_total_minimum_leaves_room_for_the_viewport_at_960():
    """The five entries together must fit the panel's own minimum width."""
    from phd.ui import theme

    bar = _Bar(WORKSPACES)
    total_minimum = sum(
        bar.minimumTabSizeHint(i).width() for i in range(bar.count())
    )

    assert total_minimum <= theme.CONTROL_PANEL_MIN_WIDTH, (
        f"nav bar minimum {total_minimum} would hold the panel wider than "
        f"its {theme.CONTROL_PANEL_MIN_WIDTH} px floor"
    )


def test_compressed_entry_keeps_room_for_icon_and_some_label():
    bar = _Bar(WORKSPACES)
    width = bar.minimumTabSizeHint(WORKSPACES.index("Control")).width()

    assert width >= nav_tabs.ICON_SIZE + nav_tabs.MIN_LABEL_WIDTH


def test_preferred_width_still_fits_the_whole_label():
    bar = _Bar(WORKSPACES)
    index = WORKSPACES.index("Control")

    assert bar.tabSizeHint(index).width() >= 8 * len("Control")


def test_height_is_the_same_for_both_hints():
    bar = _Bar(WORKSPACES)
    for index in range(bar.count()):
        assert (
            bar.minimumTabSizeHint(index).height()
            == bar.tabSizeHint(index).height()
            == nav_tabs.TAB_HEIGHT
        )

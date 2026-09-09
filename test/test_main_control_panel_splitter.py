from phd.ui.ui_ping import UI


class _Panel:
    def __init__(self):
        self.visible = False

    def setVisible(self, visible):
        self.visible = bool(visible)


class _Handle:
    def __init__(self):
        self.panel_visible = None

    def set_panel_visible(self, visible):
        self.panel_visible = bool(visible)


class _Harness:
    def __init__(self):
        self.splitter_2 = _Panel()
        self.handle_widget = _Handle()
        self.current_sizes = [800, 620]
        self._main_control_panel_collapsed = False
        self._main_control_panel_last_width = 620
        self._main_control_panel_state_syncing = False

    def sizes(self):
        return list(self.current_sizes)

    @staticmethod
    def width():
        return 1420

    def setSizes(self, sizes):
        self.current_sizes = list(sizes)

    def handle(self, _index):
        return self.handle_widget

    def _sync_main_control_panel_handle(self):
        UI._sync_main_control_panel_handle(self)

    def _set_main_control_panel_visible(self, visible):
        UI._set_main_control_panel_visible(self, visible)


def test_middle_arrow_collapses_and_restores_main_tabs_panel():
    harness = _Harness()

    harness._set_main_control_panel_visible(False)

    assert harness.splitter_2.visible is True
    assert harness.current_sizes[1] == 0
    assert harness._main_control_panel_collapsed is True
    assert harness.handle_widget.panel_visible is False

    UI.toggle_main_control_panel(harness)

    assert harness.current_sizes[1] == 620
    assert harness._main_control_panel_collapsed is False
    assert harness.handle_widget.panel_visible is True

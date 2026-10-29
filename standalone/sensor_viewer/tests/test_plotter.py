import time

import numpy as np
import pytest
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QMouseEvent, QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QScrollArea

from standalone.sensor_viewer.viewer.app import MainWindow, STYLE
from standalone.sensor_viewer.viewer.core import Grid
from standalone.sensor_viewer.viewer.plotter import PlotterPanel, scene_geometry


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    return app


def pump(app, predicate, timeout=3):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    return False


def test_signed_geometry_does_not_change_measurements():
    values = np.array([-200., -50., 0., 50., 200., np.nan])
    original = values.copy()
    base, points, colours, valid = scene_geometry(Grid(2, 3), values, "difference", 100, 2)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_allclose(points[:, 2], [-1.3, -.65, 0, .65, 1.3, 0])
    assert not valid[-1]
    assert base[1, 1] < base[0, 1]  # Index increases down each column.
    assert base[2, 0] > base[0, 0]
    assert colours[0, 2] > colours[0, 0]  # Blue negative, orange positive.
    assert colours[4, 0] > colours[4, 2]


def test_raw_geometry_is_relative_not_force():
    _base, points, _colours, _valid = scene_geometry(Grid(1, 3), [100, 150, 200], "raw", 1, 1)
    np.testing.assert_allclose(points[:, 2], [0, .325, .65])
    _base, points, _colours, _valid = scene_geometry(Grid(1, 1), [100], "raw", 1, 1)
    assert np.isfinite(points).all() and points[0, 2] == 0


@pytest.mark.parametrize("shape", [(1, 1), (5, 5), (8, 10), (20, 20)])
def test_all_views_render_and_selection_stays_aligned(app, shape):
    grid = Grid(*shape)
    panel = PlotterPanel()
    panel.resize(900, 640)
    panel.show()
    values = np.linspace(-100, 100, grid.size)
    panel.set_frame(grid, values, "difference", 100, 0, True)
    chosen = []
    panel.taxel_selected.connect(chosen.append)
    for index in range(4):
        panel.view_mode.setCurrentIndex(index)
        app.processEvents()
        assert not panel.grab().isNull()
    panel.view_mode.setCurrentIndex(0)
    panel.plotter.top_view()
    app.processEvents()
    pick = min(12, grid.size-1)
    points, _ = panel.plotter.projected_taxels()
    QTest.mouseClick(panel.plotter, Qt.LeftButton, pos=QPoint(*map(round, points[pick])))
    assert panel.plotter.selected == pick
    assert pick == 0 or chosen[-1] == pick
    panel.close()


def test_rotate_pan_zoom_do_not_select_or_change_signal(app):
    panel = PlotterPanel()
    panel.resize(900, 640)
    panel.show()
    values = np.arange(25, dtype=float)
    panel.set_frame(Grid(5, 5), values, "difference", 100, 12, True)
    plot = panel.plotter
    chosen = []
    panel.taxel_selected.connect(chosen.append)
    start, end = QPointF(220, 170), QPointF(270, 190)
    old_yaw = plot.yaw
    plot.mousePressEvent(QMouseEvent(QMouseEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    plot.mouseMoveEvent(QMouseEvent(QMouseEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
    plot.mouseReleaseEvent(QMouseEvent(QMouseEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
    assert plot.yaw != old_yaw and not chosen
    plot.mousePressEvent(QMouseEvent(QMouseEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton, Qt.ShiftModifier))
    plot.mouseMoveEvent(QMouseEvent(QMouseEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton, Qt.ShiftModifier))
    plot.mouseReleaseEvent(QMouseEvent(QMouseEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton, Qt.ShiftModifier))
    assert np.linalg.norm(plot.pan) > 0 and not chosen
    plot.wheelEvent(QWheelEvent(start, start, QPoint(), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert plot.zoom > 1
    np.testing.assert_array_equal(values, np.arange(25))
    plot.reset_view()
    assert plot.zoom == 1 and np.all(plot.pan == 0)
    panel.close()


def test_popout_selection_stream_and_offline_state(app):
    window = MainWindow()
    window.show()
    window.start(True)
    assert pump(app, lambda: window.packet is not None)
    assert window.plot_panel.view_mode.currentData() == "point_grid"
    window.open_plotter()
    dialog = window.plotter_dialog
    assert dialog.isVisible()
    dialog.panel.plotter.select(17)
    assert window.selected.value() == 17
    assert window.heatmap.selected == window.plot_panel.plotter.selected == 17
    assert "Taxel 17" in window.chart.label
    window.mode.setCurrentIndex(3)
    assert dialog.panel.plotter.signal_mode == "percent"
    window.disconnect()
    assert pump(app, lambda: window.worker is None)
    assert "OFFLINE" in dialog.windowTitle()
    window.close()
    app.processEvents()


def test_small_window_has_no_horizontal_crop(app):
    window = MainWindow()
    window.resize(920, 650)
    window.show()
    app.processEvents()
    for area in window.findChildren(QScrollArea):
        assert area.widget().width() <= area.viewport().width()
    window.close()

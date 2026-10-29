"""Exact offline event inspection without shaded regions or measurement edits."""

import copy
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QPoint, Qt, QEvent
from PyQt5.QtGui import QHelpEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QToolTip

from phd.ui import theme
from phd.ui.calibration_inspection import TimelineInspection
from phd.ui.calibration_result_dialog import CalibrationCurveWidget
from phd.ui.calibration_comparison import CalibrationComparisonDialog


@pytest.fixture(scope="session")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


def dataset():
    rows = []
    for index in range(100):
        phase = "force_baseline" if index < 20 else "approaching" if index < 60 else "returning" if index < 90 else "recovery"
        mode = "loading" if phase == "approaching" else "unloading" if 60 <= index < 80 else "retract" if 80 <= index < 90 else ""
        travel = max(0, (index - 20) / 4) if index < 60 else max(0, (90 - index) / 3)
        force = max(0, travel - 7) * 100
        signal = -travel * 10
        rows.append({"elapsed_s": float(index), "phase": phase, "step_mode": mode,
                     "step_state": "dwell", "travel_down_mm": travel,
                     "force_delta_abs_g": force, "force_delta_g": force,
                     "diff_ave": signal, "diff": signal, "raw": 1000 + signal,
                     "frame_sequence": index, "sensor_monotonic_s": index + 100})
    return {"name": "Offline test", "signal_field": "diff_ave", "rows": rows,
            "source_path": "/synthetic/trial.csv",
            "metadata": {"taxel_index": 0, "sensor_shape": [1, 1], "sensor_identity": "test"}}


def test_timeline_snaps_in_pixels_and_does_not_select_offscreen_markers():
    inspector = TimelineInspection(dataset()["rows"], {"contact_onset": {"elapsed_s": 50}})
    time, marker = inspector.snap(80.5, (0, 100), 1000)
    assert time == 80 and marker["events"][0]["label"] == "Fast return"
    assert inspector.snap(82, (0, 100), 1000) == (82, None)
    assert inspector.snap(80.5, (80.4, 81), 1000) == (80.5, None)
    assert inspector.snap(80, (80, 80), 0) == (80, None)
    assert TimelineInspection([], {}).snap(80, (0, 100), 1000) == (80, None)


def test_coincident_events_share_a_target_and_saved_samples_are_not_interpolated():
    rows = dataset()["rows"]
    before = copy.deepcopy(rows)
    inspector = TimelineInspection(rows, {"contact_onset": {"elapsed_s": 60}})
    _, marker = inspector.snap(60, (0, 100), 1000)
    assert {event["label"] for event in marker["events"]} >= {"Contact onset", "Return begins"}
    details = inspector.details(60, "diff_ave", "Signal change (sensor units)", marker)
    assert "Contact onset" in details and "Return begins" in details
    assert "Stage: Slow unloading" in details
    assert "Nearest saved sample: 60.0000 s" in details
    assert "Raw sensor value: 900" in details
    assert "Distance from surface: unavailable" in details
    assert inspector.sample_at(80.4) is rows[80]
    assert inspector.sample_at(-1) is None and inspector.sample_at(101) is None
    assert rows == before


def test_missing_nan_and_legacy_values_are_explicit_not_invented():
    rows = [{"elapsed_s": 0, "phase": "returning", "diff_ave": float("nan")},
            {"elapsed_s": 1, "phase": "returning"}, {"elapsed_s": "bad"}]
    inspector = TimelineInspection(rows, {"signal_onset": {"elapsed_s": float("nan")}})
    assert [event["key"] for marker in inspector.markers for event in marker["events"]] == ["return"]
    text = inspector.details(.4, "diff_ave", "Signal")
    assert "Stage: Return (unspecified)" in text
    assert "Signal: not recorded" in text
    assert "Force change (magnitude): not recorded" in text
    assert "Robot downward travel: not recorded" in text
    assert "No saved sample" in inspector.details(2, "diff_ave", "Signal")


def curve_position(chart, time, context_index=1, offset_x=0):
    context = chart._hover_contexts[context_index]
    sample = chart._sample_at_x(context["points"], time)
    x, y = chart._map_point(*sample, context["x_range"], context["y_range"], context["rect"])
    return QPoint(round(x + offset_x), round(y))


def test_qt_marker_hover_keeps_inline_values_without_popup_in_all_panels_and_zoom(app, monkeypatch):
    data = dataset()
    original = copy.deepcopy(data)
    chart = CalibrationCurveWidget(data["rows"], "diff_ave", "process")
    popups, labels = [], []
    monkeypatch.setattr(QToolTip, "showText", lambda *args: popups.append(args))
    original_label = chart._draw_hover_label
    def capture_label(*args, **kwargs):
        labels.append(args[3])
        return original_label(*args, **kwargs)
    monkeypatch.setattr(chart, "_draw_hover_label", capture_label)
    QToolTip.hideText()
    try:
        chart.resize(1100, 820)
        chart.show()
        app.processEvents()
        for index in range(3):
            pos = curve_position(chart, 80, index, offset_x=3)
            assert chart._process_hover_selection(pos)[0] == 80
            labels.clear()
            QTest.mouseMove(chart, pos)
            app.processEvents()
            chart.grab()
            assert len(labels) >= 3
            assert all("t=80.0000 s, value=" in text for text in labels)
            QApplication.sendEvent(chart, QHelpEvent(QEvent.ToolTip, pos, chart.mapToGlobal(pos)))
            assert not popups and not QToolTip.isVisible()
        assert not chart.toolTip()  # No delayed generic description either.
        chart._view_x_range = (79, 81)
        chart._base_image = None
        chart.grab()
        pos = curve_position(chart, 80, offset_x=4)
        assert chart._process_hover_selection(pos)[0] == 80
        chart.reset_view()
        assert not popups
        assert data == original
    finally:
        chart.close()


def test_qt_process_has_only_dotted_events_no_stage_cards_or_background_bands(app, monkeypatch):
    chart = CalibrationCurveWidget(dataset()["rows"], "diff_ave", "process")
    markers = []
    original = chart._draw_vertical_marker
    def capture(painter, value, x_range, rect, color, style):
        markers.append((value, color, style))
        return original(painter, value, x_range, rect, color, style)
    monkeypatch.setattr(chart, "_draw_vertical_marker", capture)
    try:
        chart.resize(1100, 820)
        chart.grab()
        assert any(time == 80 and color == chart.EVENT_COLORS["fast_return"] for time, color, _ in markers)
        assert all(style == Qt.DotLine for _, _, style in markers)
        assert not hasattr(chart, "_paint_return_stage_bands")
        assert not hasattr(chart, "_paint_return_stage_keys")
        assert "First at" not in chart._process_legend_document(chart.font(), 1000).toPlainText()
        # A blank point in the fast-return region remains the plot background.
        context = chart._hover_contexts[-1]
        rect = context["rect"]
        x = rect.left() + 85 / 99 * rect.width()
        y = rect.top() + .13 * rect.height()
        assert chart._base_image.pixelColor(round(x), round(y)).name() == theme.INPUT_BG.lower()
    finally:
        chart.close()


def test_comparison_snap_readout_without_popup_and_clean_export(app, tmp_path, monkeypatch):
    data = dataset()
    original = copy.deepcopy(data)
    dialog = CalibrationComparisonDialog([data])
    popups = []
    monkeypatch.setattr(QToolTip, "showText", lambda *args: popups.append(args))
    QToolTip.hideText()
    try:
        dialog.add_node(0, 0)
        dialog.mode_selector.setCurrentIndex(dialog.mode_selector.findData("process"))
        dialog.resize(1250, 920)
        dialog.show()
        app.processEvents()
        dialog.canvas.draw()
        assert all(not axis.patches for axis in dialog.plot_axes)
        assert all(axis.get_legend() is None for axis in dialog.plot_axes)
        axis = dialog.plot_axes[1]
        x, y = axis.transData.transform((80, -100 / 3))
        time = axis.transData.inverted().transform((x + 3, y))[0]
        dialog._hover(SimpleNamespace(inaxes=axis, xdata=time, x=x+3, y=y))
        assert dialog._last_hover[1] == 80
        assert "Fast return" in dialog.readout.text()
        assert "elapsed_s=80" in dialog.readout.text()
        for field in ("surface_distance_mm", "diff_ave", "force_delta_abs_g"):
            assert f"{field}=" in dialog.readout.text()
        ratio = dialog.canvas.devicePixelRatioF()
        pos = QPoint(round(x / ratio), round(dialog.canvas.height() - y / ratio))
        QApplication.sendEvent(dialog.canvas, QHelpEvent(QEvent.ToolTip, pos, dialog.canvas.mapToGlobal(pos)))
        assert not dialog.canvas.toolTip()
        assert not popups and not QToolTip.isVisible()
        assert all(list(cursor.get_xdata()) == [80, 80] for _, _, cursor in dialog._cursors)
        image, sidecar = dialog.export_comparison(tmp_path / "clean.png")
        assert image.exists() and sidecar.exists() and not popups
        assert all(not marker.get_visible() for marker in dialog._markers)
        assert data == original
    finally:
        dialog.close()

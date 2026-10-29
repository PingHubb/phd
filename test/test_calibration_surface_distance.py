"""Contact-referenced time graphs without rewriting source measurements."""

import copy

import pytest
from PyQt5.QtWidgets import QApplication

from phd.ui.calibration_analysis import analyze_calibration_rows, surface_referenced_rows
from phd.ui.calibration_comparison import (
    CalibrationComparisonDialog, comparison_entry, shared_axis_ranges,
)
from phd.ui.calibration_graph_analysis import CalibrationGraphAnalysisDialog
from phd.ui.calibration_result_dialog import CalibrationCurveWidget, SURFACE_DISTANCE_LABEL


@pytest.fixture
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


def dataset():
    rows = []
    samples = [("force_baseline", 0.0)] * 20
    samples += [("approaching", i * .5) for i in range(30)]
    samples += [("returning", i * .5) for i in reversed(range(30))]
    for index, (phase, travel) in enumerate(samples):
        force = max(0, travel - 10) * 100
        rows.append({
            "elapsed_s": index * .1, "phase": phase,
            "travel_down_mm": travel, "diff_ave": -travel * 10,
            "force_delta_abs_g": force, "force_delta_g": force,
            "frame_sequence": index, "sensor_monotonic_s": 100 + index * .1,
        })
    return {"name": "Surface reference test", "signal_field": "diff_ave", "rows": rows,
            "source_path": "/recordings/surface-reference.json",
            "metadata": {"taxel_index": 0, "sensor_shape": [1, 1],
                         "sensor_identity": "test-sensor"}}


def test_surface_distance_uses_each_trials_contact_and_keeps_input_unchanged():
    source = dataset()["rows"]
    before = copy.deepcopy(source)
    analysis = analyze_calibration_rows(source)
    contact = analysis["contact_onset"]
    rows = surface_referenced_rows(source, analysis=analysis)
    assert rows[0]["surface_distance_mm"] == contact["distance_mm"]
    assert rows[contact["row_index"]]["surface_distance_mm"] == 0
    assert min(row["surface_distance_mm"] for row in rows) < 0
    assert rows[-1]["surface_distance_mm"] == rows[0]["surface_distance_mm"]
    assert all(row["surface_distance_mm"] == -row["contact_relative_travel_mm"] for row in rows)
    assert source == before
    assert [row["travel_down_mm"] for row in rows] == [row["travel_down_mm"] for row in source]


def test_surface_distance_zero_contact_and_missing_positions():
    source = [{"travel_down_mm": value} for value in (0, 1, -1, None, float("nan"))]
    rows = surface_referenced_rows(source, analysis={"contact_onset": {"distance_mm": 0}})
    assert [row["surface_distance_mm"] for row in rows] == [0, -1, 1, None, None]


def test_no_contact_does_not_use_a_saved_stale_reference():
    source = dataset()["rows"]
    for row in source:
        row.update(force_delta_abs_g=0, force_delta_g=0,
                   contact_relative_travel_mm=123, surface_distance_mm=456)
    rows = surface_referenced_rows(source)
    assert all(row["surface_distance_mm"] is None for row in rows)
    assert all(row["contact_relative_travel_mm"] is None for row in rows)


def test_process_chart_signed_axis_hover_and_export(app, tmp_path):
    data = dataset()
    before = copy.deepcopy(data)
    chart = CalibrationCurveWidget(data["rows"], "diff_ave", "process")
    try:
        chart.resize(1100, 760)
        chart.grab()
        context = chart._hover_contexts[0]
        assert context["label"] == SURFACE_DISTANCE_LABEL
        assert context["y_range"][0] < 0 < context["y_range"][1]
        contact = chart.analysis["contact_onset"]
        assert chart._sample_at_x(context["points"], contact["elapsed_s"])[1] == 0
        assert chart.save_png(tmp_path / "surface_distance.png")
        assert data == before
    finally:
        chart.close()


def test_process_without_contact_keeps_signal_and_force_visible(app):
    data = dataset()
    for row in data["rows"]:
        row["force_delta_abs_g"] = row["force_delta_g"] = 0
    chart = CalibrationCurveWidget(data["rows"], "diff_ave", "process")
    try:
        chart.resize(1100, 760)
        chart.grab()
        assert len(chart._hover_contexts) == 2
        assert all("surface" not in context["label"] for context in chart._hover_contexts)
        assert all(row["surface_distance_mm"] is None for row in chart.rows)
    finally:
        chart.close()


def test_comparison_preserves_stationary_contact_detection():
    data = dataset()
    for row in data["rows"]:
        row["step_state"] = "dwell"
    for row in data["rows"][22:24]:
        row.update(step_state="moving", force_delta_abs_g=100)
    analysis = analyze_calibration_rows(data["rows"])
    entry = comparison_entry(data, 0, 0)
    assert entry["analysis"]["contact_onset"] == analysis["contact_onset"]
    assert entry["rows"][0]["surface_distance_mm"] == analysis["contact_onset"]["distance_mm"]


def test_multi_node_process_uses_signed_surface_scales_and_actual_points(app):
    data = dataset()
    dialog = CalibrationComparisonDialog([data])
    try:
        dialog.add_node(0, 0)
        dialog.mode_selector.setCurrentIndex(dialog.mode_selector.findData("process"))
        limits = shared_axis_ranges([data])["surface_distance_mm"]
        assert dialog.plot_axes[0].get_ylim() == pytest.approx(limits)
        assert dialog.plot_axes[0].get_ylabel() == SURFACE_DISTANCE_LABEL
        curve = dialog._curves[0]
        assert curve[2] == "surface_distance_mm"
        contact = dialog.entries[0]["analysis"]["contact_onset"]
        assert CalibrationCurveWidget._sample_at_x(curve[3], contact["elapsed_s"])[1] == 0
        assert min(y for _, y in curve[3]) < 0
    finally:
        dialog.close()


def test_analysis_and_enlarged_process_view_have_identical_surface_reference(app):
    data = dataset()
    dialog = CalibrationGraphAnalysisDialog({"datasets": [data]})
    try:
        dialog._open_chart_detail("process")
        detail = dialog._detail_dialog
        assert detail.chart.rows == dialog.charts["process"].rows
        detail.chart.grab()
        assert detail.chart._hover_contexts[0]["label"] == SURFACE_DISTANCE_LABEL
    finally:
        dialog.close()

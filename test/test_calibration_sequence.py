"""Return-stage labels and speed provenance, entirely offline."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QApplication

from phd.ui.calibration_sequence import sequence_segments, stage_at_time, return_stage_legend
from phd.ui.calibration_result_dialog import CalibrationCurveWidget
from phd.ui.calibration_comparison import CalibrationComparisonDialog, comparison_entry
from phd.ui.calibration_graph_analysis import analyze_calibration_dataset, build_trial_report
from phd.ui.experiment_tasks import CalibrationTask


def dataset():
    rows = []
    for i in range(100):
        phase, mode = ("force_baseline", "") if i < 20 else ("approaching", "loading")
        if i >= 60:
            phase, mode = "returning", "unloading"
        if i >= 80:
            phase, mode = "returning", "retract"
        if i >= 90:
            phase, mode = "recovery", ""
        travel = max(0, (i - 20) / 4) if i < 60 else max(0, (90 - i) / 3)
        force = max(0, travel - 7) * 100
        rows.append({"elapsed_s": float(i), "phase": phase, "step_mode": mode,
                     "step_state": "dwell", "travel_down_mm": travel,
                     "force_delta_g": force, "force_delta_abs_g": force,
                     "diff_ave": -travel * 10, "tool_z_m": .3 - travel / 1000})
    return {"name": "Synthetic sequence", "source_path": "/synthetic/trial.csv",
            "signal_field": "diff_ave", "rows": rows,
            "metadata": {"sensor_shape": [1, 1], "taxel_index": 0,
                         "sensor_identity": "offline-test", "return_speed_m_s": .0001}}


@pytest.fixture(scope="session")
def app():
    # Keep Qt alive while other modules drain deferred Matplotlib paint events.
    # Recreating QApplication between charts can crash in native teardown.
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def test_return_stages_are_recorded_tags_not_inferred_speed_or_force():
    data = dataset()
    before = copy.deepcopy(data)
    segments = sequence_segments(data["rows"])
    keys = return_stage_legend(segments)
    assert [(s["label"], s["start_s"], s["end_s"]) for s in keys] == [
        ("Slow unloading", 60, 80), ("Fast return", 80, 90), ("Recovery (stationary)", 90, 99)]
    assert stage_at_time(segments, 79.999)["kind"] == "unloading"
    assert stage_at_time(segments, 80)["kind"] == "fast_return"
    assert stage_at_time(segments, 99)["kind"] == "recovery"
    assert stage_at_time(segments, 100) is None
    assert stage_at_time(segments, -1) is None
    assert data == before


def test_legacy_and_missing_stage_tags_never_invent_fast_motion():
    rows = dataset()["rows"]
    for row in rows:
        row.pop("step_mode")
    keys = return_stage_legend(sequence_segments(rows))
    assert [s["kind"] for s in keys] == ["return", "recovery"]
    assert keys[0]["label"] == "Return (unspecified)"
    for row in rows:
        row.pop("phase")
    assert return_stage_legend(sequence_segments(rows)) == []


def test_pauses_split_stages_and_bad_timestamps_are_ignored():
    rows = dataset()["rows"]
    rows[84]["phase"] = "waiting_force"
    rows.append({"elapsed_s": "nan", "phase": "recovery"})
    rows.append({"elapsed_s": None, "phase": "returning"})
    segments = sequence_segments(rows)
    assert stage_at_time(segments, 84)["label"] == "Waiting force"
    assert stage_at_time(segments, 85)["kind"] == "fast_return"
    assert len(return_stage_legend(segments)) == 3
    assert sequence_segments([]) == []


def test_process_labels_hover_zoom_and_png_do_not_change_samples(app, tmp_path):
    data = dataset()
    before = copy.deepcopy(data)
    chart = CalibrationCurveWidget(data["rows"], "diff_ave", "process")
    try:
        chart.resize(1100, 760)
        chart.grab()
        assert len(chart._hover_contexts) == 3
        chart._view_x_range = (75, 99)
        chart._base_image = None
        chart.grab()
        rect = chart._last_plot_rect
        chart._hover_position = QPoint(int(rect.left() + rect.width() * 10 / 24), int(rect.center().y()))
        chart.grab()  # Includes stage-at-cursor text without a live device.
        assert stage_at_time(chart.sequence_segments, 85)["label"] == "Fast return"
        assert chart.save_png(tmp_path / "return_stages.png")
        chart.set_data(data["rows"][:50], "diff_ave")
        assert return_stage_legend(chart.sequence_segments) == []
        chart.grab()
        assert data == before
    finally:
        chart.close()


def test_event_legend_text_has_the_same_readable_colours_as_markers(app, monkeypatch):
    chart = CalibrationCurveWidget(dataset()["rows"], "diff_ave", "process")
    try:
        document = chart._process_legend_document(chart.font(), 1000)
        for event, label in chart.EVENT_LEGEND:
            selection = document.find(label)
            assert not selection.isNull()
            assert selection.charFormat().foreground().color().name() == chart.EVENT_COLORS[event].lower()
            # White plotting surface: text must remain readable, not pale yellow.
            rgb = selection.charFormat().foreground().color().getRgbF()[:3]
            linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in rgb]
            luminance = sum(c * weight for c, weight in zip(linear, (.2126, .7152, .0722)))
            assert 1.05 / (luminance + .05) >= 4.5
        compact = chart._process_legend_document(chart.font(), 450)
        assert compact.size().height() > document.size().height()
        assert compact.toPlainText() == document.toPlainText()
        colours = []
        original = chart._draw_vertical_marker
        def capture(painter, value, x_range, rect, colour, style):
            colours.append(colour)
            return original(painter, value, x_range, rect, colour, style)
        monkeypatch.setattr(chart, "_draw_vertical_marker", capture)
        chart.resize(1100, 760)
        chart.grab()
        assert set(chart.EVENT_COLORS.values()) <= set(colours)
    finally:
        chart.close()


def test_comparison_keeps_stage_tags_and_hover_uses_each_trial_time(app, tmp_path):
    first = dataset()
    second = copy.deepcopy(first)
    second["name"] = "Delayed trial"
    for row in second["rows"]:
        row["elapsed_s"] += 10
    entry = comparison_entry(first, 0, 0)
    assert entry["rows"][80]["step_mode"] == "retract"
    dialog = CalibrationComparisonDialog([first, second])
    try:
        dialog.add_node(0, 0)
        dialog.add_node(1, 0)
        dialog.mode_selector.setCurrentIndex(dialog.mode_selector.findData("process"))
        assert dialog.plot_axes[0].get_legend() is None
        assert all(not axis.patches for axis in dialog.plot_axes)
        assert any(list(line.get_xdata()) == [80, 80] and line.get_linestyle() == ":"
                   for line in dialog.plot_axes[0].lines)
        for layout in (0, 1):
            dialog.layout_selector.setCurrentIndex(layout)
            dialog._hover(SimpleNamespace(inaxes=dialog.plot_axes[0], xdata=85))
            assert "Synthetic sequence Fast return" in dialog.readout.text()
            assert "Delayed trial Slow unloading" in dialog.readout.text()
        image, sidecar = dialog.export_comparison(tmp_path / "comparison.png")
        assert image.exists()
        payload = json.loads(sidecar.read_text())
        assert payload["entries"][0]["sequence_segments"] == entry["sequence_segments"]
    finally:
        dialog.close()


@pytest.mark.parametrize("mode,protocol,scope,fast", [
    ("stepwise", "standard", "nominal_stepwise_unloading_rate", .01),
    ("velocity", "standard", "velocity_return_command", None),
    ("stepwise", "proximity_profile", "not_applicable_to_proximity_profile_steps", None),
])
def test_metadata_qualifies_only_applicable_return_speeds(mode, protocol, scope, fast):
    task = CalibrationTask(storage_path=None, output_directory=None)
    task.motion_mode, task.protocol = mode, protocol
    details = task._return_speed_metadata()
    assert details["return_speed_m_s_scope"] == scope
    assert details["fast_return_command_speed_m_s"] == fast


def test_saved_and_in_memory_metadata_agree_without_rewriting_legacy_rate(tmp_path):
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    task._samples = dataset()["rows"]
    task._trial_speed_m_s = .0001
    result = task._save_result()
    saved = json.loads(Path(result["metadata_path"]).read_text())
    for payload in (result, saved):
        assert payload["return_speed_m_s"] == .0001
        assert payload["fast_return_command_speed_m_s"] == .01
        assert payload["return_speed_m_s_scope"] == "nominal_stepwise_unloading_rate"
        assert payload["measurement_quality"]["recovery_observation_s"] == 5
        motion = payload["measurement_quality"]["motion"]
        assert motion["nominal_unloading_rate_mm_s"] == .1
        assert motion["fast_return_command_speed_mm_s"] == 10


@pytest.mark.parametrize("at_anchor", [False, True])
def test_fast_return_commands_are_still_ten_mm_per_second(monkeypatch, at_anchor):
    task = CalibrationTask(storage_path=None, output_directory=None)
    task._start_tool_position = [.1, .2, .3]
    task._step_target = [.1, .2, .3 if at_anchor else .25]
    calls = []
    monkeypatch.setattr(task, "_send_step_move", lambda *args: calls.append(args) or True)
    if at_anchor:
        task._next_unloading_step(None, 100)
    else:
        monkeypatch.setattr(task, "_fresh_force_reading", lambda *args: None)
        monkeypatch.setattr(task, "_tick_step", lambda *args, **kwargs: 0)
        task._tick_stepwise_return(None, 100, [.1, .2, .25], "next_trial")
    assert calls[0][-2:] == ("retract", .01)
    assert np.array_equal(calls[0][2], task._start_tool_position)


def test_report_does_not_treat_old_nominal_return_speed_as_fast_return():
    data = dataset()
    report = build_trial_report(analyze_calibration_dataset(data))
    assert "nominal slow-unloading rate: 0.1000 mm/s" in report
    assert "Fast return commanded speed: not recorded in this file" in report
    assert "Fast return: 80.00–90.00 s" in report
    data["metadata"]["fast_return_command_speed_m_s"] = .01
    report = build_trial_report(analyze_calibration_dataset(data))
    assert "Fast return commanded speed: 10.0000 mm/s" in report

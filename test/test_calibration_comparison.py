"""Common scales, independent windows and offline multi-node comparisons."""

import copy
import csv
import json
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from phd.ui.calibration_comparison import (
    CalibrationComparisonDialog, comparison_entry, comparison_group, comparison_source,
    shared_axis_ranges,
)
from phd.ui.calibration_graph_analysis import CalibrationGraphAnalysisDialog
from phd.ui.calibration_result_dialog import CalibrationCurveWidget


@pytest.fixture(scope="session")
def app():
    # QApplication must outlive all Qt/Matplotlib objects, including deferred
    # canvas draws in later test modules (as it does in the real viewer).
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()
    instance.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.fixture(autouse=True)
def drain_qt_events(app):
    yield
    app.processEvents()
    app.sendPostedEvents(None, QEvent.DeferredDelete)


def dataset(name="Trial 1", *, field="diff_ave", shape=(2, 2), identity="specimen-A"):
    rows = []
    for i in range(60):
        distance = i / 10 if i < 30 else (59 - i) / 10
        row = {
            "elapsed_s": i / 10, "phase": "approaching" if i < 30 else "returning",
            "travel_down_mm": distance,
            "force_delta_abs_g": max(0, distance - 1.5) * 100,
            field: -distance * 10, "frame_sequence": i, "sensor_monotonic_s": i / 10 + 100,
        }
        for node in range(shape[0] * shape[1]):
            row[f"taxel_{node:03d}_{field}"] = -distance * (10 if node == 0 else 1000)
        rows.append(row)
    return {"name": name, "signal_field": field, "rows": rows,
            "source_path": f"/recordings/{name}.csv",
            "metadata": {"sensor_shape": list(shape), "taxel_index": 0,
                         "measurement_quality": {"sensor_identity": identity}}}


def test_shared_scales_include_all_signed_native_columns_and_largest_trial():
    first = dataset()
    second = dataset("Trial 2")
    second["rows"][-1]["taxel_003_diff_ave"] = 5000
    second["rows"][-1]["force_delta_abs_g"] = 800
    before = copy.deepcopy(first)
    ranges = shared_axis_ranges([first, second])
    assert ranges["diff_ave"][0] < -2900
    assert ranges["diff_ave"][1] > 5000
    assert ranges["force_delta_abs_g"] == (0, 864)
    assert ranges["elapsed_s"] == (0, 5.9)
    assert ranges["travel_down_mm"][0] == 0
    assert ranges["contact_relative_travel_mm"][0] < 0
    assert first == before


def test_legacy_single_node_and_newton_force_still_get_shared_limits():
    data = dataset()
    for row in data["rows"]:
        for key in list(row):
            if key.startswith("taxel_"):
                del row[key]
        del row["force_delta_abs_g"]
        row["force_delta_abs_n"] = .980665
    ranges = shared_axis_ranges([data])
    assert ranges["diff_ave"][0] > -40
    assert ranges["force_delta_abs_g"][1] == pytest.approx(108)
    assert comparison_entry(data, 0, 0)["rows"][0]["force_delta_abs_g"] == pytest.approx(100)
    with pytest.raises(ValueError, match="no recorded"):
        comparison_entry(data, 1, 0)


def test_units_shape_and_sensor_identity_are_separate_groups():
    base = comparison_group(dataset())
    assert base == comparison_group(dataset("Trial 2"))
    for changed in (dataset(field="diff_percent_ave"), dataset(shape=(5, 5)), dataset(identity="specimen-B")):
        assert comparison_group(changed) != base


def test_compact_snapshot_keeps_fresh_frame_keys_and_contact_reference():
    data = dataset()
    result = comparison_entry(data, 1, 3)
    assert result["key"] == (3, 1)
    assert result["physical_target_taxel_index"] == 0
    row = result["rows"][10]
    assert row["diff_ave"] == -1000
    assert row["frame_sequence"] == 10
    assert row["sensor_monotonic_s"] == 101
    assert not any(key.startswith("taxel_") for key in row)
    assert result["analysis"]["contact_onset"] is not None
    assert result["rows"][0]["contact_relative_travel_mm"] < 0
    data["rows"][10]["taxel_001_diff_ave"] = 9000
    assert row["diff_ave"] == -1000


def test_missing_contact_reference_is_not_fabricated():
    data = dataset()
    for row in data["rows"]:
        row["force_delta_abs_g"] = 0
    result = comparison_entry(data, 0, 0)
    assert all(row["contact_relative_travel_mm"] is None for row in result["rows"])


def test_shared_limits_apply_to_all_process_panels_and_reset_after_zoom(app):
    data = dataset()
    chart = CalibrationCurveWidget(data["rows"], "diff_ave", "process")
    try:
        ranges = shared_axis_ranges([data])
        chart.set_shared_axis_ranges(ranges)
        chart.resize(1000, 750)
        chart.grab()
        assert chart._last_x_range == ranges["elapsed_s"]
        assert [context["y_range"] for context in chart._hover_contexts] == [
            ranges["surface_distance_mm"], ranges["diff_ave"], ranges["force_delta_abs_g"],
        ]
        chart.zoom_view(.5)
        assert chart._view_x_range is not None
        chart.reset_view()
        chart.grab()
        assert chart._last_x_range == ranges["elapsed_s"]
    finally:
        chart.close()


def test_analysis_keeps_multiple_details_open_and_unchanged(app):
    data = dataset()
    dialog = CalibrationGraphAnalysisDialog({"datasets": [data], "selected_path": data["source_path"]})
    try:
        assert dialog.shared_scales_check.isChecked()
        dialog._open_chart_detail("signal_distance")
        first = dialog._detail_dialog
        saved = copy.deepcopy(first.chart.rows)
        dialog._select_taxel(1)
        dialog._open_chart_detail("signal_distance")
        second = dialog._detail_dialog
        assert first is not second
        assert first.isVisible() and second.isVisible()
        assert first.chart.rows == saved
        assert len(dialog._detail_dialogs) == 2
        for detail in (first, second):
            detail.chart.grab()
        assert first.chart._last_x_range == second.chart._last_x_range
        assert first.chart._last_y_range == second.chart._last_y_range
        dialog.shared_scales_check.setChecked(False)
        assert not dialog.charts["signal_distance"].shared_axis_ranges
        assert first.chart.shared_axis_ranges
    finally:
        dialog.close()
    assert not first.isVisible() and not second.isVisible()


def test_repeatability_uses_common_band_envelope_and_force_x_units(app):
    data = dataset()
    ranges = shared_axis_ranges([data])
    chart = CalibrationCurveWidget([], "diff_ave", "repeatability")
    try:
        chart.set_trials([{"rows": data["rows"]}, {"rows": data["rows"]}], "diff_ave")
        chart.set_shared_axis_ranges(ranges)
        chart.resize(1000, 750)
        chart.grab()
        assert chart._last_x_range == ranges["travel_down_mm"]
        assert chart._hover_contexts[0]["y_range"] == ranges["repeatability:diff_ave"]
        assert chart._hover_contexts[1]["y_range"] == ranges["repeatability:force_delta_abs_g"]
        chart.repeatability_x_field = "force_delta_abs_g"
        chart.reset_view()
        chart.grab()
        assert chart._last_x_range == ranges["force_delta_abs_g"]
    finally:
        chart.close()


@pytest.fixture
def comparison(app):
    dialog = CalibrationComparisonDialog([dataset(), dataset("Trial 2")])
    yield dialog
    dialog.close()


def test_comparison_add_remove_duplicate_and_limit(comparison):
    assert comparison.add_node(0, 0)
    assert not comparison.add_node(0, 0)
    assert "already pinned" in comparison.status_label.text()
    for index, node in ((0, 1), (0, 2), (0, 3), (1, 0), (1, 1)):
        assert comparison.add_node(index, node)
    assert not comparison.add_node(1, 2)
    assert len(comparison.entries) == 6
    comparison._remove_selected()
    assert len(comparison.entries) == 5
    comparison._clear()
    assert not comparison.entries


def test_comparison_rejects_incompatible_units_but_allows_different_grid(app):
    dialog = CalibrationComparisonDialog([dataset(), dataset(field="diff_percent_ave"), dataset(shape=(5, 5))])
    try:
        assert dialog.add_node(0, 0)
        assert not dialog.add_node(1, 0)
        assert "Different signal types/units" in dialog.status_label.text()
        assert dialog.add_node(2, 0)
        assert len(dialog.entries) == 2
        assert "Different sensors/grids" in dialog.context.text()
    finally:
        dialog.close()


def test_side_by_side_uses_identical_plot_area_and_linked_limits(comparison, app):
    comparison.add_node(0, 0)
    comparison.add_node(0, 1)
    comparison.resize(1250, 900)
    comparison.show()
    app.processEvents()
    comparison.canvas.draw()
    first, second = comparison.plot_axes
    assert first.get_xlim() == second.get_xlim()
    assert first.get_ylim() == second.get_ylim()
    assert first.get_window_extent().width == pytest.approx(second.get_window_extent().width)
    assert first.get_window_extent().height == pytest.approx(second.get_window_extent().height)
    first.set_xlim(.5, 1)
    first.set_ylim(-500, 0)
    assert second.get_xlim() == (.5, 1)
    assert second.get_ylim() == (-500, 0)
    comparison.redraw()
    assert comparison.plot_axes[0].get_ylim()[0] < -2900
    assert "physically probed IDs" in comparison.context.text()


def test_overlay_preserves_each_node_and_approach_return_branches(comparison):
    comparison.add_node(0, 0)
    comparison.add_node(0, 1)
    comparison.layout_selector.setCurrentIndex(1)
    assert len(comparison.plot_axes) == 1
    assert len(comparison._curves) == 4
    assert {curve[5] for curve in comparison._curves} == {"approach", "return"}
    assert min(point[1] for point in comparison._curves[2][3]) < -2000
    axis = comparison.plot_axes[0]
    comparison._hover(SimpleNamespace(inaxes=axis, xdata=1.0))
    assert "Node 0" in comparison.readout.text() and "Node 1" in comparison.readout.text()
    assert all(marker.get_visible() for marker in comparison._markers)
    comparison._leave()
    assert not any(marker.get_visible() for marker in comparison._markers)


@pytest.mark.parametrize("mode", ["force_distance", "force_signal", "process", "contact_signal"])
def test_comparison_modes_share_axes_and_export_scales(comparison, tmp_path, mode):
    comparison.add_node(0, 0)
    comparison.add_node(0, 1)
    comparison.mode_selector.setCurrentIndex(comparison.mode_selector.findData(mode))
    assert len(comparison.plot_axes) == (6 if mode == "process" else 2)
    if mode == "process":
        comparison.plot_axes[0].set_xlim(.5, 2)
        assert all(axis.get_xlim() == (.5, 2) for axis in comparison.plot_axes)
    path, sidecar = comparison.export_comparison(tmp_path / f"compare_{mode}.png")
    payload = json.loads(sidecar.read_text())
    assert path.stat().st_size > 10000
    assert payload["mode"] == mode
    assert not payload["amplitude_normalisation"]
    assert len(payload["entries"]) == 2
    assert payload["axes"][0]["x"] == list(comparison.plot_axes[0].get_xlim())
    assert payload["entries"][1]["physical_target_taxel_index"] == 0
    with pytest.raises(ValueError, match="already exists"):
        comparison.export_comparison(path)


def test_export_never_overwrites_source_recording(comparison, tmp_path):
    comparison.add_node(0, 0)
    comparison.entries[0]["source_path"] = str(tmp_path / "input.json")
    with pytest.raises(ValueError, match="input recording"):
        comparison.export_comparison(tmp_path / "input.png")


def test_main_compare_button_pins_current_selection(comparison, app):
    data = dataset()
    dialog = CalibrationGraphAnalysisDialog({"datasets": [data]})
    try:
        dialog.compare_button.click()
        workspace = dialog._comparison_dialog
        assert workspace.isVisible()
        assert workspace.entries[0]["taxel_index"] == 0
        dialog._select_taxel(1)
        dialog.compare_button.click()
        assert dialog._comparison_dialog is workspace
        assert [entry["taxel_index"] for entry in workspace.entries] == [0, 1]
    finally:
        dialog.close()


def test_full_400_node_profile_does_not_fit_each_node(monkeypatch):
    import phd.ui.calibration_analysis as analysis
    monkeypatch.setattr(analysis, "analyze_calibration_rows", lambda *_a, **_k: pytest.fail("Must scan columns, not fit every node"))
    data = dataset(shape=(20, 20))
    ranges = shared_axis_ranges([data])
    assert ranges["diff_ave"][0] < -2900


def write_recording(directory, *, identity="specimen-B", sensor="Sensor B", field="diff_ave", shape=(5, 5)):
    directory.mkdir(parents=True)
    data = dataset("Trial 0001", identity=identity, field=field, shape=shape)
    data["metadata"].update(sensor_name=sensor, trial_number=1, signal_field=field)
    data["metadata"]["measurement_quality"]["trial_completed"] = True
    path = directory / "trial_0001.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data["rows"][0]))
        writer.writeheader()
        writer.writerows(data["rows"])
    data["metadata"]["csv_path"] = str(path)
    path.with_suffix(".json").write_text(json.dumps(data["metadata"]))
    return path.with_suffix(".json")


def test_cross_sensor_import_preserves_pins_labels_shared_axes_and_export(comparison, tmp_path):
    assert comparison.add_node(0, 0)
    before = copy.deepcopy(comparison.entries[0])
    path = write_recording(tmp_path / "other_run")
    result = comparison.import_recordings([path])
    assert result == {"added": 1, "duplicates": 0, "errors": [], "warnings": []}
    assert comparison.entries == [before]
    assert comparison.source_selector.count() == 2
    assert comparison.trial_selector.currentData() == 2
    assert comparison.node_selector.count() == 25
    comparison.node_selector.setCurrentIndex(comparison.node_selector.findData(12))
    comparison.add_button.click()
    second = comparison.entries[1]
    assert second["key"] == (2, 12)
    assert "Sensor B" in second["label"] and "other_run" in second["label"]
    assert second["source_path"] == str(path.with_suffix(".csv"))
    assert second["rows"][10]["diff_ave"] == -1000
    assert comparison.plot_axes[0].get_ylim() == comparison.plot_axes[1].get_ylim()
    assert comparison.plot_axes[0].get_ylim()[0] < -2900
    assert "Equal digital units" in comparison.context.text()
    comparison.layout_selector.setCurrentIndex(1)
    assert len(comparison.plot_axes) == 1
    comparison._hover(SimpleNamespace(inaxes=comparison.plot_axes[0], xdata=1))
    assert "Sensor B" in comparison.readout.text() and "specimen-A" in comparison.readout.text()
    _, sidecar = comparison.export_comparison(tmp_path / "cross_sensor.png")
    exported = json.loads(sidecar.read_text())
    assert exported["shared_scale_scope"] == "pinned_nodes"
    assert exported["entries"][1]["sensor_name"] == "Sensor B"
    assert exported["entries"][1]["source_path"] == str(path.with_suffix(".csv"))


def test_import_deduplicates_json_csv_png_and_keeps_existing_snapshots(comparison, tmp_path):
    path = write_recording(tmp_path / "other")
    graph = path.with_suffix(".png")
    graph.write_bytes(b"graph")  # The loader resolves its CSV; it does not read pixels.
    result = comparison.import_recordings([path, path.with_suffix(".csv"), graph])
    assert result["added"] == 1 and result["duplicates"] == 2
    assert len(comparison.datasets) == 3
    comparison.add_button.click()
    before = copy.deepcopy(comparison.entries)
    again = comparison.import_recordings([path])
    assert again["added"] == 0 and again["duplicates"] == 1
    assert comparison.entries == before
    comparison.add_button.click()
    assert "already pinned" in comparison.status_label.text()


def test_failed_and_cancelled_import_do_not_replace_current_data(comparison, tmp_path, monkeypatch):
    from phd.ui.calibration_comparison import QFileDialog
    comparison.add_node(0, 0)
    before = copy.deepcopy(comparison.entries)
    result = comparison.import_recordings([tmp_path / "missing.csv"])
    assert result["added"] == 0 and len(result["errors"]) == 1
    assert len(comparison.datasets) == 2 and comparison.entries == before
    assert "missing.csv" in comparison.status_label.toolTip()
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *_: ([], ""))
    monkeypatch.setattr(comparison, "import_recordings", lambda *_: pytest.fail("Cancelled picker must not import"))
    comparison.import_button.click()
    assert comparison.entries == before


def test_import_full_run_is_lazy_and_selects_requested_trial(comparison, tmp_path, monkeypatch):
    from test_calibration_taxel_browsing import saved_grid
    from phd.ui import calibration_recording_catalog as catalog
    folder = saved_grid(tmp_path)
    monkeypatch.setattr(catalog, "read_calibration_csv_rows", lambda *_a, **_k: pytest.fail("Import must not read all samples"))
    result = comparison.import_recordings([folder / "taxel_12/trial_0002.json"])
    assert result["added"] == 5
    assert all(not d["rows"] for d in comparison.datasets[2:])
    selected = comparison.datasets[comparison.trial_selector.currentData()]
    assert selected["metadata"]["taxel_index"] == 12
    assert selected["metadata"]["trial_number"] == 2
    assert comparison.node_selector.currentData() == 12


def test_pinned_scales_never_scan_unselected_external_trials(comparison, monkeypatch):
    import phd.ui.calibration_comparison as module
    comparison.datasets.append({"deferred_csv": True, "source_path": "/missing.csv", "signal_field": "diff_ave"})
    monkeypatch.setattr(module, "dataset_axis_bounds", lambda *_: pytest.fail("Only pinned snapshots should set scales"))
    comparison.add_node(0, 0)
    assert comparison.plot_axes[0].get_ylim()[0] > -40
    comparison.add_node(0, 1)
    assert comparison.plot_axes[0].get_ylim()[0] < -2900
    comparison._remove_selected()
    assert comparison.plot_axes[0].get_ylim()[0] > -40


def test_main_compare_selection_stays_correct_after_external_import(app, tmp_path):
    dialog = CalibrationGraphAnalysisDialog({"datasets": [dataset(), dataset("Trial 2")]})
    try:
        dialog.compare_button.click()
        workspace = dialog._comparison_dialog
        path = write_recording(tmp_path / "other")
        workspace.import_recordings([path])
        workspace.add_button.click()
        assert workspace.trial_selector.currentData() == 2
        dialog._select_taxel(1)
        dialog.compare_button.click()
        assert workspace.trial_selector.currentData() == 0
        assert workspace.node_selector.currentData() == 1
        assert [e["key"] for e in workspace.entries] == [(0, 0), (2, 0), (0, 1)]
    finally:
        dialog.close()


def test_run_label_uses_actual_folder_and_saved_probe_not_stale_metadata():
    data = dataset()
    data["source_path"] = "/moved/calibration_full_taxel_20261028_125345_sensor-A_probe-insulative/taxel_12/trial_0001.csv"
    data["metadata"].update(sensor_name="Testing Sensor 1", experiment_directory="/old/path")
    source = comparison_source(data)
    assert source["run"] == "2026-10-28 12:53:45"
    assert source["probe"] == "insulative"
    assert source["path"].startswith("/moved/")
    assert "Testing Sensor 1" in source["label"]


def test_fullscreen_fits_all_panels_and_restores_view_without_losing_zoom(comparison, app):
    comparison.add_node(0, 0)
    comparison.add_node(0, 1)
    comparison.mode_selector.setCurrentIndex(comparison.mode_selector.findData("process"))
    comparison.show()
    app.processEvents()
    axes = list(comparison.plot_axes)
    axes[0].set_xlim(.5, 2)
    snapshots = copy.deepcopy(comparison.entries)
    comparison.fullscreen_button.click()
    app.processEvents()
    assert comparison.isFullScreen()
    assert comparison.controls_panel.isHidden()
    assert comparison.controls_button.text() == "Show Controls"
    assert comparison.fit_check.isChecked()
    assert comparison.canvas.minimumHeight() == 1
    assert comparison.scroll.verticalScrollBar().maximum() == 0
    assert comparison.canvas.height() == comparison.scroll.viewport().height()
    assert comparison.plot_axes == axes
    assert all(axis.get_xlim() == (.5, 2) for axis in axes)
    comparison.reject()  # QDialog's Esc fallback must exit, not close.
    app.processEvents()
    assert not comparison.isFullScreen() and comparison.isVisible()
    assert not comparison.controls_panel.isHidden()
    assert not comparison.fit_check.isChecked()
    assert comparison.canvas.minimumHeight() == 620
    assert comparison.entries == snapshots
    assert all(axis.get_xlim() == (.5, 2) for axis in axes)


def test_fullscreen_keyboard_shortcuts_and_maximized_restore(comparison, app):
    comparison.add_node(0, 0)
    comparison.controls_button.setChecked(False)
    comparison.fit_check.setChecked(True)
    comparison.showMaximized()
    comparison.activateWindow()
    comparison.fullscreen_button.setFocus()
    app.processEvents()
    QTest.keyClick(comparison.fullscreen_button, Qt.Key_F11)
    app.processEvents()
    assert comparison.isFullScreen()
    QTest.keyClick(comparison.mode_selector, Qt.Key_Escape)
    app.processEvents()
    assert comparison.isVisible() and comparison.isMaximized()
    assert not comparison.isFullScreen()
    assert comparison.fit_check.isChecked()
    assert comparison.controls_panel.isHidden()


def test_fit_mode_handles_six_full_sequences_without_clipping(comparison, app):
    for index, node in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)):
        comparison.add_node(index, node)
    comparison.mode_selector.setCurrentIndex(comparison.mode_selector.findData("process"))
    comparison.controls_button.setChecked(False)
    comparison.fit_check.setChecked(True)
    comparison.resize(1280, 900)
    comparison.show()
    app.processEvents()
    assert len(comparison.plot_axes) == 18
    assert comparison.scroll.verticalScrollBar().maximum() == 0
    comparison.readout.setText("Detailed hover values " * 500)
    app.processEvents()
    assert comparison.readout_scroll.height() <= 76
    assert comparison.readout_scroll.verticalScrollBar().maximum() > 0
    assert comparison.scroll.viewport().height() > 400
    comparison.fit_check.setChecked(False)
    app.processEvents()
    assert comparison.canvas.minimumHeight() == 1860
    assert comparison.scroll.verticalScrollBar().maximum() > 0
    comparison.layout_selector.setCurrentIndex(1)
    assert comparison.canvas.minimumHeight() == 620
    comparison.fit_check.setChecked(True)
    comparison.mode_selector.setCurrentIndex(comparison.mode_selector.findData("force_signal"))
    app.processEvents()
    assert comparison.canvas.minimumHeight() == 1
    assert comparison.scroll.verticalScrollBar().maximum() == 0

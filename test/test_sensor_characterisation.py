"""No hardware: acquisition, provenance and workbench regression checks."""
import copy
import csv
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from phd.dependence.characterisation_study import (
    CharacterisationRecorder, analyse_study, compare_map_reference,
    compare_studies, display_vectors, evaluate_prediction_csv,
    export_study_report, import_event_csv, nominal_pressure_kpa,
    study_plot_data, validate_condition,
)
from phd.dependence.touch_tracking_data import load_trial


@pytest.fixture
def recorded(tmp_path):
    def make(*, reason=None, mount="flat1", specimen="woven1", filtered=True):
        recorder = CharacterisationRecorder(tmp_path, {
            "protocol": "spatial", "specimen_id": specimen, "session_id": "day1",
            "mounting_id": mount, "sensor_shape": [4, 4],
            "orientation_confirmed": True, "baseline_confirmed": True,
        }, baseline_sec=2, queue_size=1024, chunk_size=64)
        base = np.full((4, 4), 100., np.float32)
        for i in range(801):
            t = i*.01
            raw = base.copy()
            if 3 <= t < 6:
                raw[3, 0] -= 30  # Display taxel zero, preserving negative sign.
                raw[0, 3] += 20  # Display taxel fifteen, separate peak.
            assert recorder.sensor_frame(recorder.started+t, i+1, raw, base,
                                         filtered_signed=raw-base if filtered else None)
        recorder.mark("condition_start", {"contact_count": 2, "gesture_label": "two_contacts"}, timestamp=recorder.started+3)
        recorder.mark("contact_reference", timestamp=recorder.started+3.1)
        recorder.mark("command_reference", timestamp=recorder.started+3.3)
        recorder.mark("release", timestamp=recorder.started+6)
        recorder.mark("condition_start", {"contact_count": 0, "gesture_label": "idle"}, timestamp=recorder.started+6.1)
        recorder.mark("release", timestamp=recorder.started+7.9)
        recorder.reference({"force_n": .25, "force_source_monotonic": recorder.started+4}, timestamp=recorder.started+4.01)
        recorder.reference({"force_n": 100, "force_source_monotonic": recorder.started}, timestamp=recorder.started+4.1)
        recorder.close(reason)
        return recorder.path
    return make


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("change", [
    {"contact_count": -1}, {"contact_count": True}, {"contact_count": 6},
    {"x_mm": 1}, {"x_mm": "nan", "y_mm": 1},
    {"position_source": "measured_fixture"}, {"probe_diameter_mm": 0},
    {"nominal_speed_mm_s": -1}, {"second_x_mm": 2},
])
def test_invalid_condition_fails_closed(change):
    with pytest.raises(ValueError):
        validate_condition(change)


@pytest.mark.parametrize("change", [
    {"specimen_id": ""}, {"session_id": ""}, {"mounting_id": ""},
    {"orientation_confirmed": False}, {"baseline_confirmed": False},
    {"protocol": "not_a_protocol"},
])
def test_required_preflight(tmp_path, change):
    metadata = dict(protocol="spatial", specimen_id="one", session_id="s1", mounting_id="m1",
                    sensor_shape=[4, 4], orientation_confirmed=True, baseline_confirmed=True)
    metadata.update(change)
    with pytest.raises(ValueError):
        CharacterisationRecorder(tmp_path, metadata)
    assert not list(tmp_path.iterdir())


def test_native_signed_roundtrip_and_spatial_orientation(recorded):
    path = recorded()
    metadata, arrays = load_trial(path)
    assert metadata["status"] == "complete"
    assert arrays["raw"].shape == (801, 4, 4)
    assert np.array_equal(arrays["filtered_signed"], arrays["raw"]-arrays["baseline"])
    report = analyse_study(path)
    segment = report["segments"][0]
    assert segment["signed_response_by_taxel"][0] == -30
    assert segment["signed_response_by_taxel"][15] == 20
    assert segment["active_taxels"] == 2
    assert {p["taxel_index"] for p in segment["peak_candidates"]} == {0, 15}
    assert segment["median_latest_force_n"] == .25  # Stale 100 N excluded.
    assert segment["centroid_column_row"] == pytest.approx([1.2, 1.2])
    assert report["segments"][1]["declared_no_contact_false_activation_episodes_per_min"] == 0
    assert report["independent_latency_events"] == []  # Manual markers are not latency.
    assert report["acquisition_quality"]["observed_rate_hz"] == pytest.approx(100)
    packet = study_plot_data(path, report)
    assert packet["peak"].max() == 30
    assert packet["filtered_peak"].max() == 30
    export_study_report(path, report)
    assert (path / "condition_summary.csv").exists()
    assert json.loads((path / "analysis.json").read_text())["schema"] == report["schema"]


def test_optional_filtered_signal_and_display_order(recorded):
    path = recorded(filtered=False)
    report = analyse_study(path)
    assert not report["acquisition_quality"]["filtered_signed_available"]
    assert "filtered_peak" not in study_plot_data(path, report)
    values = np.arange(12).reshape(1, 3, 4)
    assert display_vectors(values).tolist() == [[8, 4, 0, 9, 5, 1, 10, 6, 2, 11, 7, 3]]


def test_interrupted_data_kept_but_not_treated_as_independent_complete_trials(recorded):
    failed = analyse_study(recorded(reason="simulated disconnection"))
    good = analyse_study(recorded())
    assert not failed["acquisition_quality"]["complete"]
    result = compare_studies([good, failed])
    assert result["excluded_trials"] == [failed["metadata"]["trial_id"]]
    assert all(g["independent_trials"] == 1 for g in result["groups"])
    assert all(g["between_trial_sd_units"] is None for g in result["groups"])


def test_comparison_separates_mounts_and_averages_per_trial(recorded):
    first = analyse_study(recorded(mount="flat1"))
    second = analyse_study(recorded(mount="remount2"))
    first["segments"].append(copy.deepcopy(first["segments"][0]))
    groups = compare_studies([first, second])["groups"]
    assert {g["condition"]["mounting_id"] for g in groups} == {"flat1", "remount2"}
    assert all(g["independent_trials"] == 1 for g in groups)
    third = copy.deepcopy(second)
    third["metadata"]["specimen_id"] = "another_sensor"
    with pytest.raises(ValueError, match="one specimen"):
        compare_studies([first, third])


def test_external_event_import_requires_clock_and_deduplicates(recorded, tmp_path):
    path = recorded()
    events = write_csv(tmp_path/"events.csv", ["t_s", "kind", "uncertainty_ms"], [
        [3.2, "contact_reference", .5], [3.2, "contact_reference", .5],
        [3.4, "command_reference", .7],
        [7, "command_reference", 1],  # Must not pair across release/new condition.
    ])
    with pytest.raises(ValueError, match="aligned"):
        import_event_csv(path, events)
    import_event_csv(path, events, clock_alignment_confirmed=True)
    import_event_csv(path, events, clock_alignment_confirmed=True)
    report = analyse_study(path)
    latency = report["independent_latency_events"]
    assert len(latency) == 1
    assert latency[0]["contact_to_command_ms"] == pytest.approx(200)
    assert latency[0]["timing_uncertainty_ms"] == pytest.approx(1.2)


@pytest.mark.parametrize("row", [[1, "contact_reference", 1], [20, "contact_reference", 1],
                                 [4, "contact_reference", -1], [4, "unknown", 1],
                                 ["nan", "contact_reference", 1]])
def test_invalid_external_event_rejected_without_modifying_trial(recorded, tmp_path, row):
    path = recorded()
    previous = (path/"characterisation.json").read_bytes()
    events = write_csv(tmp_path/"events.csv", ["t_s", "kind", "uncertainty_ms"], [row])
    with pytest.raises(ValueError):
        import_event_csv(path, events, clock_alignment_confirmed=True)
    assert (path/"characterisation.json").read_bytes() == previous


def test_prediction_labels_are_independent_annotations_not_sensor_pseudolabels(recorded, tmp_path):
    path = recorded()
    predictions = write_csv(tmp_path/"prediction.csv", ["t_s", "predicted_label"], [
        [3.01, "wrong_boundary_excluded"], [4, "two_contacts"], [5, "idle"], [7, "idle"],
    ])
    result = evaluate_prediction_csv(path, predictions)
    assert result["prediction_windows"] == 3
    assert result["accuracy"] == pytest.approx(2/3)
    assert "wrong_boundary_excluded" not in result["labels"]
    assert "cannot verify" in result["interpretation"]


def test_independent_map_error_in_mm_and_frame_preflight():
    result = {"task": "taxel_position_mapping", "positions": [
        {"status": "measured", "taxel_index": 12, "tool_position_m": [.103, .204, .4]},
    ]}
    reference = [{"taxel_index": "12", "x_mm": "100", "y_mm": "200"}]
    with pytest.raises(ValueError, match="SAME"):
        compare_map_reference(result, reference)
    report = compare_map_reference(result, reference, same_frame_confirmed=True)
    assert report["xy_rmse_mm"] == pytest.approx(5)
    assert report["points"][0]["dx_mm"] == pytest.approx(3)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_map_reference(result, reference*2, same_frame_confirmed=True)
    reference[0]["taxel_index"] = "12.2"
    with pytest.raises(ValueError, match="integer"):
        compare_map_reference(result, reference, same_frame_confirmed=True)


def test_nominal_pressure_requires_verified_force_and_full_known_contact_area():
    setup = {"force_reference_verified": True, "contact_area_confirmed": True,
             "probe_geometry": "flat_circular", "probe_diameter_mm": 10}
    assert nominal_pressure_kpa(1, setup) == pytest.approx(12.73239545)
    for field in ("force_reference_verified", "contact_area_confirmed"):
        assert nominal_pressure_kpa(1, {**setup, field: False}) is None
    assert nominal_pressure_kpa(1, {**setup, "probe_geometry": "finger"}) is None


def test_force_sweep_cycles_by_whole_pass(monkeypatch):
    from test_characterisation_quality import running_task
    from phd.ui.experiment_tasks import FullTaxelCalibrationTask
    clock, task, api, host = running_task(monkeypatch)
    task.protocol = "force_sweep"
    task.force_schedule_g = [5, 10, 20]
    task.force_reference_verified = True
    observed = []
    for number in range(1, 5):
        task._trial_number = number
        task._begin_trial(host, clock[0], api.tool_position)
        observed.append(task.contact_threshold_g)
    assert observed == [5, 10, 20, 5]
    assert "threshold-sweep-5-10-20g" in task._experiment_folder_name("test")
    full = FullTaxelCalibrationTask(storage_path=None, output_directory=None)
    full.protocol, full.force_schedule_g = "force_sweep", [5, 10, 20]
    full._pass_number, full._trial_number = 2, 9
    full._begin_trial(host, clock[0], api.tool_position)
    assert full.contact_threshold_g == 10
    assert "threshold-sweep-5-10-20g" in full._full_folder_name("test")


def test_force_sweep_preflight_rejects_unverified_or_out_of_range():
    from phd.ui.experiment_tasks import CalibrationTask
    from test_experiment_tasks import _Host, _RobotApi
    task = CalibrationTask(storage_path=None, output_directory=None)
    api = _RobotApi()
    host = _Host(api, task)
    task.protocol = "force_sweep"
    for schedule, verified in (([5, 10], False), ([900], True), ([float("nan")], True), ([], True)):
        task.force_schedule_g, task.force_reference_verified = schedule, verified
        assert not task._validate_measurement_settings(host)
    assert not api.velocity_commands


def test_workbench_opens_offline_and_refuses_active_motion(tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication, QWidget
    from phd.ui.sensor_characterisation_window import SensorCharacterisationWindow
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    host.ui_ros = None
    host._sidebar_active_task = None
    window = SensorCharacterisationWindow(host)
    assert window.tabs.count() == 4
    assert window.analysis_buttons[0].isEnabled()
    window._start()
    assert window.recorder is None
    assert "fresh tactile sensor" in window.status.text()
    host._sidebar_active_task = object()
    window._start()
    assert "Stop the active experiment" in window.status.text()
    window.close()
    app.processEvents()


def test_workbench_offline_analysis_draws_each_condition(recorded):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication, QWidget
    from phd.ui.sensor_characterisation_window import SensorCharacterisationWindow
    app = QApplication.instance() or QApplication([])
    host = QWidget()
    host.ui_ros = None
    host._sidebar_active_task = None
    window = SensorCharacterisationWindow(host)
    path = recorded()
    window._analyse_path(path)
    deadline = time.monotonic()+10
    while window._future is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert window._report is not None, window.status.text()
    assert window.plot_condition.count() == 2
    window.plot_condition.setCurrentIndex(1)
    assert (path/"sensor_response_condition_001.png").exists()
    assert (path/"sensor_response_condition_002.png").exists()
    window.close()
    app.processEvents()


def test_workbench_records_actual_completed_buffer_callbacks(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication, QWidget
    from phd.dependence.sensor_data import SensorDataBuffer
    from phd.ui.sensor_characterisation_window import SensorCharacterisationWindow
    app = QApplication.instance() or QApplication([])
    clock = [100.]
    real_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    data = SensorDataBuffer(4, 4, window_size=1)
    data.getCal(np.full((4, 4), 100.))
    data.frame_monotonic_sec = clock[0]
    host = QWidget()
    host._sidebar_active_task = None
    host.ui_ros = SimpleNamespace(sensor_functions=SimpleNamespace(
        _data=data, is_connected=True, cell_zero_mask=np.zeros((4, 4), bool)))
    window = SensorCharacterisationWindow(host)
    window.root.setText(str(tmp_path))
    window.specimen.setText("actual_buffer_test")
    window.mounting.setText("flat1")
    window.orientation.setChecked(True)
    window.clear_baseline.setChecked(True)
    window.baseline.setValue(2)
    window._start()
    assert window.recorder is not None, window.status.text()
    path = window.recorder.path
    for i in range(61):
        clock[0] = 100+i*.05
        raw = np.full((4, 4), 100.)
        if i >= 45:
            raw[3, 0] -= 30
        data.getRaw(raw)
        data.calDiff()
        data.calDiffPer()
        data.getWin(1)
        if i == 44:
            window._mark("condition_start")
        time.sleep(.001)
    window._mark("release")
    window._stop()
    assert window._on_frame not in data._frame_listeners
    deadline = real_monotonic()+10
    while window._future is not None and real_monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert window._report is not None, window.status.text()
    assert window._report["acquisition_quality"]["native_host_buffer_frames"] == 61
    assert window._report["segments"][0]["signed_response_by_taxel"][0] == -30
    metadata, arrays = load_trial(path)
    assert metadata["robot_commands_enabled"] is False
    assert arrays["filtered_signed"][-1, 3, 0] == -30
    window.close()
    app.processEvents()

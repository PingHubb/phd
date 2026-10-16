"""Hardware-free coverage for the independent-reference tracking pipeline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from phd.dependence.sensor_data import SensorDataBuffer
from phd.dependence.touch_tracking_data import (
    CausalPreprocessor, SCHEMA, TrialRecorder, align_reference, dataset_report,
    homography_from_corners, import_reference_csv, load_trial, prepare_trial,
    project_position, save_reference, split_trials, validate_reference, window_indices,
)
from phd.script.train_touch_tracking import (
    TrackingDataset, metrics, select_contact_threshold, train, training_normalization,
)


def reference(samples):
    return {"schema": SCHEMA, "source": "external_tracking", "verified": True, "samples": samples}


def sample(t, contact, x=None, y=None, reviewed=True):
    return {"t": t, "contact": contact, "x": x, "y": y, "quality": 1., "reviewed": reviewed}


def make_trial(root, session="day1", participant="p1", frames=150, seed=1):
    # Artificial reference positions drive the fake sensor, NOT vice versa.
    rng = np.random.default_rng(seed)
    shape = (4, 5)
    recorder = TrialRecorder(root, {"participant_id": participant, "session_id": session,
                                   "sensor_id": "test_sensor", "sensor_shape": list(shape),
                                   "surface_size_mm": [100., 80.], "synthetic_test_data": True},
                             chunk_size=32, queue_size=frames+10)
    yy, xx = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing="ij")
    labels = []
    baseline = np.full(shape, 10000., np.float32)
    for i in range(frames):
        t = i / 50.
        contact = int(20 <= i < frames-20)
        x = .3 + .2 * np.sin(t*2)
        y = .4 + .15 * np.cos(t*1.2)
        signal = -400 * np.exp(-((xx-x)**2 + (yy-y)**2)/.08) if contact else np.zeros(shape)
        raw = baseline + signal + rng.normal(0, .3, shape)
        assert recorder.sensor_frame(recorder.started+t, i+1, raw, baseline)
        labels.append(sample(t, contact, float(x) if contact else None, float(y) if contact else None))
    recorder.close()
    assert recorder.error is None
    save_reference(recorder.path, reference(labels))
    return recorder.path


def test_sensor_listeners_get_complete_fresh_native_frames():
    data = SensorDataBuffer(2, 3, 1)
    data.getCal(np.full((2, 3), 10.))
    seen = []
    def receive(value):
        seen.append((value.frame_sequence, value.frame_monotonic_sec, value.diffData.copy()))
    data.add_frame_listener(receive)
    data.add_frame_listener(receive)
    data.getRaw(np.full((2, 3), 8.))
    data.calDiff()
    data.calDiffPer()
    data.getWin(1)
    assert len(seen) == 1
    assert seen[0][0] == 1 and seen[0][1] > 0
    np.testing.assert_array_equal(seen[0][2], np.full((2, 3), -2.))
    data.remove_frame_listener(receive)
    data.getWin(1)
    assert len(seen) == 1


def test_sensor_bad_listener_does_not_stop_acquisition():
    data = SensorDataBuffer(2, 2, 1)
    def bad(_):
        raise ValueError("consumer failed")
    data.add_frame_listener(bad)
    data.getWin(1)
    data.getWin(1)
    assert data.frame_sequence == 2
    assert bad not in data._frame_listeners


def test_recorder_preserves_native_frames_and_marks_calibration_change(tmp_path):
    rec = TrialRecorder(tmp_path, {"participant_id": "a", "session_id": "day1", "sensor_id": "s", "sensor_shape": [2, 3]}, chunk_size=1)
    raw = np.arange(6).reshape(2, 3) + 100.
    baseline = np.full((2, 3), 100.)
    assert rec.sensor_frame(rec.started, 1, raw, baseline)
    assert not rec.sensor_frame(rec.started+.01, 1, raw, baseline)
    assert rec.sensor_frame(rec.started+.02, 2, raw+1, baseline)
    assert not rec.sensor_frame(rec.started+.04, 3, raw, baseline+1)
    rec.close()
    meta, arrays = load_trial(rec.path)
    assert meta["status"] == "failed"
    assert "Calibration" in meta["error"]
    np.testing.assert_array_equal(arrays["raw"][0], raw)
    assert len(arrays["t"]) == 2
    assert meta["robot_commands_enabled"] is False


def test_no_reference_and_old_controller_labels_are_rejected(tmp_path):
    path = make_trial(tmp_path)
    (path / "reference.json").unlink()
    report, prepared = dataset_report(tmp_path)
    assert not prepared and report["invalid_trials"] == 1
    fake = reference([sample(0, 0)])
    fake["source"] = "dfm"
    with pytest.raises(ValueError, match="reviewed camera"):
        validate_reference(fake)
    csv_path = tmp_path / "controller.csv"
    csv_path.write_text("t,vx,vy\n0,0.1,0\n")
    with pytest.raises(ValueError, match="CSV requires"):
        import_reference_csv(path, csv_path)


def test_camera_snapshot_is_unannotated_timestamped_and_copied():
    from phd.ui.camera_preview_window import _FrameGrabber
    grabber = _FrameGrabber()
    frame = np.full((3, 4, 3), 17, np.uint8)
    grabber._store_color_frame(frame, 12.5)
    frame[:] = 255
    timestamp, snapshot = grabber.color_snapshot()
    assert timestamp == 12.5 and np.all(snapshot == 17)
    snapshot[:] = 0
    assert np.all(grabber.color_snapshot()[1] == 17)
    assert grabber.color_snapshot(after=12.5) is None


def test_camera_resolution_change_fails_trial_without_losing_first_image(tmp_path):
    rec = TrialRecorder(tmp_path, {"participant_id": "p", "session_id": "s", "sensor_id": "camera_test", "sensor_shape": [2, 3]})
    assert rec.camera_frame(rec.started, np.zeros((3, 4, 3), np.uint8))
    assert not rec.camera_frame(rec.started+.04, np.zeros((4, 4, 3), np.uint8))
    rec.close()
    meta = json.loads((rec.path / "manifest.json").read_text())
    assert meta["status"] == "failed"
    assert "resolution changed" in meta["error"]
    assert len(list((rec.path / "camera").glob("*.jpg"))) == 1


def test_reference_interpolation_excludes_occlusion_transitions_and_long_gaps():
    ref = reference([sample(0, 1, .1, .3), sample(.1, 1, .2, .3),
                     sample(.2, 0), sample(.3, -1), sample(.4, 1, .6, .3, False),
                     sample(1., 1, .7, .3), sample(1.1, 1, .8, .3)])
    labels = align_reference(ref, [.05, .15, .2, .25, .4, .7, 1.05])
    assert labels["contact"].tolist() == [1, -1, 0, -1, -1, -1, 1]
    np.testing.assert_allclose(labels["xy"][0], [.15, .3])
    np.testing.assert_allclose(labels["velocity"][0], [1., 0.])
    assert not labels["position_valid"][2]
    assert labels["velocity_valid"][2]
    ref["time_offset_sec"] = .02
    shifted = align_reference(ref, [.01, .07])
    assert shifted["contact"].tolist() == [-1, 1]


def test_reference_versions_are_retained(tmp_path):
    first = reference([sample(0, 0), sample(.1, 0)])
    save_reference(tmp_path, first)
    save_reference(tmp_path, reference([sample(0, 1, .5, .5), sample(.1, 0)]))
    versions = list((tmp_path / "reference_history").glob("*.json"))
    assert len(versions) == 1
    assert json.loads(versions[0].read_text()) == first


def test_plane_calibration_orientation_and_invalid_corners():
    h = homography_from_corners([[10, 20], [110, 20], [110, 220], [10, 220]])
    np.testing.assert_allclose(project_position(h, [60, 120]), [.5, .5])
    with pytest.raises(ValueError, match="outside"):
        project_position(h, [-10, 120])
    with pytest.raises(ValueError, match="convex"):
        homography_from_corners([[0, 0], [10, 10], [0, 10], [10, 0]])


def test_preprocessing_is_causal_signed_and_gap_aware():
    pre = CausalPreprocessor(50)
    base = np.full((2, 3), 100.)
    mask = np.ones((2, 3), bool)
    first = pre.push(0., base, base, mask)
    assert len(first) == 1
    output = pre.push(.05, base-10, base, mask)
    # .02 and .04 must not see the future measurement at .05.
    assert [round(v[0], 2) for v in output] == [.02, .04]
    assert all(np.all(frame[0] == 0) for _, frame, _ in output)
    output = pre.push(.06, base-20, base, mask)
    assert np.all(output[-1][1][0] == -20)
    output = pre.push(.5, base, base, mask)
    assert any(not valid for _, _, valid in output)
    with pytest.raises(ValueError, match="strictly"):
        pre.push(.5, base, base, mask)


def test_window_validity_and_offline_live_preprocessing_parity(tmp_path):
    path = make_trial(tmp_path)
    trial = prepare_trial(path)
    _, arrays = load_trial(path)
    pre = CausalPreprocessor(50, .1)
    live = []
    for t, raw, base, mask in zip(arrays["t"], arrays["raw"], arrays["baseline"], arrays["valid_taxels"]):
        live.extend(pre.push(t, raw, base, mask))
    np.testing.assert_array_equal(trial["frames"], np.stack([v[1] for v in live]))
    assert len(window_indices(trial, 16)) > 30
    trial["frame_valid"][50] = False
    indices = window_indices(trial, 16)
    assert not any(50 <= i < 66 for i in indices)


def test_whole_group_split_and_duplicate_leakage_guard():
    trials = [{"metadata": {"participant_id": f"p{i}", "session_id": "day1"}, "data_hash": f"h{i}"} for i in range(5)]
    splits = split_trials(trials, group_by="participant")
    ids = [{t["metadata"]["participant_id"] for t in v} for v in splits.values()]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    with pytest.raises(ValueError, match="at least 3"):
        split_trials(trials[:2])
    for trial in trials:
        trial["data_hash"] = "same_raw_recording"
    with pytest.raises(ValueError, match="Duplicate"):
        split_trials(trials)


def test_augmentation_moves_reference_positions_and_velocity_together(tmp_path, monkeypatch):
    trial = prepare_trial(make_trial(tmp_path))
    norm = training_normalization([trial])
    data = TrackingDataset([trial], 16, norm, 1., augment=True)
    # Force both flips and zero drift/unit gain.
    monkeypatch.setattr(np.random, "random", lambda: 0.)
    monkeypatch.setattr(np.random, "uniform", lambda a, b: (a+b)/2)
    index = next(i for i, (_, k) in enumerate(data.indices) if trial["labels"]["position_valid"][k])
    result = data[index]
    _, k = data.indices[index]
    np.testing.assert_allclose(result["xy"], 1-trial["labels"]["xy"][k])
    np.testing.assert_allclose(result["velocity"], -trial["labels"]["velocity"][k])


def test_metrics_do_not_hide_missed_contact_positions_and_gate_velocity():
    rows = [{"truth_contact": 1, "probability": .1, "truth_xy": [0, 0], "xy": [1, 0],
             "truth_velocity": [1, 0], "velocity": [100, 0], "position_valid": True, "velocity_valid": True},
            {"truth_contact": 0, "probability": .9, "truth_xy": [0, 0], "xy": [0, 0],
             "truth_velocity": [0, 0], "velocity": [.2, 0], "position_valid": False, "velocity_valid": True}]
    report = metrics(rows, .5, [100, 80])
    assert report["position_rmse_mm"] == 100
    assert report["missed_contact_rate"] == 1
    assert report["no_touch_false_motion_rate"] == 1
    assert report["velocity_rmse_surface_units_per_sec"] < 1
    with pytest.raises(ValueError, match="AND no-touch"):
        select_contact_threshold(rows[:1])


def test_models_support_rectangular_sensors_without_robot_dependencies():
    import torch
    from phd.dependence.touch_tracking_model import TouchTracker
    torch.set_num_threads(1)
    for architecture in ("gru", "tcn"):
        model = TouchTracker(5, 8, architecture)
        result = model(torch.zeros(2, 16, 4, 5, 8))
        assert result["xy"].shape == (2, 2)
        assert result["velocity"].shape == (2, 2)
        assert result["contact_logits"].shape == (2,)


def test_end_to_end_synthetic_training_and_checkpoint(tmp_path):
    import torch
    from phd.dependence.touch_tracking_model import TrackingPreview
    root = tmp_path / "data"
    for i in range(3):
        make_trial(root, session=f"day{i}", seed=i)
    report, prepared = dataset_report(root)
    assert report["ready_trials"] == 3 and report["invalid_trials"] == 0
    args = argparse.Namespace(root=root, output=tmp_path / "models", group_by="session", seed=42,
                              epochs=1, batch_size=32, seq_len=8, hz=50., max_gap=.1, label_gap=.12,
                              threads=1, device="cpu", architectures="gru,tcn", lr=.001, patience=2, no_augment=True)
    output = train(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["selected_architecture"] in {"gru", "tcn"}
    assert (output / "test" / "predictions.csv").exists()
    assert (output / "centroid_test" / "metrics.json").exists()
    assert not list((tmp_path / "models").glob("latest*"))
    predictor = TrackingPreview(output / "selected_model.pt")
    _, arrays = load_trial(prepared[0]["path"])
    result = None
    for k in range(20):
        result = predictor.update(float(arrays["t"][k]), k+1, arrays["raw"][k], arrays["baseline"][k],
                                  arrays["valid_taxels"][k], sensor_id="test_sensor")
    assert result["valid"] and result["robot_commands_enabled"] is False
    wrong = predictor.update(.5, 100, arrays["raw"][0], arrays["baseline"][0], arrays["valid_taxels"][0], sensor_id="another_sensor")
    assert not wrong["valid"]
    bad = tmp_path / "legacy.pt"
    torch.save({"model_state": {}}, bad)
    with pytest.raises(ValueError, match="not an old robot policy"):
        TrackingPreview(bad)


def test_workbench_opens_without_devices_and_has_no_robot_output_path():
    code = '''
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget
from phd.ui.touch_tracking_window import TouchTrackingWindow
from phd.dependence.sensor_data import SensorDataBuffer
from phd.dependence.touch_tracking_data import load_trial
from unittest.mock import patch
app = QApplication([])
owner = QWidget()
owner.sensor_functions = None
owner._is_shutting_down = False
window = TouchTrackingWindow(owner)
assert window.tabs.count() == 4
assert 'torch' not in sys.modules
window._toggle_record()
assert window.recorder is None
assert 'sensor' in window.status.text().lower()
with tempfile.TemporaryDirectory(prefix='touch-ui-test-') as folder:
    data = SensorDataBuffer(2, 3, 1)
    data.getCal(np.full((2, 3), 100.))
    def frame(value):
        data.getRaw(np.full((2, 3), value))
        data.calDiff()
        data.calDiffPer()
        data.getWin(1)
    frame(100.)
    owner.sensor_functions = SimpleNamespace(_data=data, is_connected=True,
        current_reorder_mode='native', _primary_sensor_port='fake-test-only')
    window.record_mode.setCurrentIndex(1)  # Preserve independent-reference workflow.
    window.data_root.setText(folder)
    window.capture_camera.setChecked(False)
    window._toggle_record()
    assert window.recorder is not None, window.status.text()
    trial_path = window.recorder.path
    for value in (99., 98., 97.):
        frame(value)
    window._stop_record()
    window._finalizing[0].result(timeout=5)
    window._tick()
    meta, arrays = load_trial(trial_path)
    assert meta['status'] == 'complete'
    assert arrays['raw'].shape == (3, 2, 3)
    window._toggle_record()
    assert window.recorder is not None, window.status.text()
    recorder = window.recorder
    owner.sensor_functions.current_reorder_mode = 'changed'
    window._record_tick()
    assert window.recorder is None
    window._finalizing[0].result(timeout=5)
    assert 'taxel ordering changed' in recorder.error
    window._tick()
    window.record_mode.setCurrentIndex(0)
    window.data_root.setText(folder)
    window.protocol_confirm.setChecked(True)
    owner.camera_preview_window = SimpleNamespace(color_snapshot=lambda *args: (_ for _ in ()).throw(AssertionError('Camera was accessed')))
    frame(100.)
    window._toggle_record()
    assert window.recorder is not None, window.status.text()
    assert window.recorder.manifest['capture_camera'] is False
    assert not window.record_mode.isEnabled()
    path = window.recorder.path
    with patch('phd.ui.touch_tracking_window.time.monotonic', return_value=window.recorder.started+6):
        window._mark_protocol('touch')
    frame(99.)
    frame(98.)
    window._stop_record()
    window._finalizing[0].result(timeout=5)
    window._tick()
    import json
    protocol = json.loads((path/'camera_free_protocol.json').read_text())
    assert protocol['complete'] and protocol['events'][-1]['state'] == 'touch'
    assert not (path/'reference.json').exists()
    assert not list((path/'camera').glob('*.jpg'))
    window.preview_hands_away.setChecked(True)
    frame(100.)
    window._toggle_preview()
    assert window._preview_active, window.status.text()
    window._pending[2].result(timeout=5)
    window._preview_tick()
    frame(100.)
    window._preview_tick()
    if window._pending:
        window._pending[2].result(timeout=5)
        window._preview_tick()
    assert 'AWAY' in window.preview_status.text()
    assert 'torch' not in sys.modules
    window._stop_preview()
window.shutdown()
'''
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

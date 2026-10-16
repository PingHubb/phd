"""Camera-free pipeline tests. Synthetic success is not physical accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from phd.dependence.touch_tracking_camera_free import (
    CameraFreePreview, CameraFreeTracker, PROTOCOL_SCHEMA, SUPERVISION,
    prepare_camera_free_trial, validate_protocol,
)
from phd.dependence.touch_tracking_data import TrialRecorder, dataset_report, load_trial, save_json
from phd.script.train_touch_tracking import train


SHAPE = (6, 8)
BASE = np.full(SHAPE, 10000.)
MASK = np.ones(SHAPE, bool)


def warm(tracker):
    for i in range(261):
        result = tracker.update(i*.02, i+1, BASE, BASE, MASK, sensor_id="s")
    assert tracker.offset is not None
    return result


def bump(x=.4, y=.45, sign=-1):
    yy, xx = np.meshgrid((np.arange(SHAPE[0])+.5)/SHAPE[0], (np.arange(SHAPE[1])+.5)/SHAPE[1], indexing="ij")
    return BASE + sign*400*np.exp(-((xx-x)**2+(yy-y)**2)/.008)


def protocol():
    return {"schema": PROTOCOL_SCHEMA, "operator_confirmed": True, "complete": True,
            "events": [{"t": 0., "state": "no_touch"}, {"t": 6., "state": "touch"},
                       {"t": 10., "state": "unknown"}, {"t": 11., "state": "no_touch"}]}


def make_trial(root, session="a", seed=0):
    rng = np.random.default_rng(seed)
    recorder = TrialRecorder(root, {"participant_id": "p1", "session_id": session,
        "sensor_id": "s", "sensor_shape": list(SHAPE), "surface_size_mm": [200., 150.],
        "supervision": SUPERVISION, "reorder_mode": "native", "synthetic_test_data": True}, queue_size=1000)
    for i in range(650):
        t = i*.02
        raw = bump(x=.25+.075*(t-6)) if 6 <= t < 10 else BASE.copy()
        raw = raw + rng.normal(0, .1, SHAPE)
        assert recorder.sensor_frame(recorder.started+t, i+1, raw, BASE, MASK)
    recorder.close()
    save_json(recorder.path / "camera_free_protocol.json", protocol())
    return recorder.path


@pytest.mark.parametrize("sign", [-1, 1])
def test_tracker_accepts_both_polarities_and_stops_on_release(sign):
    tracker = CameraFreeTracker(SHAPE, "s")
    idle = warm(tracker)
    assert idle["valid"] and idle["xy"] is None
    for i in range(261, 300):
        result = tracker.update(i*.02, i+1, bump(sign=sign), BASE, MASK, sensor_id="s")
    assert result["valid"] and result["contact"]
    np.testing.assert_allclose(result["xy"], [.4, .45], atol=.04)
    assert np.linalg.norm(result["velocity"]) == 0
    result = tracker.update(6., 301, BASE, BASE, MASK, sensor_id="s")
    assert result["xy"] is None and result["velocity"] == [0., 0.]
    assert "contact_probability" not in result


@pytest.mark.parametrize("kind", ["row", "col", "global", "two_fingers", "spike"])
def test_artifacts_and_multiple_regions_do_not_create_motion(kind):
    tracker = CameraFreeTracker(SHAPE, "s")
    warm(tracker)
    raw = BASE.copy()
    if kind == "row":
        raw[2, :] -= 500
    elif kind == "col":
        raw[:, 3] -= 500
    elif kind == "global":
        raw -= 500
    elif kind == "two_fingers":
        raw[1, 1] -= 500
        raw[4, 6] -= 500
    else:
        raw[2, 3] -= 500
    result = tracker.update(5.22, 262, raw, BASE, MASK, sensor_id="s")
    assert not result["valid"] and result["xy"] is None
    assert result["velocity"] == [0., 0.]
    result = tracker.update(5.24, 263, BASE, BASE, MASK, sensor_id="s")
    assert result["xy"] is None


def test_recalibration_fault_never_silently_learns_a_touch():
    tracker = CameraFreeTracker(SHAPE, "s")
    warm(tracker)
    result = tracker.update(5.22, 262, bump(), BASE+1, MASK, sensor_id="s")
    assert not result["valid"] and "restart" in result["state"]
    for i in range(263, 550):
        result = tracker.update(i*.02, i+1, bump(), BASE+1, MASK, sensor_id="s")
    assert not result["valid"] and result["xy"] is None


def test_gap_and_jump_clear_motion_instead_of_connecting_distant_touches():
    tracker = CameraFreeTracker(SHAPE, "s")
    warm(tracker)
    for i in range(261, 290):
        result = tracker.update(i*.02, i+1, bump(x=.25), BASE, MASK, sensor_id="s")
    assert result["xy"] is not None
    jumped = tracker.update(5.8, 291, bump(x=.75), BASE, MASK, sensor_id="s")
    assert jumped["xy"] is None and jumped["velocity"] == [0., 0.]
    after_gap = tracker.update(7., 292, bump(x=.75), BASE, MASK, sensor_id="s")
    assert after_gap["xy"] is None and after_gap["velocity"] == [0., 0.]


def test_preview_uses_actual_source_timing_and_rejects_reset():
    preview = CameraFreePreview(SHAPE, "s")
    first = preview.update_many([(0., 1, BASE, BASE, MASK)], sensor_id="s")
    between = preview.update_many([(.005, 2, BASE, BASE, MASK)], sensor_id="s")
    assert first == between  # No new 50-Hz model frame: retain result, no fabricated frame.
    reset = preview.update_many([(.01, 1, BASE, BASE, MASK)], sensor_id="s")
    assert not reset["valid"] and "reset" in reset["state"]


def test_noisy_cells_do_not_dominate_quiet_localized_signal():
    tracker = CameraFreeTracker(SHAPE, "s")
    rng = np.random.default_rng(9)
    for i in range(270):
        raw = BASE.copy()
        raw[0, 0] += rng.normal(0, 200)
        result = tracker.update(i*.02, i+1, raw, BASE, MASK, sensor_id="s")
    assert tracker.offset is not None
    for i in range(270, 300):
        raw = bump()
        raw[0, 0] += 200
        result = tracker.update(i*.02, i+1, raw, BASE, MASK, sensor_id="s")
    assert result["valid"]
    np.testing.assert_allclose(result["xy"], [.4, .45], atol=.04)


def test_camera_free_preparation_does_not_create_independent_references(tmp_path):
    path = make_trial(tmp_path)
    trial = prepare_camera_free_trial(path)
    assert not (path / "reference.json").exists()
    labels = trial["labels"]
    assert labels["position_valid"].sum() > 100
    assert np.all(labels["contact"][(trial["t"] > 10.36) & (trial["t"] < 10.64)] == -1)
    assert np.all(labels["contact"][np.abs(trial["t"]-6.) < .34] == -1)
    assert np.max(labels["velocity"][:, 0]) > .02
    report, prepared = dataset_report(tmp_path)
    assert report["invalid_trials"] == 1 and not prepared
    report, prepared = dataset_report(tmp_path, supervision="camera_free")
    assert report["ready_trials"] == 1 and prepared[0]["supervision"] == SUPERVISION


def test_protocol_rejects_unconfirmed_short_initial_or_invalid_intervals():
    p = protocol()
    p["operator_confirmed"] = False
    with pytest.raises(ValueError, match="Confirm"):
        validate_protocol(p)
    p = protocol()
    p["events"][1]["t"] = 2.
    with pytest.raises(ValueError, match="5.35"):
        validate_protocol(p)
    p = protocol()
    p["events"][2]["state"] = "fake_ground_truth"
    with pytest.raises(ValueError, match="invalid"):
        validate_protocol(p)


def test_offline_and_live_direct_position_targets_are_identical(tmp_path):
    path = make_trial(tmp_path)
    prepared = prepare_camera_free_trial(path)
    _, arrays = load_trial(path)
    preview = CameraFreePreview(SHAPE, "s", "native")
    for k in range(len(arrays["t"])):
        result = preview.update_many([(arrays["t"][k], k+1, arrays["raw"][k], arrays["baseline"][k], arrays["valid_taxels"][k])], sensor_id="s")
        if prepared["labels"]["position_valid"][k]:
            np.testing.assert_allclose(result["xy"], prepared["labels"]["xy"][k], atol=1e-7)
            np.testing.assert_allclose(result["velocity"], prepared["labels"]["velocity"][k], atol=1e-7)


def test_camera_free_end_to_end_training_has_honest_metrics_and_guarded_preview(tmp_path):
    from phd.dependence.touch_tracking_model import TrackingPreview
    root = tmp_path / "data"
    for i in range(3):
        make_trial(root, session=f"day{i}", seed=i)
    args = argparse.Namespace(root=root, output=tmp_path / "models", group_by="session", seed=42,
        epochs=1, batch_size=64, seq_len=8, hz=50., max_gap=.1, label_gap=.12, threads=1,
        device="cpu", architectures="gru,tcn", lr=.001, patience=2, no_augment=True, supervision="camera_free")
    out = train(args)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["supervision"] == "camera_free"
    assert summary["independent_physical_accuracy_measured"] is False
    assert "position_rmse_mm" not in summary["test"]
    assert "sensor_target_agreement_position_rmse_mm" in summary["test"]
    assert "estimated_target_x" in (out / "test" / "predictions.csv").read_text().splitlines()[0]
    preview = TrackingPreview(out / "selected_model.pt")
    for i in range(20):
        result = preview.update(i*.02, i+1, BASE, BASE, MASK, sensor_id="s")
    assert not result["valid"] and "AWAY" in result["state"]
    for i in range(20, 270):
        result = preview.update(i*.02, i+1, BASE, BASE, MASK, sensor_id="s")
    assert result["valid"] and result["xy"] is None
    result = preview.update(5.4, 271, bump(), BASE+1, MASK, sensor_id="s")
    assert not result["valid"]

"""Synthetic fixtures verify plumbing, never research/model performance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from phd.dependence.intention_data import (
    CLASSES, LAYOUTS, SCHEMA, IntentionRecorder, align_declarations,
    dataset_report, intention_indices, prepare_intention_trial, validate_goals,
)
from phd.dependence.touch_tracking_data import save_json, split_trials
from phd.script.train_intention import calibrate, metrics, probabilities, temporal_metrics

ROOT = Path(__file__).resolve().parents[1]


def make_trial(root, participant="P01", seed=1, layout=0, frames=160):
    rec = IntentionRecorder(root, {
        "participant_id": participant, "session_id": "session1", "sensor_id": "fabric01",
        "sensor_shape": [4, 5], "sensor_model": "test4x5", "reorder_mode": "native",
        "orientation_confirmed": True, "synthetic_fixture": True,
    }, LAYOUTS[layout], queue_size=frames+5, chunk_size=32)
    for t, goal in ((.1, 0), (2., 1), (4., 2), (6., 3)):
        rec.declare(goal, rec.started+t)
    rng = np.random.default_rng(seed)
    base = np.full((4, 5), 10000, np.float32)
    for k in range(frames):
        raw = base + rng.normal(0, 2, base.shape).astype(np.float32)
        raw[k//40 % 4, k//10 % 5] += 20
        assert rec.sensor_frame(rec.started+k/20, k, raw, base)
    rec.close()
    assert rec.error is None
    return rec.path


def test_protocol_requires_registration_and_distinct_goals(tmp_path):
    with pytest.raises(ValueError, match="Confirm"):
        IntentionRecorder(tmp_path, {}, LAYOUTS[0])
    for invalid in ([[0., 0.]]*3, [[float("nan"), 0.]]*3, [[2., 0.], [.5, .5], [0., 1.]]):
        with pytest.raises(ValueError):
            validate_goals(invalid)
    assert len(LAYOUTS) == 6
    assert validate_goals(LAYOUTS[0]).shape == (3, 2)


def test_label_boundaries_are_unknown_not_interpolated():
    events = [{"t": 0., "goal": -1}, {"t": 1., "goal": 1}, {"t": 3., "goal": 2}]
    result = align_declarations(events, [0., .9, 1.1, 1.3, 2.5, 2.9, 3.1, 3.3], margin=.25)
    assert result.tolist() == [-1, -1, -1, 1, 1, -1, -1, 2]
    with pytest.raises(ValueError):
        align_declarations([{"t": 0., "goal": 1.5}], [0.])


def test_recording_roundtrip_and_label_feature_separation(tmp_path):
    path = make_trial(tmp_path)
    first = prepare_intention_trial(path)
    assert np.unique(first["labels"]).tolist() == [-1, 0, 1, 2, 3]
    ends = intention_indices(first, 20)
    assert len(ends) > 80
    assert (first["labels"][ends] >= 0).all()
    protocol = json.loads((path/"intention.json").read_text())
    protocol["events"][2]["goal"] = 3
    save_json(path/"intention.json", protocol)
    second = prepare_intention_trial(path)
    np.testing.assert_array_equal(first["frames"], second["frames"])
    np.testing.assert_array_equal(first["goals"], second["goals"])
    assert np.any(first["labels"] != second["labels"])
    assert first["data_hash"] == second["data_hash"]
    assert first["protocol_hash"] != second["protocol_hash"]


def test_failed_trials_and_legacy_data_are_refused(tmp_path):
    path = make_trial(tmp_path)
    protocol = json.loads((path/"intention.json").read_text())
    protocol["status"] = "failed"
    save_json(path/"intention.json", protocol)
    report, trials = dataset_report(tmp_path)
    assert not trials and report["invalid_trials"] == 1
    assert "Incomplete" in report["trials"][0]["errors"][0]
    protocol["status"] = "complete"
    protocol["schema"] = "legacy"
    save_json(path/"intention.json", protocol)
    with pytest.raises(ValueError, match="Not a human"):
        prepare_intention_trial(path)


def test_group_split_no_participant_or_duplicate_leakage(tmp_path):
    trials = [prepare_intention_trial(make_trial(tmp_path, f"P{i}", seed=i)) for i in range(3)]
    splits = split_trials(trials, group_by="participant")
    groups = [{t["metadata"]["participant_id"] for t in ts} for ts in splits.values()]
    assert len(set.union(*groups)) == 3
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
    trials[1]["data_hash"] = trials[0]["data_hash"]
    with pytest.raises(ValueError, match="Duplicate"):
        split_trials(trials, group_by="participant")
    with pytest.raises(ValueError, match="at least 3"):
        split_trials(trials[:2], group_by="participant")


def test_temporal_model_is_target_permutation_equivariant():
    torch = pytest.importorskip("torch")
    from phd.dependence.intention_model import IntentionNet
    torch.set_num_threads(2)
    x = torch.randn(2, 8, 4, 4, 5)
    goals = torch.tensor(np.repeat(np.asarray(LAYOUTS[0], np.float32)[None], 2, axis=0))
    model = IntentionNet(4, 5).eval()
    with torch.inference_mode():
        a = model(x, goals)
        b = model(x, goals[:, [2, 0, 1]])
    torch.testing.assert_close(b, a[:, [0, 3, 1, 2]])
    # Context-only must not accidentally consume sensor input.
    model = IntentionNet(4, 5, mode="context_only").eval()
    torch.testing.assert_close(model(x, goals), model(x*100, goals))
    model = IntentionNet(4, 5, mode="tactile_only").eval()
    torch.testing.assert_close(model(x, goals), model(x, goals.flip(1)))


def test_metrics_calibration_and_withholding():
    logits = np.zeros((40, 4))
    y = np.arange(40) % 4
    temp, threshold = calibrate(logits, y)
    assert temp > 0 and threshold > 1
    result = metrics(y, probabilities(logits, temp), threshold)
    assert result["coverage"] == 0 and result["accepted_accuracy"] is None
    assert result["accuracy"] == .25
    perfect = np.eye(4)[y]*10
    _, threshold = calibrate(perfect, y)
    assert threshold <= 1


def test_switch_misses_remain_visible(tmp_path):
    from types import SimpleNamespace
    trial = prepare_intention_trial(make_trial(tmp_path))
    ends = intention_indices(trial, 4)
    ds = SimpleNamespace(trials=[trial], indices=[(0, k) for k in ends])
    p = np.eye(4)[trial["labels"][ends]]
    result = temporal_metrics(ds, p, 1.01)
    assert result["switch_count"] == 2
    assert result["switch_detected"] == 0
    assert result["median_detected_switch_latency_sec"] is None
    result = temporal_metrics(ds, p, .9)
    assert result["switch_detected"] == 2
    assert .2 <= result["median_detected_switch_latency_sec"] < 1


def test_synthetic_train_export_and_live_guards(tmp_path):
    torch = pytest.importorskip("torch")
    from phd.script.train_intention import train
    from phd.dependence.intention_model import IntentionPreview
    root = tmp_path/"data"
    for i in range(3):
        make_trial(root, f"P{i}", seed=i, layout=i)
    args = argparse.Namespace(root=str(root), output=str(tmp_path/"models"), command="train",
        epochs=1, batch_size=32, seq_len=4, stride=2, seed=42, group_by="participant",
        hz=20., max_gap=.15, boundary_margin=.25)
    out = train(args)
    summary = json.loads((out/"summary.json").read_text())
    assert summary["status"] == "complete"
    assert set(summary["models"]) == {"fusion", "context_only", "tactile_only"}
    checkpoint = torch.load(out/"intention_preview.pt", weights_only=True)
    assert checkpoint["schema"] == SCHEMA
    assert checkpoint["config"]["classes"] == CLASSES
    preview = IntentionPreview(out/"intention_preview.pt")
    base = np.full((4, 5), 10000, np.float32)
    frames = [(1.+i*.05, i, base+i, base, np.ones_like(base, bool)) for i in range(5)]
    kwargs = dict(sensor_id="fabric01", sensor_model="test4x5", reorder_mode="native", goals=LAYOUTS[0])
    result = preview.update_many(frames, **kwargs)
    assert result["valid"] and len(result["probabilities"]) == 4
    assert result["robot_commands_enabled"] is False
    result = preview.update_many([(1.3, 5, base, base+1, np.ones_like(base, bool))], **kwargs)
    assert not result["valid"] and "calibration" in result["reason"]
    assert len(preview.history) == 0
    result = preview.update_many(frames, **{**kwargs, "sensor_id": "different-unit"})
    assert not result["valid"] and "identity" in result["reason"]
    preview.update_many(frames, **kwargs)
    result = preview.update_many([(5., 9, base, base, np.ones_like(base, bool))], **kwargs)
    assert not result["valid"] and "Warming" in result["reason"]


def test_workbench_opens_without_hardware_or_torch_and_handles_capture(tmp_path):
    script = r'''
import sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QTimer
from phd.ui.intention_window import IntentionWindow
app = QApplication([])
owner = QWidget()
window = IntentionWindow(owner)
window.show()
app.processEvents()
assert "torch" not in sys.modules
assert window.tabs.count() == 3
assert not window.cancel_button.isEnabled()
window._toggle_record()
assert "stream the sensor" in window.status.text()
assert window.mode == "idle"
assert not window.prediction.text().startswith("Predicted")
class Data:
    rawData = np.full((4,5), 10000, np.float32)
    calData = rawData.copy()
    frame_monotonic_sec = time.monotonic()
    frame_sequence = 0
    listeners = []
    def add_frame_listener(self, fn): self.listeners.append(fn)
    def remove_frame_listener(self, fn): self.listeners.remove(fn)
data = Data()
owner.sensor_functions = SimpleNamespace(_data=data, is_connected=True, _primary_sensor_port="fake",
    current_model_name="test4x5", current_reorder_mode="native", cell_zero_mask=None)
window.data_root.setText(sys.argv[1])
window.participant.setText("P01")
window.sensor_id.setText("fabric01")
window.orientation.setChecked(True)
window._toggle_record()
assert window.mode == "record", window.status.text()
assert not window.layout_combo.isEnabled()
window._declare(1)
for i in range(4):
    data.frame_sequence += 1
    data.frame_monotonic_sec = time.monotonic()
    data.rawData = data.rawData+1
    for fn in list(data.listeners): fn(data)
window._stop_record()
deadline = time.monotonic()+5
while window._future is not None and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(.01)
assert window.mode == "idle", window.status.text()
assert not data.listeners
assert len(list(Path(sys.argv[1]).glob("*/intention.json"))) == 1
window.shutdown()
assert window._closing
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=ROOT,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT)),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr


def test_real_ai_action_buttons_open_the_correct_intention_pages():
    # Construct the actual AI widgets without starting device/core services.
    script = r'''
from types import MethodType
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from phd.ui.ui_ping import UI
from phd.ui.ui_ping_ai_controls import AiControlsMixin
app = QApplication([])
owner = QWidget()
owner.widget_func = QWidget(owner)
for name in dir(UI):
    if name.startswith("_") and not name.startswith("__") and callable(getattr(UI, name)):
        setattr(owner, name, lambda *args, **kwargs: None)
owner._scrollable = UI._scrollable
owner._open_intention_page = MethodType(AiControlsMixin._open_intention_page, owner)
owner._shutdown_intention_workbench = MethodType(AiControlsMixin._shutdown_intention_workbench, owner)
owner.AI_PROXIMITY_ENVIRONMENT_TRIAL_COUNT = 10
owner.AI_PROXIMITY_ENVIRONMENT_TRIAL_SECONDS = 60
UI._build_control_workspace(owner)
UI.setup_tab3(owner, QVBoxLayout(owner.widget_func))
owner.ai_data_sections.setCurrentIndex(owner.ai_intention_section_index)
buttons = owner.intention_actions.findChildren(QPushButton)
assert len(buttons) == 3
for index, button in enumerate(buttons):
    button.click()
    app.processEvents()
    assert owner.intention_window.tabs.currentIndex() == index
    assert owner.intention_window.mode == "idle"
    assert owner.intention_window.recorder is None
owner._shutdown_intention_workbench()
assert owner.intention_window._closing
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT)),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr

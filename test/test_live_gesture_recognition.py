"""Fresh-frame boundaries and safe sensor-only UI inference."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from phd.dependence.live_gesture_recognition import GestureTrialSegmenter


def frame(touch=False):
    values = np.zeros((8, 8), np.float32)
    if touch:
        values[2, 5] = -10
    return values


def feed(segmenter, seq, touch=False):
    return segmenter.feed(seq, seq * .07, frame(touch), frame(touch))


def test_release_arms_once_and_preserves_native_layout_for_repeated_gestures():
    segmenter = GestureTrialSegmenter()
    for seq in range(3):
        assert feed(segmenter, seq, touch=True) is None
    assert segmenter.state == "release"
    for seq in range(3, 6):
        assert feed(segmenter, seq) is None
    assert segmenter.state == "ready"
    for start in (6, 11):
        assert feed(segmenter, start, True) is None
        assert feed(segmenter, start + 1, True) is None
        assert feed(segmenter, start + 2) is None
        assert feed(segmenter, start + 3) is None
        raw, avg, times = feed(segmenter, start + 4)
        assert raw.shape == (5, 8, 8)
        assert raw[0, 2, 5] == -10 and raw[0, 5, 2] == 0
        np.testing.assert_array_equal(raw, avg)
        assert times[0] == 0 and np.all(np.diff(times) > 0)
        assert segmenter.state == "ready"


def test_duplicate_frames_cannot_finish_a_trial_and_one_frame_glitch_is_ignored():
    segmenter = GestureTrialSegmenter()
    for seq in range(3):
        feed(segmenter, seq)
    feed(segmenter, 3, True)
    feed(segmenter, 4)
    for _ in range(50):
        assert feed(segmenter, 4) is None
    assert segmenter.state == "recording"
    feed(segmenter, 5)
    assert feed(segmenter, 6) is None
    assert segmenter.state == "ready"


@pytest.mark.parametrize("kind", ["gap", "reset", "nan", "shape", "timestamp"])
def test_bad_or_interrupted_stream_is_not_mistaken_for_release(kind):
    segmenter = GestureTrialSegmenter()
    for seq in range(5):
        feed(segmenter, seq, seq >= 3)
    seq, stamp, raw = 5, .35, frame()
    if kind == "gap":
        stamp = 2
    elif kind == "reset":
        seq = 0
    elif kind == "nan":
        raw[0, 0] = np.nan
    elif kind == "shape":
        raw = np.zeros((5, 5))
    else:
        stamp = 0
    with pytest.raises(ValueError):
        segmenter.feed(seq, stamp, raw, raw)


def test_overlong_touch_is_discarded_until_confirmed_release():
    segmenter = GestureTrialSegmenter()
    segmenter.MAX_FRAMES = 3
    for seq in range(3):
        feed(segmenter, seq)
    for seq in range(3, 9):
        assert feed(segmenter, seq, True) is None
    assert segmenter.state == "release"
    assert not segmenter.frames
    for seq in range(9, 12):
        feed(segmenter, seq)
    assert segmenter.state == "ready"


def run_qt(script):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", script], cwd=root,
                            env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_panel_validates_live_source_stop_late_result_and_offline_replay():
    run_qt('''
import time
from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
from PyQt5.QtWidgets import QApplication
from phd.dependence.sensor_data import SensorDataBuffer
from phd.ui.gesture_recognition_panel import GestureRecognitionPanel
app = QApplication([])
data = SensorDataBuffer(8, 8)
data.frame_monotonic_sec = time.monotonic()
sensor = SimpleNamespace(is_connected=True, _data=data, _primary_sensor_port="actual-stream")
panel = GestureRecognitionPanel(lambda: sensor)
class Predictor:
    bundle = {"sensor_shape": [8,8], "preprocess": {
        "touch_threshold_percent": -3, "minimum_touch_frames": 2}}
panel.predictor = Predictor()
panel._binding = panel._signature(sensor, data)
panel._calibration = data.calData.copy()
panel._attach()
assert panel.active and len(data._frame_listeners) == 1
assert "actual-stream" in panel.source_label.text()
sensor.selected_port = "wrong-highlighted-port"
panel._validate_live()
assert "wrong-highlighted-port" not in panel.source_label.text()
data.calData[0,0] = 5
panel._tick()
assert not panel.active and not data._frame_listeners
assert "calibration changed" in panel.status_detail.text()
future = Future()
future.set_running_or_notify_cancel()
panel._future, panel._operation = future, "load"
panel.stop()
future.set_result(Predictor())
panel._tick()
assert not panel.active and not data._frame_listeners
sensor.is_connected = False
panel.start()
assert panel.status.text() == "Cannot start"
sensor.is_connected = True
data.frame_monotonic_sec = time.monotonic() - 3
panel.start()
assert "fresh" in panel.status_detail.text()
panel.shutdown()
assert not panel.start_button.isEnabled()
''')


def test_real_checkpoint_replay_matches_file_prediction_without_sensor():
    root = Path(__file__).resolve().parents[1] / "phd/resource/ai"
    if not (root / "models/robot_dog_gestures/20261020_10class_v1/best_gesture_model.pt").exists():
        pytest.skip("Local trained artifact not installed")
    run_qt('''
import time
from pathlib import Path
import torch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from phd.ui.gesture_recognition_panel import GestureRecognitionPanel
from phd.dependence.robot_dog_gesture_model import GesturePredictor
torch.set_num_threads(2)
app = QApplication([])
panel = GestureRecognitionPanel(lambda: None)
trial = sorted(Path("phd/resource/ai/data/tactile_recordings/session_robot_dog_swipe_up").glob("*.npz"))[-1]
panel.test_saved_trial(trial)
deadline = time.monotonic() + 20
while panel._future is not None and time.monotonic() < deadline:
    QTest.qWait(20)
assert panel.last_result, panel.status_detail.text()
expected = GesturePredictor(panel.checkpoint).predict_file(trial)
assert panel.last_result == expected
assert panel.last_result["robot_command_sent"] is False
assert not panel.active and panel.status.text() == "Replay complete"
assert "Saved trial" in panel.confidence_label.text()
assert panel.history.count() == 1
panel.shutdown()
''')


def test_live_stream_changes_and_backlog_discard_pending_results():
    run_qt('''
import queue
import time
from concurrent.futures import Future
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication
from phd.dependence.sensor_data import SensorDataBuffer
from phd.ui.gesture_recognition_panel import GestureRecognitionPanel
app = QApplication([])
for failure in ("disconnect", "stale", "new_source", "new_data", "averaging", "reorder", "backlog"):
    data = SensorDataBuffer(8, 8)
    data.frame_monotonic_sec = time.monotonic()
    sensor = SimpleNamespace(is_connected=True, _data=data, _primary_sensor_port="stream")
    panel = GestureRecognitionPanel(lambda: sensor)
    panel.predictor = SimpleNamespace(bundle={"sensor_shape": [8,8], "preprocess": {
        "touch_threshold_percent": -3, "minimum_touch_frames": 2}})
    panel._binding = panel._signature(sensor, data)
    panel._calibration = data.calData.copy()
    panel._attach()
    future = Future()
    future.set_result({"gesture": "must_not_be_shown"})
    panel._future, panel._operation = future, "predict"
    if failure == "disconnect":
        sensor.is_connected = False
    elif failure == "stale":
        data.frame_monotonic_sec = time.monotonic() - 2
    elif failure == "new_source":
        sensor._primary_sensor_port = "other-stream"
    elif failure == "new_data":
        sensor._data = SensorDataBuffer(8, 8)
        sensor._data.frame_monotonic_sec = time.monotonic()
    elif failure == "averaging":
        data.windowSize += 1
    elif failure == "reorder":
        sensor.current_effective_reorder_logic = "rotate_180"
    else:
        panel._queue = queue.Queue(maxsize=1)
        panel._on_frame(data)
        panel._on_frame(data)
    panel._tick()
    assert panel.last_result is None, failure
    assert not panel.active and not data._frame_listeners, failure
    assert panel.status.state() == "error", failure
    panel.shutdown()
# Wrong grid must not subscribe to frames or request robot/force availability.
data = SensorDataBuffer(5, 5)
data.frame_monotonic_sec = time.monotonic()
sensor = SimpleNamespace(is_connected=True, _data=data)
panel = GestureRecognitionPanel(lambda: sensor)
panel._binding = panel._signature(sensor, data)
panel._calibration = data.calData.copy()
panel.predictor = SimpleNamespace(bundle={"sensor_shape": [8,8]})
try:
    panel._attach()
    raise AssertionError("Wrong sensor dimensions accepted")
except ValueError as exc:
    assert "8 × 8" in str(exc)
assert not panel.active and not data._frame_listeners
panel.shutdown()
''')

"""Read-only metadata follows the model, not its filename or a gesture preset."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from phd.ui.touch_policy_model_details import describe_touch_checkpoint, read_touch_checkpoint_details


def bundle(labels, architecture="cnn_tcn"):
    return {"format": "pinglab_touch_gesture_policy_v1", "class_names": labels,
        "members": [{"config": {"architecture": architecture, "classes": len(labels)}}],
        "sensor_shape": [10, 10], "preprocess": {"prediction_unit": "recent_touch_window", "window_duration_sec": .2}}


@pytest.mark.parametrize("labels", [["swipe_up", "swipe_down", "swipe_left", "swipe_right"],
                                    ["swipe_up", "swipe_down", "swipe_left", "swipe_right", "two_finger_push", "two_finger_pull"],
                                    ["up", "down", "left", "right", "push", "pull"]])
@pytest.mark.parametrize("architecture", ["cnn_gru", "cnn_tcn"])
def test_preview_reads_architecture_and_only_the_declared_labels(tmp_path, labels, architecture):
    path = tmp_path / "no_model_information_in_filename.pt"
    torch.save(bundle(labels, architecture), path)
    text = read_touch_checkpoint_details(path)
    assert architecture.replace("_", "–").upper() in text
    assert f"{len(labels)} gestures" in text
    assert ("Push" in text) == ("push" in labels)
    assert ("Two finger push" in text) == ("two_finger_push" in labels)
    assert ("Two finger pull" in text) == ("two_finger_pull" in labels)
    assert "latest 200 ms" in text


def test_velocity_policy_is_not_given_swipe_labels():
    text = describe_touch_checkpoint({"model_class": "TactileCNNGRUPolicy"})
    assert "velocity and control mode" in text
    assert "4 gestures" not in text and "6 gestures" not in text


def test_single_touch_preview_is_not_described_as_continuous():
    value = bundle(["swipe_up", "swipe_down", "swipe_left", "swipe_right", "two_finger_push", "two_finger_pull"])
    value["preprocess"] = {"prediction_unit": "available_touch_prefix", "maximum_duration_sec": 2.0}
    text = describe_touch_checkpoint(value)
    assert "Input: current touch history" in text
    assert "Single-gesture mode: lift your finger before the next gesture." in text
    assert "200 ms" not in text


def test_bad_output_count_and_wrong_task_are_reported():
    value = bundle(["up", "down"])
    value["members"][0]["config"]["classes"] = 6
    with pytest.raises(ValueError, match="Output size"):
        describe_touch_checkpoint(value)
    with pytest.raises(ValueError, match="different-task"):
        describe_touch_checkpoint({"format": "pinglab_robot_dog_gesture_v1"})


def test_async_preview_discards_obsolete_reads_and_uses_loaded_model():
    script = r'''
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from PyQt5.QtWidgets import QApplication
from phd.ui.touch_policy_model_details import TouchPolicyModelDetails
app = QApplication([])
with TemporaryDirectory() as folder:
    a,b = Path(folder)/'a.pt', Path(folder)/'b.pt'
    a.touch(); b.touch()
    widget = TouchPolicyModelDetails()
    first, second = Future(), Future()
    first.set_running_or_notify_cancel()
    executor = widget._executor = Mock()
    executor.submit.side_effect = [first, second]
    widget.set_source(str(a))
    widget.set_source(str(b))
    first.set_result('OLD 6 gestures')
    widget._poll()
    assert 'OLD' not in widget.label.text()
    second.set_result('NEW 4 gestures')
    widget._poll()
    assert widget.label.text() == 'NEW 4 gestures'
    widget.set_source(str(b))
    assert executor.submit.call_count == 2
    widget.set_source(str(b), loaded_bundle={'class_names': ['stale']}, loaded_kind='legacy_transformer')
    assert 'CNN–Transformer' in widget.label.text() and 'stale' not in widget.label.text()
    assert executor.submit.call_count == 2
    widget.shutdown()
    widget.set_source(str(a))
    assert executor.submit.call_count == 2
    executor.shutdown.assert_called_once()
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr

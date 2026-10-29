"""Four-swipe data isolation and backwards-compatible safe model loading."""
import json

import numpy as np
import pytest
import torch

from phd.dependence.robot_dog_gesture_model import GestureClassifier
from phd.dependence.touch_gesture_policy import (
    CLASS_NAMES, SWIPE_CLASS_NAMES, SIX_GESTURE_CLASS_NAMES, CONTINUOUS_PREPROCESS_CONFIG, FORMAT,
    TouchGesturePolicy, policy_directions,
)
from phd.script.training.train_touch_gesture_policy import discover, metrics


def bundle_for(architecture, names):
    config = dict(architecture=architecture, classes=len(names), width=96, dropout=.2, sequence_length=32)
    return {"format": FORMAT, "class_names": list(names), "sensor_shape": [10, 10],
        "members": [{"config": config, "state_dict": GestureClassifier(**config).state_dict()}],
        "normalization": {"mean": [0.] * 3, "std": [1.] * 3}, "temperature": 1.,
        "preprocess": dict(CONTINUOUS_PREPROCESS_CONFIG),
        "control": {"directions": policy_directions(names), "base_speed_mps": .02,
                    "maximum_speed_mps": .02, "confidence_threshold": .8,
                    "margin_threshold": .15, "acceleration_limit_mps2": .2}}


@pytest.mark.parametrize("architecture", ["cnn_gru", "cnn_tcn"])
@pytest.mark.parametrize("names", [CLASS_NAMES, SWIPE_CLASS_NAMES, SIX_GESTURE_CLASS_NAMES])
def test_supported_checkpoints_keep_label_count_and_normalized_predictions(tmp_path, architecture, names):
    torch.set_num_threads(2)
    path = tmp_path / "model.pt"
    torch.save(bundle_for(architecture, names), path)
    model = TouchGesturePolicy(torch.load(path, weights_only=True)).eval()
    with torch.inference_mode():
        probabilities = model(torch.zeros(2, 32, 3, 10, 10))
    assert probabilities.shape == (2, len(names))
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))


@pytest.mark.parametrize("mutation", ["direction", "labels", "classes", "architecture"])
def test_mismatched_or_unsafe_bundle_is_rejected(mutation):
    bundle = bundle_for("cnn_gru", SWIPE_CLASS_NAMES)
    if mutation == "direction":
        bundle["control"]["directions"]["swipe_up"] = [0, 0, 0, 1, 0, 0]
    elif mutation == "labels":
        bundle["class_names"].reverse()
    elif mutation == "classes":
        bundle["members"][0]["config"]["classes"] = 6
    else:
        bundle["members"][0]["config"]["architecture"] = "cnn_transformer"
    with pytest.raises(ValueError):
        TouchGesturePolicy(bundle)


@pytest.mark.parametrize("names", [SWIPE_CLASS_NAMES, SIX_GESTURE_CLASS_NAMES])
def test_gesture_discovery_excludes_other_folders_and_splits_whole_trials(tmp_path, names):
    for c, label in enumerate(names):
        folder = tmp_path / ("session_10x10_" + label)
        folder.mkdir()
        for i in range(20):
            raw = np.full((15, 10, 10), -8 - (c * 20 + i) / 100, np.float32)
            metadata = {"teacher_layout": "column_major_matrix_view", "sensor_rows": 10, "sensor_cols": 10,
                "sensor_reorder_mode": "row_to_col", "sensor_average_window_size": 3,
                "completion_reason": "touch_released", "started_at": i, "trial_number": i}
            np.savez(folder / f"trial_{i:02}.npz", diffPerData=raw, diffPerDataAve=raw,
                     elapsed_sec=np.arange(15) * .02, metadata_json=json.dumps(metadata))
    (tmp_path / "session_10x10_push").mkdir()
    records, _, splits = discover(tmp_path, names, CONTINUOUS_PREPROCESS_CONFIG)
    assert len(records) == 20 * len(names)
    assert [len(splits[s]) for s in ("train", "validation", "test")] == [n * len(names) for n in (12, 4, 4)]
    assert set(splits["train"]).isdisjoint(splits["validation"] + splits["test"])
    assert set(splits["validation"]).isdisjoint(splits["test"])
    for record in records:
        assert record["label"] in names
        assert record["split"] == ("train" if record["trial_number"] < 12 else
                                   "validation" if record["trial_number"] < 16 else "test")
    first = tmp_path / "session_10x10_swipe_up/trial_00.npz"
    duplicate = tmp_path / "session_10x10_swipe_down/trial_00.npz"
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="Duplicate signal"):
        discover(tmp_path, names, CONTINUOUS_PREPROCESS_CONFIG)


def test_four_class_metrics_are_not_padded_with_absent_push_pull_classes():
    result = metrics(np.eye(4), np.arange(4), SWIPE_CLASS_NAMES)
    assert result["macro_f1"] == 1
    assert len(result["confusion_matrix"]) == 4
    assert [r["gesture"] for r in result["per_class"]] == list(SWIPE_CLASS_NAMES)

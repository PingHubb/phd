"""Classification data are trial-separated and do not use teacher labels."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phd.dependence.robot_dog_gesture_model import (
    ARCHITECTURES, CLASS_NAMES, PREPROCESS, GestureClassifier, GesturePredictor,
    load_trial, prepare_trial,
)
from phd.script.train_robot_dog_gestures import (
    augment_batch, classification_metrics, discover_dataset, split_trials, train,
)


def write_trial(path, shape=(8, 8), trial=1, offset=0.0):
    raw = np.zeros((12, *shape), dtype=np.float32)
    for step in range(8):
        raw[step + 1, step % shape[0], (step // 2) % shape[1]] = -10 - offset
    np.savez_compressed(
        path, diffPerData=raw.transpose(0, 2, 1), diffPerDataAve=raw.transpose(0, 2, 1),
        elapsed_sec=np.arange(12) * 0.04,
        metadata_json=np.array(json.dumps({"teacher_layout": "column_major_matrix_view",
            "sensor_rows": shape[0], "sensor_cols": shape[1], "started_at": 1000 + trial,
            "trial_number": trial, "completion_reason": "touch_released"})),
    )
    return raw


def test_layout_conversion_and_preprocessing_do_not_need_teacher_outputs(tmp_path):
    original = write_trial(tmp_path / "trial.npz", shape=(5, 8))
    raw, avg, times, meta = load_trial(tmp_path / "trial.npz")
    np.testing.assert_array_equal(raw, original)
    x = prepare_trial(raw, avg, times)
    assert x.shape == (32, 3, 5, 8)
    assert np.isfinite(x).all()
    assert np.count_nonzero(x[0, 2]) == 0
    np.testing.assert_allclose(x[1:, 2], np.diff(x[:, 1], axis=0), atol=1e-6)


def test_idle_tails_and_uniform_time_scaling_do_not_change_motion_features(tmp_path):
    write_trial(tmp_path / "trial.npz")
    raw, avg, times, _ = load_trial(tmp_path / "trial.npz")
    expected = prepare_trial(raw, avg, times)
    np.testing.assert_allclose(expected, prepare_trial(raw, avg, times * 3 + 99), atol=1e-6)
    padded = np.pad(raw, ((10, 20), (0, 0), (0, 0)))
    np.testing.assert_allclose(expected, prepare_trial(padded, padded, np.arange(len(padded)) * .04), atol=1e-6)


def test_invalid_and_no_touch_episodes_are_rejected():
    raw = np.zeros((8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="No valid touch"):
        prepare_trial(raw, raw, np.arange(8))
    raw[2, 1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        prepare_trial(raw, raw, np.arange(8))
    raw[:] = -10
    with pytest.raises(ValueError, match="increase"):
        prepare_trial(raw, raw, np.zeros(8))


def test_each_class_split_is_chronological_and_disjoint():
    records = [{"label": label, "started_at": trial, "trial_number": trial, "path": f"{label}/{trial}"}
               for label in CLASS_NAMES for trial in reversed(range(100))]
    split = split_trials(records)
    assert {name: len(indices) for name, indices in split.items()} == {"train": 600, "validation": 200, "test": 200}
    assert not set(split["train"]) & set(split["validation"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["validation"]) & set(split["test"])
    for label in CLASS_NAMES:
        for name, bounds in (("train", (0, 59)), ("validation", (60, 79)), ("test", (80, 99))):
            numbers = [records[i]["trial_number"] for i in split[name] if records[i]["label"] == label]
            assert (min(numbers), max(numbers)) == bounds


def test_discovery_alias_and_duplicate_guard(tmp_path):
    folder = tmp_path / "session_robot_dog_anitclockwise"
    folder.mkdir()
    write_trial(folder / "one.npz")
    records, x, y = discover_dataset(tmp_path)
    assert records[0]["label"] == "anticlockwise"
    assert y.tolist() == [0]
    write_trial(folder / "two.npz", trial=2)
    with pytest.raises(ValueError, match="Duplicate"):
        discover_dataset(tmp_path)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_all_candidate_models_have_finite_gradients(architecture):
    torch.set_num_threads(2)
    model = GestureClassifier(architecture=architecture)
    x = augment_batch(torch.randn(3, 32, 3, 8, 8))
    logits = model(x)
    assert logits.shape == (3, 10)
    logits.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpoint_reload_and_array_file_predictions_agree(tmp_path):
    write_trial(tmp_path / "trial.npz")
    model = GestureClassifier(dropout=0).eval()
    bundle = {"format": "pinglab_robot_dog_gesture_v1", "members": [
        {"config": {"dropout": 0}, "state_dict": model.state_dict()}],
        "class_names": list(CLASS_NAMES), "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]},
        "temperature": 1.0, "preprocess": PREPROCESS, "sensor_shape": [8, 8]}
    torch.save(bundle, tmp_path / "model.pt")
    predictor = GesturePredictor(tmp_path / "model.pt")
    raw, avg, times, _ = load_trial(tmp_path / "trial.npz")
    first = predictor.predict_file(tmp_path / "trial.npz")
    assert first == predictor.predict_arrays(raw, avg, times)
    assert first["robot_command_sent"] is False
    assert abs(sum(first["probabilities"].values()) - 1) < 1e-6


def test_training_pipeline_smoke_on_synthetic_data_only(tmp_path):
    for class_index, label in enumerate(CLASS_NAMES):
        folder = tmp_path / "data" / f"session_robot_dog_{label}"
        folder.mkdir(parents=True)
        for trial in range(20):
            write_trial(folder / f"trial_{trial}.npz", trial=trial, offset=class_index + trial / 100)
    args = SimpleNamespace(output=str(tmp_path / "result"), dataset_root=str(tmp_path / "data"),
        device="cpu", architectures=["cnn_gru"], seeds=[17], epochs=1, patience=1,
        batch_size=64, learning_rate=0.001)
    output = train(args)
    predictor = GesturePredictor(output / "best_gesture_model.pt")
    assert len(predictor.class_names) == 10
    report = json.loads((output / "evaluation.json").read_text())
    assert report["test"]["samples"] == 40
    assert sum(item["support"] for item in report["test"]["per_class"]) == 40
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert len(manifest["records"]) == 200


def test_metrics_are_whole_trial_accuracy_and_macro_f1():
    labels = np.arange(10)
    metrics = classification_metrics(np.eye(10), labels)
    assert metrics["accuracy"] == metrics["macro_f1"] == 1.0
    assert metrics["correct"] == metrics["samples"] == 10

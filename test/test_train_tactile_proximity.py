import json
from pathlib import Path

import numpy as np
import torch

from phd.dependence.tactile_models import TactileProximityCNNGRU
from phd.dependence.tactile_proximity import (
    LocalizedChangeAccumulator,
    TactileProximityDetector,
    build_tactile_proximity_channel_frame,
)
from phd.script.train_tactile_proximity import (
    EnvironmentRecording,
    NormalSequenceDataset,
    RecordingBalancedSampler,
    balanced_recording_subset_indices,
    compute_normalization,
    fit_anomaly_calibration,
    load_environment_recordings,
    resolve_physical_sensor_shape,
    resolve_sensor_shape,
    split_recordings,
)


def _write_ai_dfm_recording(path, offset=0.0, frame_count=24):
    base = np.arange(frame_count * 100, dtype=np.float32).reshape(
        frame_count,
        10,
        10,
    )
    base = base * 0.001 + float(offset)
    np.savez_compressed(
        path,
        diffPerData=base,
        diffPerDataAve=base * 0.8,
        metadata_json=np.asarray('{"kind":"normal"}'),
    )


def test_load_environment_recording_builds_cnn_gru_channels(tmp_path):
    path = tmp_path / "normal_001.npz"
    _write_ai_dfm_recording(path)

    recordings, channels = load_environment_recordings([path])

    assert channels == ("diffPerData", "diffPerDataAve", "frameDiff")
    assert recordings[0].frames.shape == (24, 3, 10, 10)
    np.testing.assert_allclose(recordings[0].frames[0, 2], 0.0)
    np.testing.assert_allclose(
        recordings[0].frames[1, 2],
        recordings[0].frames[1, 1] - recordings[0].frames[0, 1],
    )


def test_sensor_shape_is_inferred_from_recorded_frames(tmp_path):
    path = tmp_path / "normal_8x10.npz"
    frames = np.zeros((24, 8, 10), dtype=np.float32)
    np.savez_compressed(
        path,
        diffPerData=frames,
        diffPerDataAve=frames,
    )

    assert resolve_sensor_shape([path]) == (8, 10)


def test_column_major_recording_uses_model_facing_array_shape(tmp_path):
    path = tmp_path / "normal_physical_8x10.npz"
    frames = np.zeros((24, 10, 8), dtype=np.float32)
    metadata = {
        "sensor_rows": 8,
        "sensor_cols": 10,
        "teacher_layout": "column_major_matrix_view",
    }
    np.savez_compressed(
        path,
        diffPerData=frames,
        diffPerDataAve=frames,
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    assert resolve_sensor_shape([path]) == (10, 8)
    recordings, _channels = load_environment_recordings(
        [path],
        rows=10,
        cols=8,
    )
    assert recordings[0].frames.shape == (24, 3, 10, 8)
    assert resolve_physical_sensor_shape([path]) == (8, 10)


def test_mixed_sensor_shapes_are_rejected_before_training(tmp_path):
    paths = []
    for rows, cols in ((8, 10), (10, 10)):
        path = tmp_path / f"normal_{rows}x{cols}.npz"
        frames = np.zeros((24, rows, cols), dtype=np.float32)
        np.savez_compressed(
            path,
            diffPerData=frames,
            diffPerDataAve=frames,
        )
        paths.append(path)

    try:
        resolve_sensor_shape(paths)
    except ValueError as exc:
        assert "different sensor dimensions" in str(exc)
    else:
        raise AssertionError("Mixed sensor dimensions should be rejected")


def test_split_recordings_keeps_complete_files_in_each_partition():
    recordings = [
        EnvironmentRecording(
            path=Path(f"normal_{index}.npz"),
            frames=np.zeros((20, 1, 10, 10), dtype=np.float32),
            channels=("frames",),
            metadata={},
        )
        for index in range(4)
    ]

    train, validation = split_recordings(
        recordings,
        validation_count=1,
        seed=7,
    )

    assert len(train) == 3
    assert len(validation) == 1
    assert {item.path for item in train}.isdisjoint(
        {item.path for item in validation}
    )


def test_normal_sequence_dataset_uses_history_to_predict_next_frame(tmp_path):
    path = tmp_path / "normal_001.npz"
    _write_ai_dfm_recording(path)
    recordings, _channels = load_environment_recordings([path])
    normalization = compute_normalization(recordings)
    dataset = NormalSequenceDataset(
        recordings,
        sequence_length=8,
        normalization=normalization,
        augment=False,
    )

    sample = dataset[0]

    assert sample["history"].shape == (8, 3, 10, 10)
    assert sample["target"].shape == (3, 10, 10)
    assert sample["target_index"] == 8


def test_balanced_sampler_gives_each_recording_equal_epoch_weight():
    recordings = [
        EnvironmentRecording(
            path=Path(f"normal_{index}.npz"),
            frames=np.zeros((frame_count, 1, 2, 2), dtype=np.float32),
            channels=("frames",),
            metadata={},
        )
        for index, frame_count in enumerate((14, 22, 30))
    ]
    normalization = compute_normalization(recordings)
    dataset = NormalSequenceDataset(
        recordings,
        sequence_length=4,
        normalization=normalization,
        augment=False,
    )
    sampler = RecordingBalancedSampler(dataset, seed=7)

    selected = list(iter(sampler))
    selected_recordings = [dataset.indices[index][0] for index in selected]

    assert len(sampler) == 30
    assert [selected_recordings.count(index) for index in range(3)] == [
        10,
        10,
        10,
    ]
    assert len(balanced_recording_subset_indices(dataset)) == 30


def test_balanced_normalization_weights_trials_not_their_durations():
    recordings = [
        EnvironmentRecording(
            path=Path("short.npz"),
            frames=np.zeros((10, 1, 2, 2), dtype=np.float32),
            channels=("frames",),
            metadata={},
        ),
        EnvironmentRecording(
            path=Path("long.npz"),
            frames=np.full((100, 1, 2, 2), 10.0, dtype=np.float32),
            channels=("frames",),
            metadata={},
        ),
    ]

    balanced = compute_normalization(recordings, balance_recordings=True)
    duration_weighted = compute_normalization(
        recordings,
        balance_recordings=False,
    )

    np.testing.assert_allclose(balanced["mean"], 5.0)
    assert float(duration_weighted["mean"].mean()) > 9.0


def test_proximity_cnn_gru_outputs_next_frame_and_embeddings():
    model = TactileProximityCNNGRU(
        in_channels=3,
        sensor_rows=10,
        sensor_cols=10,
        d_model=16,
        gru_hidden=24,
        dropout=0.0,
    )
    history = torch.randn(2, 8, 3, 10, 10)

    output = model(history)
    current_embedding = model.encode_frames(torch.randn(2, 3, 10, 10))

    assert output["predicted_frame"].shape == (2, 3, 10, 10)
    assert output["context_embedding"].shape == (2, 24)
    assert output["history_embeddings"].shape == (2, 8, 16)
    assert current_embedding.shape == (2, 16)


def test_live_proximity_channels_match_training_channel_order():
    diff = np.full((2, 3), 2.0, dtype=np.float32)
    averaged = np.full((2, 3), 5.0, dtype=np.float32)
    previous = np.full((2, 3), 3.5, dtype=np.float32)

    frame = build_tactile_proximity_channel_frame(
        ("diffPerData", "diffPerDataAve", "frameDiff"),
        diff,
        averaged,
        previous,
    )

    assert frame.shape == (3, 2, 3)
    np.testing.assert_allclose(frame[0], diff)
    np.testing.assert_allclose(frame[1], averaged)
    np.testing.assert_allclose(frame[2], 1.5)


def test_localized_change_accumulates_a_persistent_sparse_decrease():
    detector = LocalizedChangeAccumulator(
        warmup_frames=4,
        top_taxels=2,
        allowance=0.1,
        decay=0.95,
        baseline_alpha=0.0,
    )
    for _ in range(4):
        assert detector.update(np.zeros((10, 10), dtype=np.float32)) is None

    changed = np.zeros((10, 10), dtype=np.float32)
    changed[2, 3] = -0.8
    changed[2, 4] = -0.8
    scores = [detector.update(changed) for _ in range(4)]

    assert all(score is not None for score in scores)
    assert scores == sorted(scores)
    assert scores[-1] > scores[0] * 2.0
    assert np.isclose(detector.center_row, 2.0)
    assert np.isclose(detector.center_col, 3.5)


def test_anomaly_calibration_uses_held_out_normal_quantile():
    train = {
        "prediction_error": np.linspace(0.1, 0.3, 100),
        "embedding": np.linspace(-1.0, 1.0, 800).reshape(100, 8),
    }
    validation = {
        "prediction_error": np.linspace(0.12, 0.35, 50),
        "embedding": np.linspace(-0.9, 1.1, 400).reshape(50, 8),
    }

    calibration, train_score, validation_score, latent_distance = (
        fit_anomaly_calibration(
            train,
            validation,
            threshold_quantile=0.98,
        )
    )

    assert np.isfinite(calibration["threshold"])
    assert calibration["threshold"] == np.quantile(validation_score, 0.98)
    assert train_score.shape == (100,)
    assert validation_score.shape == (50,)
    assert latent_distance.shape == (50,)


def test_streaming_detector_waits_for_history_then_scores_frame(tmp_path):
    model = TactileProximityCNNGRU(
        in_channels=1,
        sensor_rows=10,
        sensor_cols=10,
        d_model=8,
        gru_hidden=12,
        dropout=0.0,
    )
    checkpoint_path = tmp_path / "proximity.pt"
    torch.save(
        {
            "model_type": "tactile_proximity_cnn_gru",
            "model_state": model.state_dict(),
            "config": {
                "channels": ["frames"],
                "sequence_length": 4,
                "sensor_rows": 10,
                "sensor_cols": 10,
                "in_channels": 1,
                "d_model": 8,
                "gru_hidden": 12,
                "gru_layers": 1,
                "dropout": 0.0,
                "input_clip": 12.0,
                "top_error_fraction": 0.1,
            },
            "normalization": {
                "mean": np.zeros((1, 10, 10), dtype=np.float32),
                "std": np.ones((1, 10, 10), dtype=np.float32),
                "std_floor": 1.0,
            },
            "anomaly_calibration": {
                "latent_mean": np.zeros(8, dtype=np.float32),
                "latent_std": np.ones(8, dtype=np.float32),
                "prediction_center": 0.0,
                "prediction_scale": 1.0,
                "latent_center": 0.0,
                "latent_scale": 1.0,
                "prediction_weight": 0.75,
                "threshold": 100.0,
            },
        },
        checkpoint_path,
    )
    detector = TactileProximityDetector(
        checkpoint_path,
        device="cpu",
    )

    for _ in range(4):
        assert not detector.update(np.zeros((1, 10, 10))).ready
    result = detector.update(np.zeros((1, 10, 10)))

    assert result.ready
    assert np.isfinite(result.anomaly_score)
    assert not result.detected

import json

import numpy as np

from phd.ui.heatmap_signal_recording import (
    HeatmapSignalPlayback,
    HeatmapSignalRecorder,
)


class _Clock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        return self.value


def _snapshot(value):
    heatmap = np.full((2, 3), float(value))
    return {
        "heatmap_values": heatmap,
        "raw_values": heatmap + 100.0,
        "calibration_values": np.full((2, 3), 100.0),
    }


def test_heatmap_signal_recording_round_trip(tmp_path):
    clock = _Clock()
    recorder = HeatmapSignalRecorder(fps=60.0, clock=clock)
    output_path = tmp_path / "scan_sensor.npz"
    metadata = {
        "model": "2d",
        "n_row": 2,
        "n_col": 3,
        "zero_mask": [[0, 1, 0], [0, 0, 0]],
        "geometry": {"shape": "custom", "rotation_deg": [0, 0, 15]},
    }

    assert recorder.start(
        output_path,
        _snapshot(1.0),
        metadata=metadata,
        video_path=tmp_path / "scan.mp4",
    )
    clock.value += 0.021
    assert recorder.capture(_snapshot(2.0))
    clock.value += 0.018
    assert recorder.capture(_snapshot(3.0))
    assert recorder.stop() == output_path

    playback = HeatmapSignalPlayback.load(output_path)

    assert playback.fps == 60.0
    assert playback.frame_count == 3
    assert playback.frame_shape == (2, 3)
    assert playback.video_filename == "scan.mp4"
    assert playback.metadata == metadata
    np.testing.assert_allclose(playback.heatmap_frames[:, 0, 0], [1, 2, 3])
    np.testing.assert_allclose(playback.raw_frames[:, 0, 0], [101, 102, 103])
    np.testing.assert_allclose(playback.calibration_values, 100.0)
    np.testing.assert_allclose(playback.timestamps_s, [0.0, 1 / 60, 2 / 60])
    np.testing.assert_allclose(playback.capture_timestamps_s, [0.0, 0.021, 0.039])


def test_heatmap_signal_playback_holds_latest_frame_between_timestamps(tmp_path):
    recorder = HeatmapSignalRecorder(fps=10.0)
    output_path = tmp_path / "scan_sensor.npz"
    assert recorder.start(output_path, _snapshot(1.0))
    assert recorder.capture(_snapshot(2.0))
    assert recorder.capture(_snapshot(3.0))
    assert recorder.stop()
    playback = HeatmapSignalPlayback.load(output_path)

    assert playback.frame_index_at(0.0) == 0
    assert playback.frame_index_at(0.099) == 0
    assert playback.frame_index_at(0.1) == 1
    assert playback.frame_index_at(10.0) == 2
    np.testing.assert_allclose(playback.frame_at(0.15), 2.0)


def test_heatmap_signal_recorder_can_remove_unmatched_video_frame(tmp_path):
    recorder = HeatmapSignalRecorder(fps=30.0)
    output_path = tmp_path / "scan_sensor.npz"
    assert recorder.start(output_path, _snapshot(1.0))
    assert recorder.capture(_snapshot(2.0))
    assert recorder.discard_last_frame()
    assert recorder.frame_count == 1
    assert recorder.stop()

    playback = HeatmapSignalPlayback.load(output_path)
    assert playback.frame_count == 1
    np.testing.assert_allclose(playback.heatmap_frames[0], 1.0)


def test_heatmap_signal_recorder_holds_previous_signal_across_frame_gaps(tmp_path):
    clock = _Clock()
    recorder = HeatmapSignalRecorder(fps=60.0, clock=clock)
    output_path = tmp_path / "timed_sensor.npz"
    assert recorder.start(output_path, _snapshot(1.0))

    clock.value += 0.01
    assert recorder.align_timeline_start(clock.value)
    clock.value += 0.05
    assert recorder.capture_to_frame_count(_snapshot(4.0), 4)
    assert recorder.frame_count == 4
    assert recorder.stop()

    playback = HeatmapSignalPlayback.load(output_path)
    np.testing.assert_allclose(
        playback.heatmap_frames[:, 0, 0], [1.0, 1.0, 1.0, 4.0]
    )
    np.testing.assert_allclose(
        playback.capture_timestamps_s, [0.0, 0.0, 0.0, 0.05]
    )


def test_heatmap_signal_recorder_rejects_shape_changes(tmp_path):
    recorder = HeatmapSignalRecorder(fps=60.0)
    assert recorder.start(tmp_path / "scan_sensor.npz", _snapshot(1.0))

    assert not recorder.capture(
        {
            "heatmap_values": np.zeros((3, 3)),
            "raw_values": np.zeros((3, 3)),
        }
    )
    assert "does not match" in recorder.last_error


def test_heatmap_signal_playback_rejects_incomplete_file(tmp_path):
    output_path = tmp_path / "invalid.npz"
    np.savez_compressed(
        output_path,
        metadata_json=np.asarray(json.dumps({"model": "2d"})),
    )

    try:
        HeatmapSignalPlayback.load(output_path)
    except ValueError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("Incomplete recording should not load")

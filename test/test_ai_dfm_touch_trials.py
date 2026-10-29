"""Touch-triggered policy recordings use fresh sensor frames, never hardware."""

import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import Mock

import numpy as np
from PyQt5.QtWidgets import QApplication
import pytest

from phd.dependence.gesture import gesture_logic_direct_finger_motion as recording
from phd.dependence.robot_dog_gesture_model import load_trial
from phd.ui.ui_ping_direct_finger_motion import DirectFingerMotionMixin


_QT_APP = QApplication.instance() or QApplication([])
CHANNELS = ("rawData", "diffData", "diffPerData", "diffDataAve", "diffPerDataAve")


@pytest.fixture
def make_recorder(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=10.0, strftime=time.strftime, localtime=time.localtime)
    clock.perf_counter = lambda: clock.now
    clock.monotonic = lambda: clock.now
    clock.time = lambda: 1800000000.0 + clock.now
    monkeypatch.setattr(recording, "time", clock)
    monkeypatch.setattr(recording.DirectFingerMotion, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    helpers = []

    def make(*, shape=(8, 8), start_on_touch=True):
        data = SimpleNamespace(frame_sequence=0, **{k: np.zeros(shape) for k in CHANNELS})
        sensor = SimpleNamespace(n_row=shape[0], n_col=shape[1], _data=data, updateCal=Mock())
        robot = SimpleNamespace(**{
            name: Mock(side_effect=AssertionError("Robot command in no-robot recording"))
            for name in ("enter_end_effector_velocity_mode", "exit_end_effector_velocity_mode",
                         "send_end_effector_velocity_in_frame", "send_request")
        })
        host = SimpleNamespace(robot_api=robot, ai_selected_frame="tool", log_display=None)
        helper = recording.AI_DirectFingerMotion(host, sensor)
        helper.dataset_root = str(tmp_path / f"dataset_{len(helpers)}")
        helper.toggle_ai_direct_finger_motion("touch_trials", start_on_touch=start_on_touch)
        helper.control_timer.stop()  # Supply deterministic frames, not a live timer.
        helpers.append(helper)

        def feed(value=0.0, count=1, *, dt=0.05, fresh=True, position=(0, 0)):
            for _ in range(count):
                clock.now += dt
                if fresh:
                    data.frame_sequence += 1
                    for key in CHANNELS:
                        matrix = np.zeros(shape, dtype=np.float32)
                        matrix[position] = value
                        setattr(data, key, matrix)
                helper.run_step()

        return SimpleNamespace(
            helper=helper, feed=feed, sensor=sensor, robot=robot,
            files=lambda: sorted(Path(helper.dataset_root).rglob("*.npz")),
        )

    yield make
    for helper in helpers:
        helper.control_timer.stop()


@pytest.mark.parametrize("shape", [(5, 5), (8, 8), (8, 10)])
def test_touch_release_saves_and_rearms_without_robot(make_recorder, shape):
    run = make_recorder(shape=shape)
    run.feed(count=20)
    assert run.helper.current_episode["timestamps"] == []
    assert run.files() == []
    for trial in (1, 2):
        run.feed(-10.0, 6)
        run.feed(count=5)
        assert len(run.files()) == trial
        assert run.helper.is_running
        assert not run.helper._touch_trial_active
        assert run.helper.trial_number is None
        with np.load(run.files()[-1], allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            assert metadata["trial_number"] == trial
            assert metadata["record_trigger_mode"] == "trigger"
            assert metadata["completion_reason"] == "touch_released"
            assert metadata["dataset_role"] == "touch_gesture_recording"
            assert metadata["robot_commands_sent"] is False
            assert set(data.files) == {
                "metadata_json", "timestamps", "elapsed_sec", "dt_sec",
                "sensor_frame_sequence", "touch_present", *CHANNELS,
            }
            assert data["touch_present"][0] == 1
            assert data["touch_present"][-1] == 0
            assert data["elapsed_sec"][0] == 0
            assert data["rawData"].shape[1:] == shape[::-1]
            assert len(np.unique(data["sensor_frame_sequence"])) == len(data["timestamps"])
        run.feed(count=20)
        assert len(run.files()) == trial
    run.helper.toggle_ai_direct_finger_motion()
    assert len(run.files()) == 2  # Stopping while armed never writes an empty trial.
    run.sensor.updateCal.assert_not_called()
    for command in vars(run.robot).values():
        command.assert_not_called()


def test_brief_release_is_not_a_new_trial_and_long_touch_is_not_chunked(make_recorder):
    run = make_recorder()
    run.feed(-10, 80)
    assert run.files() == []
    run.feed(count=2)
    run.feed(-10, 6)
    assert run.files() == []
    run.feed(count=5)
    assert len(run.files()) == 1
    with np.load(run.files()[0]) as data:
        assert data["touch_present"].sum() == 86


def test_stale_or_invalid_frames_cannot_finish_a_trial(make_recorder):
    run = make_recorder()
    run.feed(-10, 6)
    run.feed(count=2)
    run.feed(count=20, fresh=False)
    assert run.files() == []
    assert run.helper._release_frames == 0
    run.feed(count=2)
    run.feed(float("nan"), 5)
    assert run.files() == []
    assert run.helper._release_frames == 0
    run.feed(count=5)
    assert len(run.files()) == 1
    with np.load(run.files()[0]) as data:
        assert np.isfinite(data["diffPerDataAve"]).all()


def test_release_requires_both_multiple_fresh_frames_and_elapsed_time(make_recorder):
    run = make_recorder()
    run.feed(-10, 6)
    run.feed(count=4, dt=0.005)
    assert run.files() == []
    run.feed(dt=0.15)
    assert len(run.files()) == 1


def test_single_frame_glitch_is_discarded_without_consuming_trial_number(make_recorder):
    run = make_recorder()
    run.feed(-10)
    run.feed(count=5)
    assert run.files() == []
    assert not run.helper._touch_trial_active
    run.feed(-10, 6)
    run.feed(count=5)
    assert len(run.files()) == 1
    assert run.files()[0].name.startswith("trial_0001_")


def test_manual_stop_saves_active_touch_once_and_restart_is_clean(make_recorder):
    run = make_recorder()
    run.feed(-10, 6)
    run.helper.toggle_ai_direct_finger_motion()
    assert len(run.files()) == 1
    with np.load(run.files()[0]) as data:
        assert json.loads(str(data["metadata_json"].item()))["completion_reason"] == "manual_stop"
    run.helper.toggle_ai_direct_finger_motion("touch_trials", start_on_touch=True)
    run.helper.control_timer.stop()
    run.feed(count=20)
    assert len(run.files()) == 1
    run.feed(-10, 6)
    run.feed(count=5)
    assert len(run.files()) == 2


def test_manual_and_environment_recording_remain_continuous(make_recorder):
    run = make_recorder(start_on_touch=False)
    run.feed(count=6)
    run.feed(-10, 6)
    run.feed(count=10)
    assert run.files() == []
    run.helper.toggle_ai_direct_finger_motion()
    with np.load(run.files()[0]) as data:
        assert len(data["timestamps"]) == 22
        assert json.loads(str(data["metadata_json"].item()))["record_trigger_mode"] == "no_trigger"
    # An environment trial calls the API without a trigger argument, even if
    # the previous user recording used touch-triggered mode.
    run.helper.start_on_touch = True
    run.helper.toggle_ai_direct_finger_motion("environment")
    run.helper.control_timer.stop()
    assert run.helper.start_on_touch is False
    run.feed(count=6)
    run.helper.toggle_ai_direct_finger_motion()
    assert len(run.files()) == 2


def test_saved_trial_loads_in_the_touch_gesture_trainer(make_recorder):
    run = make_recorder(shape=(10, 10))
    run.feed(-10.0, 6, position=(2, 7))
    run.feed(count=5)
    raw, averaged, times, metadata = load_trial(run.files()[0])
    assert raw.shape[1:] == (10, 10) and averaged.shape == raw.shape
    assert averaged[0, 2, 7] == -10.0  # Native (row, column) after the loader's transpose.
    assert times[0] == 0 and metadata["session_tag"] == "touch_trials"


@pytest.mark.parametrize("mode,expected", [("trigger", True), ("no_trigger", False)])
def test_ui_passes_selected_touch_mode_to_actual_recorder(mode, expected):
    helper = SimpleNamespace(toggle_ai_direct_finger_motion=Mock())
    controls = DirectFingerMotionMixin()
    controls._set_ai_direct_finger_record_buttons = Mock()
    controls._get_ai_direct_finger_motion_helper = lambda: helper
    controls._current_ai_dfm_session_tag = lambda: "test_session"
    controls.record_trigger_mode_combo = SimpleNamespace(currentData=lambda: mode)
    controls._on_toggle_ai_direct_finger_motion()
    helper.toggle_ai_direct_finger_motion.assert_called_once_with(
        session_tag="test_session", start_on_touch=expected,
    )

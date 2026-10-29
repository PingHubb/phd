"""Streaming gesture/velocity boundaries, without any connected hardware."""
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PyQt5.QtCore import QCoreApplication

from phd.dependence.sensor_data import SensorDataBuffer
from phd.dependence.gesture.gesture_logic_direct_finger_motion import AI_DirectFingerMotion_execution
from phd.dependence.touch_gesture_policy import (
    CLASS_NAMES, PREPROCESS_CONFIG, CONTINUOUS_PREPROCESS_CONFIG, TouchPrefixBuffer, VELOCITY_DIRECTIONS,
    gesture_velocity, prepare_prefix,
)

_APP = QCoreApplication.instance() or QCoreApplication([])


class FixedModel:
    def __call__(self, x):
        return torch.tensor([[0.96, 0.01, 0.01, 0.01, 0.005, 0.005]], device=x.device)


@pytest.fixture
def live(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    data = SensorDataBuffer(10, 10, 3)
    data.frame_monotonic_sec = now[0]
    sensor = SimpleNamespace(n_row=10, n_col=10, _data=data, is_connected=True,
                             current_reorder_mode="row_to_col", _sensor_calibration_in_progress=False)
    helper = AI_DirectFingerMotion_execution(SimpleNamespace(robot_api=None, ai_selected_frame="tool"), sensor)
    helper.model_kind, helper.model_loaded, helper.is_running = "cnn_tcn_gesture", True, True
    helper._torch, helper.device, helper.model = torch, torch.device("cpu"), FixedModel()
    helper.inference_in_background = False
    helper.dry_run_predictions_only = True
    helper.model_conf_threshold, helper.max_linear_speed = 0.8, 0.05
    helper._touch_gesture_bundle = {
        "sensor_shape": [10, 10], "sensor_reorder_mode": "row_to_col", "sensor_average_window_size": 3,
        "preprocess": dict(PREPROCESS_CONFIG), "control": {"base_speed_mps": 0.02, "margin_threshold": 0.15},
    }
    helper._start_touch_gesture_stream()

    def frame(touch=False, poll=True, invalid=False, strength=8.0):
        now[0] += 0.02
        data.frame_sequence += 1
        data.frame_monotonic_sec = now[0]
        a = np.zeros((10, 10), np.float32)
        if touch:
            a[4:6, 4:6] = -strength
        if invalid:
            a[0, 0] = np.nan
        data.diffPerData, data.diffPerDataAve = a, a
        data.completed_frame_snapshot = SimpleNamespace(frame_sequence=data.frame_sequence,
            frame_monotonic_sec=now[0], diffPerData=a.copy(), diffPerDataAve=a.copy())
        for callback in tuple(data._frame_listeners):
            callback(data)
        if poll:
            helper.run_step()

    yield helper, frame, now, data
    helper.is_running = False
    helper._reset_execution_buffers()
    helper._shutdown_inference_executor()


def arm_and_touch(frame):
    for _ in range(10):
        frame(False)
    for _ in range(12):
        frame(True)


def _publish_snapshot(data, now, averaged):
    data.frame_sequence += 1
    data.frame_monotonic_sec = now[0]
    data.diffPerData = np.array(averaged, copy=True)
    data.diffPerDataAve = np.array(averaged, copy=True)
    data.completed_frame_snapshot = SimpleNamespace(
        frame_sequence=data.frame_sequence, frame_monotonic_sec=now[0],
        diffPerData=data.diffPerData, diffPerDataAve=data.diffPerDataAve)


def test_hand_already_away_accepts_the_first_touch(live):
    helper, frame, now, data = live
    _publish_snapshot(data, now, np.zeros((10, 10), np.float32))
    helper._start_touch_gesture_stream()
    assert helper._touch_prefix.state == "ready"
    assert helper.last_prediction["reason"] == "Hand is away. Swipe to start."
    for _ in range(12):
        frame(True)
    assert helper.last_prediction["gesture"] == "up"
    np.testing.assert_allclose(helper.last_prediction["velocity_sent"], gesture_velocity("up"))


def test_finger_down_at_execute_still_requires_release(live):
    helper, frame, now, data = live
    down = np.zeros((10, 10), np.float32)
    down[4:6, 4:6] = -8
    _publish_snapshot(data, now, down)
    helper._start_touch_gesture_stream()
    assert helper._touch_prefix.state == "release"
    for _ in range(12):
        frame(True)
    assert helper.last_prediction["mode"] == "stop"
    assert "arm" in helper.last_prediction["reason"]


def test_starting_with_finger_down_requires_release(live):
    helper, frame, _, _ = live
    for _ in range(12):
        frame(True)
    assert helper.last_prediction["mode"] == "stop"
    arm_and_touch(frame)
    assert helper.last_prediction["gesture"] == "up"
    np.testing.assert_allclose(helper.last_prediction["velocity_sent"], [0, 0, -0.05, 0, 0, 0])


@pytest.mark.parametrize("index,expected", [(0, [0, 0, -0.05, 0, 0, 0]),
    (1, [0, 0, 0.05, 0, 0, 0]), (2, [-0.05, 0, 0, 0, 0, 0]), (3, [0.05, 0, 0, 0, 0, 0])])
def test_four_swipe_labels_drive_correct_axes_and_feedback(live, index, expected):
    from phd.dependence.touch_gesture_policy import SWIPE_CLASS_NAMES
    helper, frame, _, _ = live
    helper._touch_gesture_bundle["class_names"] = list(SWIPE_CLASS_NAMES)
    probabilities = torch.full((1, 4), .01)
    probabilities[0, index] = .97
    helper.model = lambda x: probabilities.to(x.device)
    arm_and_touch(frame)
    assert helper.last_prediction["gesture"] == SWIPE_CLASS_NAMES[index]
    assert tuple(helper.last_prediction["gesture_probabilities"]) == SWIPE_CLASS_NAMES
    np.testing.assert_allclose(helper.last_prediction["velocity_sent"], expected)
    for _ in range(10):
        frame(False)
    assert not any(helper.last_prediction["velocity_sent"])
    assert helper.touch_gesture_feedback.completed["gesture"] == SWIPE_CLASS_NAMES[index]


def test_wrong_class_count_stops_touch_policy(live):
    from phd.dependence.touch_gesture_policy import SWIPE_CLASS_NAMES
    helper, frame, _, _ = live
    helper._touch_gesture_bundle["class_names"] = list(SWIPE_CLASS_NAMES)
    arm_and_touch(frame)  # The old fake model returns six probabilities.
    assert not helper.is_running
    assert not any(helper.last_prediction["velocity_sent"])


@pytest.mark.parametrize("index,mode,expected", [
    (4, "push", [0, -.05, 0, 0, 0, 0]), (5, "pull", [0, .05, 0, 0, 0, 0]),
])
@pytest.mark.parametrize("continuous", [False, True])
def test_two_finger_gestures_keep_labels_axes_modes_and_release_feedback(live, index, mode, expected, continuous):
    from phd.dependence.touch_gesture_policy import SIX_GESTURE_CLASS_NAMES
    helper, frame, _, _ = live
    helper._touch_gesture_bundle["class_names"] = list(SIX_GESTURE_CLASS_NAMES)
    if continuous:
        configure_continuous(helper)
    probabilities = torch.full((1, 6), .005)
    probabilities[0, index] = .975
    helper.model = lambda x: probabilities.to(x.device)
    for _ in range(14):
        frame(False)
    for i in range(20):
        frame(True, strength=8 + i * .5)
    label = SIX_GESTURE_CLASS_NAMES[index]
    assert helper.last_prediction["gesture"] == label
    assert helper.last_prediction["mode"] == mode
    np.testing.assert_allclose(helper.last_prediction["velocity_sent"], expected)
    if continuous:
        # The current user's hold behaviour must retain the correct push/pull mode,
        # including both the paused-signal and uncertain-prediction paths.
        for _ in range(8):
            frame(True, strength=17.5)
        assert helper.last_prediction["reason"] == "Holding last gesture"
        assert helper.last_prediction["mode"] == mode
        helper.model = lambda x: torch.full((1, 6), 1/6, device=x.device)
        for i in range(8):
            frame(True, strength=18 + i)
        assert helper.last_prediction["gesture"] == label
        assert helper.last_prediction["mode"] == mode
        np.testing.assert_allclose(helper.last_prediction["velocity_sent"], expected)
    frame(False)
    assert not any(helper.last_prediction["velocity_sent"])
    for _ in range(14):
        frame(False)
    assert helper.touch_gesture_feedback.completed["gesture"] == label



def test_release_discards_completed_background_motion(live):
    helper, frame, _, _ = live
    arm_and_touch(frame)
    helper.inference_in_background = True
    future = Future()
    future.set_result({"mode": "move", "mode_conf": 0.99, "gesture": "up", "finger_idx": 0,
                       "velocity_sent": np.array([0, 0, -0.02, 0, 0, 0])})
    helper._pending_inference = (helper._inference_epoch, future)
    frame(False)
    assert helper.last_prediction["mode"] == "stop"
    assert helper._pending_inference is None
    assert not any(helper.last_prediction["velocity_sent"])


@pytest.mark.parametrize("failure", ["exception", "expired", "stalled"])
def test_background_failures_cannot_latch_motion(live, failure):
    helper, frame, now, _ = live
    arm_and_touch(frame)
    helper.inference_in_background = True
    future = Future()
    if failure == "exception":
        future.set_exception(RuntimeError("test inference failure"))
    elif failure == "expired":
        future.set_result({"mode": "move", "mode_conf": 0.99, "gesture": "up", "finger_idx": 0,
                           "velocity_sent": gesture_velocity("up"), "source_monotonic_sec": now[0] - 0.25})
    helper._pending_inference = (helper._inference_epoch, future)
    helper._touch_pending_started = now[0] - (0.25 if failure == "stalled" else 0.01)
    frame(True)
    assert not helper.is_running
    assert helper.last_prediction["mode"] == "stop"
    assert not any(helper.last_prediction["velocity_sent"])


@pytest.mark.parametrize("fault", ["stale", "calibration", "layout", "control_frame", "disconnect", "nan", "queue", "sequence"])
def test_faults_stop_and_unsubscribe(live, fault):
    helper, frame, now, data = live
    arm_and_touch(frame)
    if fault == "stale":
        now[0] += 0.3
    elif fault == "calibration":
        data.calData[0, 0] += 1
    elif fault == "layout":
        helper.my_sensor.current_reorder_mode = "factory"
    elif fault == "control_frame":
        helper.ros_splitter.ai_selected_frame = "base"
    elif fault == "disconnect":
        helper.my_sensor.is_connected = False
    elif fault == "nan":
        frame(True, poll=False, invalid=True)
    elif fault == "queue":
        for _ in range(65):
            frame(True, poll=False)
    elif fault == "sequence":
        data.frame_sequence = 0
        frame(True, poll=False)
    helper.run_step()
    assert not helper.is_running
    assert not any(helper.last_prediction["velocity_sent"])
    assert not data._frame_listeners
    assert helper.last_prediction["reason"]


def test_run_mode_cannot_be_changed_before_live_stop(live):
    helper, _, _, _ = live
    helper.dry_run_predictions_only = False
    assert helper.set_dry_run_predictions_only(True) is False
    helper.is_running = False
    assert helper.set_dry_run_predictions_only(True) is True


def test_direction_change_requires_new_confirmation(live):
    helper, frame, _, _ = live
    arm_and_touch(frame)
    prediction = {"mode": "move", "mode_conf": 0.99, "gesture": "down", "finger_idx": 0,
                  "velocity_sent": gesture_velocity("down")}
    helper._apply_prediction(prediction)
    assert helper.last_prediction["mode"] == "stop"
    helper._apply_prediction(prediction)
    assert helper.last_prediction["mode"] == "move"
    assert helper.last_prediction["velocity_sent"][2] > 0


def test_low_confidence_does_not_leave_previous_velocity(live):
    helper, frame, _, _ = live
    arm_and_touch(frame)
    helper.model = lambda x: torch.ones((1, 6), device=x.device) / 6
    frame(True)
    assert helper.last_prediction["gesture"] == "uncertain"
    assert not any(helper.last_prediction["velocity_sent"])


def test_overlong_contact_stops_until_release(live):
    helper, frame, _, _ = live
    arm_and_touch(frame)
    for _ in range(110):
        frame(True)
    assert helper.last_prediction["mode"] == "stop"
    assert helper._touch_prefix.state == "release"
    arm_and_touch(frame)
    assert helper.last_prediction["mode"] == "move"


def test_mapping_has_six_opposite_axes_and_fixed_speed():
    for name in CLASS_NAMES:
        velocity = gesture_velocity(name, scale=20)
        np.testing.assert_allclose(velocity, np.array(VELOCITY_DIRECTIONS[name]) * 1.0)
        assert np.count_nonzero(velocity) == 1
        assert not np.any(velocity[3:])
    for a, b in [("up", "down"), ("left", "right"), ("push", "pull")]:
        np.testing.assert_allclose(gesture_velocity(a), -gesture_velocity(b))
    assert not np.any(gesture_velocity("uncertain"))
    with pytest.raises(ValueError):
        gesture_velocity("up", scale=float("nan"))


def test_prefix_is_past_only_and_has_no_teacher_features():
    a = np.zeros((16, 10, 10), np.float32)
    a[:, 2:4, 3:5] = -8
    times = np.arange(16) * 0.02
    before = prepare_prefix(a[:8], a[:8], times[:8])
    a[8:] = 999  # Future motion must have no influence on the first prediction.
    after = prepare_prefix(a[:8], a[:8], times[:8])
    np.testing.assert_array_equal(before, after)
    assert before.shape == (32, 3, 10, 10)
    with pytest.raises(ValueError):
        prepare_prefix(a[:2], a[:2], times[:2])


def test_duplicate_frames_cannot_accumulate_evidence():
    buffer = TouchPrefixBuffer()
    empty = np.zeros((10, 10))
    for i in range(10):
        buffer.feed(i, 1 + i * 0.02, empty, empty)
    touch = empty - 8
    buffer.feed(10, 1.2, touch, touch)
    for _ in range(20):
        buffer.feed(10, 1.2, touch, touch)
    assert len(buffer.frames) == 1
    assert buffer.snapshot() is None


def test_completed_feedback_waits_for_release_and_keeps_real_confidence(live):
    helper, frame, _, _ = live
    arm_and_touch(frame)
    feedback = helper.touch_gesture_feedback
    assert feedback.completed is None
    frame(False)
    assert not any(helper.last_prediction["velocity_sent"])
    assert feedback.completed is None  # display debounce never delays stop
    for _ in range(8):
        frame(False)
    result = feedback.completed
    assert result["gesture"] == "up" and result["accepted"]
    assert result["confidence"] == pytest.approx(0.96)
    for _ in range(10):
        frame(False)
    assert feedback.completed is result  # idle "Stop 100%" cannot overwrite it
    for _ in range(12):
        frame(True)
    assert feedback.completed is result  # previous result stays during next touch


def test_short_touch_does_not_reuse_previous_completed_gesture(live):
    helper, frame, _, _ = live
    arm_and_touch(frame)
    for _ in range(10):
        frame(False)
    assert helper.touch_gesture_feedback.completed["gesture"] == "up"
    frame(True)
    frame(True)
    for _ in range(10):
        frame(False)
    result = helper.touch_gesture_feedback.completed
    assert result["gesture"] == "not_recognised"
    assert result["confidence"] is None


def test_uncertain_completed_gesture_is_not_displayed_as_accepted(live):
    helper, frame, _, _ = live
    helper.model = lambda x: torch.ones((1, 6), device=x.device) / 6
    arm_and_touch(frame)
    for _ in range(10):
        frame(False)
    result = helper.touch_gesture_feedback.completed
    assert result["gesture"] == "uncertain" and not result["accepted"]
    assert result["confidence"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("failure", ["stale", "stop", "timeout"])
def test_incomplete_or_faulted_touch_never_becomes_completed(live, failure):
    helper, frame, now, _ = live
    arm_and_touch(frame)
    if failure == "stale":
        now[0] += 0.3
        helper.run_step()
    elif failure == "stop":
        helper.toggle_ai_direct_finger_motion_execution()
    else:
        for _ in range(110):
            frame(True)
        for _ in range(10):
            frame(False)
    assert helper.touch_gesture_feedback.completed is None


def test_continuous_completed_result_is_last_direction_not_whole_path(live):
    helper, frame, _, _ = live
    configure_continuous(helper)
    for _ in range(14):
        frame(False)
    for i in range(20):
        frame(True, strength=8 + i * 0.5)
    helper.model = lambda x: torch.tensor([[0.005, 0.005, 0.005, 0.975, 0.005, 0.005]], device=x.device)
    for i in range(12):
        frame(True, strength=18 + i * 0.5)
    assert helper.touch_gesture_feedback.completed is None
    for _ in range(8):
        frame(True, strength=23.5)  # pause leaves last model result intact
    for _ in range(14):
        frame(False)
    result = helper.touch_gesture_feedback.completed
    assert result["continuous"]
    assert result["gesture"] == "right" and result["confidence"] == pytest.approx(0.975)


def configure_continuous(helper):
    helper._touch_gesture_bundle["preprocess"] = dict(CONTINUOUS_PREPROCESS_CONFIG)
    helper._touch_gesture_bundle["control"]["acceleration_limit_mps2"] = 0.2
    helper.max_linear_speed = 1.0e6
    helper._start_touch_gesture_stream()


def test_continuous_crop_forgets_old_direction_and_never_uses_future():
    rng = np.random.default_rng(23)
    raw = rng.normal(-8, 1, (40, 10, 10)).astype(np.float32)
    times = np.arange(40) * 0.02
    config = CONTINUOUS_PREPROCESS_CONFIG
    first = prepare_prefix(raw, raw, times, config)
    raw[:25] = -99  # Completely different initial gesture must not matter.
    np.testing.assert_allclose(first, prepare_prefix(raw, raw, times, config))
    np.testing.assert_allclose(first, prepare_prefix(raw[30:], raw[30:], times[30:], config))


def test_single_touch_history_retains_early_movement():
    rng = np.random.default_rng(23)
    raw = rng.normal(-8, 1, (40, 10, 10)).astype(np.float32)
    times = np.arange(40) * .02
    initial = prepare_prefix(raw, raw, times, PREPROCESS_CONFIG)
    raw[:20, 2:4, 2:4] -= 20
    assert not np.allclose(initial, prepare_prefix(raw, raw, times, PREPROCESS_CONFIG))


def test_single_touch_uncertainty_stops_without_reusing_a_latched_command(live):
    helper, frame, _, _ = live
    arm_and_touch(frame)
    assert not helper._touch_prefix.continuous
    assert helper.last_prediction["mode"] != "stop"
    helper.model = lambda x: torch.full((1, 6), 1/6, device=x.device)
    frame(True)
    assert helper.last_prediction["mode"] == "stop"
    assert not any(helper.last_prediction["velocity_sent"])
    assert helper._touch_latched_gesture is None
    for _ in range(10):
        frame(False)
    assert helper._touch_prefix.frames == []


def test_single_touch_timeout_requires_release_before_another_prediction(live):
    helper, frame, _, _ = live
    arm_and_touch(frame)
    for _ in range(110):
        frame(True)
    assert helper._touch_prefix.state == "release"
    assert not any(helper.last_prediction["velocity_sent"])
    assert helper.touch_gesture_feedback.completed is None
    arm_and_touch(frame)
    assert helper.last_prediction["mode"] != "stop"


def test_continuous_can_change_direction_without_lifting_and_outlast_two_seconds(live):
    helper, frame, _, _ = live
    configure_continuous(helper)
    for _ in range(14):
        frame(False)
    for i in range(120):
        frame(True, strength=8 + i * 0.5)
    assert helper.last_prediction["gesture"] == "up"
    assert helper.last_prediction["velocity_sent"][2] < 0
    assert len(helper._touch_prefix.frames) <= 11
    assert helper._touch_prefix.episode == 1
    helper.model = lambda x: torch.tensor([[0.005, 0.005, 0.005, 0.975, 0.005, 0.005]], device=x.device)
    for i in range(6):
        frame(True, strength=68 + i * 0.5)
    assert helper.last_prediction["gesture"] == "right"
    assert helper.last_prediction["velocity_sent"][0] > 0
    assert helper._touch_prefix.episode == 1
    assert helper.is_running


def test_hold_keeps_last_direction_until_a_new_gesture_or_release(live):
    helper, frame, _, _ = live
    configure_continuous(helper)
    for _ in range(14):
        frame(False)
    for _ in range(12):
        frame(True, strength=17.5)
    assert helper.last_prediction["mode"] == "stop"
    assert "paused" in helper.last_prediction["reason"]
    assert helper._touch_latched_gesture is None
    for i in range(20):
        frame(True, strength=8 + i * 0.5)
    np.testing.assert_allclose(helper.last_prediction["velocity_sent"], gesture_velocity("up"))
    for _ in range(8):
        frame(True, strength=17.5)
    assert helper.last_prediction["gesture"] == "up"
    assert helper.last_prediction["reason"] == "Holding last gesture"
    np.testing.assert_allclose(helper.last_prediction["velocity_sent"], gesture_velocity("up"))
    helper.model = lambda x: torch.tensor([[0.005, 0.005, 0.005, 0.975, 0.005, 0.005]], device=x.device)
    for i in range(8):
        frame(True, strength=18 + i)
    np.testing.assert_allclose(helper.last_prediction["velocity_sent"], gesture_velocity("right"))
    for _ in range(8):
        frame(True, strength=30)
    assert helper.last_prediction["gesture"] == "right"
    np.testing.assert_allclose(helper.last_prediction["velocity_sent"], gesture_velocity("right"))
    frame(False)
    assert not any(helper.last_prediction["velocity_sent"])
    assert helper._touch_latched_gesture is None


def test_continuous_pause_discards_already_finished_inference(live):
    helper, frame, now, _ = live
    configure_continuous(helper)
    for _ in range(14):
        frame(False)
    for i in range(20):
        frame(True, strength=8 + i * 0.5)
    for _ in range(8):
        frame(True, strength=17.5, poll=False)
    future = Future()
    future.set_result({"mode": "move", "mode_conf": 0.99, "gesture": "down", "finger_idx": 0,
                       "velocity_sent": gesture_velocity("down"), "source_monotonic_sec": now[0]})
    helper.inference_in_background = True
    helper._pending_inference = (helper._inference_epoch, future)
    helper.run_step()
    assert helper._pending_inference is None
    assert helper.last_prediction["gesture"] == "up"
    np.testing.assert_allclose(helper.last_prediction["velocity_sent"], gesture_velocity("up"))


def test_accepted_command_is_immediate_and_stop_has_no_tail(live):
    helper, _, _, _ = live
    configure_continuous(helper)
    helper.max_linear_speed = 1.0e6
    target = gesture_velocity("up")
    prediction = {"mode": "move", "mode_conf": 0.99, "gesture": "up", "finger_idx": 0,
                  "velocity_sent": target}
    helper._apply_prediction(prediction)
    assert not any(helper.last_prediction["velocity_sent"])
    helper._apply_prediction(prediction)
    np.testing.assert_allclose(helper.last_prediction["velocity_sent"], target)
    helper._apply_prediction({"mode": "stop", "mode_conf": 1.0, "gesture": "stop", "finger_idx": 0,
                              "velocity_sent": np.zeros(6, dtype=np.float32)})
    assert not any(helper.last_prediction["velocity_sent"])


def test_continuous_maximum_contact_remains_bounded():
    buffer = TouchPrefixBuffer(config=CONTINUOUS_PREPROCESS_CONFIG)
    zero = np.zeros((10, 10))
    for i in range(15):
        buffer.feed(i, 10 + i * 0.02, zero, zero)
    for i in range(15, 1530):
        buffer.feed(i, 10 + i * 0.02, zero - 8, zero - 8)
    assert buffer.state == "release"
    assert not buffer.frames
    assert buffer.snapshot() is None


def test_continuous_noise_guard_learns_hand_away_noise_only():
    buffer = TouchPrefixBuffer(config=CONTINUOUS_PREPROCESS_CONFIG)
    for i in range(20):
        value = np.zeros((10, 10)) + (0.5 if i % 2 else 0)
        buffer.feed(i, 10 + i * 0.02, value, value)
    assert buffer.motion_threshold == pytest.approx(1.5)
    for i in range(20, 40):
        value = np.zeros((10, 10)) - i
        buffer.feed(i, 10 + i * 0.02, value, value)
    assert buffer.motion_threshold == pytest.approx(1.5)


@pytest.mark.parametrize("raises", [False, True])
def test_robot_send_failure_is_not_reported_as_running(live, raises):
    helper, frame, _, data = live
    arm_and_touch(frame)
    helper.dry_run_predictions_only = False
    if raises:
        def failed(*args, **kwargs):
            raise RuntimeError("Disconnected transport")
        helper._send_robot_velocity_execution = failed
    # There is deliberately no robot API in this fixture.
    frame(True)
    assert not helper.is_running
    assert not data._frame_listeners
    assert not any(helper.last_prediction["velocity_sent"])
    assert "Robot command" in helper.last_prediction["reason"]

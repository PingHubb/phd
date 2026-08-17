from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import QCoreApplication

from phd.dependence.gesture.gesture_logic_direct_finger_motion import (
    AI_DirectFingerMotion_execution,
    DirectFingerMotion,
)
from phd.dependence.func_sensor import MySensor
from phd.dependence.robot_api import RobotController
from phd.script.benchmark_ai_direct_finger_motion_robustness import (
    combine_scores,
    default_trials,
    score,
    stable_perturbation_seed,
)
from phd.script.train_ai_direct_finger_motion import (
    Episode,
    split_episodes_with_test,
)


_QT_APP = QCoreApplication.instance() or QCoreApplication([])


class _Sensor:
    n_row = 2
    n_col = 2

    def __init__(self):
        self._data = SimpleNamespace(
            diffPerData=np.zeros((2, 2), dtype=np.float32),
            diffPerDataAve=np.zeros((2, 2), dtype=np.float32),
            frame_sequence=0,
        )


class _RobotApi:
    def __init__(self, results):
        self.results = list(results)
        self.commands = []

    @staticmethod
    def enter_end_effector_velocity_mode():
        return True

    def send_end_effector_velocity_in_frame(
        self, linear, angular, frame="tool", ensure_mode=False
    ):
        self.commands.append((list(linear), list(angular), frame, ensure_mode))
        return self.results.pop(0) if self.results else True


def _splitter(robot_api=None):
    return SimpleNamespace(
        robot_api=robot_api,
        ai_selected_frame="tool",
        ai_frame_input=None,
        log_display=None,
    )


def test_dfm_retries_failed_velocity_and_only_caches_success():
    robot = _RobotApi([False, True, True])
    helper = DirectFingerMotion(_splitter(robot), _Sensor())

    assert not helper._send_robot_velocity([0.1, 0, 0, 0, 0, 0])
    assert helper.last_robot_velocity_cmd is None

    assert helper._send_robot_velocity([0.1, 0, 0, 0, 0, 0])
    assert helper.last_robot_velocity_cmd == [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert len(robot.commands) == 2

    assert helper._send_robot_velocity([0.1, 0, 0, 0, 0, 0])
    assert len(robot.commands) == 2
    helper._last_velocity_send_time -= (
        helper.motion_command_keepalive_sec + 0.01
    )
    assert helper._send_robot_velocity([0.1, 0, 0, 0, 0, 0])
    assert len(robot.commands) == 3


def test_ai_requires_two_consistent_motion_predictions():
    helper = AI_DirectFingerMotion_execution(_splitter(), _Sensor())
    helper.dry_run_predictions_only = True
    prediction = {
        "mode": "move",
        "mode_conf": 0.95,
        "finger_idx": 1,
        "velocity_sent": np.array([0.02, 0, 0, 0, 0, 0], dtype=np.float32),
    }

    helper._apply_prediction(prediction)
    assert helper.last_prediction["mode"] == "stop"
    helper._apply_prediction(prediction)
    assert helper.last_prediction["mode"] == "move"


def test_checkpoint_signature_detects_replaced_model(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"first")
    first = AI_DirectFingerMotion_execution._checkpoint_signature(checkpoint)
    checkpoint.write_bytes(b"second-version")
    second = AI_DirectFingerMotion_execution._checkpoint_signature(checkpoint)
    assert first != second


def test_sensor_render_consumes_only_the_latest_pending_snapshot():
    sensor = object.__new__(MySensor)
    sensor._is_shutting_down = False
    sensor.is_connected = True
    sensor.main_visualization_enabled = True
    sensor._heatmap_playback_active = False
    sensor._pending_sensor_visualization_matrix = np.ones((2, 2), dtype=float)
    sensor.saved_camera = 0
    sensor.rendered = []
    sensor.saveCameraPara = lambda: setattr(
        sensor, "saved_camera", sensor.saved_camera + 1
    )
    sensor.update_visualization = lambda matrix: sensor.rendered.append(
        np.array(matrix, copy=True)
    )

    sensor._render_pending_sensor_visualization()
    sensor._render_pending_sensor_visualization()

    assert sensor.saved_camera == 1
    assert len(sensor.rendered) == 1
    assert sensor._pending_sensor_visualization_matrix is None


class _ReadyClient:
    def __init__(self, ready):
        self.ready = bool(ready)

    def service_is_ready(self):
        return self.ready


def test_robot_availability_refreshes_after_services_appear(monkeypatch):
    import phd.dependence.robot_api as robot_api_module

    monkeypatch.setattr(robot_api_module, "HAND_SRVS_AVAILABLE", True)
    api = object.__new__(RobotController)
    api.use_ros = True
    api._node_started = True
    api.client = _ReadyClient(False)
    api.send_script_client = _ReadyClient(False)
    api.event_client = _ReadyClient(False)
    api.hand_set_angle_client = _ReadyClient(False)
    api.hand_set_speed_client = _ReadyClient(False)
    api.hand_set_force_client = _ReadyClient(False)
    api.hand_get_angle_client = _ReadyClient(False)
    api._hand_topic_commands_available = lambda refresh=False: False
    api.hand_tactile_publisher_count = lambda: 0

    assert not api.refresh_availability()["robot"]
    api.send_script_client.ready = True
    assert api.refresh_availability()["robot"]


def test_robot_velocity_state_tracks_nonzero_and_stop_commands():
    api = object.__new__(RobotController)
    api._velocity_state_lock = threading.Lock()
    api._end_effector_velocity_mode_active = False
    api._joint_velocity_mode_active = False
    api._velocity_command_last_at = 0.0
    api._velocity_command_nonzero = False

    api._note_script_mode_transition("ContinueVLine(20000,100000)")
    api._note_script_mode_transition("SetContinueVLine(0.02,0,0,0,0,0)")
    assert api._end_effector_velocity_mode_active
    assert api._velocity_command_nonzero
    assert api._velocity_command_last_at > 0.0

    api._note_script_mode_transition("StopContinueVmode()")
    assert not api._end_effector_velocity_mode_active
    assert not api._velocity_command_nonzero


def test_benchmark_combines_global_error_counts_not_episode_rates():
    first = {
        "intended": np.zeros((1, 6), dtype=np.float32),
        "intended_mode": np.array(["stop"]),
    }
    second = {
        "intended": np.zeros((9, 6), dtype=np.float32),
        "intended_mode": np.array(["stop"] * 9),
    }
    first_score = score(
        {
            "velocity": np.array([[1, 1, 1, 0, 0, 0]], dtype=np.float32),
            "modes": np.array(["move"]),
        },
        first,
    )
    second_score = score(
        {
            "velocity": np.zeros((9, 6), dtype=np.float32),
            "modes": np.array(["stop"] * 9),
        },
        second,
    )

    combined = combine_scores([first_score, second_score])
    assert np.isclose(combined["rmse_linear"], np.sqrt(3.0 / 30.0))
    assert np.isclose(combined["stop_false_move_rate"], 0.1)
    assert np.isclose(combined["mode_agreement"], 0.9)


def test_perturbation_seed_is_stable_and_input_sensitive():
    seed = stable_perturbation_seed("dead_taxels", 0.2, 3, 42)
    assert seed == stable_perturbation_seed("dead_taxels", 0.2, 3, 42)
    assert seed != stable_perturbation_seed("dead_taxels", 0.2, 4, 42)


def test_benchmark_prefers_checkpoint_test_trials(tmp_path, monkeypatch):
    session = tmp_path / "session_demo"
    session.mkdir()
    validation = session / "trial_validation.npz"
    test = session / "trial_test.npz"
    validation.touch()
    test.touch()
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    fake_torch = SimpleNamespace(
        load=lambda *_args, **_kwargs: {
            "config": {
                "val_files": [str(validation)],
                "test_files": [str(test)],
            }
        }
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert default_trials(tmp_path, checkpoint) == [test]


def _episode(path):
    return Episode(
        path=Path(path),
        frames=np.zeros((2, 1, 2, 2), dtype=np.float32),
        aux=np.zeros((2, 1), dtype=np.float32),
        target=np.zeros((2, 6), dtype=np.float32),
        mode=np.zeros(2, dtype=np.int64),
        elapsed_sec=np.arange(2, dtype=np.float32),
        metadata={},
    )


def test_training_split_keeps_test_trials_out_of_train_and_validation():
    episodes = [_episode(f"trial_{index}.npz") for index in range(5)]
    train, validation, test = split_episodes_with_test(
        episodes, val_trials=1, test_trials=1
    )

    assert [item.path.name for item in train] == [
        "trial_0.npz",
        "trial_1.npz",
        "trial_2.npz",
    ]
    assert [item.path.name for item in validation] == ["trial_3.npz"]
    assert [item.path.name for item in test] == ["trial_4.npz"]

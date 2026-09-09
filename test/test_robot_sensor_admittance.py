import numpy as np

from phd.dependence.func_meshLab import MyMeshLab
from phd.ui.ui_ping_ai_controls import AiControlsMixin


class _FakeRobotApi:
    def __init__(self):
        self.commands = []
        self.exit_calls = []
        self.enter_calls = []
        self.tool_position = np.zeros(3, dtype=float)

    def enter_end_effector_velocity_mode(self, suspend_existing=False):
        self.enter_calls.append(bool(suspend_existing))
        return True

    def get_current_positions(self):
        return [0.0] * 6

    def get_current_tool_position(self):
        return tuple(self.tool_position), (1.0, 0.0, 0.0, 0.0)

    def send_end_effector_velocity_in_frame(
        self, v_lin, v_rot=(0.0, 0.0, 0.0), frame="tool", ensure_mode=True
    ):
        self.commands.append((list(v_lin), list(v_rot), frame, ensure_mode))
        return True

    def exit_end_effector_velocity_mode(self, send_zero=True):
        self.exit_calls.append(bool(send_zero))
        return True


class _FakeSensor:
    def __init__(self, estimate):
        self.n_row = 1
        self.n_col = 1
        self.normals = np.array([[0.0, 0.0, 1.0]], dtype=float)
        self.is_connected = True
        self._estimate = estimate
        self._data = type(
            "Data",
            (),
            {
                "diffPerData": np.zeros((1, 1)),
                "diffPerDataAve": np.zeros((1, 1)),
            },
        )()

    @staticmethod
    def _sensor_reader_is_running():
        return True

    def _estimate_contact_force_signal(self, _matrix, peak_threshold=None):
        return self._estimate


class _FakeParent:
    def __init__(self, sensor, robot_api):
        self.sensor_functions = sensor
        self.robot_api = robot_api

    def window(self):
        return self


class _FakeTimer:
    def __init__(self):
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def isActive(self):
        return self.active


class _FakeToggleButton:
    def __init__(self, checked=False, enabled=True):
        self.checked = bool(checked)
        self.enabled = bool(enabled)
        self.active_style = False

    def blockSignals(self, _blocked):
        return None

    def setChecked(self, checked):
        self.checked = bool(checked)

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _FakeAiControls(AiControlsMixin):
    @staticmethod
    def _set_button_active(button, active):
        button.active_style = bool(active)


class _FakeProximityMotionHelper:
    def __init__(self):
        self.stop_calls = 0
        self.start_calls = 0

    def start_ai_proximity_motion(self):
        self.start_calls += 1
        return True, "started"

    def stop_ai_proximity_motion(self):
        self.stop_calls += 1


def _make_lab(estimate):
    lab = object.__new__(MyMeshLab)
    robot_api = _FakeRobotApi()
    sensor = _FakeSensor(estimate)
    lab.parent = _FakeParent(sensor, robot_api)
    config = lab._default_robot_sensor_mapping_config()
    config.update(
        {
            "link_index": 0,
            "admittance_max_speed_mps": 0.03,
            "admittance_contact_threshold_pct": 3.0,
            "admittance_full_scale_pct": 12.0,
            "admittance_smoothing_alpha": 1.0,
        }
    )
    lab._robot_dialog_sensor_mapping_config = config
    lab._robot_dialog_admittance_active = True
    lab._robot_dialog_admittance_velocity_mode_on = True
    lab._robot_dialog_admittance_filtered_velocity = np.zeros(3, dtype=float)
    lab._robot_dialog_drag_active = False
    return lab, sensor, robot_api


def test_pressure_speed_mapping_is_bounded():
    speed = MyMeshLab._admittance_speed_from_pressure
    assert speed(2.9, 3.0, 12.0, 0.03) == 0.0
    assert np.isclose(speed(7.5, 3.0, 12.0, 0.03), 0.015)
    assert np.isclose(speed(100.0, 3.0, 12.0, 0.03), 0.03)
    assert np.isclose(speed(100.0, 3.0, 12.0, 2.0), 0.1)


def test_link5_overlay_uses_kinematic_frame_origin_and_axes():
    transform = np.array(
        [
            [0.0, -1.0, 0.0, 0.40],
            [1.0, 0.0, 0.0, -0.20],
            [0.0, 0.0, 1.0, 0.70],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    origin, axes = MyMeshLab._link_frame_origin_and_axes(transform)

    assert np.allclose(origin, [0.40, -0.20, 0.70])
    assert np.allclose(axes[:, 0], [0.0, 1.0, 0.0])
    assert np.allclose(axes[:, 1], [-1.0, 0.0, 0.0])
    assert np.allclose(axes[:, 2], [0.0, 0.0, 1.0])


def test_old_mapping_defaults_to_surface_normal_mode():
    config = MyMeshLab._normalize_robot_sensor_mapping_config(
        {"link_index": 5, "admittance_max_speed_mps": 0.02}
    )

    assert config["admittance_direction_mode"] == "surface_normal"
    assert config["admittance_control_center_m"] == [0.0, 0.0, 0.0]


def _set_square_sensor_geometry(sensor):
    sensor.n_row = 2
    sensor.n_col = 2
    # Column-major order: bottom-left, top-left, bottom-right, top-right.
    sensor.points_origin = np.array(
        [
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    sensor.normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    sensor._2D_map = type(
        "Map", (), {"points": np.array(sensor.points_origin, copy=True)}
    )()


def test_centre_directed_top_and_bottom_left_have_different_vertical_motion():
    estimate = {"center_row": 0.0, "center_col": 0.0}
    lab, sensor, _robot_api = _make_lab(estimate)
    _set_square_sensor_geometry(sensor)
    config = dict(lab._robot_dialog_sensor_mapping_config)
    config.update(
        {
            "admittance_direction_mode": "centre_directed",
            "admittance_control_center_m": [0.0, 0.0, -1.0],
        }
    )

    bottom_left = lab._robot_dialog_admittance_direction_base(
        sensor,
        {"center_row": 0.0, "center_col": 0.0},
        [0.0] * 6,
        config,
    )
    top_left = lab._robot_dialog_admittance_direction_base(
        sensor,
        {"center_row": 1.0, "center_col": 0.0},
        [0.0] * 6,
        config,
    )

    expected_bottom = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
    expected_top = np.array([1.0, -1.0, -1.0]) / np.sqrt(3.0)
    assert np.allclose(bottom_left, expected_bottom)
    assert np.allclose(top_left, expected_top)


def test_centre_directed_reverse_flips_complete_motion_vector():
    estimate = {"center_row": 1.0, "center_col": 0.0}
    lab, sensor, _robot_api = _make_lab(estimate)
    _set_square_sensor_geometry(sensor)
    config = dict(lab._robot_dialog_sensor_mapping_config)
    config.update(
        {
            "admittance_direction_mode": "centre_directed",
            "admittance_control_center_m": [0.0, 0.0, -1.0],
        }
    )
    forward = lab._robot_dialog_admittance_direction_base(
        sensor, estimate, [0.0] * 6, config
    )
    config["admittance_reverse_direction"] = True
    reverse = lab._robot_dialog_admittance_direction_base(
        sensor, estimate, [0.0] * 6, config
    )

    assert np.allclose(reverse, -forward)


def test_contact_normal_uses_saved_mount_rotation():
    estimate = {"center_row": 0.0, "center_col": 0.0}
    lab, sensor, _robot_api = _make_lab(estimate)
    config = dict(lab._robot_dialog_sensor_mapping_config)
    config["rotation_deg"] = [90.0, 0.0, 0.0]

    normal = lab._robot_dialog_contact_surface_normal_base(
        sensor, estimate, [0.0] * 6, config
    )

    assert np.allclose(normal, [0.0, -1.0, 0.0], atol=1e-7)


def test_contact_normal_handles_nonuniform_mount_scaling():
    estimate = {"center_row": 0.0, "center_col": 0.0}
    lab, sensor, _robot_api = _make_lab(estimate)
    sensor.normals = np.array([[1.0, 0.0, 1.0]], dtype=float) / np.sqrt(2.0)
    config = dict(lab._robot_dialog_sensor_mapping_config)
    config["horizontal_scale"] = 2.0
    config["vertical_scale"] = 0.5

    normal = lab._robot_dialog_contact_surface_normal_base(
        sensor, estimate, [0.0] * 6, config
    )

    expected = np.array([0.5, 0.0, 1.0], dtype=float)
    expected /= np.linalg.norm(expected)
    assert np.allclose(normal, expected, atol=1e-7)


def test_no_contact_sends_immediate_zero_velocity():
    lab, _sensor, robot_api = _make_lab(None)

    lab._robot_dialog_admittance_tick()

    assert robot_api.commands[-1][0] == [0.0, 0.0, 0.0]
    assert robot_api.commands[-1][2] == "base"
    assert robot_api.commands[-1][3] is False


def test_raw_contact_loss_overrides_residual_averaged_pressure():
    residual_contact = {
        "center_row": 0.0,
        "center_col": 0.0,
        "peak_pressure": 12.0,
    }
    lab, sensor, robot_api = _make_lab(None)

    def _estimate(matrix, peak_threshold=None):
        if matrix is sensor._data.diffPerData:
            return None
        return residual_contact

    sensor._estimate_contact_force_signal = _estimate
    lab._robot_dialog_admittance_filtered_velocity = np.array(
        [0.0, 0.0, -0.03]
    )

    lab._robot_dialog_admittance_tick()

    assert robot_api.commands[-1][0] == [0.0, 0.0, 0.0]


def test_contact_moves_opposite_surface_normal_at_bounded_speed():
    estimate = {
        "center_row": 0.0,
        "center_col": 0.0,
        "peak_pressure": 12.0,
    }
    lab, _sensor, robot_api = _make_lab(estimate)

    lab._robot_dialog_admittance_tick()

    velocity = np.asarray(robot_api.commands[-1][0], dtype=float)
    assert np.allclose(velocity, [0.0, 0.0, -0.03], atol=1e-7)
    assert np.linalg.norm(velocity) <= 0.03 + 1e-9


def test_teardown_exits_velocity_mode_with_zero():
    lab, _sensor, robot_api = _make_lab(None)

    lab._teardown_robot_dialog_admittance(update_status=False)

    assert robot_api.exit_calls == [True]
    assert lab._robot_dialog_admittance_active is False
    assert lab._robot_dialog_admittance_velocity_mode_on is False


def test_ai_proximity_retreats_then_returns_to_start():
    lab, _sensor, robot_api = _make_lab(None)
    lab._robot_dialog_admittance_active = False
    lab._robot_dialog_admittance_velocity_mode_on = False
    lab._ai_proximity_velocity_mode_on = False
    lab._ai_proximity_filtered_velocity = np.zeros(3, dtype=float)
    lab._ai_proximity_mapping_config = None
    config = dict(lab._robot_dialog_sensor_mapping_config)
    lab._load_robot_sensor_mapping_config = lambda: config

    started, _message = lab.start_ai_proximity_motion()
    moving = lab.update_ai_proximity_motion(
        0.0,
        0.0,
        2.0,
        detected=True,
        dry_run=False,
    )
    robot_api.tool_position[:] = [0.0, 0.0, -0.02]
    returning = lab.update_ai_proximity_motion(
        None,
        None,
        0.5,
        detected=False,
        dry_run=False,
    )
    robot_api.tool_position[:] = [0.0, 0.0, -0.001]
    arrived = lab.update_ai_proximity_motion(
        None,
        None,
        0.5,
        detected=False,
        dry_run=False,
    )
    lab.stop_ai_proximity_motion()

    assert started is True
    assert moving["ok"] is True
    assert np.allclose(robot_api.commands[1][0], [0.0, 0.0, -0.03])
    assert returning["ok"] is True
    assert returning["returning"] is True
    assert np.allclose(robot_api.commands[2][0], [0.0, 0.0, 0.03])
    assert arrived["ok"] is True
    assert arrived["at_start"] is True
    assert robot_api.commands[3][0] == [0.0, 0.0, 0.0]
    assert robot_api.exit_calls == [True]


def test_ai_proximity_dry_run_estimates_motion_without_robot_command():
    lab, _sensor, robot_api = _make_lab(None)
    lab._robot_dialog_admittance_active = False
    lab._ai_proximity_velocity_mode_on = False
    lab._ai_proximity_filtered_velocity = np.zeros(3, dtype=float)
    config = dict(lab._robot_dialog_sensor_mapping_config)
    lab._load_robot_sensor_mapping_config = lambda: config

    preview = lab.update_ai_proximity_motion(
        0.0,
        0.0,
        2.0,
        detected=True,
        dry_run=True,
    )

    assert preview["ok"] is True
    assert np.allclose(preview["direction"], [0.0, 0.0, -1.0])
    assert np.isclose(preview["speed_mps"], 0.03)
    assert robot_api.commands == []
    assert robot_api.enter_calls == []


def test_stopping_ai_detection_disables_proximity_admittance_button():
    controls = object.__new__(_FakeAiControls)
    controls.mesh_functions = _FakeProximityMotionHelper()
    controls.ai_proximity_detection_button = _FakeToggleButton(checked=True)
    controls.ai_proximity_admittance_button = _FakeToggleButton(checked=True)
    controls._ai_proximity_detection_active = True
    controls._ai_proximity_robot_motion_active = True
    controls._ai_proximity_timer = None
    controls._ai_proximity_pending = None
    controls._ai_proximity_executor = None
    controls._ai_proximity_last_frame_sequence = 10
    controls._ai_proximity_previous_averaged_frame = np.zeros((1, 1))
    controls._ai_proximity_state = "detected"

    controls._stop_ai_proximity_detection()

    assert controls.mesh_functions.stop_calls == 1
    assert controls._ai_proximity_robot_motion_active is False
    assert controls.ai_proximity_detection_button.checked is False
    assert controls.ai_proximity_admittance_button.checked is False
    assert controls.ai_proximity_admittance_button.enabled is False
    assert controls.ai_proximity_admittance_button.active_style is False


def test_proximity_admittance_starts_without_confirmation_dialog():
    controls = object.__new__(_FakeAiControls)
    controls.mesh_functions = _FakeProximityMotionHelper()
    controls._ai_proximity_motion_confirmed = False
    controls._ai_proximity_robot_motion_active = False

    started = controls._prepare_ai_proximity_robot_motion()

    assert started is True
    assert controls.mesh_functions.start_calls == 1
    assert controls._ai_proximity_motion_confirmed is True
    assert controls._ai_proximity_robot_motion_active is True


def test_ai_tab_can_start_admittance_without_robot_dialog_or_actor():
    lab, _sensor, robot_api = _make_lab(None)
    lab._robot_dialog_admittance_active = False
    lab._robot_dialog_admittance_velocity_mode_on = False
    lab._robot_dialog_admittance_confirmed = True
    lab._robot_dialog_admittance_timer = _FakeTimer()
    lab._admittance_source = None
    lab._admittance_pending_start = False
    lab._admittance_mapping_config = None

    lab.set_ai_admittance_control_enabled(True)

    assert robot_api.enter_calls == [True]
    assert lab._robot_dialog_admittance_active is True
    assert lab._admittance_source == "ai_tab"
    assert lab._robot_dialog_admittance_timer.isActive() is True


def test_ai_tab_auto_starts_sensor_before_admittance():
    lab, sensor, robot_api = _make_lab(None)
    sensor.is_connected = False
    sensor.running = False
    sensor.start_calls = 0
    sensor.main_visualization_enabled = False
    sensor._sensor_reader_is_running = lambda: sensor.running

    def _start_sensor():
        sensor.start_calls += 1
        sensor.running = True
        sensor.is_connected = True
        return True

    def _show_main_sensor(enabled, render=True):
        sensor.main_visualization_enabled = bool(enabled)

    sensor.start_external_visualization_stream = _start_sensor
    sensor.set_main_visualization_enabled = _show_main_sensor
    lab._robot_dialog_admittance_active = False
    lab._robot_dialog_admittance_velocity_mode_on = False
    lab._robot_dialog_admittance_confirmed = True
    lab._robot_dialog_admittance_timer = _FakeTimer()
    lab._admittance_source = None
    lab._admittance_pending_start = False
    lab._admittance_mapping_config = None

    lab.set_ai_admittance_control_enabled(True)

    assert sensor.start_calls == 1
    assert sensor.main_visualization_enabled is True
    assert robot_api.enter_calls == [True]
    assert lab._robot_dialog_admittance_active is True

import numpy as np

from phd.dependence.func_meshLab import MyMeshLab


class _FakeRobotApi:
    def __init__(self):
        self.commands = []
        self.exit_calls = []
        self.enter_calls = []

    def enter_end_effector_velocity_mode(self, suspend_existing=False):
        self.enter_calls.append(bool(suspend_existing))
        return True

    def get_current_positions(self):
        return [0.0] * 6

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

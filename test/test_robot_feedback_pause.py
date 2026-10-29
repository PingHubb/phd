"""Read-only controller feedback used by click-to-taxel navigation."""

from types import SimpleNamespace

from phd.dependence.robot_api import RobotController


def test_feedback_tracks_pause_without_sending_commands():
    # No ROS node, services or robot connection is constructed by these tests.
    api = object.__new__(RobotController)
    api._set_offline_defaults()
    assert api.current_project_paused is None
    api._feedback_state_cb(SimpleNamespace(project_speed=30, project_pause=True))
    assert api.current_project_paused is True
    assert api.current_project_speed_percent == 30
    assert api.current_feedback_state_time is not None
    api._feedback_state_cb(SimpleNamespace(project_speed=30, project_pause=False))
    assert api.current_project_paused is False


def test_legacy_feedback_without_pause_field_is_unknown_not_paused():
    api = object.__new__(RobotController)
    api._set_offline_defaults()
    api._feedback_state_cb(SimpleNamespace(project_speed=30))
    assert api.current_project_paused is None

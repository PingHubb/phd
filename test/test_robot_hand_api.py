from types import SimpleNamespace

from phd.dependence.robot_api import RobotController


class _FakeClient:
    @staticmethod
    def call_async(_request):
        return object()


def _service_api(values):
    api = object.__new__(RobotController)
    api.hand_get_angle_client = _FakeClient()
    api._hand_service_commands_available = lambda: True
    api._wait_future = lambda _future, timeout_sec=1.5: SimpleNamespace(
        curangleact=list(values)
    )
    return api


def test_hand_angle_service_response_is_normalized_for_ui_sliders():
    api = _service_api([1731, 1727, 1726, 1735, 1351, 1803])

    result = api.hand_get_actual_angles()

    assert [getattr(result, f"angle{index}") for index in range(6)] == [
        1731,
        1727,
        1726,
        1735,
        1351,
        1803,
    ]


def test_short_hand_angle_service_response_is_rejected():
    api = _service_api([1000, 1000])

    assert api.hand_get_actual_angles() is None

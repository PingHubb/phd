"""Four-way tactile swipe recognition for the Robot Dog controls."""

import numpy as np
import pytest

from phd.ui.ui_ping_robot_dog import (
    ROBOT_DOG_SWIPE_COMMANDS,
    RobotDogControlsMixin,
    SingleFingerSwipeDetector,
)


def _frame(*points):
    values = np.zeros((8, 8), dtype=float)
    for row, column in points:
        values[row, column] = -10.0
    return values


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([(3, 1), (3, 2), (3, 3)], "right"),
        ([(3, 6), (3, 5), (3, 4)], "left"),
        ([(6, 3), (5, 3), (4, 3)], "up"),
        ([(1, 3), (2, 3), (3, 3)], "down"),
    ],
)
def test_detects_each_single_finger_swipe_direction(points, expected):
    detector = SingleFingerSwipeDetector()
    detections = [
        detector.process(_frame(point), timestamp=index * 0.05)
        for index, point in enumerate(points)
    ]

    assert expected in detections


def test_emits_only_once_until_the_finger_is_released():
    detector = SingleFingerSwipeDetector()

    detections = [
        detector.process(_frame((3, column)), timestamp=index * 0.05)
        for index, column in enumerate((1, 2, 3, 4, 5))
    ]

    assert detections.count("right") == 1


def test_diagonal_motion_is_not_forced_into_a_cardinal_direction():
    detector = SingleFingerSwipeDetector()

    detections = [
        detector.process(_frame(point), timestamp=index * 0.05)
        for index, point in enumerate(((1, 1), (2, 2), (4, 4)))
    ]

    assert all(direction is None for direction in detections)


def test_multiple_touch_clusters_block_the_gesture_until_release():
    detector = SingleFingerSwipeDetector()
    detector.process(_frame((3, 1), (3, 6)), timestamp=0.0)

    blocked = [
        detector.process(
            _frame((3, column)),
            timestamp=0.1 + index * 0.05,
        )
        for index, column in enumerate((1, 2, 3, 4))
    ]
    detector.process(_frame(), timestamp=0.4)
    detector.process(_frame(), timestamp=0.45)
    after_release = [
        detector.process(
            _frame((3, column)),
            timestamp=0.5 + index * 0.05,
        )
        for index, column in enumerate((1, 2, 3))
    ]

    assert all(direction is None for direction in blocked)
    assert "right" in after_release


def test_swipe_directions_map_to_requested_open_cat_commands():
    assert ROBOT_DOG_SWIPE_COMMANDS == {
        "up": "kup",
        "down": "ksit",
        "left": "kstr",
        "right": "khi",
    }


def test_robot_dog_command_uses_the_existing_primary_sensor_port():
    class _Serial:
        port = "/dev/ttyACM6"
        is_open = True

        def __init__(self):
            self.writes = []

        def write(self, payload):
            self.writes.append(bytes(payload))
            return len(payload)

    serial_port = _Serial()
    sensor = type(
        "_Sensor",
        (),
        {
            "ser_list": [serial_port],
            "_primary_sensor_port": "/dev/ttyACM6",
        },
    )()
    harness = type(
        "_Harness",
        (RobotDogControlsMixin,),
        {"sensor_functions": sensor},
    )()

    sent, error = harness._send_robot_dog_command("kup")

    assert sent is True
    assert error == ""
    assert serial_port.writes == [b"kup\n"]

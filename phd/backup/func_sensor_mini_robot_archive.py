"""
Archived mini-robot sensor model hooks extracted from `dependence/func_sensor.py`.

Removed from active flow:
- `init_mini_robot_large_skin()`
- `init_mini_robot_small_skin()`
- sensor-choice buildScene routing for mini-robot skins

The active sensor UI no longer offers the mini-robot skin models.
"""


class MiniRobotSensorArchiveReference:
    def init_mini_robot_large_skin(self):
        raise NotImplementedError("Archived mini-robot sensor model is not active.")

    def init_mini_robot_small_skin(self):
        raise NotImplementedError("Archived mini-robot sensor model is not active.")

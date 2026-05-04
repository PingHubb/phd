"""
Archived Geneva-specific sensor support extracted from `dependence/func_sensor.py`.

This file is intentionally not imported by the active application. It exists as a
local archive after removing the Geneva demo entry points from the main UI and
sensor runtime.

Archived feature surface:
- `init_geneva_demo()`
- `startSensor_geneva()`
- `update_animation_geneva()`
- `update_visualization_geneva()`
- `updateCal_geneva()`

Notes:
- The old Geneva demo was a separate multi-sensor visualization / calibration flow.
- The active app no longer exposes the `Geneva Demo` sensor option.
- The active app no longer exposes the `startSensorGeneva` / `updateSensorGeneva`
  testing buttons.
- If Geneva support is needed again later, restore the implementation from version
  history or reintroduce the archived methods into a dedicated runtime module.
"""


class GenevaArchiveReference:
    """
    Lightweight archive marker for the retired Geneva demo path.

    This class is intentionally non-functional. It documents the public method
    names that were removed from the active runtime so future restoration work
    has a single obvious starting point inside `backup/`.
    """

    def init_geneva_demo(self):
        raise NotImplementedError("Archived Geneva demo support is not active.")

    def startSensor_geneva(self):
        raise NotImplementedError("Archived Geneva demo support is not active.")

    def update_animation_geneva(self):
        raise NotImplementedError("Archived Geneva demo support is not active.")

    def update_visualization_geneva(self, data, sensor_index):
        raise NotImplementedError("Archived Geneva demo support is not active.")

    def updateCal_geneva(self):
        raise NotImplementedError("Archived Geneva demo support is not active.")

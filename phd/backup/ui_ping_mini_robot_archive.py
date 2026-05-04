"""
Archived mini-robot UI support extracted from `ui/ui_ping.py`.

This archive marker documents the mini-robot runtime/UI pieces that were removed
from the active application flow:
- `MiniRobotToolPositionController`
- `setup_tab5()` mini-robot subtab
- mini-robot connect/status/motion controls

The active app no longer exposes the `Mini Robot` robots subtab.
"""


class MiniRobotUiArchiveReference:
    def setup_tab5(self, layout):
        raise NotImplementedError("Archived mini-robot UI support is not active.")


class MiniRobotToolPositionControllerArchiveReference:
    def send_coords(self):
        raise NotImplementedError("Archived mini-robot UI support is not active.")

    def send_angles(self):
        raise NotImplementedError("Archived mini-robot UI support is not active.")

    def get_coords(self):
        raise NotImplementedError("Archived mini-robot UI support is not active.")

    def stop(self):
        raise NotImplementedError("Archived mini-robot UI support is not active.")

    def pause(self):
        raise NotImplementedError("Archived mini-robot UI support is not active.")

    def resume(self):
        raise NotImplementedError("Archived mini-robot UI support is not active.")

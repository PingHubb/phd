"""RealSense preview device selection and window focusing."""

import os

from phd.ui.camera_preview_window import realsense_color_candidates
from phd.ui.ui_initial import MyMainWindow


def test_realsense_color_candidates_use_stable_index_order(tmp_path):
    names = [
        "usb-Intel_R__RealSense_TM__Depth_Camera_555-video-index2",
        "usb-Intel_R__RealSense_TM__Depth_Camera_555-video-index0",
        "usb-Logitech_Webcam-video-index0",
        "notes.txt",
    ]
    for name in names:
        (tmp_path / name).write_text("")

    found = realsense_color_candidates(str(tmp_path))

    assert [os.path.basename(path) for path in found] == [
        "usb-Intel_R__RealSense_TM__Depth_Camera_555-video-index0",
        "usb-Intel_R__RealSense_TM__Depth_Camera_555-video-index2",
    ]


class _Preview:
    def __init__(self):
        self.visible = True
        self.shown = 0

    def isVisible(self):
        return self.visible

    def show(self):
        self.shown += 1
        self.visible = True


def test_camera_action_focuses_an_open_preview():
    preview = _Preview()
    calls = []
    window = type("Host", (), {})()
    window.camera_preview_window = preview
    window._focus_window = lambda target: calls.append(target)

    MyMainWindow.open_camera_preview_window(window)

    assert calls == [preview]
    assert preview.shown == 1

"""RealSense preview device selection and window focusing."""

import os

import numpy as np
import pytest

from phd.ui.camera_preview_window import (
    available_view_modes,
    depth_meters,
    depth_sample_meters,
    depth_to_rgb,
    format_distance,
    image_pixel_at,
    overlay_color_and_depth,
    point_cloud_to_rgb,
    realsense_color_candidates,
    vertices_from_points,
)
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


def test_color_and_depth_enable_derived_overlay_and_point_cloud_views():
    assert available_view_modes(["color", "depth"]) == [
        "color",
        "depth",
        "overlay",
        "points",
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


def test_depth_colors_near_red_and_leaves_missing_pixels_black():
    depth = np.array([[0, 150], [3000, 3000]], dtype=np.uint16)

    rgb = depth_to_rgb(depth, depth_scale=0.001, near_m=0.15, far_m=3.0)

    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0].tolist() == [0, 0, 0]
    assert rgb[0, 1, 0] > rgb[1, 0, 0]
    assert rgb[1, 0, 2] > rgb[0, 1, 2]


def test_overlay_keeps_color_where_depth_is_missing():
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    color[..., 0] = 10
    color[..., 1] = 20
    color[..., 2] = 30
    depth = np.zeros((2, 2), dtype=np.uint16)
    depth[0, 0] = 1000

    rgb = overlay_color_and_depth(color, depth, depth_scale=0.001)

    assert rgb[1, 1].tolist() == [30, 20, 10]
    assert not np.array_equal(rgb[0, 0], [30, 20, 10])


def test_point_cloud_draws_a_finite_point_and_ignores_empty_depth():
    vertices = np.array(
        [[0.0, 0.0, 1.0], [np.nan, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    image = point_cloud_to_rgb(vertices, width=40, height=30)

    assert image.shape == (30, 40, 3)
    assert image.sum() > 0


def test_named_realsense_vertices_become_rows():
    raw = np.zeros(2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    raw["z"] = [1, 2]

    vertices = vertices_from_points(raw)

    assert vertices.shape == (2, 3)
    assert vertices[1, 2] == 2


def test_depth_meters_and_distance_text():
    meters = depth_meters(np.array([[0, 1250]], dtype=np.uint16), 0.001)

    assert meters[0, 0] == 0
    assert meters[0, 1] == 1.25
    assert format_distance(1.25) == "1.25 m  (125 cm)"
    assert format_distance(0) == "no depth"


def test_depth_readout_uses_local_median_and_ignores_holes():
    depth = np.zeros((7, 7), dtype=np.float32)
    depth[2:5, 2:5] = 0.8
    depth[3, 3] = 9.0
    depth[2, 2] = np.nan

    measured = depth_sample_meters(depth, column=3, row=3, radius=2)

    assert measured == pytest.approx(0.8)
    assert depth_sample_meters(depth, column=-1, row=3) is None


def test_cursor_maps_into_the_centered_picture():
    assert image_pixel_at(50, 0, 100, 100, 200, 100) == (0, 0)
    assert image_pixel_at(0, 0, 100, 100, 200, 100) is None
    assert image_pixel_at(149, 50, 100, 100, 200, 100) == (99, 50)

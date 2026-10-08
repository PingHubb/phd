"""Unit tests for depth processing used by live object detection."""

import numpy as np
import pytest

from phd.dependence.camera_api import median_depth_in_box


def test_object_depth_uses_valid_median_instead_of_center_pixel():
    depth = np.zeros((20, 20), dtype=np.uint16)
    depth[7:14, 7:14] = 420
    depth[10, 10] = 0
    depth[9, 9] = 5000

    distance = median_depth_in_box(depth, (5, 5, 15, 15), 0.001)

    assert distance == pytest.approx(0.42)


def test_object_depth_returns_zero_when_no_valid_measurement_exists():
    depth = np.zeros((10, 10), dtype=np.uint16)

    assert median_depth_in_box(depth, (2, 2, 8, 8), 0.001) == 0.0
    assert median_depth_in_box(depth, (2, 2, 8, 8), 0.0) == 0.0

import numpy as np
import pyvista as pv

from phd.dependence.func_sensor import MySensor


def _sensor_for_visibility_tests():
    sensor = MySensor.__new__(MySensor)
    sensor.n_row = 2
    sensor.n_col = 3
    sensor.points = np.asarray(
        [[float(col), float(row), 0.0] for col in range(3) for row in range(2)],
        dtype=float,
    )
    sensor.points_origin = np.array(sensor.points, copy=True)
    sensor.normals = np.tile([0.0, 0.0, 1.0], (6, 1))
    sensor.colors_3d = np.ones((6, 4), dtype=float)
    sensor.colors = np.ones((12, 4), dtype=float)
    sensor.array_positions = [
        [2 * index, 2 * index + 1] for index in range(6)
    ]
    sensor.line_poly = pv.PolyData(sensor.points)
    sensor.line_poly.lines = np.asarray(
        [
            [2, 0, 1],
            [2, 2, 3],
            [2, 4, 5],
            [2, 0, 2],
            [2, 2, 4],
            [2, 1, 3],
            [2, 3, 5],
        ],
        dtype=int,
    )
    sensor._2D_map = pv.PolyData(np.repeat(sensor.points, 2, axis=0))
    sensor.cell_zero_mask = np.zeros((2, 3), dtype=bool)
    sensor.matrixLineColors = None
    sensor.matrixLinePoly = None
    sensor._matrix_line_dense_shape = (0, 0)
    sensor._matrix_line_base_indices = None
    sensor._matrix_line_top_indices = None
    sensor.sensorPointLabelActor = None
    sensor.show_sensor_point_labels = False
    return sensor


def test_setting_mask_hides_coarse_and_dense_points_immediately():
    sensor = _sensor_for_visibility_tests()
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True

    assert sensor.set_cell_zero_mask(mask)

    masked_index = 2
    assert np.all(np.isnan(sensor.line_poly.points[masked_index]))
    assert np.all(np.isfinite(np.delete(sensor.line_poly.points, masked_index, axis=0)))
    assert sensor.colors_3d[masked_index, 3] == 0.0
    np.testing.assert_allclose(sensor.colors_3d[np.arange(6) != masked_index, 3], 1.0)
    np.testing.assert_allclose(sensor.colors[[4, 5], 3], 0.0)
    assert np.count_nonzero(sensor.colors[:, 3] == 0.0) == 2


def test_clearing_mask_restores_points_and_opacity():
    sensor = _sensor_for_visibility_tests()
    mask = np.zeros((2, 3), dtype=bool)
    mask[1, 2] = True
    sensor.set_cell_zero_mask(mask)

    sensor.clear_cell_zero_mask()

    np.testing.assert_allclose(sensor.line_poly.points, sensor.points)
    np.testing.assert_allclose(sensor.colors_3d[:, 3], 1.0)
    np.testing.assert_allclose(sensor.colors[:, 3], 1.0)


def test_zero_mask_expands_to_nearest_dense_taxel_regions():
    sensor = _sensor_for_visibility_tests()
    sensor.cell_zero_mask[1, 1] = True

    dense_mask = sensor._dense_cell_zero_mask(4, 6)

    expected_rows = np.rint(np.linspace(0, 1, 4)).astype(int)
    expected_cols = np.rint(np.linspace(0, 2, 6)).astype(int)
    expected = sensor.cell_zero_mask[
        expected_rows[:, None], expected_cols[None, :]
    ]
    np.testing.assert_array_equal(dense_mask, expected)
    assert dense_mask.any()


def test_stereo_field_alpha_hides_both_ends_of_masked_lines():
    sensor = _sensor_for_visibility_tests()
    sensor.cell_zero_mask[0, 1] = True
    dense_shape = (4, 6)
    line_count = dense_shape[0] * dense_shape[1]
    base_indices = np.arange(line_count) * 2
    top_indices = base_indices + 1
    colors = np.full((line_count * 2, 4), 255, dtype=np.uint8)

    sensor._apply_zero_mask_to_matrix_colors(
        colors,
        dense_shape,
        base_indices=base_indices,
        top_indices=top_indices,
    )

    dense_mask = sensor._dense_cell_zero_mask(*dense_shape).reshape(-1)
    np.testing.assert_array_equal(colors[base_indices, 3] == 0, dense_mask)
    np.testing.assert_array_equal(colors[top_indices, 3] == 0, dense_mask)


class _LabelPlotter:
    def __init__(self):
        self.labels = None
        self.points = None

    def add_point_labels(self, points, labels, **_kwargs):
        self.points = np.array(points.points, copy=True)
        self.labels = list(labels)
        return object()

    def remove_actor(self, *_args, **_kwargs):
        return None


def test_masked_taxel_is_omitted_from_point_labels():
    sensor = _sensor_for_visibility_tests()
    sensor.plotter = _LabelPlotter()
    sensor.show_sensor_point_labels = True
    sensor.cell_zero_mask[0, 1] = True

    sensor._refresh_sensor_point_label_actor()

    assert sensor.plotter.labels == [
        "P0 r0 c0",
        "P1 r1 c0",
        "P3 r1 c1",
        "P4 r0 c2",
        "P5 r1 c2",
    ]
    assert len(sensor.plotter.points) == 5


def test_point_labels_use_same_top_down_coordinates_as_zero_mask():
    sensor = _sensor_for_visibility_tests()

    labels = sensor._sensor_point_labels()

    assert labels[0] == "P0 r0 c0"
    assert labels[1] == "P1 r1 c0"
    assert sensor.get_cell_point_label(0, 1) == "P2 r0 c1"
    assert sensor.get_cell_point_label(1, 1) == "P3 r1 c1"


def test_7x7_top_right_mask_hides_reported_displayed_point_ids():
    sensor = MySensor.__new__(MySensor)
    sensor.n_row = 7
    sensor.n_col = 7
    sensor.points = np.asarray(
        [[float(col), float(row), 0.0] for col in range(7) for row in range(7)],
        dtype=float,
    )
    sensor.cell_zero_mask = np.zeros((7, 7), dtype=bool)
    sensor.cell_zero_mask[0:2, 5:7] = True

    displayed = sensor._sensor_points_for_plotter(sensor.points)
    hidden_ids = set(np.flatnonzero(np.all(np.isnan(displayed), axis=1)).tolist())

    assert hidden_ids == {35, 36, 42, 43}
    assert hidden_ids.isdisjoint({40, 41, 47, 48})


class _ActorCapturePlotter:
    def __init__(self):
        self.mesh_calls = []

    def add_mesh(self, mesh, **kwargs):
        self.mesh_calls.append((mesh, kwargs))
        return object()


def test_original_position_actor_keeps_low_reference_opacity():
    sensor = MySensor.__new__(MySensor)
    sensor.plotter = _ActorCapturePlotter()
    sensor._2D_map = pv.PolyData(np.asarray([[0.0, 0.0, 0.0]]))
    sensor.colors = np.ones((1, 4), dtype=float)
    sensor.objActor = None
    sensor.actionMesh = None
    sensor.line_poly = None
    sensor.matrixLineActor = None
    sensor.sensorPointLabelActor = None
    sensor.show_sensor_point_labels = False

    sensor._ensure_main_sensor_visualization_actors()

    assert len(sensor.plotter.mesh_calls) == 1
    assert sensor.plotter.mesh_calls[0][1]["opacity"] == 0.12

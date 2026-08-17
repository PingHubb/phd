import numpy as np

import phd.dependence.func_sensor as func_sensor_module
from phd.dependence.func_sensor import MySensor
from phd.dependence.sensor_geometry import (
    compute_grid_normals,
    grid_selection_weights,
    heatmap_corner_points_from_centres,
    heatmap_surface_from_corner_lattice,
    interpolate_coarse_deformation,
    smooth_grid_points,
    structured_grid_edge_pairs,
)


def _grid_points(n_row, n_col, z_function=None):
    points = []
    for col in range(n_col):
        for row in range(n_row):
            z = 0.0 if z_function is None else float(z_function(col, row))
            points.append([float(col), float(row), z])
    return np.asarray(points, dtype=float)


def test_grid_selection_uses_a_physical_circular_neighborhood():
    indices, weights = grid_selection_weights(
        center_index=4,
        n_row=3,
        n_col=3,
        radius=1,
        soft=True,
    )

    assert set(indices.tolist()) == {1, 3, 4, 5, 7}
    assert weights[indices.tolist().index(4)] == 1.0
    assert np.all((weights > 0.0) & (weights <= 1.0))


def test_flat_and_tilted_grids_produce_unit_surface_normals():
    flat = _grid_points(3, 4)
    flat_normals = compute_grid_normals(flat, 3, 4)

    np.testing.assert_allclose(
        flat_normals,
        np.tile([0.0, 0.0, 1.0], (len(flat), 1)),
        atol=1e-9,
    )

    tilted = _grid_points(3, 4, lambda col, row: 0.2 * col - 0.1 * row)
    tilted_normals = compute_grid_normals(tilted, 3, 4)
    expected = np.asarray([-0.2, 0.1, 1.0], dtype=float)
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(
        tilted_normals,
        np.tile(expected, (len(tilted), 1)),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        compute_grid_normals(tilted, 3, 4, normal_flip=True),
        np.tile(-expected, (len(tilted), 1)),
        atol=1e-9,
    )


def test_coarse_deformation_is_blended_onto_dense_sensor_points():
    base = _grid_points(2, 2)
    deformed = np.array(base, copy=True)
    deformed[3, 2] = 1.0
    fine = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )

    result = interpolate_coarse_deformation(base, deformed, fine)

    np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(result[1], [1.0, 1.0, 1.0], atol=1e-9)
    assert 0.0 < result[2, 2] < 1.0


def test_smoothing_reduces_an_isolated_height_spike():
    points = _grid_points(3, 3)
    points[4, 2] = 1.0

    smoothed = smooth_grid_points(
        points,
        selected_indices=[4],
        n_row=3,
        n_col=3,
        strength=0.5,
    )

    assert smoothed[4, 2] == 0.5
    np.testing.assert_allclose(smoothed[np.arange(9) != 4], points[np.arange(9) != 4])


def test_custom_geometry_config_is_sanitized_for_persistence():
    sensor = MySensor.__new__(MySensor)
    points = _grid_points(2, 2)
    heatmap_points = np.array(points, copy=True)
    heatmap_points[3, 2] = 0.4
    heatmap_corners = _grid_points(3, 3)
    heatmap_corners[4, 2] = 0.3
    heatmap_edge_offsets = np.zeros((12, 3), dtype=float)
    heatmap_edge_offsets[2, 2] = 0.1

    config = sensor._normalize_sensor_geometry_config(
        {
            "use_selected_shape": True,
            "shape": "custom",
            "normal_flip": True,
            "rotation_deg": [10.0, 20.0, 30.0],
            "custom_points": points,
            "use_custom_heatmap_shape": True,
            "custom_heatmap_points": heatmap_points,
            "custom_heatmap_corners": heatmap_corners,
            "use_curved_heatmap_edges": True,
            "custom_heatmap_edge_offsets": heatmap_edge_offsets,
        }
    )

    assert config["shape"] == "custom"
    assert config["custom_points"] == points.tolist()
    assert config["normal_flip"] is True
    assert config["rotation_deg"] == [10.0, 20.0, 30.0]
    assert config["use_custom_heatmap_shape"] is True
    assert config["custom_heatmap_points"] == heatmap_points.tolist()
    assert config["custom_heatmap_corners"] == heatmap_corners.tolist()
    assert config["use_curved_heatmap_edges"] is True
    assert config["custom_heatmap_edge_offsets"] == heatmap_edge_offsets.tolist()

    invalid = sensor._normalize_sensor_geometry_config(
        {"shape": "custom", "custom_points": [[0.0, float("nan"), 0.0]]}
    )
    assert invalid["custom_points"] == []


def test_editor_data_uses_custom_points_and_recomputed_normals():
    sensor = MySensor.__new__(MySensor)
    sensor.current_model_name = "2d"
    sensor.current_sensor_geometry_config = {}
    sensor.n_row = 2
    sensor.n_col = 3
    sensor._sensor_geometry_base_points_origin = _grid_points(2, 3)
    sensor._sensor_geometry_base_normals = np.tile([0.0, 0.0, 1.0], (6, 1))
    sensor.edges = np.asarray(
        [[2, 0, 1], [2, 2, 3], [2, 4, 5], [2, 0, 2], [2, 2, 4],
         [2, 1, 3], [2, 3, 5]],
        dtype=int,
    )
    custom = _grid_points(2, 3, lambda col, row: 0.15 * col)
    heatmap_custom = _grid_points(2, 3, lambda col, row: 0.2 * row)
    heatmap_corners = _grid_points(3, 4, lambda col, row: 0.1 * row)
    heatmap_edge_offsets = np.zeros((17, 3), dtype=float)
    heatmap_edge_offsets[3, 2] = 0.2

    editor_data = sensor.get_sensor_geometry_editor_data(
        {
            "use_selected_shape": True,
            "shape": "custom",
            "custom_points": custom.tolist(),
            "use_custom_heatmap_shape": True,
            "custom_heatmap_points": heatmap_custom.tolist(),
            "custom_heatmap_corners": heatmap_corners.tolist(),
            "use_curved_heatmap_edges": True,
            "custom_heatmap_edge_offsets": heatmap_edge_offsets.tolist(),
        }
    )

    np.testing.assert_allclose(editor_data["points"], custom)
    np.testing.assert_allclose(
        np.linalg.norm(editor_data["normals"], axis=1),
        1.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        editor_data["flat_points"],
        sensor._sensor_geometry_base_points_origin,
    )
    np.testing.assert_allclose(editor_data["heatmap_points"], heatmap_custom)
    np.testing.assert_allclose(editor_data["heatmap_default_points"], custom)
    np.testing.assert_allclose(editor_data["heatmap_corners"], heatmap_corners)
    assert editor_data["heatmap_n_row"] == 3
    assert editor_data["heatmap_n_col"] == 4
    np.testing.assert_array_equal(
        editor_data["heatmap_edge_pairs"], structured_grid_edge_pairs(3, 4)
    )
    np.testing.assert_allclose(
        editor_data["heatmap_edge_offsets"], heatmap_edge_offsets
    )


def test_custom_geometry_round_trips_through_sensor_settings(tmp_path, monkeypatch):
    settings_file = tmp_path / "sensor_reorder_logic.json"
    monkeypatch.setattr(
        func_sensor_module,
        "SENSOR_REORDER_LOGIC_FILE",
        str(settings_file),
    )
    sensor = MySensor.__new__(MySensor)
    sensor.current_model_name = "2d"
    custom = _grid_points(2, 3, lambda col, row: 0.25 * col + 0.1 * row)
    geometry = {
        "use_selected_shape": True,
        "shape": "custom",
        "normal_flip": False,
        "rotation_deg": [5.0, -10.0, 15.0],
        "custom_points": custom.tolist(),
        "use_custom_heatmap_shape": True,
        "custom_heatmap_points": (custom + [0.0, 0.0, 0.1]).tolist(),
        "custom_heatmap_corners": _grid_points(3, 4).tolist(),
        "use_curved_heatmap_edges": True,
        "custom_heatmap_edge_offsets": np.zeros((17, 3)).tolist(),
    }

    assert sensor.set_saved_sensor_geometry_config(
        "2d",
        geometry,
        n_row=2,
        n_col=3,
    )
    loaded = sensor.get_saved_sensor_geometry_config(
        "2d",
        n_row=2,
        n_col=3,
    )

    assert loaded["shape"] == "custom"
    assert loaded["rotation_deg"] == [5.0, -10.0, 15.0]
    np.testing.assert_allclose(loaded["custom_points"], custom)
    assert loaded["use_custom_heatmap_shape"] is True
    np.testing.assert_allclose(
        loaded["custom_heatmap_points"], custom + [0.0, 0.0, 0.1]
    )
    np.testing.assert_allclose(
        loaded["custom_heatmap_corners"], _grid_points(3, 4)
    )
    assert loaded["use_curved_heatmap_edges"] is True
    np.testing.assert_allclose(
        loaded["custom_heatmap_edge_offsets"], np.zeros((17, 3))
    )


def test_heatmap_corner_lattice_has_one_shared_vertex_per_taxel_corner():
    centres = _grid_points(2, 3)
    normals = compute_grid_normals(centres, 2, 3)

    corners = heatmap_corner_points_from_centres(
        centres,
        normals,
        n_row=2,
        n_col=3,
    )

    assert corners.shape == (12, 3)
    corner_grid = corners.reshape(4, 3, 3)
    assert np.all(corner_grid[:, :, 2] > 0.0)
    assert np.ptp(corner_grid[:, :, 2]) < 1e-12


def test_curved_heatmap_edge_is_identical_for_both_neighboring_taxels():
    corners = _grid_points(2, 3)
    edge_pairs = structured_grid_edge_pairs(2, 3)
    offsets = np.zeros((len(edge_pairs), 3), dtype=float)
    shared_edge_index = np.flatnonzero(
        np.all(edge_pairs == np.asarray([2, 3]), axis=1)
    )[0]
    offsets[shared_edge_index, 2] = 0.5

    surface = heatmap_surface_from_corner_lattice(
        corners,
        corner_n_row=2,
        corner_n_col=3,
        edge_offsets=offsets,
        curve_resolution=4,
    )

    resolution = 4
    left = surface["points"][surface["tile_vertices"][0, 0]]
    right = surface["points"][surface["tile_vertices"][0, 1]]
    left_shared = left[np.arange(resolution + 1) * (resolution + 1) + resolution]
    right_shared = right[np.arange(resolution + 1) * (resolution + 1)]
    np.testing.assert_allclose(left_shared, right_shared)
    assert left_shared[resolution // 2, 2] == 0.25

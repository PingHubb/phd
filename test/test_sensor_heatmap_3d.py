from types import SimpleNamespace

import numpy as np
import pyvista as pv

from phd.dependence.func_sensor import MySensor
from phd.dependence.sensor_geometry import (
    heatmap_corner_points_from_centres,
    structured_grid_edge_pairs,
)
from phd.dependence.sensor_heatmap import (
    DEFAULT_HEATMAP_3D_COLOR_GAIN,
    DEFAULT_HEATMAP_3D_PALETTE,
    DEFAULT_PROXIMITY_KNEE,
    DEFAULT_PROXIMITY_NOISE_FLOOR,
    DEFAULT_PROXIMITY_SATURATION,
    HEATMAP_RESPONSE_LINEAR_RELATIVE,
    HEATMAP_RESPONSE_PROXIMITY_ENHANCED,
    HEATMAP_3D_PALETTE_LIGHT_DEEP_BLUE,
    HEATMAP_3D_PALETTE_WHITE_BLUE_RED,
    apply_3d_heatmap_color_gain,
    heatmap_3d_rgb,
    heatmap_rgb,
)
from phd.dependence.sensor_signal_window import SensorSignalWindow


class _Actor:
    def __init__(self):
        self.visible = True

    def SetVisibility(self, visible):
        self.visible = bool(visible)


class _Plotter:
    def __init__(self):
        self.meshes = []

    def add_mesh(self, mesh, **kwargs):
        self.meshes.append((mesh, kwargs))
        return _Actor()

    def remove_actor(self, *_args, **_kwargs):
        return None

    def render(self):
        return None


def _heatmap_sensor(rows=2, cols=3):
    sensor = MySensor.__new__(MySensor)
    sensor.n_row = rows
    sensor.n_col = cols
    sensor.points_origin = np.asarray(
        [[float(col), float(row), 0.0] for col in range(cols) for row in range(rows)],
        dtype=float,
    )
    sensor.normals = np.tile([0.0, 0.0, 1.0], (rows * cols, 1))
    sensor.cell_zero_mask = np.zeros((rows, cols), dtype=bool)
    sensor.sensor_visual_offset_scale = 0.0
    sensor.plotter = _Plotter()
    sensor.heatmapActor = None
    sensor.heatmapPoly = None
    sensor.heatmapColors = None
    sensor._heatmap_tile_vertices = None
    return sensor


def test_shared_heatmap_color_scale_matches_viewer_thresholds():
    colors = heatmap_rgb(np.asarray([0.0, 0.5, 2.75, 5.0, 10.0]))

    np.testing.assert_array_equal(colors[0], [255, 255, 255])
    np.testing.assert_array_equal(colors[1], [255, 255, 255])
    np.testing.assert_array_equal(colors[2], [255, 128, 128])
    np.testing.assert_array_equal(colors[3], [255, 0, 0])
    np.testing.assert_array_equal(colors[4], [255, 0, 0])


def test_proximity_enhanced_scale_expands_the_20_to_100_range():
    values = np.asarray([0.0, 20.0, 30.0, 60.0, 100.0, 300.0, 1000.0])
    colors = heatmap_rgb(
        values,
        response_mode=HEATMAP_RESPONSE_PROXIMITY_ENHANCED,
        proximity_noise_floor=20.0,
        proximity_knee=100.0,
        proximity_saturation=1000.0,
    )

    np.testing.assert_array_equal(colors[1], [255, 255, 255])
    assert colors[2, 1] < 240
    assert np.all(np.diff(colors[1:, 1].astype(int)) <= 0)
    assert 90 < colors[4, 1] < 140
    np.testing.assert_array_equal(colors[-1], [255, 0, 0])


def test_3d_color_gain_strengthens_only_intermediate_heatmap_colors():
    colors = np.asarray(
        [[255, 255, 255], [255, 205, 205], [255, 115, 115], [255, 0, 0]],
        dtype=np.uint8,
    )

    boosted = apply_3d_heatmap_color_gain(colors, gain=1.5)

    np.testing.assert_array_equal(boosted[0], colors[0])
    np.testing.assert_array_equal(boosted[-1], colors[-1])
    assert boosted[1, 1] < colors[1, 1]
    assert boosted[2, 1] < colors[2, 1]


def test_original_3d_palette_keeps_existing_white_to_red_colors():
    values = np.asarray([0.0, 1.0, 2.75, 5.0])
    expected = apply_3d_heatmap_color_gain(
        heatmap_rgb(values), gain=DEFAULT_HEATMAP_3D_COLOR_GAIN
    )

    np.testing.assert_array_equal(heatmap_3d_rgb(values), expected)


def test_white_blue_red_3d_palette_avoids_cyan_yellow_and_orange():
    values = np.linspace(0.5, 5.0, 101)
    colors = heatmap_3d_rgb(
        values,
        palette=HEATMAP_3D_PALETTE_WHITE_BLUE_RED,
        color_gain=1.0,
    )

    np.testing.assert_array_equal(colors[0], [255, 255, 255])
    assert colors[25, 2] > colors[25, 0]
    np.testing.assert_array_equal(colors[-1], [130, 0, 25])
    assert np.all(colors[:, 1] <= np.maximum(colors[:, 0], colors[:, 2]))
    assert np.max(np.abs(np.diff(colors.astype(int), axis=0))) < 20


def test_light_deep_blue_3d_palette_darkens_smoothly_with_signal():
    values = np.linspace(0.5, 5.0, 101)
    colors = heatmap_3d_rgb(
        values,
        palette=HEATMAP_3D_PALETTE_LIGHT_DEEP_BLUE,
        color_gain=1.0,
    )

    np.testing.assert_array_equal(colors[0], [245, 250, 255])
    np.testing.assert_array_equal(colors[-1], [5, 25, 80])
    assert np.all(colors[:, 2] >= colors[:, 1])
    assert np.all(colors[:, 1] >= colors[:, 0])
    assert np.all(np.diff(colors.astype(int), axis=0) <= 0)
    assert np.max(np.abs(np.diff(colors.astype(int), axis=0))) < 10


def test_signal_viewer_and_3d_heatmap_return_identical_colors():
    viewer = SensorSignalWindow.__new__(SensorSignalWindow)
    viewer._heatmap_response_mode = HEATMAP_RESPONSE_LINEAR_RELATIVE
    viewer._heatmap_saturation_pct = 12.0
    viewer._heatmap_noise_floor_pct = 0.1
    viewer._proximity_noise_floor = DEFAULT_PROXIMITY_NOISE_FLOOR
    viewer._proximity_knee = DEFAULT_PROXIMITY_KNEE
    viewer._proximity_saturation = DEFAULT_PROXIMITY_SATURATION

    for value in (0.0, 0.1, 0.11, 1.0, 6.0, 12.0, 20.0):
        expected = heatmap_rgb(
            value,
            saturation_pct=12.0,
            noise_floor_pct=0.1,
        )
        actual = viewer._color_for_cell(value)
        assert (actual.red(), actual.green(), actual.blue()) == tuple(expected)


def test_signal_viewer_publishes_its_settings_to_3d_heatmap():
    received = []
    sensor_functions = SimpleNamespace(
        set_heatmap_settings=lambda settings: received.append(dict(settings)) or True
    )
    viewer = SensorSignalWindow.__new__(SensorSignalWindow)
    viewer._resolve_sensor_functions = lambda: sensor_functions
    viewer._heatmap_response_mode = HEATMAP_RESPONSE_PROXIMITY_ENHANCED
    viewer._heatmap_saturation_pct = 5.0
    viewer._heatmap_noise_floor_pct = 0.5
    viewer._proximity_noise_floor = 20.0
    viewer._proximity_knee = 100.0
    viewer._proximity_saturation = 1000.0
    viewer._heatmap_3d_color_gain = DEFAULT_HEATMAP_3D_COLOR_GAIN
    viewer._heatmap_3d_palette = HEATMAP_3D_PALETTE_WHITE_BLUE_RED

    assert viewer._publish_heatmap_settings_to_sensor()
    assert received == [{
        "palette_3d": HEATMAP_3D_PALETTE_WHITE_BLUE_RED,
        "response_mode": HEATMAP_RESPONSE_PROXIMITY_ENHANCED,
        "saturation_pct": 5.0,
        "noise_floor_pct": 0.5,
        "proximity_noise_floor": 20.0,
        "proximity_knee": 100.0,
        "proximity_saturation": 1000.0,
        "color_gain_3d": DEFAULT_HEATMAP_3D_COLOR_GAIN,
    }]


def test_heatmap_builds_one_independent_quad_per_taxel():
    sensor = _heatmap_sensor(rows=2, cols=3)

    poly, colors, tile_vertices = sensor._build_heatmap_polydata()

    assert poly.n_cells == 6
    assert poly.n_points == 24
    assert colors.shape == (24, 4)
    assert tile_vertices.shape == (2, 3, 4)
    assert len(np.unique(tile_vertices)) == 24


def test_live_heatmap_actor_uses_a_separate_solid_grid_overlay():
    sensor = _heatmap_sensor(rows=2, cols=3)

    sensor._rebuild_heatmap_visualization_actor(render=False)

    _surface, surface_options = sensor.plotter.meshes[-2]
    grid, grid_options = sensor.plotter.meshes[-1]
    assert surface_options["show_edges"] is False
    assert grid_options["name"] == "sensor_heatmap_grid"
    assert grid_options["color"] == "#495057"
    assert grid_options["show_edges"] is False
    assert grid_options["smooth_shading"] is False
    assert grid_options["lighting"] is False
    assert grid is sensor.heatmapGridDisplayPoly
    assert sensor.heatmapGridPoly.n_lines > 0
    assert grid.n_cells > 0


def test_heatmap_grid_omits_edges_that_touch_only_zero_masked_taxels():
    sensor = _heatmap_sensor(rows=2, cols=3)
    sensor.current_model_name = "2d"
    sensor._sensor_geometry_base_points_origin = np.array(
        sensor.points_origin, copy=True
    )
    sensor._sensor_geometry_base_normals = np.array(sensor.normals, copy=True)
    corners = heatmap_corner_points_from_centres(
        sensor.points_origin,
        sensor.normals,
        sensor.n_row,
        sensor.n_col,
    )
    sensor.current_sensor_geometry_config = {
        "use_custom_heatmap_shape": True,
        "custom_heatmap_corners": corners.tolist(),
    }
    sensor._rebuild_heatmap_visualization_actor(render=False)

    assert sensor.heatmapGridPoly.n_lines == 17
    sensor.cell_zero_mask[1, 0] = True
    masked_grid = sensor._build_visible_heatmap_grid_polydata()
    assert masked_grid.n_lines == 15

    sensor.cell_zero_mask[:] = True
    assert sensor._build_visible_heatmap_grid_polydata().n_lines == 0


def test_shared_corner_heatmap_moves_without_moving_sensor_points_or_opening_edges():
    sensor = _heatmap_sensor(rows=2, cols=3)
    sensor.current_model_name = "2d"
    sensor._sensor_geometry_base_points_origin = np.array(
        sensor.points_origin, copy=True
    )
    sensor._sensor_geometry_base_normals = np.array(sensor.normals, copy=True)
    custom_corners = heatmap_corner_points_from_centres(
        sensor.points_origin,
        sensor.normals,
        sensor.n_row,
        sensor.n_col,
    )
    custom_corners[4, 2] = 0.6
    sensor.current_sensor_geometry_config = {
        "use_selected_shape": False,
        "use_custom_heatmap_shape": True,
        "custom_heatmap_corners": custom_corners.tolist(),
    }
    original_sensor_points = np.array(sensor.points_origin, copy=True)

    poly, _colors, tile_vertices = sensor._build_heatmap_polydata()

    np.testing.assert_allclose(sensor.points_origin, original_sensor_points)
    left_tile = poly.points[tile_vertices[0, 0]]
    right_tile = poly.points[tile_vertices[0, 1]]
    np.testing.assert_allclose(left_tile[1], right_tile[0])
    np.testing.assert_allclose(left_tile[3], right_tile[2])
    assert left_tile[3, 2] == 0.6


def test_curved_heatmap_renderer_tessellates_shared_bezier_edges():
    sensor = _heatmap_sensor(rows=2, cols=3)
    sensor.current_model_name = "2d"
    sensor._sensor_geometry_base_points_origin = np.array(
        sensor.points_origin, copy=True
    )
    sensor._sensor_geometry_base_normals = np.array(sensor.normals, copy=True)
    corners = heatmap_corner_points_from_centres(
        sensor.points_origin,
        sensor.normals,
        sensor.n_row,
        sensor.n_col,
    )
    edge_pairs = structured_grid_edge_pairs(3, 4)
    offsets = np.zeros((len(edge_pairs), 3), dtype=float)
    offsets[3, 2] = 0.25
    sensor.current_sensor_geometry_config = {
        "use_custom_heatmap_shape": True,
        "custom_heatmap_corners": corners.tolist(),
        "use_curved_heatmap_edges": True,
        "custom_heatmap_edge_offsets": offsets.tolist(),
    }

    poly, colors, tile_vertices = sensor._build_heatmap_polydata()

    assert poly.n_cells == sensor.n_row * sensor.n_col * 36
    assert tile_vertices.shape == (2, 3, 49)
    assert colors.shape == (2 * 3 * 49, 4)
    assert max(
        np.max(edge["points"][:, 2])
        for edge in sensor._heatmap_logical_edges
    ) > 0.1


def test_heatmap_uses_same_top_down_row_mapping_as_signal_viewer():
    sensor = _heatmap_sensor(rows=2, cols=3)
    values = np.zeros((2, 3), dtype=float)
    values[0, 1] = 10.0

    sensor._update_heatmap_visualization(values)

    top_viewer_tile = sensor._heatmap_tile_vertices[1, 1]
    opposite_tile = sensor._heatmap_tile_vertices[0, 1]
    np.testing.assert_array_equal(
        sensor.heatmapColors[top_viewer_tile, :3],
        np.tile([255, 0, 0], (4, 1)),
    )
    np.testing.assert_array_equal(
        sensor.heatmapColors[opposite_tile, :3],
        np.tile([255, 255, 255], (4, 1)),
    )


def test_3d_heatmap_uses_latest_frame_instead_of_averaged_frame():
    sensor = _heatmap_sensor(rows=1, cols=1)
    sensor._data = SimpleNamespace(diffPerData=np.asarray([[4.0]]))
    sensor.sensor_visualization_mode = "heatmap_3d"
    sensor.touch_sensitivity_scale = 0.05
    sensor.points = np.array(sensor.points_origin, copy=True)
    sensor.colors_3d = np.ones((1, 4), dtype=float)
    sensor.colors = np.ones((1, 4), dtype=float)
    sensor.array_positions = [[0]]
    sensor.line_poly = pv.PolyData(sensor.points)
    sensor._2D_map = pv.PolyData(sensor.points)
    sensor._record_visualization_tick = lambda: None
    sensor._update_contact_force_status = lambda _matrix: None
    sensor._update_contact_normal_visualization = lambda _matrix: None
    received = []
    sensor._update_heatmap_visualization = (
        lambda matrix: received.append(np.array(matrix, copy=True))
    )

    sensor.update_visualization(np.asarray([[1.0]]))

    np.testing.assert_array_equal(received[0], [[4.0]])
    np.testing.assert_array_equal(sensor._last_sensor_visualization_matrix, [[4.0]])


def test_heatmap_recording_snapshot_preserves_display_raw_and_calibration_values():
    sensor = _heatmap_sensor(rows=2, cols=2)
    sensor._data = SimpleNamespace(
        rawData=np.asarray([[101.0, 102.0], [103.0, 104.0]]),
        calData=np.full((2, 2), 100.0),
        diffPerData=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
    )

    snapshot = sensor.get_heatmap_recording_snapshot()

    np.testing.assert_array_equal(
        snapshot["heatmap_values"], [[1.0, 2.0], [3.0, 4.0]]
    )
    np.testing.assert_array_equal(
        snapshot["raw_values"], [[101.0, 102.0], [103.0, 104.0]]
    )
    np.testing.assert_array_equal(snapshot["calibration_values"], 100.0)


def test_recorded_heatmap_frame_uses_existing_live_heatmap_actor():
    sensor = _heatmap_sensor(rows=2, cols=2)
    sensor.sensor_visualization_mode = "heatmap_3d"
    sensor._rebuild_heatmap_visualization_actor(render=False)
    actor = sensor.heatmapActor

    assert sensor.render_recorded_heatmap_frame(
        np.asarray([[0.0, 1.0], [2.0, 4.0]])
    )

    assert sensor.heatmapActor is actor
    np.testing.assert_array_equal(
        sensor._last_sensor_visualization_matrix,
        [[0.0, 1.0], [2.0, 4.0]],
    )
    assert not sensor.render_recorded_heatmap_frame(np.zeros((3, 3)))


def test_proximity_heatmap_uses_absolute_difference_frame():
    sensor = _heatmap_sensor(rows=1, cols=1)
    sensor.heatmap_response_mode = HEATMAP_RESPONSE_PROXIMITY_ENHANCED
    sensor._data = SimpleNamespace(
        diffPerData=np.asarray([[4.0]]),
        diffData=np.asarray([[60.0]]),
    )

    np.testing.assert_array_equal(sensor._current_heatmap_sensor_matrix(), [[60.0]])


def test_signal_viewer_baseline_is_used_by_3d_proximity_heatmap():
    sensor = _heatmap_sensor(rows=2, cols=2)
    sensor.heatmap_response_mode = HEATMAP_RESPONSE_PROXIMITY_ENHANCED
    sensor._data = SimpleNamespace(
        rawData=np.asarray([[130.0, 340.0], [220.0, 500.0]]),
        diffData=np.zeros((2, 2)),
        diffPerData=np.zeros((2, 2)),
    )

    assert sensor.set_heatmap_calibration_baseline(
        [100.0, 200.0, 300.0, 400.0], n_row=2, n_col=2
    )

    np.testing.assert_array_equal(
        sensor.get_heatmap_sensor_matrix(),
        [[30.0, 40.0], [20.0, 100.0]],
    )


def test_signal_viewer_reads_exact_3d_heatmap_values_in_column_major_order():
    sensor_functions = SimpleNamespace(
        _data=SimpleNamespace(),
        get_heatmap_sensor_matrix=lambda: np.asarray(
            [[30.0, 40.0], [20.0, 100.0]]
        ),
    )
    viewer = SensorSignalWindow.__new__(SensorSignalWindow)
    viewer.table_rows = 2
    viewer.table_columns = 2
    viewer._shared_sensor_functions_ref = sensor_functions
    viewer._resolve_sensor_functions = lambda: sensor_functions

    assert viewer._read_shared_heatmap_list() == [30.0, 20.0, 40.0, 100.0]


def test_heatmap_zero_mask_hides_the_matching_display_tile():
    sensor = _heatmap_sensor(rows=2, cols=3)
    sensor.cell_zero_mask[0, 2] = True

    sensor._update_heatmap_visualization(np.full((2, 3), 5.0))

    hidden_tile = sensor._heatmap_tile_vertices[1, 2]
    assert np.all(sensor.heatmapColors[hidden_tile, 3] == 0)
    assert np.count_nonzero(sensor.heatmapColors[:, 3] == 0) == 4


def test_heatmap_sensitivity_setting_recolors_current_frame_immediately():
    sensor = _heatmap_sensor(rows=2, cols=2)
    sensor.sensor_visualization_mode = "heatmap_3d"
    values = np.full((2, 2), 1.0, dtype=float)
    sensor._last_sensor_visualization_matrix = values.copy()
    sensor._update_heatmap_visualization(values)
    original_green = int(sensor.heatmapColors[0, 1])

    assert sensor.set_heatmap_settings({
        "saturation_pct": 1.0,
        "noise_floor_pct": 0.0,
    })

    assert original_green > 0
    assert sensor.heatmapColors[0, 1] == 0
    assert sensor.get_heatmap_settings() == {
        "palette_3d": DEFAULT_HEATMAP_3D_PALETTE,
        "response_mode": HEATMAP_RESPONSE_LINEAR_RELATIVE,
        "saturation_pct": 1.0,
        "noise_floor_pct": 0.0,
        "proximity_noise_floor": DEFAULT_PROXIMITY_NOISE_FLOOR,
        "proximity_knee": DEFAULT_PROXIMITY_KNEE,
        "proximity_saturation": DEFAULT_PROXIMITY_SATURATION,
        "color_gain_3d": DEFAULT_HEATMAP_3D_COLOR_GAIN,
    }


def test_heatmap_sensitivity_is_saved_per_sensor_layout():
    sensor = MySensor.__new__(MySensor)
    sensor.current_model_name = "2d"
    sensor.n_row = 7
    sensor.n_col = 7
    payload = {"version": 1, "settings": {}}
    sensor.get_sensor_reorder_key = lambda *_args, **_kwargs: "2d_7x7"
    sensor._read_reorder_logic_file = lambda: payload
    sensor._write_reorder_logic_file = lambda _payload: True

    assert sensor.set_saved_sensor_heatmap_config(
        "2d",
        {
            "saturation_pct": 12.0,
            "noise_floor_pct": 0.1,
        },
        n_row=7,
        n_col=7,
    )

    assert payload["settings"]["2d_7x7"]["heatmap"] == {
        "palette_3d": DEFAULT_HEATMAP_3D_PALETTE,
        "response_mode": HEATMAP_RESPONSE_LINEAR_RELATIVE,
        "saturation_pct": 12.0,
        "noise_floor_pct": 0.1,
        "proximity_noise_floor": DEFAULT_PROXIMITY_NOISE_FLOOR,
        "proximity_knee": DEFAULT_PROXIMITY_KNEE,
        "proximity_saturation": DEFAULT_PROXIMITY_SATURATION,
        "color_gain_3d": DEFAULT_HEATMAP_3D_COLOR_GAIN,
    }
    assert sensor.get_saved_sensor_heatmap_config("2d", n_row=7, n_col=7) == {
        "palette_3d": DEFAULT_HEATMAP_3D_PALETTE,
        "response_mode": HEATMAP_RESPONSE_LINEAR_RELATIVE,
        "saturation_pct": 12.0,
        "noise_floor_pct": 0.1,
        "proximity_noise_floor": DEFAULT_PROXIMITY_NOISE_FLOOR,
        "proximity_knee": DEFAULT_PROXIMITY_KNEE,
        "proximity_saturation": DEFAULT_PROXIMITY_SATURATION,
        "color_gain_3d": DEFAULT_HEATMAP_3D_COLOR_GAIN,
    }


def test_heatmap_tiles_follow_curved_sensor_geometry():
    sensor = _heatmap_sensor(rows=3, cols=5)
    radius = 2.0
    points = []
    normals = []
    for col, theta in enumerate(np.linspace(-0.7, 0.7, sensor.n_col)):
        for row in range(sensor.n_row):
            points.append([radius * np.sin(theta), float(row), radius * np.cos(theta)])
            normals.append([np.sin(theta), 0.0, np.cos(theta)])
    sensor.points_origin = np.asarray(points, dtype=float)
    sensor.normals = np.asarray(normals, dtype=float)

    poly, _colors, _tile_vertices = sensor._build_heatmap_polydata()

    assert poly.n_cells == sensor.n_row * sensor.n_col
    assert np.ptp(poly.points[:, 2]) > 0.25
    assert np.all(np.isfinite(poly.points))

"""Pure planning and simulated routing: no live robot/sensor access."""
import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest

from phd.ui.experiment_tasks import TaxelPositionMappingTask


def make_grid(rows=20, columns=20, skip=True, warped=False):
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task.sensor_rows, task.sensor_columns = rows, columns
    anchor_row, anchor_column = (rows-1)//2, (columns-1)//2
    task.anchor_taxel = anchor_row+anchor_column*rows
    task.skip_outer_border = skip
    theta = np.deg2rad(27.)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    origin = np.array([.1, .2, .29])
    points = {}
    for index in range(rows*columns):
        dr, dc = index % rows-anchor_row, index//rows-anchor_column
        curvature = .00008 if warped else 0.
        offset = np.array([.011*dr+curvature*dr**2, .011*dc+curvature*dc**2])
        point = origin.copy()
        point[:2] += rotation@offset
        points[index] = point
    for index in (task.anchor_taxel, *task.anchor_neighbour_indices()):
        save_measured(task, index, points[index])
    fitted = task.fit_lattice_from_positions(task._positions)
    task._row_vector_xy, task._column_vector_xy = fitted[:2]
    task._initial_row_vector_xy, task._initial_column_vector_xy = fitted[:2]
    task.anchor_pose = {"taxel_index": task.anchor_taxel,
                        "tool_position_m": [.1, .2, .32],
                        "tool_quaternion_wxyz": [1., 0., 0., 0.]}
    task._reference_onset_z, task._probe_start_z = .30, .302
    task.max_xy_offset_m = .2
    return task, points


def save_measured(task, index, point):
    task._positions[index] = {"taxel_index": index, "status": "measured",
                              "tool_position_m": np.asarray(point).tolist()}


def test_five_saved_centres_grow_locally_instead_of_jumping_to_361():
    task, _points = make_grid()
    known = set(task._positions)
    plan = task.plan_frontier_target([.1, .2])
    assert plan["taxel_index"] != 361
    row, column = task._taxel_coords(plan["taxel_index"])
    assert abs(row-9)+abs(column-9) == 2
    assert len(plan["verified_neighbours"]) == 2  # Prefer a supported corner of this ring.
    assert set(plan["verified_neighbours"]) <= known
    for index in plan["verified_neighbours"]:
        known_row, known_column = task._taxel_coords(index)
        assert abs(row-known_row)+abs(column-known_column) == 1
        assert np.linalg.norm(np.array(plan["position_xy_m"])-np.array(task._positions[index]["tool_position_m"][:2])) < .02
    assert not any(key in task._positions for key in [plan["taxel_index"]])


@pytest.mark.parametrize("rows,columns,skip", [(5, 5, False), (8, 10, False), (8, 10, True), (20, 20, True)])
def test_frontier_visits_every_allowed_cell_exactly_once(rows, columns, skip):
    task, points = make_grid(rows, columns, skip)
    current = points[task.anchor_taxel][:2]
    visited = set(task._positions)
    rings = []
    for _ in range(rows*columns):
        plan = task.plan_frontier_target(current)
        if plan is None:
            break
        index = plan["taxel_index"]
        assert index not in visited
        assert index in task.mapping_target_indices()
        assert all(neighbour in visited for neighbour in plan["verified_neighbours"])
        np.testing.assert_allclose(plan["position_xy_m"], points[index][:2], atol=1e-10)
        row, column = task._taxel_coords(index)
        anchor_row, anchor_column = task._taxel_coords(task.anchor_taxel)
        rings.append(abs(row-anchor_row)+abs(column-anchor_column))
        save_measured(task, index, points[index])
        visited.add(index)
        current = points[index][:2]
    assert set(task.mapping_target_indices()) <= visited
    assert rings == sorted(rings)
    assert task.plan_frontier_target(current) is None
    if skip:
        assert not (visited & set(task.skipped_taxel_indices()))


def test_locally_updated_edges_follow_warp_instead_of_fixed_grid_extrapolation():
    task, points = make_grid(warped=True)
    old_positions, _row, _column, _score = task.predict_lattice_positions()
    old_error = np.linalg.norm(old_positions[361][:2]-points[361][:2])*1000
    assert old_error > 6
    errors = []
    while True:
        plan = task.plan_frontier_target()
        if plan is None:
            break
        index = plan["taxel_index"]
        errors.append(np.linalg.norm(np.array(plan["position_xy_m"])-points[index][:2])*1000)
        save_measured(task, index, points[index])
    assert len(task._positions) == 324
    assert max(errors) < 2.5  # Planning-only synthetic fixture, not a hardware accuracy claim.
    assert max(errors) < old_error/2


def test_new_verified_measurement_changes_the_next_prediction_and_references():
    task, points = make_grid()
    first = task.plan_frontier_target()
    index = first["taxel_index"]
    before = copy.deepcopy(task._positions)
    task._positions[index] = {"status": "predicted", "tool_position_m": points[index].tolist()}
    plan_unverified = task.plan_frontier_target()
    assert index not in plan_unverified["verified_neighbours"]
    task._positions = before
    save_measured(task, index, points[index]+[.0003, -.0002, 0.])
    assert task.plan_frontier_target()["taxel_index"] != index
    # A recorded failure alone is never a seed.
    task._failures[361] = "verification failed"
    assert 361 not in task.plan_frontier_target()["verified_neighbours"]


def test_transformed_unverified_or_nonfinite_positions_are_not_motion_seeds():
    task, points = make_grid()
    save_measured(task, 361, points[361])
    task._positions[361]["relocation_verified"] = False
    task._positions[362] = {"status": "measured", "tool_position_m": [float("nan"), 0, 0]}
    verified = task._verified_mapping_positions()
    assert 361 not in verified and 362 not in verified


def test_missing_frontier_refuses_unknown_gap_instead_of_finishing():
    task, _points = make_grid()
    task._positions.clear()
    with pytest.raises(ValueError, match="refusing a jump"):
        task.plan_frontier_target()


def test_resume_keeps_saved_points_clears_failed_attempt_and_uses_new_policy():
    task, _points = make_grid()
    old_positions = copy.deepcopy(task._positions)
    result = {"positions": list(old_positions.values()), "stop_reason": "mapping_target_failed",
              "failures": {"361": "stronger neighbour"}, "attempt_counts": {"361": 2},
              "reference_onset_z_m": .30, "probe_start_z_m": .302,
              "initial_row_vector_xy_m": task._initial_row_vector_xy.tolist(),
              "initial_column_vector_xy_m": task._initial_column_vector_xy.tolist(),
              "row_vector_xy_m": task._row_vector_xy.tolist(),
              "column_vector_xy_m": task._column_vector_xy.tolist()}
    task._restore_resume_state(result)
    assert task._positions == old_positions
    assert 361 not in task._attempt_counts
    assert not task._failures
    assert task._resume_history[-1]["previous_mapping_order"] == "fixed_serpentine"
    assert task._settings()["mapping_order"] == task.MAPPING_ORDER
    assert task.plan_frontier_target()["taxel_index"] != 361


def test_old_queue_is_replanned_and_a_failed_target_does_not_start_distant_motion(monkeypatch):
    task, _points = make_grid()
    task._mapping_queue = [361, 341, 321]
    started, messages = [], []
    host = SimpleNamespace(ui_ros=None, _append_sidebar_control_message=messages.append)
    monkeypatch.setattr(task, "_start_target", lambda _host, index, xy, context: started.append((index, list(xy), context)))
    task._start_next_regular_taxel(host)
    assert len(started) == 1
    assert started[0][0] != 361
    assert started[0][2] == "regular"
    assert task._current_prediction["verified_neighbours"]
    assert "one-step estimate" in messages[-1]


def test_full_twenty_by_twenty_state_machine_verifies_all_324_interior_taxels():
    from test_experiment_tasks import _SimRobotApi, _Host, _tick, _SIM_SURFACE_Z
    _fixture_task, centres_by_index = make_grid()
    centres = np.asarray([centres_by_index[index][:2] for index in range(400)])
    api = _SimRobotApi()
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task.move_settle_sec = 0.
    # This deterministic geometry/state-machine test needs no noise averaging;
    # production 100/20-frame defaults are asserted in separate tests.
    task.baseline_frame_count = 5
    task.probe_sample_frames = 5
    task.scan_frame_count = 3
    task.verify_frame_count = 5
    host = _Host(api, task)
    sensor = host.ui_ros.sensor_functions
    sensor.n_row = sensor.n_col = 20

    def frame_at_pose():
        position = np.asarray(api.tool_position)
        height = max(float(position[2])-_SIM_SURFACE_Z, 0.)
        # Synthetic steep proximity field: the target is reachable before the configured
        # limit. A shallow field must fail instead (separate safety tests).
        level = 400.*np.exp(-height/.0025)
        distance_sq = np.sum((centres-position[:2])**2, axis=1)
        return (-level*np.exp(-distance_sq/(2.*.006**2))).reshape(20, 20).T.copy()

    sensor._data.diffDataAve = frame_at_pose()
    assert task.capture_anchor_pose(host)
    assert task.on_start(host)
    lowest_z = api.tool_position[2]
    for _ in range(700000):
        if not task._active:
            break
        api.step(.05)
        lowest_z = min(lowest_z, api.tool_position[2])
        sensor._data.diffDataAve = frame_at_pose()
        host.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        _tick(task, host)
    result = task.last_result
    assert result is not None, host.messages[-5:]
    assert result["completed"], (result["stop_reason"], host.messages[-5:])
    assert result["mapped_count"] == 324
    assert result["failed_count"] == 0
    assert set(task._positions) == set(task.mapping_target_indices())
    assert lowest_z > _SIM_SURFACE_Z
    assert api.tool_position == pytest.approx([.1, .2, .3])
    errors = [1000*np.linalg.norm(np.array(record["tool_position_m"][:2])-centres[index])
              for index, record in task._positions.items()]
    assert max(errors) < .5  # Synthetic Gaussian field, not physical measurement accuracy.
    regular = [record for record in result["positions"] if record.get("position_prediction")]
    assert len(regular) == 319
    assert all(record["position_prediction"]["verified_neighbours"] for record in regular)

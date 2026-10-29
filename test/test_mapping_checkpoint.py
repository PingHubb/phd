"""Relocation saves/resumes use fake hardware only; never open a device."""

import copy
import json

import numpy as np
import pytest

from phd.ui.experiment_tasks import TaxelPositionMappingTask
from test_experiment_tasks import _Host, _RobotApi


def relocating_task(tmp_path):
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task.sensor_rows = task.sensor_columns = 5
    task.anchor_taxel = 12
    task.skip_outer_border = False
    task.anchor_pose = dict(taxel_index=12, tool_position_m=[.1, .2, .3],
                            tool_quaternion_wxyz=[1., 0., 0., 0.])
    task._reference_onset_z = .29
    task._probe_start_z = .295
    task._initial_row_vector_xy = np.array([0., .004])
    task._initial_column_vector_xy = np.array([.004, 0.])
    source = []
    for index in range(25):
        row, col = task.taxel_row_column(index, 5)
        source.append(dict(
            taxel_index=index, row=row, column=col, status="measured",
            tool_position_m=[.1 + (col-2)*.004, .2 + (row-2)*.004, .29],
            safe_approach_position_m=[.1 + (col-2)*.004, .2 + (row-2)*.004, .3],
            tool_quaternion_wxyz=[1., 0., 0., 0.],
            signal_counts=100., competing_signal_counts=10., dominance_ratio=10.,
            confidence=1., method="cross_scan_verified", attempt_count=1,
        ))
    task._relocation_active = True
    task._relocation_source_result = dict(
        task=task.id, completed=True, settings=task._settings(),
        sensor_identity=task.active_sensor_key_name, positions=source,
    )
    task._relocation_source_path = tmp_path / "old.json"
    task._relocation_source_positions = {r["taxel_index"]: r for r in source}
    task._relocation_reference_indices = [7, 11, 12, 13, 17]
    task._positions = {i: copy.deepcopy(source[i]) for i in task._relocation_reference_indices}
    task._positions[4] = copy.deepcopy(source[4])
    task._relocation_transform = task._fit_relocation_transform()
    task._output_folder = tmp_path
    return task


def test_numpy_transform_checkpoint_round_trip_and_no_in_memory_mutation(tmp_path):
    task = relocating_task(tmp_path)
    task._relocation_transform["nested"] = [np.float32(2.5), np.int64(6), np.bool_(True)]
    result = task._save_results(checkpoint=True)
    saved = json.loads((tmp_path / "taxel_position_map.json").read_text())
    assert saved == result == task.last_result
    assert isinstance(task._relocation_transform["rotation"], np.ndarray)
    np.testing.assert_allclose(saved["relocation"]["rotation"], np.eye(2), atol=1e-12)
    assert saved["relocation"]["nested"] == [2.5, 6, True]
    assert saved["measured_position_count"] == 6
    assert not (tmp_path / "taxel_position_map.json.tmp").exists()
    assert saved["relocation_checkpoint"]["source_map"]["completed"]
    assert task.validate_resume_result(saved)[0]


@pytest.mark.parametrize("bad", [np.array([np.nan, 0.]), np.array([np.inf, 0.]), object()])
def test_invalid_snapshot_preserves_all_previous_files(tmp_path, bad):
    task = relocating_task(tmp_path)
    task._save_results(checkpoint=True)
    previous = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    task._relocation_transform["translation"] = bad
    task._positions[4]["signal_counts"] = 1234.
    with pytest.raises((TypeError, ValueError)):
        task._save_results(checkpoint=True)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == previous


def test_in_memory_result_is_json_compatible_too(tmp_path):
    task = relocating_task(tmp_path)
    task._output_folder = None
    result = task._save_results()
    json.dumps(result, allow_nan=False)


def test_resume_restores_remaining_verifications_not_whole_grid(tmp_path, monkeypatch):
    original = relocating_task(tmp_path)
    result = original._save_results(checkpoint=True)
    remaining = [i for i in original._choose_relocation_verification_indices(3) if i != 4]
    assert len(remaining) == 2
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task._apply_resume_settings(result)
    task._restore_resume_state(result, tmp_path / "taxel_position_map.json")
    assert task._relocation_active and not task._relocation_finished
    assert not task._relocation_continues_mapping
    assert task._relocation_verification_queue == remaining
    assert set(task._positions) == {4, 7, 11, 12, 13, 17}
    assert len(task._relocation_source_positions) == 25
    calls = []
    monkeypatch.setattr(task, "_start_target", lambda host, index, xy, context: calls.append((index, context)))
    task._continue_after_resume_anchor_check(_Host(_RobotApi(), task))
    assert calls == [(remaining[0], "relocation_verify")]
    # A further interruption after popping the target must not skip it.
    second = task._save_results(checkpoint=True)
    task._restore_resume_state(second, tmp_path / "taxel_position_map.json")
    assert task._relocation_verification_queue == remaining


def test_resume_start_preserves_anchor_safety_check(tmp_path):
    original = relocating_task(tmp_path)
    result = original._save_results(checkpoint=True)
    result["stop_reason"] = "possible_contact"
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    api = _RobotApi()
    host = _Host(api, task)
    sensor = host.ui_ros.sensor_functions
    sensor.n_row = sensor.n_col = 5
    sensor._data.diffDataAve = np.zeros((5, 5))
    assert task.request_resume(result, tmp_path / "taxel_position_map.json")[0]
    api.tool_position[2] = .28
    assert not task.on_start(host)
    assert not api.sent and not api.velocity_commands
    api.tool_position[2] = .3
    assert task.on_start(host)
    assert task._move_action == "resume_anchor_arrived"
    assert task._relocation_active and len(task._positions) == 6


def test_resume_before_first_fit_keeps_relocation_intent(tmp_path):
    original = relocating_task(tmp_path)
    original._relocation_transform = None
    original._relocation_reference_indices = []
    original._positions = {12: original._positions[12]}
    result = original._save_results(checkpoint=True)
    assert original.validate_resume_result(result)[0]
    resumed = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    resumed._apply_resume_settings(result)
    resumed._restore_resume_state(result, tmp_path / "taxel_position_map.json")
    assert resumed._relocation_active
    assert not resumed._relocation_reference_indices
    assert not resumed._relocation_verification_queue


def test_regular_resume_clears_stale_relocation_state(tmp_path):
    task = relocating_task(tmp_path)
    result = task._save_results(checkpoint=True)
    result["relocation"] = result["relocation_checkpoint"] = None
    task._restore_resume_state(result, tmp_path / "taxel_position_map.json")
    assert not task._relocation_active
    assert not task._relocation_source_positions
    assert task._relocation_transform is None


@pytest.mark.parametrize("fault", ["legacy", "legacy_unfitted", "grid", "identity", "references", "missing_references", "completion"])
def test_invalid_relocation_resume_is_rejected(tmp_path, fault):
    task = relocating_task(tmp_path)
    result = task._save_results(checkpoint=True)
    state = result["relocation_checkpoint"]
    if fault == "legacy":
        del result["relocation_checkpoint"]
    elif fault == "legacy_unfitted":
        del result["relocation_checkpoint"]
        result["relocation"] = None
        result["output_folder"] = "/maps/taxel_position_map_relocated_20261026"
    elif fault == "grid":
        state["source_map"]["settings"]["sensor_rows"] = 6
    elif fault == "identity":
        state["source_map"]["sensor_identity"] = "another_sensor"
    elif fault == "references":
        state["reference_taxels"].append(4)
    elif fault == "missing_references":
        state["reference_taxels"] = []
    else:
        state["finished"] = "false"
    valid, message = task.validate_resume_result(result)
    assert not valid, message


def test_completed_relocation_partial_mapping_resumes_as_mapping(tmp_path):
    task = relocating_task(tmp_path)
    task._relocation_source_result["completed"] = False
    task._relocation_finished = True
    task._relocation_transform = {"rotation_matrix": np.eye(2)}
    result = task._save_results(checkpoint=True)
    assert task.validate_resume_result(result)[0]
    task._restore_resume_state(result, tmp_path / "taxel_position_map.json")
    assert task._relocation_active and task._relocation_finished
    assert task._relocation_continues_mapping
    assert not task._relocation_verification_queue


def test_resumed_verification_finishes_relocation_without_remapping_grid(tmp_path, monkeypatch):
    original = relocating_task(tmp_path)
    result = original._save_results(checkpoint=True)
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task._apply_resume_settings(result)
    task._restore_resume_state(result, tmp_path / "taxel_position_map.json")
    host = _Host(_RobotApi(), task)
    selected = []
    monkeypatch.setattr(task, "_start_target", lambda host, index, xy, context: selected.append(index))
    task._continue_after_resume_anchor_check(host)
    for _ in range(2):
        index = selected[-1]
        task._positions[index] = copy.deepcopy(task._relocation_source_positions[index])
        # At the same clearance checkpoint that failed in the real run.
        task._save_results(checkpoint=True)
        task._start_next_relocation_verification(host)
    assert len(selected) == 2 and 4 not in selected
    assert task._relocation_finished and len(task._positions) == 25
    assert task._move_action == "complete_anchor_arrived"
    final = task._save_results(checkpoint=True)
    assert len(final["relocation"]["verified_taxels"]) == 8
    assert sum(r["relocation_verified"] for r in final["positions"]) == 8
    assert len(final["positions"]) == 25

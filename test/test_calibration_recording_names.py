"""Recording identities, filename limits and recoverable legacy renames."""

import json
from pathlib import Path
import runpy

import pytest

from phd.dependence.calibration_recording_names import (
    calibration_folder_name,
    sensor_name_for_recording,
)
from phd.ui.experiment_tasks import CalibrationTask, FullTaxelCalibrationTask


MIGRATION = runpy.run_path(str(
    Path(__file__).resolve().parents[1]
    / "docs/maintenance/rename_calibration_recordings_2026_10_23.py"
))


def test_sensor_names_use_identity_without_guessing_from_shape():
    assert sensor_name_for_recording("", "sensor:testing_5x5_1", (5, 5)) == "Testing Sensor 1"
    assert sensor_name_for_recording("My sensor", "sensor:testing_5x5_1") == "My sensor"
    assert sensor_name_for_recording(shape=(8, 10)) == "Unknown 8x10 Sensor"


def test_long_unicode_sensor_names_are_bounded_and_distinct():
    kwargs = dict(kind="calibration_full_taxel", timestamp="20261023_160000_123456",
                  settings="taxels-324_signal-diff_ave_speed-0p1mmps_threshold-500g_travel-100mm_timeout-7200s_full-passes-5",
                  protocol="proximity_profile")
    name = calibration_folder_name(sensor_name="织物传感器" * 100 + "A", **kwargs)
    assert len(name.encode("utf-8")) <= 255
    assert name != calibration_folder_name(sensor_name="织物传感器" * 100 + "B", **kwargs)
    assert "protocol-proximity-profile_taxels-324" in name
    assert name.endswith("full-passes-5")


def test_full_taxel_names_and_metadata_keep_the_sensor_and_protocol(tmp_path):
    task = FullTaxelCalibrationTask(storage_path=None, output_directory=tmp_path)
    task.active_sensor_key_name = "sensor:testing_20x20_3"
    task.active_sensor_label = "Testing Sensor 3"
    task.protocol = "force_sweep"
    task._targets = [{"taxel_index": i} for i in range(3)]
    name = task._full_folder_name("20261023_160000_123456")
    assert "sensor-Testing-Sensor-3_protocol-force-level-sweep_taxels-3_" in name
    assert "threshold-sweep-5-10-20-50-100g" in name
    task._full_directory = tmp_path / name
    task._full_directory.mkdir()
    task._full_run_id = "20261023_160000_123456"
    task._full_target_results = [{"taxel_index": 0, "summary": {"trial_number": 1}}]
    _csv, metadata_path = task._save_full_summary()
    metadata = json.loads(Path(metadata_path).read_text())
    assert metadata["protocol"] == "force_sweep"
    assert metadata["sensor_name"] == "Testing Sensor 3"
    assert metadata["sensor_identity"] == "sensor:testing_20x20_3"


def _legacy_recording(root, *, old=False):
    folder = root / "calibration_20261023_160000_123456_taxel-12_signal-diff_ave_speed-0p1mmps_threshold-500g_travel-100mm_timeout-6000s_reps-1"
    folder.mkdir(parents=True)
    (folder / "trial_0001.csv").write_bytes(b"elapsed_s,diff_ave\r\n0,-3.125\r\n1,-12.3456789012345\r\n")
    quality = {} if old else {"sensor_identity": "sensor:testing_5x5_1", "protocol": "proximity_profile"}
    metadata = {"version": 7 if old else 8, "sensor_shape": [8, 10] if old else [5, 5],
                "stop_reason": "contact_threshold" if old else "profile_complete",
                "measurement_quality": quality, "csv_path": str(folder / "trial_0001.csv")}
    (folder / "trial_0001.json").write_text(json.dumps(metadata))
    (folder / "batch_summary.csv").write_text("csv_path\n" + str(folder / "trial_0001.csv") + "\n")
    (folder / "batch_summary.json").write_text(json.dumps({"trials": [{"csv_path": metadata["csv_path"]}]}))
    return folder


def test_migration_preserves_samples_updates_references_and_backs_up_edits(tmp_path):
    root = tmp_path / "recordings"
    old = _legacy_recording(root)
    samples = (old / "trial_0001.csv").read_bytes()
    old_metadata = (old / "trial_0001.json").read_bytes()
    original_mode = (old / "trial_0001.json").stat().st_mode
    plan = MIGRATION["plan_recordings"](root)
    assert old.exists()  # Planning is read-only.
    assert plan[0]["sensor_name"] == "Testing Sensor 1"
    manifest = MIGRATION["apply_renames"](plan, tmp_path / "backup")
    new = Path(plan[0]["new_path"])
    assert not old.exists()
    assert (new / "trial_0001.csv").read_bytes() == samples
    metadata = json.loads((new / "trial_0001.json").read_text())
    assert Path(metadata["csv_path"]).is_file()
    assert metadata["sensor_name"] == "Testing Sensor 1"
    assert metadata["protocol"] == "proximity_profile"
    assert (new / "trial_0001.json").stat().st_mode == original_mode
    assert new.name in (new / "batch_summary.csv").read_text()
    assert (tmp_path / "backup/originals" / old.name / "trial_0001.json").read_bytes() == old_metadata
    assert manifest["status"] == "complete"
    assert manifest["unchanged_files_verified"] == 1
    assert MIGRATION["plan_recordings"](root) == []


def test_migration_marks_unknown_legacy_sensor_and_never_overwrites(tmp_path):
    root = tmp_path / "recordings"
    old = _legacy_recording(root, old=True)
    plan = MIGRATION["plan_recordings"](root)
    assert plan[0]["sensor_name"] == "Unknown 8x10 Sensor"
    assert plan[0]["protocol"] == "standard"
    assert "inferred" in plan[0]["protocol_source"]
    new = Path(plan[0]["new_path"])
    new.mkdir()
    with pytest.raises(FileExistsError):
        MIGRATION["apply_renames"](plan, tmp_path / "backup")
    assert old.is_dir() and new.is_dir()


def test_migration_rolls_back_if_a_metadata_write_fails(tmp_path, monkeypatch):
    root = tmp_path / "recordings"
    old = _legacy_recording(root)
    original_files = {p.name: p.read_bytes() for p in old.iterdir()}
    plan = MIGRATION["plan_recordings"](root)
    apply = MIGRATION["apply_renames"]
    atomic = apply.__globals__["_atomic_bytes"]

    def fail_trial_write(path, data):
        if path.name == "trial_0001.json":
            raise OSError("Simulated write failure")
        atomic(path, data)

    monkeypatch.setitem(apply.__globals__, "_atomic_bytes", fail_trial_write)
    with pytest.raises(OSError, match="Simulated"):
        apply(plan, tmp_path / "backup")
    assert not Path(plan[0]["new_path"]).exists()
    assert {p.name: p.read_bytes() for p in old.iterdir()} == original_files
    assert json.loads((tmp_path / "backup/manifest.json").read_text())["status"] == "rolled_back"


def test_migration_preserves_external_changes_detected_during_preflight(tmp_path, monkeypatch):
    root = tmp_path / "recordings"
    old = _legacy_recording(root)
    plan = MIGRATION["plan_recordings"](root)
    apply = MIGRATION["apply_renames"]
    digest = apply.__globals__["_hash"]
    counts = {}
    changed = old / "trial_0001.json"
    external_data = b'{"note": "Updated by another process"}\n'

    def change_on_second_read(path):
        counts[path] = counts.get(path, 0) + 1
        if path == changed and counts[path] == 2:
            path.write_bytes(external_data)
        return digest(path)

    monkeypatch.setitem(apply.__globals__, "_hash", change_on_second_read)
    with pytest.raises(RuntimeError, match="changed during preflight"):
        apply(plan, tmp_path / "backup")
    assert old.is_dir() and not Path(plan[0]["new_path"]).exists()
    assert changed.read_bytes() == external_data


def test_single_trial_and_batch_metadata_are_explicit(tmp_path):
    task = CalibrationTask(storage_path=None, output_directory=tmp_path)
    task.active_sensor_key_name = "sensor:testing_5x5_2"
    task.protocol = "hold_recovery"
    task._samples = [{"elapsed_s": 0, "phase": "approaching", "travel_down_mm": 0,
                      "tool_z_m": 0.2, "force_delta_abs_g": 0, "force_delta_abs_n": 0,
                      "diff_ave": -1, "taxel_035_diff_ave": -1}]
    task._batch_id = "20261023_160000_123456"
    task._batch_directory = tmp_path / task._experiment_folder_name(task._batch_id)
    result = task._save_result()
    task._batch_results = [result]
    _csv, batch_path, _graphs = task._save_batch_summary()
    for path in (result["metadata_path"], batch_path):
        metadata = json.loads(Path(path).read_text())
        assert metadata["sensor_name"] == "Testing Sensor 2"
        assert metadata["sensor_identity"] == "sensor:testing_5x5_2"
        assert metadata["protocol"] == "hold_recovery"

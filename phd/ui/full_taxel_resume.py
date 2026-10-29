"""Disk-only full-taxel resume plans. Loading never writes or moves hardware.

Only orderly stopped runs can be resumed. Trial metadata, not aggregate counts,
is authoritative. Old runs without a manifest are supported for loading/unloading.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone


MANIFEST = "full_taxel_run.json"
SUMMARY = "full_taxel_summary.json"
SETTINGS = (
    "protocol", "signal_field", "approach_speed_m_s", "line_speed_m_s",
    "route_force_guard_g", "contact_threshold_n", "contact_hold_sec",
    "max_travel_m", "timeout_sec", "repeat_count", "motion_mode",
    "far_field_speed_mm_s", "slow_zone_mm", "step_far_mm", "step_near_mm",
    "step_loading_mm", "step_dwell_sec", "sliding_step_mm", "velocity_rest_sec",
    "sensor_baseline_frames", "sensor_baseline_duration_sec", "sensor_baseline_timeout_sec",
    "proximity_window_frames", "proximity_min_margin", "recalibrate_between_trials",
    "recovery_observation_sec", "speed_schedule_mm_s", "force_schedule_g",
    "light_contact_threshold_g", "loading_extra_travel_mm",
    "sliding_distance_mm", "sliding_speed_mm_s", "sliding_direction",
    "specimen_id", "session_id", "mounting_id", "probe_geometry",
    "probe_material", "probe_material_note", "probe_diameter_mm",
    "contact_area_confirmed", "force_reference_verified", "experiment_notes",
    "return_velocity_rad_s", "return_timeout_sec", "return_timeout_factor",
    "return_stall_timeout_sec", "return_progress_epsilon_m", "return_settle_sec",
    "return_arrival_tolerance_rad", "return_height_tolerance_m",
    "recalibration_timeout_sec", "force_baseline_duration_sec", "force_baseline_min_samples",
    "position_tolerance_m", "move_settle_sec", "move_timeout_sec", "stationary_tolerance_m",
)


def settings_snapshot(task):
    return {name: copy.deepcopy(getattr(task, name)) for name in SETTINGS}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {Path(path).name}.")
    return result


def atomic_json(path, data):
    path = Path(path)
    # Replace only after the complete new document is on disk.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def acquire_run_lock(folder):
    # POSIX advisory lock: released by the OS even if the UI process exits.
    import fcntl
    path = _inside(Path(folder).resolve(), Path(folder) / ".full_taxel.lock")
    stream = path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise ValueError("This experiment folder is already in use by another process.")
    return stream


def _legacy_settings(summary, trial, defaults):
    quality = trial.get("measurement_quality") or {}
    if summary.get("protocol") != "standard":
        raise ValueError("Older runs can only resume the Loading / unloading protocol.")
    settings = copy.deepcopy(defaults)
    for name in ("signal_field", "approach_speed_m_s", "max_travel_m", "timeout_sec", "protocol"):
        settings[name] = summary[name]
    settings.update(
        repeat_count=int(summary["repetitions_per_taxel"]),
        line_speed_m_s=summary["route_speed_m_s"],
        route_force_guard_g=summary["route_force_guard_g"],
        contact_threshold_n=float(summary["contact_threshold_g"]) * 0.00980665,
        contact_hold_sec=trial["contact_hold_sec"],
        return_height_tolerance_m=trial["return_height_tolerance_m"],
        recalibrate_between_trials=quality["recalibrate_between_trials"],
        recovery_observation_sec=quality["recovery_observation_s"],
        speed_schedule_mm_s=quality["speed_schedule_mm_s"],
        force_schedule_g=quality["force_schedule_g"],
        experiment_notes=quality.get("notes", ""),
    )
    motion = quality["motion"]
    settings["motion_mode"] = motion["mode"]
    if motion["mode"] == "stepwise":
        for target, source in (
            ("step_far_mm", "far_step_mm"), ("step_near_mm", "near_step_mm"),
            ("step_loading_mm", "loading_step_mm"), ("step_dwell_sec", "time_at_rest_per_step_s"),
            ("light_contact_threshold_g", "light_contact_threshold_g"),
        ):
            settings[target] = motion[source]
    elif motion["mode"] != "velocity":
        raise ValueError("Unsupported saved motion mode.")
    far = quality["far_field_approach"]
    settings.update(far_field_speed_mm_s=far["speed_mm_s"], slow_zone_mm=far["slow_zone_mm"])
    for name in ("specimen_id", "session_id", "mounting_id", "probe_geometry", "probe_diameter_mm",
                 "contact_area_confirmed", "force_reference_verified"):
        value = quality.get("setup", {}).get(name)
        if value is not None:
            settings[name] = value
    return settings


def _inside(folder, path):
    if path.is_symlink() or not path.resolve().is_relative_to(folder):
        raise ValueError(f"Refusing linked/outside recording path: {path.name}")
    return path


def load_plan(source, prepare_map, defaults):
    """Validate a stopped run and return compact metadata, without loading CSV arrays."""
    source = Path(source).expanduser().resolve()
    folder = source if source.is_dir() else source.parent
    summary_path = _inside(folder, folder / SUMMARY)
    if not summary_path.is_file():
        raise ValueError("Stop the experiment normally and wait for full_taxel_summary.json to save first.")
    summary = read_json(summary_path)
    if summary.get("task") != "full_taxel_calibration" or summary.get("status") not in {"aborted", "complete"}:
        raise ValueError("Select the summary of a stopped Full Taxel Calibration run.")
    manifest_path = _inside(folder, folder / MANIFEST)
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    if manifest is not None:
        if manifest.get("version") != 1 or manifest.get("status") != "stopped":
            raise ValueError("This run is active or was not stopped cleanly. Stop it and wait for saving before resuming.")
        if manifest.get("run_id") != summary.get("run_id"):
            raise ValueError("The saved run manifest and summary do not match.")
        if digest(manifest["position_map"]) != manifest["map_digest"]:
            raise ValueError("The saved map snapshot has changed.")
        raw_map = manifest["position_map"]
        settings = manifest["settings"]
        if digest(settings) != manifest.get("settings_digest"):
            raise ValueError("The saved settings snapshot has changed.")
        missing = set(SETTINGS) - set(settings)
        if not set(settings) <= set(SETTINGS) or not missing <= {"probe_material", "probe_material_note"}:
            raise ValueError("The saved settings are incomplete or from an unsupported version.")
        settings = {**{name: copy.deepcopy(defaults[name]) for name in missing}, **settings}
    else:
        raw_map = read_json(summary["position_map_path"])
        settings = None
    prepared = prepare_map(raw_map)
    shape = [int(prepared["settings"][key]) for key in ("sensor_rows", "sensor_columns")]
    if summary["sensor_shape"] != shape:
        raise ValueError("The saved sensor shape does not match its map.")
    requested = summary["requested_taxels"]
    records = {int(item["taxel_index"]): item for item in prepared["positions"]}
    if not requested or len(set(requested)) != len(requested) or set(requested) != set(records):
        raise ValueError("The requested taxels do not match the original position map.")
    prepared["positions"] = [records[index] for index in requested]
    home = summary["home_position"]
    for key in ("tool_position_m", "tool_quaternion_wxyz"):
        if (len(home[key]) != len(prepared["anchor_pose"][key])
                or any(abs(float(a) - float(b)) > 1e-8 for a, b in zip(home[key], prepared["anchor_pose"][key]))):
            raise ValueError("The map anchor has changed since this experiment. Restore the original map.")
    passes = int(summary["repetitions_per_taxel"])
    if not 1 <= passes <= 1000:
        raise ValueError("Invalid number of saved full passes.")
    trials, completed, partial = {}, set(), []
    tracked = [summary_path] + ([manifest_path] if manifest else [])
    first_trial = None
    for index in requested:
        directory = _inside(folder, folder / f"taxel_{index:02d}")
        for number in range(1, passes + 1):
            stem = f"trial_{number:04d}"
            files = sorted(directory.glob(stem + ".*")) if directory.exists() else []
            for path in files:
                _inside(folder, path)
            tracked.extend(files)
            metadata = directory / (stem + ".json")
            if not metadata.exists():
                partial.extend(files)
                continue
            trial = read_json(metadata)
            if (trial.get("taxel_index") != index or trial.get("trial_number") != number
                    or trial.get("sensor_shape") != shape or trial.get("requested_trials") != passes
                    or trial.get("sensor_identity") != summary.get("sensor_identity")
                    or trial.get("protocol") != summary.get("protocol")
                    or trial.get("signal_field") != summary.get("signal_field")
                    or trial.get("batch_id") != f"{summary['run_id']}_taxel{index:02d}"):
                raise ValueError(f"Taxel {index}, pass {number}: saved trial identity/settings mismatch.")
            pose = trial.get("initial_position") or {}
            for key, expected in (
                ("tool_position_m", records[index]["calibration_start_position_m"]),
                ("tool_quaternion_wxyz", records[index]["tool_quaternion_wxyz"]),
            ):
                actual = pose.get(key, [])
                if len(actual) != len(expected) or any(abs(float(a)-float(b)) > 1e-8 for a,b in zip(actual, expected)):
                    raise ValueError(f"Taxel {index}: the map pose differs from the recorded trial.")
            first_trial = first_trial or trial
            if trial.get("measurement_quality", {}).get("trial_completed") is not True:
                partial.extend(files)
                continue
            data_path = directory / (stem + ".csv")
            if not data_path.is_file() or int(trial.get("sample_count", 0)) <= 0:
                raise ValueError(f"Taxel {index}, pass {number}: completed trial data is missing.")
            with data_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                header, row = next(reader, []), next(reader, [])
            if "elapsed_s" not in header or len(row) != len(header):
                raise ValueError(f"Taxel {index}, pass {number}: invalid CSV data.")
            trial.update(csv_path=str(data_path), metadata_path=str(metadata),
                         experiment_directory=str(directory), rows=[])
            graph = directory / (stem + ".png")
            if graph.is_file():
                trial["graph_path"] = str(graph)
            trials.setdefault(index, []).append(trial)
            completed.add((number, index))
    if settings is None:
        if first_trial is None:
            raise ValueError("This older run has no trial metadata to recover its settings. Start a new run.")
        settings = _legacy_settings(summary, first_trial, defaults)
        tracked.append(Path(summary["position_map_path"]))
    if (settings["protocol"] == "drift" or not settings["recalibrate_between_trials"]):
        raise ValueError("Resume needs a fresh sensor baseline. Fixed-baseline/drift runs must start as a new experiment.")
    if (settings["repeat_count"] != passes or settings["protocol"] != summary["protocol"]
            or settings["signal_field"] != summary["signal_field"]):
        raise ValueError("Saved manifest settings and summary do not match.")
    # Saved settings are checked against data, not just the summary's totals.
    for values in trials.values():
        for trial in values:
            for name in ("max_travel_m", "timeout_sec", "contact_hold_sec"):
                if abs(float(trial[name]) - float(settings[name])) > 1e-8:
                    raise ValueError(f"Taxel {trial['taxel_index']}: inconsistent saved {name}.")
            number = int(trial["trial_number"])
            speed = settings["approach_speed_m_s"]
            force = settings["contact_threshold_n"]
            if settings["protocol"] == "speed_sweep":
                schedule = settings["speed_schedule_mm_s"]
                speed = schedule[(number - 1) % len(schedule)] / 1000.
            if settings["protocol"] == "force_sweep":
                schedule = settings["force_schedule_g"]
                force = schedule[(number - 1) % len(schedule)] * .00980665
            if (abs(float(trial["approach_speed_m_s"]) - float(speed)) > 1e-8
                    or abs(float(trial["contact_threshold_n"]) - float(force)) > 1e-8):
                raise ValueError(f"Taxel {trial['taxel_index']}: inconsistent force/speed settings.")
    remaining = [(p, i) for p in range(1, passes + 1) for i in requested if (p, i) not in completed]
    if not remaining:
        raise ValueError("All requested trials are already complete; nothing needs resuming.")
    signature = digest([(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in tracked])
    return dict(folder=folder, summary=summary, settings=settings, position_map=prepared,
                map_source=str(summary["position_map_path"]), trials=trials, completed=completed,
                remaining=remaining, partial=partial, signature=signature,
                history=copy.deepcopy((manifest or {}).get("resume_history", [])), legacy=manifest is None)


def archive_attempts(plan):
    """Called only after explicit Run, lock acquisition and a second disk validation."""
    folder = plan["folder"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    history = _inside(folder, folder / "resume_history")
    archive = history / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for path in (folder / SUMMARY, folder / "full_taxel_summary.csv", folder / MANIFEST):
        if path.is_file():
            shutil.copy2(path, archive / path.name)
    for directory in folder.glob("taxel_*"):
        if directory.is_dir():
            _inside(folder, directory)
            for path in directory.glob("batch_summary.*"):
                _inside(folder, path)
                target = archive / directory.name / path.name
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(path, target)
    for path in plan["partial"]:
        target = archive / "interrupted_attempts" / path.relative_to(folder)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    return dict(resumed_at=datetime.now(timezone.utc).isoformat(), archive=str(archive.relative_to(folder)),
                completed_trials=len(plan["completed"]), remaining_trials=len(plan["remaining"]),
                next_pass=plan["remaining"][0][0], next_taxel=plan["remaining"][0][1],
                baseline_policy="new sensor calibration at safe anchor; new force baseline for every trial",
                legacy_settings_recovered=plan["legacy"])

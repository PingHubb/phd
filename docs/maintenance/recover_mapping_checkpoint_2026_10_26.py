"""Recover the known ndarray checkpoint failure; dry-run unless --apply.

Uses the intact prefix of the failed JSON and measured-position CSV. It never
opens hardware. Stop the UI first. Original files are backed up byte-for-byte;
only taxel_position_map.json is replaced. No positions are inferred from the
old map: that map is retained solely as the relocation fit's source.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from phd.ui.experiment_tasks import TaxelPositionMappingTask, _mapping_json_default
from phd.ui.taxel_position_navigation import prepare_taxel_map


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(folder, source_path):
    folder = Path(folder).resolve()
    source_path = Path(source_path).resolve()
    paths = [folder / name for name in (
        "taxel_position_map.json", "taxel_position_map.json.tmp", "taxel_positions.csv",
        "scan_samples.csv", "baseline_checks.csv", "probe_levels.csv",
    )]
    hashes = {str(path): sha(path) for path in [*paths, source_path]}
    old = json.loads(paths[0].read_text())
    text = paths[1].read_text()
    # Reject any other truncation rather than guessing how to repair it.
    marker = '  "relocation": {\n    "rotation": '
    if not text.endswith(marker) or text.count(marker) != 1:
        raise ValueError("Not the known relocation.rotation serialization failure")
    prefix = json.loads(text[:-len(marker)] + '  "relocation": null\n}')
    if (prefix.get("stop_reason") != "checkpoint_failed"
            or "ndarray is not JSON serializable" not in prefix.get("stop_message", "")):
        raise ValueError("Not an ndarray checkpoint failure")
    for key in ("task", "started_at", "sensor_identity", "settings", "anchor_pose"):
        if prefix[key] != old[key]:
            raise ValueError(f"Older JSON and temporary JSON disagree on {key}")
    source = prepare_taxel_map(json.loads(source_path.read_text()))
    if source.get("sensor_identity") != old.get("sensor_identity"):
        raise ValueError("Source sensor identity differs")
    source_valid, message = TaxelPositionMappingTask.validate_relocation_result(source)
    if not source_valid:
        raise ValueError(message)
    old_positions = {r["taxel_index"]: r for r in old["positions"]}
    recovered = []
    with paths[2].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    coords = {"x_m", "y_m", "z_m", "safe_x_m", "safe_y_m", "safe_z_m"}
    for row in rows:
        index = int(row["taxel_index"])
        if row["status"] != "measured":
            raise ValueError("CSV contains a non-measured taxel")
        record = copy.deepcopy(old_positions.get(index, {}))
        record["tool_position_m"] = [float(row[key]) for key in ("x_m", "y_m", "z_m")]
        record["safe_approach_position_m"] = [float(row[key]) for key in ("safe_x_m", "safe_y_m", "safe_z_m")]
        if index in old_positions:
            for key in ("tool_position_m", "safe_approach_position_m"):
                if record[key] != old_positions[index][key]:
                    raise ValueError(f"Taxel {index} changed coordinates between checkpoints")
        else:
            # All mapping motion uses this fixed quaternion. It is not an
            # independently recovered orientation measurement for this taxel.
            record["tool_quaternion_wxyz"] = list(old["anchor_pose"]["tool_quaternion_wxyz"])
            record["recovery_note"] = "Measured CSV pose; commanded orientation from run anchor. Fields absent from CSV were not reconstructed."
        for key, value in row.items():
            if key in coords:
                continue
            if value == "":
                parsed = None
            elif value in ("True", "False"):
                parsed = value == "True"
            else:
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = value
            record[key] = parsed
        recovered.append(record)
    if len(recovered) != prefix["measured_position_count"]:
        raise ValueError("CSV measurement count differs from failed checkpoint")
    result = dict(old, **prefix, positions=recovered)
    # Preserve paths and histories; this remains incomplete, never a finished
    # or validated relocated map.
    result["completed"] = False
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    task._apply_resume_settings(result)
    task._positions = {record["taxel_index"]: record for record in recovered}
    task._relocation_active = True
    task._relocation_source_result = source
    task._relocation_source_path = source_path
    task._relocation_source_positions = {r["taxel_index"]: r for r in source["positions"]}
    references = sorted([task.anchor_taxel, *task.anchor_neighbour_indices()])
    if set(references) != set(old_positions):
        raise ValueError("Expected the intact five-reference checkpoint before first verification")
    task._relocation_reference_indices = references
    fitted = task._fit_relocation_transform()
    if fitted is None:
        raise ValueError("Recovered measurements cannot fit a relocation transform")
    result["relocation"] = fitted
    result["relocation_checkpoint"] = task._relocation_checkpoint()
    result["lattice_fit"] = task.lattice_fit_quality()
    result["recovery"] = dict(
        recovered_at=datetime.now(timezone.utc).isoformat(),
        reason="Recovered ndarray checkpoint failure without new measurements",
        source_hashes=hashes,
        added_measured_taxels=sorted(set(task._positions) - set(old_positions)),
        transform="Recomputed from recovered measured centres and explicit source map",
        probe_note="Operator reports smaller-diameter, longer tip of changed material. Verify TCP/physical clearance before resuming; no geometry correction inferred.",
    )
    result = json.loads(json.dumps(result, default=_mapping_json_default, allow_nan=False))
    prepare_taxel_map(result)
    valid, message = task.validate_resume_result(result)
    if not valid:
        raise ValueError(message)
    remaining = [i for i in task._choose_relocation_verification_indices(3) if i not in task._positions]
    return result, paths, hashes, remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result, paths, hashes, remaining = inspect(args.folder, args.source_map)
    print(f"Validated {len(result['positions'])} measured taxels. Remaining distributed verifications: {remaining}; anchor will be checked again on Resume.")
    if not args.apply:
        print("Dry run: no files changed.")
        return
    # Mapping opens files briefly per checkpoint, so checking FDs alone cannot
    # prove it has stopped. Require the desktop UI process to be closed.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError):
            continue
        if any(token.endswith(b"/phd_ui") or token == b"phd_ui" for token in argv):
            raise ValueError("Close the research UI before recovering its recording")
    for name, expected in hashes.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Recording changed during recovery: {name}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = paths[0].parent / ".checkpoint_recovery" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    for path in paths:
        target = backup / (path.name + ".bak")
        shutil.copy2(path, target)
        if sha(target) != hashes[str(path)]:
            raise ValueError(f"Backup verification failed: {path}")
    source_backup = backup / "relocation_source_map.json.bak"
    shutil.copy2(args.source_map, source_backup)
    if sha(source_backup) != hashes[str(args.source_map.resolve())]:
        raise ValueError("Source backup verification failed")
    result["recovery"]["backup_folder"] = str(backup)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    candidate = backup / "recovered_map.json"
    candidate.write_text(text)
    shutil.copy2(candidate, paths[0].with_suffix(".json.recovery-tmp"))
    for name, expected in hashes.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Recording changed before replacement: {name}")
    os.replace(paths[0].with_suffix(".json.recovery-tmp"), paths[0])
    assert json.loads(paths[0].read_text()) == result
    print(f"Recovered: {paths[0]}\nOriginals: {backup}\nRobot was not accessed.")


if __name__ == "__main__":
    main()

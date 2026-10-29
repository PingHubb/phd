"""Rename completed calibration folders from their saved metadata.

The default is a dry run. --apply requires a new backup directory and preserves
every changed metadata/summary file there. Raw samples and figures are hashed
before and after; only metadata and text path references may change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phd.dependence.calibration_recording_names import (
    calibration_folder_name,
    sensor_name_for_recording,
)


OLD_NAME = re.compile(
    r"^(calibration(?:_full_taxel)?)_(\d{8}_\d{6}_\d{6})_(.+)$"
)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_recordings(root):
    """Resolve identities from the recordings, never from the connected device."""
    plan = []
    for folder in sorted(Path(root).resolve().iterdir()):
        match = OLD_NAME.fullmatch(folder.name)
        if folder.is_symlink() or not folder.is_dir() or not match:
            continue
        kind, stamp, settings = match.groups()
        if settings.startswith("sensor-"):
            continue
        records = [json.loads(p.read_text()) for p in sorted(folder.rglob("*.json"))]
        identities, labels, protocols, shapes = set(), set(), set(), set()
        trials = []
        for record in records:
            quality = record.get("measurement_quality") or {}
            for item in (record, quality):
                if item.get("sensor_identity"):
                    identities.add(str(item["sensor_identity"]))
                if item.get("sensor_name"):
                    labels.add(str(item["sensor_name"]))
                if item.get("protocol"):
                    protocols.add(str(item["protocol"]))
            if record.get("sensor_shape"):
                shapes.add(tuple(record["sensor_shape"]))
            if record.get("csv_path"):
                trials.append(record)
        if any(len(values) > 1 for values in (identities, labels, protocols, shapes)):
            raise ValueError(f"Conflicting sensor/protocol metadata: {folder}")
        identity = next(iter(identities), "")
        label = next(iter(labels), "")
        shape = next(iter(shapes), (0, 0))
        protocol = next(iter(protocols), "unknown")
        protocol_source = "recorded metadata"
        if not protocols:
            # Version 7 and earlier predate the protocol selector. Their
            # contact-threshold trials implement the original loading cycle.
            if trials and all(
                0 < int(item.get("version", 0)) <= 7
                and item.get("stop_reason") == "contact_threshold"
                for item in trials
            ):
                protocol = "standard"
                protocol_source = "inferred from legacy loading/unloading trials"
            else:
                protocol_source = "not recorded; left unknown"
        sensor_name = sensor_name_for_recording(label, identity, shape)
        new_name = calibration_folder_name(
            kind, stamp, settings, sensor_name=sensor_name, protocol=protocol,
        )
        target = folder.with_name(new_name)
        if target.exists():
            raise FileExistsError(f"Rename target already exists: {target}")
        plan.append({
            "old_path": str(folder), "new_path": str(target),
            "sensor_name": sensor_name, "sensor_identity": identity,
            "protocol": protocol, "protocol_source": protocol_source,
            "sensor_name_source": "recorded identity/name" if identity or label else "unknown; grid shape only",
        })
    return plan


def _replace_names(text, plan):
    for item in plan:
        text = text.replace(Path(item["old_path"]).name, Path(item["new_path"]).name)
    return text


def _atomic_bytes(path, data):
    handle, name = tempfile.mkstemp(prefix=".recording-rename-", dir=path.parent)
    try:
        if path.exists():
            os.fchmod(handle, stat.S_IMODE(path.stat().st_mode))
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def apply_renames(plan, backup_directory):
    """Apply a preflighted plan, back up edits, and verify unchanged samples."""
    backup = Path(backup_directory).resolve()
    if backup.exists():
        raise FileExistsError(f"Backup directory already exists: {backup}")
    before, changes = {}, {}
    for item in plan:
        old = Path(item["old_path"])
        if Path(item["new_path"]).exists():
            raise FileExistsError(item["new_path"])
        for path in sorted(old.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink inside recording: {path}")
            if not path.is_file():
                continue
            relative = str(path.relative_to(old))
            before[(str(old), relative)] = _hash(path)
            if path.suffix not in {".json", ".csv"}:
                continue
            original = path.read_bytes()
            text = _replace_names(original.decode("utf-8"), plan)
            if path.suffix == ".json":
                data = json.loads(text)
                if any(key in data for key in ("csv_path", "trials", "taxel_results", "target_taxel")):
                    for key in ("sensor_name", "sensor_identity", "protocol"):
                        data.setdefault(key, item[key])
                    quality = data.get("measurement_quality")
                    if isinstance(quality, dict):
                        quality.setdefault("sensor_name", item["sensor_name"])
                text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            updated = text.encode("utf-8")
            if updated != original:
                changes[(str(old), relative)] = updated

    backup.mkdir(parents=True)
    for old, relative in changes:
        source = Path(old) / relative
        copy = backup / "originals" / Path(old).name / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copy)
    manifest = {
        "status": "prepared", "renames": plan,
        "changed_files": [
            {"old_path": str(Path(old) / rel), "backup": str(backup / "originals" / Path(old).name / rel)}
            for old, rel in changes
        ],
        "original_sha256": [
            {"old_directory": old, "relative_path": rel, "sha256": digest}
            for (old, rel), digest in before.items()
        ],
    }
    manifest_path = backup / "manifest.json"
    _atomic_bytes(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
    renamed = []
    written = []
    try:
        # Fail before renaming if a recording changed during inspection.
        for item in plan:
            old = Path(item["old_path"])
            current_files = {str(p.relative_to(old)) for p in old.rglob("*") if p.is_file()}
            expected_files = {rel for folder, rel in before if folder == str(old)}
            if current_files != expected_files:
                raise RuntimeError(f"Recording file list changed during preflight: {old}")
        for (old, relative), digest in before.items():
            if _hash(Path(old) / relative) != digest:
                raise RuntimeError(f"Recording changed during preflight: {Path(old) / relative}")
        for item in plan:
            old, new = Path(item["old_path"]), Path(item["new_path"])
            if new.exists():
                raise FileExistsError(str(new))
            old.rename(new)
            renamed.append(item)
        destinations = {item["old_path"]: Path(item["new_path"]) for item in plan}
        for (old, relative), data in changes.items():
            written.append((old, relative))
            _atomic_bytes(destinations[old] / relative, data)
        preserved = 0
        for (old, relative), digest in before.items():
            path = destinations[old] / relative
            if (old, relative) in changes:
                expected = hashlib.sha256(changes[(old, relative)]).hexdigest()
            else:
                expected = digest
                preserved += 1
            if _hash(path) != expected:
                raise RuntimeError(f"Verification failed: {path}")
        manifest.update(status="complete", unchanged_files_verified=preserved)
    except Exception:
        for item in reversed(renamed):
            Path(item["new_path"]).rename(item["old_path"])
        for old, relative in written:
            saved = backup / "originals" / Path(old).name / relative
            shutil.copy2(saved, Path(old) / relative)
        manifest["status"] = "rolled_back" if renamed else "aborted_before_rename"
        _atomic_bytes(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
        raise
    _atomic_bytes(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-directory", type=Path)
    args = parser.parse_args()
    plan = plan_recordings(args.root)
    if not args.apply:
        print(json.dumps(plan, indent=2))
        return
    if args.backup_directory is None:
        parser.error("--apply requires --backup-directory")
    result = apply_renames(plan, args.backup_directory)
    print(json.dumps({key: result[key] for key in ("status", "renames", "unchanged_files_verified")}, indent=2))


if __name__ == "__main__":
    main()

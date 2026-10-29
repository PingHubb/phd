"""Remove non-target taxel columns from an explicitly selected stopped run.

Dry-run by default. --apply permanently discards other-taxel measurements.
Every retained CSV value is verified before any original is replaced. Small
metadata backups and a verification report remain in .target_taxel_compaction.
No hardware imports, no motion, and no recursive search of other experiments.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from phd.ui.full_taxel_resume import acquire_run_lock, atomic_json, read_json


TAXEL = re.compile(r"^taxel_(\d+)_(.+)$")


def fingerprint(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def local_file(folder, path):
    if any(p.is_symlink() for p in (path, *path.parents) if p != folder.parent):
        raise ValueError(f"Linked recording path: {path}")
    if not path.resolve().is_relative_to(folder) or not path.is_file():
        raise ValueError(f"Missing or outside recording file: {path}")
    return path


def check_writers(folder):
    """Also protect legacy UI versions that predate the run lock (Linux)."""
    proc = Path('/proc')
    if not proc.is_dir():
        return
    prefix = str(folder) + os.sep
    for process in proc.iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            descriptors = list((process / 'fd').iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for fd in descriptors:
            try:
                target = os.readlink(fd)
                if not target.startswith(prefix):
                    continue
                info = (process / 'fdinfo' / fd.name).read_text()
                flags = int(re.search(r'^flags:\s+(\d+)', info, re.M)[1], 8)
                if flags & os.O_ACCMODE != os.O_RDONLY:
                    raise ValueError(f"Run has an open writer (PID {process.name}): {target}")
            except FileNotFoundError:
                continue


def retained_columns(header, target):
    if len(set(header)) != len(header):
        raise ValueError('Duplicate CSV columns')
    if not {'elapsed_s', 'cell_index', 'raw', 'diff'}.issubset(header):
        raise ValueError('Not a calibration sample CSV')
    keep, found = [], set()
    for i, name in enumerate(header):
        match = TAXEL.fullmatch(name)
        if name.startswith('taxel_') and match is None:
            raise ValueError(f'Unrecognised taxel column: {name}')
        if match is None or int(match[1]) == target:
            keep.append(i)
        if match is not None:
            found.add(int(match[1]))
    if target not in found:
        raise ValueError(f'No recorded columns for target taxel {target}')
    return keep


def inspect_run(folder):
    folder = Path(folder).absolute()
    local_file(folder, folder / 'full_taxel_summary.json')
    summary = read_json(folder / 'full_taxel_summary.json')
    if summary.get('task') != 'full_taxel_calibration' or summary.get('status') not in {'complete', 'aborted'}:
        raise ValueError('Select an orderly stopped Full Taxel Calibration run')
    guards = {folder / 'full_taxel_summary.json': fingerprint(folder / 'full_taxel_summary.json')}
    manifest = folder / 'full_taxel_run.json'
    if manifest.exists():
        local_file(folder, manifest)
        data = read_json(manifest)
        if data.get('status') != 'stopped' or data.get('run_id') != summary['run_id']:
            raise ValueError('Run is active or its manifest does not match')
        guards[manifest] = fingerprint(manifest)
    check_writers(folder)
    if list(folder.glob('.target_taxel_compaction/*/full_grid_restore/**/*.csv')):
        raise ValueError('A previous conversion was interrupted; recover its full_grid_restore files first')
    trials = []
    for path in sorted(folder.glob('taxel_*/trial_*.csv')):
        if re.fullmatch(r'trial_\d+\.csv', path.name) is None:
            continue  # Keep derived proximity-profile tables unchanged.
        local_file(folder, path)
        metadata_path = local_file(folder, path.with_suffix('.json'))
        metadata = read_json(metadata_path)
        target = metadata['taxel_index']
        if type(target) is not int or target < 0 or int(path.parent.name[6:]) != target:
            raise ValueError(f'Taxel identity mismatch: {path}')
        if (metadata['batch_id'] != f"{summary['run_id']}_taxel{target:02d}"
                or metadata.get('sensor_identity') != summary.get('sensor_identity')
                or metadata.get('sensor_shape') != summary.get('sensor_shape')):
            raise ValueError(f'Run identity mismatch: {path}')
        with path.open(newline='', encoding='utf-8-sig') as stream:
            header = next(csv.reader(stream))
        keep = retained_columns(header, target)
        guards[path] = fingerprint(path)
        guards[metadata_path] = fingerprint(metadata_path)
        trials.append(dict(path=path, metadata=metadata, header=header, keep=keep, target=target))
    if not trials:
        raise ValueError('No saved trial CSVs in this run')
    for path in folder.glob('taxel_*/batch_summary.json'):
        local_file(folder, path)
        if read_json(path).get('taxel_index') != int(path.parent.name[6:]):
            raise ValueError(f'Batch taxel mismatch: {path}')
        guards[path] = fingerprint(path)
    return folder, trials, guards


def unchanged(guards):
    for path, expected in guards.items():
        if fingerprint(path) != expected:
            raise ValueError(f'Recording changed during conversion: {path}')


def hash_row(digest, row):
    # JSON separates fields unambiguously, preserving blanks and numeric spelling.
    digest.update(json.dumps(row, ensure_ascii=True, separators=(',', ':')).encode())
    digest.update(b'\n')


def project_csv(source, destination, target, expected_samples):
    expected_hash, actual_hash = hashlib.sha256(), hashlib.sha256()
    with source.open(newline='', encoding='utf-8-sig') as src, destination.open('x', newline='', encoding='utf-8') as dst:
        reader = csv.reader(src, strict=True)
        header = next(reader)
        keep = retained_columns(header, target)
        selected = [header[i] for i in keep]
        writer = csv.writer(dst)
        writer.writerow(selected)
        hash_row(expected_hash, selected)
        count = 0
        cell = header.index('cell_index')
        for row in reader:
            if len(row) != len(header) or row[cell] != str(target):
                raise ValueError(f'Malformed row or wrong target at sample {count + 1}: {source}')
            projected = [row[i] for i in keep]
            writer.writerow(projected)
            hash_row(expected_hash, projected)
            count += 1
        if count != expected_samples:
            raise ValueError(f'Sample count mismatch: {source}: {count} != {expected_samples}')
        dst.flush()
        os.fsync(dst.fileno())
    # Read the staged output independently before permitting replacement.
    with destination.open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream, strict=True)
        output_header = next(reader)
        if output_header != selected:
            raise ValueError('Staged header mismatch')
        hash_row(actual_hash, output_header)
        actual_count = 0
        for row in reader:
            if len(row) != len(selected):
                raise ValueError('Staged row width mismatch')
            hash_row(actual_hash, row)
            actual_count += 1
    if actual_count != count or actual_hash.digest() != expected_hash.digest():
        raise ValueError(f'Retained values failed verification: {source}')
    shutil.copymode(source, destination)
    return dict(samples=count, retained_columns=len(selected), original_columns=len(header),
                retained_values_sha256=actual_hash.hexdigest(),
                original_bytes=source.stat().st_size, retained_bytes=destination.stat().st_size)


def trim_metadata(payload, target):
    """Preserve all analyses/settings; correct only the recorded-column declaration."""
    payload = dict(payload)
    payload.update(recorded_taxel_indices=[target], recording_scope='target_taxel_only')
    return payload


def compact_run(folder, *, apply=False, progress=print):
    folder, trials, guards = inspect_run(folder)
    plan = dict(folder=str(folder), trial_files=len(trials),
                original_csv_bytes=sum(t['path'].stat().st_size for t in trials),
                columns_to_remove=sum(len(t['header']) - len(t['keep']) for t in trials))
    if not apply:
        return plan
    with acquire_run_lock(folder):
        unchanged(guards)
        check_writers(folder)
        if not plan['columns_to_remove']:
            return dict(plan, status='already_target_only')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        audit = folder / '.target_taxel_compaction' / stamp
        audit.mkdir(parents=True, exist_ok=False)
        report_path = audit / 'verification.json'
        report = dict(plan, status='staging', created_at=stamp, trials=[],
                      warning='Other-taxel measurements are permanently removed. Metadata backups are not signal backups.')
        atomic_json(report_path, report)
        # Stage and verify ALL trials before replacing ANY original.
        for number, trial in enumerate(trials, 1):
            path = trial['path']
            relative = path.relative_to(folder)
            staged = audit / 'staged' / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            stats = project_csv(path, staged, trial['target'], trial['metadata']['sample_count'])
            report['trials'].append(dict(path=str(relative), taxel_index=trial['target'], **stats))
            progress(f"Verified {number}/{len(trials)}: Taxel {trial['target']}, {stats['samples']} samples", flush=True)
        unchanged(guards)
        check_writers(folder)
        report['status'] = 'verified_before_replacement'
        atomic_json(report_path, report)
        # Tiny metadata backups remain available; the full-grid data do not.
        for path in guards:
            if path.suffix == '.json':
                backup = audit / 'original_metadata' / path.relative_to(folder)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
        for trial, entry in zip(trials, report['trials']):
            unchanged({p: guards[p] for p in (folder / 'full_taxel_summary.json', trial['path'], trial['path'].with_suffix('.json'))})
            path = trial['path']
            relative = path.relative_to(folder)
            backup = audit / 'full_grid_restore' / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, backup)
            try:
                os.replace(audit / 'staged' / relative, path)
                atomic_json(path.with_suffix('.json'), trim_metadata(trial['metadata'], trial['target']))
                entry['replaced'] = True
                atomic_json(report_path, report)
            except BaseException:
                os.replace(backup, path)
                shutil.copy2(audit / 'original_metadata' / relative.with_suffix('.json'), path.with_suffix('.json'))
                raise
            # Delete only the verified original's extra hard link, after both files commit.
            backup.unlink()
        for path in folder.glob('taxel_*/batch_summary.json'):
            unchanged({path: guards[path]})
            payload = read_json(path)
            target = int(path.parent.name[6:])
            if payload['taxel_index'] != target:
                raise ValueError(f'Batch taxel mismatch: {path}')
            atomic_json(path, trim_metadata(payload, target))
        report.update(status='complete', retained_csv_bytes=sum(e['retained_bytes'] for e in report['trials']))
        report['bytes_removed'] = report['original_csv_bytes'] - report['retained_csv_bytes']
        atomic_json(report_path, report)
        return {k: v for k, v in report.items() if k != 'trials'} | {'verification_report': str(report_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folder', type=Path)
    parser.add_argument('--apply', action='store_true', help='Permanently remove other-taxel columns after verification')
    args = parser.parse_args()
    print(json.dumps(compact_run(args.run_folder, apply=args.apply), indent=2))


if __name__ == '__main__':
    main()

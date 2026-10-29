"""Read-only, on-demand browsing of trials from one Full Taxel experiment."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

from phd.ui.calibration_analysis import available_taxel_indices, read_calibration_csv_rows


SIGNALS = ('diff_ave', 'diff', 'diff_percent_ave', 'diff_percent', 'raw_ave', 'raw')


def _stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def catalog_full_taxel_run(selected):
    """Only discover siblings inside the enclosing run, never another run/history."""
    selected = Path(selected).resolve()
    root = None
    for directory in list(selected.parents)[:3]:
        summary = directory / 'full_taxel_summary.json'
        if summary.is_file():
            root = directory
            break
    if root is None:
        return None
    summary = json.loads((root / 'full_taxel_summary.json').read_text())
    if summary.get('task') != 'full_taxel_calibration':
        return None
    # Do not substitute current trials when the user deliberately opens history.
    relative = selected.relative_to(root)
    if relative.parts[0] == 'resume_history' or relative.parts[0].startswith('.'):
        return None
    datasets, warnings = [], []
    for directory in sorted(root.glob('taxel_*'), key=lambda p: p.name):
        if not re.fullmatch(r'taxel_\d+', directory.name) or directory.is_symlink():
            continue
        target = int(directory.name[6:])
        for path in sorted(directory.glob('trial_*.json')):
            if not re.fullmatch(r'trial_\d+\.json', path.name):
                continue
            try:
                if path.is_symlink() or path.with_suffix('.csv').is_symlink():
                    raise ValueError('linked trial file')
                meta = json.loads(path.read_text())
                if int(meta['taxel_index']) != target:
                    raise ValueError('taxel identity does not match its folder')
                rows_n, columns_n = map(int, meta['sensor_shape'])
                if min(rows_n, columns_n) < 1 or not 0 <= target < rows_n * columns_n:
                    raise ValueError('taxel index is outside the saved sensor shape')
                for key in ('sensor_identity', 'sensor_shape'):
                    if summary.get(key) is not None and meta.get(key) != summary[key]:
                        raise ValueError(f'{key} does not match the experiment')
                if summary.get('run_id') and meta.get('batch_id') != f"{summary['run_id']}_taxel{target:02d}":
                    raise ValueError('trial belongs to a different run')
                csv_path = path.with_suffix('.csv')
                with csv_path.open(newline='', encoding='utf-8-sig') as stream:
                    reader = csv.reader(stream)
                    header, first = next(reader), next(reader)
                if len(header) != len(first) or 'elapsed_s' not in header:
                    raise ValueError('invalid sample CSV')
                preferred = meta.get('signal_field') or summary.get('signal_field')
                field = next((f for f in (preferred, *SIGNALS) if f and f in header), None)
                if not field:
                    raise ValueError('no supported signal column')
                if 'cell_index' in header and int(float(first[header.index('cell_index')])) != target:
                    raise ValueError('CSV target does not match metadata')
                indices = sorted({int(m[1]) for key in header
                                  if (m := re.fullmatch(r'taxel_(\d+)_' + re.escape(field), key))}) or [target]
                if target not in indices:
                    raise ValueError('target signal is missing')
                number = int(meta.get('trial_number', int(path.stem[6:])))
                if number != int(path.stem[6:]):
                    raise ValueError('trial number does not match its filename')
                partial = (meta.get('measurement_quality') or {}).get('trial_completed') is False
                datasets.append(dict(
                    name=f"Trial {number:04d}" + (' · partial' if partial else ''),
                    rows=[], signal_field=field, source_path=str(csv_path), metadata=meta,
                    deferred_csv=True, csv_stamp=_stamp(csv_path), recorded_indices=indices,
                ))
            except (OSError, ValueError, TypeError, KeyError, StopIteration) as exc:
                warnings.append(f'{directory.name}/{path.name}: {exc}')
    if not datasets:
        raise ValueError('No readable calibration trials in this experiment. ' + '; '.join(warnings[:3]))
    datasets.sort(key=lambda d: (int(d['metadata']['taxel_index']), int(d['metadata'].get('trial_number', 0))))
    initial = next((i for i, d in enumerate(datasets)
                    if Path(d['source_path']).stem == selected.stem
                    and Path(d['source_path']).parent in selected.parents), None)
    if initial is None:
        initial = next((i for i, d in enumerate(datasets) if Path(d['source_path']).parent in selected.parents), 0)
    return dict(selected_path=str(selected), datasets=datasets, initial_dataset_index=initial,
                full_taxel_run=str(root), catalog_warnings=warnings)


def recorded_taxels(dataset):
    if dataset.get('deferred_csv'):
        return list(dataset['recorded_indices'])
    return available_taxel_indices(dataset.get('rows') or [], dataset.get('signal_field') or 'diff_ave',
                                   legacy_index=int((dataset.get('metadata') or {}).get('taxel_index', 0) or 0))


def _unchanged(dataset):
    if _stamp(dataset['source_path']) != tuple(dataset['csv_stamp']):
        raise ValueError('The recording changed on disk. Close Graph Analysis and reopen the saved experiment.')


def dataset_rows(dataset):
    if dataset.get('deferred_csv'):
        _unchanged(dataset)
        if not dataset.get('rows'):
            rows = read_calibration_csv_rows(dataset['source_path'])
            _unchanged(dataset)
            if not rows:
                raise ValueError('The trial has no saved samples.')
            count = dataset['metadata'].get('sample_count')
            if count is not None and len(rows) != int(count):
                raise ValueError('Trial sample count differs from its metadata. Reopen after saving finishes.')
            for row in rows:
                if row.get('force_delta_abs_g') is None and row.get('force_delta_abs_n') is not None:
                    row['force_delta_abs_g'] = abs(float(row['force_delta_abs_n'])) / .00980665
            dataset['rows'] = rows
    return dataset.get('rows') or []


def dataset_axis_bounds(dataset):
    """Scan only display-axis values, without retaining every trial in RAM."""
    _unchanged(dataset)
    if 'axis_bounds' in dataset:
        return dataset['axis_bounds']
    field = dataset['signal_field']
    bounds = {}
    def add(key, raw):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        low, high = bounds.get(key, (value, value))
        bounds[key] = min(low, value), max(high, value)
    with Path(dataset['source_path']).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        signal_columns = [f'taxel_{i:03d}_{field}' for i in recorded_taxels(dataset)]
        signal_columns = [key for key in signal_columns if key in header] or [field]
        columns = [(header.index(key), field) for key in signal_columns]
        columns += [(header.index(key), key) for key in ('elapsed_s', 'travel_down_mm') if key in header]
        grams = header.index('force_delta_abs_g') if 'force_delta_abs_g' in header else None
        newtons = header.index('force_delta_abs_n') if 'force_delta_abs_n' in header else None
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f'Malformed sample row in {dataset["source_path"]}')
            for position, key in columns:
                add(key, row[position])
            force = row[grams] if grams is not None else ''
            try:
                valid_force = math.isfinite(float(force))
            except (TypeError, ValueError):
                valid_force = False
            if not valid_force and newtons is not None and row[newtons]:
                force = abs(float(row[newtons])) / .00980665
            add('force_delta_abs_g', force)
    _unchanged(dataset)
    dataset['axis_bounds'] = bounds
    return bounds

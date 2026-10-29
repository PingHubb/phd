import csv
import json
from pathlib import Path

import pytest

from phd.script.conversion import compact_full_taxel_recording as compact
from phd.ui.full_taxel_resume import acquire_run_lock


def saved_run(tmp_path):
    folder = tmp_path / 'run'
    trial = folder / 'taxel_12' / 'trial_0001.csv'
    trial.parent.mkdir(parents=True)
    summary = dict(task='full_taxel_calibration', status='aborted', run_id='test',
                   sensor_identity='sensor:test', sensor_shape=[5, 5])
    (folder / 'full_taxel_summary.json').write_text(json.dumps(summary))
    metadata = dict(taxel_index=12, batch_id='test_taxel12', sensor_identity='sensor:test',
                    sensor_shape=[5, 5], sample_count=2, recorded_taxel_indices=[0, 12, 24],
                    analysis={'turning_point': 1.2345}, measurement_quality={'trial_completed': False})
    trial.with_suffix('.json').write_text(json.dumps(metadata))
    (trial.parent / 'batch_summary.json').write_text(json.dumps(metadata))
    header = ['elapsed_s', 'cell_index', 'raw', 'diff', 'force_raw_g', 'tool_z_m', 'phase',
              'taxel_000_diff', 'taxel_012_diff', 'taxel_024_diff']
    rows = [['0.0000001', '12', '1.2345678901234567', '-0.0', '', '1e-09', 'a,b\nc', '-9', '-0.0', '4'],
            ['1.1', '12', '2', '-3.50', '0', '0.0005', 'returning', '-5', '-3.50', '3']]
    with trial.open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
    return folder, trial, header, rows, metadata


def silent(*args, **kwargs):
    pass


def test_compaction_preserves_every_retained_value_and_metadata(tmp_path):
    folder, trial, header, rows, metadata = saved_run(tmp_path)
    original_summary = (folder / 'full_taxel_summary.json').read_bytes()
    result = compact.compact_run(folder, apply=True, progress=silent)
    with trial.open(newline='') as stream:
        actual = list(csv.reader(stream))
    keep = [i for i, key in enumerate(header) if key not in {'taxel_000_diff', 'taxel_024_diff'}]
    assert actual == [[row[i] for i in keep] for row in [header, *rows]]
    edited = json.loads(trial.with_suffix('.json').read_text())
    assert edited['recorded_taxel_indices'] == [12]
    assert edited['analysis'] == metadata['analysis']
    assert edited['measurement_quality'] == metadata['measurement_quality']
    assert (folder / 'full_taxel_summary.json').read_bytes() == original_summary
    assert result['status'] == 'complete' and result['bytes_removed'] > 0
    assert not list(folder.glob('.target_taxel_compaction/*/full_grid_restore/**/*.csv'))
    assert compact.compact_run(folder, apply=True, progress=silent)['status'] == 'already_target_only'


def test_dry_run_does_not_write(tmp_path):
    folder, *_ = saved_run(tmp_path)
    before = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    assert compact.compact_run(folder)['columns_to_remove'] == 2
    assert {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('failure', ['samples', 'cell', 'duplicate', 'missing_target', 'identity', 'active', 'symlink'])
def test_invalid_input_never_replaces_data(tmp_path, failure):
    folder, trial, _, _, metadata = saved_run(tmp_path)
    if failure == 'samples':
        metadata['sample_count'] = 3
    elif failure == 'cell':
        trial.write_text(trial.read_text().replace(',12,', ',13,'))
    elif failure == 'duplicate':
        trial.write_text(trial.read_text().replace('taxel_000_diff', 'taxel_012_diff'))
    elif failure == 'missing_target':
        trial.write_text(trial.read_text().replace('taxel_012_diff', 'taxel_013_diff'))
    elif failure == 'identity':
        metadata['sensor_identity'] = 'different'
    elif failure == 'active':
        (folder / 'full_taxel_run.json').write_text(json.dumps(dict(status='running', run_id='test')))
    elif failure == 'symlink':
        external = tmp_path / 'external.csv'
        trial.rename(external)
        trial.symlink_to(external)
    trial.with_suffix('.json').write_text(json.dumps(metadata))
    original = trial.read_bytes()
    with pytest.raises(ValueError):
        compact.compact_run(folder, apply=True, progress=silent)
    assert trial.read_bytes() == original


def test_active_lock_blocks_conversion(tmp_path):
    folder, trial, *_ = saved_run(tmp_path)
    original = trial.read_bytes()
    with acquire_run_lock(folder), pytest.raises(ValueError, match='in use'):
        compact.compact_run(folder, apply=True, progress=silent)
    assert trial.read_bytes() == original


def test_metadata_failure_rolls_back_csv(tmp_path, monkeypatch):
    folder, trial, *_ = saved_run(tmp_path)
    original, meta = trial.read_bytes(), trial.with_suffix('.json').read_bytes()
    write_json = compact.atomic_json
    def fail(path, data):
        if path == trial.with_suffix('.json'):
            raise OSError('simulated write failure')
        return write_json(path, data)
    monkeypatch.setattr(compact, 'atomic_json', fail)
    with pytest.raises(OSError):
        compact.compact_run(folder, apply=True, progress=silent)
    assert trial.read_bytes() == original
    assert trial.with_suffix('.json').read_bytes() == meta


def test_compacted_run_still_loads_resume_and_target_graph(tmp_path):
    from test_full_taxel_resume import stopped_run, staged
    from phd.ui.calibration_graph_analysis import load_calibration_recording
    from phd.ui.calibration_analysis import available_taxel_indices
    folder, *_ = stopped_run(tmp_path)
    before, _ = staged(folder)
    compact.compact_run(folder, apply=True, progress=silent)
    after, _ = staged(folder)
    assert after.resume_plan['remaining'] == before.resume_plan['remaining']
    assert after.resume_plan['completed'] == before.resume_plan['completed']
    loaded = load_calibration_recording(folder / 'taxel_35' / 'trial_0001.json')
    dataset = loaded['datasets'][0]
    assert available_taxel_indices(dataset['rows'], dataset['signal_field']) == [35]

import pytest

from phd.ui.experiment_tasks import CalibrationTask, FullTaxelCalibrationTask


@pytest.mark.parametrize('shape,target', [((5, 5), 12), ((8, 10), 35), ((20, 20), 189)])
def test_full_taxel_recording_keeps_core_and_only_current_target(shape, target):
    task = FullTaxelCalibrationTask(output_directory=None)
    task.sensor_rows, task.sensor_columns = shape
    task.taxel_index = target
    row = dict(raw=100., diff=-3., diff_ave=-2., calibration=103., cell_index=target,
               elapsed_s=1., tool_z_m=.1, force_raw_g=30., phase='returning',
               force_baseline_g=2., force_delta_g=28., travel_down_mm=15.)
    for index in range(shape[0] * shape[1]):
        for field in ('raw', 'calibration', 'diff', 'diff_ave', 'diff_percent_ave', 'raw_ave'):
            row[f'taxel_{index:03d}_{field}'] = index + .123456789
    projected = task._profile_trimmed_taxel(row)
    assert projected == task._rows_for_storage([dict(row)])[0]
    assert all(projected[key] == value for key, value in row.items() if not key.startswith('taxel_'))
    assert len(projected) == 6 + sum(not key.startswith('taxel_') for key in row)
    assert projected[f'taxel_{target:03d}_raw_ave'] == row[f'taxel_{target:03d}_raw_ave']
    assert task._recorded_taxel_indices() == [target]
    task.taxel_index = target + 1
    assert f'taxel_{target + 1:03d}_raw' in task._profile_trimmed_taxel(row)
    assert f'taxel_{target:03d}_raw' not in task._profile_trimmed_taxel(row)


def test_proximity_reference_channels_remain_until_storage_projection():
    task = FullTaxelCalibrationTask(output_directory=None)
    task.protocol = 'proximity_profile'
    task.taxel_index = 35
    task._phase = 'profile_collect'
    task._profile_kind = 'reference'
    row = {'raw': 100., 'taxel_035_diff': -1., 'taxel_079_diff': -2.}
    assert task._profile_trimmed_taxel(row) is row
    assert 'taxel_079_diff' not in task._rows_for_storage([dict(row)])[0]
    normal = CalibrationTask(output_directory=None)
    assert normal._rows_for_storage([row]) == [row]
    assert normal._profile_trimmed_taxel(row) is row
    assert len(normal._recorded_taxel_indices()) == normal.sensor_rows * normal.sensor_columns


def test_full_profile_is_analysed_before_reducing_saved_csv(tmp_path, monkeypatch):
    import csv
    from pathlib import Path
    from phd.ui import experiment_tasks as tasks
    task = FullTaxelCalibrationTask(output_directory=tmp_path)
    task.taxel_index = 35
    task.protocol = 'proximity_profile'
    task._samples = [{'elapsed_s': 1., 'raw': 100., 'diff_ave': -1.,
                      'taxel_035_diff_ave': -1., 'taxel_079_diff_ave': -2.}]
    observed = []
    def analyse(rows, **kwargs):
        observed.append(rows[0]['taxel_079_diff_ave'])
        return {'common_mode': {'reference_count': 1}}
    monkeypatch.setattr(tasks, 'analyse_proximity_profile', analyse)
    monkeypatch.setattr(tasks, 'analyze_calibration_rows', lambda *a, **kw: {})
    monkeypatch.setattr(tasks, 'contact_referenced_rows', lambda rows, *a, **kw: rows)
    monkeypatch.setattr(tasks, 'save_profile_outputs', lambda *a, **kw: {})
    result = task._save_result()
    assert observed == [-2.]
    assert result['proximity_profile']['common_mode'] == {'reference_count': 1}
    with Path(result['csv_path']).open(newline='') as stream:
        header = next(csv.reader(stream))
    assert 'taxel_035_diff_ave' in header and 'taxel_079_diff_ave' not in header

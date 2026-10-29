import copy
import csv
import json
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QApplication

from phd.ui.calibration_graph_analysis import CalibrationGraphAnalysisDialog, load_calibration_recording
from phd.ui.calibration_comparison import comparison_entry, shared_axis_ranges
from phd.ui.calibration_recording_catalog import dataset_rows
from test_calibration_graph_analysis import _synthetic_rows, _write_csv


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def saved_grid(tmp_path, *, full_grid=False):
    folder = tmp_path / 'run'
    folder.mkdir()
    summary = dict(task='full_taxel_calibration', run_id='test', sensor_identity='sensor:test',
                   sensor_shape=[5, 5], signal_field='diff_ave', status='aborted')
    (folder / 'full_taxel_summary.json').write_text(json.dumps(summary))
    for target, number in ((6, 1), (6, 2), (12, 1), (12, 2), (18, 1)):
        directory = folder / f'taxel_{target:02d}'
        directory.mkdir(exist_ok=True)
        path = directory / f'trial_{number:04d}.csv'
        rows = _synthetic_rows(signal_offset=float(target * 10 + number))
        for row in rows:
            row.update(cell_index=target, raw=row['diff_ave']+1000, diff=row['diff_ave'])
            for node in range(25) if full_grid else (target,):
                row[f'taxel_{node:03d}_diff_ave'] = row['diff_ave'] if node == target else 2000. + node
        _write_csv(path, rows)
        meta = dict(sensor_shape=[5, 5], sensor_identity='sensor:test', taxel_index=target,
                    batch_id=f'test_taxel{target:02d}', trial_number=number, signal_field='diff_ave',
                    csv_path=str(path), sample_count=len(rows),
                    measurement_quality={'trial_completed': target != 18})
        path.with_suffix('.json').write_text(json.dumps(meta))
    return folder


def test_catalog_discovers_only_same_run_without_reading_sample_arrays(tmp_path, monkeypatch):
    folder = saved_grid(tmp_path)
    from phd.ui import calibration_recording_catalog as catalog
    monkeypatch.setattr(catalog, 'read_calibration_csv_rows', lambda *_: pytest.fail('Eager CSV load'))
    selected = folder / 'taxel_12/trial_0002.json'
    loaded = load_calibration_recording(selected, browse_full_run=True)
    assert len(loaded['datasets']) == 5
    assert not any(d['rows'] for d in loaded['datasets'])
    assert loaded['datasets'][loaded['initial_dataset_index']]['metadata']['trial_number'] == 2
    assert loaded['full_taxel_run'] == str(folder)


def test_actual_map_click_switches_physical_node_and_preserves_trial_number(app, tmp_path):
    folder = saved_grid(tmp_path)
    dialog = CalibrationGraphAnalysisDialog(load_calibration_recording(folder/'taxel_06/trial_0002.json', browse_full_run=True))
    try:
        assert dialog.taxel_map._available_taxels == {6, 12, 18}
        assert not dialog.taxel_map.buttons[0].isEnabled()
        ranges = copy.deepcopy(dialog.charts['signal_distance'].shared_axis_ranges)
        dialog.taxel_map.buttons[12].click()
        assert dialog.selected_taxel_index == 12
        current = dialog.datasets[dialog._current_index()]
        assert current['metadata']['taxel_index'] == 12
        assert current['metadata']['trial_number'] == 2
        assert 'taxel_12/trial_0002.csv' in dialog.status_label.text()
        assert dialog.charts['signal_distance'].rows[0]['diff_ave'] == 122.
        assert dialog.taxel_map.buttons[12].isChecked()
        assert dialog.taxel_map.buttons[6].isEnabled()
        assert all(not d['rows'] for d in dialog.datasets if d['metadata']['taxel_index'] != 12)
        assert dialog.charts['signal_distance'].shared_axis_ranges == ranges
        dialog.taxel_map.buttons[18].click()
        assert 'Partial / interrupted' in dialog.status_label.text()
        assert dialog.selected_taxel_index == 18
    finally:
        dialog.close()


def test_legacy_full_grid_uses_own_trial_and_does_not_offer_unmeasured_targets(app, tmp_path):
    folder = saved_grid(tmp_path, full_grid=True)
    dialog = CalibrationGraphAnalysisDialog(load_calibration_recording(folder/'full_taxel_summary.json', browse_full_run=True))
    try:
        assert dialog.taxel_map._available_taxels == {6, 12, 18}
        dialog.taxel_map.buttons[12].click()
        assert dialog.charts['signal_distance'].rows[0]['diff_ave'] == 121.
        assert len(dialog.projected_datasets) == 2
        # Direct dropdown change also selects the new physical target, without recursion.
        dialog.trial_selector.setCurrentIndex(0)
        assert dialog.selected_taxel_index == 6
        assert dialog.charts['signal_distance'].rows[0]['diff_ave'] == 61.
    finally:
        dialog.close()


def test_shared_bounds_stream_all_trials_without_retaining_them(tmp_path):
    folder = saved_grid(tmp_path)
    lazy = load_calibration_recording(folder/'full_taxel_summary.json', browse_full_run=True)['datasets']
    eager = [load_calibration_recording(d['source_path'])['datasets'][0] for d in lazy]
    assert shared_axis_ranges(lazy) == shared_axis_ranges(eager)
    assert all(not d['rows'] for d in lazy)
    entry = comparison_entry(lazy[2], 12, 2)
    assert entry['rows'][0]['diff_ave'] == 121.
    assert len([d for d in lazy if d['rows']]) == 1


def test_missing_trial_keeps_current_graph_and_reports_error(app, tmp_path):
    folder = saved_grid(tmp_path)
    dialog = CalibrationGraphAnalysisDialog(load_calibration_recording(folder/'full_taxel_summary.json', browse_full_run=True))
    try:
        previous = list(dialog.charts['signal_distance'].rows)
        (folder/'taxel_12/trial_0001.csv').unlink()
        dialog.taxel_map.buttons[12].click()
        assert dialog.selected_taxel_index == 6
        assert dialog.charts['signal_distance'].rows == previous
        assert 'Could not load Taxel 12' in dialog.status_label.text()
        assert dialog.taxel_map.buttons[6].isChecked()
    finally:
        dialog.close()


def test_catalog_excludes_mismatched_and_derived_files(tmp_path):
    folder = saved_grid(tmp_path)
    path = folder/'taxel_18/trial_0001.json'
    meta=json.loads(path.read_text()); meta['sensor_identity']='other'; path.write_text(json.dumps(meta))
    (folder/'taxel_12/trial_0001_proximity_profile.json').write_text('{}')
    loaded=load_calibration_recording(folder/'full_taxel_summary.json', browse_full_run=True)
    assert len(loaded['datasets']) == 4
    assert len(loaded['catalog_warnings']) == 1
    dataset=loaded['datasets'][0]
    with Path(dataset['source_path']).open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='changed on disk'):
        dataset_rows(dataset)


def test_preloaded_target_only_datasets_offer_union_of_map_nodes(app, tmp_path):
    folder = saved_grid(tmp_path)
    first=load_calibration_recording(folder/'taxel_06/trial_0001.json')['datasets'][0]
    second=load_calibration_recording(folder/'taxel_12/trial_0001.json')['datasets'][0]
    dialog=CalibrationGraphAnalysisDialog({'datasets':[first,second]})
    try:
        assert dialog.taxel_map._available_taxels == {6,12}
        dialog.taxel_map.buttons[12].click()
        assert dialog.selected_taxel_index == 12 and dialog._current_index() == 1
        assert dialog.taxel_map.buttons[6].isEnabled()
    finally:
        dialog.close()


def test_compare_button_can_pin_different_on_demand_nodes(app, tmp_path):
    folder = saved_grid(tmp_path)
    dialog = CalibrationGraphAnalysisDialog(load_calibration_recording(folder/'full_taxel_summary.json', browse_full_run=True))
    try:
        dialog.compare_button.click()
        comparison = dialog._comparison_dialog
        assert comparison.entries[0]['taxel_index'] == 6
        first = copy.deepcopy(comparison.entries[0]['rows'])
        dialog.taxel_map.buttons[12].click()
        dialog.compare_button.click()
        assert [entry['physical_target_taxel_index'] for entry in comparison.entries] == [6,12]
        assert comparison.entries[0]['rows'] == first
        assert comparison.entries[1]['rows'][0]['diff_ave'] == 121.
    finally:
        dialog.close()


def test_png_opens_corresponding_trial_and_relocated_folder_uses_local_csv(tmp_path):
    folder = saved_grid(tmp_path)
    graph = folder/'taxel_12/graphs/trial_0002.png'
    graph.parent.mkdir()
    graph.write_bytes(b'graph placeholder')
    moved = tmp_path/'moved'
    folder.rename(moved)
    loaded = load_calibration_recording(moved/'taxel_12/graphs/trial_0002.png', browse_full_run=True)
    current = loaded['datasets'][loaded['initial_dataset_index']]
    assert current['metadata']['taxel_index'] == 12 and current['metadata']['trial_number'] == 2
    assert dataset_rows(current)[0]['diff_ave'] == 122.
    assert str(moved) in current['source_path']


def test_map_does_not_mix_sensor_identities_but_dropdown_can_switch(app, tmp_path):
    folder = saved_grid(tmp_path)
    first=load_calibration_recording(folder/'taxel_06/trial_0001.json')['datasets'][0]
    second=load_calibration_recording(folder/'taxel_12/trial_0001.json')['datasets'][0]
    second['metadata']['sensor_identity']='sensor:other'
    dialog=CalibrationGraphAnalysisDialog({'datasets':[first,second]})
    try:
        assert dialog.taxel_map._available_taxels == {6}
        dialog.trial_selector.setCurrentIndex(1)
        assert dialog.selected_taxel_index == 12
        assert dialog.taxel_map._available_taxels == {12}
        assert dialog.charts['signal_distance'].rows[0]['diff_ave'] == 121.
    finally:
        dialog.close()

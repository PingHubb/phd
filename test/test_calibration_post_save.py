"""No hardware: saving must yield before a verified next motion can start."""
import json
from pathlib import Path

import numpy as np
import pytest

from phd.ui.experiment_tasks import CalibrationTask, FullTaxelCalibrationTask
from test_experiment_tasks import _Host, _RobotApi, _saved_full_calibration_map


def completed_trial(monkeypatch, tmp_path, *, full=True):
    clock = [100.0]
    monkeypatch.setattr('phd.ui.experiment_tasks.time.monotonic', lambda: clock[0])
    cls = FullTaxelCalibrationTask if full else CalibrationTask
    task = cls(storage_path=None, output_directory=tmp_path)
    api = _RobotApi()
    host = _Host(api, task)
    task.post_save_feedback_stable_sec = 0.25  # Exercise production readiness.
    task._test_active = True
    task._phase = 'recovery'
    task._stop_reason = 'contact_threshold'
    task._trial_number = 1
    task._batch_total = 1 if full else 2
    task._batch_directory = tmp_path / 'taxel_35'
    task._batch_id = 'test_taxel35'
    if full:
        assert task.load_position_map(_saved_full_calibration_map())[0] is False  # Cannot replace a live map.
        task._test_active = False
        assert task.load_position_map(_saved_full_calibration_map())[0]
        task._test_active = True
        task._targets = task.position_map['positions']
        task._target_cursor = 0
        task._current_target = task._targets[0]
        task.taxel_index = task._current_target['taxel_index']
        task._full_directory = tmp_path
        task._full_run_id = 'test'
    else:
        task.repeat_count = 2
    result = {
        'taxel_index': task.taxel_index, 'trial_number': 1,
        'stop_reason': 'contact_threshold', 'rows': [],
        'measurement_quality': {'trial_completed': True},
        'summary': {'trial_number': 1, 'stop_reason': 'contact_threshold'},
    }
    def slow_save(_host):
        clock[0] += 2.5  # Simulate GUI-thread analysis while delivery is queued.
        task._batch_results.append(result)
        task._trial_result_saved = True
        return result
    monkeypatch.setattr(task, '_save_current_trial', slow_save)
    data = host.ui_ros.sensor_functions._data
    data.frame_monotonic_sec = clock[0]
    api.current_feedback_state_time = clock[0]
    host.ui_ros._force_meter_last_sample_monotonic = clock[0]
    task._complete_returned_trial(host, clock[0], 'next_trial')
    assert task._phase == 'waiting_post_save_feedback'
    assert not host.stops and not api.sent
    return task, host, api, clock


def fresh(task, host, api, clock, *, missing=None, increment=0.1):
    clock[0] += increment
    if missing != 'tactile sensor':
        data = host.ui_ros.sensor_functions._data
        data.frame_monotonic_sec = clock[0]
        data.frame_sequence += 1
    if missing != 'force transmitter':
        host.ui_ros._force_meter_last_sample_monotonic = clock[0]
    if missing != 'robot position':
        api.current_feedback_state_time = clock[0]
    task.on_tick(host)


@pytest.mark.parametrize('full', [False, True])
def test_slow_save_yields_and_requires_new_stable_feedback_before_continuing(monkeypatch, tmp_path, full):
    task, host, api, clock = completed_trial(monkeypatch, tmp_path, full=full)
    task.on_tick(host)  # The old 2.5-second-old snapshot must not abort or move.
    assert not host.stops and not api.sent
    for _ in range(3):
        fresh(task, host, api, clock)
        assert not api.sent and task._phase == 'waiting_post_save_feedback'
    fresh(task, host, api, clock)
    assert not host.stops
    assert task._post_save_started_at is None
    if full:
        assert task._phase == 'full_moving' and len(api.sent) == 1
        assert task._completed_taxels == [35]
    else:
        assert task._phase == 'recalibrating'
        assert host.ui_ros.sensor_functions.update_cal_calls == 1


def test_repeated_cached_sample_does_not_satisfy_readiness(monkeypatch, tmp_path):
    task, host, api, clock = completed_trial(monkeypatch, tmp_path)
    fresh(task, host, api, clock)
    for _ in range(5):
        clock[0] += 0.1
        task.on_tick(host)
    assert task._post_save_frame_counts == [1, 1, 1]
    assert not api.sent and not host.stops
    fresh(task, host, api, clock)
    fresh(task, host, api, clock)
    assert task._phase == 'full_moving'


@pytest.mark.parametrize('missing', ['tactile sensor', 'force transmitter', 'robot position'])
def test_real_feedback_failure_times_out_without_motion_and_records_reason(monkeypatch, tmp_path, missing):
    task, host, api, clock = completed_trial(monkeypatch, tmp_path)
    fresh(task, host, api, clock, missing=missing)
    assert not host.stops
    fresh(task, host, api, clock, missing=missing, increment=10.0)
    assert task._stop_reason == 'post_save_feedback_timeout'
    assert missing in task._stop_message and not api.sent
    assert not task._test_active
    metadata = json.loads(Path(task.last_result['metadata_path']).read_text())
    assert metadata['stop_reason'] == task._stop_reason
    assert metadata['stop_message'] == task._stop_message
    assert metadata['stop_phase'] == 'waiting_post_save_feedback'
    assert metadata['stopped_at'] and metadata['completed_taxels'] == [35]
    assert metadata['last_taxel_index'] == 35
    assert task._batch_results[0]['measurement_quality']['trial_completed']


def test_manual_stop_cancels_continuation_permanently(monkeypatch, tmp_path):
    task, host, api, clock = completed_trial(monkeypatch, tmp_path)
    fresh(task, host, api, clock)
    host._stop_sidebar_control(show_result=False, message='Manual stop')
    for _ in range(6):
        fresh(task, host, api, clock)
    assert not api.sent and task._post_save_started_at is None
    assert task._stop_reason == 'manual_stop'
    assert task.last_result['stop_reason'] == 'manual_stop'


@pytest.mark.parametrize('raw_g,baseline_g', [(900, 0), (0, 900)])
def test_force_guard_is_not_disabled_during_wait(monkeypatch, tmp_path, raw_g, baseline_g):
    task, host, api, clock = completed_trial(monkeypatch, tmp_path)
    host.ui_ros._force_meter_last_raw_newtons = raw_g * task.NEWTONS_PER_GRAM
    task._force_baseline_n = baseline_g * task.NEWTONS_PER_GRAM
    task.on_tick(host)
    assert task._stop_reason == 'force_safety_limit' and not api.sent


def test_bad_matrix_or_dead_force_reader_never_releases_next_move(monkeypatch, tmp_path):
    task, host, api, clock = completed_trial(monkeypatch, tmp_path)
    host.ui_ros.sensor_functions._data.diffDataAve[0, 0] = np.nan
    for _ in range(5):
        fresh(task, host, api, clock)
    assert not api.sent and not host.stops
    host.ui_ros.sensor_functions._data.diffDataAve[0, 0] = 0
    host.ui_ros._force_meter_thread.running = False
    for _ in range(5):
        fresh(task, host, api, clock)
    assert not api.sent and not host.stops


def test_motion_phase_stale_sensor_still_stops_immediately(monkeypatch, tmp_path):
    task, host, api, clock = completed_trial(monkeypatch, tmp_path)
    task._clear_post_save_wait()
    task._phase = 'approaching'
    task.on_tick(host)
    assert task._stop_reason == 'sensor_stale' and not api.sent


def test_save_failure_never_queues_continuation(monkeypatch, tmp_path):
    task, host, api, clock = completed_trial(monkeypatch, tmp_path)
    task._clear_post_save_wait()
    task._phase = 'recovery'
    def failed_save(_host):
        raise OSError('disk full')
    monkeypatch.setattr(task, '_save_current_trial', failed_save)
    task._complete_returned_trial(host, clock[0], 'next_trial')
    assert task._stop_reason == 'save_failed' and 'disk full' in task._stop_message
    assert task._post_save_started_at is None and not api.sent


def test_full_summary_preserves_reason_even_before_first_saved_taxel(tmp_path):
    task = FullTaxelCalibrationTask(output_directory=tmp_path)
    host = _Host(_RobotApi(), task)
    task._full_directory = tmp_path
    task._test_active = True
    task._phase = 'full_moving'
    task._request_stop(host, 'robot_feedback_lost', 'Robot feedback was lost before the first trial.')
    metadata = json.loads((tmp_path / 'full_taxel_summary.json').read_text())
    assert metadata['status'] == 'aborted' and metadata['completed_taxels'] == []
    assert metadata['stop_reason'] == 'robot_feedback_lost'
    assert metadata['stop_phase'] == 'full_moving'


def test_save_reuses_analysis_without_changing_contact_distance(monkeypatch):
    from phd.ui import experiment_tasks as module
    task = CalibrationTask(storage_path=None, output_directory=None)
    task._samples = [{'travel_down_mm': 8.0, 'elapsed_s': 0, 'phase': 'baseline'}]
    calls = []
    def analyse(rows, field):
        calls.append((rows, field))
        return {'contact_onset': {'distance_mm': 6.0}}
    monkeypatch.setattr(module, 'analyze_calibration_rows', analyse)
    result = task._save_result()
    assert len(calls) == 1
    assert result['rows'][0]['contact_relative_travel_mm'] == 2.0
    # The derived column is added in place: a second copy of every frame doubles RAM.
    assert result['rows'][0] is task._samples[0]


def test_queued_qt_feedback_can_run_before_next_route(monkeypatch, tmp_path):
    from PyQt5.QtCore import QCoreApplication, QTimer
    app = QCoreApplication.instance() or QCoreApplication([])
    task, host, api, clock = completed_trial(monkeypatch, tmp_path)
    delivered = []
    def deliver_frame():
        delivered.append(True)
        fresh(task, host, api, clock)
    # Data may already be queued while the save callback is still unwinding.
    QTimer.singleShot(0, deliver_frame)
    task.on_tick(host)
    assert not delivered and not host.stops and not api.sent
    app.processEvents()
    assert delivered and not api.sent
    for _ in range(3):
        QTimer.singleShot(0, deliver_frame)
        app.processEvents()
    assert len(delivered) == 4 and len(api.sent) == 1


def test_freshness_guard_is_checked_again_immediately_before_dispatch(monkeypatch, tmp_path):
    task, host, api, clock = completed_trial(monkeypatch, tmp_path)
    calls = []
    def block_route(_host, _now):
        calls.append(True)
        task._request_stop(host, 'robot_feedback_stale', 'Robot feedback stale before dispatch.')
        return False
    monkeypatch.setattr(task, '_measurement_guard', block_route)
    for _ in range(4):
        fresh(task, host, api, clock)
    assert calls == [True] and not api.sent
    assert task._stop_reason == 'robot_feedback_stale'


def test_analysis_reuse_matches_previous_two_pass_results():
    from phd.ui.calibration_analysis import analyze_calibration_rows, contact_referenced_rows
    rows = [
        {'elapsed_s': i / 20, 'phase': 'approaching', 'travel_down_mm': i / 10,
         'diff_ave': -max(i - 10, 0) * 2, 'force_delta_abs_g': max(i - 40, 0) * 3}
        for i in range(70)
    ]
    old_rows = contact_referenced_rows(rows)
    old_analysis = analyze_calibration_rows(old_rows)
    new_analysis = analyze_calibration_rows(rows)
    assert new_analysis == old_analysis
    assert contact_referenced_rows(rows, analysis=new_analysis) == old_rows

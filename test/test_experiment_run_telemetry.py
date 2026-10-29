"""Read-only force baseline presentation and long-duration run monitoring."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from phd.ui.experiment_run_view import (
    NEWTONS_PER_GRAM,
    ended_run_state,
    force_baseline_readout,
    format_elapsed,
)


@pytest.mark.parametrize("fields,expected", [
    ({"_stop_reason": "complete"}, "active"),
    ({"_stop_reason": "contact_threshold", "_batch_completed": True}, "active"),
    ({"_stop_reason": "contact_threshold", "_batch_completed": False}, "warning"),
    ({"id": "full_taxel_calibration", "_stop_reason": "contact_threshold",
      "_batch_completed": True, "_full_completed": False}, "warning"),
    ({"id": "full_taxel_calibration", "_stop_reason": "contact_threshold",
      "_full_completed": True}, "active"),
    ({"_stop_reason": "profile_complete", "_batch_completed": True}, "active"),
    ({"_stop_reason": "manual_stop"}, "warning"),
    ({"_stop_reason": "partial_complete"}, "warning"),
    ({"_stop_reason": "sensor_stale", "_batch_completed": True}, "error"),
    ({"last_result": {"stop_reason": "checkpoint_failed"}}, "error"),
    ({"last_result": []}, "warning"),
])
def test_latest_status_colour_does_not_hide_faults_or_infer_completion(fields, expected):
    task = SimpleNamespace(**fields)
    before = vars(task).copy()
    assert ended_run_state(task) == expected
    assert vars(task) == before


def force_ui(grams=12.5, stamp=10):
    return SimpleNamespace(
        _force_meter_last_raw_newtons=grams * NEWTONS_PER_GRAM,
        _force_meter_last_sample_monotonic=stamp,
    )


@pytest.mark.parametrize("task_id", ["calibration", "full_taxel_calibration", "taxel_position_mapping"])
@pytest.mark.parametrize("raw,baseline,expected", [
    (12.5, 2.5, "+10.0"), (2.5, 12.5, "-10.0"), (2.5, -2.5, "+5.0"), (2.5, 0, "+2.5"),
])
def test_baseline_subtraction_preserves_sign_and_uses_live_task_reference(task_id, raw, baseline, expected):
    task = SimpleNamespace(id=task_id, _phase="approaching", _force_baseline_n=baseline * NEWTONS_PER_GRAM)
    ui = force_ui(raw)
    before = vars(task).copy(), vars(ui).copy()
    assert force_baseline_readout(ui, task, 10) == (
        f"Corrected: {expected} g", f"Baseline: {baseline:+.1f} g",
    )
    assert before == (vars(task), vars(ui))  # No tare, baseline or command mutation.
    task._force_baseline_n = raw * NEWTONS_PER_GRAM
    assert force_baseline_readout(ui, task, 10)[0] == "Corrected: +0.0 g"


@pytest.mark.parametrize("baseline", [None, float("nan"), float("inf")])
def test_missing_baseline_is_not_assumed_zero(baseline):
    task = SimpleNamespace(_force_baseline_n=baseline)
    assert force_baseline_readout(force_ui(), task, 10) == (
        "Corrected: —", "Baseline: waiting for measurement",
    )


@pytest.mark.parametrize("phase", ["idle", "connecting_force_meter", "force_baseline", "returning"])
def test_new_trial_start_does_not_show_old_reference(phase):
    task = SimpleNamespace(_phase=phase, _return_purpose="start", _force_baseline_n=0.01)
    assert "waiting for measurement" in force_baseline_readout(force_ui(), task, 10)[1]


@pytest.mark.parametrize("value,stamp", [(None, 10), (float("nan"), 10), (1, None), (1, 11)])
def test_invalid_sample_hides_corrected_force_but_keeps_valid_baseline(value, stamp):
    ui = SimpleNamespace(_force_meter_last_raw_newtons=value, _force_meter_last_sample_monotonic=stamp)
    task = SimpleNamespace(_force_baseline_n=2.5 * NEWTONS_PER_GRAM)
    assert force_baseline_readout(ui, task, 10) == (
        "Corrected: — (no valid sample)", "Baseline: +2.5 g",
    )


def test_staleness_matches_calibration_and_mapping_limits():
    task = SimpleNamespace(_force_baseline_n=0, force_sample_stale_sec=2)
    assert force_baseline_readout(force_ui(), task, 12)[0] == "Corrected: +12.5 g"
    assert force_baseline_readout(force_ui(), task, 12.01)[0] == "Corrected: — (stale sample)"
    mapping = SimpleNamespace(_force_baseline_n=0, force_stale_sec=0.2)
    assert force_baseline_readout(force_ui(), mapping, 10.5)[0] == "Corrected: — (stale sample)"


def test_tasks_without_baseline_do_not_inherit_previous_task_values():
    assert force_baseline_readout(force_ui(), SimpleNamespace(), 10) == (
        "Corrected: —", "Baseline: not used by this task",
    )


def test_full_taxel_routing_reference_is_distinct_from_trial_reference_even_when_paused():
    task = SimpleNamespace(id="full_taxel_calibration", _phase="full_moving",
                           _force_baseline_n=2.5 * NEWTONS_PER_GRAM,
                           _routing_force_baseline_n=10 * NEWTONS_PER_GRAM)
    assert force_baseline_readout(force_ui(), task, 10) == (
        "Corrected: +2.5 g", "Route baseline: +10.0 g",
    )
    task._phase = "paused_force_meter"
    task._force_paused_phase = "full_moving"
    assert force_baseline_readout(force_ui(), task, 10)[1] == "Route baseline: +10.0 g"
    task._phase = "approaching"
    assert force_baseline_readout(force_ui(), task, 10)[1] == "Baseline: +2.5 g"
    task._phase = "full_waiting_force_route"
    assert force_baseline_readout(force_ui(), task, 10)[1] == "Route baseline: waiting for measurement"


@pytest.mark.parametrize("seconds,expected", [
    (0, "0 h 00 min 00 s"), (59.9, "0 h 00 min 59 s"), (60, "0 h 01 min 00 s"),
    (3599, "0 h 59 min 59 s"), (3600, "1 h 00 min 00 s"), (3661, "1 h 01 min 01 s"),
    (90061, "25 h 01 min 01 s"), (360000, "100 h 00 min 00 s"), (-2, "0 h 00 min 00 s"),
    (None, "—"), (float("nan"), "—"), (float("inf"), "—"),
])
def test_elapsed_uses_explicit_hours_minutes_seconds_without_daily_rollover(seconds, expected):
    assert format_elapsed(seconds) == expected


def test_run_widget_updates_freezes_and_clears_baseline_without_hardware():
    script = '''
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtCore import Qt
from phd.ui import theme
from phd.ui.experiment_run_view import ExperimentRunView, NEWTONS_PER_GRAM as G
app = QApplication([])
now = [10.0]
task = SimpleNamespace(id="calibration", label="Calibration", _phase="approaching", _force_baseline_n=2.5*G)
ui = SimpleNamespace(_force_meter_last_raw_newtons=12.5*G, _force_meter_last_sample_monotonic=10.)
host = SimpleNamespace(_sidebar_active_task=task, ui_ros=ui, info_process=QLabel("Running"))
view = ExperimentRunView(clock=lambda: now[0])
view.refresh(host)
assert theme.SUCCESS in view.message.styleSheet()
assert 'font-size: 16px' in view.message.styleSheet()
assert 'font-weight: 600' in view.message.styleSheet()
assert view.message.wordWrap() and view.message.textFormat() == Qt.PlainText
assert view.message.textInteractionFlags() & Qt.TextSelectableByMouse
task._phase = 'paused_force_meter'
view.refresh(host)
assert theme.WARNING in view.message.styleSheet()
task._phase = 'approaching'
view.refresh(host)
assert theme.SUCCESS in view.message.styleSheet()
assert view.force.text() == "12.5 g"
assert view.force_corrected.text() == "Corrected: +10.0 g"
assert view.force_baseline.text() == "Baseline: +2.5 g"
task._force_baseline_n = 5*G
now[0] += 3661
ui._force_meter_last_sample_monotonic = now[0]
view.refresh(host)
assert view.elapsed.text() == "1 h 01 min 01 s"
assert view.force_corrected.text() == "Corrected: +7.5 g"
task._stop_reason = 'sensor_stale'
view.finish(host)
assert theme.DANGER in view.message.styleSheet()
host._sidebar_active_task = None
task._force_baseline_n = None
ui._force_meter_last_raw_newtons = None
now[0] += 1000
view.refresh(host)
assert view.force_corrected.text() == "Corrected: +7.5 g"
assert view.force_baseline.text() == "Baseline: +5.0 g"
assert view.elapsed.text() == "1 h 01 min 01 s"
assert "not live" in view.force_age.text()
host._sidebar_active_task = task
task._stop_reason = ''
task._phase = "force_baseline"
view.refresh(host)
assert theme.SUCCESS in view.message.styleSheet()
assert view.elapsed.text() == "0 h 00 min 00 s"
assert view.force_corrected.text() == "Corrected: —"
assert "waiting" in view.force_baseline.text()
view.close()
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

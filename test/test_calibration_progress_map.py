"""Per-pass Run map correctness, geometry and read-only behavior without hardware."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from phd.ui.calibration_progress_map import calibration_progress_snapshot
from test_research_workspace import run_qt


def result(number, completed=True):
    return {"trial_number": number, "measurement_quality": {"trial_completed": completed}}


def task(rows=5, columns=5):
    indices = [i for i in range(rows * columns) if i % rows not in (0, rows - 1)
               and i // rows not in (0, columns - 1)]
    targets = [{"taxel_index": i, "status": "measured"} for i in indices]
    return SimpleNamespace(
        id="full_taxel_calibration", label="Full calibration", _phase="approaching",
        position_map={"settings": {"sensor_rows": rows, "sensor_columns": columns}, "positions": targets},
        _targets=deepcopy(targets), _current_target=targets[1], _target_cursor=1,
        _pass_number=2, repeat_count=3, active_sensor_label="Testing Sensor",
        _trials_by_taxel={indices[0]: [result(1), result(2)], indices[1]: [result(1), result(2, False)]},
        _completed_taxels=[indices[0], indices[1]],
    )


@pytest.mark.parametrize("shape", [(5, 5), (8, 10), (20, 20)])
def test_only_completed_current_pass_trials_count_and_shape_matches(shape):
    active = task(*shape)
    before = deepcopy(vars(active))
    view = calibration_progress_snapshot(active)
    indices = sorted(view.planned)
    assert (view.rows, view.columns) == shape
    assert len(view.planned) == (shape[0] - 2) * (shape[1] - 2)
    assert view.completed == {indices[0]}
    assert view.current == indices[1]
    assert (view.number, view.total) == (2, 3)
    assert 0 not in view.planned
    assert vars(active) == before
    active._pass_number = 3
    active._target_cursor = -1  # Between passes, previous target is still cached.
    following = calibration_progress_snapshot(active)
    assert not following.completed and following.current is None
    assert view.completed == {indices[0]}  # Immutable snapshot survives task mutation.


def test_connecting_does_not_show_old_run_targets_or_results():
    active = task()
    active._phase = "connecting_force_meter"
    active.position_map["positions"] = [{"taxel_index": 12}]
    view = calibration_progress_snapshot(active)
    assert view.planned == {12} and not view.completed and view.current is None
    assert view.number == 1


def test_single_taxel_repeat_and_proximity_profile_use_same_readonly_progress():
    active = task()
    active.id = "calibration"
    active.protocol = "proximity_profile"
    active.taxel_index = 12
    active._trial_number = 2
    active._batch_results = [result(1), result(2, False)]
    view = calibration_progress_snapshot(active)
    assert view.planned == {12} and view.current == 12 and view.unit == "Trial"
    assert not view.completed
    active._batch_results[-1] = result(2)
    assert calibration_progress_snapshot(active).completed == {12}
    active._phase = "paused_force_meter"
    assert calibration_progress_snapshot(active).paused


@pytest.mark.parametrize("kind", ["taxel_position_mapping", "mapped_taxel_move", "graph_analysis"])
def test_other_tasks_hide_calibration_map(kind):
    active = task()
    active.id = kind
    assert calibration_progress_snapshot(active) is None


def test_incomplete_or_bad_metadata_does_not_break_run_monitor():
    assert calibration_progress_snapshot(SimpleNamespace(id="calibration")) is None
    active = task()
    active.position_map["settings"]["sensor_rows"] = -1
    assert calibration_progress_snapshot(active) is None


def test_run_map_freezes_on_stop_resets_on_next_run_and_cannot_control_robot():
    run_qt('''
from types import SimpleNamespace
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QLabel, QAbstractButton
from phd.ui.experiment_run_view import ExperimentRunView
app = QApplication([])
targets = [{'taxel_index': i} for i in (6, 7, 11, 12)]
task = SimpleNamespace(id='full_taxel_calibration', label='Full calibration', _phase='approaching',
    position_map={'settings': {'sensor_rows': 5, 'sensor_columns': 5}, 'positions': targets},
    _targets=targets, _current_target=targets[1], _target_cursor=1, _pass_number=1, repeat_count=2,
    _trials_by_taxel={6: [{'trial_number': 1, 'measurement_quality': {'trial_completed': True}}]})
host = SimpleNamespace(_sidebar_active_task=task, ui_ros=None, info_process=QLabel('Approaching'))
view = ExperimentRunView(clock=lambda: 10)
view.show()
view.refresh(host)
app.processEvents()
grid = view.taxel_progress
assert not grid.isHidden()
assert grid.cells[6].property('progressState') == 'completed'
assert grid.cells[7].property('progressState') == 'current'
assert grid.cells[11].property('progressState') == 'waiting'
assert grid.cells[0].property('progressState') == 'excluded'
assert 'row 2, column 1' in grid.cells[7].toolTip()
assert not grid.findChildren(QAbstractButton)
assert not hasattr(grid, 'taxelSelected')
QTest.mouseClick(grid.cells[11], Qt.LeftButton)
assert task._current_target == targets[1] and grid.snapshot.current == 7
old_cell = grid.cells[7]
view.refresh(host)
assert grid.cells[7] is old_cell
task._phase = 'paused_force_meter'
view.refresh(host)
assert grid.cells[7].property('progressState') == 'paused'
view.finish(host, message='Manual stop')
assert grid.cells[7].property('progressState') == 'stopped'
assert 'Last run snapshot' in grid.summary.text()
frozen = grid.snapshot
task._trials_by_taxel.clear()
task._current_target = None
host._sidebar_active_task = None
view.update_end_details(host)
view.refresh(host)
assert grid.snapshot == frozen and grid.cells[6].property('progressState') == 'completed'
host._sidebar_active_task = task
task._phase = 'connecting_force_meter'
view.refresh(host)
assert not grid.frozen and not grid.snapshot.completed and grid.snapshot.current is None
assert grid.cells[6].property('progressState') == 'waiting'
host._sidebar_active_task = SimpleNamespace(id='graph_analysis')
view.refresh(host)
assert grid.isHidden()
view.close()
''')


def test_map_fits_small_window_scrolls_tall_grid_and_preserves_stop_buttons():
    run_qt('''
import os, tempfile
from types import SimpleNamespace
config = tempfile.TemporaryDirectory()
os.environ['XDG_CONFIG_HOME'] = config.name
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from phd.ui.ui_initial import MyMainWindow
app = QApplication([])
QTimer.singleShot = lambda *a, **k: None
window = MyMainWindow()
window.open_experiments_window()
window.experiments_window.resize(860, 620)
for rows, cols in ((5, 5), (20, 20), (5, 5)):
    target = {'taxel_index': rows + 1}
    task = SimpleNamespace(id='full_taxel_calibration', label='Full calibration', _phase='approaching',
        position_map={'settings': {'sensor_rows': rows, 'sensor_columns': cols}, 'positions': [target]},
        _targets=[target], _current_target=target, _target_cursor=0, _pass_number=1, repeat_count=1)
    window._sidebar_active_task = task
    window._refresh_experiment_status()
    for _ in range(5): app.processEvents()
    grid = window.experiment_run_view.taxel_progress
    assert len(grid.cells) == rows * cols
    assert grid.grid.rowCount() == rows and grid.grid.columnCount() == cols
    assert window.experiments_window.width() == 860
    assert window.sidebar_btn_stop.isVisible() and window.experiment_stop_all_button.isVisible()
    scroll = window.experiment_view_tabs.widget(1)
    assert scroll.verticalScrollBar().maximum() > 0
    assert grid.scroll.verticalScrollBar().maximum() == 0
window._sidebar_active_task = None
window.close()
config.cleanup()
''')

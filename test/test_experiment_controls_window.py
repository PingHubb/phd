"""Experiment controls stay reusable when their modeless window is closed."""

import os
from pathlib import Path
import subprocess
import sys


def test_experiment_controls_window_hides_without_destroying_panel():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import os
import tempfile

config_dir = tempfile.TemporaryDirectory()
os.environ["XDG_CONFIG_HOME"] = config_dir.name

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QFrame
from phd.ui.ui_initial import ExperimentControlsWindow, MyMainWindow

app = QApplication([])
panel = QFrame()
window = ExperimentControlsWindow(panel)

window.show()
app.processEvents()
assert window.isVisible()
assert panel.parent() is window

window.close()
app.processEvents()
assert not window.isVisible()
assert panel.parent() is window

window.show()
app.processEvents()
assert window.isVisible()
window.reject()
app.processEvents()
assert not window.isVisible()
assert panel.parent() is window

window.deleteLater()
app.processEvents()

original_single_shot = QTimer.singleShot
QTimer.singleShot = lambda *_args, **_kwargs: None
try:
    main_window = MyMainWindow()
    assert main_window.h_splitter.count() == 1
    assert main_window.taxel_mapping_scan_strength_spin.value() == 2.0
    assert "default 2×" in main_window.taxel_mapping_scan_strength_spin.toolTip()
    assert main_window.taxel_mapping_scan_extra_descent_spin.maximum() == 20.0
    assert main_window.taxel_mapping_scan_extra_descent_spin.value() == 20.0
    assert "5 g" in main_window.taxel_mapping_noncontact_note.text()
    assert "cannot certify" in main_window.taxel_mapping_noncontact_note.text()
    assert main_window.sidebar.parent() is main_window.experiments_window
    main_window.sidebar_btn_start.setEnabled(False)
    assert main_window.select_sidebar_task('graph_analysis')
    main_window._on_sidebar_task_changed()
    assert main_window.sidebar_btn_start.text() == 'Select Graph'
    assert main_window.sidebar_btn_start.isEnabled()
    opened = []
    main_window._open_calibration_graph_analysis = lambda: opened.append(True)
    main_window.sidebar_btn_start.click()
    app.processEvents()
    assert opened == [True]
    assert main_window._sidebar_active_task is None
    assert main_window.select_sidebar_task('sensor_characterisation')
    workbench_opened = []
    main_window._open_sensor_characterisation = lambda: workbench_opened.append(True)
    preserved_task = object()
    main_window._sidebar_active_task = preserved_task
    main_window._start_sidebar_control()
    assert workbench_opened == [True]
    assert main_window._sidebar_active_task is preserved_task
    main_window._sidebar_active_task = None
    main_window.open_experiments_window()
    app.processEvents()
    assert main_window.experiments_window.isVisible()
    main_window.experiments_window.close()
    app.processEvents()
    assert not main_window.experiments_window.isVisible()
    assert main_window.sidebar.parent() is main_window.experiments_window
    tree = main_window.sidebar_task_list
    assert [tree.topLevelItem(index).text(0) for index in range(tree.topLevelItemCount())] == [
        'TRO2026',
        'Archive',
    ]
    assert tree.topLevelItem(0).isExpanded()
    assert not tree.topLevelItem(1).isExpanded()
    assert [
        tree.topLevelItem(0).child(index).data(0, Qt.UserRole)
        for index in range(tree.topLevelItem(0).childCount())
    ] == ['taxel_position_mapping', 'calibration', 'graph_analysis', 'sensor_characterisation']
    assert tree.topLevelItem(1).child(0).data(0, Qt.UserRole) == 'sigraph2026_scanning'
    assert main_window.select_sidebar_task('calibration')
    tabs = main_window.calibration_mode_tabs
    assert [tabs.tabText(index) for index in range(tabs.count())] == [
        'Full taxel',
        'Single taxel',
    ]
    assert tabs.currentIndex() == 0
    assert main_window.sidebar_btn_start.text() == 'Run Full Taxel Calibration'
    tabs.setCurrentIndex(0)
    app.processEvents()
    assert main_window._current_sidebar_control_id() == 'full_taxel_calibration'
    assert main_window.sidebar_btn_start.text() == 'Run Full Taxel Calibration'
    assert not hasattr(main_window, 'full_taxel_remember_position_button')
    assert 'safe anchor' in main_window.full_taxel_initial_position_label.text()
    splitter = main_window.experiment_controls_splitter
    assert splitter.orientation() == Qt.Horizontal
    assert main_window.sidebar_task_list.parent() is splitter.widget(0)
    assert main_window.sidebar_btn_start.parent().parent() is splitter.widget(1)
    assert main_window.experiments_window.minimumWidth() >= 860
    assert main_window.experiments_window.minimumHeight() >= 620
    assert [
        main_window.full_taxel_calibration_signal_combo.itemData(index)
        for index in range(main_window.full_taxel_calibration_signal_combo.count())
    ] == ['diff_ave', 'diff_percent_ave', 'raw_ave']
    main_window.close()
finally:
    QTimer.singleShot = original_single_shot

config_dir.cleanup()
"""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(project_root), env.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_file_dialogs_open_over_experiment_controls_and_keep_it_in_front():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import os
import tempfile

config_dir = tempfile.TemporaryDirectory()
os.environ['XDG_CONFIG_HOME'] = config_dir.name

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QFileDialog
from phd.ui.ui_initial import MyMainWindow

app = QApplication([])
original_single_shot = QTimer.singleShot
original_file_picker = QFileDialog.getOpenFileName
QTimer.singleShot = lambda *_args, **_kwargs: None
parents = []
QFileDialog.getOpenFileName = lambda parent, *_args, **_kwargs: (parents.append(parent), ('', ''))[1]
window = MyMainWindow()
try:
    controls = window.experiments_window
    controls.hide()
    window._choose_experiment_file('t', '', '*')
    assert parents[-1] is window
    controls.show()
    raised = []
    controls.raise_ = lambda: raised.append('raise')
    window._load_single_calibration_map()
    assert parents[-1] is controls
    assert raised == ['raise']
finally:
    window.close()
    QTimer.singleShot = original_single_shot
    QFileDialog.getOpenFileName = original_file_picker
    config_dir.cleanup()
"""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(project_root), env.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=project_root, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mapped_taxel_grid_routes_clicks_through_monitored_experiment_lifecycle():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

config_dir = tempfile.TemporaryDirectory()
os.environ['XDG_CONFIG_HOME'] = config_dir.name

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QFileDialog
from phd.ui.ui_initial import MyMainWindow

class Robot:
    @property
    def current_feedback_state_time(self):
        return time.monotonic()
    def __init__(self):
        self.position = [0.1, 0.2, 0.29]
        self.quaternion = [1.0, 0.0, 0.0, 0.0]
        self.sent = []
    def get_current_tool_position(self):
        return self.position, self.quaternion
    def send_positions_tool_position(self, position, quaternion, **kwargs):
        self.position = list(position)
        self.quaternion = list(quaternion)
        self.sent.append((list(position), kwargs))
        return True
    def exit_end_effector_velocity_mode(self, send_zero=True):
        return True

record = {
    'task': 'taxel_position_mapping',
    'completed': False,
    'settings': {'sensor_rows': 8, 'sensor_columns': 10, 'max_xy_offset_mm': 60},
    'anchor_pose': {
        'taxel_index': 35, 'tool_position_m': [0.1, 0.2, 0.3],
        'tool_quaternion_wxyz': [1., 0., 0., 0.],
    },
    'positions': [{
        'taxel_index': 43, 'row': 3, 'column': 5, 'status': 'measured',
        'tool_position_m': [0.104, 0.2, 0.292],
        'safe_approach_position_m': [0.104, 0.2, 0.3],
        'tool_quaternion_wxyz': [1., 0., 0., 0.],
    }],
}
app = QApplication([])
original_single_shot = QTimer.singleShot
original_file_picker = QFileDialog.getOpenFileName
QTimer.singleShot = lambda *_args, **_kwargs: None
window = MyMainWindow()
try:
    assert window.select_sidebar_task('taxel_position_mapping')
    grid = window.taxel_mapping_map
    assert len(grid.buttons) == 80
    assert not any(button.isEnabled() for button in grid.buttons.values())
    assert grid.layout().itemAtPosition(3, 5).widget() is grid.buttons[43]

    api = Robot()
    data = SimpleNamespace(frame_sequence=1, diffDataAve=np.zeros((8, 10)))
    window.ui_ros = SimpleNamespace(
        robot_api=api,
        sensor_functions=SimpleNamespace(is_connected=True, n_row=8, n_col=10, _data=data),
        _force_meter_thread=SimpleNamespace(isRunning=lambda: True),
        _force_meter_last_raw_newtons=0.0,
        _force_meter_last_sample_monotonic=time.monotonic(),
    )
    path = Path(config_dir.name) / 'taxel_position_map.json'
    path.write_text(json.dumps(record))
    QFileDialog.getOpenFileName = lambda *_args, **_kwargs: (str(path), '')
    window._load_taxel_position_map()
    assert grid.buttons[43].isEnabled()
    assert not grid.buttons[35].isEnabled()
    assert not api.sent  # File loading must never move the robot.

    grid.buttons[43].click()
    task = window._sidebar_active_task
    assert task.id == 'mapped_taxel_move'
    assert grid.buttons[43].isChecked()
    task.move_settle_sec = 0
    assert window._sidebar_control_timer.isActive()
    assert grid.isEnabled()
    assert not grid.buttons[43].isEnabled()
    assert grid.buttons[43].isChecked()
    assert not window.taxel_mapping_load_map_button.isEnabled()
    assert not window.action_keyboard_tool_velocity.isEnabled()
    window._move_above_mapped_taxel(43)
    assert window._sidebar_active_task is task  # No overlapping/restarted move.
    for _ in range(20):
        data.frame_sequence += 1
        window._run_sidebar_control_action()
        if window._sidebar_active_task is None:
            break
    assert window._sidebar_active_task is None
    assert not window._sidebar_control_timer.isActive()
    assert [entry[0] for entry in api.sent] == [[0.1, 0.2, 0.3], [0.104, 0.2, 0.3]]
    assert grid.buttons[43].isEnabled()
    assert grid.buttons[43].isChecked()
    assert not grid.buttons[35].isEnabled()
    assert window.taxel_mapping_load_map_button.isEnabled()
    assert window.action_keyboard_tool_velocity.isEnabled()
    assert 'Arrived above Taxel 43' in window.taxel_mapping_move_status_label.text()

    grid.buttons[43].click()
    assert window._sidebar_active_task is not None
    window.sidebar_btn_stop.click()
    assert window._sidebar_active_task is None
    assert not window._sidebar_control_timer.isActive()
    assert grid.buttons[43].isEnabled()
    assert 'stopped' in window.taxel_mapping_move_status_label.text()

    # The same grid click recovers an outside/rotated starting pose first.
    api.sent.clear()
    api.position = [0.35, 0.2, 0.29]
    api.quaternion = [float(np.cos(.3)), 0., 0., float(np.sin(.3))]
    window.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
    grid.buttons[43].click()
    task = window._sidebar_active_task
    assert task is not None
    task.move_settle_sec = 0
    for _ in range(40):
        data.frame_sequence += 1
        window.ui_ros._force_meter_last_sample_monotonic = time.monotonic()
        window._run_sidebar_control_action()
        if window._sidebar_active_task is None:
            break
    assert window._sidebar_active_task is None
    positions = [entry[0] for entry in api.sent]
    assert positions[0] == [0.35, 0.2, 0.3]
    assert positions.index([0.1, 0.2, 0.3]) < positions.index([0.104, 0.2, 0.3])
    assert 'Arrived above Taxel 43' in window.taxel_mapping_move_status_label.text()

    sensor = window.ui_ros.sensor_functions
    sensor.n_row = sensor.n_col = 20
    data.diffDataAve = np.zeros((20, 20))
    window._sync_taxel_maps_to_streaming_sensor()
    assert 'Taxel 189' in window.taxel_mapping_anchor_button.text()
    assert 'Taxel 189' in window.taxel_mapping_anchor_button.toolTip()
    assert 'Taxel 35' not in window.taxel_mapping_anchor_button.toolTip()
    assert 'Taxel 189' in window.sidebar_task_info.text()
    assert '20x20' in window.sidebar_task_info.text()
    assert 'Taxel 35' not in window.taxel_mapping_note.text()
    for field in (window.taxel_mapping_reference_speed_spin,
                  window.taxel_mapping_start_margin_spin,
                  window.taxel_mapping_xy_limit_spin):
        assert 'Taxel 35' not in field.toolTip()
    assert len(grid.buttons) == 400
    assert window.taxel_mapping_xy_limit_spin.value() == 200
    assert window.taxel_mapping_coarse_radius_spin.value() == 20
    assert window.taxel_mapping_coarse_step_spin.value() == 2
    assert window.taxel_mapping_progress_label.text() == 'Ready | 0/324 located'
    assert len(window._taxel_mapping_task().skipped_taxel_indices()) == 76
    window._taxel_navigation_result = None
    window._refresh_taxel_mapping_controls()
    assert len(grid._skipped_taxels) == 76
    assert 'outer border' in grid.buttons[0].toolTip()
finally:
    window.ui_ros = None
    window.close()
    QTimer.singleShot = original_single_shot
    QFileDialog.getOpenFileName = original_file_picker
    config_dir.cleanup()
"""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(project_root), env.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=project_root, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_partial_taxel_map_can_resume_from_the_loaded_json():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

config_dir = tempfile.TemporaryDirectory()
os.environ['XDG_CONFIG_HOME'] = config_dir.name

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QFileDialog
from phd.ui.ui_initial import MyMainWindow

class Robot:
    @property
    def current_feedback_state_time(self):
        return time.monotonic()
    def __init__(self):
        self.position = [0.1, 0.2, 0.3]
        self.sent = []
    def get_current_tool_position(self):
        return self.position, [1.0, 0.0, 0.0, 0.0]
    def send_positions_tool_position(self, position, quaternion, **kwargs):
        self.position = list(position)
        self.sent.append((list(position), list(quaternion), kwargs))
        return True
    def exit_end_effector_velocity_mode(self, send_zero=True):
        return True

def measured(index, x, y):
    return {
        'taxel_index': index, 'row': index % 8, 'column': index // 8,
        'status': 'measured', 'attempt_count': 1,
        'tool_position_m': [x, y, 0.275],
        'safe_approach_position_m': [x, y, 0.3],
        'tool_quaternion_wxyz': [1., 0., 0., 0.],
        'signal_counts': 100., 'competing_signal_counts': 20.,
        'dominance_ratio': 5., 'confidence': 1.,
        'method': 'cross_scan_verified',
    }

folder = Path(config_dir.name) / 'partial_map'
folder.mkdir()
path = folder / 'taxel_position_map.json'
record = {
    'task': 'taxel_position_mapping', 'version': 2, 'completed': False,
    'stop_reason': 'manual_stop', 'started_at': '2026-09-28T00:00:00+00:00',
    'settings': {
        'sensor_rows': 8, 'sensor_columns': 10, 'max_xy_offset_mm': 60.,
        'skip_outer_border': True,
        'signal_threshold_counts': 77., 'line_speed_mm_s': 20.,
    },
    'anchor_pose': {
        'taxel_index': 35, 'tool_position_m': [0.1, 0.2, 0.3],
        'tool_quaternion_wxyz': [1., 0., 0., 0.],
    },
    'reference_onset_z_m': 0.275, 'probe_start_z_m': 0.277,
    'reference_attempt_count': 1,
    'initial_row_vector_xy_m': [0.004, 0.0],
    'initial_column_vector_xy_m': [0.0, 0.005],
    'row_vector_xy_m': None, 'column_vector_xy_m': None,
    'positions': [
        measured(27, 0.1, 0.195), measured(34, 0.096, 0.2),
        measured(35, 0.1, 0.2), measured(36, 0.104, 0.2),
    ],
}
path.write_text(json.dumps(record))

# Resuming a map persists its anchor. XDG_CONFIG_HOME only isolates Qt settings;
# this task normally writes under the project's resource/config directory.
from phd.ui.experiment_tasks import get_task
mapping_task = get_task('taxel_position_mapping')
mapping_task.storage_path = Path(config_dir.name) / 'mapping_anchor.json'
mapping_task._anchor_profiles = {}
mapping_task.anchor_pose = None

app = QApplication([])
original_single_shot = QTimer.singleShot
original_file_picker = QFileDialog.getOpenFileName
QTimer.singleShot = lambda *_args, **_kwargs: None
window = MyMainWindow()
try:
    assert window.select_sidebar_task('taxel_position_mapping')
    api = Robot()
    data = SimpleNamespace(frame_sequence=1, diffDataAve=np.zeros((8, 10)))
    window.ui_ros = SimpleNamespace(
        robot_api=api,
        sensor_functions=SimpleNamespace(is_connected=True, n_row=8, n_col=10, _data=data),
        _force_meter_thread=SimpleNamespace(isRunning=lambda: True),
        _force_meter_last_raw_newtons=0.0,
        _force_meter_last_sample_monotonic=time.monotonic(),
    )
    five = {
        'task': 'taxel_position_mapping', 'completed': True,
        'settings': {
            'sensor_rows': 5, 'sensor_columns': 5,
            'max_xy_offset_mm': 100.0, 'skip_outer_border': False,
            'signal_threshold_counts': 77.0,
        },
        'anchor_pose': {
            'taxel_index': 12, 'tool_position_m': [0.1, 0.2, 0.3],
            'tool_quaternion_wxyz': [1., 0., 0., 0.],
        },
        'positions': [
            {
                'taxel_index': index,
                'row': index % 5,
                'column': index // 5,
                'status': 'measured',
                'tool_position_m': [0.1, 0.2, 0.29],
                'safe_approach_position_m': [0.1, 0.2, 0.3],
                'tool_quaternion_wxyz': [1., 0., 0., 0.],
            }
            for index in range(25)
        ],
    }
    threshold_before = window.taxel_mapping_signal_threshold_spin.value()
    window.ui_ros.sensor_functions.n_row = 5
    window.ui_ros.sensor_functions.n_col = 5
    window._show_taxel_position_map(five)
    grid = window.taxel_mapping_map
    assert (grid.rows, grid.columns, len(grid.buttons)) == (5, 5, 25)
    assert grid.layout().rowCount() == 5
    assert grid.layout().columnCount() == 5
    assert grid.layout().getItemPosition(grid.layout().indexOf(grid.buttons[12]))[:2] == (2, 2)
    assert grid.buttons[12].isEnabled()
    assert window.taxel_mapping_signal_threshold_spin.value() == threshold_before
    window.ui_ros.sensor_functions.n_row = 8
    window.ui_ros.sensor_functions.n_col = 10

    QFileDialog.getOpenFileName = lambda *_args, **_kwargs: (str(path), '')
    window._load_taxel_position_map()
    assert window.taxel_mapping_resume_button.isEnabled()
    assert '4/48' in window.taxel_mapping_resume_button.toolTip()
    assert window.taxel_mapping_map.buttons[0].property('taxelState') == 'skipped'
    assert window.taxel_mapping_map.buttons[35].property('taxelState') == 'mapped'

    window.taxel_mapping_resume_button.click()
    task = window._sidebar_active_task
    assert task is not None and task.id == 'taxel_position_mapping'
    assert task is mapping_task
    assert task.storage_path.is_file()
    saved_anchor = json.loads(task.storage_path.read_text())
    assert saved_anchor == {'version': 2, 'profiles': {}}
    assert set(task._positions) == {27, 34, 35, 36}
    assert task._move_action == 'resume_anchor_arrived'
    assert task._output_folder == folder.resolve()
    assert window.taxel_mapping_signal_threshold_spin.value() == 77.
    assert not window.taxel_mapping_resume_button.isEnabled()
    assert 'Mapping live · 4/48' in window.taxel_mapping_map_info.text()
    assert window.taxel_mapping_map.buttons[35].property('taxelState') == 'mapped'
    assert not window.taxel_mapping_map.buttons[35].isEnabled()
    assert api.sent

    window.sidebar_btn_stop.click()
    saved = json.loads(path.read_text())
    assert saved['mapped_count'] == 4
    assert saved['resume_history'][-1]['mapped_count_before_resume'] == 4
    assert window.taxel_mapping_resume_button.isEnabled()
finally:
    window.ui_ros = None
    window.close()
    QTimer.singleShot = original_single_shot
    QFileDialog.getOpenFileName = original_file_picker
    config_dir.cleanup()
"""
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(project_root), env.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=project_root, env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

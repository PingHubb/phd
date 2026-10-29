"""Sensor-specific map discovery and automatic UI loading never start motion."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from phd.ui.taxel_map_catalog import TaxelMapCatalog


def saved_map(identity="sensor:first", saved_at="2026-10-23T10:00:00+00:00"):
    return {
        "task": "taxel_position_mapping", "version": 3, "completed": False,
        "sensor_identity": identity, "saved_at": saved_at,
        "settings": {"sensor_rows": 5, "sensor_columns": 5, "max_xy_offset_mm": 100},
        "anchor_pose": {"taxel_index": 12, "tool_position_m": [.1, .2, .3],
                        "tool_quaternion_wxyz": [1., 0., 0., 0.]},
        "probe_start_z_m": .295,
        "positions": [{"taxel_index": 12, "row": 2, "column": 2, "status": "measured",
                       "tool_position_m": [.1, .2, .29],
                       "safe_approach_position_m": [.1, .2, .3],
                       "tool_quaternion_wxyz": [1., 0., 0., 0.]}],
    }


def write_map(folder, result):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "taxel_position_map.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def test_catalog_matches_identity_and_shape_and_prefers_save_time(tmp_path):
    old = write_map(tmp_path / "copied-later", saved_map(saved_at="2026-10-22T10:00:00Z"))
    newest = write_map(tmp_path / "partial", saved_map())
    other = write_map(tmp_path / "same-size-other-sensor", saved_map("sensor:second", "2026-10-24T10:00:00Z"))
    os.utime(old, (2000000000, 2000000000))
    catalog = TaxelMapCatalog(tmp_path)
    assert catalog.latest("sensor:first", (5, 5)).path == newest
    assert catalog.latest("sensor:second", (5, 5)).path == other
    assert catalog.latest("sensor:first", (20, 20)) is None
    assert catalog.latest("sensor:missing", (5, 5)) is None
    assert catalog.latest("layout:5x5", (5, 5)) is None
    assert catalog.latest("", (5, 5)) is None


def test_catalog_ignores_corrupt_and_unsafe_maps_and_reloads_changed_files(tmp_path):
    valid = write_map(tmp_path / "valid", saved_map())
    bad = write_map(tmp_path / "bad", saved_map())
    bad.write_text("{", encoding="utf-8")
    unsafe = saved_map(saved_at="2026-10-24T10:00:00Z")
    unsafe["positions"][0]["safe_approach_position_m"][0] = 99
    invalid = write_map(tmp_path / "invalid-position", unsafe)
    catalog = TaxelMapCatalog(tmp_path)
    entry = catalog.latest("sensor:first", (5, 5))
    assert entry.path == valid
    assert set(catalog.errors) == {bad, invalid}
    assert catalog.latest("sensor:first", (5, 5)) is entry  # unchanged JSON is cached
    write_map(bad.parent, saved_map(saved_at="2026-10-25T10:00:00Z"))
    assert catalog.latest("sensor:first", (5, 5)).path == bad
    assert bad not in catalog.errors
    bad.unlink()
    assert catalog.latest("sensor:first", (5, 5)).path == valid


def test_catalog_legacy_identity_and_empty_new_map_are_not_guessed(tmp_path):
    legacy = saved_map()
    legacy.pop("sensor_identity")
    write_map(tmp_path / "legacy", legacy)
    catalog = TaxelMapCatalog(tmp_path)
    assert catalog.latest("sensor:first", (5, 5)) is None
    old = saved_map()
    old.pop("saved_at")
    write_map(tmp_path / "old", old)
    empty = saved_map(saved_at="2099-01-01T00:00:00Z")
    empty["positions"] = []
    path = write_map(tmp_path / "new-placement", empty)
    # Do not silently reuse old coordinates when the newest mapping attempt
    # has not measured any taxels at the new placement yet.
    entry = catalog.latest("sensor:first", (5, 5))
    assert entry.path == path and entry.result["positions"] == []


@pytest.mark.parametrize("kind", ["single", "full", "move", "resume", "relocate"])
def test_wrong_sensor_map_is_rejected_before_any_robot_command(monkeypatch, kind):
    from phd.ui import experiment_tasks
    from phd.ui.taxel_position_navigation import MappedTaxelMoveTask
    from test_experiment_tasks import _Host, _RobotApi, _saved_full_calibration_map

    result = _saved_full_calibration_map()
    result["sensor_identity"] = "sensor:first"
    monkeypatch.setattr(experiment_tasks, "connected_calibration_sensor",
                        lambda host: ("sensor:second", "Second sensor", 8, 10))
    if kind == "move":
        task = MappedTaxelMoveTask(result, 35, line_speed_m_s=.02, force_guard_g=20)
    elif kind in {"resume", "relocate"}:
        task = experiment_tasks.TaxelPositionMappingTask(storage_path=None, output_directory=None)
        setattr(task, "_resume_request" if kind == "resume" else "_relocation_request", result)
    else:
        cls = experiment_tasks.CalibrationTask if kind == "single" else experiment_tasks.FullTaxelCalibrationTask
        task = cls(storage_path=None, output_directory=None)
        task.position_map = result
    api = _RobotApi()
    host = _Host(api, task)
    assert not task.on_start(host)
    assert not api.sent and not api.velocity_commands and not api.scripts
    assert any("loaded map belongs to sensor:first" in text for text in host.messages)


def test_ui_autoload_switch_reconnect_busy_manual_override_and_no_motion(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    script = r'''
import os, tempfile, json
from pathlib import Path
from types import SimpleNamespace
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from phd.ui import experiment_tasks
from phd.ui.ui_initial import MyMainWindow
from phd.ui.taxel_map_catalog import TaxelMapCatalog
from test_taxel_map_autoload import saved_map, write_map

config = tempfile.TemporaryDirectory()
os.environ['XDG_CONFIG_HOME'] = config.name
folder = Path(config.name) / 'maps'
first = write_map(folder / 'first', saved_map())
second = write_map(folder / 'second', saved_map('sensor:second'))
original_get_task = experiment_tasks.get_task
tasks = {
    'calibration': experiment_tasks.CalibrationTask(storage_path=None, output_directory=None),
    'full_taxel_calibration': experiment_tasks.FullTaxelCalibrationTask(storage_path=None, output_directory=None),
    'taxel_position_mapping': experiment_tasks.TaxelPositionMappingTask(storage_path=None, output_directory=None),
}
experiment_tasks.get_task = lambda key: tasks[key] if key in tasks else original_get_task(key)
sensor = SimpleNamespace(is_connected=True, n_row=5, n_col=5, key='sensor:first')
experiment_tasks.connected_calibration_sensor = lambda host: (
    (sensor.key, sensor.key, sensor.n_row, sensor.n_col) if host.ui_ros else ('layout:unknown', 'sensor', 0, 0)
)
app = QApplication([])
QTimer.singleShot = lambda *args, **kwargs: None
window = MyMainWindow()
window._taxel_map_catalog = TaxelMapCatalog(folder)
window.ui_ros = SimpleNamespace(sensor_functions=sensor)
def no_motion(*args, **kwargs):
    raise AssertionError('Loading must never start motion or an experiment')
window._move_above_mapped_taxel = no_motion
window._start_sidebar_control = no_motion
def poll():
    window._auto_taxel_map_next_check = 0
    window._sync_taxel_maps_to_streaming_sensor()
def loaded(identity):
    assert window._taxel_navigation_result['sensor_identity'] == identity
    assert tasks['calibration'].position_map['sensor_identity'] == identity
    assert tasks['full_taxel_calibration'].position_map['sensor_identity'] == identity
    full = tasks['full_taxel_calibration']
    assert full.initial_position['tool_position_m'] == full.position_map['anchor_pose']['tool_position_m']
    assert full.initial_position['source'] == 'taxel_position_map_anchor'
    assert 'map safe anchor (Taxel 12)' in window.full_taxel_initial_position_label.text()
    expected = first if identity == 'sensor:first' else second
    for name in ('taxel_mapping_map_folder_label', 'calibration_map_folder_label',
                 'full_taxel_calibration_map_folder_label'):
        label = getattr(window, name)
        assert label.text() == 'Map folder: ' + expected.parent.name
        assert str(expected.parent) in label.toolTip()
        assert label.property('mapFile') == str(expected)
        assert label.accessibleName() == 'Loaded map folder'
        label.actions()[0].trigger()
        assert app.clipboard().text() == str(expected.parent)
        label.actions()[1].trigger()
        assert app.clipboard().text() == str(expected)
try:
    poll()
    loaded('sensor:first')
    assert window._taxel_navigation_source_path == str(first)
    assert 'Auto-loaded' in window.calibration_map_info.text()
    assert window._sidebar_active_task is None
    assert tasks['calibration'].initial_position is None  # user still chooses a taxel
    assert 12 in window.calibration_taxel_map._available_taxels
    full = tasks['full_taxel_calibration']
    home_text = window.full_taxel_initial_position_label.text()
    full._home_position = dict(full.initial_position)
    full._test_active = True
    full.initial_position = {'tool_position_m': [.11, .22, .3], 'taxel_index': 13}
    window._refresh_full_taxel_calibration_controls()
    assert window.full_taxel_initial_position_label.text() == home_text
    full._test_active = False
    window._refresh_full_taxel_calibration_controls()

    # Identical dimensions must not retain a different physical sensor's map/pose.
    tasks['calibration'].use_mapped_taxel(tasks['calibration'].position_map['positions'][0], rows=5, columns=5)
    sensor.key = 'sensor:second'
    poll()
    loaded('sensor:second')
    assert tasks['calibration'].initial_position is None
    assert tasks['taxel_position_mapping'].active_sensor_key_name == 'sensor:second'

    sensor.key = 'sensor:missing'
    poll()
    assert window._taxel_navigation_result is None
    assert tasks['calibration'].position_map is None
    assert tasks['full_taxel_calibration'].position_map is None
    assert tasks['full_taxel_calibration'].initial_position is None
    assert tasks['full_taxel_calibration']._home_position is None
    assert not window.calibration_taxel_map._available_taxels
    assert not window.full_taxel_calibration_map._available_taxels
    assert not window.taxel_mapping_resume_button.isEnabled()
    assert 'No saved position map matches' in window.calibration_map_info.text()
    for name in ('taxel_mapping_map_folder_label', 'calibration_map_folder_label',
                 'full_taxel_calibration_map_folder_label'):
        label = getattr(window, name)
        assert label.text() == 'Map folder: no map loaded'
        assert not label.property('mapFolder')
        assert not any(action.isEnabled() for action in label.actions())

    # Connection changes defer until the live experiment is over.
    window._sidebar_active_task = object()
    sensor.key = 'sensor:first'
    poll()
    assert window._taxel_navigation_result is None
    window._sidebar_active_task = None
    poll()
    loaded('sensor:first')

    # Browsing an older map is allowed until the next matching save/reconnect.
    manual = saved_map(saved_at='2026-01-01T00:00:00Z')
    window._show_taxel_position_map(manual, source_path='/manual/old-map.json')
    poll()
    assert window._taxel_navigation_source_path == '/manual/old-map.json'
    assert window.taxel_mapping_map_folder_label.text() == 'Map folder: manual'
    assert window.calibration_map_folder_label.property('mapFile') == '/manual/old-map.json'
    assert window.full_taxel_calibration_map_folder_label.property('mapFile') == '/manual/old-map.json'

    # Each calibration page reports its own map, including a pinned resume map,
    # rather than blindly repeating the navigation page's newer source path.
    tasks['full_taxel_calibration'].position_map_path = '/pinned-run/source-map.json'
    window._refresh_full_taxel_calibration_controls()
    assert window.full_taxel_calibration_map_folder_label.text() == 'Map folder: pinned-run'
    assert window.taxel_mapping_map_folder_label.text() == 'Map folder: manual'
    resumed = saved_map(saved_at='2026-10-25T10:00:00Z')
    write_map(first.parent, resumed)
    poll()
    assert window._taxel_navigation_source_path == str(first)
    assert window._taxel_navigation_result['saved_at'] == resumed['saved_at']

    window._show_taxel_position_map(manual, source_path='/manual/old-map.json')
    sensor.is_connected = False
    poll()
    sensor.is_connected = True
    poll()
    assert window._taxel_navigation_source_path == str(first)

    # A newest empty partial map must remove old usable calibration coordinates.
    empty = saved_map(saved_at='2026-10-26T10:00:00Z')
    empty['positions'] = []
    write_map(first.parent, empty)
    poll()
    assert window._taxel_navigation_result['positions'] == []
    assert tasks['calibration'].position_map is None
    assert tasks['full_taxel_calibration'].position_map is None
    assert tasks['full_taxel_calibration'].initial_position is None
    assert 'Continue mapping first' in window.calibration_map_info.text()
    assert window._sidebar_active_task is None
    assert window.taxel_mapping_map_folder_label.property('mapFile') == str(first)
    assert window.calibration_map_folder_label.text() == 'Map folder: no map loaded'
    assert window.full_taxel_calibration_map_folder_label.text() == 'Map folder: no map loaded'

    # Unsaved in-memory results never retain an old disk location or copy action.
    window._show_taxel_position_map(saved_map())
    for name in ('taxel_mapping_map_folder_label', 'calibration_map_folder_label',
                 'full_taxel_calibration_map_folder_label'):
        label = getattr(window, name)
        assert label.text() == 'Map folder: current result has no saved path'
        assert not label.property('mapFile')
        assert not any(action.isEnabled() for action in label.actions())

    # Long names remain a single compact line and copy the complete path.
    long_folder = 'taxel_position_map_relocated_20261026_sensor-' + 'long-name-' * 30
    long_path = '/tmp/maps/' + long_folder + '/taxel_position_map.json'
    window._show_taxel_position_map(saved_map(), source_path=long_path)
    label = window.taxel_mapping_map_folder_label
    label.resize(350, 26)
    assert label.sizePolicy().horizontalPolicy() == label.sizePolicy().Ignored
    assert not label.wordWrap()
    assert label.text() == 'Map folder: ' + long_folder
    label.actions()[0].trigger()
    assert app.clipboard().text() == '/tmp/maps/' + long_folder
    assert not label.grab().isNull()
finally:
    window.ui_ros = None
    window.close()
    config.cleanup()
'''
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join((str(project_root), str(project_root / "test")))
    result = subprocess.run([sys.executable, "-c", script], cwd=project_root,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

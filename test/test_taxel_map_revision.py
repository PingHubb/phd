"""Remember confirmations never rewrite measurements or command hardware."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from phd.ui import experiment_tasks
from phd.ui.experiment_tasks import TaxelPositionMappingTask, FullTaxelCalibrationTask, CalibrationTask
from phd.ui.taxel_map_revision import (
    normalize_setup, read_revision_source, revision_motion_error, revision_setup, save_setup_revision,
)
from phd.ui.taxel_position_navigation import MappedTaxelMoveTask, prepare_taxel_map
from phd.ui.taxel_map_catalog import TaxelMapCatalog
from test_taxel_map_autoload import saved_map, write_map
from test_experiment_tasks import _Host, _RobotApi


def proposal(result, *, z=.28, scan=.25):
    anchor = copy.deepcopy(result["anchor_pose"])
    anchor["tool_position_m"][2] = z
    return dict(sensor_identity=result["sensor_identity"], sensor_label="Test sensor",
                sensor_rows=5, sensor_columns=5, anchor_pose=anchor, fixed_scan_z_m=scan)


def test_revision_preserves_original_bytes_and_all_measurements_and_chains(tmp_path):
    result = saved_map()
    result["failures"] = {"8": "low signal"}
    result["positions"].append(dict(taxel_index=8, status="failed", reason="low signal"))
    source = write_map(tmp_path / "original", result)
    original = source.read_bytes()
    _, digest = read_revision_source(source)
    revision, path = save_setup_revision(source, proposal(result), digest)
    assert path.parent != source.parent
    assert source.read_bytes() == original
    assert revision["positions"] == result["positions"]
    assert revision["anchor_pose"] == result["anchor_pose"]
    assert revision["settings"] == result["settings"]
    assert revision["failures"] == result["failures"]
    assert revision_motion_error(revision)
    assert revision_setup(revision)["anchor_pose"]["tool_position_m"][2] == .28
    assert prepare_taxel_map(revision)["positions"] == result["positions"][:1]
    assert json.loads(path.read_text()) == revision
    before = path.read_bytes()
    _, parent_digest = read_revision_source(path)
    second, second_path = save_setup_revision(path, proposal(result, scan=.24), parent_digest)
    assert path.read_bytes() == before
    assert second["setup_revision"]["parent_json_path"] == str(path)
    assert second["positions"] == result["positions"]
    assert TaxelMapCatalog(tmp_path).latest("sensor:first", (5, 5)).path == second_path


def test_identical_geometry_and_quaternion_sign_do_not_require_verification(tmp_path):
    result = saved_map()
    source = write_map(tmp_path / "source", result)
    _, digest = read_revision_source(source)
    setup = proposal(result, z=.3, scan=None)
    setup["anchor_pose"]["tool_quaternion_wxyz"] = [-1., 0., 0., 0.]
    revision, _ = save_setup_revision(source, setup, digest)
    assert not revision_motion_error(revision)
    FullTaxelCalibrationTask.prepare_position_map(revision)


@pytest.mark.parametrize("kind", ["move", "calibration", "full", "resume", "load"])
def test_pending_geometry_is_rejected_by_motion_consumers(tmp_path, kind):
    result = saved_map()
    source = write_map(tmp_path / "source", result)
    _, digest = read_revision_source(source)
    revision, _ = save_setup_revision(source, proposal(result), digest)
    if kind == "move":
        with pytest.raises(ValueError, match="Verify"):
            MappedTaxelMoveTask(revision, 12, line_speed_m_s=.02, force_guard_g=20)
    elif kind == "load":
        with pytest.raises(ValueError, match="Verify"):
            FullTaxelCalibrationTask.prepare_position_map(revision)
    elif kind == "resume":
        valid, message = TaxelPositionMappingTask.validate_resume_result(revision)
        assert not valid and "Verify" in message
    else:
        cls = CalibrationTask if kind == "calibration" else FullTaxelCalibrationTask
        task = cls(storage_path=None, output_directory=None)
        task.position_map = revision
        api = _RobotApi()
        host = _Host(api, task)
        assert not task._validate_start_prerequisites(host)
        assert any("Verify" in message for message in host.messages)
        assert not api.sent and not api.velocity_commands and not api.scripts


def test_concurrent_change_and_write_failure_do_not_damage_original(tmp_path, monkeypatch):
    source = write_map(tmp_path / "source", saved_map())
    _, digest = read_revision_source(source)
    source.write_text(source.read_text() + "\n")
    original = source.read_bytes()
    with pytest.raises(ValueError, match="changed"):
        save_setup_revision(source, proposal(saved_map()), digest)
    _, digest = read_revision_source(source)
    import phd.ui.taxel_map_revision as module
    monkeypatch.setattr(module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError, match="disk failure"):
        save_setup_revision(source, proposal(saved_map()), digest)
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source.parent]


def test_preview_has_no_side_effects_and_rejects_stale_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_tasks, "connected_calibration_sensor", lambda host: ("sensor:first", "First", 5, 5))
    task = TaxelPositionMappingTask(storage_path=tmp_path / "anchor.json", output_directory=None)
    api = _RobotApi()
    host = _Host(api, task)
    before = copy.deepcopy(task.__dict__)
    setup, _ = task.preview_remembered_setup(host, "anchor", saved_map())
    assert task.__dict__ == before
    assert not task.storage_path.exists()
    task.apply_remembered_setup(setup, persist=True)
    api.tool_position[2] = .27
    plane, _ = task.preview_remembered_setup(host, "scan_height", saved_map())
    assert plane["fixed_scan_z_m"] == .27
    assert task.fixed_scan_z_m is None
    api.current_feedback_state_time = time.monotonic() - 4
    with pytest.raises(ValueError, match="Fresh"):
        task.preview_remembered_setup(host, "anchor", saved_map())
    assert not api.sent and not api.velocity_commands and not api.scripts


def test_explicit_unset_plane_does_not_recover_a_previous_geometry(tmp_path, monkeypatch):
    task = TaxelPositionMappingTask(storage_path=None, output_directory=tmp_path)
    task.apply_remembered_setup(proposal(saved_map(), scan=None))
    monkeypatch.setattr(task, "_scan_height_recorded_for_this_anchor", lambda: pytest.fail("must not recover an old plane"))
    task._restore_scan_plane_choice()
    assert task.fixed_scan_z_m is None


@pytest.mark.parametrize("scan", [.299, .301, .199, float("nan")])
def test_invalid_plane_is_rejected(scan):
    with pytest.raises(ValueError, match="5–100"):
        normalize_setup(proposal(saved_map(), z=.3, scan=scan))


def test_relocation_uses_proposed_setup_but_original_measured_centres(tmp_path, monkeypatch):
    import numpy as np
    from test_experiment_tasks import _partial_relocation_source
    monkeypatch.setattr(experiment_tasks, "connected_calibration_sensor", lambda host: ("sensor:first", "First", 5, 5))
    source = _partial_relocation_source()
    source["sensor_identity"] = "sensor:first"
    source["anchor_pose"] = dict(taxel_index=12, tool_position_m=[.008, .008, .25],
                                 tool_quaternion_wxyz=[1., 0., 0., 0.])
    source["settings"]["max_xy_offset_mm"] = 100
    path = write_map(tmp_path / "original", source)
    _, digest = read_revision_source(path)
    setup = proposal(source, z=.32, scan=.27)
    setup["anchor_pose"]["tool_position_m"][:2] = [.1, .2]
    revision, revision_path = save_setup_revision(path, setup, digest)
    task = TaxelPositionMappingTask(storage_path=None, output_directory=None)
    api = _RobotApi()
    host = _Host(api, task)
    host.ui_ros.sensor_functions.n_row = host.ui_ros.sensor_functions.n_col = 5
    host.ui_ros.sensor_functions._data.diffDataAve = np.zeros((5, 5))
    assert task.request_relocation(revision, revision_path)[0]
    assert task.on_start(host)
    assert task.anchor_pose["tool_position_m"] == [.1, .2, .32]
    assert task.fixed_scan_z_m == .27
    assert task._relocation_source_positions[12]["tool_position_m"] == source["positions"][0]["tool_position_m"]
    assert task._relocation_continues_mapping
    task.on_stop(host, show_result=False)


def test_corrupt_or_mismatched_revision_cannot_load_or_relocate(tmp_path):
    source = write_map(tmp_path / "source", saved_map())
    _, digest = read_revision_source(source)
    result, _ = save_setup_revision(source, proposal(saved_map()), digest)
    bad = copy.deepcopy(result)
    bad["setup_revision"]["requested_setup"]["sensor_identity"] = "sensor:other"
    with pytest.raises(ValueError, match="sensor/anchor"):
        prepare_taxel_map(bad)
    assert not TaxelPositionMappingTask.validate_relocation_result(bad)[0]
    bad = copy.deepcopy(result)
    bad["setup_revision"]["verification_required"] = False
    with pytest.raises(ValueError, match="verification-required"):
        prepare_taxel_map(bad)


def test_ui_confirm_cancel_revision_reload_and_resume_protection(tmp_path):
    # Isolate singleton task registry and real Qt controls from other tests.
    script = r'''
import copy, os, tempfile
from pathlib import Path
from types import SimpleNamespace
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from phd.ui import experiment_tasks, components
from phd.ui.ui_initial import MyMainWindow
from phd.ui.taxel_map_catalog import TaxelMapCatalog
from test_taxel_map_autoload import saved_map, write_map
from test_experiment_tasks import _RobotApi
from phd.ui.taxel_map_revision import revision_setup

config = tempfile.TemporaryDirectory()
os.environ['XDG_CONFIG_HOME'] = config.name
root = Path(config.name)
source = write_map(root / 'maps' / 'original', saved_map())
original = source.read_bytes()
old_get_task = experiment_tasks.get_task
tasks = {key: cls(storage_path=None, output_directory=None) for key, cls in (
    ('calibration', experiment_tasks.CalibrationTask),
    ('full_taxel_calibration', experiment_tasks.FullTaxelCalibrationTask),
    ('taxel_position_mapping', experiment_tasks.TaxelPositionMappingTask))}
mapping = tasks['taxel_position_mapping']
mapping.storage_path = root / 'anchor.json'
mapping.output_directory = root / 'maps'
experiment_tasks.get_task = lambda key: tasks.get(key) or old_get_task(key)
experiment_tasks.connected_calibration_sensor = lambda host: ('sensor:first', 'First', 5, 5)
app = QApplication([])
real_confirm = components.confirm
single_shot = QTimer.singleShot
QTimer.singleShot = lambda *args, **kwargs: None
window = MyMainWindow()
api = _RobotApi()
window.ui_ros = SimpleNamespace(robot_api=api, sensor_functions=SimpleNamespace(is_connected=True, n_row=5, n_col=5))
window._current_sidebar_control_id = lambda: 'taxel_position_mapping'
window._taxel_map_catalog = TaxelMapCatalog(root / 'maps')
window._show_taxel_position_map(saved_map(), str(source))
window._start_sidebar_control = lambda: (_ for _ in ()).throw(AssertionError('Unexpected start'))
messages = []
window._append_sidebar_control_message = messages.append
seen = []
def confirm(parent, title, message, **kwargs):
    assert parent is window
    assert not kwargs['destructive']
    assert kwargs['confirm_text'] == 'Create and load revision'
    assert str(source) in kwargs['detail']
    seen.append(kwargs['detail'])
    return False
components.confirm = confirm
api.tool_position[2] = .32
before = copy.deepcopy(mapping._anchor_profiles)
window._remember_taxel_mapping_anchor()
assert len(seen) == 1
assert mapping._anchor_profiles == before
assert source.read_bytes() == original
assert list((root / 'maps').iterdir()) == [source.parent]
assert not mapping.storage_path.exists()

# Exercise the actual native modal too: default/escape is Cancel, with the
# experiment utility window as parent, and no robot action from the dialog.
window.experiments_window.show()
def cancel_native():
    box = app.activeModalWidget()
    assert box.parentWidget() is window.experiments_window
    assert box.defaultButton().text() == 'Cancel'
    assert box.escapeButton() is box.defaultButton()
    capture = os.environ.get('REVISION_DIALOG_CAPTURE')
    if capture:
        assert box.grab().save(capture)
    box.defaultButton().click()
single_shot(0, cancel_native)
assert not real_confirm(window.experiments_window, 'Save mapping setup?',
    'Remember this safe anchor?', detail=seen[0],
    confirm_text='Create and load revision', destructive=False)
window.experiments_window.hide()

components.confirm = lambda *args, **kwargs: True
window._remember_taxel_mapping_anchor()
assert 'Created and loaded' in messages[-1], messages
path1 = Path(window._taxel_navigation_source_path)
assert path1 != source and path1.exists()
assert mapping.anchor_pose['tool_position_m'][2] == .32
assert mapping.fixed_scan_z_m is None
assert 'verification required' in window.taxel_mapping_map_info.text()
assert not window.taxel_mapping_resume_button.isEnabled()
assert not window.taxel_mapping_map.buttons[12].isEnabled()
for key in ('calibration', 'full_taxel_calibration'):
    assert tasks[key].position_map['positions'] == []
    assert tasks[key].initial_position is None
assert tasks['full_taxel_calibration']._home_position is None
assert not window.full_taxel_calibration_map._available_taxels
assert source.read_bytes() == original

api.tool_position[2] = .27
previous = path1.read_bytes()
window._remember_taxel_mapping_scan_height()
path2 = Path(window._taxel_navigation_source_path)
assert path2 != path1
assert path1.read_bytes() == previous
assert mapping.fixed_scan_z_m == .27
assert revision_setup(window._taxel_navigation_result)['anchor_pose']['tool_position_m'][2] == .32

# Reopening/automatic discovery reapplies the proposed setup, not root geometry.
mapping._anchor_profiles.clear()
mapping.anchor_pose = None
window._auto_taxel_map_context = None
window._auto_load_connected_sensor_map()
assert window._taxel_navigation_source_path == str(path2)
assert mapping.anchor_pose['tool_position_m'][2] == .32
assert mapping.fixed_scan_z_m == .27
assert 'verification required' in window.taxel_mapping_map_info.text()

# A staged resume cannot be changed by the Remember buttons or a manual load.
full = tasks['full_taxel_calibration']
full.resume_plan = {'original': True}
snapshot = copy.deepcopy(full.position_map)
window._remember_taxel_mapping_anchor()
assert 'staged' in messages[-1]
assert full.position_map == snapshot
window._show_taxel_position_map(window._taxel_navigation_result, str(path2))
assert full.position_map == snapshot
full.resume_plan = None

# A changed file while the modal is open is refused, not silently revised.
def changed(*args, **kwargs):
    path2.write_text(path2.read_text() + '\n')
    return True
components.confirm = changed
window._remember_taxel_mapping_scan_height()
assert 'changed while confirmation' in messages[-1], messages[-1]
assert window._taxel_navigation_source_path == str(path2)
assert not api.sent and not api.velocity_commands and not api.scripts
window.ui_ros = None
window.close()
print('revision UI smoke passed')
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            env={**os.environ, "QT_QPA_PLATFORM": "offscreen",
                                 "PYTHONPATH": str(Path(__file__).parent)},
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "revision UI smoke passed" in result.stdout

"""Render the real checkpoint-row construction without a robot/UI runtime."""
import os
from pathlib import Path
import subprocess
import sys


def test_checkpoint_selector_layout_and_preview_action():
    script = r'''
import ast
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QPushButton, QComboBox, QSizePolicy, QCheckBox, QDoubleSpinBox
from phd.ui.ui_ping_direct_finger_motion import DirectFingerMotionMixin
from phd.ui import theme
app = QApplication([])
self = DirectFingerMotionMixin()
temporary = TemporaryDirectory()
folder = Path(temporary.name)
default = str(folder / 'first.pt')
other = str(folder / 'second.pt')
labels = ['swipe_up', 'swipe_down', 'swipe_left', 'swipe_right']
for path, architecture, classes in [(default, 'cnn_tcn', labels), (other, 'cnn_gru', labels + ['push', 'pull'])]:
    torch.save(dict(format='pinglab_touch_gesture_policy_v1', class_names=classes,
        sensor_shape=[10,10], members=[dict(config=dict(architecture=architecture, classes=len(classes)))],
        preprocess=dict(prediction_unit='recent_touch_window', window_duration_sec=.2)), path)
self._get_default_ai_execution_model_path = lambda: default
self._get_ai_direct_finger_motion_execution_helper = lambda: (_ for _ in ()).throw(AssertionError('must not construct controller'))
DisabledSensorFunctions = SimpleNamespace(DEFAULT_AI_DIRECT_EXECUTION_MODEL_PATH=default)
tree = ast.parse(Path('phd/ui/ui_ping.py').read_text())
setup = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'setup_tab3')
start = next(i for i,n in enumerate(setup.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'model_row')
end = next(i for i,n in enumerate(setup.body[start:],start) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'execute_safety_row')
exec(compile(ast.Module(body=setup.body[start:end], type_ignores=[]), 'real_checkpoint_row', 'exec'))
assert [button.text() for button in model_row.findChildren(QPushButton)] == ['Select Model']
assert not model_row.findChildren(QComboBox)
self.ai_direct_execution_dry_run_checkbox = QCheckBox()
self.ai_direct_execution_velocity_scale_spin = QDoubleSpinBox()
self.ai_direct_execution_velocity_scale_spin.valueChanged.connect(self._on_ai_direct_execution_velocity_scale_changed)
self.ai_direct_execution_select_model_button.clicked.connect(self._on_select_ai_direct_execution_model)
def wait_for_text(text):
    for i in range(100):
        if text in self.ai_direct_execution_model_details.label.text():
            return
        QTest.qWait(25)
    raise AssertionError(self.ai_direct_execution_model_details.label.text())
wait_for_text('CNN–TCN · 4 gestures')
assert 'Push' not in self.ai_direct_execution_model_details.label.text()
for width in (375, 850):
    model_row.resize(width, 300)
    model_row.show()
    app.processEvents()
    assert model_row.width() == width
    for widget in (self.ai_direct_execution_model_details, self.ai_direct_execution_select_model_button):
        assert widget.geometry().right() < width
        assert widget.width() > 80
class Dialog:
    ExistingFile, DontUseNativeDialog, Detail, Accepted = range(4)
    def __init__(self, *args): pass
    def setFileMode(self, *args): pass
    def setOption(self, *args): pass
    def setViewMode(self, *args): pass
    def exec_(self): return self.Accepted
    def selectedFiles(self): return [other]
with patch('phd.ui.ui_ping_direct_finger_motion.QFileDialog', Dialog):
    self.ai_direct_execution_select_model_button.click()
wait_for_text('CNN–GRU · 6 gestures')
assert 'Push' in self.ai_direct_execution_model_details.label.text()
assert self.ai_direct_execution_dry_run_checkbox.isChecked()
assert self.ai_direct_execution_velocity_scale_spin.value() == 1.0
assert self.ai_direct_execution_model_path_input.text() == other
assert not getattr(self, '_ai_direct_finger_execution_active', False)
helper = SimpleNamespace(is_running=True, model_loaded=False, loaded_model_checkpoint_path='')
self._refresh_ai_direct_execution_model_status(helper)
assert not self.ai_direct_execution_select_model_button.isEnabled()
self.ai_direct_execution_model_details.shutdown()
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", script], cwd=root,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen"), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

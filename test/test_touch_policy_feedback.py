"""Result cards render the executor's evidence without starting hardware."""
import os
from pathlib import Path
import subprocess
import sys


def test_live_completed_uncertain_stopped_and_legacy_cards():
    script = r'''
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication
from phd.ui.gesture_feedback import TouchPolicyGestureFeedback, GestureResultCard
from phd.ui.gesture_recognition_panel import GestureRecognitionPanel
app = QApplication([])
panel = TouchPolicyGestureFeedback()
recognition = GestureRecognitionPanel(lambda: None)
assert isinstance(recognition.result_card, GestureResultCard)
assert recognition.confidence_bar.height() == panel.completed.confidence_bar.height() == 6
panel.update_from_helper(None)
assert panel.current.result_label.text() == 'Waiting for a gesture'
completed = dict(gesture='left', accepted=True, confidence=.93, margin=.83, continuous=True)
helper = SimpleNamespace(model_loaded=True, model_kind='cnn_tcn_gesture', is_running=True,
    touch_gesture_feedback=SimpleNamespace(completed=completed),
    last_prediction=dict(gesture='right', mode='move', mode_conf=.96, gesture_margin=.9, dry_run=True))
panel.update_from_helper(helper)
assert panel.current.result_label.text() == 'Right'
assert '96.00%' in panel.current.confidence_label.text()
assert panel.current.confidence_bar.value() == 960
assert panel.completed.result_label.text() == 'Left'
assert 'Last direction before release' in panel.completed.confidence_label.text()
assert panel.completed.confidence_bar.value() == 930
helper.last_prediction['gesture'] = 'swipe_up'
completed['gesture'] = 'swipe_left'
panel.update_from_helper(helper)
assert panel.current.result_label.text() == 'Swipe up'
assert panel.completed.result_label.text() == 'Swipe left'
helper.last_prediction['gesture'] = 'two_finger_push'
helper.last_prediction['mode'] = 'push'
completed['gesture'] = 'two_finger_pull'
panel.update_from_helper(helper)
assert panel.current.result_label.text() == 'Two finger push'
assert panel.completed.result_label.text() == 'Two finger pull'
helper.last_prediction['gesture'] = 'right'
completed['gesture'] = 'left'
helper.last_prediction['mode'] = 'stop'
panel.update_from_helper(helper)
assert panel.current.result_label.text() == 'Confirming Right'
helper.last_prediction = dict(gesture='stop', mode='stop', mode_conf=1., reason='Released / waiting for touch')
panel.update_from_helper(helper)
assert panel.current.result_label.text() == 'Waiting for touch'
assert panel.current.confidence_bar.value() == 0
assert panel.completed.result_label.text() == 'Left'
helper.last_prediction = dict(gesture='uncertain', mode='stop', mode_conf=.4, gesture_margin=.05, top_gesture='up')
panel.update_from_helper(helper)
assert panel.current.result_label.text() == 'Uncertain'
assert 'Closest match: Up' in panel.current.confidence_label.text()
helper.is_running = False
panel.update_from_helper(helper)
assert panel.current.result_label.text() == 'Stopped'
assert panel.current.confidence_bar.value() == 0
assert panel.completed.result_label.text() == 'Left'
for width in (400, 850):
    panel.resize(width, 650)
    panel.show()
    app.processEvents()
    assert panel.width() == width
    assert panel.current.result_label.wordWrap()
    assert panel.completed.result_label.wordWrap()
panel.clear()
assert panel.completed.result_label.text() == 'Waiting for a gesture'
assert panel.completed.confidence_bar.value() == 0
helper.model_kind = 'cnn_gru'
panel.update_from_helper(helper)
assert panel.completed.result_label.text() == 'Gesture labels unavailable'
recognition.shutdown()
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", script], cwd=root,
                            env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

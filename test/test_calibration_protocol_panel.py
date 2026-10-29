"""Protocol UI wiring without opening or commanding any hardware."""
import os
from pathlib import Path
import subprocess
import sys


def test_protocol_fields_follow_selected_mode_and_presets():
    script = """
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QCheckBox, QFormLayout, QLabel,
    QToolButton, QWidget,
)
from phd.ui.experiment_tasks import CalibrationTask, FullTaxelCalibrationTask
from phd.ui.calibration_protocol_panel import build_protocol_panel

app = QApplication([])
task = CalibrationTask(storage_path=None, output_directory=None)
holder = QWidget()
main_form = QFormLayout(holder)
force = QDoubleSpinBox()
force.setRange(1, 800)
force.setValue(task.contact_threshold_g)
base_speed = QDoubleSpinBox()
main_form.addRow("Base -Z speed", base_speed)
main_form.addRow("Max contact threshold", force)
repeat = QSpinBox()
repeat.setRange(1, 1000)
panel = build_protocol_panel(task, force, repeat, base_speed)
mode = panel.findChild(QComboBox, "characterisationProtocol")


def hidden(widget):
    # Optional settings are hidden as whole sections, not row by row.
    return widget.isHidden() or widget.parentWidget().isHidden()


assert not mode.isHidden()  # The protocol choice is never inside a collapsed section.
assert mode.findData("proximity") == -1 and mode.count() == 7
toggles = {button.text() for button in panel.findChildren(QToolButton)}
assert "Record details · IDs, probe and notes" in toggles
assert any(text.startswith("Advanced settings (") for text in toggles)
assert any("To set:" in label.text() for label in panel.findChildren(QLabel))
speeds = panel.findChild(QLineEdit, "speedSchedule")
slide = panel.findChild(QDoubleSpinBox, "sliding_distance_mm")
confirm = panel.findChild(QCheckBox, "slidingPathConfirmed")
assert hidden(speeds) and hidden(slide)
mode.setCurrentIndex(mode.findData("speed_sweep"))
assert not hidden(speeds) and hidden(slide)
assert repeat.value() == 3
speeds.setText("0.05, 0.2, 0.5")
assert task.speed_schedule_mm_s == [0.05, 0.2, 0.5]
mode.setCurrentIndex(mode.findData("sliding"))
assert hidden(speeds) and not hidden(slide)
assert force.value() == 100 and not confirm.isChecked()
confirm.setChecked(True)
assert task.sliding_path_confirmed
mode.setCurrentIndex(mode.findData("drift"))
assert not task.recalibrate_between_trials
assert task.contact_hold_sec == 30 and task.recovery_observation_sec == 30
mode.setCurrentIndex(mode.findData("sliding"))
assert not confirm.isChecked()
mode.setCurrentIndex(mode.findData("force_sweep"))
forces = panel.findChild(QLineEdit, "forceSchedule")
assert not hidden(forces) and not force.isEnabled()
assert repeat.value() == 5
forces.setText("5, 10, 25")
assert task.force_schedule_g == [5, 10, 25]
panel.findChild(QCheckBox, "forceReferenceVerified").setChecked(True)
assert task.force_reference_verified
mode.setCurrentIndex(mode.findData("standard"))
assert hidden(forces) and force.isEnabled()
motion = panel.findChild(QComboBox, "motionMode")
loading_step = panel.findChild(QDoubleSpinBox, "step_loading_mm")
assert task.motion_mode == "stepwise" and not hidden(loading_step)
loading_step.setValue(0.015)
assert task.step_loading_mm == 0.015
light_contact = panel.findChild(QDoubleSpinBox, "light_contact_threshold_g")
assert not hidden(light_contact) and task.light_contact_threshold_g == 0.5
light_contact.setValue(0.8)
assert task.light_contact_threshold_g == 0.8
motion.setCurrentIndex(motion.findData("velocity"))
assert task.motion_mode == "velocity" and hidden(loading_step) and hidden(light_contact)
motion.setCurrentIndex(motion.findData("stepwise"))
mode.setCurrentIndex(mode.findData("sliding"))
assert not hidden(panel.findChild(QDoubleSpinBox, "sliding_step_mm"))
mode.setCurrentIndex(mode.findData("standard"))
assert hidden(panel.findChild(QDoubleSpinBox, "sliding_step_mm"))
profile_gaps = panel.findChild(QLineEdit, "profileGaps")
assert hidden(profile_gaps)
mode.setCurrentIndex(mode.findData("proximity_profile"))
assert not hidden(profile_gaps) and not force.isEnabled() and repeat.value() == 1
assert not hidden(light_contact)
# Settings the profile does not use are hidden, with their labels; values are kept.
assert base_speed.isHidden() and main_form.labelForField(base_speed).isHidden() and force.isHidden()
assert hidden(panel.findChild(QDoubleSpinBox, "contact_hold_sec"))
assert hidden(panel.findChild(QDoubleSpinBox, "far_field_speed_mm_s"))
assert hidden(panel.findChild(QCheckBox, "contactAreaConfirmed"))
assert hidden(panel.findChild(QComboBox, "motionMode"))
profile_gaps.setText("50, 10, 5, 2, 1")
assert task.profile_gaps_mm == [50, 10, 5, 2, 1]
panel.findChild(QSpinBox, "profile_sweeps").setValue(4)
assert task.profile_sweeps == 4
mode.setCurrentIndex(mode.findData("standard"))
assert hidden(profile_gaps) and force.isEnabled()
assert not base_speed.isHidden() and not main_form.labelForField(base_speed).isHidden() and not force.isHidden()
panel.findChild(QDoubleSpinBox, "far_field_speed_mm_s").setValue(0.8)
panel.findChild(QDoubleSpinBox, "slow_zone_mm").setValue(8)
assert task.far_field_speed_mm_s == 0.8 and task.slow_zone_mm == 8
panel.close()

full = FullTaxelCalibrationTask(output_directory=None)
full_panel = build_protocol_panel(full, QDoubleSpinBox(), QSpinBox(), allow_profile=False)
full_mode = full_panel.findChild(QComboBox, "characterisationProtocol")
assert full_mode.findData("proximity_profile") == -1 and full_mode.findData("standard") >= 0
assert full_mode.findData("proximity") == -1 and full_mode.count() == 6
full_panel.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

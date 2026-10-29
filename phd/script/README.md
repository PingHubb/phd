# Standalone tools

Edit implementations in the grouped subdirectories below. The flat `.py` files
beside this README are compatibility entry points, not a second implementation.
They preserve existing imports, private helper access, direct-file commands,
ROS console commands and UI training buttons.

## Browse without starting anything

```bash
python3 -m phd.script list
python3 -m phd.script paths
```

After rebuilding ROS, `ros2 run phd pinglab_tools list` provides the same index.
`paths` reports resource locations for the package actually imported in that
shell; it does not move data or change environment variables.

## Grouped implementations

### Training

- [train_ai_direct_finger_motion.py](training/train_ai_direct_finger_motion.py): Train a recorded tactile control policy.
- [train_intention.py](training/train_intention.py): Check/train A/B/C goal-intention models.
- [train_robot_dog_gestures.py](training/train_robot_dog_gestures.py): Compare gesture classifiers from whole trials.
- [train_tactile_proximity.py](training/train_tactile_proximity.py): Train a tactile proximity/anomaly model.
- [train_touch_tracking.py](training/train_touch_tracking.py): Check/train camera-free or independently referenced tracking.
- [train_touch_gesture_policy.py](training/train_touch_gesture_policy.py): Train a six-gesture CNN-TCN for live Touch Policy.

### Analysis

- [analyse_height_profile.py](analysis/analyse_height_profile.py): Summarise a saved calibration height profile.
- [analyse_proximity_profile.py](analysis/analyse_proximity_profile.py): Analyse a saved proximity-profile trial.
- [benchmark_ai_direct_finger_motion_robustness.py](analysis/benchmark_ai_direct_finger_motion_robustness.py): Offline robustness replay of recorded tactile policies.
- [benchmark_robot_dog_gestures.py](analysis/benchmark_robot_dog_gestures.py): Run gesture-classifier benchmark studies.
- [check_ai_direct_finger_motion_dataset.py](analysis/check_ai_direct_finger_motion_dataset.py): Check a recorded dataset before policy training.
- [replay_touch_gesture_policy.py](analysis/replay_touch_gesture_policy.py): Replay the actual touch-to-velocity adapter without hardware.

### Inference

- [predict_robot_dog_gesture.py](inference/predict_robot_dog_gesture.py): Predict one saved gesture trial without robot commands.

### Conversion

- [compact_full_taxel_recording.py](conversion/compact_full_taxel_recording.py): Preview storage reduction for one stopped Full Taxel Calibration run. `--apply` permanently removes other-taxel columns after verifying every retained sample; keeps metadata backups and a verification report.
- [iges_xyz_to_csv.py](conversion/iges_xyz_to_csv.py): Extract XYZ points from an IGES file.
- [igs_to_csv_freecad.py](conversion/igs_to_csv_freecad.py): Convert an IGES path with FreeCAD (separate dependency).

### Hardware

- [rh56f1_hand_menu.py](hardware/rh56f1_hand_menu.py): Interactive ROS hand control; can command physical hardware.
- [setup_goodix_usb_permissions.py](hardware/setup_goodix_usb_permissions.py): Install USB permissions; intentionally modifies system udev rules.

## Examples

```bash
python3 -m phd.script.training.train_touch_tracking check /path/to/trials --supervision camera_free
python3 -m phd.script.training.train_intention check /path/to/intention_dataset
python3 -m phd.script.analysis.analyse_height_profile /path/to/trial.csv --help
python3 -m phd.script.inference.predict_robot_dog_gesture --help
python3 -m phd.script.conversion.iges_xyz_to_csv --help
```

You can also dispatch by name:

```bash
python3 -m phd.script train_touch_tracking --help
```

The old `python3 -m phd.script.train_touch_tracking ...` and
`python3 phd/script/train_touch_tracking.py ...` commands still work.
Existing ROS command names (such as `train_touch_tracking`,
`train_ai_dfm_model`, `check_ai_dfm_dataset` and `analyse_height_profile`)
are unchanged.

Training and analysis retain the original algorithms, default resource roots,
output rules and dataset schemas. A tool that writes reports/trains models still
writes when explicitly run; the folder organisation does not make those tools
read-only. Hardware utilities still need their normal ROS/FreeCAD/system
prerequisites and can act on real devices or system configuration. They were
not executed for this cleanup. Use the catalogue's hardware `--help` route for
a safe static description; the legacy direct hardware scripts do not implement
argparse help.

[Project layout](../../docs/PROJECT_LAYOUT.md) · [Workflow guides](../../docs/README.md)


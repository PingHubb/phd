"""Lightweight command index; listing tools imports no device or ML modules."""

TOOLS = {
    "analyse_height_profile": ("analysis", "Summarise a saved calibration height profile."),
    "analyse_proximity_profile": ("analysis", "Analyse a saved proximity-profile trial."),
    "benchmark_ai_direct_finger_motion_robustness": ("analysis", "Offline robustness replay of recorded tactile policies."),
    "benchmark_robot_dog_gestures": ("analysis", "Run gesture-classifier benchmark studies."),
    "check_ai_direct_finger_motion_dataset": ("analysis", "Check a recorded dataset before policy training."),
    "compact_full_taxel_recording": ("conversion", "Verify and permanently remove non-target taxel columns from a stopped run (--apply required)."),
    "iges_xyz_to_csv": ("conversion", "Extract XYZ points from an IGES file."),
    "igs_to_csv_freecad": ("conversion", "Convert an IGES path with FreeCAD (separate dependency)."),
    "predict_robot_dog_gesture": ("inference", "Predict one saved gesture trial without robot commands."),
    "replay_touch_gesture_policy": ("analysis", "Replay a six-gesture policy with a synthetic sensor and no robot."),
    "rh56f1_hand_menu": ("hardware", "Interactive ROS hand control; can command physical hardware."),
    "setup_goodix_usb_permissions": ("hardware", "Install USB permissions; intentionally modifies system udev rules."),
    "train_ai_direct_finger_motion": ("training", "Train a recorded tactile control policy."),
    "train_intention": ("training", "Check/train A/B/C goal-intention models."),
    "train_robot_dog_gestures": ("training", "Compare gesture classifiers from whole trials."),
    "train_tactile_proximity": ("training", "Train a tactile proximity/anomaly model."),
    "train_touch_tracking": ("training", "Check/train camera-free or independently referenced tracking."),
    "train_touch_gesture_policy": ("training", "Train the six-gesture CNN-TCN touch-to-velocity policy."),
}


def tool_module(name):
    """Return a canonical module path for a known command."""
    group, _description = TOOLS[name]
    return f"phd.script.{group}.{name}"


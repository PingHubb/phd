# Mutable device and experiment settings

- Sensor registration/order: `humanoid_sensor_mappings.json`,
  `robot_sensor_mapping.json`, `sensor_reorder_logic.json`, `sensor_zero_masks.json`.
- Robot/control presets: `robot_position_presets.json`, `console_control.json`,
  `direct_finger_motion.json`, `tool_pose_target_rotation.json`.
- Experiment poses: `taxel_position_mapping_anchor.json`,
  `calibration_initial_position.json`. Full Taxel Calibration takes its home
  from the loaded map's `anchor_pose`; the legacy
  `full_taxel_calibration_initial_position.json` is preserved but no longer used.
- Taught scan points: `sigraph2026_scanning_points.json`; dated backup files
  remain beside it because they are recovery records, not duplicates to delete.
- Hand settings: `dexterous_hand_pose_presets.json`,
  `dexterous_hand_sensor_adjustments.json`, `dexterous_hand_sensor_regions.json`.

These filenames/locations were not changed. The UI can update them while
running. Back them up before changing a setup, and revalidate remembered robot
poses after any physical relocation. Do not treat a saved anchor as a measured
guarantee of collision-free clearance.

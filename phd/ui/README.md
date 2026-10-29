# UI code map

Stable module paths are intentionally retained. This is a responsibility map,
not a claim that every large UI module has already been split/refactored.

## App composition and navigation

- `ui_initial.py`: main window, setup forms, experiment lifecycle and actions.
- `ui_ping.py`: device/control widgets and main UI composition.
- `ui_ping_ui_interactions.py`: wiring and general interaction handlers.
- `components.py`, `theme.py`, `icons.py`, `motion.py`, `nav_tabs.py`,
  `combo_popup.py`: shared visual components, theme and interaction primitives.

## Experiments and analysis

- `experiment_tasks.py`: task state machines, motion guards and recording.
- `experiment_run_view.py`, `experiment_quality.py`: run monitor and quality helpers.
- `mapping_safety.py`, `taxel_position_navigation.py`: mapping bounds/recovery
  and monitored navigation to saved taxels.
- `mapping_reference_dialog.py`, `mapping_repeatability_dialog.py`: map checks.
- `calibration_protocol_panel.py`: calibration protocol settings.
- `calibration_analysis.py`, `calibration_graph_analysis.py`,
  `calibration_result_dialog.py`, `calibration_comparison.py`: saved graph
  selection, derived measurements and cross-taxel comparison.
- `characterisation_analysis.py`, `sensor_characterisation_window.py`,
  `proximity_profile.py`, `proximity_profile_dialog.py`: characterisation and
  proximity-profile analysis.

## AI / human interaction

- `ui_ping_ai_controls.py`, `ui_ping_direct_finger_motion.py`: collection,
  policy execution and direct-motion controls.
- `touch_tracking_window.py`, `intention_window.py`,
  `gesture_recognition_panel.py`: tracking, goal-intention and gesture workflows.

## Sensors, robots and peripheral views

- `ui_ping_robot_sensor_controls.py`, `ui_ping_robot_dog.py`: robot controls.
- `ui_ping_camera_control.py`, `camera_preview_window.py`: camera integration.
- `humanoid_viewer.py`, `sensor_shape_editor.py`, `sensor_zero_mask_window.py`:
  model/sensor display and editing.
- `force_meter_chart.py`, `heatmap_signal_recording.py`,
  `plotter_video_recorder.py`: charts and visual capture.

[Project layout](../../docs/PROJECT_LAYOUT.md) · [Workflow guides](../../docs/README.md)

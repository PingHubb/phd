# Test map

Test files remain together: several simulated experiment tests share fixtures
by their existing module names. Moving them into arbitrary new subfolders
would risk changing discovery/imports without improving runtime behaviour.

## Find tests by area

- `test_mapping_*`, `test_experiment_tasks.py`: simulated robot mapping,
  clearance, retries and experiment state machines.
- `test_calibration_*`, `test_characterisation_*`, `test_height_profile.py`,
  `test_proximity_profile.py`, `test_sensor_characterisation.py`: measurement,
  graph analysis and characterisation workflows.
- `test_touch_tracking*`, `test_intention_learning.py`,
  `test_robot_dog_gesture_model.py`, `test_live_gesture_recognition.py`,
  `test_train_tactile_proximity.py`, `test_control_safety_and_ai_evaluation.py`:
  data/model/preprocessing and evaluation checks.
- `test_sensor_*`, `test_humanoid_*`, `test_robot_*`: device interfaces,
  geometry, layouts and fake-device behaviour.
- `test_ui_*`, `test_*workspace*`, navigation/status/window/layout tests:
  Qt presentation, controls and interaction state.
- `test_project_organization.py`: old/new tool entry points, path resolution,
  catalogue safety, resource aliases and documentation navigation.
- `test_flake8.py`, `test_pep257.py`, `test_copyright.py`: separate style/license
  checks inherited from the ROS package template.

## Run

From the repository root:

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest -q test \
  --ignore=test/test_flake8.py --ignore=test/test_pep257.py \
  --ignore=test/test_copyright.py
```

For one area, specify files or use `-k mapping`, `-k calibration`, etc.
The excluded checks can be run separately; a functional-suite pass is not a
claim that repository-wide formatting/license checks pass.

Experiment tests use fake robot APIs. Passing them does not certify a physical
air gap, collision-free workspace or safe unattended robot operation. Do not
substitute a live robot/device API into a test fixture.

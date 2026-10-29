# Device, data and model code map

Import paths remain stable. Do not rename `dependence/` or these modules without
updating and testing all UI, training and saved-model callers.

## Sensor acquisition and geometry

- `sensor_api.py`, `sensor_protocol.py`, `sensor_serial.py`,
  `goodix_usb_sensor.py`: transport/protocol implementations.
- `sensor_data.py`, `sensor_layout.py`: native frames, averaging and indexing.
- `sensor_geometry.py`, `sensor_heatmap.py`, `func_sensor.py`,
  `sensor_signal_window.py`: sensor geometry and signal display.
- `humanoid_sensor_registry.py`, `humanoid_signal_parts.py`, `humanoid_urdf.py`:
  model/sensor association and humanoid support.

## Robot and peripheral interfaces

- `robot_api.py`: ROS robot API and velocity helpers.
- `force_meter_api.py`, `camera_api.py`, `gripper_api.py`: peripheral interfaces.
- `func_meshLab.py`: shared mesh/visualisation and related device integrations.
- `gesture/`: sensor/control policies, controller input and gesture recording.

## Learning and research data

- `tactile_models.py`, `tactile_proximity.py`: tactile-policy and proximity models.
- `touch_tracking_data.py`, `touch_tracking_model.py`,
  `touch_tracking_camera_free.py`: tracking schemas, models and camera-free targets.
- `intention_data.py`, `intention_model.py`: A/B/C intention dataset/model.
- `robot_dog_gesture_model.py`, `live_gesture_recognition.py`: gesture
  preprocessing, classifier bundles and live recognition.
- `characterisation_study.py`, `height_profile.py`: measurement-study data and
  height-profile calculations.

## Shared paths and assets

- `paths.py`: canonical resource/experiment/AI path helpers and overrides.
- `embedded_toolbar_icons.py`: embedded toolbar artwork.

Training/analysis command implementations are under [script](../script/README.md),
not this directory. Data and model files belong under [resource](../resource/README.md).

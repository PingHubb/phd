# PingLab Tactile Robot Interface

PingLab is a ROS 2 application for tactile-sensor acquisition, 2D/3D signal
visualization, learned and rule-based robot control, dexterous-hand control,
and experiment recording. The desktop interface is built with PyQt5 and
PyVista.

## Run

From a sourced ROS 2 workspace:

```bash
ros2 run phd phd_ui
```

The primary sensor controller currently uses the `readRaw` serial protocol at
9600 baud. Select one serial device in the Sensor tab before building or
updating a sensor scene.

## Development

Run the package tests from this repository:

```bash
python3 -m pytest -q test
```

Build only this ROS package and its required dependencies:

```bash
colcon build --packages-up-to phd
```

Important modules are grouped by responsibility:

- `phd/dependence/sensor_api.py`: synchronous sensor command API.
- `phd/dependence/sensor_protocol.py`: serial protocol defaults and parsing.
- `phd/dependence/sensor_serial.py`: non-blocking Qt sensor reader.
- `phd/dependence/sensor_data.py`: calibrated frames and rolling averages.
- `phd/dependence/func_sensor.py`: sensor scene and live visualization.
- `phd/dependence/robot_api.py`: ROS robot commands and velocity safety.
- `phd/dependence/gesture/`: rule-based and learned control policies.
- `phd/ui/`: Qt views, controls, and experiment workflows.

Detailed sensor row/column conventions, AI compatibility constraints, and
known legacy layout issues are documented in
[`docs/SENSOR_DATA_LAYOUT_AND_COMPATIBILITY.md`](../docs/SENSOR_DATA_LAYOUT_AND_COMPATIBILITY.md).

## External AI assets

Large AI datasets and checkpoints should live outside the Python package.  By
default the app still uses `phd/resource/ai`, but you can point it at an
external asset folder:

```bash
export PINGLAB_AI_RESOURCE_ROOT=/home/ping2/phd_assets/ai
```

Expected layout:

```text
/home/ping2/phd_assets/ai/
  data/
  models/
```

You can also point at a full resource mirror; the app will use its `ai`
subfolder:

```bash
export PINGLAB_RESOURCE_ROOT=/home/ping2/phd_assets
```

`PINGLAB_AI_RESOURCE_ROOT` takes priority when both variables are set.

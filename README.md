# PingLab Tactile Robot Interface

PingLab is a ROS 2 application for tactile-sensor acquisition, 2D/3D signal
visualization, learned and rule-based robot control, dexterous-hand control,
and experiment recording. The desktop interface is built with PyQt5 and
PyVista.

## Find things quickly

- [Project layout and file ownership](docs/PROJECT_LAYOUT.md): where code,
  settings, recordings, models and papers belong.
- [All workflow guides](docs/README.md): sensor conventions, touch tracking,
  intention learning, gesture recognition and sensor characterisation.
- [Standalone tools](phd/script/README.md): training, analysis, inference,
  hardware utilities and CAD conversion, grouped by purpose.
- [Runtime UI map](phd/ui/README.md) and [device/model code map](phd/dependence/README.md).
- Experiment results: `phd/resource/experiment/`; datasets and models:
  `phd/resource/ai/`; papers: `phd/resource/paper/`.
- Historical material and recoverable originals: [archive](archive/README.md).

The UI, sensor/robot control modules, dataset names, model names and device
configuration paths are deliberately unchanged. The old flat script paths
forward to the grouped implementations; existing UI buttons, imports, CLI
commands and direct-script launches remain supported.

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
[`SENSOR_DATA_LAYOUT_AND_COMPATIBILITY.md`](docs/guides/SENSOR_DATA_LAYOUT_AND_COMPATIBILITY.md).

## Desktop workspace

The desktop uses a shared slate/teal instrument theme with a dark sensor canvas.
The full-width status strip describes the actual sensor stream, active control,
model and force meter. The familiar Sensor, Robots, Control, AI, Hand and Tools
tabs retain their behaviour. Main navigation stays in a fixed full-width bar,
outside the adaptive sensor/control split; selecting a workspace also reveals
its controls from Focus sensor mode. The Layout menu is accessible beside it.

Motion is deliberately short: the navigation highlight and ordinary task-page
reveal take 140 ms, disclosure sections 150 ms and action hover outlines 100 ms.
Selections and commands take effect immediately; live sensor/3D rendering and
workspace splitter widths are not animated. **View → Reduce Motion** disables
these effects immediately and remembers the preference between launches.

- Experiments, Sensor Signal and Camera have labelled toolbar shortcuts on wide
  windows. On smaller screens they become icons with tooltips; Stop All Motion,
  Disconnect and Calibrate remain visible. Controller tests remain in Menu.
- With no scene loaded, the canvas offers Sensor setup and Touch tracking
  shortcuts. These only open interfaces; they do not connect hardware or move robots.
- Sensor geometry uses a compact dropdown. It shares the existing selection
  model, so geometry indices and saved settings are unchanged.
- Each dense workspace has a persistent task selector: unrelated controls are
  shown on separate pages instead of one long form. Switching pages preserves
  values and does not start or stop devices, controllers or recordings.
- Main navigation uses filled selections, subtabs use an underline and task
  selectors use a soft highlight. Task selectors reflow with available width;
  forms stay bounded on wide displays. Duplicate workspace headings are hidden.
- Sensor separates Setup, Display and Diagnostics. TM Robot separates Read state
  from Move / send. Control separates touch/proximity, controller input, tool paths
  and pressure response. Tools separates Force meter, Gripper and Camera control.
- AI opens on Run. AI → Data opens on Gesture recording: touch trials for the
  touch policy, started on touch by default, with no robot commands. The session
  folder name is the gesture label. Touch tracking, proximity background and
  intention learning have separate pages. Space only toggles gesture recording
  while AI → Data → Gesture recording is selected; Escape remains available to
  stop it. Opening a tracking page never starts capture or inference.
- Touch tracking's Collect, Train and Preview actions have separate responsive
  cards with short descriptions; the buttons still open the original workbench
  pages without starting an action automatically.
- Advanced hardware, policy and legacy options start collapsed under descriptive
  headings. Hand motion separates Whole hand, Individual fingers and Saved poses.
- Recording setup and training forms scroll on short displays; touch-state
  marking and Stop/Save remain visible while collecting. Experiments, camera,
  signal inspection and logs remain separate utility windows.
- Touch tracking separates trial settings from participant/session identity.
  Optional sensor dimensions and the collection checklist expand on demand.
  Training parameters have their own disclosure, with Check, Train, Cancel
  and the output log outside the scrolling setup area.
- Camera keeps View, Save frame and Reconnect in its main toolbar. Exposure,
  white balance and emitter controls are under Image adjustments, leaving
  more room for the preview. The preview also fits smaller windows.
- **View → Workspace Layout** defaults to **Auto · follow task**. AI → Data,
  Hand → Motion controls and Tools use the full width for controls; other tasks
  keep the sensor/control split. Split view, Focus controls and Focus sensor
  override Auto until you select it again. Dragging the divider selects a manual
  split, and your chosen control width is restored after a focused view. These
  choices hide containers only: they do not disconnect streams or rebuild scenes.
  Returning from a full-width page restores the previous split width in one
  update; responsive task selectors do not impose their old expanded width.
- Experiments separates **Setup** from a read-only **Run** monitor. Starting a
  timed task opens Run once; you can return to Setup without being pulled back.
  Stop Experiment and the existing software Stop All Motion action remain outside
  both pages, separated visually to clarify their scope. Task descriptions can
  be expanded with About. Single-taxel calibration keeps its settings visible
  first; expand Choose a target on the taxel map to select a cell. Loading a map
  opens that selector automatically. Unavailable cells stay readable but disabled.
- Taxel Position Mapping separates Map / anchor, Motion / safety, Signal
  detection and XY search. The live map remains on the first page and mapping
  progress stays below every page. Full-taxel calibration's map and sequence
  explanation can be expanded without pushing the settings down by default.
  These are presentation changes only; switching pages does not command motion
  or alter the sampling, force limits, saved map or experiment settings.
  Fixed-threshold and legacy-margin fields appear only for their respective
  detection modes; switching modes preserves their stored values.
- Run shows the task phase, existing trial/taxel progress, elapsed time including
  pauses, raw force in g with sample age/staleness, calibration TCP-Z travel and
  the output location when reported by the task. Z travel is downward displacement
  from this trial's start, not total path length. Unsupported measurements show
  unavailable rather than a fabricated zero. The last snapshot stays after the
  run ends; “Run ended” and an output path are not claims of successful completion
  or complete data saving. Detailed results still belong in Graph Analysis/logs.

Restart the desktop UI to load presentation changes. The software stop does
not replace the physical emergency stop. Heatmap meanings, calibration values,
recording schemas and model calculations are not changed by the theme.

## External AI assets

For continuous finger-position and speed training, open **AI → Data → Open Touch
Tracking workspace**. Camera-free collection and immediate direct tracking are
now the default; optional independent camera/CSV references remain available.
Sensor-estimated training targets are explicitly distinguished from physical
ground truth. The recorder, model comparison and sensor-only preview
are described in [the Touch Tracking guide](docs/guides/TOUCH_TRACKING_WORKFLOW.md).
Existing imitation-policy recordings/models remain separate and unchanged.

For the new human-goal task, open **AI → Data → Intention learning**. Collect
camera-free, human-declared **A/B/C surface-target** trials, compare a causal
touch-plus-context model against touch-only/context-only baselines, and run a
sensor-only intention preview. This is a separate research prototype, not robot
control or a novelty/performance claim. Read [the Intention Learning guide](docs/guides/INTENTION_LEARNING.md)
for physical target setup, collection examples, held-out evaluation and limits.

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

# PingLab Sensor Viewer — Windows test build

A separate sensor-only app. No ROS, robot, force meter, experiment, camera, AI
or research recordings are included. The original research project is unchanged.

## Try it on Windows

1. Copy `PingLab-Sensor-Viewer-Windows-x64-3D-v0.2.zip` to a Windows 10/11 **64-bit Intel/AMD** PC.
2. Right-click the ZIP → **Extract All**. Do not run the EXE from inside the ZIP.
3. Open `PingLabSensorViewer.exe` in the extracted folder. Keep `app` and `runtime`
   beside it. No Python installation, ROS, administrator access or internet is needed.
4. Try **Demo** first. Synthetic data is clearly marked and never opens a serial port.
5. Plug in the sensor's USB controller. Close Arduino Serial Monitor and any other
   program holding the COM port. Click **Refresh ports**, select its COM port,
   choose the grid (Auto, 5×5, 8×10, 20×20 or Custom), then **Connect sensor**.
6. Leave the sensor untouched and press **Update Calibration**. The default view
   is now **3D Point Grid**. Click a node to inspect its signed signal.
7. **Open 3D Plotter** opens a larger interactive sensor visualization;
   **Open Signal Viewer** opens the selected taxel's time-history graph.

## New in 0.2 — interactive sensor plotter

- **3D Point Grid**: connected taxels move above/below their reference plane with
  positive/negative signal changes, like the research UI's point-grid concept.
- **3D Surface**: neighbouring measured nodes form a coloured response surface.
  Faces between nodes are visual interpolation, not extra measured taxels.
- **3D Heatmap**: a flat coloured sensor grid that you can rotate in 3D.
- **2D Heatmap**: the original top-down view remains available in the dropdown.
- Drag to rotate, use the mouse wheel to zoom, and Shift-drag to pan. **Top** gives
  the same index orientation as the 2D map; **Reset view** or **R** restores the camera.
- Click a node or use arrow keys to select it. Selection is shared by the main
  plotter, pop-out plotter, heatmap and signal graph. **Height** changes display
  exaggeration only. Taxel IDs are shown for grids up to 100 cells; larger grids
  use hover tooltips and the selected-taxel readout to avoid unreadable labels.
- Difference views use the existing fixed ±range; values beyond it clip visually
  but their full measured numbers remain in the readout, graph and recording.
  Raw/baseline heights use a relative min-to-max scale, not absolute displacement.
- The plotter is software-rendered with the existing Qt runtime, without a new
  OpenGL/VTK dependency. No new USB protocol or firmware changes are needed.

**Important:** Height is a signal visualization, not measured indentation,
proximity distance or force. This build uses a regular flat taxel layout;
it does not import the research application's curved/OBJ meshes, edited geometry,
mask profiles, Stereo Field effect or its exact rendering/smoothing pipeline.
Changing views, gain or camera does not change the acquired or recorded data.

This build supports the project's **serial/Arduino/ESP32 tactile firmware** with
`channelCheck`, `readRaw`, `readCal` and `updateCal`. It does not support direct
Goodix USB/HID controllers, Bluetooth robot connections or raw I²C devices.
No firmware upload is required when your controller already uses this protocol.
Your USB-to-serial adapter may need its manufacturer's Windows driver.

## Layout and numbers

- Taxel index = column × number of rows + row. Index 0 is top-left; IDs run down
  each column, matching the standalone Sensor Signal Viewer numbering.
- Auto grid assumes the firmware's `channelDrive` includes one spare/reference
  column, and `channelSensor` is the row count. If incorrect, choose the known
  physical grid manually. Optional trailing reference data is excluded.
- Signed difference = raw − baseline. Blue means negative; orange means positive.
  These are sensor units, not calibrated force or physical capacitance units.
- Percent difference = 100 × (raw − baseline) / baseline; a zero baseline gives
  an undefined (NaN) percentage, not a made-up zero.
- The heatmap uses a fixed ±colour range for differences. Raw/baseline auto-scale.
- No smoothing is applied. Graph history holds up to 3,000 recent frames;
  re-zeroing clears the displayed history to avoid joining different baselines.
- Hover over a graph for timestamp/value; wheel changes its 2–120 second window.
  **Freeze graph** does not stop acquisition or recording.
- No geometry transforms, sensor masks or custom reorder profiles from the main
  research UI are loaded. Confirm orientation with a known corner before use.

## Recording

**Record CSV** saves all taxels' raw and signed-difference values for every frame
delivered by the reader, plus UTC time, elapsed time, frame sequence and baseline
version. A companion JSON stores shape, port, baseline history and dropped-frame
count. Recording stops cleanly on disconnect/close. Existing files are never
overwritten. Demo recordings are marked synthetic in metadata and filename.

Disconnecting retains the last display but labels it OFFLINE. Five consecutive
missing frames end a connection; the app never silently reconnects to another
device. UI backlog is bounded and any dropped frames are shown/counted.

## Packaging and test status

This is an **unsigned test build**, not an installed or certified release. The
native Windows x64 launcher is cross-compiled with Zig; it launches the official
CPython embeddable runtime and Windows wheels bundled in `runtime`.

The user confirmed the 0.1.0 Windows executable works. The updated 0.2.0 plotter
is tested with Linux/offscreen simulations; its actual Windows rendering and
real-sensor behaviour still require your test. The previous ZIP is preserved.
Do not disable antivirus or organizational security controls. If Windows blocks
the unsigned executable, follow your organization's approval process.

If startup fails, run **Diagnose.cmd**. Logs are at
`%LOCALAPPDATA%\PingLabSensorViewer\viewer.log`. Share that log and your COM port,
grid and baud setting when reporting an issue. Log files rotate at 2 MB.

## Source and dependencies

The source in this folder runs from the research checkout with
`python3 standalone/sensor_viewer/run.py --demo`. It reuses only `sensor_api.py`
and `sensor_protocol.py` from `phd/dependence`; production robot modules are not
imported. Runtime requirements are PyQt5, NumPy and pyserial.

The ZIP includes source and third-party licence notices. PyQt5 is GPLv3 (unless
separately commercially licensed), Qt has its own LGPL/GPL terms, NumPy is BSD,
pyserial is BSD and CPython uses the PSF licence. This is a private research test
package; review licensing before public/commercial distribution of the app.
The bundled components are unmodified and replaceable. This packaging does not
grant rights to distribute the user's research code.

Build inputs and SHA-256 hashes are listed in `BUILD_INFO.json`; `SHA256SUMS.txt`
lists each bundled file. The reproducible packager is `build_portable.py`.

# Sensor characterisation — updated workflow

[All guides](../README.md) · [Project layout](../PROJECT_LAYOUT.md)

## Before the next hardware run

The installed DYLY-106 is **1 kg rated**. Display range and overload survival
are not a calibrated measuring range. No code can verify the cell's condition,
the probe attachment, load alignment, emergency stop or free robot workspace.
Check the supplier's exact datasheet and verify zero plus known loads across
the intended working range before using results as a force reference. Do not
continue the old 2,000 g programme without a suitable higher-capacity reference.
Recorded above-range values are preserved, not certified accurate.

Restart PingLab to load these changes. Reload the position map before a new
calibration: newly prepared map targets use the saved **clearance/anchor height**
for a stationary baseline, not the near-surface mapping probe plane. Confirm
that height really is out of proximity range and that the sensor has not moved.
Existing JSON/CSV files and saved physical poses are not overwritten.

Current software precautions for the installed cell:

- Default target **500 g**; maximum target **800 g**.
- Independent stop at **900 g** of either raw or baseline-relative force.
- Slows to ¼ of the nominal approach speed above 80% of the target, including
  speed-comparison protocols. Return uses the trial's nominal speed.
- Default maximum downward travel **30 mm** and whole-trial timeout **600 s**.
  Set finite limits appropriate to the measured clearance and speed; timeout
  includes approach, hold, return, recovery and communications pauses.
- Stale (>1 s), disconnected or invalid tactile data, and stale/missing robot
  feedback, stop the task. Force dropouts stop motion and retain the existing
  stable-stream resume workflow; sliding aborts instead of auto-resuming.
- The emergency stop ends motion; it does **not** automatically retract after
  a safety fault. Inspect and recover the robot manually.

These are conservative software settings, not manufacturer-approved ratings or
guarantees against overshoot. Supervise initial validation with the physical
emergency stop accessible. Start below the default target if the setup requires it.

## Existing Calibration → Single taxel / Full taxel

Choose **Protocol** below the main settings; it is always visible. The panel
then shows only what that protocol uses:

- A **To set** line names the fields to fill in for this protocol (for the
  proximity profile also the expected duration). Main settings the protocol
  ignores are hidden: Base -Z speed and Max contact threshold for the proximity
  profile.
- Protocol inputs (speed list, force-level list, sliding path) appear beside
  the protocol choice; Motion is hidden for the proximity profile.
- **Advanced settings** (collapsed, with a count) holds settings whose
  defaults are fine: step sizes, light-contact threshold, far-field approach,
  hold/recovery, baseline frames and the proximity-profile plan.
- **Record details** (collapsed) holds specimen/session/mounting IDs, probe
  geometry and diameter, force-reference verification and notes; they go into
  the metadata and do not change the motion.
- **About this protocol** (collapsed) explains what the run does.

Hiding a setting never changes its value. Full Taxel does not offer the
proximity profile, which runs from Single taxel only. The existing Run buttons
and full-map pass order are unchanged.

The **Run** view includes a read-only taxel progress map for Single taxel and
Full taxel calibration. It follows the loaded sensor grid and the same
top-down, column-major numbering as Setup. The current taxel has a ▶ marker,
completed trials have ✓, waiting taxels have ·, and nodes excluded from the
run (including unmapped/skipped borders) have —. Pauses use Ⅱ. Labels, counts
and hover descriptions accompany the colours. Clicking the map has no action.
Full-taxel completion is counted **for the displayed pass**, based on recorded
trials marked complete, not merely visited taxels or partial recordings. A new
pass resets that view of completion. After stopping, the last snapshot stays
visible and the last unfinished taxel is marked !; cleanup cannot erase it.
The map is observational only and changes no movement, limits or recording.

After a completed trial is saved, **Full taxel** and repeating **Single taxel**
return control to the UI event loop before any next route or recalibration.
The robot remains stopped at the returned start height while **Waiting after
save** checks tactile, force and robot-position feedback. Each stream must
provide at least three new timestamps after saving, with all three streams
fresh for at least 0.25 s. The ordinary 1 s freshness, force and trial-time
guards are checked again before continuing. A real feedback failure has a
10 s bounded wait, then stops without movement and preserves the completed
trial. Pressing Stop cancels this continuation; it cannot restart itself.
The 900 g raw/baseline-relative emergency check remains active during the wait.
Analysis is computed once per save rather than twice; saving is still
synchronous, so the UI may briefly be busy while it writes the recording.
Batch/full-run JSON summaries now include the stop reason, message, phase
and time; full-run summaries also identify the last taxel and pass, even
when stopped before recording the first taxel.

Full Taxel uses the loaded position map's **safe anchor as its common home**.
Loading a new or relocated map updates that home automatically while idle;
there is no separate Remember Initial Position button. It visits the anchor
before the first taxel and returns there after a successful full run. Between
taxels it retracts vertically to each taxel's clearance pose before moving XY.
The old separate Full Taxel home file is preserved but ignored. Loading a map
does not move the robot, and identity/orientation/force checks still apply.

For **Loading / unloading**, the visible **Before running** section checks the
selected taxel (or deepest node of a full map) against Maximum travel. A map's
scan plane is a known non-contact height, **not** a measured surface. A limit
that cannot pass that plane is rejected before any connection/motion starts.
**Automatically plan travel and timeout from map** is enabled by default in
both calibration UI tabs for Loading / unloading. Loading a map (Full taxel)
or selecting a mapped taxel (Single taxel) updates those two limits while idle.
Under **Plan travel and time limits**, review the extra allowance below the
mapped plane (default 5 mm) and check physical clearance before pressing Run.
The displayed automatic limits are read-only; uncheck the automatic option to
edit them manually. Manual mode also retains **Apply suggested travel and
timeout** as a one-click action. Switching off automatic planning keeps the
current values for editing, rather than resetting them.

Run rechecks the automatic proposal using cached robot project speed, logs
the limits, and then freezes them for acquisition/force connection and the
whole batch. Automatic planning never starts motion, changes force/speed or
repetitions, learns a larger allowance from failed trials, or retries a stopped
experiment. A plan outside 100 mm / 7,200 s is blocked before force connection
or motion, without partial application or clipping. No map/selected target
means no automatic plan. Other protocols keep manual limits. Programmatic
task callers remain manual unless they explicitly enable `loading_auto_plan`.
The load-cell limits remain 800 g target / 900 g emergency stop.

The timeout suggestion is **per taxel/trial**, not the full batch. It models
far/near steps, the step/dwell loading rate, conservative small loading steps,
unloading, baseline, force-zero settling, hold, return and recovery, with 25%
time margin plus 60 s. The final 30 mm above the mapped plane are budgeted as
near steps; all extra travel below it is budgeted as loading/unloading. Robot
project speed comes from cached feedback when available, otherwise the panel
labels its 25% assumption. These are planning assumptions, not guarantees of
contact depth, stiffness, clearance or completion time. Runtime pauses can
exhaust the timeout. Suggestions above 100 mm / 7,200 s are not silently clipped
or applied. In manual mode, a short timeout shows a warning rather than treating an uncertain
estimate as a hard physical limit. Run a supervised, lower-force single-taxel
pilot before the full scan, then refine travel/time from the measured cycle.
The proximity-profile protocol and recordings are unchanged and are not used
to silently transfer one taxel's contact height to the whole sensor.

Downward step commands, including an optional far-field move, are now bounded
by Maximum travel before being sent. Reaching that bound before the force
target stops the task; no automatic force-limit increase or fault retraction
is introduced. The setup planning assumptions are recorded in trial metadata.

1. **Loading / unloading:** the normal force–signal–travel experiment.
2. **Hold & recovery:** preset 30 s fixed-position hold plus 30 s unloaded
   observation. The force can relax while position is fixed. Do not call this
   a constant-force creep experiment; that requires a validated force controller
   or suitable independent loading apparatus.
3. **Speed comparison:** editable list, initially 0.05, 0.1, 0.2 mm/s.
   Each repetition uses the next nominal speed. Full Taxel changes speed only
   after a whole pass. Six repetitions give two trials/passes per speed.
   Near-target slowdown is identical in all conditions and recorded in metadata.
4. **Drift / repeated cycling:** disables tactile recalibration between trials.
   Each trial still measures stationary noise and a force baseline. Raw values
   and per-frame calibration values preserve long-term tactile baseline changes.
5. **Spatial sliding:** supervised fixed-Z out-and-back in a chosen **robot-base**
   X/Y direction. Default 2 mm at 0.1 mm/s; allowed 0.1–5 mm and 0.01–0.5 mm/s;
   target must be ≤200 g. Check the entire path and tick the explicit preflight
   checkbox. The robot does not regulate normal force, measure shear or know
   the sensor boundary. Path deviation, lost contact and excessive normal force
   stop the run. This is a research acquisition option, not a validated safe
   sliding controller. First validate a short path at low load under supervision.
6. **Proximity profile (Single taxel only):** estimate the surface with light-contact
   probes, then collect stationary frames at specified gaps over repeated sweeps.
   Far-field reference revisits compensate baseline drift. Reports signal versus
   gap, noise-based detection ranges and repeatability; see the detailed section
   below. Contact-defined distance remains an estimate of air gap.
7. **Low-force / force-level sweep:** editable levels, initially 5, 10, 20,
   50, 100 g. Requires the force-reference verification checkbox after an actual
   known-load check. Retracts completely between levels. One repetition means
   one level, not a repetition of the whole list: 15 repetitions give three
   cycles of five levels. In Full Taxel, one **whole pass** uses one level before
   moving to the next level. These are target forces, not controlled constant
   forces; low-force resolution must be checked against actual load-cell noise.

The simpler **Proximity approach** option has been removed from both protocol
menus. Existing recordings with protocol `proximity` remain readable; their
metadata and interpretation are preserved.

Record the physical specimen, session/day, mounting/remount ID, probe geometry
and diameter in the new structured fields. A verified force reference plus a
known fully contacting flat circular tip permits a **nominal** force/area pressure
estimate. Finger/hemispherical tips and unknown contact areas do not. The loading
slope and first observed above-noise force are exploratory summaries, not proof
of linearity or a validated minimum detectable force/pressure.

Adjust hold, recovery, stationary baseline time/frames and baseline update policy
as needed. Protocol selection applies its displayed presets, so check the main
force and repetition fields afterwards. Use the notes field for sensor mounting,
probe dimensions/material, grounding, temperature, session and any changes.
Changing these notes deliberately separates conditions in automatic comparisons.

## What is recorded

One experiment still has one output folder. New trial JSON schema is version 8.

- All taxels: raw value, calibration reference, instantaneous signed difference,
  averaged signed difference and averaged relative difference. The chosen plot
  still starts on the physically pressed taxel.
- Acquisition, tactile-source, force-source and robot-feedback monotonic times;
  source ages; frame sequence; XYZ tool position.
- `elapsed_s` is real monotonic wall time, including pauses.
  `active_elapsed_s` separately excludes completed force pauses.
- Stationary baseline rows, loading, fixed-position hold, sliding (if chosen),
  return and unloaded recovery rows are labelled by phase.
- Interrupted/completed flags, pause events, protocol, nominal speed, slowdown,
  cell rating, force guard, baseline policy and setup notes.
- `contact_relative_travel_mm`: robot TCP displacement relative to detected
  contact. Negative is before estimated contact, positive after it. This is
  **not certified fabric compression**; fixture/load-cell compliance contributes.

Duplicate tactile frames are not logged repeatedly. Published tactile snapshots
keep raw/averaged/baseline matrices from one completed frame together. Acquisition
is still **UI polling at up to 20 Hz**, not a native-rate synchronised DAQ. Source
timestamps expose delay and missing samples; they do not remove it. Do not claim
millisecond response times from these files.

## Motion: stepwise position moves (default)

Calibration (single and full taxel, every protocol) and the Taxel Mapping
depth probe now move with **TM line (position) moves** by default
(`Motion → Stepwise position moves` in the protocol panel; mapping records
`probe_motion` in its settings). Live control modes (keyboard, controller,
gesture/DFM/proximity control) still stream velocity.

- **Approach**: 0.2 mm steps until the target signal has dropped by
  max(20 counts, 10 × start-pose noise), 0.05 mm steps until light contact, then
  **0.02 mm loading steps** (halved above 80 % of the target). With far-field
  memory enabled, one line move goes to the previous contact height plus the
  slow zone first.
- **Force zero and light contact**: the trial-start force zero shifts by up to
  ~1 g. On 22 Oct 2026 the start reading was 0.7 g while the meter was still
  recovering from a ~500 g press 17 min earlier; it decayed to 0.0 g over 8 min
  and the absolute change then read 0.7 g even 100 mm above the sensor. Cable
  drag when the robot moves adds to this. The far steps therefore follow slow
  drift (median of the last 5 at-rest readings), and before the near steps the
  robot holds still until the force is steady (2 s within 0.15 g, waiting up to
  15 s) and that reading becomes the zero for every later force in the trial.
  Light contact is **Light-contact threshold** (default 0.5 g) above it, or
  max(3σ, 2 × range) of the zero window if larger; an unsteady window raises
  the threshold and says so. Metadata `motion.force_zeros` records each zero;
  every row keeps `force_baseline_g` and `force_raw_g`.
- **Measurement at rest**: after each step the TCP must be still, 0.25 s of
  force-meter lag is ignored, and the step lasts at least the time at rest
  (default 0.5 s). Those rows are tagged `step_state=dwell`; onset, turning
  point, fits and hysteresis use only them.
- **Unloading**: 0.02 mm steps up until the force is below the contact
  threshold, then one line move to the start height.
- **Loading rate** = step ÷ step period. Base -Z speed (or the speed list) is the
  nominal rate; with 0.02 mm steps and 0.5 s at rest the fastest is about
  0.025 mm/s. Speed comparison rejects unattainable rates and presets
  0.005, 0.01, 0.02 mm/s. Metadata `motion` records the nominal and measured rate.
- **Sliding**: XY step-and-hold (default 0.05 mm) at fixed Z; the rate is again
  step ÷ step period. There is still no shear measurement.
- **Mapping depth probe**: each 0.5 mm step is a line move clamped to the same
  descent, neighbour-plane and scan-depth limits.

Legacy velocity motion remains selectable to reproduce older recordings; do
not mix the two in one comparison.

## Stopping in TM velocity mode

At low commanded speeds (≤0.1 mm/s, and to a lesser extent 0.5 mm/s) TM
velocity mode moves in **stick-slip** fashion: the TCP stalls or creeps the
wrong way for seconds, then jumps 0.35–0.6 mm at several mm/s. This is visible
in every recording since September (7–16 % of 2 s windows creep upwards while
commanded down), and the force rises on average 328 g in the 0.6 s after such
a jump versus 28 g otherwise, so it is real motion. Velocity-mode loading curves
therefore rise in steps; treat their fine force–distance detail and the
turning-point force with caution. The proximity profile uses position steps
instead.

An abrupt stop (Suspend/Stop/StopAndClearBuffer) while the robot moves in
velocity mode makes the next velocity command start in the opposite direction.
Normal-flow stops (contact, end of the +Z return, sliding path ends, profile
light contacts) therefore stream **zero velocity** every tick until the TCP has
been still for 0.5 s, and only then leave velocity mode or start the next
motion; the contact hold and return settle wait for that. Faults, Stop and the
emergency force limit still stop immediately. As a backstop the run stops if a
+Z return moves more than 0.1 mm down, or a commanded descent moves more than
0.5 mm up.

## Three calibration targets per trial

Every trial (single taxel, full taxel and offline Graph Analysis) now reports:

1. **Proximity range** — robot travel from the first sustained proximity
   response to the first sustained force change. If the response starts within
   1 mm of the start pose, or the signal already falls by more than the noise
   threshold over the first 2 mm, the start pose was already inside the range:
   the value is reported only as a **lower bound** (`start_inside_range`).
   Recalibrating at such a pose also hides part of the response. Raise the
   start pose until the status reads `measured`.
2. **Force–signal** — the old single line is kept, plus separate fits for the
   **drop branch** (contact → turning point) and the **rise branch** (turning
   point onwards). Use the rise branch as the monotonic force calibration.
3. **Drop-to-rise turning point** — the loading-signal minimum (zero-lag median
   smoothing, then the raw minimum inside its flat band), with its force, the
   force band within tolerance of the minimum, and travel after force contact.
   An unaveraged-signal cross-check shows averaging lag. If the signal has not
   started rising when the target is reached, raise the force target.

The force where the signal **returns to its no-touch baseline** is also
reported: below it, signal alone cannot tell light contact from no touch.
Batch Graph Analysis gives mean ± SD for all of these.

**Far-field approach speed** (protocol panel; 0 = off) lets you start far above
the surface without hour-long approaches. It is used only on repeat trials of
the same taxel in the same run, above that taxel's previously measured contact
height plus the **slow zone** (default 5 mm), and is dropped at the first 1 g
force change. Once the slow zone is entered the approach stays slow. Samples
record `approach_zone`.

Turning forces of a few grams are close to the 1 kg cell's noise. Check the
force reference with small known weights before quoting them.

## Proximity profile (precise proximity range and resolution)

The ramp-based proximity onset depends strongly on the detection threshold
far from the surface, where the signal changes by well under one noise sigma
per millimetre. Use **Protocol → Proximity profile (stepwise,
contact-referenced)** on the Single taxel tab for proximity claims.

1. **Surface reference.** The profile never uses velocity mode. From the start
   pose it steps down with TM line (position) moves and, after each step,
   waits until the TCP is still, ignores 0.25 s of force-meter lag and takes the
   median of fresh force samples. Steps are 0.2 mm until the target signal has
   dropped by max(20 counts, 10 × start-pose noise), then 0.05 mm until light
   contact. It then makes light-contact probes (default 3) from 0.3 mm above
   that contact in 0.01 mm steps; each probe's contact Z is interpolated
   between the last step below and the first step at the threshold. Overshoot
   is at most one step. A force of 300 g during the search stops the run.
   The force zero follows slow drift during the far steps and is re-taken at
   rest (as for stepwise calibration) before the near steps and **before every
   probe**, at the probe start; light contact is the Light-contact threshold
   (default 0.5 g) above that zero. Rows before a probe's zero are tagged
   `force_zero`, not `surface_probe`. Surface Z is the median TCP Z where force
   first stays above the probe's threshold. The report lists each probe's
   zero and threshold, the probe spread (SD), the zero-force extrapolation
   offset (fabric movement before the threshold) and, optionally, the surface
   shift at the end. Trials recorded before 23 Oct 2026 used one trial-start
   zero and a 1 g floor.
2. **Stationary levels.** The robot moves (gap-scaled closed-loop Z, within
   0.01 mm) to each gap above that surface, settles, and records N unique frames
   of every taxel. Default gaps: 100 (far reference), 70, 50, 45, 40, then
   every 3 mm to 22, every 2 mm to 10, then 8, 6, 5, 4, 3, 2.5 … 0.4 mm. The
   2–3 mm spacing over 10–40 mm is where the range edge lies on the 20×20
   sensor; wider spacing leaves the range to interpolation (the report warns).
3. **Drift control.** The far reference gap is revisited at the start and end
   of every sweep and every 6 levels (default). Each taxel's reference medians
   are interpolated in time and subtracted, so slow baseline drift does not
   appear as a distance effect. Sweeps alternate far→near and near→far by
   default (hysteresis); random order is available. Default 5 sweeps.
4. **Common-mode rejection.** Drift is not linear between references: on
   22 Oct 2026 (taxel 189, 20×20) predicting each reference from its
   neighbours missed by 3.0 counts RMS, comparable to the 3σ threshold. Most of
   it is shared by the whole sensor, so the *localized* signal subtracts, frame
   by frame, the median of taxels at least 6 cells from the target (fewer cells
   on small sensors, never below 3; none if fewer than 8 taxels qualify). On
   that run this cut the reference error to 1.1 counts RMS and the 5σ range
   spread over 3 sweeps from 21.5/20.8/27.2 mm to 22.3/22.4/23.5 mm. The
   localized range excludes coupling shared by the whole sensor (robot arm,
   tool); the target-only range includes it. Report the one that matches the
   claim and name it.

Results (`trial_XXXX_proximity_profile.{json,csv,png}`, `_proximity_levels.csv`):
signal drop versus gap (target and localized, mean ± SD), **detection distance
at 3σ and 5σ of single-frame noise** for the target and the localized signal
(interpolated between levels; every sweep separately with the Student-t 95% CI
of their mean; a bootstrap 95% CI over sweeps only with ≥5 sweeps), the gap up to
which the target is the strongest taxel (dominance ≥1 and ≥1.2), **distance
resolution** (noise ÷ local slope: single frame, window median and sweep
repeatability), neighbour dominance, reference drift and its leave-one-out
interpolation error. The figure's fourth panel shows the per-sweep ranges.
Warnings flag drift-limited target-only ranges, fewer than 5 sweeps and ranges
that fall between widely spaced levels. Re-analyse a saved trial with
`analyse_proximity_profile trial_0001.csv --k 3 5 --suffix _v2`; without
`--suffix` the earlier analysis files are replaced.

Set Timeout above the estimate shown in the protocol panel and Maximum travel
larger than the start-to-surface distance. Line moves run at the profile speed
limit scaled by the TM project speed override (10 mm/s at 25% is 2.5 mm/s), so
every far-reference visit costs more than a minute. The panel assumes 25% until
the robot reports its override; Run re-checks Timeout with the reported value
(10% if none) and includes the stepped descent within Maximum travel. With the
defaults at 25% expect about 53 min after the surface and up to about 65 min in
total with 70 mm Maximum travel.
Moves stop on an unexpected force at or above the light-contact stop, except
when moving up from a surface probe. Gaps are robot-frame distances from the
light-contact surface, not independent air-gap measurements, and proximity
depends on the probe: keep one grounded probe and record it in the notes.

## Proximity/contact analysis

Default stationary baseline: at least 100 unique frames and 5 seconds. A drifting
baseline or insufficient fresh samples aborts the trial before descent.
Mapping and calibration share the robust downward-shift calculation:
rolling median (7 frames), median centre, robust MAD noise, low-percentile
downward excursion and a noise allowance. Positive spikes cannot be proximity.
Offline calibration requires sustained fresh samples and detects contact from
stationary force noise, not a percentage of the eventual peak force. Mapping
uses stationary depth steps; calibration is a moving ramp, so confirmation
latency still differs and should be reported.

Holds are excluded from loading equations. Older recordings still open, but
their early-approach baseline is labelled exploratory. Missing old data cannot
be reconstructed, including neighbours that were never recorded.

## Graph Analysis

- **Open Graph Analysis…**, below the experiment task list, is always available,
  including during mapping, calibration and scanning. It opens the file picker
  and analysis in a separate offline process; it does not change the selected
  experiment, stop/restart its timer, connect to devices or replace its latest
  status. The task-list **Graph Analysis → Select Recording** action uses the
  same independent viewer. Closing that viewer does not stop the experiment.
  Prefer completed trial files; a CSV/JSON still being written may be incomplete
  and is a snapshot, not a live plot (reopen after saving finishes). Numeric
  worker threads are limited in the viewer, and it runs at reduced CPU priority
  where supported. Large analyses still share the computer's CPU/RAM, so avoid
  opening many large viewers during acquisition.
- On the full-sequence graph, **Amber: proximity onset**, **Red: contact onset**,
  **Blue: peak force**, **Purple: return begins**, **Teal: fast return**, and
  **Green: recovery** are drawn in the same
  colours as their event markers, using readable shades on the light background.
  The coloured legend wraps in smaller windows and is retained in PNG exports.
- **Complete Calibration Sequence / Full Sequence** distinguishes recorded
  **Slow unloading**, **Fast return**, and **Recovery (stationary)**. The single
  chart and side-by-side comparison use dotted event lines only, without
  shaded regions or start-time cards above the graph. Hover near a dotted line
  to snap the shared cursor to its exact event time. Small labels beside the
  single-chart intersections show time and curve values; comparisons show them
  in the readout below the graphs. There is no large descriptive hover popup.
  Readouts follow the displayed curves (interpolated, and smoothed where enabled),
  not necessarily an individual raw sample. Zoom in to
  separate tightly spaced markers. Overlay reports each trial's own stage in the hover readout
  (trials may enter return at different times). Zooming does not alter time or
  distance. Stage boundaries come from `phase` and `step_mode`, not an inferred
  force-zero/no-contact event. A `retract` stage includes arrival settling;
  legacy returns without a step-mode tag remain **Return (unspecified)**.
  Exported charts retain the dotted markers. Reports still list stage times.
  Future trial JSONs explicitly distinguish the nominal unloading rate from
  the fast-return **command** speed (currently 10 mm/s); acceleration and robot
  project speed can affect actual speed. Older files without this field report
  it as not recorded. Existing measurements, motion and safety settings are
  unchanged. Recovery is observation at the anchor, not a mid-trial recalibration.
- **Shared scales across nodes** is on by default. Time, travel, signed signal
  and force axes use common physical limits across all compatible loaded
  recordings (same sensor identity, grid and signal field), including recorded
  nodes not currently selected. Values are not peak-normalised or shifted.
  Uncheck for individual auto-scaling. Wheel/drag changes are temporary;
  **Reset View** restores the common limits. Contact-relative travel keeps
  negative pre-contact values and uses a conservative common travel envelope.
- **Compare Nodes…** pins the current node/trial in a separate offline window.
  Add up to six different node/trial pairs using that window's selectors, or
  select another node in the main window and press **Compare Nodes…** again.
  **Import Experiment…** adds saved CSV/JSON recordings (or a saved graph with
  its associated recording), including recordings from other sensors. A
  full-taxel run is indexed without loading every trial's samples. Choose
  **Sensor / experiment**, then the trial and node, and press **Add Node**.
  Existing pinned graphs stay open; selecting the same recording via its JSON,
  CSV or PNG does not duplicate it. Sensor name, run and probe information are
  shown with each pinned node; tooltips show source paths.
  Different sensor grids are allowed, but all pinned curves must use the same
  recorded signal field (for example, Difference cannot share axes with %).
  Node numbers are local to each sensor, and equal digital units do not imply
  equal sensitivity or matched test conditions. No normalisation is applied.
  Use **Side by side** for equal-sized plotting areas with linked axes, or
  **Overlay** for coloured node curves. Solid lines are approach, dashed lines
  are return. Hover shows a common x cursor and per-curve values; pan/zoom is
  linked. **Reset Shared View** restores common limits across the pinned
  nodes only; unrelated imported trials do not change the comparison scale.
  **Full Screen (F11)** uses the whole display, hides the setup controls and
  enables **Fit All Plots**, so the full-sequence panels fit in the viewport.
  Press **Esc**, **F11** or **Exit Full Screen** to restore the previous window,
  control visibility and fit setting without changing pinned data or zoom.
  **Hide/Show Controls** also works in the ordinary resizable/maximisable window.
  For many nodes, uncheck **Fit All Plots** for taller, scrollable graphs rather
  than shrinking every panel. Long hover readouts scroll independently and no
  longer push the graphs out of view.
  For older full-taxel recordings storing one node per trial, load
  `full_taxel_summary.json` to compare different physically probed nodes.
- Detail windows no longer replace each other. Open one node's selected graph,
  choose another node and open another: both keep their own data and the same
  initial axis limits. Their manual zoom is independent; the comparison window
  is the option for linked zoom. Equally sized detail windows have equal scales.
- **Export Comparison…** saves a PNG and a JSON sidecar with displayed limits,
  source paths, sensor/run/probe labels, node/trial identities and smoothing settings. Existing files
  must use a new output name. Faint curves retain raw samples; optional bold
  smoothing is display-only, not a fitted equation. The comparison window
  accepts legacy/incomplete trials for inspection; it does not certify them
  as matched-condition repeatability or force-reference validation. Physically
  probed and viewed nodes are labelled separately. Missing contact references
  are shown as missing, never inferred as zero.
- Existing graphs, phase-specific fits and distance-based hysteresis remain.
- **Contact-referenced Signal** adds the clearly labelled contact-relative TCP axis.
- **Quality & Characterisation** reports stationary noise, hold signal/force
  changes, unloaded recovery offset, full-grid response ratios at one common
  measured force instant, and held-out model errors.
- Comparisons group matching sensor/layout, pressed/viewed taxel, nominal speed,
  force target, hold, protocol, baseline policy and notes. They exclude interrupted,
  incomplete and legacy-unverified recordings by default.
  The immediate result window and automatic PNG exports follow the same policy;
  different speed conditions get separate repeatability files.
- Held-out equations use complete trials as validation folds, never random rows
  from the same trial. Evaluation is restricted to the training signal range.
- **Matched-force Variation** plots signal mean ± sample SD for eligible matching
  trials. The arrays and validation fold details are included in exported JSON.
  SD describes trial-to-trial variation, not a confidence interval.
- Spatial ratios include both electrical and mechanical spreading. A large ratio
  is not by itself a faulty taxel.

The ordinary R² on a fitted curve is still in-sample. Use the separate held-out
errors and repetitions for claims about prediction accuracy.

## Remembering a mapping anchor or scan plane

Both **Remember Current Pose Above Taxel …** and **Remember Current Height as
Scan Plane** ask for confirmation. Cancel (also the default keyboard action)
changes nothing. The prompt compares the saved and proposed setup. Remembering
an anchor captures TCP position and orientation; remembering the scan plane
captures only TCP Z, retaining the anchor orientation. Fresh robot feedback and
a matching connected sensor are required. Saving does not move the robot.

With a map loaded, confirmation creates and loads a separate
`taxel_position_map_revision_…_sensor-…/taxel_position_map.json`. Original JSON,
CSV measurements and completed calibration recordings are untouched. The new
JSON retains the measured geometry and stores the proposed setup separately in
`setup_revision.requested_setup`, with a parent path and SHA-256 for provenance.
Measurement CSV paths still reference the original recording, not duplicated
data. Keep the original recording folders.

Changing the anchor position/orientation or scan plane marks the revision
**verification required**. Retained taxel measurements are visible but cannot be
used for navigation, mapping Resume, or calibration with the changed setup yet.
Use **Relocate loaded map** to re-register and verify reference taxels, or start
a fresh mapping if probe changes require remeasuring every centre. These remain
explicit robot actions; saving a revision never starts them. A changed anchor
clears the old scan-plane choice: remember the intended plane again if needed.
Reopening the revision restores its proposed setup for the matching sensor.

Without a loaded map, confirmation saves only the setup for a new mapping run.
A staged Full Taxel Calibration resume keeps its original map snapshot; clear
that resume before using the Remember buttons to edit the mapping setup.

## Mapping repeatability

For the 20×20 sensor, the selected anchor is **Taxel 189** (zero-based display
row 9, column 9). Its four reference neighbours are 188, 190, 169 and 209.
The layout defaults use a **200 mm radial XY limit**, a 20 mm neighbour-search
radius and a 2 mm dense fallback step. The initial search first tries 4 mm
spacing (**121 positions rather than 441**) and falls back to the original
2 mm grid if reference signals are missing, a peak reaches the search boundary,
the lattice is unclear, or anchor-centre refinement/verification fails. Smaller
grids (169 points or fewer) use the full-resolution search directly. Sparse
spacing is also disabled if it would exceed twice the local scan half-width.
Local centre scans retain **0.5 mm steps**, but use **9 stationary readings per
point**, combined by median. Both local measurement lines are traversed in the
shortest endpoint order; their positions and measurement density are unchanged.
The reverse-order option reverses that route. Dense neighbour search uses at
least 20 readings per point. Baselines remain 100 frames, proximity samples
20 frames per depth level, and final centre verification 20 frames. Entering
this layout updates the controls; subsequent
refreshes preserve edits made in the UI. New 20×20 maps skip the 76 outermost
border nodes and scan the **18×18 interior: 324 nodes**. The anchor and its four
reference neighbours remain inside this region. Other sensor layouts keep their
own border policy.
Older loaded/resumed maps retain their anchor and measurement settings. When
resuming a 20×20 map, the requested border exclusion also applies to any
remaining work; previously recorded positions are preserved.

The workspace limit is Euclidean distance from the anchor, including corners.
It does not mean ±200 mm independently along both X and Y. The latter would
need a corner radius near 283 mm, beyond the current 200 mm navigation limit.
The initial coarse scan must encompass the anchor and all four reference nodes;
review its radius against actual node spacing before starting. A 400-node run
can take hours. Partial positions are written on a normal Stop or completion
and after each successful clearance/recalibration cycle. The JSON is replaced
atomically; it contains the authoritative measured positions. A green point
may appear before that cycle/checkpoint has finished.

### Stronger signal, non-contact intent

The working probe now seeks **2× the learned proximity-detection threshold**
by default, after preserving the original onset height. It still uses short Z
steps and 20 stationary readings per level. Near-sensor working probes use
the lower of the selected probe and confirmation speeds from the start
(default at most 0.5 mm/s); a lost candidate cannot restore fast search speed.
The first height-discovery probe slows after its first candidate and stays slow.
The onset threshold itself is not raised. The working probe uses the current
far-field batch, not a near-start baseline that could zero away proximity.

- Maximum/default extra descent after onset is **20 mm**, as requested by the
  operator. This is an allowance, not a required travel distance: descent stops
  once the 2× target is reached. Smaller saved limits remain smaller.
  This is not the only bound: working probes have an **onset-independent**
  boundary, so a missed detection cannot leave a 100 mm blind working search.
  A saved target response plane or the shallowest cardinal neighbour's plane
  permits only a 0.5 mm search margin below it. Without those planes, the
  learned anchor onset permits at most 5 mm discovery (or a smaller selected
  allowance). These planes are software search bounds, not surface/air-gap
  measurements; they can legitimately stop a weak or tilted-region search early.
  **20 mm below detected onset may reach or pass the physical surface.**
- There is **no weak-signal fallback**. If the requested stronger signal cannot
  be reached within the limits, the probe returns through the normal bounded
  failure/retry sequence and eventually stops; it does not scan at a weak plane
  or keep descending until contact. A strong sample beyond the depth limit is
  also rejected before XY scanning.
- A conservative **5 g change from the unloaded force reference** stops all
  motion as possible contact, before the existing 20 g guard. A smaller selected
  force guard remains effective. This stop does not automatically retract or
  retry: inspect the setup and air gap first. Noise/drift can cause a false stop.
  Resuming after a saved force/contact stop refuses automatic movement while
  below the clearance-anchor height: inspect and retract manually first.
- Force is referenced at the clear/safe pose; starting the lower working probe
  does not re-zero away force acquired during descent.
- Old maps without `centre_scan_policy=strong_noncontact_v1` upgrade a lower
  strength setting to 2× when resumed. Larger custom strengths are kept; old
  extra-descent limits are capped at 20 mm. Completed measured positions are not
  edited by loading/upgrading settings. The next save records the new policy.
  New position records also carry their own strength/limit/force-stop policy;
  retained old points are not retroactively certified under a new strength setting.

These are **best-effort non-contact precautions, not proof of clearance**.
The sensor's proximity onset is not a measured surface position, and very light
contact can be below the load cell's noise/resolution or occur before a force
stop. The software cannot guarantee that 20 mm below onset is still in air.
Supervise an initial low-speed pilot with independently checked physical gap;
reduce the descent limit if necessary. Do not run a blind deeper scan based only
on a force reading near zero. A shallow/nonuniform response may legitimately
fail to reach 2× within this limit; that is not permission to exceed it.

### Between-taxel clearance and full calibration

After every verified taxel, and after each recoverable attempt failure, the
task first lifts vertically to the saved clearance height, then returns in XY
to the remembered anchor. At that stationary pose it collects 100 unique
frames, checks stability and recovery against the earlier **same-calibration**
anchor batch, runs the full Update Calibration, waits for reconnection and
fresh data, and collects another 100-frame stable baseline before continuing.
Successful taxels are checkpointed before the next target. A failed target
gets one retry, using a bounded local-spacing correction when measured
neighbours support it; a second failure performs the clearance cycle then stops.

The remembered anchor must be independently established as out of proximity
range, with a clear return path. Stable capacitance does **not** prove absence
of proximity. Residual response or unstable readings at the anchor stop the
task **without recalibrating**; it does not blindly search upward or zero a
possibly loaded/nearby probe. Force/contact, stale feedback, and connection
faults stop immediately without automatic return/calibration. The force guard
stays active throughout tactile recalibration.

The JSON now records `search_safety_policy`, clearance checks, calibration
history and a stop snapshot (TCP XYZ, force and measured descent when available).
`probe_levels.csv` records stationary depth evaluations, feedback Z/force,
threshold, signal strength and the active boundary. These describe sample/stop
feedback, **not** an independently measured first-contact point. Older runs
without these diagnostics cannot retroactively supply that distance.

After the anchor and its four reference neighbours are verified, mapping now
grows **outwards through verified neighbours**, rather than starting at a distant
corner of a fixed whole-grid serpentine route. Each next target must be one
row/column step from a measured centre. It prioritises the innermost available
ring, then multiple verified supports and shorter travel. Nearby measured
row/column edges update the step vectors; predictions from the adjacent known
centres are combined by median. It re-plans after every accepted centre.

The initial fitted grid remains a geometry/workspace preflight, not the origin
of long-range motion targets. The 200 mm workspace, clearance routing, local
scan radius, existing force/stale-feedback guards, two-attempt limit and final dominance
verification are unchanged. Failure still returns to the anchor and stops;
the planner does not silently label an unverified location or jump across a gap.
The reverse scan-point checkbox still concerns XY measurement-line traversal,
not this outward target order.

Saved/resumed maps use this new planner automatically after the existing safe
anchor recalibration/alignment check. Previously measured positions are kept;
failed unsaved taxels remain to be mapped. No JSON edit is required. Settings
record `mapping_order=verified_neighbour_frontier_v1`, resume history records the
old/new policy, and newly accepted regular centres record their verified
neighbours, step vectors and prediction spread. That spread is an agreement
diagnostic, **not** measured position accuracy. The all-point lattice fit remains
a separate diagnostic. Restart the UI after normally stopping/saving any active
run; do not hot-reload or assume a running process has changed its planner.

Clicking a mapped taxel goes to that taxel's **saved safe pose**, without a detour
through the anchor XY. After checking live tactile, force and robot feedback,
navigation lifts vertically if below the saved clearance height, then restores
the saved orientation in place if needed. From clearance or above, it takes a
straight Cartesian line to the selected taxel's safe pose. This is not its lower
scanning/proximity position. It waits for both position and orientation to settle
at every waypoint. Stop and the force/freshness guards apply throughout. A paused
robot project blocks the click; a pause during movement cancels remaining
waypoints, with no automatic resume. The clearance belongs to the loaded map.
If the sensor has moved, relocate the map first. Mapping scans and calibration
trials keep their own existing anchor/recalibration routines.

This Cartesian route assumes the approach and clearance space are clear; it
does not perform obstacle detection or collision planning. Validate the direct
path under supervision when first using a saved setup.

Mapping defaults to 9 frames per XY point (adjustable in mapping search
settings; choose 20 or more if the shorter window is noisy). JSON settings
record the adaptive-search policy and median reducer; scan CSV rows record
the pass and actual frame count. Loaded maps preserve their saved frame count.
The controller's project-speed override is not bypassed: a 20 mm/s line command
at 5% runs at approximately 1 mm/s. These software changes reduce search
positions and sampling overhead, not physical safety margins. Runtime depends
on actual robot speed, sensor rate and fallback frequency; supervise the first
hardware validation. Restart the UI only after stopping and saving the active
run normally; changes are not hot-applied to an active experiment.

The **Reverse scan-point order** checkbox enables a second independent
run from opposite scan traversal. Preserve the first map, leave the setup unchanged,
and run mapping again rather than resuming the completed map.

Use **Compare Repeated Maps…** on the mapping page to select at least two saved
map JSONs. It reports common-taxel XYZ standard deviations and XY spread in mm,
and can export a separate JSON. Never interpret relocation as repeatability.
Maps locate response centres; confidence is a heuristic signal/dominance score,
and lattice residual is internal consistency, not geometric accuracy. Independent
physical references are needed to establish absolute centre accuracy.

**Validate Against Independent XY References…** opens an offline comparison
with a CSV containing `taxel_index,x_mm,y_mm`. Use zero-based display taxel IDs
and independently measured robot-base millimetres from the unchanged setup.
Do not copy the map's predicted/measured positions into the reference file.
The comparison reports per-point XY errors and RMSE without fitting an alignment
to hide error. Z is deliberately excluded: mapping proximity-plane height is
not independently measured surface height. A few references do not prove the
accuracy of every node.

The fast-versus-dense comparison is optional if mapping speed is not a paper
contribution. Independent centre checks and repeatability still support the
validity of the characterisation measurements; neither guarantees acceptance.

## New task: Sensor Characterisation

Select **Experiments → TRO2026 → Sensor Characterisation → Open Workbench**.
This is a separate fixture-guided native-frame recorder, **not an automatic
robot XY scanner or a new force controller**. It never sends robot commands,
starts a force meter, or stops another controller. Stop/save a running automated
experiment first. Opening the window or analysing saved data needs no devices.

With the current single robot probe you can use existing Calibration for
automated force/distance/spatial-spread acquisition, and this workbench for
independently labelled stationary spatial contacts, noise/interference and
manual finger/size/speed recordings. It does not make a one-probe system produce
simultaneous calibrated two-point contacts. Do not run the existing flat -Z or
fixed-Z paths unchanged on a curved mounting.

### Collect a workbench trial

1. Build/calibrate and live-stream the intended sensor. Check orientation against
   Sensor signal. Use stable physical specimen IDs rather than ttyACM numbers;
   keep the 10×10 paper specimen and 20×20 variant separate.
2. Under **Setup**, choose Spatial response field, Multi-touch separation,
   Finger size / gesture speed, Mounting / remounting, Acquisition / latency audit,
   or Noise / drift / robot interference. Enter specimen, session, unchanged
   mounting and anonymous participant IDs; record setup/environment notes.
3. Confirm orientation and a clear initial baseline. Set duration (10–300 s)
   longer than baseline (2–60 s) plus two seconds. Choose a parent folder.
4. Under **Record conditions**, press **Start native recording**. Keep the
   hand/probe clear for the entire baseline. Native completed-buffer callbacks
   collect every delivered frame, independently of GUI polling. The host-buffer
   timestamp is not a hardware scan timestamp or guaranteed 200 Hz acquisition.
5. Set the declared contact count, label, optional diameter, nominal speed and
   reference XY. Blank means unknown. Measured positions must share the stated
   robot-base frame. The TCP copy button is read-only and needs fresh feedback
   plus explicit stationary/registration confirmation; the software cannot verify
   the physical probe alignment. It is not a measured finger position.
6. Mark **Begin contact / gesture** for each declared condition and **Release /
   no contact** at its end. To quantify false activations after baseline, begin
   a separate condition with contact count **0**, keep clear, then end it.
   Change labels/conditions between intervals, not within one. First 0.2 s of
   each interval is excluded from stationary response summaries.
7. **Stop and save**, or let duration expire. Recording fails visibly on stale
   tactile data, geometry/source/order/calibration/mask changes, writer overload,
   or another automated controller starting. Partial data are retained, but
   excluded from final comparisons. This does not send a robot stop command.

Manual motion/probe loading remains the operator's responsibility, including
force protection and emergency-stop access. Workbench reference readings are
optional latest samples with source ages, not synchronised force measurements.
Do not take the workbench as a replacement for calibration's motion safeguards.

### Inspect and compare

**Analyse saved data → Open saved trial…** accepts the trial folder offline.
All related files stay inside that unique specimen-and-timestamp folder:
`manifest.json`, `characterisation.json`, native compressed `sensor/*.npz`,
optional `references.csv`, `analysis.json`, `condition_summary.csv`, and
`sensor_response_condition_*.png`. The shared recorder may also create an empty
camera directory/index; this workbench does not collect camera images.

- Native raw, calibration and optional signed filtered matrices retain their
  original row order. Analysis converts them to the existing display taxel IDs.
- Reports include observed native delivery rate/intervals, sequence gaps,
  baseline median/MAD noise and baseline half-to-half shift, signed spatial
  fields, active taxels, weighted grid centroids and local peak candidates.
- A noise threshold is initially max(3 sensor units, 5×baseline robust sigma).
  A declared no-contact interval reports above-threshold activation episodes/min.
  It is descriptive, not a validated operating-point false-positive rate.
- **Graphs** has zoom/pan plus a marked-condition selector. It overlays the
  native raw-derived peak change and delivered filtered change when available.
  Spatial maps retain positive/negative signs and mask disabled taxels.
- **Compare trials…** selects `characterisation.json` from several completed
  trials of the same specimen/layout. Mounting, robot condition, size/speed,
  label and declared positions remain separate groups. It averages intervals
  within each trial first; between-trial SD is not a confidence interval and
  repeated windows are not independent participants.

Response spreading includes mechanical/electrical effects. Grid-centroid units
are not millimetre localisation accuracy. Two response maxima do not by
themselves prove two physical contacts or calibrated two-point resolution.
For the latter, obtain a two-contact fixture with independently known tip
positions and separations. Until then, manual two-finger trials are exploratory.

### Timing and recognition robustness imports

The onset/command/motion buttons are operator annotations only. For measured
contact-to-command latency, use an independent event detector/controller log
aligned to this trial's host-monotonic clock, then import CSV columns
`t_s,kind,uncertainty_ms`. Supported kinds are `contact_reference`,
`command_reference`, `motion_reference`; `t_s` is recording-relative seconds,
after baseline and within the saved native frames. Declare clock alignment and
each reference's timing uncertainty. The analysis pairs independent contact
and subsequent command events, not manual buttons, and reports summed timing
uncertainty. Motion events are retained for inspection; this is not an automatic
physical robot-motion-latency measurement. Unaligned video/device clocks or
model inference runtime alone cannot establish end-to-end latency.

For external recognition predictions, label intervals independently with exact
gesture names and import `t_s,predicted_label`. Evaluation excludes 0.25 s at
label boundaries and exports `prediction_evaluation.json` with confusion,
accuracy and macro-F1. It does **not** train/retrain a model or verify that its
test data were held out. Split original participants/sessions/source recordings
before augmentation and evaluate unseen sizes/speeds/remounts separately.

## Suggested pilot

After checking the hardware/reference, use a few centre, edge and corner nodes
before a long full scan. At validated safe loads, collect repeated load/unload
cycles, then hold/recovery, speed conditions and controlled short sliding paths.
Keep mounting and probe geometry consistent. Repeat on another session/day;
use a separate session note. Test proximity separately with documented targets,
because a metal indenter and human finger need not have the same coupling.

Software was tested with simulated devices and offline UI/recordings. Physical
load accuracy, stop distance, sliding contact and real hardware timing still need
supervised validation; no physical robot motion is part of these software tests.

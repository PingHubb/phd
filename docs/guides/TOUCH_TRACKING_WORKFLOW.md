# Continuous touch tracking: collection to evaluation

[All guides](../README.md) · [Project layout](../PROJECT_LAYOUT.md)

## Camera-free workflow (new default)

No camera, force meter or robot connection is required. Open **AI → Data → Open
Touch Tracking workspace**. The existing independent-reference workflow remains
available by selecting its collection/training mode; its recordings and models
are not converted or overwritten.

### Try the direct tracker before collecting a large dataset

1. Stream the sensor. Keep its baseline and taxel ordering fixed.
2. Open **4 · Live preview** and choose **Camera-free robust tracker — no model/training needed**.
3. Remove your hand completely from the sensing area. Confirm the hands-away
   checkbox and press **Start sensor-only preview**.
4. Keep your hand away until the five-second noise calibration finishes. It
   needs at least 50 usable frames; missing frames can prolong calibration.
5. Touch with one finger, hold still, slide, reverse direction, then release.
   The displayed point/trail estimates the responding region. Signal quality is
   NOT a contact probability, and a localized proximity signal can still appear.

This direct tracker uses per-taxel noise statistics, both response polarities,
connected responding regions, temporal confirmation, and causal speed-adaptive
smoothing. Its filter follows the [1 Euro filter approach](https://gery.casiez.net/1euro/).
Complete row/column responses, broad signals, multiple strong disconnected
regions and abrupt jumps are withheld instead of being assigned a finger centre.
This is intentionally conservative: real broad touches can also be withheld.
Baseline/mask changes or counter resets require restarting with the hand away;
the tracker never quietly recalibrates an ongoing touch as zero.

Positions assume a regular rectangular taxel grid, located at cell centres.
Edge estimates cannot automatically resolve positions beyond those centres;
irregular spacing, curved mounting and nonuniform sensitivity need additional
calibration. A persistent local fault can resemble a real finger. These checks
are not universal fault detection or a robot-safety certification.

### Record without camera images or per-frame annotation

In **1 · Record**, leave **Camera-free — guided sensor collection** selected.
Enter participant/session/physical sensor ID and optional measured dimensions.
Use 30–60 seconds for a first trial. Confirm the collection instructions.

An example:

| Period | Action and recording state |
| --- | --- |
| First 6 seconds | Hand completely away; leave state as **No touch**. |
| Repositioning | Select **Uncertain / reposition**, then bring the finger down. |
| Next 10–15 seconds | Once touching, select **Touch / move — keep contact**, then slide and hold. |
| Lifting/repositioning | Select **Uncertain / reposition** before lifting. |
| Next 5 seconds | Once clear, select **No touch — hand away**. |
| Remaining time | Repeat with another direction/location, marking transitions as uncertain. |

Touch/no-touch states are your declarations, not automatically measured truth.
Use **Uncertain** when unsure, during multiple contacts, or whenever the action
does not match the selected state. The initial confirmation records agreement
to follow the protocol; software cannot verify that you actually followed it.
The trainer excludes 0.35 seconds around state changes and the recording end.
Keep touch intervals long enough to include holds and movement after that margin.

Every trial contains the original sensor chunks and a separate
`camera_free_protocol.json`. There is no camera capture and no `reference.json`
is fabricated. Noise statistics come only from the initial hands-away portion;
held-out touch data is never used to fit thresholds. Strong stable, localized
responses supply estimated position/velocity targets. Ambiguous positions are
masked; operator-declared contact labels remain separate. Changing calibration,
source or ordering stops the recording as before.

New default folders:

```text
resource/ai/data/touch_tracking_camera_free/<sensor>_<timestamp>_<unique-id>/
resource/ai/models/touch_tracking_camera_free/camera_free_<timestamp>_<unique-id>/
```

### Train and compare honestly

Skip **Reference labels** for camera-free recordings. In **3 · Train and evaluate**,
select **Camera-free — operator states + estimated positions**, check the dataset,
then train. At least three genuinely separate sessions/people are still required;
each split must include usable motion and no-touch windows. The check reports
position-target coverage, including how much ambiguous data was excluded.

CNN–GRU and CNN–TCN training still uses grouped train/validation/test partitions,
train-only normalization, causal preprocessing and saved immutable runs. The
network learns these weak targets; it is NOT guaranteed to outperform the direct
tracker. The test plots explicitly say **Sensor-estimated target (not ground
truth)**. CSV columns use `estimated_target_*`; metrics use
`sensor_target_agreement_*` and `operator_label_*`. Even a small millimetre
agreement error is NOT an independently measured physical position error.
Comparisons to the simple centroid are likewise not independent validation.

Example CLI:

```bash
python -m phd.script.train_touch_tracking check /path/to/camera_free_trials --supervision camera_free
python -m phd.script.train_touch_tracking train /path/to/camera_free_trials --supervision camera_free --output /path/to/new_models
```

Independent-reference mode remains the CLI default to prevent silently changing
the meaning of existing training scripts. Camera-free trials are rejected by
independent mode, and vice versa. Choose the matching dataset folder explicitly.

To preview the trained model, select the neural-model backend and its
`selected_model.pt`. Camera-free checkpoints require a fresh hands-away noise
calibration and apply the direct tracker's quality gate before displaying neural
predictions. Gross model/signal disagreement is withheld. Offline model metrics
are before that additional live gate; they do not describe final gated coverage.
Compare the direct tracker and neural tracker on new sessions before choosing
one. Both are sensor-only and cannot send robot commands.

No claim of perfect tracking, physical contact/proximity separation, or unseen
user robustness follows from these training scores. Independent spot checks or
known-location measurements remain useful even when collection is camera-free.

---

## Independent-reference workflow (optional)

The remainder describes the original camera/external-reference mode. Select
**Independent reference — camera or measured CSV** to use it.

## What changed

Open **AI → Data → Open Touch Tracking workspace** after restarting the UI.
The new workspace has four stages: Record, Reference Labels, Train and Evaluate,
and Live Preview. Existing policy datasets, trained models, controls and firmware
are unchanged. New assets go under:

```text
resource/ai/data/touch_tracking/<sensor>_<timestamp>_<unique-id>/
resource/ai/models/touch_tracking/tracking_<timestamp>_<unique-id>/
```

The existing `PINGLAB_AI_RESOURCE_ROOT` override is respected.

This first version estimates **one finger's contact position and velocity on a
sensor surface**. It is not full hand-pose reconstruction, force measurement,
multi-finger identity tracking, or a proximity controller. It does not send
robot commands. Keep the robot stationary while collecting camera references.

The old controller's `intended_velocity_target`, inferred touch mask, and finger
count estimates are **not** new training labels. The simple centre tracker is an
evaluation baseline only, not a teacher for the new neural networks.

## 1. Prepare the experiment

1. Build and stream the intended sensor in the Sensor tab. The recording follows
   the actual streaming data buffer, not the highlighted port in a list.
2. Fix the sensor in place. Start with one sensor geometry and one finger.
3. Let the baseline settle. Run your normal calibration **with no hand touching
   or hovering over the sensor**. This workspace never recalibrates automatically.
4. Open the existing camera preview. Object-detection/auto-centering must be
   stopped manually first; the workspace will not interrupt robot motion.
5. Position the camera so the surface coordinate reference and contact locations
   can be measured. Avoid camera/sensor movement after calibration.
6. Enter real participant, session and physical sensor IDs. Keep the same session
   ID for trials from the same sitting. Use different IDs for genuinely different
   days/setups, not for copies of the same trial.
7. Enter sensor width and height in mm if measured. Leaving either unknown keeps
   evaluation in normalized surface coordinates; it does not invent mm values.

The four-corner camera annotation method assumes a **flat, stationary surface**.
It does not account for curvature, strong lens distortion, or an elevated marker
on a fingernail. For curved sensors, use externally measured surface coordinates
and import the reference CSV. Never label an occluded contact point by assuming
that the visible top of the finger lies in the sensor plane.

## 2. Record

Click **Start trial — no robot**. Stop manually or let the duration limit finish.
Default: 120 seconds; configurable from 10 to 600 seconds. Camera capture is on
by default. You may explicitly disable it if you have another independent
reference system and will import its measurements later.

Record small blocks containing all of these:

- No hand near the sensor; nearby hand without physical contact.
- Touch and hold still, with light/medium/firm contact.
- Straight lines in several directions, including diagonals.
- Slow/fast motion and gradual acceleration/deceleration.
- Curves, circles and figure-eights.
- Stops, reversals, lift-off, and touching down at a different location.
- Centre, edges, and corners—not only the most responsive taxels.

Vary the order, contact angle and speed. A useful pilot is several short trials
in each of three independent sessions on different days. More people and days
are needed to make a credible general-use claim. Dataset sufficiency must be
judged by learning curves and genuinely unseen recordings, not total frame count.

### Saved files and failure behaviour

Each trial folder contains:

```text
manifest.json               IDs, geometry, layout, clock, status and limits
sensor/000000.npz ...        small recoverable raw/baseline/mask/timestamp chunks
camera.csv                  exact host timestamps for the saved images
camera/000000.jpg ...        unannotated RGB reference images
reference.json              added after independent labels are verified
reference_history/*.json    previous saved reference versions
```

Sensor data is recorded from completed buffer updates, not by repeatedly reading
the same UI frame. Disabled cells are saved with a validity mask. "Raw" means
the observed native sensor buffer: any existing firmware/driver filtering or
zero-mask operation may already have affected it. Raw data is never rewritten
by annotation or training.

Source/geometry/taxel-order changes, calibration changes, sensor resets, stale streams and
writer overload stop the trial visibly. Already saved chunks remain available.
Incomplete/failed trials are **not silently included in training**. Keep them
for diagnosis, or explicitly review/recover copies before adding them to a
curated training dataset. Do not rename incomplete recordings to make them look
like independent successful sessions.

### Timing is measured, not assumed

Sensor timestamps are host monotonic **buffer-update** times. Camera timestamps
are host monotonic **color-frame receipt** times. These share a host clock but
are not hardware-synchronised exposure/ADC timestamps. Camera buffering and
sensor transport can introduce different delays.

Check timing against an independently observed event before using velocity
labels. The Reference Labels page provides an explicit time offset in seconds:

`sensor-aligned reference time = reference time + offset`

Document the measured offset. Do not tune it to improve held-out test results.
If this timing accuracy is insufficient for fast movements, improve acquisition
timestamping/reference hardware before claiming accurate fast tracking.

## 3. Create independent reference labels

### Camera annotation

1. Open a trial folder in **Reference Labels**.
2. Select four corners in the sensor's coordinate order:
   `(x=0,y=0) → (1,0) → (1,1) → (0,1)`.
   **x follows native columns; y follows native rows.** These are not necessarily
   image top-left/top-right after mounting rotation. Confirm the correspondence
   using known physical taxels. Do not mix the legacy flipped/transposed view
   with the new native coordinate system.
3. Check projected centre/edge locations against known positions. The four clicks
   alone are not an independent accuracy check of calibration.
4. Click the actual visible contact location to mark a touch position.
5. Use **No touch** for physically non-contacting frames. Use **Unknown / occluded**
   if contact or position cannot be verified, including ambiguous/multiple contacts.
6. **Suggest next position** uses optical flow with a forward/backward consistency
   check. It is only a visual suggestion. **Accept visible suggestion** marks it
   as a label after your inspection; an unchecked suggestion is never used.
7. For confirmed no-touch intervals, set the interval start, inspect through its
   end, and apply **Mark reviewed interval: no touch**. Unknown intervals can also
   be marked in bulk.
8. Check the reference verification box and save. Earlier reference versions are
   retained. Changing the plane calibration clears in-memory labels to prevent
   mixing incompatible coordinate systems; old saved labels remain on disk until
   a replacement is explicitly saved and archived.

Labels need to be close together in time. By default, the trainer interpolates
only between reviewed, same-contact measurements at most **0.12 seconds apart**.
It does not extrapolate across lift-off, unknown labels or long gaps. Label fast
curves more densely; sparse straight-line interpolation is not a substitute for
measuring their actual shape. Reference velocity is the local position slope
between nearby measurements, so annotation noise directly affects velocity.

### External reference CSV

Use a calibrated tracking system or a controlled probe if camera contact points
are not observable. The CSV format is:

```csv
t,contact,x,y,quality,reviewed
0.000,0,,,1.0,1
0.033,0,,,1.0,1
0.067,1,0.250,0.400,1.0,1
0.100,1,0.260,0.400,1.0,1
0.133,-1,,,0.0,0
```

- `t`: seconds relative to trial start, before the explicit offset.
- `contact`: 1=touch, 0=no touch, -1=unknown.
- `x,y`: normalized measured surface position, each in [0,1]. Not robot velocity.
- `quality`: reference measurement quality [0,1]; below 0.8 is excluded by default.
- `reviewed`: 1 only for accepted measurements.

Unknown is not equivalent to no touch. A failed optical tracker should produce
unknown labels, not negative examples. Probe references are useful supplementary
data, but a probe's contact pattern is not identical to a human finger's.

From the package repository root:

```bash
python -m phd.script.train_touch_tracking import-reference /path/to/trial reference.csv --source external_tracking --verified
python -m phd.script.train_touch_tracking import-reference /path/to/trial reference.csv --source robot_probe --time-offset-sec 0.02 --verified
```

The verification flag is your declaration that the measurements are independent
and aligned. Software cannot certify physical reference accuracy from a CSV.

## 4. Check and train

Use **Train and Evaluate → Check dataset** first. Select a curated folder of
complete, reviewed trials. The original data can remain elsewhere; do not delete
unlabelled trials merely to pass the check.

Training refuses missing/unverified references, incompatible geometry, invalid
timestamps, duplicate recordings across splits, insufficient independent groups,
or splits lacking both movement and no-touch examples.

Select:

- **Unseen sessions / days** for the same person's future sessions.
- **Unseen people** to test use by people excluded from training.

At least three real groups are needed: train, validation and test. This minimum
is a software requirement, not evidence that three groups are enough for a
publication or deployment. Splits are fixed by seed and saved in `split.json`.
Never split overlapping windows of a single recording into different subsets.

Default training compares a compact CNN–GRU and CNN–TCN:

- Shared native row/column input convention.
- Signed relative signal change, change per **second**, taxel validity and sample age.
- Causal 50 Hz resampling from past measurements; no future sensor interpolation.
- The same preprocessing code is used offline and in live preview.
- 16-frame history by default, adjustable in the UI.
- Position, velocity and contact heads; no forced discrete motion direction.
- Losses ignore unknown positions/velocities.
- Training-only normalization and velocity scaling.
- Sampling balances no-contact, stationary contact and moving contact.
- Optional flips transform positions and velocity signs together; small gain/drift
  augmentation retains signed changes. It does not fabricate independent trials.
- Early stopping on validation. Contact threshold selected on validation only.
- Final architecture selected by validation loss, never by test performance.

The test set is evaluated after selection. A causal, lightly smoothed contact-
centre baseline is evaluated on the same test windows for comparison.

CLI equivalents:

```bash
python -m phd.script.train_touch_tracking check /path/to/reviewed_data
python -m phd.script.train_touch_tracking train /path/to/reviewed_data --group-by session --architectures gru,tcn
python -m phd.script.train_touch_tracking train /path/to/reviewed_data --group-by participant --seq-len 32 --seed 42
```

Every training run uses a new output folder. Neither the old robot-policy
checkpoint nor its `latest_*` aliases is overwritten. Cancellation retains the
partial run for inspection; it does not produce a completed final report.

## 5. Read the results

Each completed run contains:

- `dataset_check.json`, `split.json`: coverage, exact trial lists and data/reference hashes.
- `gru_history.json`, `tcn_history.json`: training/validation loss by epoch.
- Model-specific validation folders with metrics, predictions and figures.
- `test/`: selected model's held-out metrics, CSV predictions and per-trial plots.
- `centroid_test/`: the comparison tracker on the same held-out data.
- `summary.json`: model selection, evaluation and explicit limitations.
- `selected_model.pt`: the independently supervised model for preview.

Inspect position error, velocity error, false/missed contact, unwanted movement
when stationary/no-touch, and per-trial failures. Position errors include missed
contact examples rather than hiding them. Velocity metrics use contact-gated
output. Frame-level false-motion rates are not false activations per hour or a
robot-safety guarantee.

Per-trial diagnostic velocity lag is estimated only where there is sufficient
varying, correlated motion. Positive means delayed tracking. `null` means the
data does not justify a lag estimate. Predictions are **never shifted** to make
the reported errors look better. The figures show x/y position, x/y velocity and
contact probability over time.

Repeat model comparisons with additional seeds and genuinely new sessions before
claiming a winner. Do not keep tuning against the same test set. For stronger
research evidence, add grouped cross-validation and uncertainty intervals over
independent people/sessions—not over highly correlated individual frames.

## 6. Sensor-only live preview

Select `selected_model.pt` in **Live Preview**, check the physical sensor ID in
the Record page, stream that sensor and start preview. The camera is not needed.

The preview shows normalized position, signed velocity in surface units/second,
contact probability, model forward-pass duration and a motion trail. Forward-pass
duration is **not** the end-to-end tracking delay. Probability is **not** a
certified measure of tracking reliability or an out-of-distribution detector.

Preview ingests every observed buffer frame and performs inference on the most
recent complete window. It clears history on gaps/resets/calibration changes,
refuses incompatible sensor geometry/ID/taxel ordering, and discards stale predictions. It stops
on overload rather than silently skipping acquisition frames.

There is deliberately **no robot command interface** in this new tracker.
Robot integration is a later, explicit step after independent reference checks,
unseen-session tests, and stationary robot-free preview. Keep robot speed,
acceleration, workspace, emergency-stop and stale-data safeguards independent
from the learned model. Multiple fingers and proximity require separately
labelled data and validation before adding those outputs.

## Verification scope

Automated tests exercise artificial recording → independent reference → dataset
check → both architectures → held-out evaluation → checkpoint loading → preview,
plus timing, masking, leakage guards, layout, safe imports and offline UI startup.
Synthetic test success proves the software path runs; it does not establish
accuracy on the physical sensor. No new real-finger model is supplied until you
collect and verify independent reference data.

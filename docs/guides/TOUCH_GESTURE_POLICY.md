# Touch gesture policies

## Current default: single-touch / non-continuous six gestures (27 October 2026)

The initial checkpoint is **`six_gestures_single_touch_10x10.pt`**, a CNN–TCN
(seed 43, epoch 22). It was trained in single-touch mode on the same 600 files
and six folder labels: swipe up/down/left/right and two-finger push/pull.
This is a separately trained checkpoint with matching preprocessing, not a
continuous checkpoint with its input-window metadata changed.

Restart the UI, open **AI → Run → Touch policy**, and leave **Dry run** checked.
The checkpoint details show **Input: current touch history** and
**Single-gesture mode: lift your finger before the next gesture.** Use
**Select Model** if an older checkpoint is still selected. Connect and calibrate
the same 10×10 sensor with `row_to_col` layout and averaging 3, then Execute.
Keep the hand away briefly to arm; perform one gesture, lift, then perform the
next. Allow about 0.2 seconds of release between gestures.

- Each prediction uses the available history since that touch began, resampled
  to 32 time steps. It does not discard everything older than 200 ms.
- Prediction can begin before release, after at least 5 contact frames spanning
  80 ms and two confident matching predictions. It is not a release-only action.
- Release commands zero immediately and clears the history for the next touch.
  The gesture feedback is completed after the release debounce.
- The continuous-mode last-command fallback is inactive. An uncertain prediction
  commands zero instead of retaining the previous confirmed direction.
- **A stationary finger is not a trained stop class.** The classifier can still
  command motion from the earlier gesture history while contact remains. Lift
  to stop; this mode is not a continuous finger-speed tracker.
- Contact exceeding 2 seconds stops prediction and requires release to re-arm.
  Gesture axes, speed settings, confirmation and freshness guards are unchanged.

Training compared CNN–GRU/TCN with seeds 17/43/79 and the existing chronological
360/120/120 whole-trial split. Selection and temperature calibration used
validation only. The selected TCN classified **120/120 final-contact samples**
correctly and **1956/1966 (99.49%) partial histories** correctly. Actual-executor
offline replay recognised every gesture before release but produced **4 brief
wrong-direction decisions in 3 trials**, among 1837 nonzero decisions. Release
stopped commands in all 120 trials. Median first-command time was 103 ms,
95th percentile 133 ms in recorded time, not physical robot latency.

Therefore this switch matches the requested one-gesture-per-touch use, but does
not establish better early-command accuracy than the continuous model. These
are reused same-collection test trials, not an independent new-day/user test.
Try fresh gestures in Dry Run before enabling robot commands. No hardware was
moved during training or replay.

Run artifacts: `six_gestures_single_touch_10x10_20261027_v1/`. The old continuous
checkpoint `six_gestures_recommended_10x10.pt`, all earlier models and all
recordings are unchanged. To reproduce the new mode, omit `--continuous`:

```sh
/usr/bin/python3 -m phd.script.training.train_touch_gesture_policy \
  --gesture-set six-gestures --architectures cnn_gru cnn_tcn \
  --output NEW_OUTPUT_DIRECTORY
```

## Previous continuous six-gesture model (27 October 2026)

The previous continuous checkpoint is **`six_gestures_recommended_10x10.pt`**,
a CNN–TCN (seed 79, epoch 23). It was retrained from all six current
`ai/data/tactile_recordings/session_10x10_*` gesture folders: **600 complete,
unique recordings, 100 per gesture**. Only `swipe_up`, `swipe_down`, `swipe_left`,
`swipe_right`, `two_finger_push`, and `two_finger_pull` are included. The push
folder's older internal session tag is `10x10_push`; its folder supplies the
explicit `two_finger_push` label. Original recordings and older models are unchanged.

To try that continuous version, select its checkpoint under **AI → Run → Touch policy**. The checkpoint details
should show **CNN–TCN · 6 gestures**, including **Two finger push / pull**.
Use **Select Model** to choose that continuous checkpoint
from `phd/resource/ai/models/touch_policy/`. Leave **Dry run** checked initially,
start the same 10×10 sensor with `row_to_col` layout and averaging 3, calibrate
with the hand away, press **Execute**, and leave the hand away for at least
0.2 seconds before touching. Current/last-completed results now support both
two-finger labels. This remains one checkpoint selector, with no gesture preset.

Input is still the latest 200 ms of sensor data, resampled to 32 time steps,
requiring at least 5 measured contact frames spanning 80 ms. The explicit
velocity mapping is up → Z−, down → Z+, left → X−, right → X+, **two-finger
push → Y−, two-finger pull → Y+** in the UI-selected frame; rotation is zero.
Current speed and hold behaviour are unchanged: 1× commands 50 mm/s; a confirmed
direction persists during a stationary hold until a new gesture is confirmed
or the finger lifts. Test all six directions in Dry Run before commanding hardware.

### Training and checks

- Chronological whole-trial split per gesture: 60 train / 20 validation / 20 test,
  totalling **360 / 120 / 120**. No recording contributes windows to multiple splits.
- Both CNN–GRU and CNN–TCN trained with seeds 17/43/79, identical augmentation,
  balanced class/trial sampling, and the same epoch budget. Normalization used
  training data only. Best epoch, model/ensemble, and confidence temperature
  were selected using validation only, before evaluating this run's test split.
- CNN–TCN: **119/120 final-contact samples** correct and **1962/1966 (99.80%)**
  eligible overlapping windows correct. CNN–GRU: **120/120** final-contact
  samples and **1958/1966 (99.59%)** windows. The selected TCN had better
  validation negative log likelihood with equal validation macro-F1.
- Actual-executor offline replay: TCN recognised **all 120 gestures before
  release**, with **0 wrong nonzero decisions / 1829**. GRU had **6 wrong
  decisions in 3 trials / 1832**. Both stopped commanded velocity on every
  release. The executor's confirmation/hold logic explains why TCN's isolated
  final-window error did not become a wrong command in this replay.
- TCN median first-command time: **103 ms**, 95th percentile **135 ms**, in
  recorded sensor time. Replay used synthetic quiet arming samples, synchronous
  CPU inference and no robot API; this is not measured physical robot latency.

Run artifacts, file hashes, split membership, comparison, predictions and both
replay reports are under `six_gestures_comparison_10x10_20261027_v1/`. Alternative
checkpoints `six_gestures_cnn_gru_10x10.pt` and `six_gestures_cnn_tcn_10x10.pt`
remain selectable. Reproduce training from the repository root with:

```sh
/usr/bin/python3 -m phd.script.training.train_touch_gesture_policy \
  --gesture-set six-gestures --architectures cnn_gru cnn_tcn --continuous \
  --output NEW_OUTPUT_DIRECTORY
```

These are same-sensor, same-collection results, not a guarantee for new days,
users, unknown gestures or uninterrupted gesture changes. The swipe test
recordings were also evaluated during the earlier four-class development, so
this is **not a new independent external test set**. No test data was used for
gradient training or model selection in this retraining. Collect a fresh mixed
session for a stronger real-world assessment. Proximity is not included.

## Previous four-swipe comparison (27 October 2026)

The four new `tactile_recordings/session_10x10_swipe_*` folders contain
**100 recordings each**: up, down, left and right. This run uses only these
400 files, not the old six-direction, robot-dog or proximity datasets. Folder
names supply the labels; no camera, teacher velocities or robot pose are used.

Restart the UI, then open **AI → Run → Touch policy → Policy setup → Checkpoint**.
There is a single **Select Model** button. That run's recommended checkpoint is
`four_swipes_recommended_10x10.pt` (CNN–TCN seed 17, epoch 34). To compare models,
select `four_swipes_cnn_gru_10x10.pt` or `four_swipes_cnn_tcn_10x10.pt` from the
same model folder. Selecting a model checks **Dry run** and sets **1.00×** speed;
it does **not** start execution. Selection is disabled during execution.

Architecture, supported output labels, sensor shape and input window are read
from the selected checkpoint in a background, restricted CPU metadata preview.
They appear **before Execute**, without a sensor or robot. During execution,
details reflect the actual loaded model, not a changed file on disk. Invalid or
unsupported preview files show an explicit message instead of guessing labels.
Full model validation still happens when Execute loads the policy.

The redundant **Use Default**, **Use Continuous**, and **Use 4 Swipes** buttons
have been removed; their saved checkpoints are unchanged and remain selectable.
The older **Predict / Latch** controls are now under **AI → Run → Legacy
controller**, explicitly labelled as a separate three-level Transformer that
can command robot motion. They do not use the Touch Policy checkpoint.
The separate **Gesture recognition** page remains a sensor-only completed-gesture
classifier. Current/last-completed feedback stays in Touch Policy.

Connect the same 10×10 sensor with `row_to_col` layout and averaging of 3 frames.
Calibrate with the hand away, press **Execute**, then swipe. If the hand is
already away, the first swipe is accepted. If a finger is down when Execute
is pressed, lift briefly before the first swipe. Watch **Current gesture** and **Last completed**
gesture**. Stop before selecting the other model and repeat the same gestures.
No position map, force meter or connected robot is needed for Dry Run.

Each prediction uses up to the latest **200 ms** of contact, resampled to 32
time steps, with at least 5 measured touch frames spanning 80 ms. The output is
`swipe_up`, `swipe_down`, `swipe_left` or `swipe_right`; low confidence/margin
or stopped contact produces a stop. It does not output push/pull, continuous
finger coordinates, or learned speed. Command mapping remains **up → Z−,
down → Z+, left → X−, right → X+** in the selected control frame, with zero
rotation. 1× commands 50 mm/s on the recognised axis. The speed scale is not capped.
After a gesture is confirmed, holding the finger still keeps that direction until
another gesture is confirmed or the finger lifts. Confirmation, release and
freshness guards remain.

### Fair comparison and limitations

Both architectures use the same chronological whole-trial split: **240 train,
80 validation, 80 test** (60/20/20 per gesture). Windows from one recording never
cross splits. Normalization uses training data only; each family uses seeds
17/43/79, the same training budget and augmentation. Best epochs, single-model
versus ensemble selection and confidence temperatures use validation only.
The CNN–GRU is bidirectional within the available past-only window; neither
architecture sees later recording frames. Test recordings were evaluated after
selection and were not used to retune the model or safety thresholds.

- **CNN–TCN:** 80/80 final-contact classifications; 1076/1081 eligible partial
  observations correct (**99.54%**). Selected by validation macro-F1, then NLL.
- **CNN–GRU:** 80/80 final-contact classifications; 1074/1081 partial observations
  correct (**99.35%**). Selected GRU seed 17, epoch 11.
- Actual-adapter offline CPU replay recognised all 80 gestures before release
  for both. TCN issued **2 brief wrong-direction decisions in 2 trials** among
  982 nonzero decisions; GRU issued **1 in 1 trial** among 978. Thus TCN's small
  window-accuracy advantage does not establish universal superiority.
- Median first command was **101 ms TCN / 102 ms GRU**, with 95th percentiles
  133/144 ms, measured in recorded time. Release commanded zero on every test.
  Replay used synthetic quiet arming frames and no robot API; it does not measure
  physical robot latency, stopping distance or noisy real-world arming.

Overlapping windows are correlated, not 1081 independent tests. All data came
from one sensor/collection; new-day, new-person, stationary/unknown-gesture and
uninterrupted direction-change performance is **not established**. Test fresh
swipes, pauses and release in Dry Run before any supervised low-speed robot use.
Both models can briefly be confidently wrong. Neither is an unattended-motion
safety system. No robot was moved during training, evaluation or UI testing.

### Saved files and reproduction

Run folder: `phd/resource/ai/models/touch_policy/four_swipes_comparison_10x10_20261027_v1/`.
Contains `dataset_manifest.json` with hashes/splits, each candidate/history,
`model_selection.json`, `architecture_comparison.json`, `evaluation.json`,
`test_predictions.json`, `test_confusion_matrix.png`, and both
`runtime_replay_cnn_gru.json` / `runtime_replay_cnn_tcn.json`.

Selectable aliases in `phd/resource/ai/models/touch_policy/`:

- `four_swipes_recommended_10x10.pt` (selected TCN).
- `four_swipes_cnn_tcn_10x10.pt`.
- `four_swipes_cnn_gru_10x10.pt`.

All 400 source files and all three previous model aliases retain their hashes.
The UI additions follow the existing components and clear disabled/preview
states; no visual redesign or control-axis change is included.

```bash
python3 -m phd.script.training.train_touch_gesture_policy \
  --gesture-set four-swipes --architectures cnn_gru cnn_tcn --continuous \
  --output /new/four_swipe_comparison --device cuda
```

## Original six-gesture policy

The original six-gesture checkpoint recognises `up`, `down`,
`left`, `right`, `push`, and `pull` from the available touch history, then maps
that gesture to a velocity. It can respond during contact, before finger release.
It uses only capacitance percentage-change frames. No camera, force meter,
centroid teacher, robot pose or recorded rule-based velocity enters the model.

## Gesture feedback in Touch Policy

The Run page now uses the same result-card component as Gesture Recognition:
**Current gesture** updates during contact, with a large name, confidence and
bar; **Last completed gesture** retains the last model result after confirmed
release. In continuous mode the caption explicitly says **Last direction before
release**: this is not a classification of the entire multi-direction path.

The display reuses the running policy's predictions. It does not start another
model, sensor listener or controller, and does not change velocity commands.
Arming, paused and stop states have no gesture confidence; uncertain or too-short
touches are labelled accordingly. Manual Stop, timeout and stream faults do not
create a completed gesture. A completed result remains visible after Stop, while
selecting a different checkpoint or starting a new session clears the display.
Legacy velocity-only checkpoints report that named gesture labels are unavailable.

## Continuous-touch candidate (27 October 2026)

For changing direction **without lifting**, restart the UI, then open
**AI → Run → Touch policy → Policy setup → Checkpoint → Select Model**.
Choose `latest_cnn_tcn_continuous_touch_10x10.pt`. Selection checks **Dry run**, and
sets speed to **1.00× (50 mm/s commanded)**. Selection never starts execution.
Selection is disabled while execution is running. Select
`latest_cnn_tcn_touch_policy_10x10.pt` to return to the original single-gesture
policy; neither its checkpoint nor the source
recordings were overwritten. No taxel-position map is needed for either policy.

Press **Execute**, leave your hand away for at least **0.2 seconds**, then try
up → right → down → left while maintaining contact. Watch the prediction and
velocity. Check pauses and release as well as direction changes. Only after
checking this preview should you stop, uncheck Dry run and start again in a
clear robot workspace with the emergency stop accessible. Do not run another
motion controller or experiment at the same time. No robot motion was used to
develop or validate this candidate.

### What changed

- A separately trained CNN–TCN uses **only the most recent 200 ms** of the
  current touch, resampled to 32 time steps. Training and inference use the
  same crop. It still needs at least 5 touch frames spanning 80 ms to start.
  The old whole-touch history no longer influences predictions once outside
  this window. Direction-change latency is not guaranteed to be 200 ms.
- No release is required between directions. A continuous contact can last
  up to **30 seconds**, after which release is required. History is bounded
  independently of contact duration. The old model retains its 2-second limit.
- Two confident matching predictions are still required. An accepted command
  is sent at the selected speed immediately; there is no software acceleration
  ramp and no speed cap. Once a direction is confirmed, holding still keeps
  that command until a new gesture is confirmed or the finger lifts. Before any
  direction is confirmed, a still finger commands zero. Release and faults
  command zero immediately. Robot/network dynamics still determine the physical
  stopping time.
- The signal-change check is no longer a stop. It decides whether the latest
  200 ms should be classified. While the finger is still, the last confirmed
  direction is repeated instead. The check compares the mean of the first/last
  two averaged frames over the last 100 ms, taking the mean absolute change of
  the strongest 5% of taxels. The threshold is the larger of 0.25 percentage
  points and 3× the 90th percentile of recent hand-away frame-change scores
  (up to 50). This noise estimate is frozen during contact.
- Sensor and inference freshness checks remain. Robot command submission
  failures now stop the policy and display a connection reason instead of
  silently showing it as running. A successful service submission is not proof
  of physical movement or a controller acknowledgement.

### Validation and remaining limits

Run directory: `phd/resource/ai/models/touch_policy/cnn_tcn_continuous_10x10_20261027_v1/`.
The same 320 recordings and chronological whole-trial split were used: 192
training, 64 validation and 64 test. Seeds 17/43/79 and their ensemble were
compared on validation only; seed 17, epoch 33 was selected. Temperature 0.5
was fitted on validation only. Test results: 64/64 final gestures and 1124/1124
eligible overlapping recent windows classified correctly. These windows are
not independent trials.

Actual-adapter synchronous CPU replay recognised all 64 gestures before release,
with zero wrong-motion decisions among 803 nonzero decisions. Release stopped
commands in all trials. Median first-command latency was 104 ms and the 95th
percentile was 124 ms in recorded time, excluding robot/network delays.

**These are isolated-gesture results, not continuous-path validation.** The
recordings do not contain uninterrupted corners, reversals, diagonals or long
stationary holds. Unit tests exercise rolling memory, no-lift direction changes,
pause/resume, stale-result cancellation and stop behaviour, but cannot establish
model accuracy on new continuous motions. The model still selects one of six
axis directions at a fixed speed: it does not estimate actual finger velocity
or accurately reproduce a curved/diagonal path. The next dataset should include
continuous squares/zigzags, reversals and pauses with time-aligned direction and
idle labels, held out by recording session, before making those claims.

Reproduce the candidate with the existing trainer by adding `--continuous`:

```bash
python3 -m phd.script train_touch_gesture_policy --continuous \
  --output /new/continuous/run --device cuda
```

## Use

The following describes the original six-class single-gesture checkpoint.

1. Restart the UI to load the updated implementation. Connect the same 10×10
   sensor, with `row_to_col` reorder and averaging of 3 frames, and calibrate
   with the hand away.
2. Open **AI → Run → Touch Policy → Policy setup → Checkpoint**, and select
   **Select Model**. Choose `latest_cnn_tcn_touch_policy_10x10.pt`.
3. Enable **Dry run**, press **Execute**, and leave the sensor untouched for
   at least 0.12 seconds to arm. Then make one gesture per touch. The prediction
   shows the gesture, confidence, command direction and XYZ velocity.
4. Lift to stop. Each new gesture starts a new history. Stop execution before
   changing the sensor, its calibration/layout or its averaging setting.
5. After checking the directions in dry run, stop, uncheck **Dry run**, and
   press **Execute** to use the existing robot velocity path. Select the intended
   control frame in the UI; the axes below refer to that frame.

The mapping preserves the existing direct-finger command convention:

- `left` → X−; `right` → X+.
- `up` → Z−; `down` → Z+.
- `push` → Y−; `pull` → Y+.
- Rotational velocities are always zero.

At **1×**, commanded speed is **0.050 m/s (50 mm/s)** on the recognised axis.
The scale control multiplies that speed and is not capped. This is a
fixed-speed gesture command; it does not estimate the finger's physical speed
or pressing force. There is no software acceleration ramp.

Low confidence (below 0.80), a top-two probability margin below 0.15, or no
contact produces zero velocity. Two consecutive matching predictions are
required, including when switching between two swipe directions. Contact uses
the dataset's −3% averaged-difference threshold. Release stops on the first
fresh non-contact frame; 3 frames spanning 0.12 seconds re-arm a new episode.
A contact lasting over 2 seconds is stopped and requires release; sustained
holds and changing gestures without release were not included in this dataset.

Stale frames/predictions (0.2 seconds), changed calibration/layout/sensor,
invalid samples, frame-counter resets and frame-queue overflow stop execution.
The UI displays the reason. Start again once the cause is resolved.

## Data and provenance

To add trials, open **AI → Data → Gesture recording**, keep **Start on touch**,
and name the session `10x10_<gesture>` (for example `10x10_up`); the trainer reads
folders named `session_10x10_<gesture>`. From 27 October 2026 (dataset version 4)
a trial stores only tactile frames, timing and touch state. It no longer runs or
saves the rule-based DFM velocity, teaching labels or robot pose, none of which
this policy uses. The older CNN-GRU velocity trainer needs those fields and
cannot use version-4 trials.

All six folders are under
`phd/resource/ai/data/tactile_recordings/`:

- `session_10x10_up`, `session_10x10_down`, `session_10x10_left`,
  `session_10x10_right`, `session_10x10_pull`: 50 recordings each.
- `session_10x10_push`: 70 recordings.

All **320** recordings are used according to their folder label, including
recordings whose older metadata session tag differs from their current folder.
Their source files are unchanged. The proximity datasets are excluded.

Each class is split by capture time into 60% training, 20% validation and 20%
test: **192 / 64 / 64 whole trials**. Prefixes are extracted only after this
split. The manifest records source paths, hashes, labels and exact membership;
duplicate sensor sequences are rejected. Longer recordings and push's extra
trials do not dominate training: sampling balances classes and trials.

The historic column-major NPZ layout is transposed once to native row/column.
Inputs are instantaneous and averaged percentage change plus temporal change
of the average. Spatial-median subtraction, an asinh transform, and resampling
to 32 steps match the previous CNN-TCN gesture work. Normalization is fitted on
training prefixes only. A prefix needs at least 5 touch frames and 80 ms of
contact. Prefixes end at a contact frame; release cannot reveal the answer.
Convolutions may use all frames *within that past-only prefix*, but never future
frames from the recording. This is not a claim of causal padding inside the TCN.

Three CNN-TCN seeds (17, 43, 79) and their fixed ensemble were compared on
validation data. Seed 17, epoch 23, was selected by validation macro-F1/NLL.
Confidence temperature was also fitted on validation data. The test recordings
were evaluated after selection, without retraining or test-based parameter tuning.

## Results from 27 October 2026

- Final contact frame, before release: **64/64 correct**, macro-F1 **1.000**.
- All eligible partial-gesture observations: **1122/1124 correct (99.82%)**.
  These prefixes overlap; they are not 1,124 independent trials.
- Offline replay through the actual execution adapter: all **64** gestures
  produced the correct command before release; release stopped motion in every
  replay. There was **one brief wrong-direction command decision in one trial**
  before correction, out of 1,059 nonzero prediction decisions.
- Median first-command latency: **104 ms** after contact; 95th percentile
  **124 ms**, using recorded time and synchronous CPU inference. This excludes
  robot/network delays and is not a measured physical-robot latency.

These are held-out recordings from the same sensor/collection, not an independent
person/day/placement evaluation. Folder labels are assumed correct. Unknown
gestures, long holds and background are not trained classes; confidence screening
and release gating do not establish unknown-gesture recognition. Test fresh
recordings in dry run before relying on physical control. No hardware was moved
as part of training or replay.

## Files and reproduction

Training run:
`phd/resource/ai/models/touch_policy/cnn_tcn_touch_10x10_20261027_v1/`

- `best_touch_gesture_policy.pt`: selected CNN-TCN, labels, preprocessing,
  normalization, confidence calibration and velocity mapping.
- `dataset_manifest.json`: recording hashes and split assignments.
- `evaluation.json`, `test_predictions.json`, `test_confusion_matrix.png`:
  held-out classification results.
- `runtime_replay.json`: individual trial command/release results.
- `model_selection.json`, `candidates.json`, `cnn_tcn_seed*_history.json`:
  training arguments and validation-based selection.

The original six-gesture model remains available at
`phd/resource/ai/models/touch_policy/latest_cnn_tcn_touch_policy_10x10.pt`.
The old `latest_cnn_gru_model_10x10.pt` is preserved and can still be selected.
The separate 8×8 robot-dog classifier and proximity checkpoints are unchanged.

From the project root:

```bash
python3 -m phd.script train_touch_gesture_policy --output /new/run/directory --device cuda
python3 -m phd.script replay_touch_gesture_policy \
  --checkpoint /new/run/directory/best_touch_gesture_policy.pt \
  --manifest /new/run/directory/dataset_manifest.json \
  --output /new/run/directory/runtime_replay.json
```

Training and replay require new output paths. Training does not automatically
replace the deployed default. Use the bundled `best_touch_gesture_policy.pt`,
not an individual seed training artifact, when selecting a model in the UI.

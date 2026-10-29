# Ten-gesture tactile classifier

This is a **separate classification model**, not the older CNN-GRU robot-velocity
policy. It recognises a **completed touch-to-release trial**, using only the
sensor. Neither training nor prediction connects to the robot or camera.

## Current dataset

The October 20, 2026 collection contains 100 trials per gesture (1,000 total) from
an 8×8 sensor. Labels come from `session_robot_dog_*` folder names:

- Swipe up, down, left and right.
- Diagonal bottom-left to top-right, bottom-right to top-left, top-left to
  bottom-right and top-right to bottom-left.
- Clockwise and anticlockwise.

The existing folder spelling `anitclockwise` is accepted as `anticlockwise`.
The source folder is not renamed. No rule-based velocity, finger-count, position
or teaching labels are used as model inputs or training targets.

## Training and evaluation

Run from the repository root using the project Python environment:

```bash
python3 -m phd.script.training.train_robot_dog_gestures \
  --output /absolute/path/to/a/new/training-run-folder --device cuda
```

The output directory must be new; existing checkpoints are never overwritten.
By default the comparison trains CNN-GRU, CNN-TCN and CNN-Transformer, each with
three seeds. It also evaluates fixed seed-ensemble options on validation data.

For each class, recordings are ordered by capture time and split into the first
60% for training, middle 20% for validation and last 20% for the held-out test.
**Whole trials remain together.** Exact duplicate signal sequences are rejected.
The manifest records each input path, label, split and SHA-256 hash.

Sensor-only preprocessing:

1. Convert the recording's column-major layout to physical row/column layout.
2. Detect contact at the existing −3% averaged signal threshold; remove idle
   time outside the touch, retaining one release frame.
3. Subtract each frame's spatial median to reduce global baseline shifts.
4. Apply an `asinh` transform and resample the ordered motion to 32 time steps.
   Absolute gesture duration is not supplied to the classifier.
5. Use instantaneous percentage difference, averaged percentage difference and
   the averaged signal's temporal change. Fit channel normalization on training
   trials only.

Training augmentation changes amplitude, adds small noise, shifts the contact
pattern by less than one taxel and varies timing monotonically. It never flips,
rotates or reverses a gesture while keeping its old direction label.

Architecture, seed/ensemble and stopping epoch are selected using validation
macro-F1, with validation negative log-likelihood as the tie-breaker. Probability
temperature is also fitted on validation data. **The test set is not used for
selection or further tuning.** The saved model remains the evaluated model;
it is not retrained using the held-out trials afterwards.

## Use in the UI: AI → Run → Gesture recognition

1. Restart the UI after the update. Build/connect the **same 8×8 sensor** used
   for collection, and keep its orientation, reorder and averaging settings.
2. With your hand away, calibrate the sensor and check that it is streaming.
3. Open **AI → Run → Gesture recognition**. The selected CNN-TCN checkpoint
   is already the default; do not put it in the older Touch policy Execute field.
4. Press **Start recognition**, then keep your hand away until **Listening**.
5. Touch, perform one gesture, and lift. **Recording gesture** changes back to
   **Listening**, with the recognised name, confidence and top-two score lead.
   Repeat without pressing Start again. Low-confidence results say **Uncertain**.
6. Press **Stop** to finish. Sensor disconnect, recalibration, layout/averaging
   changes or a stream interruption stop recognition and discard incomplete trials.
   Restart it once the sensor is ready. Global **Stop All** also stops recognition.

**Details · scores, recent gestures and model** contains all ten scores, the last
ten results, checkpoint selection and **Test saved trial…** for offline NPZ replay.
No robot, force meter or camera is required. This page does not connect hardware,
save new training recordings, change calibration or send any robot commands.
It does not stop unrelated control modes; stop those yourself before testing if
you want the entire system to be motion-free.

Touch uses the trained −3% averaged-difference threshold. A release requires at
least three fresh frames spanning 0.12 seconds. Single-frame touches are ignored;
overlong gestures (30 seconds / 2,000 frames) are discarded until release. There
must be a clear release between gestures. Data is copied in native row/column
layout, and CPU model loading/inference runs outside the GUI thread.

## Predict a saved trial from the terminal

```bash
python3 -m phd.script.inference.predict_robot_dog_gesture \
  --model /absolute/path/to/best_gesture_model.pt \
  --trial /absolute/path/to/trial_0001_YYYYMMDD_HHMMSS.npz
```

The output includes the gesture, all ten probabilities, the best score,
top-two score margin and `accepted`. Default screening requires probability
≥0.80 and margin ≥0.15; otherwise the result is `uncertain`. Invalid/no-touch
trials are rejected. Those thresholds are conservative interface defaults,
**not a proven open-set safety guarantee**.

Python integration uses `GesturePredictor.predict_arrays(raw, averaged, times)`.
Here `raw` means **instantaneous signed percentage difference**, not absolute
capacitance; both arrays have shape `(time, rows, columns)`. The second array is
averaged percentage difference. Use the same baseline, sensor orientation and
averaging settings as collection. Buffer a whole touch episode and classify
after release. The predictor validates the trained sensor dimensions.

The current robot-velocity “Execute” control does not accept this checkpoint.
Use the separate Gesture recognition page above. Gesture-to-dog-command mapping
is not assigned or enabled.

## Reading results and limitations

- `best_gesture_model.pt`: selected weights, class names, preprocessing,
  normalization and temperature in one safe-loadable PyTorch bundle.
- `evaluation.json`: final held-out accuracy, macro-F1, per-class results,
  confusion matrix and binomial Wilson interval.
- `test_confusion_matrix.png`: whole-trial counts, true labels versus predictions.
- `dataset_manifest.json`: original file hashes and exact split membership.
- `candidate_comparison.json`, `model_selection.json` and individual history
  files: validation comparison and training provenance.
- Individual candidate `.pt` files are training artifacts; use the complete
  `best_gesture_model.pt` bundle for inference.

These are trials from the same collection session, sensor and presumed operator.
A good held-out result is **not** evidence of performance on another person,
day, placement or sensor. Folder labels are assumed correct. Idle/background,
other gestures, ambiguous touches and multi-finger gestures were not collected
as negative classes. Collect a fresh independent session and those negatives
before enabling physical robot actions. Confidence scores alone do not make
unknown gestures safe.

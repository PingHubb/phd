# Intention Learning: first camera-free research task

Open **AI → Data → Intention learning** after restarting the UI. The three
actions open a separate workbench: **Collect**, **Train & evaluate**, and
**Live preview**. Existing touch tracking and CIC/robot-policy workflows have
not been replaced. Opening the workbench does not open devices or send motion.

## What this first version measures

The question is: **given the tactile history and the available target positions,
which surface target does the person intend to reach now?**

For example, a participant starts near the centre, intends to reach A, moves
towards it, then changes their mind and heads towards B. The model predicts
`No current goal / A / B / C` at each usable time window. It can withhold the
prediction when its empirical confidence criterion is not met. Withholding is
not the same as predicting no goal.

This is a **surface target-selection prototype**, not yet a robot task-goal
inference system. The known context consists of all three target coordinates,
not robot pose, objects or a hidden chosen target. A/B/C are goals, not fixed
swipe names. Reassigning the letters between locations helps separate goal
selection from simply classifying a direction.

It does NOT independently measure finger position, speed, force, contact or
private mental decisions. Human declarations are approximate intention labels.
No camera is required for these goal labels, training, or deployment. Proximity
and robotic execution are intentionally outside this first version.

If two intentions produce indistinguishable tactile histories and candidate
context, this model cannot reliably tell them apart. More network layers do not
create missing information: retain those ambiguous cases and evaluate when the
system should withhold a prediction or request clarification.

The CNN–GRU + shared candidate scorer is a **baseline**, not a novelty claim.
Research value must come from a clear hypothesis and evidence: early prediction,
response to changed goals, generalization to people/sessions, and useful
assistance in a later robot experiment. A working implementation alone does not
establish cognition, novelty, or thesis-level performance.

## 1. Prepare the physical task

1. Keep robot motion and automated experiments stopped. Stream your chosen
   sensor using the existing Sensor workflow; calibrate with no nearby hand.
2. Give the physical sensor a stable ID, such as `woven_5x5_unit1`. Do not use
   `ttyACM6` as its identity: ports can change. This ID is an operator declaration,
   not an automatically verified hardware serial number.
3. Check native row/column orientation using Sensor signal. The workbench uses
   unflipped raw arrays: x = column/(columns−1), y = row/(rows−1). Row zero is at
   the top of the diagram. Physically identify that row and column on the fabric;
   do not assume the main heatmap uses the same display orientation.
4. Mark three reachable positions on/alongside the surface without changing the
   sensor's touch response: normalized (0.15, 0.20), (0.85, 0.20), (0.50, 0.85).
   These are approximate task targets, not calibrated robot coordinates. Place
   removable A/B/C labels according to the selected one of six assignments.
5. Confirm the orientation/target-location checkbox. Change layout only between
   trials, then physically check it again. Do not recalibrate mid-trial.

The full taxel array is recorded; no single-taxel selection limits the model.
This version trains one sensor identity, grid shape, model and reorder mode per
run. It does not claim arbitrary-sensor transfer.

## 2. Collect a small feasibility dataset first

Use anonymous participant IDs. A session is a genuine separate acquisition
session, not a name invented to get past the split check. Start with yourself to
check the workflow, but do not present that as unseen-person evaluation.

Within a trial:

1. Press **Start intention trial**. The initial label is Unknown, not an assumed
   goal or an automatically inferred no-touch state.
2. Press **No goal** and collect ordinary idle/resting-hand or incidental-touch
   examples where you genuinely do not intend to select A/B/C. No goal does not
   mean zero capacitance or no contact.
3. Decide the target, press its declaration button, then move your finger towards
   its physical position. For example: declare A, approach A at a comfortable
   speed, pause while still pursuing A, then explicitly declare No goal when
   the selection task is over.
4. Repeat with B and C. Vary starting position, route, speed, pressure and pause
   duration. Randomize order and timing; avoid a repeated A–B–C timed script.
5. Add genuine changes of intention, such as A → B partway through a movement.
   An assistant can click the declaration while you keep moving. Mark the time
   of the intended change, not when the finger finally arrives. Avoid an Unknown
   interval for a clearly timed switch if you want it counted in switch metrics.
6. Use **Unknown** when the goal or timing cannot be labelled confidently. Those
   endpoints are excluded from supervised training and evaluation.
7. Stop and save. Every trial has one folder containing raw chunks, manifest and
   `intention.json`. Select another layout and move the physical letter labels
   before the next trial.

A useful **initial engineering check**, not a powered study, is several short
trials in each layout, covering all four known states and some direct target
switches in every session. Review the recordings and dataset counts before
collecting a large volume. Do not record hours of an unbalanced fixed gesture
sequence. Frames within a trial are not independent samples.

For training/evaluation the software requires at least **three independent
participants** by default, one or more in each train/validation/test group.
Each resulting split must contain all four classes. That is a software minimum,
not a defensible final study size. A same-person feasibility pilot can choose
**Hold out sessions**, with at least three genuine sessions; its results support
only session transfer, not unseen-user generalization. Plan the final participant
count and repeated sessions with your supervisor after inspecting pilot variance.

Record participants across layouts, rather than assigning one target/layout to
each person. Retain hard/failed examples for diagnosis; don't quietly discard
them to improve scores. Failed or interrupted trials are retained on disk but
are not automatically used for training.

## 3. Check, train, and inspect evidence

Select the common dataset folder, then **Check dataset**. Resolve every invalid
trial before **Train & compare**. A data check validates individual recordings;
the training stage additionally enforces split-level class coverage, grouping,
sensor compatibility and duplicate-recording separation.

All runs use:

- Causal, past-only resampling at 20 Hz; default history is 20 frames (~1 s).
  Every resampled frame includes signed percent change, its rate of change,
  valid-taxel mask and sample age. Offline and preview preprocessing are shared.
- A 0.25 s exclusion around operator label changes, acknowledging timing
  ambiguity. This is a target/evaluation mask, not an input feature. A model
  window may contain a previous goal; otherwise switches could not be learned.
- Participant-held-out train/validation/test splits by default; no random window
  split. Raw duplicate detection rejects copies crossing the split.
- Normalization fitted only on training data. No chosen goal, declaration time,
  participant/session ID, filename, or absolute trial clock enters the network.
- Three comparisons: touch + candidate context (the predeclared main model),
  touch-only, and context-only. The fusion scorer shares weights across target
  candidates, so permuting A/B/C coordinates permutes their scores.
- Validation-based checkpoint choice, temperature scaling, and an empirical
  abstention threshold. If no threshold reaches 85% validation accepted-window
  accuracy with at least 30 accepted windows, preview withholds all predictions.
  Those correlated windows do not make this a statistical or safety guarantee.
- Test scores computed after the model and thresholds are frozen. The main model
  is not selected by its test score. All three comparisons use the same split.

Each immutable training-run folder contains `split.json`, `dataset_report.json`,
`run.json`, `summary.json`, three checkpoints and detailed model metric files.
The live-compatible checkpoint is `intention_preview.pt`.

Inspect accuracy, macro F1, class confusion, Brier score/NLL, accepted accuracy
**together with coverage**, and per-participant/per-trial results. High accepted
accuracy with almost no coverage is not a useful system. Macro F1 on a group
without a class includes a zero for that missing class; check its class support.

Temporal reports give counts of usable/correct/accepted-correct predictions at
0.5, 1 and 2 seconds after a declared goal, and response to direct A/B/C switches.
A switch succeeds only after an accepted correct goal remains stable for at
least 0.2 s, before the next declaration or 5 s. Misses remain explicit; median
latency is reported only for detected events. These are operator-event-relative
metrics, not claims about when the participant privately decided or touched.

If context-only performs well, investigate target frequency/layout shortcuts.
If fusion only succeeds at the destination, it may be detecting arrival rather
than anticipating intention: examine the early metrics and collection protocol.
Keep the held-out test locked during method development; repeated manual tuning
against it effectively turns it into validation and needs a new final test.

Command-line equivalent (run from the repository root):

```bash
python3 -m phd.script.training.train_intention check /path/to/intention_dataset
python3 -m phd.script.training.train_intention train /path/to/intention_dataset \
  --output /path/to/intention_models --group-by participant --epochs 30
```

## 4. Run sensor-only preview

1. Select the completed run's `intention_preview.pt` (filled in automatically
   after a successful training job).
2. Use the same physical sensor ID, model/order and native grid. Recalibrate with
   the hand away before starting, not during a movement.
3. Choose and physically verify the A/B/C layout. Do **not** declare the chosen
   target: the preview accepts no intention labels.
4. Start preview. It waits for a full fresh window, then shows four probabilities
   and the predicted or withheld goal. Stop preview to change layouts.

Missing frames, stale acquisition, an overloaded inference queue, source/order
changes, and calibration changes invalidate/reset predictions rather than
presenting a stale goal as current. Checkpoints use restricted weights-only
loading and a distinct schema. Old CIC/position-tracker files are rejected.

There is no robot command path in this model or workbench. It does not substitute
for the robot's emergency stop, stop other controllers, or prove safety. Connect
predictions to assistance only in a separately reviewed, bounded robot task with
an independent supervisor and an explicit user-enabled execution mode.

## Next research step, after this pilot works

Use pilot failures to choose a focused contribution rather than merely enlarging
the network. Candidate questions are whether task context resolves ambiguous
touch, whether a sequential belief model detects real goal changes earlier at
the same false-switch rate, and whether limited personal adaptation transfers
across users/days without losing uncertainty calibration. These are hypotheses,
not established novelty claims; check related work before selecting one.

Then add actual robot/task state (available targets, current pose and task phase)
as separately logged context, keep independent human-goal labels, and evaluate
assisted task completion against the baseline. Add proximity only with a separate
approach/contact protocol and independent labels appropriate to that claim.
Neither robot execution nor a trained real-data model was created automatically
by installing this workbench. Synthetic unit tests exercise software paths only.

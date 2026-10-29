# Resume Full Taxel Calibration

1. Press **Stop** and wait for the full-taxel summary to finish saving. Do not
   close/kill the UI during saving. Stop does **not** automatically retract the robot.
2. If installing this update while a run is already active, restart the UI only
   after stopping and saving. The old running process does not acquire new code.
3. Open **Experiments → Calibration → Full taxel → Resume Full Taxel Calibration…**.
4. Select `full_taxel_summary.json` in that experiment's folder. Loading is
   read-only: it does not connect devices or move the robot.
5. Review the completed/remaining trial counts, next pass/taxel and saved settings.
   Keep the same sensor, mounting, probe, robot base and tool configuration. Leave
   the probe unloaded, inside the mapped workspace, with the mapped tool orientation.
   Check the vertical path up to clearance and the route to the anchor, then confirm
   the setup checkbox. For sliding, this also requires checking the sliding path and
   supervising it.
6. Press **Resume Full Taxel Calibration**. Fresh sensor, robot and force feedback
   are required; force auto-connection remains available. If the probe is below the
   saved clearance, the robot first lifts straight up to that height, then follows
   the guarded route to the original anchor. It updates sensor calibration there
   and starts the next unfinished trial with a new force zero. A loaded probe
   (force at or above the route guard) is refused before any motion.

Completed trial CSV/JSON/PNG files stay unchanged. An unfinished trial restarts
from its safe start, not from its interrupted pressing position. Its old files
are moved to `resume_history/<timestamp>/interrupted_attempts/`. Prior summaries
are also backed up in that history directory. New data and updated summaries stay
in the **same experiment folder**. Full passes retain their original taxel order.
Completed trials that had force-stream pauses keep their quality flags; resuming
does not silently turn them into good-quality trials or repeat them.

Settings and the original map are locked while Resume is loaded. Automatic map
selection and travel replanning cannot replace them. **Clear resume · set up a
new run** leaves saved data untouched and returns to normal setup.

New runs store their complete resume settings and map snapshot. Pre-update
loading/unloading runs can recover the recorded settings from trial metadata and
the original map file, which must still be available and unchanged. The UI warns
that timing/safety options not recorded in older runs use current defaults.
The resume history records this migration and each new session; trial metadata
includes `measurement_quality.resume_session` (older files remain untouched).
Do not interpret a resumed experiment as one uninterrupted acquisition.

Full Taxel Calibration now records only the physically tested taxel in each
trial, including its raw, baseline and signed signal channels. Force, robot
position, timestamps, phase and quality metadata are retained. Normal Calibration
is unchanged. Proximity-profile reports still use the other channels internally
before the Full Taxel trial CSV is reduced; those other raw channels cannot be
reanalysed from that reduced CSV. Use a separate full-grid recording for crosstalk.
Resuming an older full-grid run is supported: completed files are not rewritten
by Resume, and newly acquired trials use the target-only format.

To reduce a stopped existing run, preview with
`python3 -m phd.script.conversion.compact_full_taxel_recording /exact/run/folder`.
Add `--apply` only to permanently discard its other-taxel columns. All trials
are staged and value-verified before replacement. The small
`.target_taxel_compaction/<timestamp>/verification.json` report records the
retained-value checksums and space removed. Its metadata backups do **not**
restore discarded signals. Reload Resume or Graph Analysis after conversion.

## Browse the saved taxels in Graph Analysis

Open Graph Analysis and select the full experiment summary **or any trial
CSV/JSON/PNG inside that full-taxel experiment**. The left map discovers saved
trials in the same experiment, including interrupted trials. Click a recorded
node to load that node's own force/signal/distance graphs and report. Grey nodes
have no readable saved trial. Partial trials are labelled; they are not treated
as completed calibrations. Clicking here never moves the robot.

The analysed-trial selector follows the selected node, keeping the same trial
number where possible. Use Compare Nodes to pin different nodes side by side.
Shared scales are calculated once from the experiment's recorded values; the
first opening can take a few seconds for a large run. Only the selected node's
trial arrays are retained for analysis, not the entire experiment. If files
change while the window is open, close and reopen Graph Analysis.

For safety, Resume refuses active runs, unclean shutdowns, missing completed data,
changed settings/maps, a different sensor, and fixed-baseline/drift experiments
whose baseline continuity would be lost. An already completed experiment needs no
resume. These cases must be inspected rather than silently skipped.

Verification is automated with simulated robot/force/tactile feedback; it is not
a physical robot validation. Supervise the first resumed trial and retain access
to the emergency stop.

# Taxel Position Mapping quality update — 26 October 2026

## Scope

Improves borderline-signal handling, centre repeatability and partial-map recovery.
No hardware was commanded, no saved experiment/map was rewritten, and no stored
speed, force threshold, scan height or travel allowance was changed.

The original source and changed legacy tests are backed up under
`archive/2026-10-26-mapping-quality/`. This is an incremental update on top of the
earlier NumPy-safe checkpoint/relocation-resume fix.

## Behaviour

1. **Keep the requested strong-signal target.** The automatic mode default remains
   2× the noise-derived onset threshold; explicit larger settings are preserved.
   At the depth boundary, a qualified signal at least 90% of that target gets a
   stationary check. A marginal pass between 100% and 110% also gets the check
   before XY scanning, even above the boundary. Clearly weaker signals at the
   boundary still fail without extra descent.
2. **Three disjoint stationary batches.** With the normal 20-frame setting this
   means 60 frames total, not three overlapping windows. Acceptance requires the
   combined median, at least two batch medians, and the final batch to meet the
   unchanged target, with valid downward response/dominance. Repeated sequence
   numbers do not count. A single spike or an older peak at a different Z cannot
   authorize the XY scan. Significant TCP drift during the hold stops the task.
3. **Repeat ambiguous centres once.** A broad/flat peak, dominance below 1.20,
   weak final response, or a successful borderline-strength check triggers a
   reverse-order scan of the exact same XY lines at the same Z. The repeat uses
   at least 20 stationary readings per point. The two estimated centres must
   agree within one configured local step (0.5 mm at the default setting), or
   twice the robot-position tolerance if larger. Final signal/dominance checks
   still apply. Disagreeing centres are rejected, not averaged. Clear centres
   retain the single-pass path. Repeatability is **not** absolute geometric
   accuracy; independent validation is still needed for that claim.
4. **Continue only from trustworthy geometry.** After bounded attempts, an
   ordinary signal/centre-quality failure can be marked unresolved after the
   clearance return, recovery check and full recalibration. Automatic-height
   mode requires a valid grid defined by verified reference neighbours before
   continuing. Frontier planning still refuses jumps across unknown cells.
   Essential anchor/neighbour checks and unfinished relocation references still
   stop on failure. Force/contact/staleness stops are not converted into skips.
   Existing explicitly enabled auto-reruns remain bounded; this update does not
   enable them.
5. **Incomplete means incomplete.** A pass with skipped failures saves
   `completed: false`, `stop_reason: partial_complete`, the unresolved IDs and
   failure reasons. Explicit Resume retries unresolved taxels with a new bounded
   attempt budget, retains measured centres and old failure evidence, and still
   verifies the anchor first.

## Recorded evidence

- JSON: `stationary_strength_checks`, `centre_repeat_checks`,
  `unresolved_taxels`, and per-position quality/repeatability details.
- `probe_levels.csv`: confirmation batch number, decision and combined median.
- `scan_samples.csv`: `local_repeat` identifies the second XY pass.
- `taxel_positions.csv`: repeat status and centre separation.
- Application Log/status: same-height confirmation progress, repeat scans,
  measured centre agreement and explicit unresolved/stop reasons.

Old maps remain readable. Old points are not retroactively labelled as having
passed the new checks. Resume history retains prior failures, while the existing
sensor-identity, orientation and anchor-alignment requirements remain in force.

## Unchanged safety limits and remaining limitation

No Z boundary, maximum descent, extra-descent allowance, workspace limit or
force-stop threshold was increased. Fresh force, robot and sensor feedback are
still required throughout confirmation and repeated scans. The conservative
5 g force-change stop remains active (or an earlier operator-configured guard).

More samples cannot create a real signal that is absent. For example, if a
reference taxel stays at 18.2 units while its target is 19.3, it must still stop
at the depth boundary. A changed probe/sensor setup needs physical clearance
verification; a zero force reading or a saved neighbour response plane alone
does not prove that deeper travel is non-contact. The software guards are not
certified protective equipment.

## Verification and use

383 tests passed in the combined mapping, experiment-task, map-autoload,
full-taxel-resume, characterisation-quality and robot-feedback-pause suites.
These include 22 new quality regressions and existing full-grid simulations.
`git diff --check` passed. Tests use simulated hardware; physical accuracy and
contact-free operation have not been validated on the robot by this update.

Restart the UI to load the rebuilt package. With the same sensor/probe setup,
load the partial mapping JSON and choose Resume; do not start a fresh map merely
to apply the software update. Supervise the initial resumed taxels and inspect
their results before considering a long scan. If the physical setup has changed
again, establish the correct safe anchor/relocation first.

# Mapping boundary and Resume repair — 26 October 2026

## Failure reproduced

The stopped Testing Sensor 3 anchor check reported TCP Z 238.758728 mm
against a working boundary of 238.761292 mm: 0.002563 mm below it.
The probe accepted an arrival within 0.1 mm, but the subsequent XY guard
refused more than 0.001 mm below the boundary. A sampled pose and a later
feedback pose could consequently disagree at the scan handoff.

Resume also loaded the anchor's historical measurement attempt count (3)
into its live retry budget, so the first check became attempt 4 despite a
two-attempt limit.

## Changes

- Automatic working depth steps now target the safe side of the active depth
  limit with twice the position tolerance as headroom (0.2 mm by default).
  This shortens, never extends, allowed downward travel. The boundary and
  force limits have not been relaxed. Operator-fixed-height behavior is
  unchanged.
- If a settled automatic probe or its XY handoff has an undershoot within
  the existing position tolerance, it may make **one vertical-only upward
  correction per probe**. It preserves current XY and orientation, discards
  previous strength batches, settles, and takes fresh readings at the raised
  height. Weak readings then trigger normal bounded recovery, not another
  descent. Larger/repeated undershoots also trigger normal recovery; failure
  to confirm the lift stops without scanning.
- Live force, robot-feedback and tactile freshness checks remain in effect
  during correction. A contact/force stop is not converted into an automatic
  lift. This software boundary is not an independently measured air gap.
- Explicit Resume starts a fresh bounded target-attempt budget, including
  anchor validation. Existing measured-position metadata is retained;
  previous live counts are preserved in `resume_history.previous_attempt_counts`.
  Subsequent failures within that resume still exhaust the configured limit.
  Automatic rerun limits/settings are unchanged.
- New snapshots record the boundary and resume accounting policies in settings.

## Verification and deployment

`test/test_mapping_boundary_resume.py` uses fake hardware exclusively and
covers the recorded undershoot, a stale sampled/current-pose mismatch, fresh
strong/weak measurements after correction, excessive/repeated undershoot,
failed lift confirmation, force/stale feedback stops, depth command headroom,
and two-attempt Resume accounting with retained history.

Validation: all 397 tests in the combined mapping, experiment, map-autoload,
full-taxel-resume, characterisation-quality and robot-feedback suites passed
(102.12 seconds). The ROS `phd` package rebuilt successfully; installed and
source `experiment_tasks.py` SHA-256 hashes match:
`75be6b12110cddb82879377c90eec7a4c11c70fbaf7f6aa659e258a391c0ef97`.
The test run reported existing plotting-library deprecation warnings.

The pre-repair source and touched pre-existing tests are preserved under
`archive/2026-10-26-mapping-boundary-resume/`. No live map, anchor or experiment
data is modified by this repair. The running UI must be restarted after the
ROS package rebuild to load the updated installed Python code. Resume the
existing map only with the same physical setup and a verified clear anchor.
No physical robot validation or automatic experiment restart is performed.

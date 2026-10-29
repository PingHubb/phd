# Mapping scan-plane selection / Resume repair — 26 October 2026

## Confirmed fault

`capture_scan_height()` only changed the in-memory `fixed_scan_z_m`.
`_apply_resume_settings()` then unconditionally replaced that value with
`settings.fixed_scan_z_mm` from the checkpoint. The Testing Sensor 3 checkpoint
had `null`, so Resume silently switched back to proximity-threshold descent.
The remembered height was also absent from the persistent anchor configuration.

## Changes

- Remember/Clear now atomically persist an explicit scan-plane choice inside the
  current sensor's anchor profile. Legacy anchor files remain readable.
- Resume gives that choice precedence over the checkpoint for a matching
  sensor, grid and anchor pose. Clear is an explicit choice, not a missing value.
- Remembering a new anchor invalidates its previous plane; switching sensors or
  layouts cannot transfer the height. A conflicting anchor on Resume is rejected
  before motion, with a relocation instruction.
- Fixed-plane Resume rebuilds the approach height above the selected plane.
  It does not approach using a potentially lower, old threshold-derived height.
- A changed plane discards the previous fixed-plane strength reference. The
  existing anchor measurement establishes a new one. Unchanged planes retain
  their saved reference.
- Effective scan mode/height are logged before motion, and the setting's source
  plus previous/current heights are recorded in the settings/resume history.
- Marking a plane requires fresh feedback, the existing XY workspace check and
  5 mm minimum anchor clearance, plus the existing 100 mm maximum travel bound
  applied to the anchor-to-plane distance. Persistence failures are reported.

No centre-repeatability tolerance, force guard, alignment check, or automatic
restart policy was relaxed. Existing measured map records are retained. A
marked plane does not certify an air gap and does not guarantee a successful
centre scan. The sensor placement, robot base and probe must still match the map.
The existing 1.5 mm XY / 2 mm Z anchor-resume alignment checks still apply.

## Operator handoff

Restart the UI to load the repaired code. For the existing map, the previously
lost height cannot be recovered from its `null` setting: mark the intended safe,
non-contact plane once again, then Resume Loaded Map. Do not remember a new
anchor merely to resume an unchanged setup. Confirm that the fixed-height label
and Application Log show the selected Z. Re-establish the setup/relocate if the
sensor, robot base or probe has changed. No robot motion was performed during
the repair.

## Verification

`test/test_mapping_scan_plane_resume.py` covers manual overrides, explicit Clear,
restart persistence, legacy maps, sensor/layout/anchor isolation, persistence
failure, stale/invalid feedback, approach replanning, reference invalidation,
full Resume startup and force-stop behaviour. Its simulated robot also reaches
the selected plane and collects a fresh reference, without opening devices.

Final verification: 421 tests passed across the scan-plane, boundary, mapping
quality/non-contact/recovery/checkpoint/fixed-plane/frontier/auto-rerun,
experiment lifecycle, map-autoload, full-calibration resume, characterisation
quality and robot-feedback suites (100.65 seconds). Only existing plotting
dependency deprecation warnings were reported. `colcon build --packages-select
phd --event-handlers console_direct+` succeeded; source and installed
`experiment_tasks.py` have matching SHA-256
`a39f0b7afae3e8f346c5501696bd7c523cc1a3b09565c5155f97418fcf6d1e59`.

The pre-edit source is retained in
`archive/2026-10-26-scan-plane-resume/experiment_tasks.py.before-fix`.

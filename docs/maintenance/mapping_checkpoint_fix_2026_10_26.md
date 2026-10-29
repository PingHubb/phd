# Relocation checkpoint repair — 26 October 2026

## Cause and changes

The intermediate rigid-fit dictionary held NumPy rotation/translation arrays.
`json.dump` failed after the CSVs had already been written. The valid JSON
therefore contained five measured taxels; the CSV and truncated JSON prefix
contained the newer six-taxel checkpoint.

- Encode NumPy arrays/scalars as JSON numbers/lists; unknown objects and
  non-finite values still fail. Validate the full JSON snapshot before writing
  any output, then replace the main JSON atomically.
- Persist a non-recursive source-map snapshot and original relocation reference
  IDs, including before the first fit. Resume reconstructs outstanding
  distributed verification targets, including a target interrupted in flight.
- Keep the existing anchor recheck, calibration, force/staleness guards, motion
  limits and completion checks. Do not treat relocation as ordinary full-grid
  mapping on resume. Legacy intermediate checkpoints lacking source state are
  rejected with a recovery explanation.

## Recovered recording

Run folder under `phd/resource/experiment/taxel_position_maps/`:

`taxel_position_map_relocated_20261026_171937_660466_sensor-Testing-Sensor-1_anchor-12_detection-robust-shift-win7-min6units_probe-3mmps_step-0p5mm_samples-20_scanstrength-2x_scanextra-20mm`

The main `taxel_position_map.json` now retains measured taxels **4, 7, 11, 12,
13, 17** and remains **incomplete**. Its original `checkpoint_failed` stop
reason and time are preserved; recovery time/provenance are separate. The fit
was recomputed from those measured centres and the explicit 23 October source
map. No new measurement or physical geometric correction was invented.

All six original run files and the source map were backed up, byte-for-byte,
under the run's `.checkpoint_recovery/20261026T100118029351Z/` directory. SHA-256
verification confirmed the CSVs, truncated temp JSON and source map themselves
are unchanged. Only the main run JSON was replaced. The recovery script is
`recover_mapping_checkpoint_2026_10_26.py`; it defaults to dry-run and refuses
application while the research UI process is open.

## Continue

Open/restart the UI, load this run's recovered `taxel_position_map.json`, and
select **Resume Loaded Map**, not a new Relocate run. With the same probe/TCP
and sensor setup as the failed run, it rechecks anchor taxel 12, then verifies
taxels 20 and 0 before completing the transform. This does not mean the entire
25-taxel map has already been physically verified. Changing the probe/TCP or
sensor again invalidates the assumption of unchanged geometry.

The operator reported a smaller-diameter, longer tip of different material.
Check physical tip clearance before resuming; this software repair did not
validate the TCP, change robot poses, change scan heights or relax limits.

## Verification and rollback

- 18 new checkpoint/resume regression tests passed.
- 138 mapping-focused tests passed, including those 18.
- Broader initial run: 298 passed, one existing calibration test failed:
  `test_stepwise_contact_uses_a_zero_re_taken_near_the_surface`. It also failed
  independently against the pre-fix installed module. No calibration logic
  was changed in this repair.
- `colcon build --packages-select phd` succeeded. Source and installed
  `experiment_tasks.py` hashes match. The installed code validates the recovered
  map and restores all six measured positions plus remaining queue `[20, 0]`.
- No hardware was accessed or moved; all motion tests used fake robot APIs.

Pre-fix source is retained in
`archive/2026-10-26-mapping-checkpoint/experiment_tasks.py.before-fix`.
For rollback, close the UI first. Restore code/data only from the appropriate
backups and rebuild. Do not roll back over any subsequent experiment progress.

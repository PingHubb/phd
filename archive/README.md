# Archive and recoverable originals

- `2026-10-19/`: previously retired code/model/mesh material. Its existing
  `manifest.json` records original locations and checksums. It is not active
  runtime code; do not import it into the application.
- `2026-10-23-organization/originals/`: pre-organisation copies of the moved
  guides and standalone tools, plus `setup.py`. These capture the actual
  modified working files, not an older Git revision.

The current move/hash record is
[docs/maintenance/organization_2026-10-23.json](../docs/maintenance/organization_2026-10-23.json).
Original files can be restored from the dated directory if needed. Restore
related files consistently and rebuild; do not overwrite unrelated current
work or experiment data. Nothing was permanently deleted by this organisation.

`COLCON_IGNORE` prevents this directory from being treated as a ROS workspace.

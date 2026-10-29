# Research assets and runtime resources

- [experiment/](experiment/README.md): mapping, calibration and saved scans.
- [ai/](ai/README.md): training data, checkpoints and benchmark results.
- [config/](config/README.md): sensor/device settings, presets and remembered poses.
- [paper/](paper/README.md): manuscripts, LaTeX source folders and upload ZIPs.
- `sensor/`, `robot/`, `humanoid/`, `dexterous_hand/`: geometry/model assets.
- `icon/`, `stylesheets/`: UI artwork and styles.

The repository-root `resource/phd` file is a ROS package-index marker; it is
unrelated to this asset directory and must remain in its original location.

Calibration recordings, taxel position maps and scanning recordings live only
under `experiment/`; code and saved JSON/CSV paths point there. A
`calibration_recordings`, `taxel_position_maps` or
`sigraph2026_scanning_recordings` folder appearing directly here means an old
build wrote to the former location: move its contents into `experiment/`.
Moving the whole repository to another machine can still require explicit
resource-root overrides and relocation/validation of hardware pose records.

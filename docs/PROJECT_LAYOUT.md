# Project layout and file ownership

Paths below are relative to the repository root containing `setup.py` and
`package.xml`, normally `/home/ping2/ros2_ws/src/phd`.

## Main folders

```text
phd/                         repository root
├── README.md                start here
├── docs/
│   ├── guides/              maintained research/workflow guides
│   └── maintenance/         move records and verification metadata
├── phd/                     importable Python package
│   ├── phd_ui.py            desktop entry point
│   ├── ui/                  Qt UI and experiment workflows
│   ├── dependence/          devices, data, models and control policies
│   ├── script/
│   │   ├── training/        train models
│   │   ├── analysis/        validate datasets, analyse trials, benchmark
│   │   ├── inference/       offline predictions
│   │   ├── hardware/        explicit hardware/permission utilities
│   │   └── conversion/      CAD-to-CSV utilities
│   └── resource/
│       ├── experiment/     one folder per experiment run
│       ├── ai/             datasets, checkpoints and benchmark outputs
│       ├── config/         mutable device/robot/sensor settings
│       ├── paper/          manuscripts, sources and submission ZIPs
│       └── …               sensor/robot/hand meshes, icons and styles
├── test/                    automated tests; original test paths retained
├── archive/                 historical work and recoverable originals
├── resource/phd             ROS package-index marker; not research data
├── setup.py                 Python packages and named ROS/CLI commands
└── package.xml              ROS package metadata
```

## What stayed in place, and why

- `phd/ui/` and `phd/dependence/` have stable imports and lazy-loaded modules.
  Moving them would be a runtime refactor, not just tidying. Their README maps
  group modules by responsibility without changing control logic.
- Datasets, checkpoint names, paper filenames, recorded map positions and
  `resource/config/` files are unchanged. They can be referenced by running
  sessions, saved JSON, training manifests and checkpoints.
- Flat `phd/script/*.py` files forward to the grouped implementations. Keep
  them: UI training buttons and older imports/direct-script commands use them.
- Former document paths are short pointers to the maintained guides.
- Calibration recordings, taxel position maps and scanning recordings exist
  only under `resource/experiment/`. The former `resource/<folder>` aliases
  were removed; no code or saved JSON/CSV path used them.
- `venv/`, `.git/` and IDE files are not moved or deleted. Some embed absolute
  paths or may be in use. They are environment material, not the main source
  tree.

Build, install and log output belongs only to the workspace root,
`/home/ping2/ros2_ws/{build,install,log}`. Old colcon output inside this
repository (`build/`, `install/`, `log/` and `phd/{build,install,log}`) and an
empty detector `phd/runs/` folder were deleted on 23 Oct 2026; none was tracked
or used. Always build from `/home/ping2/ros2_ws`, never inside `src/phd` or
`phd/phd`, and do not edit installed/build copies as your source of truth.

## Where to put new work

- New standalone tools belong in the appropriate `phd/script/` subpackage.
  Add them to `phd/script/catalog.py` if they should appear in the command list.
- New workflow guides belong in `docs/guides/` and should be linked from
  `docs/README.md`.
- Keep each experiment's CSV, JSON, graphs and diagnostics together in its
  existing run folder. Do not rename an active run or rewrite its provenance
  just to match a new naming preference.
- Put training runs under the relevant `resource/ai/models/<task>/` and
  benchmarks under `resource/ai/benchmarks/`; retain their original manifests.
- Keep device settings in `resource/config/`, not mixed with source scripts.
- Put retired material in a dated archive with a manifest, not an ambiguous
  `old`, `final2` or `backup_new` directory.

## Commands and resource roots

The desktop launch remains `ros2 run phd phd_ui`. Existing named CLI commands
also remain unchanged. To browse tools without loading Qt, ML or hardware:

```bash
python3 -m phd.script list
python3 -m phd.script paths
```

`paths` reports the roots resolved by the **currently imported package**.
Source, installed and external resources may differ. Existing
`PINGLAB_RESOURCE_ROOT` and `PINGLAB_AI_RESOURCE_ROOT` overrides still apply;
this organisation does not change either override or migrate installed assets.

## Recovery and verification

Pre-organisation copies of every moved guide/tool and `setup.py` are retained
in `archive/2026-10-23-organization/originals/`. The move manifest records old
and new locations and SHA-256 hashes. No hardware was operated for this cleanup.
Tests verify old/new module identity, unchanged function/class syntax, direct
script launches, path resolution, document links and protected runtime files.

If rolling back, stop relevant commands first and restore matching originals
from the dated archive; do not use a hard Git reset in this already modified
worktree. Restore the whole related tool group and `setup.py` consistently,
then rebuild the ROS package. Do not restore/delete experiment data as part of
a source-layout rollback.

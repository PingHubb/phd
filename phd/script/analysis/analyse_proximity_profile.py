# Grouped implementation; the former flat script path remains supported.
"""Re-analyse a saved Calibration "Proximity profile" trial.

Usage:
    analyse_proximity_profile path/to/trial_0001.csv [--k 3 5] [--signal diff_ave]
        [--common-mode-distance 6] [--suffix _v2]

Writes ``<trial><suffix>_proximity_profile.{json,csv,png}`` and
``<trial><suffix>_proximity_levels.csv`` next to the trial and prints the report.
Without ``--suffix`` the existing analysis files are replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from phd.ui.proximity_profile import (
    DEFAULT_COMMON_MODE_DISTANCE,
    analyse_proximity_profile,
    report_lines,
    save_profile_outputs,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trial_csv", type=Path)
    parser.add_argument("--k", type=float, nargs="+", default=[3.0, 5.0],
                        help="Detection thresholds in multiples of single-frame noise")
    parser.add_argument("--signal", default=None, help="Signal field (default: the recorded one)")
    parser.add_argument("--common-mode-distance", type=int, default=DEFAULT_COMMON_MODE_DISTANCE,
                        help="Taxels at least this many cells from the target form the common-mode reference")
    parser.add_argument("--suffix", default="", help="Appended to the output file stem to keep earlier results")
    args = parser.parse_args(argv)
    metadata_path = args.trial_csv.with_suffix(".json")
    if not metadata_path.is_file():
        raise SystemExit(f"Missing trial metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    settings = (metadata.get("measurement_quality") or {}).get("proximity_profile_settings") or {}
    with args.trial_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows_n, columns_n = metadata["sensor_shape"]
    analysis = analyse_proximity_profile(
        rows, rows_n=rows_n, columns_n=columns_n, target=metadata["taxel_index"],
        signal_field=args.signal or metadata.get("signal_field", "diff_ave"),
        force_threshold_g=float(settings.get("surface_contact_threshold_g", 1.0)),
        k_values=tuple(args.k), common_mode_distance=args.common_mode_distance,
    )
    files = save_profile_outputs(analysis, args.trial_csv.parent, args.trial_csv.stem + args.suffix)
    print("\n".join(report_lines(analysis)))
    print(f"Files: {', '.join(files.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

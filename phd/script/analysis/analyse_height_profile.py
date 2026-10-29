# Grouped implementation; the former flat script path remains supported.
"""Summarise a Calibration "Proximity approach" trial as a height profile.

Usage:
    analyse_height_profile path/to/trial_0001.csv [--bin-mm 0.5]

Writes ``<trial>_height_profile.csv`` (and a PNG when matplotlib is present)
next to the trial and prints the recommended non-contact scan height.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from phd.dependence.height_profile import analyse_height_profile


def _load(trial_csv: Path):
    metadata_path = trial_csv.with_suffix(".json")
    if not metadata_path.is_file():
        raise SystemExit(f"Missing trial metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with trial_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return metadata, rows


def _write_outputs(trial_csv: Path, result: dict) -> Path:
    out_csv = trial_csv.with_name(trial_csv.stem + "_height_profile.csv")
    fields = list(result["profile"][0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["profile"])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return out_csv
    travel = [level["travel_mm"] for level in result["profile"]]
    figure, (signal_axis, dominance_axis) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    signal_axis.plot(travel, [level["target_signal"] for level in result["profile"]], label="target")
    signal_axis.plot(travel, [level["strongest_neighbour"] for level in result["profile"]], label="strongest neighbour")
    signal_axis.set_ylabel("signal change (counts)")
    signal_axis.legend()
    dominance_axis.plot(travel, [level["dominance"] for level in result["profile"]])
    dominance_axis.axhline(result["criteria"]["min_dominance"], linestyle="--", color="grey")
    dominance_axis.set_ylabel("target / strongest other")
    dominance_axis.set_xlabel("travel down from start (mm)")
    if result["contact_travel_mm"] is not None:
        for axis in (signal_axis, dominance_axis):
            axis.axvline(result["contact_travel_mm"], color="red", linestyle=":")
    figure.tight_layout()
    figure.savefig(trial_csv.with_name(trial_csv.stem + "_height_profile.png"), dpi=120)
    plt.close(figure)
    return out_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trial_csv", type=Path)
    parser.add_argument("--bin-mm", type=float, default=0.5)
    args = parser.parse_args(argv)
    metadata, rows = _load(args.trial_csv)
    rows_n, columns_n = metadata["sensor_shape"]
    result = analyse_height_profile(
        rows,
        rows_n,
        columns_n,
        metadata["taxel_index"],
        signal_field=metadata.get("signal_field", "diff_ave"),
        bin_mm=args.bin_mm,
    )
    out_csv = _write_outputs(args.trial_csv, result)

    print(f"Taxel {result['target_taxel']} (neighbours {result['neighbours']}), "
          f"baseline noise {result['baseline_noise_counts']:.2f} counts")
    contact = result["contact_travel_mm"]
    print("First force change "
          + (f">= {result['contact_force_g']:.0f} g at {contact:.2f} mm travel" if contact is not None
             else "not reached"))
    print(f"{'travel':>7} {'Z mm':>7} {'target':>7} {'nbr':>6} {'dom':>5} {'SNR':>5} {'force g':>7}")
    for level in result["profile"]:
        z = "" if level["tool_z_mm"] is None else f"{level['tool_z_mm']:.2f}"
        force = "" if level["force_g"] is None else f"{level['force_g']:.1f}"
        print(f"{level['travel_mm']:7.2f} {z:>7} {level['target_signal']:7.1f} "
              f"{level['strongest_neighbour']:6.1f} {level['dominance']:5.2f} "
              f"{level['snr']:5.1f} {force:>7}")
    best = result["recommended_non_contact"]
    if best is None:
        print("No non-contact height reached the minimum signal-to-noise ratio.")
    else:
        print(f"Best non-contact height: Z {best['tool_z_mm']:.2f} mm "
              f"(travel {best['travel_mm']:.2f} mm), dominance {best['dominance']:.2f}, "
              f"SNR {best['snr']:.1f}")
    at_contact = result["best_at_contact"]
    if at_contact is not None:
        print(f"Best at light contact: dominance {at_contact['dominance']:.2f}, "
              f"signal {at_contact['target_signal']:.1f}")
    print("Non-contact mapping looks feasible." if result["non_contact_feasible"]
          else "Non-contact mapping is NOT recommended for this sensor at these heights.")
    print(f"Profile written to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

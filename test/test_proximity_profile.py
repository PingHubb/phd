import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from phd.ui.calibration_analysis import taxel_signal_column
from phd.ui.proximity_profile import (
    analyse_proximity_profile,
    common_mode_taxels,
    detection_distance,
    report_lines,
    surface_reference,
)


def test_detection_distance_interpolates_beyond_and_not_detected():
    gaps = [1.0, 2.0, 4.0, 8.0]

    crossing = detection_distance(gaps, [40.0, 20.0, 10.0, 2.0], 6.0)
    assert crossing["status"] == "interpolated"
    assert np.isclose(crossing["distance_mm"], 4.0 + 4.0 * 4.0 / 8.0)
    assert detection_distance(gaps, [40.0, 20.0, 10.0, 7.0], 6.0)["status"] == "beyond_farthest_level"
    assert detection_distance(gaps, [5.0, 2.0, 1.0, 0.0], 6.0)["distance_mm"] is None


def _probe_rows(surface_z, probe, final=False, stiffness_g_per_mm=50.0):
    rows = []
    for index in range(60):
        z_value = surface_z + 0.0003 - index * 0.00001
        force = max(0.0, (surface_z - z_value) * 1000.0 * stiffness_g_per_mm)
        rows.append({
            "elapsed_s": probe * 10 + index * 0.05, "profile_kind": "surface_probe",
            "profile_probe": probe, "profile_probe_final": final,
            "tool_z_m": z_value, "force_delta_abs_g": force,
        })
    return rows


def test_surface_reference_uses_crossing_and_reports_zero_force_offset():
    surface_z = 0.14
    rows = _probe_rows(surface_z, 0) + _probe_rows(surface_z, 1) + _probe_rows(surface_z + 0.00002, 0, final=True)

    surface = surface_reference(rows, 1.0)

    # 1 g at 50 g/mm is 0.02 mm below the zero-force surface.
    assert abs(surface["surface_z_m"] - (surface_z - 0.00002)) < 0.000011
    assert abs(surface["zero_force_offset_mm"] - 0.02) < 0.011
    assert abs(surface["final_shift_mm"] - 0.02) < 0.011


def test_stepped_probe_crossing_on_the_last_step_is_interpolated():
    rows = []
    for index, (z_value, force) in enumerate(((0.14003, 0.0), (0.14002, 0.0), (0.14001, 0.4), (0.14000, 2.4))):
        rows.append({
            "elapsed_s": index, "profile_kind": "surface_probe", "profile_probe": 0,
            "profile_probe_final": False, "tool_z_m": z_value, "force_delta_abs_g": force,
        })

    surface = surface_reference(rows, 1.0)

    # 1 g lies 30% of the way from the 0.4 g step to the 2.4 g step.
    assert abs(surface["surface_z_m"] - (0.14001 - 0.3 * 0.00001)) < 1e-9


def _window(step, kind, elapsed, gap_mm, target, other, sweep=1, direction="down", frames=20, seed=0):
    rng = np.random.default_rng(seed + step)
    rows = []
    for index in range(frames):
        rows.append({
            "elapsed_s": elapsed + index * 0.05, "profile_kind": kind, "profile_step": step,
            "profile_sweep": sweep, "profile_direction": direction if kind == "level" else "",
            "profile_target_gap_mm": gap_mm, "tool_z_m": 0.14 + gap_mm / 1000.0,
            "force_delta_abs_g": 0.0,
            taxel_signal_column("diff_ave", 0): target + rng.normal(0.0, 0.5),
            taxel_signal_column("diff_ave", 1): other + rng.normal(0.0, 0.5),
        })
    return rows


def test_profile_removes_linear_baseline_drift_between_references():
    surface_rows = _probe_rows(0.14 + 0.00002, 0)
    drift = 0.1  # counts per second
    rows = list(surface_rows)
    step = 0
    true_drop = {1.0: 40.0, 2.0: 20.0, 5.0: 8.0}
    timeline = [("reference", 20.0, 0.0)]
    for gap in (5.0, 2.0, 1.0):
        timeline.append(("level", gap, true_drop[gap]))
        timeline.append(("reference", 20.0, 0.0))
    for kind, gap, drop in timeline:
        elapsed = 100.0 + 50.0 * step
        baseline = drift * (elapsed + 0.5)
        rows += _window(step, kind, elapsed, gap, baseline - drop, baseline - 0.2 * drop)
        step += 1

    analysis = analyse_proximity_profile(rows, rows_n=1, columns_n=2, target=0, force_threshold_g=1.0)

    by_gap = {row["target_gap_mm"]: row for row in analysis["profile"]}
    for gap, drop in true_drop.items():
        assert abs(by_gap[gap]["departure_mean"] - drop) < 0.5
        assert abs(by_gap[gap]["dominance_median"] - 5.0) < 1.0
    assert analysis["reference_drift"]["range"] > 10.0
    assert 0.3 < analysis["single_frame_sigma"] < 0.8
    assert any(line.startswith("Detection distance at 3 sigma") for line in report_lines(analysis))


def test_common_mode_taxels_use_column_major_cells_and_shrink_on_small_sensors():
    far, cells = common_mode_taxels(20, 20, 189)
    assert cells == 6 and 189 not in far
    assert all(max(abs(index % 20 - 9), abs(index // 20 - 9)) >= 6 for index in far)

    # Taxel 35 of an 8-row sensor is row 3, column 4; only column 9 is 5 cells away.
    far, cells = common_mode_taxels(8, 10, 35)
    assert cells == 5 and far == list(range(72, 80))

    assert common_mode_taxels(5, 5, 12) == ([], None)


_GRID_ROWS, _GRID_COLUMNS = 4, 10  # Target 0 is row 0, column 0; columns 6–9 are 6+ cells away.


def _grid_rows(drops, sweeps, shifted_sweep=None, shift=6.0):
    """Target drops per gap; every taxel also drops by ``shift`` between references in one sweep."""
    rng = np.random.default_rng(1)
    rows = _probe_rows(0.14 + 0.00002, 0)
    step = 0
    for sweep in range(1, sweeps + 1):
        timeline = [("reference", 30.0, 0.0)] + [("level", gap, drop) for gap, drop in drops.items()]
        timeline.append(("reference", 30.0, 0.0))
        for kind, gap, drop in timeline:
            common = shift if kind == "level" and sweep == shifted_sweep else 0.0
            for index in range(20):
                row = {
                    "elapsed_s": 100.0 + 30.0 * step + index * 0.05, "profile_kind": kind,
                    "profile_step": step, "profile_sweep": sweep,
                    "profile_direction": "down" if kind == "level" else "",
                    "profile_target_gap_mm": gap, "tool_z_m": 0.14 + gap / 1000.0, "force_delta_abs_g": 0.0,
                }
                for taxel in range(_GRID_ROWS * _GRID_COLUMNS):
                    level = -drop if taxel == 0 else 0.0
                    row[taxel_signal_column("diff_ave", taxel)] = level - common + rng.normal(0.0, 0.5)
                rows.append(row)
            step += 1
    return rows


_DROPS = {1.0: 40.0, 2.0: 20.0, 4.0: 10.0, 8.0: 5.0, 16.0: 2.5, 24.0: 0.5}


def test_localized_range_rejects_a_shared_shift_that_references_miss():
    rows = _grid_rows(_DROPS, sweeps=3, shifted_sweep=2)

    analysis = analyse_proximity_profile(rows, rows_n=_GRID_ROWS, columns_n=_GRID_COLUMNS, target=0)

    common = analysis["common_mode"]
    assert common["available"] and common["distance_cells"] == 6 and common["taxels"] == 16
    by_gap = {row["target_gap_mm"]: row for row in analysis["profile"]}
    for gap, drop in _DROPS.items():
        assert abs(by_gap[gap]["localized_mean"] - drop) < 0.5
        assert by_gap[gap]["departure_mean"] > drop + 1.0  # Target-only keeps the sweep-2 shift.
    target = analysis["detection"]["3_sigma"]
    localized = analysis["detection_localized"]["3_sigma"]["per_sweep_mm"]
    assert target["per_sweep_status"][1] == "beyond_farthest_level"
    assert max(target["per_sweep_mm"]) - min(target["per_sweep_mm"]) > 3.0
    assert len(localized) == 3 and max(localized) - min(localized) < 1.5
    entry = analysis["detection_localized"]["3_sigma"]
    assert entry["per_sweep_ci95_mm"][0] < entry["per_sweep_mean_mm"] < entry["per_sweep_ci95_mm"][1]
    assert entry["bootstrap_ci95_mm"] is None
    assert any(warning.startswith("Only 3 sweep(s)") for warning in analysis["warnings"])
    assert any("between levels 16.0 and 24.0 mm" in warning for warning in analysis["warnings"])
    lines = report_lines(analysis)
    assert any(line.startswith("Localized detection distance at 3 sigma") for line in lines)
    assert any(line.startswith("Target strongest taxel") for line in lines)


def test_bootstrap_interval_needs_five_sweeps():
    analysis = analyse_proximity_profile(
        _grid_rows(_DROPS, sweeps=5), rows_n=_GRID_ROWS, columns_n=_GRID_COLUMNS, target=0,
    )

    entry = analysis["detection_localized"]["5_sigma"]
    assert entry["bootstrap_ci95_mm"] is not None and len(entry["per_sweep_mm"]) == 5
    assert not any(warning.startswith("Only") for warning in analysis["warnings"])


def test_profile_without_surface_contact_is_reported_not_raised():
    analysis = analyse_proximity_profile([], rows_n=1, columns_n=2, target=0)

    assert analysis["surface"]["surface_z_m"] is None
    assert "profile" not in analysis
    assert any("Warning" in line for line in report_lines(analysis))


def test_profile_result_dialog_opens_without_hardware():
    script = """
from PyQt5.QtWidgets import QApplication, QPlainTextEdit
from phd.ui.proximity_profile_dialog import ProximityProfileResultDialog
app = QApplication([])
dialog = ProximityProfileResultDialog({"proximity_profile": {"surface": {}, "warnings": ["x"]}})
assert "PROXIMITY PROFILE" in dialog.findChild(QPlainTextEdit, "proximityProfileReport").toPlainText()
dialog.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

import csv

import numpy as np

from phd.ui.calibration_analysis import (
    _smooth_polynomial_fit,
    analyze_calibration_rows,
    available_taxel_indices,
    calibration_rows_for_taxel,
    read_calibration_csv_rows,
    release_trial_rows,
    taxel_signal_column,
    with_trial_rows,
)


def test_taxel_columns_share_one_key_string_per_column():
    assert taxel_signal_column("diff_ave", 7) is taxel_signal_column("diff_ave", 7)


def test_released_trials_reload_from_csv_with_or_without_the_taxel_grid(tmp_path):
    rows = [
        {"elapsed_s": 0.0, "phase": "approaching", "diff_ave": -1.5,
         "taxel_000_diff_ave": -1.5, "taxel_001_raw": 900.0, "note": ""},
        {"elapsed_s": 0.05, "phase": "returning", "diff_ave": -2.0,
         "taxel_000_diff_ave": -2.0, "taxel_001_raw": 901.0, "note": ""},
    ]
    paths = []
    for number in (1, 2):
        path = tmp_path / f"trial_{number:04d}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    trials = [
        {"trial_number": number, "rows": list(rows), "csv_path": str(path)}
        for number, path in zip((1, 2), paths)
    ]
    batch = {"trial_results": trials, "rows": list(rows)}

    release_trial_rows(batch)

    assert batch["rows"] == [] and all(trial["rows"] == [] for trial in trials)
    assert trials[0]["row_count"] == 2
    loaded = with_trial_rows(batch)
    assert loaded["trial_results"][1]["rows"][1] == {
        "elapsed_s": 0.05, "phase": "returning", "diff_ave": -2.0,
        "taxel_000_diff_ave": -2.0, "taxel_001_raw": 901.0, "note": None,
    }
    assert loaded["rows"] == loaded["trial_results"][-1]["rows"]
    assert trials[0]["rows"] == []  # Reloading does not re-fill the kept result.
    assert read_calibration_csv_rows(paths[0], taxel_columns=False)[0] == {
        "elapsed_s": 0.0, "phase": "approaching", "diff_ave": -1.5, "note": None,
    }


def test_unsaved_trials_keep_their_samples():
    trial = {"rows": [{"elapsed_s": 0.0}], "csv_path": ""}
    release_trial_rows(trial)
    assert trial["rows"] == [{"elapsed_s": 0.0}]


def _row(elapsed, phase, distance, signal, force_g):
    return {
        "elapsed_s": float(elapsed),
        "phase": phase,
        "travel_down_mm": float(distance),
        "diff_ave": float(signal),
        "force_delta_abs_g": float(force_g),
    }


def test_calibration_analysis_detects_proximity_then_contact_and_fits_loading():
    rows = [
        _row(i * .1, "approaching", i * .1, -20 * max(0, i - 4), 20 * max(0, i - 19))
        for i in range(30)
    ]

    analysis = analyze_calibration_rows(rows, "diff_ave")

    assert analysis["signal_onset"]["distance_mm"] < analysis["contact_onset"][
        "distance_mm"
    ]
    assert analysis["signal_onset"]["signal"] < 0.0
    assert analysis["contact_onset"]["force_g"] >= 5.0
    assert analysis["signal_distance_fit"]["slope"] < 0.0
    assert analysis["force_distance_fit"]["slope"] > 0.0
    assert analysis["force_signal_fit"]["slope"] < 0.0
    assert np.isfinite(analysis["force_signal_fit"]["r_squared"])


def _transition_rows(start_inside=False, rise=True):
    """Stationary baseline, proximity drop, contact, turning point, then rise."""
    rows = [
        {
            "elapsed_s": index * 0.05,
            "phase": "force_baseline",
            "travel_down_mm": 0.0,
            "diff_ave": float((-1) ** index),
            "force_delta_abs_g": 0.0,
            "force_delta_g": 0.0,
        }
        for index in range(40)
    ]
    for index in range(1301):
        distance = index * 0.01
        penetration = distance - 10.0
        if penetration <= 0.0:
            signal = (
                -60.0 * distance / 10.0 if start_inside
                else -100.0 * max(0.0, distance - 5.0) / 5.0
            )
        elif not rise or penetration < 0.25:
            signal = -100.0 - 200.0 * penetration
        else:
            signal = -150.0 + 600.0 * (penetration - 0.25)
        force_g = 100.0 * max(0.0, penetration)
        rows.append(_row(2.0 + index * 0.05, "approaching", distance, signal, force_g))
    return rows


def test_calibration_targets_range_turning_point_and_baseline_return():
    analysis = analyze_calibration_rows(_transition_rows(), "diff_ave")

    proximity = analysis["proximity_range"]
    assert proximity["status"] == "measured"
    assert 4.0 < proximity["distance_above_contact_mm"] < 5.0
    turning = analysis["turning_point"]
    assert turning["detected"]
    assert abs(turning["force_g"] - 25.0) < 2.0
    assert turning["force_low_g"] <= turning["force_g"] <= turning["force_high_g"]
    assert abs(turning["after_contact_mm"] - 0.25) < 0.03
    assert abs(analysis["baseline_return"]["force_g"] - 50.0) < 2.0
    assert np.isclose(analysis["force_signal_rise_fit"]["slope"], 1 / 6, rtol=0.05)
    assert np.isclose(analysis["force_signal_drop_fit"]["slope"], -0.5, rtol=0.1)


def test_calibration_range_is_a_lower_bound_when_the_start_is_already_in_range():
    analysis = analyze_calibration_rows(_transition_rows(start_inside=True), "diff_ave")

    proximity = analysis["proximity_range"]
    assert proximity["status"] == "start_inside_range"
    assert proximity["distance_above_contact_mm"] is None
    assert np.isclose(proximity["lower_bound_mm"], proximity["start_to_contact_mm"])


def test_calibration_turning_point_needs_a_rise_after_the_minimum():
    analysis = analyze_calibration_rows(_transition_rows(rise=False), "diff_ave")

    assert not analysis["turning_point"]["detected"]
    assert analysis["turning_point"]["reason"] == "no_rise_within_loading"
    assert analysis["force_signal_rise_fit"] is None
    assert not analysis["baseline_return"]["reached"]


def test_positive_shift_is_not_proximity_and_legacy_baseline_is_flagged():
    rows = [
        _row(index * 0.1, "approaching", index * 0.1, signal, 0.0)
        for index, signal in enumerate((0.0, 1.0, 2.0, 4.0, 5.0, 6.0))
    ]

    analysis = analyze_calibration_rows(rows, "diff_ave")

    assert analysis["signal_change_threshold"] >= 6
    assert analysis["signal_onset"] is None
    assert analysis["warnings"]


def test_smooth_polynomial_fit_recovers_a_quadratic_equation():
    points = [
        (x_value, 2.0 * x_value ** 2 - 3.0 * x_value + 4.0)
        for x_value in np.linspace(-2.0, 3.0, 21)
    ]

    fit = _smooth_polynomial_fit(points)

    assert fit["degree"] == 2
    np.testing.assert_allclose(fit["coefficients"], [4.0, -3.0, 2.0])
    assert np.isclose(fit["r_squared"], 1.0)


def test_calibration_analysis_does_not_mix_return_samples_into_loading_fit():
    loading_forces = [0.0, 0.0, 0.0, 20.0, 30.0, 40.0]
    loading_rows = [
        _row(index, "approaching", index, -2.0 * index, loading_forces[index - 1])
        for index in range(1, 7)
    ]
    return_rows = [
        _row(7 + index, "returning", 6 - index, 100.0, 500.0)
        for index in range(4)
    ]

    analysis = analyze_calibration_rows(loading_rows + return_rows, "diff_ave")

    assert analysis["signal_distance_fit"] is None  # No confirmed proximity.
    assert np.isclose(analysis["force_distance_fit"]["slope"], 10.0)


def test_calibration_rows_can_project_any_recorded_taxel_signal():
    rows = [
        {
            "elapsed_s": 0.0,
            "phase": "approaching",
            "travel_down_mm": 0.0,
            "force_delta_abs_g": 0.0,
            "diff_ave": -35.0,
            taxel_signal_column("diff_ave", 0): -1.0,
            taxel_signal_column("diff_ave", 35): -35.0,
            taxel_signal_column("diff_ave", 79): -79.0,
        }
    ]

    assert available_taxel_indices(rows, "diff_ave", legacy_index=35) == [
        0,
        35,
        79,
    ]
    projected = calibration_rows_for_taxel(
        rows,
        "diff_ave",
        79,
        legacy_index=35,
    )

    assert projected[0]["diff_ave"] == -79.0
    assert projected[0]["cell_index"] == 79
    assert rows[0]["diff_ave"] == -35.0


def test_legacy_calibration_rows_only_expose_original_taxel():
    rows = [_row(0.0, "approaching", 0.0, -4.0, 0.0)]

    assert available_taxel_indices(rows, "diff_ave", legacy_index=35) == [35]
    assert calibration_rows_for_taxel(rows, "diff_ave", 35, 35) == rows
    assert calibration_rows_for_taxel(rows, "diff_ave", 36, 35) == []

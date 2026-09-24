import numpy as np

from phd.ui.calibration_analysis import (
    analyze_calibration_rows,
    available_taxel_indices,
    calibration_rows_for_taxel,
    taxel_signal_column,
)


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
        _row(0.0, "approaching", 0.0, 0.0, 0.0),
        _row(0.1, "approaching", 0.1, 0.1, 0.1),
        _row(0.2, "approaching", 0.2, -0.1, 0.2),
        _row(0.3, "approaching", 0.3, 0.0, 0.2),
        _row(0.4, "approaching", 0.4, -2.0, 0.3),
        _row(0.5, "approaching", 0.5, -4.0, 0.4),
        _row(0.6, "approaching", 0.6, -6.0, 0.5),
        _row(0.7, "approaching", 0.7, -8.0, 5.0),
        _row(0.8, "approaching", 0.8, -10.0, 15.0),
        _row(0.9, "contact_hold", 0.9, -12.0, 30.0),
        _row(1.0, "returning", 0.7, -8.0, 10.0),
        _row(1.1, "returning", 0.2, -1.0, 0.5),
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

    assert np.isclose(analysis["signal_distance_fit"]["slope"], -2.0)
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

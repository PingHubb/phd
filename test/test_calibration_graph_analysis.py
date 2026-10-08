import csv
import json

import numpy as np

from phd.ui.calibration_graph_analysis import (
    _dataset_sensor_shape,
    analyze_calibration_batch,
    analyze_calibration_dataset,
    build_batch_report,
    build_trial_report,
    load_calibration_recording,
)
from phd.ui.calibration_result_dialog import CalibrationCurveWidget


def _synthetic_rows(signal_offset=0.0):
    rows = []
    elapsed = 0.0
    for index in range(20):
        distance = index * 0.1
        signal = signal_offset if index < 4 else signal_offset - 10.0 * (
            distance - 0.3
        )
        force = 0.0 if index < 10 else 100.0 * (distance - 0.9)
        rows.append(
            {
                "elapsed_s": elapsed,
                "phase": "approaching",
                "travel_down_mm": distance,
                "diff_ave": signal,
                "force_delta_abs_g": force,
            }
        )
        elapsed += 0.1
    for index in range(20):
        distance = (19 - index) * 0.1
        loading_signal = (
            signal_offset
            if distance < 0.4
            else signal_offset - 10.0 * (distance - 0.3)
        )
        loading_force = 0.0 if distance < 1.0 else 100.0 * (distance - 0.9)
        rows.append(
            {
                "elapsed_s": elapsed,
                "phase": "returning",
                "travel_down_mm": distance,
                "diff_ave": loading_signal + 2.0,
                "force_delta_abs_g": loading_force + 3.0,
            }
        )
        elapsed += 0.1
    return rows


def _add_taxel_signals(rows):
    enriched = []
    for row in rows:
        item = dict(row)
        for index in range(80):
            item[f"taxel_{index:03d}_diff_ave"] = (
                float(row["diff_ave"]) - index
            )
        enriched.append(item)
    return enriched


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_professional_analysis_reports_events_models_and_hysteresis():
    dataset = {
        "name": "Trial 0001",
        "source_path": "/tmp/trial_0001.csv",
        "signal_field": "diff_ave",
        "rows": _synthetic_rows(),
    }

    analysis = analyze_calibration_dataset(dataset)

    assert np.isclose(
        analysis["acquisition"]["median_sampling_rate_hz"], 10.0
    )
    assert analysis["events"]["signal_onset"]["distance_mm"] < analysis[
        "events"
    ]["contact_onset"]["distance_mm"]
    assert analysis["events"]["proximity_window_mm"] > 0.0
    assert np.isclose(
        analysis["models"]["signal_distance"]["slope"], -10.0
    )
    assert analysis["models"]["force_distance"]["slope"] > 0.0
    assert analysis["models"]["force_signal"]["slope"] < 0.0
    assert analysis["models"]["force_signal"]["rmse"] >= 0.0
    assert analysis["hysteresis"]["signal_distance"]["rms"] > 0.0
    assert analysis["hysteresis"]["force_distance"]["rms"] > 0.0

    report = build_trial_report(analysis)
    assert "ACQUISITION QUALITY" in report
    assert "Non-contact proximity travel window" in report
    assert "LOADING REGRESSION MODELS" in report
    assert "PIECEWISE SMOOTH APPROXIMATIONS" not in report
    assert "LOADING/RETURN HYSTERESIS" in report


def test_piecewise_approximations_separate_proximity_contact_and_return():
    dataset = {
        "name": "Trial 0001",
        "source_path": "/tmp/trial_0001.csv",
        "signal_field": "diff_ave",
        "rows": _synthetic_rows(),
    }

    analysis = analyze_calibration_dataset(dataset)
    fits = analysis["segmented_approximations"]
    contact_distance = analysis["events"]["contact_onset"]["distance_mm"]

    for relationship in (
        "signal_distance",
        "force_distance",
        "force_signal",
    ):
        assert set(fits[relationship]) == {
            "proximity",
            "contact",
            "return",
        }
        assert all(
            fits[relationship][segment] is not None
            for segment in fits[relationship]
        )
    assert fits["signal_distance"]["proximity"]["x_max"] < contact_distance
    assert np.isclose(
        fits["signal_distance"]["contact"]["x_min"],
        contact_distance,
    )


def test_graph_cursor_interpolates_local_curve_in_both_directions():
    increasing = [(0.0, -2.0), (2.0, 6.0)]
    decreasing = [(2.0, 6.0), (0.0, -2.0)]

    assert CalibrationCurveWidget._sample_at_x(increasing, 0.5) == (0.5, 0.0)
    assert CalibrationCurveWidget._sample_at_x(decreasing, 0.5) == (0.5, 0.0)
    assert CalibrationCurveWidget._sample_at_x(increasing, -0.1) is None


def test_repeatability_graph_interpolates_mean_and_sample_deviation():
    first = _synthetic_rows(signal_offset=0.0)
    second = _synthetic_rows(signal_offset=2.0)

    statistics = CalibrationCurveWidget._repeatability_statistics(
        [first, second],
        "diff_ave",
        sample_count=11,
    )

    assert statistics["trial_count"] == 2
    assert len(statistics["mean"]) == 11
    assert len(statistics["lower"]) == 11
    assert len(statistics["upper"]) == 11
    for mean, lower, upper in zip(
        statistics["mean"],
        statistics["lower"],
        statistics["upper"],
    ):
        assert np.isclose(mean[1], (lower[1] + upper[1]) / 2.0)
        assert lower[1] < mean[1] < upper[1]


def test_load_csv_and_exported_graph_uses_underlying_raw_data(tmp_path):
    batch_directory = tmp_path / "calibration_batch"
    graph_directory = batch_directory / "graphs"
    graph_directory.mkdir(parents=True)
    csv_path = batch_directory / "trial_0001.csv"
    _write_csv(csv_path, _synthetic_rows())
    graph_path = graph_directory / "trial_0001.png"
    graph_path.write_bytes(b"not parsed as image pixels")

    csv_loaded = load_calibration_recording(csv_path)
    graph_loaded = load_calibration_recording(graph_path)

    assert len(csv_loaded["datasets"]) == 1
    assert len(graph_loaded["datasets"]) == 1
    assert graph_loaded["datasets"][0]["source_path"] == str(
        csv_path.resolve()
    )
    assert graph_loaded["datasets"][0]["rows"][5]["diff_ave"] < 0.0


def test_load_csv_keeps_all_80_taxel_signal_columns(tmp_path):
    csv_path = tmp_path / "trial_0001.csv"
    _write_csv(csv_path, _add_taxel_signals(_synthetic_rows()))

    loaded = load_calibration_recording(csv_path)
    row = loaded["datasets"][0]["rows"][5]

    assert row["taxel_000_diff_ave"] == row["diff_ave"]
    assert row["taxel_079_diff_ave"] == row["diff_ave"] - 79.0


def test_batch_json_loads_trials_and_reports_repeatability(tmp_path):
    trial_paths = []
    for number, offset in ((1, 0.0), (2, 1.0)):
        trial_path = tmp_path / f"trial_{number:04d}.csv"
        _write_csv(trial_path, _synthetic_rows(signal_offset=offset))
        trial_paths.append(trial_path)
    batch_path = tmp_path / "batch_summary.json"
    batch_path.write_text(
        json.dumps(
            {
                "signal_field": "diff_ave",
                "trials": [
                    {
                        "trial_number": number,
                        "csv_path": str(path),
                    }
                    for number, path in enumerate(trial_paths, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    repeatability_path = tmp_path / "repeatability.png"
    repeatability_path.write_bytes(b"graph pixels are not parsed")

    loaded = load_calibration_recording(batch_path)
    loaded_from_repeatability = load_calibration_recording(
        repeatability_path
    )
    analyses = [
        analyze_calibration_dataset(dataset)
        for dataset in loaded["datasets"]
    ]
    batch = analyze_calibration_batch(analyses)

    assert len(loaded["datasets"]) == 2
    assert len(loaded_from_repeatability["datasets"]) == 2
    assert batch["trial_count"] == 2
    assert batch["repeatability"]["contact_onset_distance_mm"]["count"] == 2
    assert "BATCH REPEATABILITY ANALYSIS" in build_batch_report(batch)


def test_full_taxel_summary_uses_each_trial_sensor_shape(tmp_path):
    trial_directory = tmp_path / "taxel_12"
    trial_directory.mkdir()
    trial_path = trial_directory / "trial_0001.csv"
    rows = _synthetic_rows()
    for row in rows:
        row["taxel_012_diff_ave"] = row["diff_ave"]
        row["cell_index"] = 12
    _write_csv(trial_path, rows)
    metadata_path = trial_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "sensor_shape": [5, 5],
                "taxel_index": 12,
                "signal_field": "diff_ave",
                "csv_path": str(trial_path),
            }
        ),
        encoding="utf-8",
    )
    batch_path = trial_directory / "batch_summary.csv"
    _write_csv(
        batch_path,
        [
            {
                "taxel_index": 12,
                "trial_number": 1,
                "csv_path": str(trial_path),
                "metadata_path": str(metadata_path),
            }
        ],
    )
    full_csv_path = tmp_path / "full_taxel_summary.csv"
    _write_csv(
        full_csv_path,
        [
            {
                "taxel_index": 12,
                "trial_number": 1,
                "csv_path": str(trial_path),
                "metadata_path": str(metadata_path),
            }
        ],
    )
    full_json_path = full_csv_path.with_suffix(".json")
    full_json_path.write_text(
        json.dumps(
            {
                "task": "full_taxel_calibration",
                "signal_field": "diff_ave",
                "summary_csv_path": str(full_csv_path),
                "taxel_results": [
                    {
                        "taxel_index": 12,
                        "batch_summary_path": str(batch_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from_csv = load_calibration_recording(full_csv_path)
    from_json = load_calibration_recording(full_json_path)

    assert _dataset_sensor_shape(from_csv["datasets"][0]) == (5, 5)
    assert _dataset_sensor_shape(from_json["datasets"][0]) == (5, 5)
    assert from_csv["datasets"][0]["metadata"]["taxel_index"] == 12.0
    assert from_json["datasets"][0]["metadata"]["taxel_index"] == 12.0

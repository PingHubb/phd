"""Height profile: neighbour contrast and contact margin from an approach trial."""

import numpy as np

from phd.dependence.height_profile import analyse_height_profile, cardinal_neighbours
from phd.ui.calibration_analysis import taxel_signal_column

ROWS = COLUMNS = 20
TARGET = 189


def _rows(width_at, contact_travel_mm=20.0, noise=0.5, seed=0):
    """Synthetic approach: response narrows with travel, force after contact."""
    rng = np.random.default_rng(seed)
    centres = np.array(
        [[(index % ROWS) * 12.0, (index // ROWS) * 12.0] for index in range(ROWS * COLUMNS)]
    )
    target_xy = centres[TARGET]
    rows = []
    for sequence, travel in enumerate(np.arange(0.0, contact_travel_mm + 1.0, 0.05)):
        gap = max(contact_travel_mm - travel, 0.0)
        level = 200.0 / (1.0 + gap / 3.0)
        sigma = width_at(gap)
        distance = np.linalg.norm(centres - target_xy, axis=1)
        values = -level * np.exp(-distance**2 / (2 * sigma**2)) + rng.normal(0, noise, len(centres))
        force = 0.0 if travel < contact_travel_mm else 10.0 * (travel - contact_travel_mm + 0.5)
        row = {
            "phase": "approaching",
            "frame_sequence": sequence,
            "travel_down_mm": travel,
            "tool_z_m": 0.17 - travel / 1000.0,
            "force_delta_g": force,
        }
        row.update(
            {taxel_signal_column("diff_ave", index): values[index] for index in range(len(values))}
        )
        rows.append(row)
    return rows


def test_neighbours_follow_the_column_major_index():
    assert sorted(cardinal_neighbours(189, 20, 20)) == [169, 188, 190, 209]


def test_sharp_response_close_to_the_sensor_is_recommended_above_contact():
    result = analyse_height_profile(_rows(lambda gap: 4.0 + gap), ROWS, COLUMNS, TARGET)

    assert result["contact_travel_mm"] is not None
    best = result["recommended_non_contact"]
    assert best is not None
    assert best["travel_mm"] <= result["contact_travel_mm"] - 1.0 + 0.5
    assert best["dominance"] >= 1.3
    assert result["non_contact_feasible"]


def test_a_broad_response_at_every_height_is_reported_as_not_feasible():
    result = analyse_height_profile(_rows(lambda gap: 25.0), ROWS, COLUMNS, TARGET)

    assert result["recommended_non_contact"]["dominance"] < 1.3
    assert not result["non_contact_feasible"]

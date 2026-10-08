import json

import numpy as np

from phd.dependence.func_sensor import SENSOR_ZERO_MASK_FILE
from phd.dependence.sensor_layout import reshape_sensor_values_to_row_col_matrix


# 27x22 grid. Rows are sensor electrodes and columns are drive electrodes,
# both running from the lowest electrode number.
_DRIVE_ELECTRODES = (
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    20, 21, 22, 23, 24, 25,
    33, 34, 35, 36, 37, 38, 39,
)
_SENSOR_ELECTRODES = (
    0, 1, 2, 3, 4, 5, 6,
    10, 11, 12, 13,
    14, 15, 16, 17,
    18, 19, 20, 21,
    22, 23, 24, 25,
    26, 27, 28, 29,
)
_PAIRED_PATCHES = (
    (range(0, 3), range(26, 30)),
    (range(3, 6), range(22, 26)),
    (range(6, 9), range(18, 22)),
    (range(20, 23), range(14, 18)),
    (range(23, 26), range(10, 14)),
    (range(33, 40), range(0, 7)),
)


def _expected_mask():
    drive_index = {
        electrode: index for index, electrode in enumerate(_DRIVE_ELECTRODES)
    }
    sensor_index = {
        electrode: index for index, electrode in enumerate(_SENSOR_ELECTRODES)
    }
    mask = np.ones((27, 22), dtype=bool)
    for drives, sensors in _PAIRED_PATCHES:
        for drive in drives:
            for sensor in sensors:
                mask[sensor_index[sensor], drive_index[drive]] = False
    return mask


def test_dexterous_hand_zero_mask_keeps_only_six_paired_patches():
    with open(SENSOR_ZERO_MASK_FILE, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    saved = np.asarray(payload["masks"]["2d_27x22"], dtype=bool)

    expected = _expected_mask()
    np.testing.assert_array_equal(saved, expected)
    assert saved.shape == (27, 22)
    assert int((~saved).sum()) == 109
    assert int(saved.sum()) == 27 * 22 - 109


def test_dexterous_hand_frame_stays_27_by_22():
    values = np.zeros(27 * 22, dtype=int)
    values[4 * 27 + 9] = 1234
    matrix = reshape_sensor_values_to_row_col_matrix(values, 27, 22)

    assert matrix.shape == (27, 22)
    assert matrix[9, 4] == 1234

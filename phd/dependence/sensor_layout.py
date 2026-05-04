"""
Shared helpers for tactile sensor row/col layout conversions.

Internal rule:
- Active code should treat sensor matrices as `(n_row, n_col)`.
- Internal access should read as `matrix[row][col]`.

Historic compatibility rules:
- Some old pipelines expect a column-major flat frame.
- Some old pipelines expect a transposed 2D `(n_col, n_row)` view.

Example for `n_row=2`, `n_col=3`, and device payload `[1, 2, 3, 4, 5, 6]`:
- `reshape_sensor_values_to_row_col_matrix(...)` -> `[[1, 3, 5], [2, 4, 6]]`
- `flatten_column_major_view(...)` of that matrix -> `[1, 2, 3, 4, 5, 6]`
- `column_major_matrix_view(...)` of that matrix -> `[[1, 2], [3, 4], [5, 6]]`

This keeps the device boundary explicit while preserving old model/data formats.
"""

import numpy as np


def column_major_idx(n_row, col, row):
    """Map `(col, row)` onto a flat index stored in column-major order."""
    return col * n_row + row


def column_major_coords(n_row, flat_idx):
    """Decode a column-major flat index into `(col, row)` coordinates."""
    return flat_idx // n_row, flat_idx % n_row


def row_major_idx(n_col, row, col):
    """Map `(row, col)` onto a flat index stored in row-major order."""
    return row * n_col + col


def flatten_column_major_view(matrix):
    """Expose a `(n_row, n_col)` matrix in the historic column-major flat view."""
    return np.asarray(matrix).T.flatten()


def column_major_matrix_view(matrix, dtype=None, copy=False):
    """
    Expose a `(n_row, n_col)` matrix as a `(n_col, n_row)` column-major 2D view.

    This is kept for model/data pipelines that were historically trained or saved
    using the transposed layout.
    """
    return np.array(np.asarray(matrix).T, dtype=dtype, copy=copy)


def reshape_sensor_values_to_row_col_matrix(values, n_row, n_col):
    """
    Convert a column-major sensor payload into the internal `(n_row, n_col)` layout.

    The sensor stream is serialized column-by-column, while the rest of the codebase
    uses `matrix[row][col]`. This helper is the single boundary that translates
    between those two representations.
    """
    return np.asarray(values).reshape(n_col, n_row).T

"""Pure bounds/recovery helpers; recorded response planes are not surfaces."""

import numpy as np


def working_probe_boundary(positions, taxel, rows, columns, reference_onset,
                           discovery_allowance=.005, plane_margin=.0005):
    """Bound search to a small margin below a saved response plane.

    This is a conservative software search boundary, NOT a measured air gap.
    It applies before and after onset. A first anchor without measured
    neighbours gets only a small discovery window below its learned onset.
    The default 0.5 mm plane margin is a search tolerance, not an air gap.
    """
    taxel, rows, columns = int(taxel), int(rows), int(columns)

    def scan_z(index):
        record = positions.get(index) or {}
        if record.get("status") != "measured" or record.get("relocation_verified") is False:
            return None
        try:
            value = record.get("centre_scan_z_m")
            value = float(record["tool_position_m"][2] if value is None else value)
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        return value if np.isfinite(value) else None

    own = scan_z(taxel)
    if own is not None:
        return {"z_m": own-float(plane_margin), "source": "saved_target_response_plane", "taxels": [taxel]}
    row, column = taxel % rows, taxel // rows
    neighbours = []
    for r, c in ((row-1, column), (row+1, column), (row, column-1), (row, column+1)):
        if 0 <= r < rows and 0 <= c < columns:
            index = c * rows + r
            z = scan_z(index)
            if z is not None:
                neighbours.append((index, z))
    if neighbours:
        return {"z_m": max(z for _, z in neighbours)-float(plane_margin),
                "source": "shallowest_verified_neighbour_plane", "taxels": [i for i, _ in neighbours]}
    try:
        reference = float(reference_onset)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(reference):
        return None
    return {"z_m": reference - float(discovery_allowance),
            "source": "bounded_first_anchor_discovery", "taxels": []}


def clearance_recovery(frames, reference_frames, minimum_margin):
    """Check recovery BEFORE recalibration, so recalibration cannot hide load.

    The anchor itself must be independently established as clear by the
    operator. Stability/recovery only rejects obvious residual responses.
    """
    values = np.asarray(frames, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
        return {"recovered": False, "reference_available": False,
                "interpretation": "Invalid clearance sample batch; do not recalibrate."}
    if reference_frames is None:
        return {"recovered": True, "reference_available": False,
                "interpretation": "Operator-established clearance anchor; no earlier same-calibration reference."}
    reference = np.asarray(reference_frames, dtype=float)
    if (reference.ndim != 2 or reference.shape[1:] != values.shape[1:]
            or reference.shape[0] < 1 or not np.all(np.isfinite(reference))):
        return {"recovered": False, "reference_available": False,
                "interpretation": "Invalid same-calibration reference; do not recalibrate."}
    old = np.median(reference, axis=0)
    current = np.median(values, axis=0)
    sigma = 1.4826 * np.median(np.abs(reference-old), axis=0)
    tolerance = np.maximum(2. * float(minimum_margin), 4. * sigma)
    shift = current - old
    departure = np.abs(shift)
    bad = np.flatnonzero(departure > tolerance)

    def detail(index):
        return {"taxel": int(index), "shift_units": float(shift[index]),
                "tolerance_units": float(tolerance[index])}

    return {"recovered": not bool(bad.size), "reference_available": True,
            "maximum_departure_units": float(np.max(departure)),
            "unrecovered_taxels": bad.astype(int).tolist(),
            "unrecovered_details": [detail(index) for index in bad],
            "largest_shifts": [detail(index) for index in np.argsort(-departure)[:5]],
            "median_shift_units": float(np.median(shift)),
            "shift_convention": "current minus reference; proximity lowers the signal, so "
                                "a rise is not a residual proximity response",
            "interpretation": "Signal recovery check, not proof of physical clearance."}


def local_retry_xy(positions, taxel, original_xy, rows, columns, max_shift):
    """Try a bounded local-edge estimate instead of repeating an identical guess."""
    row, column = int(taxel) % rows, int(taxel) // rows
    estimates = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        coords = [(row+step*dr, column+step*dc) for step in (1, 2)]
        if not all(0 <= r < rows and 0 <= c < columns for r, c in coords):
            continue
        records = [positions.get(c*rows+r) or {} for r, c in coords]
        if not all(r.get("status") == "measured" and r.get("relocation_verified") is not False for r in records):
            continue
        try:
            near, far = [np.asarray(r["tool_position_m"][:2], dtype=float) for r in records]
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite([near, far])):
            estimates.append(2.*near-far)
    original = np.asarray(original_xy, dtype=float)
    if not estimates:
        return original.copy()
    delta = np.median(estimates, axis=0)-original
    length = float(np.linalg.norm(delta))
    return original + delta * min(1., float(max_shift)/max(length, 1e-12))

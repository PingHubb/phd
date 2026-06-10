#!/usr/bin/env python3
"""Convert an IGES/IGS CAD path to a simple XYZ CSV using FreeCAD.

Run with FreeCAD's Python executable, for example:

    FreeCADCmd phd/script/igs_to_csv_freecad.py input.igs output.csv --unit-scale 0.001

The output CSV contains:

    x,y,z

Use ``--unit-scale 0.001`` when the CAD file is in millimetres and the robot
path viewer expects metres.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_shape(path: str):
    try:
        import Part
    except Exception as exc:  # pragma: no cover - only available in FreeCAD
        raise RuntimeError(
            "FreeCAD Part module is not available. Run this script with FreeCADCmd."
        ) from exc

    try:
        return Part.read(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read IGES file with FreeCAD Part.read(): {path}") from exc


def _sample_edge(edge, samples_per_edge: int):
    try:
        first, last = edge.ParameterRange
    except Exception:
        first, last = 0.0, 1.0

    n = max(2, int(samples_per_edge))
    for i in range(n):
        t = first + (last - first) * (i / float(n - 1))
        p = edge.valueAt(t)
        yield (float(p.x), float(p.y), float(p.z))


def extract_points(input_path: str, samples_per_edge: int):
    shape = _load_shape(input_path)
    points = []

    edges = list(getattr(shape, "Edges", []) or [])
    for edge in edges:
        points.extend(_sample_edge(edge, samples_per_edge))

    if not points:
        vertices = list(getattr(shape, "Vertexes", []) or [])
        for vertex in vertices:
            p = vertex.Point
            points.append((float(p.x), float(p.y), float(p.z)))

    if not points:
        raise RuntimeError("No edges or vertices were found in the IGES file.")

    return points


def remove_near_duplicates(points, tolerance: float):
    if tolerance <= 0.0:
        return points

    kept = []
    last = None
    tol2 = float(tolerance) ** 2
    for point in points:
        if last is None:
            kept.append(point)
            last = point
            continue
        dx = point[0] - last[0]
        dy = point[1] - last[1]
        dz = point[2] - last[2]
        if (dx * dx + dy * dy + dz * dz) >= tol2:
            kept.append(point)
            last = point
    return kept


def write_csv(output_path: str, points, unit_scale: float):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = float(unit_scale)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        for x, y, z in points:
            writer.writerow([x * scale, y * scale, z * scale])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_igs", help="Input .igs/.iges file")
    parser.add_argument("output_csv", help="Output CSV path")
    parser.add_argument(
        "--samples-per-edge",
        type=int,
        default=25,
        help="How many sample points to generate along each CAD edge.",
    )
    parser.add_argument(
        "--unit-scale",
        type=float,
        default=1.0,
        help="Scale applied to every coordinate. Use 0.001 for mm -> m.",
    )
    parser.add_argument(
        "--dedupe-tolerance",
        type=float,
        default=0.0,
        help="Remove consecutive points closer than this distance after sampling, before unit scaling.",
    )
    args = parser.parse_args()

    points = extract_points(args.input_igs, args.samples_per_edge)
    points = remove_near_duplicates(points, args.dedupe_tolerance)
    write_csv(args.output_csv, points, args.unit_scale)
    print(f"Wrote {len(points)} points to {args.output_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract XYZ coordinates from an IGES file into CSV.

Default output extracts IGES point entities, type 116:
    116,x,y,z,...

Optional flags can also include line endpoints (type 110) and NURBS curve
control points (type 126). Those are useful when SolidWorks stores a path as
curves rather than explicit point entities.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")


def _parse_number(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "E"))


def _split_iges_params(record: str) -> list[str]:
    record = record.strip().rstrip(";")
    return [part.strip() for part in record.split(",")]


def _read_parameter_records(path: Path) -> list[str]:
    """Return complete IGES parameter records from the P section."""
    records: list[str] = []
    current: list[str] = []

    with path.open("r", encoding="latin1") as handle:
        for line in handle:
            if len(line) < 73 or line[72] != "P":
                continue
            current.append(line[:64].rstrip())
            if ";" in line[:64]:
                records.append("".join(current))
                current = []

    if current:
        records.append("".join(current))
    return records


def _entity_type(record: str) -> int | None:
    match = FLOAT_RE.match(record.strip())
    if not match:
        return None
    return int(float(match.group(0)))


def _extract_point_entity(record: str) -> list[tuple[str, float, float, float]]:
    parts = _split_iges_params(record)
    if len(parts) < 4 or parts[0] != "116":
        return []
    return [("point_116", _parse_number(parts[1]), _parse_number(parts[2]), _parse_number(parts[3]))]


def _extract_line_entity(record: str) -> list[tuple[str, float, float, float]]:
    parts = _split_iges_params(record)
    if len(parts) < 7 or parts[0] != "110":
        return []
    x1, y1, z1, x2, y2, z2 = (_parse_number(value) for value in parts[1:7])
    return [
        ("line_110_start", x1, y1, z1),
        ("line_110_end", x2, y2, z2),
    ]


def _extract_nurbs_control_points(record: str) -> list[tuple[str, float, float, float]]:
    """Extract IGES type 126 control points.

    Type 126 layout after entity type:
        K, M, PROP1, PROP2, PROP3, PROP4, knots..., weights..., control points...

    Number of control points is K + 1.
    """
    parts = _split_iges_params(record)
    if len(parts) < 7 or parts[0] != "126":
        return []

    k = int(float(parts[1]))
    degree = int(float(parts[2]))
    control_count = k + 1
    knot_count = k + degree + 2
    knot_start = 7
    weight_start = knot_start + knot_count
    point_start = weight_start + control_count

    points: list[tuple[str, float, float, float]] = []
    for idx in range(control_count):
        base = point_start + idx * 3
        if base + 2 >= len(parts):
            break
        points.append(
            (
                "nurbs_126_control",
                _parse_number(parts[base]),
                _parse_number(parts[base + 1]),
                _parse_number(parts[base + 2]),
            )
        )
    return points


def extract_xyz(
    iges_path: Path,
    *,
    include_line_endpoints: bool = False,
    include_nurbs_control_points: bool = False,
    dedupe: bool = True,
) -> list[tuple[str, float, float, float]]:
    points: list[tuple[str, float, float, float]] = []
    for record in _read_parameter_records(iges_path):
        entity = _entity_type(record)
        if entity == 116:
            points.extend(_extract_point_entity(record))
        elif entity == 110 and include_line_endpoints:
            points.extend(_extract_line_entity(record))
        elif entity == 126 and include_nurbs_control_points:
            points.extend(_extract_nurbs_control_points(record))

    if not dedupe:
        return points

    unique: list[tuple[str, float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for source, x, y, z in points:
        key = (round(x, 9), round(y, 9), round(z, 9))
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, x, y, z))
    return unique


def write_csv(points: list[tuple[str, float, float, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_mm", "y_mm", "z_mm"])
        for _source, x, y, z in points:
            writer.writerow([f"{x:.9g}", f"{y:.9g}", f"{z:.9g}"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract XYZ points from an IGES file.")
    parser.add_argument("iges", type=Path, help="Input .igs/.iges file")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        help="Output CSV path. Defaults to input filename with .csv extension.",
    )
    parser.add_argument(
        "--include-line-endpoints",
        action="store_true",
        help="Also output start/end points from IGES line entities, type 110.",
    )
    parser.add_argument(
        "--include-nurbs-control-points",
        action="store_true",
        help="Also output control points from IGES NURBS curve entities, type 126.",
    )
    parser.add_argument("--no-dedupe", action="store_true", help="Keep duplicate XYZ rows.")
    args = parser.parse_args()

    csv_path = args.csv or args.iges.with_suffix(".csv")
    points = extract_xyz(
        args.iges,
        include_line_endpoints=args.include_line_endpoints,
        include_nurbs_control_points=args.include_nurbs_control_points,
        dedupe=not args.no_dedupe,
    )
    write_csv(points, csv_path)
    print(f"Wrote {len(points)} XYZ points to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

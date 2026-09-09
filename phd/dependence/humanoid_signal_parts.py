"""Map packaged humanoid signal sections onto URDF links."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .humanoid_urdf import UrdfModel

PART_OBJ = re.compile(r"^(\d+)_(.+)\.obj$", re.IGNORECASE)
SIGNAL_FILE = "curves_col_signal.obj"

LINK_ALIASES = {
    "left_hand_palm_link": "left_rubber_hand",
    "right_hand_palm_link": "right_rubber_hand",
}


NO_MIRROR_TOKENS = ("elbow",)


@dataclass
class SignalPart:
    number: str
    part_name: str
    link_name: str
    signal_obj: Path
    reused_from: str | None = None


def resolve_link_name(part_name: str, model: UrdfModel) -> str | None:
    names = [part_name]
    if part_name.endswith("_rev_1_0"):
        names.append(part_name[: -len("_rev_1_0")])
    alias = LINK_ALIASES.get(part_name)
    if alias:
        names.append(alias)
    for name in names:
        if name in model.links:
            return name
    want = {n.replace("_rev_1_0", "") for n in names}
    for link in model.links.values():
        for vis in link.visuals:
            stem = vis.mesh_path.stem.replace("_rev_1_0", "")
            if stem in want or vis.mesh_path.stem in names:
                return link.name
    return None


def discover_signal_parts(done_dir: str | Path, model: UrdfModel) -> list[SignalPart]:
    root = Path(done_dir)
    found: list[SignalPart] = []
    if not root.is_dir():
        return found

    records: list[tuple[str, str]] = []
    manifest_path = root / "parts.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return found
        for item in payload.get("parts", []):
            if not isinstance(item, dict):
                continue
            number = str(item.get("number", "")).strip()
            part_name = str(item.get("part_name", "")).strip()
            if number.isdigit() and part_name:
                records.append((number, part_name))
    else:
        # Compatibility with raw WeavingMapGenerator export directories.
        for src in sorted(root.glob("*_*.obj")):
            if src.parent.resolve() != root.resolve():
                continue
            match = PART_OBJ.match(src.name)
            if match is not None:
                records.append((match.group(1), match.group(2)))

    for number, part_name in records:
        signal = root / number / SIGNAL_FILE
        if not signal.is_file():
            continue
        link = resolve_link_name(part_name, model)
        if link is None:
            continue
        found.append(
            SignalPart(
                number=number,
                part_name=part_name,
                link_name=link,
                signal_obj=signal,
            )
        )
    found.sort(key=lambda p: int(p.number))
    return expand_symmetric_parts(found, model)


def counterpart_link(link_name: str) -> str | None:
    """``left_*`` ↔ ``right_*``. Elbows are not interchangeable."""
    lowered = link_name.lower()
    if any(tok in lowered for tok in NO_MIRROR_TOKENS):
        return None
    if link_name.startswith("left_"):
        return "right_" + link_name[len("left_") :]
    if link_name.startswith("right_"):
        return "left_" + link_name[len("right_") :]
    return None


def expand_symmetric_parts(parts: list[SignalPart], model: UrdfModel) -> list[SignalPart]:
    """Reuse a left/right signal on the opposite link with the same local points."""
    have = {p.link_name for p in parts}
    extra: list[SignalPart] = []
    for part in parts:
        other = counterpart_link(part.link_name)
        if other is None or other not in model.links or other in have:
            continue
        extra.append(
            SignalPart(
                number=part.number,
                part_name=part.part_name,
                link_name=other,
                signal_obj=part.signal_obj,
                reused_from=part.link_name,
            )
        )
        have.add(other)
    return parts + extra


def load_colored_line_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ``v x y z [r g b]`` + ``l`` polylines. Returns points, colors, edges."""
    pts: list[list[float]] = []
    cols: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        kind = tok[0]
        if kind == "v" and len(tok) >= 4:
            pts.append([float(tok[1]), float(tok[2]), float(tok[3])])
            if len(tok) >= 7:
                cols.append([float(tok[4]), float(tok[5]), float(tok[6])])
            else:
                cols.append([1.0, 0.45, 0.12])
        elif kind == "l" and len(tok) >= 3:
            ids = [int(x.split("/")[0]) - 1 for x in tok[1:]]
            for a, b in zip(ids, ids[1:]):
                if a >= 0 and b >= 0:
                    edges.append((a, b))
    points = np.asarray(pts, dtype=np.float64)
    colors = np.clip(np.asarray(cols, dtype=np.float64), 0.0, 1.0)
    if len(edges) == 0:
        line_arr = np.zeros((0,), dtype=np.int64)
    else:
        line_arr = np.asarray([[2, a, b] for a, b in edges], dtype=np.int64).reshape(-1)
    return points, colors, line_arr

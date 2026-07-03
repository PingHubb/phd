#!/usr/bin/env python3
"""Check AI Direct Finger Motion recording quality before training.

This script does not train a model. It only validates recorded ``.npz`` trials
and prints dataset health statistics for the first tactile policy training pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE_LENGTH = 16
DEFAULT_SESSION = "first_trial_no_rotation"
DEFAULT_TARGET_KEY = "intended_velocity_target"
CRITICAL_ARRAY_KEYS = (
    "diffPerDataAve",
    "touch_mask",
    DEFAULT_TARGET_KEY,
)
RECOMMENDED_KEYS = (
    "metadata_json",
    "timestamps",
    "elapsed_sec",
    "dt_sec",
    "sensor_frame_sequence",
    "rawData",
    "diffData",
    "diffPerData",
    "diffDataAve",
    "diffPerDataAve",
    "touch_mask",
    "touch_present",
    "mode",
    "teacher_velocity_target",
    "intended_velocity_target",
    "intended_mode",
    "teaching_label",
    "teaching_source",
    "manual_override_active",
    "robot_feedback_valid",
)
AXIS_NAMES = ("vx", "vy", "vz", "rx", "ry", "rz")


@dataclass
class TrialSummary:
    path: Path
    frame_count: int = 0
    duration_sec: float = 0.0
    hz: float = 0.0
    sensor_shape: tuple[int, ...] | None = None
    version: str = "unknown"
    touch_count: int = 0
    target_nonzero_count: int = 0
    rotation_nonzero_count: int = 0
    manual_count: int = 0
    robot_feedback_count: int = 0
    duplicate_sensor_frames: int = 0
    stale_dt_count: int = 0
    max_abs_target: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    label_counts: Counter[str] = field(default_factory=Counter)
    mode_counts: Counter[str] = field(default_factory=Counter)
    source_counts: Counter[str] = field(default_factory=Counter)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _resource_root() -> Path:
    env_resource = os.environ.get("PINGLAB_RESOURCE_ROOT", "").strip()
    if env_resource:
        return Path(env_resource).expanduser()
    return PACKAGE_ROOT / "resource"


def _default_dataset_root() -> Path:
    explicit_ai = os.environ.get("PINGLAB_AI_RESOURCE_ROOT", "").strip()
    if explicit_ai:
        return Path(explicit_ai).expanduser() / "data" / "ai_direct_finger_motion"
    return _resource_root() / "ai" / "data" / "ai_direct_finger_motion"


def _session_path(session_or_path: str, dataset_root: Path) -> Path:
    value = Path(session_or_path).expanduser()
    if value.exists() or value.is_absolute() or "/" in session_or_path:
        return value

    tag = session_or_path
    if not tag.startswith("session_"):
        tag = f"session_{tag}"
    return dataset_root / tag


def _load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in data.files:
        return {}
    raw = data["metadata_json"]
    try:
        if raw.shape == ():
            text = str(raw.item())
        else:
            text = str(raw.reshape(-1)[0])
        return json.loads(text)
    except Exception:
        return {}


def _as_string_counter(data: np.lib.npyio.NpzFile, key: str) -> Counter[str]:
    if key not in data.files:
        return Counter()
    values = np.asarray(data[key]).reshape(-1)
    return Counter(str(value) for value in values)


def _frame_count(data: np.lib.npyio.NpzFile, target_key: str) -> int:
    for key in ("timestamps", "elapsed_sec", target_key, "diffPerDataAve", "touch_mask"):
        if key in data.files:
            return int(np.asarray(data[key]).shape[0])
    return 0


def _duration_and_hz(data: np.lib.npyio.NpzFile, frame_count: int) -> tuple[float, float]:
    if frame_count <= 1:
        return 0.0, 0.0

    elapsed = np.asarray(data["elapsed_sec"], dtype=np.float64) if "elapsed_sec" in data.files else None
    timestamps = np.asarray(data["timestamps"], dtype=np.float64) if "timestamps" in data.files else None

    timeline = elapsed
    if timeline is None or timeline.shape[0] != frame_count:
        timeline = timestamps
    if timeline is None or timeline.shape[0] != frame_count:
        return 0.0, 0.0

    duration = float(timeline[-1] - timeline[0])
    if duration <= 0.0 or not math.isfinite(duration):
        return 0.0, 0.0
    return duration, float((frame_count - 1) / duration)


def _touch_count(data: np.lib.npyio.NpzFile, frame_count: int) -> int:
    if "touch_present" in data.files:
        touch = np.asarray(data["touch_present"]).reshape(-1)
        if touch.shape[0] == frame_count:
            return int(np.count_nonzero(touch > 0))
    if "touch_mask" in data.files:
        mask = np.asarray(data["touch_mask"])
        if mask.shape[0] == frame_count:
            return int(np.count_nonzero(np.any(mask > 0, axis=tuple(range(1, mask.ndim)))))
    return 0


def _count_nonzero_rows(array: np.ndarray, *, start_col: int = 0, eps: float = 1e-8) -> int:
    if array.ndim != 2 or array.shape[1] <= start_col:
        return 0
    block = array[:, start_col:]
    return int(np.count_nonzero(np.linalg.norm(block, axis=1) > eps))


def _critical_nonfinite_count(array: np.ndarray) -> int:
    if array.dtype.kind not in {"f", "i", "u", "b"}:
        return 0
    return int(np.count_nonzero(~np.isfinite(array)))


def _duplicate_sensor_frame_count(data: np.lib.npyio.NpzFile) -> int:
    if "sensor_frame_sequence" not in data.files:
        return 0
    seq = np.asarray(data["sensor_frame_sequence"]).reshape(-1)
    if seq.shape[0] <= 1:
        return 0
    valid = (seq[1:] >= 0) & (seq[:-1] >= 0)
    return int(np.count_nonzero((seq[1:] == seq[:-1]) & valid))


def _stale_dt_count(data: np.lib.npyio.NpzFile, threshold_sec: float = 0.05) -> int:
    if "dt_sec" not in data.files:
        return 0
    dt = np.asarray(data["dt_sec"], dtype=np.float64).reshape(-1)
    if dt.shape[0] <= 1:
        return 0
    return int(np.count_nonzero(dt[1:] > threshold_sec))


def _summarize_trial(path: Path, target_key: str, expected_no_rotation: bool) -> TrialSummary:
    summary = TrialSummary(path=path)
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = _load_metadata(data)
            summary.version = str(metadata.get("dataset_version", "unknown"))
            missing = [key for key in RECOMMENDED_KEYS if key not in data.files]
            if missing:
                summary.warnings.append("missing recommended key(s): " + ", ".join(missing))

            summary.frame_count = _frame_count(data, target_key)
            if summary.frame_count <= 0:
                summary.issues.append("no frames found")
                return summary

            if "diffPerDataAve" in data.files:
                sensor = np.asarray(data["diffPerDataAve"])
                summary.sensor_shape = tuple(sensor.shape[1:])
            summary.duration_sec, summary.hz = _duration_and_hz(data, summary.frame_count)
            summary.touch_count = _touch_count(data, summary.frame_count)
            summary.label_counts = _as_string_counter(data, "teaching_label")
            summary.mode_counts = _as_string_counter(data, "intended_mode")
            summary.source_counts = _as_string_counter(data, "teaching_source")
            summary.duplicate_sensor_frames = _duplicate_sensor_frame_count(data)
            summary.stale_dt_count = _stale_dt_count(data)

            if target_key not in data.files:
                summary.issues.append(f"missing training target key: {target_key}")
            else:
                target = np.asarray(data[target_key], dtype=np.float32)
                if target.shape != (summary.frame_count, 6):
                    summary.issues.append(
                        f"{target_key} shape is {target.shape}, expected ({summary.frame_count}, 6)"
                    )
                else:
                    summary.target_nonzero_count = _count_nonzero_rows(target)
                    summary.rotation_nonzero_count = _count_nonzero_rows(target, start_col=3)
                    summary.max_abs_target = np.max(np.abs(target), axis=0)
                    if expected_no_rotation and summary.rotation_nonzero_count:
                        summary.issues.append(
                            f"rotation target is nonzero in {summary.rotation_nonzero_count} frame(s)"
                        )

            if "manual_override_active" in data.files:
                manual = np.asarray(data["manual_override_active"]).reshape(-1)
                if manual.shape[0] == summary.frame_count:
                    summary.manual_count = int(np.count_nonzero(manual > 0))
            if "robot_feedback_valid" in data.files:
                feedback = np.asarray(data["robot_feedback_valid"]).reshape(-1)
                if feedback.shape[0] == summary.frame_count:
                    summary.robot_feedback_count = int(np.count_nonzero(feedback > 0))

            for key in CRITICAL_ARRAY_KEYS:
                if key not in data.files:
                    continue
                array = np.asarray(data[key])
                if array.shape[0] != summary.frame_count:
                    summary.issues.append(
                        f"{key} frame count is {array.shape[0]}, expected {summary.frame_count}"
                    )
                nonfinite = _critical_nonfinite_count(array)
                if nonfinite:
                    summary.issues.append(f"{key} has {nonfinite} non-finite value(s)")

            if summary.hz and summary.hz < 50.0:
                summary.warnings.append(f"recording rate is low: {summary.hz:.2f} Hz")
            if summary.hz and summary.hz > 70.0:
                summary.warnings.append(f"recording rate is high: {summary.hz:.2f} Hz")
            if summary.touch_count == 0:
                summary.warnings.append("no touch frames found")
            if summary.target_nonzero_count == 0:
                summary.warnings.append("all target velocities are zero")
            if summary.duplicate_sensor_frames:
                summary.warnings.append(
                    f"{summary.duplicate_sensor_frames} consecutive duplicate sensor frame(s)"
                )
            if summary.stale_dt_count:
                summary.warnings.append(f"{summary.stale_dt_count} frame gap(s) above 50 ms")
    except Exception as exc:
        summary.issues.append(f"failed to read file: {exc}")
    return summary


def _pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{100.0 * count / total:.1f}%"


def _print_counter(title: str, counter: Counter[str], total: int) -> None:
    print(f"\n{title}:")
    if not counter:
        print("  none")
        return
    for key, count in counter.most_common():
        print(f"  {key}: {count} ({_pct(count, total)})")


def _to_jsonable(summary: TrialSummary) -> dict[str, Any]:
    return {
        "path": str(summary.path),
        "frame_count": summary.frame_count,
        "duration_sec": summary.duration_sec,
        "hz": summary.hz,
        "sensor_shape": list(summary.sensor_shape) if summary.sensor_shape else None,
        "version": summary.version,
        "touch_count": summary.touch_count,
        "target_nonzero_count": summary.target_nonzero_count,
        "rotation_nonzero_count": summary.rotation_nonzero_count,
        "manual_count": summary.manual_count,
        "robot_feedback_count": summary.robot_feedback_count,
        "duplicate_sensor_frames": summary.duplicate_sensor_frames,
        "stale_dt_count": summary.stale_dt_count,
        "max_abs_target": summary.max_abs_target.tolist(),
        "label_counts": dict(summary.label_counts),
        "mode_counts": dict(summary.mode_counts),
        "source_counts": dict(summary.source_counts),
        "issues": list(summary.issues),
        "warnings": list(summary.warnings),
    }


def _write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def check_session(
    session_path: Path,
    *,
    target_key: str = DEFAULT_TARGET_KEY,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    expected_no_rotation: bool = True,
) -> tuple[list[TrialSummary], dict[str, Any]]:
    files = sorted(session_path.glob("*.npz")) if session_path.exists() else []
    summaries = [
        _summarize_trial(path, target_key=target_key, expected_no_rotation=expected_no_rotation)
        for path in files
    ]

    total_frames = sum(item.frame_count for item in summaries)
    total_duration = sum(item.duration_sec for item in summaries)
    total_touch = sum(item.touch_count for item in summaries)
    total_target = sum(item.target_nonzero_count for item in summaries)
    total_rotation = sum(item.rotation_nonzero_count for item in summaries)
    total_manual = sum(item.manual_count for item in summaries)
    total_feedback = sum(item.robot_feedback_count for item in summaries)
    duplicate_frames = sum(item.duplicate_sensor_frames for item in summaries)
    stale_dt = sum(item.stale_dt_count for item in summaries)
    total_samples = sum(max(0, item.frame_count - sequence_length + 1) for item in summaries)

    label_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    version_counts: Counter[str] = Counter()
    sensor_shapes: Counter[str] = Counter()
    max_abs_target = np.zeros(6, dtype=np.float32)
    for item in summaries:
        label_counts.update(item.label_counts)
        mode_counts.update(item.mode_counts)
        source_counts.update(item.source_counts)
        version_counts.update([item.version])
        if item.sensor_shape:
            sensor_shapes.update(["x".join(str(value) for value in item.sensor_shape)])
        max_abs_target = np.maximum(max_abs_target, item.max_abs_target)

    issues = [f"{item.path.name}: {issue}" for item in summaries for issue in item.issues]
    warnings = [f"{item.path.name}: {warning}" for item in summaries for warning in item.warnings]
    if not session_path.exists():
        issues.append(f"session path does not exist: {session_path}")
    elif not files:
        issues.append(f"no .npz files found in {session_path}")
    if total_samples <= 0 and files:
        issues.append(f"not enough frames for sequence length {sequence_length}")

    if not issues:
        if total_frames < 1000:
            warnings.append("dataset is small for training; collect more before serious experiments")
        if expected_no_rotation and total_rotation:
            issues.append(f"rotation target is nonzero in {total_rotation} frame(s)")

    payload = {
        "session_path": str(session_path),
        "target_key": target_key,
        "sequence_length": sequence_length,
        "expected_no_rotation": expected_no_rotation,
        "file_count": len(files),
        "total_frames": total_frames,
        "total_duration_sec": total_duration,
        "average_hz": (total_frames - len(files)) / total_duration if total_duration > 0 else 0.0,
        "training_sequence_samples": total_samples,
        "touch_count": total_touch,
        "target_nonzero_count": total_target,
        "rotation_nonzero_count": total_rotation,
        "manual_count": total_manual,
        "robot_feedback_count": total_feedback,
        "duplicate_sensor_frames": duplicate_frames,
        "stale_dt_count": stale_dt,
        "label_counts": dict(label_counts),
        "mode_counts": dict(mode_counts),
        "source_counts": dict(source_counts),
        "version_counts": dict(version_counts),
        "sensor_shapes": dict(sensor_shapes),
        "max_abs_target": dict(zip(AXIS_NAMES, (float(value) for value in max_abs_target))),
        "issues": issues,
        "warnings": warnings,
        "trials": [_to_jsonable(item) for item in summaries],
    }
    return summaries, payload


def print_report(summaries: list[TrialSummary], payload: dict[str, Any]) -> None:
    total_frames = int(payload["total_frames"])
    status = "FAIL" if payload["issues"] else "READY"
    if status == "READY" and payload["warnings"]:
        status = "READY WITH WARNINGS"

    print("\nAI Direct Finger Motion Dataset Check")
    print("=" * 38)
    print(f"Session: {payload['session_path']}")
    print(f"Target key: {payload['target_key']}")
    print(f"Sequence length: {payload['sequence_length']} frames")
    print(f"Status: {status}")
    print()
    print(f"Trials: {payload['file_count']}")
    print(f"Frames: {total_frames}")
    print(f"Duration: {payload['total_duration_sec']:.2f} s")
    print(f"Average Hz: {payload['average_hz']:.2f}")
    print(f"Training sequence samples: {payload['training_sequence_samples']}")
    print(f"Touch frames: {payload['touch_count']} ({_pct(payload['touch_count'], total_frames)})")
    print(
        "Nonzero target frames: "
        f"{payload['target_nonzero_count']} ({_pct(payload['target_nonzero_count'], total_frames)})"
    )
    print(
        "Manual teaching frames: "
        f"{payload['manual_count']} ({_pct(payload['manual_count'], total_frames)})"
    )
    print(
        "Robot feedback valid: "
        f"{payload['robot_feedback_count']} ({_pct(payload['robot_feedback_count'], total_frames)})"
    )
    print(f"Rotation target frames: {payload['rotation_nonzero_count']} ({_pct(payload['rotation_nonzero_count'], total_frames)})")
    print(f"Duplicate sensor frames: {payload['duplicate_sensor_frames']}")
    print(f"Frame gaps >50 ms: {payload['stale_dt_count']}")
    print(f"Dataset versions: {payload['version_counts']}")
    print(f"Sensor shapes: {payload['sensor_shapes']}")
    print("Max abs target velocity:")
    for axis, value in payload["max_abs_target"].items():
        print(f"  {axis}: {value:.6g}")

    print("\nPer Trial:")
    for item in summaries:
        print(
            "  "
            f"{item.path.name}: frames={item.frame_count}, "
            f"duration={item.duration_sec:.2f}s, hz={item.hz:.2f}, "
            f"touch={_pct(item.touch_count, item.frame_count)}, "
            f"manual={_pct(item.manual_count, item.frame_count)}, "
            f"rotation={item.rotation_nonzero_count}"
        )

    _print_counter("Intended Modes", Counter(payload["mode_counts"]), total_frames)
    _print_counter("Teaching Labels", Counter(payload["label_counts"]), total_frames)
    _print_counter("Teaching Sources", Counter(payload["source_counts"]), total_frames)

    if payload["issues"]:
        print("\nIssues:")
        for issue in payload["issues"]:
            print(f"  - {issue}")

    if payload["warnings"]:
        print("\nWarnings:")
        for warning in payload["warnings"]:
            print(f"  - {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check AI Direct Finger Motion .npz recordings before training."
    )
    parser.add_argument(
        "session",
        nargs="?",
        default=DEFAULT_SESSION,
        help=(
            "Session tag or path. Examples: first_trial_no_rotation, "
            "session_first_trial_no_rotation, /path/to/session_dir"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_default_dataset_root(),
        help="Root containing session_* directories.",
    )
    parser.add_argument(
        "--target-key",
        default=DEFAULT_TARGET_KEY,
        help="Training target array to validate.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Input sequence length planned for training.",
    )
    parser.add_argument(
        "--allow-rotation",
        action="store_true",
        help="Do not fail when rx/ry/rz target values are nonzero.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional path to save the same report as JSON.",
    )
    args = parser.parse_args()

    session_path = _session_path(args.session, args.dataset_root.expanduser())
    summaries, payload = check_session(
        session_path,
        target_key=args.target_key,
        sequence_length=max(1, int(args.sequence_length)),
        expected_no_rotation=not args.allow_rotation,
    )
    print_report(summaries, payload)

    if args.json_report:
        _write_json_report(args.json_report.expanduser(), payload)
        print(f"\nJSON report saved: {args.json_report.expanduser()}")

    return 1 if payload["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

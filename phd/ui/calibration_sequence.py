"""Recorded sequence labels, independent of the chart or connected hardware.

These are acquisition states, not contact detections or measured velocities.
In particular, never infer a fast return from distance or from a legacy speed.
"""

import math


RETURN_STAGE_LABELS = {
    "unloading": "Slow unloading",
    "fast_return": "Fast return",
    "return": "Return (unspecified)",
    "recovery": "Recovery (stationary)",
}


def _stage(row):
    phase = str(row.get("phase") or "")
    if phase == "returning":
        mode = str(row.get("step_mode") or "")
        kind = {"unloading": "unloading", "retract": "fast_return"}.get(mode, "return")
        return kind, RETURN_STAGE_LABELS[kind]
    if phase == "recovery":
        return phase, RETURN_STAGE_LABELS[phase]
    return phase, phase.replace("_", " ").capitalize() or "Stage not recorded"


def sequence_segments(rows):
    """Contiguous recorded stages; boundaries are the first sample of a stage.

    End times are exclusive except at the final sample. Missing/unknown stage
    tags remain unknown, and interruption rows break stages rather than being
    relabelled as motion. Source rows are never modified.
    """
    samples = []
    for index, row in enumerate(rows or []):
        try:
            elapsed = float(row.get("elapsed_s"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(elapsed):
            samples.append((elapsed, index, _stage(row)))
    samples.sort(key=lambda item: (item[0], item[1]))
    segments = []
    for elapsed, _index, (kind, label) in samples:
        if segments:
            segments[-1]["end_s"] = elapsed
        if not segments or kind != segments[-1]["kind"]:
            segments.append({"kind": kind, "label": label, "start_s": elapsed, "end_s": elapsed})
    return segments


def stage_at_time(segments, elapsed):
    """Return the recorded stage at the cursor, not the nearest remote stage."""
    if not segments or not segments[0]["start_s"] <= elapsed <= segments[-1]["end_s"]:
        return None
    for segment in reversed(segments):
        if elapsed >= segment["start_s"]:
            return segment
    return None


def return_stage_legend(segments):
    """One legend entry per return stage, retaining its first recorded onset."""
    first = {}
    for segment in segments:
        if segment["kind"] in RETURN_STAGE_LABELS:
            first.setdefault(segment["kind"], segment)
    return list(first.values())

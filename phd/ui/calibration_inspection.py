"""Read-only event snapping and saved-sample details for calibration charts."""

from bisect import bisect_left
import math

from phd.ui.calibration_sequence import sequence_segments, stage_at_time


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class TimelineInspection:
    """Index original samples once; never infer motion or alter measurements."""

    def __init__(self, rows, analysis, segments=None):
        self.samples = sorted(
            [(time, index, row) for index, row in enumerate(rows)
             if (time := _finite(row.get("elapsed_s"))) is not None],
            key=lambda item: (item[0], item[1]),
        )
        self.times = [item[0] for item in self.samples]
        self.segments = sequence_segments(rows) if segments is None else segments
        events = []

        def add(key, label, time):
            time = _finite(time)
            if time is not None and self.times and self.times[0] <= time <= self.times[-1]:
                events.append({"key": key, "label": label, "time": time})

        for key, label in (("signal_onset", "Proximity onset"), ("contact_onset", "Contact onset")):
            add(key, label, (analysis.get(key) or {}).get("elapsed_s"))
        forces = [(force, time) for time, _, row in self.samples
                  if (force := _finite(row.get("force_delta_abs_g"))) is not None]
        if forces:
            add("peak_force", "Peak force", max(forces)[1])
        returning = [time for time, _, row in self.samples if row.get("phase") == "returning"]
        if returning:
            add("return", "Return begins", returning[0])
        for segment in self.segments:
            kind = segment["kind"]
            if kind in {"fast_return", "recovery"}:
                add(kind, segment["label"], segment["start_s"])
            elif kind in {"unloading", "return"} and returning and segment["start_s"] != returning[0]:
                add("return", segment["label"] + " resumes", segment["start_s"])
        # Coincident events share one target, so neither hides the other.
        self.markers = []
        for event in sorted(events, key=lambda item: item["time"]):
            if self.markers and event["time"] == self.markers[-1]["time"]:
                self.markers[-1]["events"].append(event)
            else:
                self.markers.append({"time": event["time"], "events": [event]})

    def snap(self, time, x_range, pixel_width, tolerance_px=8):
        """Snap only to visible markers within a screen-space tolerance."""
        low, high = x_range
        if pixel_width <= 0 or high <= low:
            return time, None
        visible = [marker for marker in self.markers if low <= marker["time"] <= high]
        marker = min(visible, key=lambda item: abs(item["time"] - time), default=None)
        if marker and abs(marker["time"] - time) * pixel_width / (high - low) <= tolerance_px:
            return marker["time"], marker
        return time, None

    def sample_at(self, time):
        """Nearest saved row, not an interpolated or extrapolated measurement."""
        if not self.times or not self.times[0] <= time <= self.times[-1]:
            return None
        index = bisect_left(self.times, time)
        candidates = range(max(0, index - 1), min(len(self.samples), index + 1))
        return min((self.samples[i] for i in candidates), key=lambda item: abs(item[0] - time))[2]

    def details(self, time, signal_field, signal_label, marker=None):
        """Plain text; curve interpolation is intentionally not called raw data."""
        title = " / ".join(event["label"] for event in marker["events"]) if marker else "Curve inspection"
        lines = [title, f"Cursor time: {time:.4f} s"]
        stage = stage_at_time(self.segments, time)
        if stage:
            lines.append(f"Stage: {stage['label']}")
        sample = self.sample_at(time)
        if sample is None:
            return "\n".join(lines + ["No saved sample at this time."])
        lines.append(f"Nearest saved sample: {float(sample['elapsed_s']):.4f} s")

        def value(label, field, unit="", signed=True):
            number = _finite(sample.get(field))
            if number == 0:
                number = 0.0  # Avoid displaying a misleading negative zero.
            text = "not recorded" if number is None else format(number, "+.6g" if signed else ".6g") + unit
            lines.append(f"{label}: {text}")

        value(signal_label, signal_field)
        if signal_field not in {"raw", "raw_ave"} and _finite(sample.get("raw")) is not None:
            value("Raw sensor value", "raw", signed=False)
        value("Force change (magnitude)", "force_delta_abs_g", " g", signed=False)
        if _finite(sample.get("surface_distance_mm")) is None:
            lines.append("Distance from surface: unavailable (no contact reference)")
        else:
            value("Distance from surface", "surface_distance_mm", " mm")
        value("Robot downward travel", "travel_down_mm", " mm")
        if sample.get("step_mode"):
            lines.append(f"Recorded step: {sample['step_mode']} / {sample.get('step_state') or 'state not recorded'}")
        lines.append("Surface distance is contact-referenced TCP travel, not fabric compression.")
        return "\n".join(lines)

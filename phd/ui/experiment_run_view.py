"""Read-only experiment monitor. No device calls or experiment state mutations."""

from __future__ import annotations

import math
import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from phd.ui import components, theme
from phd.ui.calibration_progress_map import CalibrationProgressMap, calibration_progress_snapshot


NEWTONS_PER_GRAM = 0.00980665


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _label_text(host, name, fallback="—"):
    label = getattr(host, name, None)
    return label.text() if label is not None else fallback


def ended_run_message(task, stop_message="", latest_message=""):
    """Keep the actual stop cause alongside subsequent save/cleanup messages."""
    result = getattr(task, "last_result", None) or {}
    if not isinstance(result, dict):
        result = {}
    reason = str(getattr(task, "_stop_reason", "") or result.get("stop_reason") or "")
    detail = str(
        getattr(task, "_stop_message", "") or result.get("stop_message") or stop_message or ""
    ).strip()
    parts = []
    if reason:
        parts.append(f"Stop reason: {reason.replace('_', ' ')}")
    if detail:
        parts.append(detail)
    latest = str(latest_message or "").strip()
    summary = "\n".join(parts)
    if summary and (latest == summary or latest.startswith(summary + "\n")):
        return latest
    if latest and latest != detail:
        parts.append(latest)
    return "\n".join(parts) or "Run ended; no stop reason was reported."


def ended_run_state(task):
    """Colour a stop without mistaking a force target for a completed run."""
    result = getattr(task, "last_result", None)
    result = result if isinstance(result, dict) else {}
    reason = str(getattr(task, "_stop_reason", "") or result.get("stop_reason") or "")
    if reason == "complete":
        return "active"
    if reason in {"contact_threshold", "profile_complete"}:
        complete_field = (
            "_full_completed" if getattr(task, "id", "") == "full_taxel_calibration"
            else "_batch_completed"
        )
        return "active" if getattr(task, complete_field, False) else "warning"
    if not reason or reason in {"manual_stop", "partial_complete"}:
        return "warning"
    return "error"


def format_elapsed(seconds):
    """Keep long runs readable without wrapping hours at midnight."""
    seconds = _finite(seconds)
    if seconds is None:
        return "—"
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours} h {minutes:02d} min {seconds:02d} s"


def _force_sample(ui, task, now):
    """Read cached telemetry only; never poll or zero the force transmitter."""
    force = _finite(getattr(ui, "_force_meter_last_raw_newtons", None))
    stamp = _finite(getattr(ui, "_force_meter_last_sample_monotonic", None))
    if force is None or stamp is None or stamp > now:
        return None
    age = now - stamp
    limit = _finite(getattr(task, "force_sample_stale_sec", getattr(task, "force_stale_sec", 1.0)))
    limit = limit if limit is not None and limit > 0 else 1.0
    return force, age, age > limit


def force_readout(ui, task, now):
    """Use the same cached raw force and freshness limit as the active task."""
    sample = _force_sample(ui, task, now)
    if sample is None:
        return "—", "No valid force sample", "idle"
    force, age, stale = sample
    return (
        f"{force / NEWTONS_PER_GRAM:.1f} g",
        f"{'STALE · ' if stale else ''}Last sample {age:.1f} s ago",
        "warning" if stale else "active",
    )


def force_baseline_readout(ui, task, now):
    """Show the task's actual software reference, not the transmitter's tare.

    Mapping and calibration use _force_baseline_n; full-taxel clearance
    moves have a separate routing reference. Keep the displayed subtraction
    signed even where a safety guard uses the absolute change.
    """
    phase = str(getattr(task, "_phase", ""))
    if phase == "paused_force_meter":
        phase = str(getattr(task, "_force_paused_phase", ""))
    routing = (
        getattr(task, "id", "") == "full_taxel_calibration"
        and phase in {"full_moving", "full_waiting_force_route"}
    )
    field = "_routing_force_baseline_n" if routing else "_force_baseline_n"
    title = "Route baseline" if routing else "Baseline"
    if not hasattr(task, field):
        return "Corrected: —", "Baseline: not used by this task"
    baseline = _finite(getattr(task, field, None))
    # A reused task can still carry the previous trial's reference before
    # startup reaches stationary baseline collection. Do not present it as new.
    if phase in {"idle", "connecting_force_meter", "force_baseline", "full_waiting_force_route"} or (
        phase == "returning" and getattr(task, "_return_purpose", "") == "start"
    ):
        baseline = None
    if baseline is None:
        return "Corrected: —", f"{title}: waiting for measurement"
    reference = f"{title}: {baseline / NEWTONS_PER_GRAM:+.1f} g"
    sample = _force_sample(ui, task, now)
    if sample is None:
        return "Corrected: — (no valid sample)", reference
    force, _age, stale = sample
    if stale:
        return "Corrected: — (stale sample)", reference
    corrected_g = (force - baseline) / NEWTONS_PER_GRAM
    return f"Corrected: {corrected_g:+.1f} g", reference


class ExperimentRunView(QWidget):
    """Retain the last run snapshot, including after task cleanup resets fields."""

    PROGRESS_LABELS = {
        "calibration": "calibration_progress_label",
        "full_taxel_calibration": "full_taxel_calibration_progress_label",
        "taxel_position_mapping": "taxel_mapping_progress_label",
        "sigraph2026_scanning": "sigraph_scanning_progress_label",
    }

    def __init__(self, parent=None, *, clock=time.monotonic):
        super().__init__(parent)
        self._clock = clock
        self._task = None
        self._started_at = None
        self._running = False
        self._last_phase = ""
        self._end_message = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self.state = components.StatusPill("No active run", "idle")
        self.state.setWordWrap(True)
        layout.addWidget(self.state, alignment=Qt.AlignLeft)
        self.title = self._label("Run monitor", large=True)
        layout.addWidget(self.title)
        self.progress = self._label("Choose a task in Setup, then start when the rig is ready.")
        layout.addWidget(self.progress)
        self.taxel_progress = CalibrationProgressMap()
        layout.addWidget(self.taxel_progress)

        metrics = QGridLayout()
        metrics.setSpacing(12)
        self.elapsed, elapsed_card = self._metric("Elapsed · includes pauses", format_elapsed(0))
        self.force, force_card = self._metric("Force · raw reading", "—")
        self.force_corrected = self._label("Corrected: —")
        self.force_corrected.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {theme.TEXT_PRIMARY};"
        )
        self.force_baseline = components.muted(self._label("Baseline: waiting for experiment"))
        force_explanation = (
            "Corrected force = raw force − the experiment's current software baseline. "
            "The sign is preserved; force guards may use the absolute change. "
            "During full-taxel clearance moves, Route baseline is the separate routing reference. "
            "These values follow the task's baseline updates; they do not zero the "
            "transmitter or change any safety limits."
        )
        for label in (self.force_corrected, self.force_baseline):
            label.setToolTip(force_explanation)
            force_card.body.addWidget(label)
        self.force_age = components.StatusPill("No valid force sample", "idle")
        self.force_age.setWordWrap(True)
        force_card.layout().addWidget(self.force_age)
        metrics.addWidget(elapsed_card, 0, 0)
        metrics.addWidget(force_card, 0, 1)
        metrics.setColumnStretch(0, 1)
        metrics.setColumnStretch(1, 1)
        layout.addLayout(metrics)

        self.travel = self._label("Z travel: — mm | Max: — mm")
        self.travel.setToolTip(
            "Calibration only: measured downward TCP-Z displacement from the trial start, "
            "not total path length. Each new trial resets the maximum."
        )
        layout.addWidget(self.travel)
        layout.addWidget(components.divider())
        layout.addWidget(components.muted(QLabel("LATEST STATUS")))
        self.message = self._label("No experiment has started in this session.")
        self.message.setAccessibleName("Latest experiment status")
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.message.setFocusPolicy(Qt.StrongFocus)
        self._message_state = None
        self._set_message(self.message.text(), "idle")
        layout.addWidget(self.message)
        layout.addStretch(1)
        layout.addWidget(components.muted(QLabel("OUTPUT LOCATION")))
        self.output = self._label("Not created yet")
        self.output.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.output.setFocusPolicy(Qt.StrongFocus)
        layout.addWidget(self.output)
        hint = components.muted(self._label(
            "Read-only view · settings remain in Setup. Closing this window does not stop a run. "
            "An output path alone does not confirm that every trial was saved."
        ))
        layout.addWidget(hint)

    @staticmethod
    def _label(text, *, large=False):
        label = QLabel(text)
        label.setTextFormat(Qt.PlainText)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        if large:
            label.setStyleSheet(f"font-size: 22px; font-weight: 600; color: {theme.TEXT_PRIMARY};")
        return label

    def _metric(self, title, value):
        card = components.CardGroup(title)
        label = self._label(value, large=True)
        card.body.addWidget(label)
        return label, card

    def _set_message(self, text, state):
        self.message.setText(text)
        if state != self._message_state:
            color = theme.STATE_COLORS[state][0]
            self.message.setStyleSheet(
                f"font-size: 16px; font-weight: 600; color: {color};"
            )
            self._message_state = state

    def refresh(self, host):
        """Sample cached UI state; return True only when a new run is observed."""
        task = getattr(host, "_sidebar_active_task", None)
        if task is None:
            if self._running:
                self.finish(host)
            return False
        new_run = not self._running or task is not self._task
        now = self._clock()
        if new_run:
            self._task = task
            self._running = True
            self._started_at = now
            self._end_message = ""
        self._render_active(host, task, now)
        return new_run

    def _render_active(self, host, task, now):
        self.title.setText(str(getattr(task, "label", "Experiment")))
        phase = str(getattr(task, "_phase", "running"))
        self._last_phase = phase.replace("_", " ").capitalize()
        waiting = any(word in phase for word in ("paus", "recover", "connect", "wait"))
        self.state.set_state("warning" if waiting else "active", self._last_phase)
        self.elapsed.setText(format_elapsed(now - self._started_at))
        task_id = str(getattr(task, "id", ""))
        self.progress.setText(_label_text(
            host, self.PROGRESS_LABELS.get(task_id, ""), "Experiment is active."
        ))
        text, age, state = force_readout(getattr(host, "ui_ros", None), task, now)
        self.force.setText(text)
        self.force_age.set_state(state, age)
        corrected, baseline = force_baseline_readout(getattr(host, "ui_ros", None), task, now)
        self.force_corrected.setText(corrected)
        self.force_baseline.setText(baseline)
        self.travel.setText(
            _label_text(host, "status_calibration_travel", "Z travel: — mm | Max: — mm")
            if task_id in {"calibration", "full_taxel_calibration"}
            else "Z travel: not reported for this task"
        )
        self._set_message(
            _label_text(host, "info_process", "Running"), "warning" if waiting else "active"
        )
        self.taxel_progress.display(calibration_progress_snapshot(task))
        self._update_output(task)

    def _update_output(self, task):
        folder = (
            getattr(task, "_full_directory", None)
            or getattr(task, "_batch_directory", None)
            or getattr(task, "_output_folder", None)
        )
        self.output.setText(str(folder) if folder else "Not created / not reported by this task")

    def finish(self, host, message=None):
        """Freeze before on_stop clears telemetry; never infer successful completion."""
        if not self._running:
            return
        self._render_active(host, self._task, self._clock())
        self._running = False
        self.taxel_progress.display(self.taxel_progress.snapshot, frozen=True)
        self._end_message = str(message or "")
        self.state.set_state("idle", f"Run ended · last phase: {self._last_phase}")
        self.force_age.set_state("idle", "Last run snapshot · not live")
        self.update_end_details(host)

    def update_end_details(self, host):
        """Capture final save/error messages after cleanup, without losing metrics."""
        if self._task is not None and not self._running:
            self._set_message(ended_run_message(
                self._task, self._end_message, _label_text(host, "info_process", "Run ended")
            ), ended_run_state(self._task))
            self._update_output(self._task)
            reason = str(getattr(self._task, "_stop_reason", "") or "")
            if reason:
                self.state.set_state("idle", f"Run ended · {reason.replace('_', ' ')}")

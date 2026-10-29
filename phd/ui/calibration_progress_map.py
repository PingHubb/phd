"""Read-only, per-pass calibration progress. Never calls devices or changes tasks."""
from dataclasses import dataclass

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from phd.ui import theme


def _integer(value, default=None):
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return default


def _finished_trial(results, number):
    # Recent trials first: the current pass is normally the latest saved entry.
    for result in reversed(results or []):
        if not isinstance(result, dict):
            continue
        if _integer(result.get("trial_number")) == number:
            return (result.get("measurement_quality") or {}).get("trial_completed") is True
    return False


@dataclass(frozen=True)
class CalibrationProgress:
    rows: int
    columns: int
    planned: frozenset
    completed: frozenset
    current: int | None
    number: int
    total: int
    unit: str
    sensor: str
    paused: bool


def calibration_progress_snapshot(task):
    """Copy small cached fields only; do not read files, frames or robot APIs."""
    task_id = getattr(task, "id", "")
    if task_id not in {"calibration", "full_taxel_calibration"}:
        return None
    position_map = getattr(task, "position_map", None) or {}
    settings = position_map.get("settings") or {}
    rows = _integer(settings.get("sensor_rows", getattr(task, "sensor_rows", None)))
    columns = _integer(settings.get("sensor_columns", getattr(task, "sensor_columns", None)))
    if rows is None or columns is None or not (0 < rows <= 64 and 0 < columns <= 64):
        return None
    phase = str(getattr(task, "_phase", ""))
    connecting = phase == "connecting_force_meter"
    paused = "paused" in phase or phase == "recovering_force_meter"
    full = task_id == "full_taxel_calibration"
    total = max(1, _integer(getattr(task, "repeat_count", 1), 1))
    number = 1 if connecting else max(1, _integer(
        getattr(task, "_pass_number" if full else "_trial_number", 1), 1))
    completed = set()
    current = None
    if full:
        # While connecting, old run fields may not have been reset yet.
        targets = None if connecting else getattr(task, "_targets", None)
        targets = targets or position_map.get("positions") or []
        planned = {_integer(record.get("taxel_index")) for record in targets if isinstance(record, dict)}
        planned = {index for index in planned if index is not None and 0 <= index < rows * columns}
        if not connecting:
            trials = getattr(task, "_trials_by_taxel", {}) or {}
            completed = {index for index in planned if _finished_trial(
                trials.get(index, trials.get(str(index), [])), number)}
            target = getattr(task, "_current_target", None)
            if isinstance(target, dict) and _integer(getattr(task, "_target_cursor", -1), -1) >= 0:
                current = _integer(target.get("taxel_index"))
    else:
        target = _integer(getattr(task, "taxel_index", None))
        planned = {target} if target is not None and 0 <= target < rows * columns else set()
        if not connecting:
            current = target
            if _finished_trial(getattr(task, "_batch_results", []), number):
                completed = set(planned)
    if not planned:
        return None
    return CalibrationProgress(
        rows, columns, frozenset(planned), frozenset(completed),
        current if current in planned else None, number, total,
        "Pass" if full else "Trial", str(getattr(task, "active_sensor_label", "") or "Sensor"), paused,
    )


class CalibrationProgressMap(QWidget):
    """Label cells, not buttons: no selection signal or movement action exists."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot = None
        self.frozen = False
        self.cells = {}
        self._shape = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)
        self.heading = QLabel("TAXEL PROGRESS · READ ONLY")
        self.heading.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-weight: 600;")
        layout.addWidget(self.heading)
        self.summary = QLabel()
        self.summary.setTextFormat(Qt.PlainText)
        self.summary.setWordWrap(True)
        self.summary.setMinimumWidth(0)
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.summary)
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.container = QWidget()
        horizontal = QHBoxLayout(self.container)
        horizontal.setContentsMargins(0, 0, 0, 0)
        self.grid_widget = QWidget()
        self.grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)
        horizontal.addWidget(self.grid_widget, alignment=Qt.AlignHCenter)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)
        self.legend = QLabel("▶ Current   ✓ Completed this pass   · Waiting   — Not in this run\n"
                             "Ⅱ Paused   ! Stopped unfinished · Hover a cell for details; clicking does not move the robot.")
        self.legend.setWordWrap(True)
        self.legend.setMinimumWidth(0)
        self.legend.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.legend.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: {theme.FONT_SIZE_SMALL};")
        layout.addWidget(self.legend)
        self.setAccessibleName("Read-only calibration taxel progress map")
        self.hide()

    def _rebuild(self, rows, columns):
        if self._shape == (rows, columns):
            return
        self._shape = rows, columns
        while self.grid.count():
            old = self.grid.takeAt(0).widget()
            old.hide()
            old.deleteLater()
        # A fresh layout also drops the old row/column minimums after shrinking.
        QWidget().setLayout(self.grid)
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)
        self.cells = {}
        cell_width = 30 if columns <= 10 else 24
        for column in range(columns):
            for row in range(rows):
                index = column * rows + row  # Same top-down, column-major order as Setup.
                cell = QLabel()
                cell.setAlignment(Qt.AlignCenter)
                cell.setTextFormat(Qt.PlainText)
                cell.setMinimumWidth(cell_width)
                cell.setFixedHeight(30)
                cell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                self.grid.addWidget(cell, row, column)
                self.cells[index] = cell
        self.grid_widget.setMinimumWidth(columns * cell_width + (columns - 1) * 2)
        self.grid_widget.setMaximumWidth(columns * 42 + (columns - 1) * 2)
        # Horizontal scrolling is a fallback for grids wider than a small window.
        self.scroll.setFixedHeight(rows * 30 + (rows - 1) * 2 + 18)

    def display(self, snapshot, *, frozen=False):
        if snapshot is None:
            self.snapshot = None
            self.hide()
            return
        self.show()
        if snapshot == self.snapshot and frozen == self.frozen:
            return  # No rebuilding/repolishing a 400-taxel grid every monitor tick.
        self.snapshot, self.frozen = snapshot, frozen
        self._rebuild(snapshot.rows, snapshot.columns)
        pending = snapshot.planned - snapshot.completed
        current = snapshot.current if snapshot.current in pending else None
        waiting = pending - ({current} if current is not None else set())
        prefix = "Last run snapshot · " if frozen else ""
        active_text = f"Taxel {current}" if current is not None else "—"
        if not frozen and snapshot.current in snapshot.completed:
            active_text = f"Taxel {snapshot.current} (completed)"
        self.legend.setText(
            f"▶ Current   ✓ Completed this {snapshot.unit.lower()}   · Waiting   — Not in this run\n"
            "Ⅱ Paused   ! Stopped unfinished · Hover a cell for details; clicking does not move the robot."
        )
        self.summary.setText(
            f"{prefix}{snapshot.sensor} · {snapshot.rows}×{snapshot.columns} · "
            f"{snapshot.unit} {snapshot.number}/{snapshot.total}\n"
            f"{'Last unfinished' if frozen else 'Current'}: {active_text} · "
            f"Completed: {len(snapshot.completed)}/{len(snapshot.planned)} · "
            f"{'Not started' if frozen else 'Waiting'}: {len(waiting)}"
        )
        palette = {
            "completed": ("✓", theme.SUCCESS_SOFT, theme.SUCCESS, theme.SUCCESS_BORDER, "Completed"),
            "current": ("▶", theme.ACCENT, theme.TEXT_ON_ACCENT, theme.ACCENT_PRESSED, "Current"),
            "paused": ("Ⅱ", theme.WARNING_SOFT, theme.WARNING, theme.WARNING_BORDER, "Paused"),
            "stopped": ("!", theme.WARNING_SOFT, theme.WARNING, theme.WARNING_BORDER, "Stopped before completion"),
            "waiting": ("·", theme.INPUT_BG, theme.TEXT_PRIMARY, theme.BORDER, "Waiting"),
            "excluded": ("—", theme.SURFACE_SUNKEN, theme.TEXT_MUTED, theme.BORDER_SUBTLE, "Not in this run / unmapped"),
        }
        for index, cell in self.cells.items():
            if index in snapshot.completed:
                state = "completed"
            elif index == current:
                state = "stopped" if frozen else "paused" if snapshot.paused else "current"
            else:
                state = "waiting" if index in snapshot.planned else "excluded"
            mark, background, foreground, border, description = palette[state]
            cell.setProperty("progressState", state)
            cell.setText(f"{index}\n{mark}")
            cell.setStyleSheet(
                f"background: {background}; color: {foreground}; border: 1px solid {border}; "
                "border-radius: 3px; font-size: 8pt; padding: 0;"
            )
            text = (f"Taxel {index}: row {index % snapshot.rows}, column {index // snapshot.rows} · "
                    f"{description} · {snapshot.unit.lower()} {snapshot.number}/{snapshot.total}")
            cell.setToolTip(text)
            cell.setAccessibleName(text)
        self.setAccessibleDescription(self.summary.text())

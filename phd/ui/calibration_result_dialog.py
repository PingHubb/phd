"""Result plots for the force-versus-tactile calibration experiment."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QRectF, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPainterPath
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from phd.ui import components, theme
from phd.ui.calibration_analysis import (
    APPROACH_PHASES,
    RETURN_PHASE,
    analyze_calibration_rows,
    available_taxel_indices,
    calibration_rows_for_taxel,
)


SIGNAL_LABELS = {
    "raw": "Raw sensor value",
    "raw_ave": "Averaged raw value",
    "diff": "Sensor difference (counts)",
    "diff_ave": "Averaged difference (counts)",
    "diff_percent": "Relative change (%)",
    "diff_percent_ave": "Averaged relative change (%)",
}


class CalibrationTaxelMap(QWidget):
    """Compact clickable map using the viewer's column-major taxel indices."""

    taxelSelected = pyqtSignal(int)

    def __init__(self, rows=8, columns=10, parent=None):
        super().__init__(parent)
        self.rows = max(1, int(rows))
        self.columns = max(1, int(columns))
        self.buttons = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(3)
        grid.setVerticalSpacing(3)
        for column in range(self.columns):
            for display_row in range(self.rows):
                index = column * self.rows + display_row
                button = QToolButton(self)
                button.setText(str(index))
                button.setCheckable(True)
                button.setAutoExclusive(True)
                button.setFixedSize(31, 25)
                button.setToolTip(
                    f"Taxel {index}: row {display_row}, column {column}"
                )
                button.clicked.connect(
                    lambda _checked=False, value=index: (
                        self.taxelSelected.emit(value)
                    )
                )
                grid.addWidget(button, display_row, column)
                self.buttons[index] = button
        self.setStyleSheet(
            f"QToolButton {{ background: {theme.INPUT_BG}; "
            f"color: {theme.TEXT_PRIMARY}; border: 1px solid {theme.BORDER}; "
            "border-radius: 4px; font-size: 8pt; padding: 0; }"
            f"QToolButton:hover {{ border-color: {theme.ACCENT}; }}"
            f"QToolButton:checked {{ background: {theme.ACCENT}; "
            f"color: {theme.TEXT_ON_ACCENT}; border-color: {theme.ACCENT}; }}"
            f"QToolButton:disabled {{ background: {theme.NEUTRAL_SOFT}; "
            f"color: {theme.TEXT_FAINT}; border-color: {theme.BORDER_SUBTLE}; }}"
        )

    def set_taxels(self, available, selected=None):
        """Enable recorded taxels and check the selected one."""
        available = {int(index) for index in available}
        for index, button in self.buttons.items():
            button.setEnabled(index in available)
            button.setChecked(selected is not None and index == int(selected))


class CalibrationCurveWidget(QWidget):
    """Draw the calibration process or one pairwise physical relationship."""

    detailRequested = pyqtSignal()

    FORCE_COLOR = "#42a5f5"
    SIGNAL_COLOR = "#ff8a65"
    DISTANCE_COLOR = "#66bb6a"
    RETURN_COLOR = "#ab47bc"
    FIT_COLOR = "#fdd835"
    MEAN_COLOR = "#fdd835"
    TRIAL_COLORS = (
        "#42a5f5",
        "#ff8a65",
        "#66bb6a",
        "#ab47bc",
        "#26c6da",
        "#ec407a",
        "#9ccc65",
        "#ffa726",
    )

    def __init__(self, rows, signal_field, mode="process", parent=None):
        super().__init__(parent)
        self.signal_field = str(signal_field)
        self.mode = str(mode)
        self.rows = []
        self.trial_results = []
        self.analysis = {}
        self._view_x_range = None
        self._view_y_range = None
        self._last_plot_rect = None
        self._last_x_range = None
        self._last_y_range = None
        self._drag_origin = None
        self._drag_x_range = None
        self._drag_y_range = None
        self._hover_position = None
        self._hover_contexts = []
        self._base_image = None
        self.setMinimumHeight(430)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setToolTip(
            "Move mouse: inspect values · Mouse wheel: zoom · Left-drag: "
            "pan · Double-click: open a large detailed view"
        )
        self.set_data(rows, signal_field)

    def set_data(self, rows, signal_field) -> None:
        self.signal_field = str(signal_field)
        self.rows = self._normalized_rows(rows)
        self.analysis = analyze_calibration_rows(
            self.rows,
            self.signal_field,
        )
        self._hover_position = None
        self.reset_view()
        self.update()

    def set_trials(self, trial_results, signal_field) -> None:
        """Set repeated trials used by the batch repeatability graph."""
        self.signal_field = str(signal_field)
        normalized = []
        for index, result in enumerate(trial_results or [], start=1):
            rows = self._normalized_rows((result or {}).get("rows") or [])
            if not rows:
                continue
            normalized.append(
                {
                    "trial_number": int(
                        (result or {}).get("trial_number", index) or index
                    ),
                    "rows": rows,
                }
            )
        self.trial_results = normalized
        self.rows = []
        self.analysis = {}
        self._hover_position = None
        self.reset_view()
        self.update()

    @staticmethod
    def _valid_range(value_range):
        return (
            value_range is not None
            and len(value_range) == 2
            and math.isfinite(float(value_range[0]))
            and math.isfinite(float(value_range[1]))
            and float(value_range[1]) - float(value_range[0]) > 1e-12
        )

    def _display_range(self, base_range, axis):
        override = (
            self._view_x_range if axis == "x" else self._view_y_range
        )
        return override if self._valid_range(override) else base_range

    def reset_view(self):
        self._view_x_range = None
        self._view_y_range = None
        self._drag_origin = None
        self._base_image = None
        self.update()

    @staticmethod
    def _zoomed_range(value_range, factor, anchor_fraction):
        lower, upper = (float(value_range[0]), float(value_range[1]))
        span = upper - lower
        factor = max(0.1, min(10.0, float(factor)))
        anchor_fraction = max(0.0, min(1.0, float(anchor_fraction)))
        anchor = lower + span * anchor_fraction
        new_span = max(1e-12, span * factor)
        return (
            anchor - new_span * anchor_fraction,
            anchor + new_span * (1.0 - anchor_fraction),
        )

    def zoom_view(self, factor, anchor=None):
        rect = self._last_plot_rect
        if (
            rect is None
            or not self._valid_range(self._last_x_range)
            or rect.width() <= 0.0
            or rect.height() <= 0.0
        ):
            return
        if anchor is None:
            x_fraction = 0.5
            y_fraction = 0.5
        else:
            x_fraction = (float(anchor.x()) - rect.left()) / rect.width()
            y_fraction = (rect.bottom() - float(anchor.y())) / rect.height()
        self._view_x_range = self._zoomed_range(
            self._last_x_range,
            factor,
            x_fraction,
        )
        if self.mode != "process" and self._valid_range(self._last_y_range):
            self._view_y_range = self._zoomed_range(
                self._last_y_range,
                factor,
                y_fraction,
            )
        self._base_image = None
        self.update()

    def wheelEvent(self, event):
        rect = self._last_plot_rect
        if rect is None or not rect.contains(event.pos()):
            super().wheelEvent(event)
            return
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        self.zoom_view(factor, event.pos())
        event.accept()

    def mousePressEvent(self, event):
        rect = self._last_plot_rect
        if (
            event.button() == Qt.LeftButton
            and rect is not None
            and rect.contains(event.pos())
            and self._valid_range(self._last_x_range)
        ):
            self._drag_origin = event.pos()
            self._drag_x_range = tuple(self._last_x_range)
            self._drag_y_range = (
                tuple(self._last_y_range)
                if self._valid_range(self._last_y_range)
                else None
            )
            self._hover_position = None
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is None:
            plot_rect = self._last_plot_rect
            hover_position = event.pos()
            if plot_rect is not None and plot_rect.contains(hover_position):
                self._hover_position = hover_position
                self.update()
                event.accept()
                return
            if self._hover_position is not None:
                self._hover_position = None
                self.update()
            super().mouseMoveEvent(event)
            return
        if self._last_plot_rect is None:
            super().mouseMoveEvent(event)
            return
        rect = self._last_plot_rect
        delta_x = float(event.pos().x() - self._drag_origin.x())
        x_span = self._drag_x_range[1] - self._drag_x_range[0]
        x_shift = -delta_x * x_span / max(1.0, rect.width())
        self._view_x_range = (
            self._drag_x_range[0] + x_shift,
            self._drag_x_range[1] + x_shift,
        )
        if self.mode != "process" and self._drag_y_range is not None:
            delta_y = float(event.pos().y() - self._drag_origin.y())
            y_span = self._drag_y_range[1] - self._drag_y_range[0]
            y_shift = delta_y * y_span / max(1.0, rect.height())
            self._view_y_range = (
                self._drag_y_range[0] + y_shift,
                self._drag_y_range[1] + y_shift,
            )
        self._base_image = None
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_origin is not None:
            self._drag_origin = None
            self._drag_x_range = None
            self._drag_y_range = None
            self.setCursor(Qt.CrossCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if self._drag_origin is None and self._hover_position is not None:
            self._hover_position = None
            self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.detailRequested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def save_png(self, path) -> bool:
        path = Path(path).expanduser()
        image = QImage(self.size(), QImage.Format_ARGB32)
        image.fill(QColor(theme.INPUT_BG))
        painter = QPainter(image)
        self.render(painter)
        painter.end()
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(image.save(str(path), "PNG"))

    @staticmethod
    def _normalized_rows(rows):
        normalized = []
        for source in rows or []:
            row = dict(source or {})
            if row.get("force_delta_abs_g") is None:
                try:
                    row["force_delta_abs_g"] = (
                        abs(float(row.get("force_delta_abs_n"))) / 0.00980665
                    )
                except (TypeError, ValueError):
                    pass
            normalized.append(row)
        return normalized

    @staticmethod
    def _finite_points(rows, x_field, y_field, phases=None):
        points = []
        for row in rows:
            if phases is not None and str(row.get("phase", "")) not in phases:
                continue
            try:
                x_value = float(row.get(x_field))
                y_value = float(row.get(y_field))
            except (TypeError, ValueError):
                continue
            if math.isfinite(x_value) and math.isfinite(y_value):
                points.append((x_value, y_value))
        return points

    @classmethod
    def _repeatability_statistics(
        cls,
        trial_rows,
        y_field,
        sample_count=180,
    ):
        """Interpolate trials over their common distance and return mean ± SD."""
        curves = []
        for rows in trial_rows or []:
            source = cls._finite_points(
                rows,
                "travel_down_mm",
                y_field,
                phases=APPROACH_PHASES,
            )
            grouped = defaultdict(list)
            for x_value, y_value in source:
                grouped[float(x_value)].append(float(y_value))
            curve = [
                (x_value, float(np.mean(grouped[x_value])))
                for x_value in sorted(grouped)
            ]
            if len(curve) >= 2:
                curves.append(curve)

        payload = {
            "curves": curves,
            "mean": [],
            "lower": [],
            "upper": [],
            "trial_count": len(curves),
        }
        if not curves:
            return payload

        common_min = max(curve[0][0] for curve in curves)
        common_max = min(curve[-1][0] for curve in curves)
        if common_max - common_min <= 1e-12:
            return payload

        grid = np.linspace(
            float(common_min),
            float(common_max),
            max(2, int(sample_count)),
        )
        interpolated = np.asarray(
            [
                np.interp(
                    grid,
                    [point[0] for point in curve],
                    [point[1] for point in curve],
                )
                for curve in curves
            ],
            dtype=float,
        )
        mean = np.mean(interpolated, axis=0)
        standard_deviation = (
            np.std(interpolated, axis=0, ddof=1)
            if len(curves) > 1
            else np.zeros_like(mean)
        )
        payload["mean"] = list(zip(grid.tolist(), mean.tolist()))
        payload["lower"] = list(
            zip(grid.tolist(), (mean - standard_deviation).tolist())
        )
        payload["upper"] = list(
            zip(grid.tolist(), (mean + standard_deviation).tolist())
        )
        return payload

    @staticmethod
    def _sample_at_x(points, x_value, x_values=None, x_bounds=None):
        """Interpolate the local curve segment nearest to a requested x value."""
        if not points:
            return None
        x_value = float(x_value)
        if x_values is None:
            x_values = np.asarray(
                [float(point[0]) for point in points], dtype=float
            )
        if x_bounds is None:
            minimum = float(np.min(x_values))
            maximum = float(np.max(x_values))
        else:
            minimum, maximum = (float(value) for value in x_bounds)
        if x_value < minimum or x_value > maximum:
            return None
        nearest_index = int(np.argmin(np.abs(x_values - x_value)))
        segment_indices = []
        if nearest_index > 0:
            segment_indices.append(nearest_index - 1)
        if nearest_index + 1 < len(points):
            segment_indices.append(nearest_index)
        for index in segment_indices:
            x_start, y_start = points[index]
            x_end, y_end = points[index + 1]
            if not min(x_start, x_end) <= x_value <= max(x_start, x_end):
                continue
            if abs(x_end - x_start) <= 1e-12:
                return (x_value, float(y_start))
            fraction = (x_value - x_start) / (x_end - x_start)
            return (
                x_value,
                float(y_start) + fraction * (float(y_end) - float(y_start)),
            )
        nearest = points[nearest_index]
        return (x_value, float(nearest[1]))

    def _add_hover_context(
        self,
        rect,
        x_range,
        y_range,
        points,
        color,
        label,
        *,
        time_axis=False,
    ):
        if points:
            x_values = np.asarray(
                [float(point[0]) for point in points], dtype=float
            )
            self._hover_contexts.append(
                {
                    "rect": QRectF(rect),
                    "x_range": tuple(x_range),
                    "y_range": tuple(y_range),
                    "points": points,
                    "x_values": x_values,
                    "x_bounds": (
                        float(np.min(x_values)),
                        float(np.max(x_values)),
                    ),
                    "color": str(color),
                    "label": str(label),
                    "time_axis": bool(time_axis),
                }
            )

    @staticmethod
    def _hover_number(value, signed=False):
        value = float(value)
        absolute = abs(value)
        if absolute >= 10000.0 or (0.0 < absolute < 0.001):
            specifier = "+.3e" if signed else ".3e"
        elif absolute >= 100.0:
            specifier = "+.2f" if signed else ".2f"
        else:
            specifier = "+.5g" if signed else ".5g"
        return format(value, specifier)

    def _draw_hover_label(
        self,
        painter,
        marker_x,
        marker_y,
        text,
        color,
        rect,
        stack_index=None,
    ):
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 14
        height = metrics.height() + 8
        if stack_index is None:
            left = marker_x + 9.0
            if left + width > rect.right():
                left = marker_x - width - 9.0
            top = marker_y - height - 7.0
            if top < rect.top():
                top = marker_y + 7.0
        else:
            left = rect.right() - width - 7.0
            top = rect.top() + 7.0 + int(stack_index) * (height + 4.0)
        left = max(rect.left(), min(left, rect.right() - width))
        top = max(rect.top(), min(top, rect.bottom() - height))
        label_rect = QRectF(left, top, width, height)
        background = QColor(theme.SURFACE_RAISED)
        background.setAlpha(238)
        painter.fillRect(label_rect, background)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(color), 1))
        painter.drawRect(label_rect)
        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(
            label_rect.adjusted(7.0, 2.0, -7.0, -2.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            text,
        )

    def _paint_hover(self, painter):
        position = self._hover_position
        plot_rect = self._last_plot_rect
        x_range = self._last_x_range
        if (
            position is None
            or plot_rect is None
            or not plot_rect.contains(position)
            or not self._valid_range(x_range)
        ):
            return
        fraction = (float(position.x()) - plot_rect.left()) / plot_rect.width()
        fraction = max(0.0, min(1.0, fraction))
        x_value = x_range[0] + fraction * (x_range[1] - x_range[0])
        line_x = plot_rect.left() + fraction * plot_rect.width()

        painter.save()
        cursor_color = QColor(theme.TEXT_PRIMARY)
        cursor_color.setAlpha(180)
        painter.setPen(QPen(cursor_color, 1.5, Qt.DashLine))
        painter.drawLine(
            int(line_x),
            int(plot_rect.top()),
            int(line_x),
            int(plot_rect.bottom()),
        )
        labels_by_rect = defaultdict(int)
        for context in self._hover_contexts:
            point = self._sample_at_x(
                context["points"],
                x_value,
                x_values=context["x_values"],
                x_bounds=context["x_bounds"],
            )
            if point is None:
                continue
            rect = context["rect"]
            marker_x, marker_y = self._map_point(
                point[0],
                point[1],
                context["x_range"],
                context["y_range"],
                rect,
            )
            if not rect.contains(marker_x, marker_y):
                continue
            color = context["color"]
            painter.setPen(QPen(QColor("#101318"), 2))
            painter.setBrush(QColor(color))
            painter.drawEllipse(int(marker_x - 5), int(marker_y - 5), 10, 10)
            x_text = self._hover_number(point[0])
            y_text = self._hover_number(point[1], signed=True)
            if context["time_axis"]:
                text = f"{context['label']}: t={x_text} s, value={y_text}"
            else:
                text = f"{context['label']}: x={x_text}, y={y_text}"
            rect_key = (
                round(rect.left()),
                round(rect.top()),
                round(rect.width()),
                round(rect.height()),
            )
            label_index = labels_by_rect[rect_key]
            labels_by_rect[rect_key] += 1
            self._draw_hover_label(
                painter,
                marker_x,
                marker_y,
                text,
                color,
                rect,
                stack_index=(
                    None if context["time_axis"] else label_index
                ),
            )
        painter.restore()

    @staticmethod
    def _range(
        values,
        padding=0.08,
        include_zero=False,
        nonnegative=False,
    ):
        minimum = min(values)
        maximum = max(values)
        if include_zero or nonnegative:
            minimum = min(0.0, minimum)
            maximum = max(0.0, maximum)
        if abs(maximum - minimum) < 1e-12:
            extra = max(1.0, abs(minimum) * 0.1)
            lower = minimum - extra
            upper = maximum + extra
            return (max(0.0, lower), upper) if nonnegative else (lower, upper)
        extra = (maximum - minimum) * float(padding)
        lower = minimum - extra
        upper = maximum + extra
        return (max(0.0, lower), upper) if nonnegative else (lower, upper)

    @staticmethod
    def _map_point(x, y, x_range, y_range, rect):
        x_ratio = (x - x_range[0]) / (x_range[1] - x_range[0])
        y_ratio = (y - y_range[0]) / (y_range[1] - y_range[0])
        return (
            rect.left() + x_ratio * rect.width(),
            rect.bottom() - y_ratio * rect.height(),
        )

    @classmethod
    def _draw_series(cls, painter, points, x_range, y_range, rect, color):
        if not points:
            return
        painter.save()
        painter.setClipRect(rect)
        painter.setBrush(Qt.NoBrush)
        path = QPainterPath()
        first = cls._map_point(points[0][0], points[0][1], x_range, y_range, rect)
        path.moveTo(*first)
        for x_value, y_value in points[1:]:
            path.lineTo(*cls._map_point(x_value, y_value, x_range, y_range, rect))
        painter.setPen(QPen(QColor(color), 2))
        painter.drawPath(path)
        painter.restore()

    @classmethod
    def _draw_points(cls, painter, points, x_range, y_range, rect, color):
        if not points:
            return
        painter.save()
        painter.setClipRect(rect)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(color))
        step = max(1, len(points) // 100)
        for x_value, y_value in points[::step]:
            px, py = cls._map_point(
                x_value,
                y_value,
                x_range,
                y_range,
                rect,
            )
            painter.drawEllipse(int(px - 2), int(py - 2), 4, 4)
        painter.restore()

    @classmethod
    def _draw_variation_band(
        cls,
        painter,
        lower_points,
        upper_points,
        x_range,
        y_range,
        rect,
        color,
    ):
        if not lower_points or len(lower_points) != len(upper_points):
            return
        painter.save()
        painter.setClipRect(rect)
        fill = QColor(color)
        fill.setAlpha(52)
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        path = QPainterPath()
        first = cls._map_point(
            lower_points[0][0],
            lower_points[0][1],
            x_range,
            y_range,
            rect,
        )
        path.moveTo(*first)
        for x_value, y_value in lower_points[1:]:
            path.lineTo(
                *cls._map_point(x_value, y_value, x_range, y_range, rect)
            )
        for x_value, y_value in reversed(upper_points):
            path.lineTo(
                *cls._map_point(x_value, y_value, x_range, y_range, rect)
            )
        path.closeSubpath()
        painter.drawPath(path)
        painter.restore()

    @staticmethod
    def _draw_grid(painter, rect):
        painter.setPen(QPen(QColor(theme.BORDER_SUBTLE), 1, Qt.DashLine))
        for index in range(1, 5):
            x_value = rect.left() + rect.width() * index / 5.0
            painter.drawLine(
                int(x_value), int(rect.top()), int(x_value), int(rect.bottom())
            )
        for index in range(1, 4):
            y_value = rect.top() + rect.height() * index / 4.0
            painter.drawLine(
                int(rect.left()), int(y_value), int(rect.right()), int(y_value)
            )
        painter.setPen(QPen(QColor("#7d8794"), 1))
        painter.drawRect(rect)

    @staticmethod
    def _axis_value_text(value):
        value = float(value)
        absolute = abs(value)
        if absolute >= 10000.0 or (0.0 < absolute < 0.001):
            return f"{value:.3e}"
        if absolute >= 100.0:
            return f"{value:.1f}"
        if absolute >= 10.0:
            return f"{value:.2f}"
        return f"{value:.4g}"

    @classmethod
    def _draw_x_ticks(cls, painter, rect, value_range):
        painter.setPen(QColor(theme.TEXT_MUTED))
        for index in range(6):
            fraction = index / 5.0
            value = value_range[0] + fraction * (
                value_range[1] - value_range[0]
            )
            x_value = rect.left() + fraction * rect.width()
            painter.drawText(
                QRectF(x_value - 40.0, rect.bottom() + 3.0, 80.0, 18.0),
                Qt.AlignHCenter | Qt.AlignTop,
                cls._axis_value_text(value),
            )

    @classmethod
    def _draw_y_ticks(cls, painter, rect, value_range, side="left"):
        painter.setPen(QColor(theme.TEXT_MUTED))
        for index in range(5):
            fraction = index / 4.0
            value = value_range[1] - fraction * (
                value_range[1] - value_range[0]
            )
            y_value = rect.top() + fraction * rect.height()
            if side == "right":
                label_rect = QRectF(
                    rect.right() + 5.0,
                    y_value - 9.0,
                    60.0,
                    18.0,
                )
                alignment = Qt.AlignLeft | Qt.AlignVCenter
            else:
                label_rect = QRectF(
                    rect.left() - 82.0,
                    y_value - 9.0,
                    76.0,
                    18.0,
                )
                alignment = Qt.AlignRight | Qt.AlignVCenter
            painter.drawText(
                label_rect,
                alignment,
                cls._axis_value_text(value),
            )

    @staticmethod
    def _draw_vertical_marker(painter, x_value, x_range, rect, color, style):
        if x_value is None or not x_range[0] <= float(x_value) <= x_range[1]:
            return
        x_ratio = (float(x_value) - x_range[0]) / (x_range[1] - x_range[0])
        x_pixel = rect.left() + x_ratio * rect.width()
        painter.setPen(QPen(QColor(color), 1.5, style))
        painter.drawLine(
            int(x_pixel),
            int(rect.top()),
            int(x_pixel),
            int(rect.bottom()),
        )

    @staticmethod
    def _finite_row_value(row, field):
        try:
            value = float((row or {}).get(field))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _event_time(self, name):
        event = self.analysis.get(name) or {}
        return self._finite_row_value(event, "elapsed_s")

    def _return_time(self):
        for row in self.rows:
            if str(row.get("phase", "")) == RETURN_PHASE:
                return self._finite_row_value(row, "elapsed_s")
        return None

    def _peak_force_time(self):
        candidates = []
        for row in self.rows:
            force = self._finite_row_value(row, "force_delta_abs_g")
            elapsed = self._finite_row_value(row, "elapsed_s")
            if force is not None and elapsed is not None:
                candidates.append((force, elapsed))
        return max(candidates, default=(None, None))[1]

    @staticmethod
    def _format_event(name, payload):
        if not payload:
            return f"{name}: not detected"
        distance = payload.get("distance_mm")
        elapsed = payload.get("elapsed_s")
        if distance is None or elapsed is None:
            return f"{name}: detected"
        return f"{name}: {float(distance):.3f} mm at {float(elapsed):.2f} s"

    def _paint_process(self, painter):
        outer = QRectF(self.rect().adjusted(86, 68, -68, -48))
        if outer.width() <= 0 or outer.height() <= 0:
            return
        elapsed_values = [
            value
            for value in (
                self._finite_row_value(row, "elapsed_s") for row in self.rows
            )
            if value is not None
        ]
        if len(elapsed_values) < 2:
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(outer, Qt.AlignCenter, "Not enough samples")
            return

        base_x_range = self._range(
            elapsed_values,
            padding=0.0,
            nonnegative=True,
        )
        x_range = self._display_range(base_x_range, "x")
        self._last_plot_rect = outer
        self._last_x_range = x_range
        self._last_y_range = None
        signal_label = SIGNAL_LABELS.get(self.signal_field, self.signal_field)
        specs = (
            (
                "Travel distance (mm)",
                "travel_down_mm",
                self.DISTANCE_COLOR,
                True,
            ),
            (signal_label, self.signal_field, self.SIGNAL_COLOR, False),
            ("Force change (g)", "force_delta_abs_g", self.FORCE_COLOR, True),
        )
        gap = 28.0
        panel_height = (outer.height() - gap * 2.0) / 3.0
        signal_time = self._event_time("signal_onset")
        contact_time = self._event_time("contact_onset")
        peak_force_time = self._peak_force_time()
        return_time = self._return_time()

        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(12, 24, "Complete Calibration Sequence")
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.drawText(
            12,
            45,
            "Amber: proximity onset   Red: contact onset   "
            "Blue: peak force   Purple: return begins",
        )

        for index, (label, field, color, include_zero) in enumerate(specs):
            panel = QRectF(
                outer.left(),
                outer.top() + index * (panel_height + gap),
                outer.width(),
                panel_height,
            )
            points = self._finite_points(self.rows, "elapsed_s", field)
            if not points:
                continue
            y_range = self._range(
                [point[1] for point in points],
                include_zero=not include_zero,
                nonnegative=include_zero,
            )
            self._draw_grid(painter, panel)
            self._draw_series(
                painter,
                points,
                x_range,
                y_range,
                panel,
                color,
            )
            self._add_hover_context(
                panel,
                x_range,
                y_range,
                points,
                color,
                label,
                time_axis=True,
            )
            self._draw_vertical_marker(
                painter, signal_time, x_range, panel, self.FIT_COLOR, Qt.DotLine
            )
            self._draw_vertical_marker(
                painter, contact_time, x_range, panel, "#ef5350", Qt.DashLine
            )
            self._draw_vertical_marker(
                painter,
                peak_force_time,
                x_range,
                panel,
                self.FORCE_COLOR,
                Qt.DotLine,
            )
            self._draw_vertical_marker(
                painter, return_time, x_range, panel, self.RETURN_COLOR, Qt.DashLine
            )
            painter.setPen(QColor(color))
            painter.drawText(8, int(panel.top() + 14), label)
            self._draw_y_ticks(painter, panel, y_range, side="right")

        self._draw_x_ticks(painter, outer, x_range)
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.drawText(
            int(outer.center().x() - 24),
            self.height() - 7,
            "Time (s)",
        )

    def _paint_repeatability(self, painter):
        outer = QRectF(self.rect().adjusted(92, 70, -50, -64))
        if outer.width() <= 0 or outer.height() <= 0:
            return
        trial_rows = [item["rows"] for item in self.trial_results]
        signal_label = SIGNAL_LABELS.get(self.signal_field, self.signal_field)
        specifications = (
            (signal_label, self.signal_field, False),
            ("Force change (g)", "force_delta_abs_g", True),
        )
        statistics = [
            self._repeatability_statistics(trial_rows, field)
            for _label, field, _nonnegative in specifications
        ]
        all_curves = [
            curve
            for payload in statistics
            for curve in payload["curves"]
        ]
        if not all_curves:
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(
                outer,
                Qt.AlignCenter,
                "Repeatability requires recorded approach samples.",
            )
            return

        x_values = [point[0] for curve in all_curves for point in curve]
        base_x_range = self._range(
            x_values,
            include_zero=True,
            nonnegative=True,
        )
        x_range = self._display_range(base_x_range, "x")
        self._last_plot_rect = outer
        self._last_x_range = x_range
        self._last_y_range = None
        gap = 42.0
        panel_height = (outer.height() - gap) / 2.0

        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(12, 24, "Repeated-Trial Consistency")
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.drawText(
            12,
            46,
            f"{len(self.trial_results)} trials · coloured lines: trials · "
            "yellow: mean · shaded band: ±1 SD",
        )

        for panel_index, (specification, payload) in enumerate(
            zip(specifications, statistics)
        ):
            label, _field, nonnegative = specification
            panel = QRectF(
                outer.left(),
                outer.top() + panel_index * (panel_height + gap),
                outer.width(),
                panel_height,
            )
            values = [
                point[1]
                for curve in payload["curves"]
                for point in curve
            ]
            values.extend(point[1] for point in payload["lower"])
            values.extend(point[1] for point in payload["upper"])
            if not values:
                continue
            y_range = self._range(
                values,
                include_zero=True,
                nonnegative=nonnegative,
            )
            self._draw_grid(painter, panel)
            for curve_index, curve in enumerate(payload["curves"]):
                color = QColor(
                    self.TRIAL_COLORS[curve_index % len(self.TRIAL_COLORS)]
                )
                color.setAlpha(150)
                self._draw_series(
                    painter,
                    curve,
                    x_range,
                    y_range,
                    panel,
                    color,
                )
            self._draw_variation_band(
                painter,
                payload["lower"],
                payload["upper"],
                x_range,
                y_range,
                panel,
                self.MEAN_COLOR,
            )
            self._draw_series(
                painter,
                payload["mean"],
                x_range,
                y_range,
                panel,
                self.MEAN_COLOR,
            )
            self._add_hover_context(
                panel,
                x_range,
                y_range,
                payload["mean"],
                self.MEAN_COLOR,
                f"Mean {label}",
            )
            painter.setPen(QColor(theme.TEXT_PRIMARY))
            painter.drawText(8, int(panel.top() - 8), label)
            self._draw_y_ticks(painter, panel, y_range, side="left")

        bottom_panel = QRectF(
            outer.left(),
            outer.top() + panel_height + gap,
            outer.width(),
            panel_height,
        )
        self._draw_x_ticks(painter, bottom_panel, x_range)
        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(
            int(outer.center().x() - 55),
            self.height() - 14,
            "Travel distance (mm)",
        )

    def _relationship_spec(self):
        signal_label = SIGNAL_LABELS.get(self.signal_field, self.signal_field)
        return {
            "signal_distance": (
                "travel_down_mm",
                self.signal_field,
                "Travel distance (mm)",
                signal_label,
                "Sensor Signal vs Travel Distance",
                "signal_distance_fit",
            ),
            "force_distance": (
                "travel_down_mm",
                "force_delta_abs_g",
                "Travel distance (mm)",
                "Force change (g)",
                "Force vs Travel Distance",
                "force_distance_fit",
            ),
            "force_signal": (
                self.signal_field,
                "force_delta_abs_g",
                signal_label,
                "Force change (g)",
                "Force vs Sensor Signal",
                "force_signal_fit",
            ),
        }.get(self.mode)

    @staticmethod
    def _equation_text(fit):
        if not fit:
            return "Linear loading fit unavailable (not enough varying samples)"
        slope = float(fit.get("slope", 0.0))
        intercept = float(fit.get("intercept", 0.0))
        sign = "+" if intercept >= 0.0 else "−"
        return (
            f"Loading fit: y = {slope:.5g}x {sign} {abs(intercept):.5g}   "
            f"R² = {float(fit.get('r_squared', 0.0)):.4f}   "
            f"n = {int(fit.get('sample_count', 0))}"
        )

    def _paint_relationship(self, painter):
        spec = self._relationship_spec()
        plot_rect = QRectF(self.rect().adjusted(92, 86, -56, -90))
        if spec is None or plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return
        x_field, y_field, x_label, y_label, title, fit_name = spec
        approach_points = self._finite_points(
            self.rows,
            x_field,
            y_field,
            phases=APPROACH_PHASES,
        )
        return_points = self._finite_points(
            self.rows,
            x_field,
            y_field,
            phases={RETURN_PHASE},
        )
        points = approach_points + return_points
        if len(points) < 2:
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(plot_rect, Qt.AlignCenter, "Not enough samples")
            return

        include_x_zero = x_field in {"travel_down_mm", "force_delta_abs_g"}
        include_y_zero = y_field in {"force_delta_abs_g", self.signal_field}
        base_x_range = self._range(
            [point[0] for point in points],
            include_zero=not include_x_zero and x_field == self.signal_field,
            nonnegative=include_x_zero,
        )
        base_y_range = self._range(
            [point[1] for point in points],
            include_zero=include_y_zero and y_field == self.signal_field,
            nonnegative=y_field == "force_delta_abs_g",
        )
        x_range = self._display_range(base_x_range, "x")
        y_range = self._display_range(base_y_range, "y")
        self._last_plot_rect = plot_rect
        self._last_x_range = x_range
        self._last_y_range = y_range
        self._draw_grid(painter, plot_rect)
        self._draw_series(
            painter,
            approach_points,
            x_range,
            y_range,
            plot_rect,
            self.SIGNAL_COLOR,
        )
        self._draw_points(
            painter,
            approach_points,
            x_range,
            y_range,
            plot_rect,
            self.SIGNAL_COLOR,
        )
        self._draw_series(
            painter,
            return_points,
            x_range,
            y_range,
            plot_rect,
            self.RETURN_COLOR,
        )
        self._add_hover_context(
            plot_rect,
            x_range,
            y_range,
            approach_points,
            self.SIGNAL_COLOR,
            "Loading",
        )
        self._add_hover_context(
            plot_rect,
            x_range,
            y_range,
            return_points,
            self.RETURN_COLOR,
            "Return",
        )
        self._draw_points(
            painter,
            return_points,
            x_range,
            y_range,
            plot_rect,
            self.RETURN_COLOR,
        )

        fit = self.analysis.get(fit_name)
        if fit:
            fit_points = [
                (
                    float(fit["x_min"]),
                    float(fit["slope"]) * float(fit["x_min"])
                    + float(fit["intercept"]),
                ),
                (
                    float(fit["x_max"]),
                    float(fit["slope"]) * float(fit["x_max"])
                    + float(fit["intercept"]),
                ),
            ]
            painter.save()
            painter.setClipRect(plot_rect)
            painter.setPen(QPen(QColor(self.FIT_COLOR), 2, Qt.DashLine))
            path = QPainterPath()
            path.moveTo(
                *self._map_point(
                    fit_points[0][0],
                    fit_points[0][1],
                    x_range,
                    y_range,
                    plot_rect,
                )
            )
            path.lineTo(
                *self._map_point(
                    fit_points[1][0],
                    fit_points[1][1],
                    x_range,
                    y_range,
                    plot_rect,
                )
            )
            painter.drawPath(path)
            painter.restore()

        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(12, 24, title)
        painter.setPen(QColor(self.FIT_COLOR if fit else theme.TEXT_MUTED))
        painter.drawText(12, 47, self._equation_text(fit))
        painter.setPen(QColor(self.SIGNAL_COLOR))
        painter.drawText(
            int(plot_rect.left()),
            int(plot_rect.bottom() + 46),
            "Approach/loading",
        )
        painter.setPen(QColor(self.RETURN_COLOR))
        painter.drawText(
            int(plot_rect.left() + 170),
            int(plot_rect.bottom() + 46),
            "Return/unloading",
        )
        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(8, int(plot_rect.top() - 10), y_label)
        painter.drawText(
            int(plot_rect.center().x() - 55),
            self.height() - 14,
            x_label,
        )
        self._draw_x_ticks(painter, plot_rect, x_range)
        self._draw_y_ticks(painter, plot_rect, y_range, side="left")

    def resizeEvent(self, event):
        self._base_image = None
        super().resizeEvent(event)

    def paintEvent(self, _event):
        if (
            self._base_image is None
            or self._base_image.size() != self.size()
        ):
            self._last_plot_rect = None
            self._last_x_range = None
            self._last_y_range = None
            self._hover_contexts = []
            image = QImage(self.size(), QImage.Format_ARGB32)
            image.fill(QColor(theme.INPUT_BG))
            base_painter = QPainter(image)
            base_painter.setRenderHint(QPainter.Antialiasing)
            if self.mode == "process":
                self._paint_process(base_painter)
            elif self.mode == "repeatability":
                self._paint_repeatability(base_painter)
            else:
                self._paint_relationship(base_painter)
            base_painter.end()
            self._base_image = image

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawImage(0, 0, self._base_image)
        self._paint_hover(painter)


class CalibrationGraphDetailDialog(QDialog):
    """Large interactive view for one calibration graph."""

    def __init__(
        self,
        title,
        rows,
        signal_field,
        mode,
        parent=None,
        trial_results=None,
    ):
        super().__init__(parent, Qt.Window)
        self.graph_title = str(title)
        self.setWindowTitle(f"Calibration Detail — {self.graph_title}")
        self.setModal(False)
        components.size_to_screen(self, 1280, 820)
        self.setStyleSheet(
            f"QDialog {{ background: {theme.WINDOW_BG}; "
            f"color: {theme.TEXT_PRIMARY}; }}"
        )

        layout = QVBoxLayout(self)
        instruction = QLabel(
            "Move the mouse to inspect x/y values · Mouse wheel to zoom · "
            "Left-drag to pan"
        )
        instruction.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        layout.addWidget(instruction)

        self.chart = CalibrationCurveWidget(rows, signal_field, mode=mode)
        if mode == "repeatability":
            self.chart.set_trials(trial_results or [], signal_field)
        self.chart.setMinimumHeight(620)
        layout.addWidget(self.chart, 1)

        controls = QHBoxLayout()
        zoom_in_button = QPushButton("Zoom In")
        zoom_out_button = QPushButton("Zoom Out")
        reset_button = QPushButton("Reset View")
        save_button = QPushButton("Save This Graph")
        close_button = QPushButton("Close")
        zoom_in_button.clicked.connect(lambda: self.chart.zoom_view(0.75))
        zoom_out_button.clicked.connect(lambda: self.chart.zoom_view(1.0 / 0.75))
        reset_button.clicked.connect(self.chart.reset_view)
        save_button.clicked.connect(self._save_graph)
        close_button.clicked.connect(self.close)
        controls.addWidget(zoom_in_button)
        controls.addWidget(zoom_out_button)
        controls.addWidget(reset_button)
        controls.addStretch(1)
        controls.addWidget(save_button)
        controls.addWidget(close_button)
        layout.addLayout(controls)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        layout.addWidget(self.status_label)

    def _save_graph(self):
        default_name = (
            self.graph_title.lower().replace(" ", "_").replace("/", "_")
            + ".png"
        )
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save calibration graph",
            default_name,
            "PNG image (*.png)",
        )
        if not path:
            return
        if not str(path).lower().endswith(".png"):
            path += ".png"
        if self.chart.save_png(path):
            self.status_label.setText(f"Saved: {path}")
        else:
            self.status_label.setText("The graph could not be saved.")


class CalibrationResultDialog(QDialog):
    GRAPH_TITLES = {
        "process": "Full Calibration Sequence",
        "signal_distance": "Sensor Signal vs Travel Distance",
        "force_distance": "Force vs Travel Distance",
        "force_signal": "Force vs Sensor Signal",
        "repeatability": "Repeated-Trial Mean and Variation",
    }

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = dict(result or {})
        trial_results = list(self.result.get("trial_results") or [])
        self.trial_results = trial_results or [self.result]
        self.exported_graph_paths: list[str] = []
        self._detail_dialog = None
        self.selected_taxel_index = int(self.result.get("taxel_index", 0) or 0)
        self.setWindowTitle("Calibration Force-Signal-Distance Result")
        components.size_to_screen(self, 1360, 820)
        self.setStyleSheet(
            f"QDialog {{ background: {theme.WINDOW_BG}; color: {theme.TEXT_PRIMARY}; }}"
        )
        layout = QVBoxLayout(self)
        requested_trials = int(self.result.get("requested_trials", 1) or 1)
        title = QLabel(
            "Repeated Force-Signal-Distance Characterization"
            if requested_trials > 1
            else "Force-Signal-Distance Characterization"
        )
        title.setStyleSheet(
            f"font-size: 18px; font-weight: 600; color: {theme.TEXT_PRIMARY};"
        )
        layout.addWidget(title)
        if requested_trials > 1:
            batch_summary = QLabel(
                f"Batch: {self.result.get('batch_status', 'unknown')}   |   "
                f"Recorded: {int(self.result.get('completed_trials', len(self.trial_results)))}/"
                f"{requested_trials}\n"
                f"Summary CSV: {self.result.get('batch_summary_path', 'not saved')}"
            )
            batch_summary.setWordWrap(True)
            batch_summary.setStyleSheet(f"color: {theme.TEXT_MUTED};")
            layout.addWidget(batch_summary)

        if len(self.trial_results) > 1:
            selector_layout = QHBoxLayout()
            selector_label = QLabel("Displayed trial")
            selector_label.setStyleSheet(f"color: {theme.TEXT_PRIMARY};")
            selector_layout.addWidget(selector_label)
            self.trial_selector = QComboBox(self)
            for result_item in self.trial_results:
                number = int(result_item.get("trial_number", 0) or 0)
                reason = str(result_item.get("stop_reason", "unknown"))
                self.trial_selector.addItem(
                    f"Trial {number:04d} | {reason}"
                )
            selector_layout.addWidget(self.trial_selector, 1)
            layout.addLayout(selector_layout)
        else:
            self.trial_selector = None

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        layout.addWidget(self.summary_label)

        self.tabs = QTabWidget(self)
        self.process_chart = CalibrationCurveWidget(
            [], "diff_ave", mode="process"
        )
        self.signal_distance_chart = CalibrationCurveWidget(
            [], "diff_ave", mode="signal_distance"
        )
        self.force_distance_chart = CalibrationCurveWidget(
            [], "diff_ave", mode="force_distance"
        )
        self.force_signal_chart = CalibrationCurveWidget(
            [], "diff_ave", mode="force_signal"
        )
        self.repeatability_chart = CalibrationCurveWidget(
            [], "diff_ave", mode="repeatability"
        )
        # Preserve the old attribute names for any external integrations.
        self.time_chart = self.process_chart
        self.relationship_chart = self.force_signal_chart
        self.tabs.addTab(self.process_chart, "Full Sequence")
        self.tabs.addTab(self.signal_distance_chart, "Signal vs Distance")
        self.tabs.addTab(self.force_distance_chart, "Force vs Distance")
        self.tabs.addTab(self.force_signal_chart, "Force vs Signal")
        repeatability_index = self.tabs.addTab(
            self.repeatability_chart,
            "Repeatability",
        )
        self.tabs.setTabEnabled(
            repeatability_index,
            len(self.trial_results) > 1,
        )
        self.repeatability_tab_index = repeatability_index

        sensor_shape = self.result.get("sensor_shape") or [8, 10]
        try:
            sensor_rows, sensor_columns = map(int, sensor_shape[:2])
        except (TypeError, ValueError):
            sensor_rows, sensor_columns = 8, 10
        taxel_panel = QWidget(self)
        taxel_layout = QVBoxLayout(taxel_panel)
        taxel_layout.setContentsMargins(0, 0, 8, 0)
        taxel_title = QLabel("Recorded taxels")
        taxel_title.setStyleSheet(
            f"font-weight: 600; color: {theme.TEXT_PRIMARY};"
        )
        taxel_hint = QLabel(
            "Click a node to inspect its signal.\n"
            "Indices are top-down, column-major."
        )
        taxel_hint.setWordWrap(True)
        taxel_hint.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self.taxel_context_label = QLabel("")
        self.taxel_context_label.setWordWrap(True)
        self.taxel_context_label.setStyleSheet(f"color: {theme.ACCENT};")
        self.taxel_map = CalibrationTaxelMap(
            sensor_rows,
            sensor_columns,
            taxel_panel,
        )
        taxel_layout.addWidget(taxel_title)
        taxel_layout.addWidget(taxel_hint)
        taxel_layout.addWidget(self.taxel_context_label)
        taxel_layout.addWidget(self.taxel_map)
        taxel_layout.addStretch()
        content_layout = QHBoxLayout()
        content_layout.addWidget(taxel_panel)
        content_layout.addWidget(self.tabs, 1)
        layout.addLayout(content_layout, 1)

        for chart in (
            self.process_chart,
            self.signal_distance_chart,
            self.force_distance_chart,
            self.force_signal_chart,
            self.repeatability_chart,
        ):
            chart.detailRequested.connect(
                lambda chart=chart: self._open_chart_detail(chart)
            )

        detail_actions = QHBoxLayout()
        detail_hint = QLabel(
            "Move the mouse for a value cursor, use the wheel to zoom, drag "
            "to pan, or open the graph in a larger window."
        )
        detail_hint.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self.open_detail_button = QPushButton("Open Selected Graph")
        self.open_detail_button.clicked.connect(
            self._open_selected_chart_detail
        )
        detail_actions.addWidget(detail_hint, 1)
        detail_actions.addWidget(self.open_detail_button)
        layout.addLayout(detail_actions)

        self.export_status_label = QLabel()
        self.export_status_label.setWordWrap(True)
        self.export_status_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        layout.addWidget(self.export_status_label)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

        if self.trial_selector is not None:
            self.trial_selector.currentIndexChanged.connect(self._select_trial)
        self.taxel_map.taxelSelected.connect(self._select_taxel)
        batch_signal_field = str(
            (self.trial_results[0] if self.trial_results else {}).get(
                "signal_field",
                "diff_ave",
            )
            or "diff_ave"
        )
        self.repeatability_chart.set_trials(
            self.trial_results,
            batch_signal_field,
        )
        self._select_trial(0)
        if self.result.get("graph_directory"):
            self.export_status_label.setText("Preparing trial graph files...")
            QTimer.singleShot(0, self.export_all_graphs)

    @staticmethod
    def _finite_float(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)
        return value if math.isfinite(value) else float(default)

    def _select_trial(self, index):
        if not self.trial_results:
            return
        index = max(0, min(int(index), len(self.trial_results) - 1))
        result = self.trial_results[index]
        source_rows = list(result.get("rows") or [])
        signal_field = str(result.get("signal_field") or "diff_ave")
        original_taxel = int(result.get("taxel_index", 0) or 0)
        available = available_taxel_indices(
            source_rows,
            signal_field,
            legacy_index=original_taxel,
        )
        if self.selected_taxel_index not in available and available:
            self.selected_taxel_index = (
                original_taxel
                if original_taxel in available
                else int(available[0])
            )
        self.taxel_map.set_taxels(available, self.selected_taxel_index)
        rows = calibration_rows_for_taxel(
            source_rows,
            signal_field,
            self.selected_taxel_index,
            legacy_index=original_taxel,
        )
        duration = max(
            (self._finite_float(row.get("elapsed_s")) for row in rows),
            default=0.0,
        )
        peak_force = max(
            (self._finite_float(row.get("force_delta_abs_g")) for row in rows),
            default=0.0,
        )
        max_distance = max(
            (self._finite_float(row.get("travel_down_mm")) for row in rows),
            default=0.0,
        )
        analysis = analyze_calibration_rows(rows, signal_field)
        signal_onset = analysis.get("signal_onset")
        contact_onset = analysis.get("contact_onset")
        self.summary_label.setText(
            f"Trial {int(result.get('trial_number', index + 1))}   |   "
            f"Viewed taxel: {self.selected_taxel_index}   |   "
            f"Stop reason: {result.get('stop_reason', 'unknown')}   |   "
            f"Samples: {len(rows)}   |   Duration: {duration:.2f} s   |   "
            f"Peak force: {peak_force:.1f} g   |   "
            f"Maximum travel: {max_distance:.3f} mm\n"
            f"{CalibrationCurveWidget._format_event('Proximity onset', signal_onset)}   |   "
            f"{CalibrationCurveWidget._format_event('Contact onset', contact_onset)}\n"
            f"CSV: {result.get('csv_path', 'not saved')}"
        )
        for chart in (
            self.process_chart,
            self.signal_distance_chart,
            self.force_distance_chart,
            self.force_signal_chart,
        ):
            chart.set_data(rows, signal_field)
        projected_trials = []
        for trial_index, trial in enumerate(self.trial_results, start=1):
            trial_signal = str(trial.get("signal_field") or signal_field)
            trial_original = int(trial.get("taxel_index", 0) or 0)
            trial_rows = calibration_rows_for_taxel(
                trial.get("rows") or [],
                trial_signal,
                self.selected_taxel_index,
                legacy_index=trial_original,
            )
            if trial_rows:
                projected_trials.append(
                    {
                        "trial_number": int(
                            trial.get("trial_number", trial_index)
                        ),
                        "rows": trial_rows,
                    }
                )
        self.repeatability_chart.set_trials(projected_trials, signal_field)
        self.tabs.setTabEnabled(
            self.repeatability_tab_index,
            len(projected_trials) > 1,
        )
        self.taxel_context_label.setText(
            f"Viewing taxel {self.selected_taxel_index}\n"
            f"Physical target: taxel {original_taxel}"
        )

    def _select_taxel(self, taxel_index):
        self.selected_taxel_index = int(taxel_index)
        current_trial = (
            self.trial_selector.currentIndex()
            if self.trial_selector is not None
            else 0
        )
        self._select_trial(current_trial)

    def _open_selected_chart_detail(self):
        chart = self.tabs.currentWidget()
        if isinstance(chart, CalibrationCurveWidget):
            self._open_chart_detail(chart)

    def _open_chart_detail(self, chart):
        if not isinstance(chart, CalibrationCurveWidget):
            return
        existing = self._detail_dialog
        if existing is not None:
            existing.close()
        title = self.GRAPH_TITLES.get(chart.mode, "Calibration Graph")
        dialog = CalibrationGraphDetailDialog(
            f"{title} — Taxel {self.selected_taxel_index}",
            chart.rows,
            chart.signal_field,
            chart.mode,
            parent=self,
            trial_results=chart.trial_results,
        )
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(
            lambda _object=None, dialog=dialog: self._detail_dialog_closed(
                dialog
            )
        )
        self._detail_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _detail_dialog_closed(self, dialog):
        if self._detail_dialog is dialog:
            self._detail_dialog = None

    def closeEvent(self, event):
        detail = self._detail_dialog
        self._detail_dialog = None
        if detail is not None:
            detail.close()
        super().closeEvent(event)

    @staticmethod
    def _render_curve(rows, signal_field, mode, width=880, height=600):
        chart = CalibrationCurveWidget(rows, signal_field, mode=mode)
        chart.resize(int(width), int(height))
        chart.ensurePolished()
        image = QImage(int(width), int(height), QImage.Format_ARGB32)
        image.fill(QColor(theme.INPUT_BG))
        painter = QPainter(image)
        chart.render(painter)
        painter.end()
        chart.deleteLater()
        return image

    @staticmethod
    def _render_repeatability(
        trial_results,
        signal_field,
        width=1120,
        height=760,
    ):
        chart = CalibrationCurveWidget(
            [],
            signal_field,
            mode="repeatability",
        )
        chart.set_trials(trial_results, signal_field)
        chart.resize(int(width), int(height))
        chart.ensurePolished()
        image = QImage(int(width), int(height), QImage.Format_ARGB32)
        image.fill(QColor(theme.INPUT_BG))
        painter = QPainter(image)
        chart.render(painter)
        painter.end()
        chart.deleteLater()
        return image

    @classmethod
    def _export_trial_graph(cls, result, path: Path) -> bool:
        rows = list((result or {}).get("rows") or [])
        if not rows:
            return False
        signal_field = str(
            (result or {}).get("signal_field") or "diff_ave"
        )
        modes = (
            "process",
            "signal_distance",
            "force_distance",
            "force_signal",
        )
        images = [
            cls._render_curve(
                rows,
                signal_field,
                mode,
                width=760,
                height=500,
            )
            for mode in modes
        ]
        margin = 18
        header_height = 44
        canvas = QImage(
            images[0].width() * 2 + margin * 3,
            images[0].height() * 2 + header_height + margin * 2,
            QImage.Format_ARGB32,
        )
        canvas.fill(QColor(theme.WINDOW_BG))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(
            margin,
            28,
            f"Calibration Trial "
            f"{int((result or {}).get('trial_number', 0)):04d} · Taxel "
            f"{int((result or {}).get('taxel_index', 0))}",
        )
        for image_index, chart_image in enumerate(images):
            row = image_index // 2
            column = image_index % 2
            painter.drawImage(
                margin + column * (chart_image.width() + margin),
                header_height + row * (chart_image.height() + margin),
                chart_image,
            )
        painter.end()
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(canvas.save(str(path), "PNG"))

    def export_all_graphs(self):
        graph_directory = str(self.result.get("graph_directory") or "")
        if not graph_directory:
            return
        output_directory = Path(graph_directory).expanduser()
        exported = []
        failures = 0
        for index, result in enumerate(self.trial_results, start=1):
            trial_number = int(result.get("trial_number", index) or index)
            graph_path = output_directory / f"trial_{trial_number:04d}.png"
            try:
                saved = self._export_trial_graph(result, graph_path)
            except Exception:
                saved = False
            if saved:
                exported.append(str(graph_path))
                result["graph_path"] = str(graph_path)
            else:
                failures += 1
        repeatability_saved = False
        if len(self.trial_results) > 1:
            signal_field = str(
                (self.trial_results[0] or {}).get("signal_field")
                or "diff_ave"
            )
            repeatability_path = output_directory / "repeatability.png"
            try:
                image = self._render_repeatability(
                    self.trial_results,
                    signal_field,
                )
                repeatability_path.parent.mkdir(parents=True, exist_ok=True)
                repeatability_saved = bool(
                    image.save(str(repeatability_path), "PNG")
                )
            except Exception:
                repeatability_saved = False
            if repeatability_saved:
                exported.append(str(repeatability_path))
                self.result["repeatability_graph_path"] = str(
                    repeatability_path
                )
            else:
                failures += 1
        self.exported_graph_paths = exported
        if exported:
            trial_graph_count = len(exported) - int(repeatability_saved)
            message = f"Saved {trial_graph_count} trial graph(s)"
            if repeatability_saved:
                message += " and the repeatability graph"
            message += f": {output_directory}"
            if failures:
                message += f" ({failures} failed)"
        else:
            message = "Trial graphs could not be exported."
        self.export_status_label.setText(message)
        parent = self.parent()
        callback = getattr(parent, "_append_sidebar_control_message", None)
        if callable(callback):
            callback(message)

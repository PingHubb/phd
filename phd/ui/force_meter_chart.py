from __future__ import annotations

import math
import time
from collections import deque

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QWidget

from phd.ui import theme


class ForceMeterChartWidget(QWidget):
    """Lightweight rolling chart for force-meter samples."""

    MAX_HISTORY_SECONDS = 300.0

    def __init__(self, parent=None, time_window_seconds: float = 30.0):
        super().__init__(parent)
        self._samples = deque()
        self._tare_newtons = 0.0
        self._time_window_seconds = max(1.0, float(time_window_seconds))
        self.setMinimumHeight(240)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def time_window_seconds(self) -> float:
        return self._time_window_seconds

    def append_sample(self, force_newtons: float, timestamp: float | None = None):
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        force_newtons = float(force_newtons)
        if not (math.isfinite(timestamp) and math.isfinite(force_newtons)):
            return

        self._samples.append((timestamp, force_newtons))
        cutoff = timestamp - self.MAX_HISTORY_SECONDS
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        self.update()

    def set_tare(self, tare_newtons: float):
        tare_newtons = float(tare_newtons)
        self._tare_newtons = tare_newtons if math.isfinite(tare_newtons) else 0.0
        self.update()

    def set_time_window(self, seconds: float):
        self._time_window_seconds = max(1.0, float(seconds))
        self.update()

    def clear(self):
        self._samples.clear()
        self.update()

    @staticmethod
    def _nice_step(span: float, interval_count: int = 4) -> float:
        raw_step = max(float(span) / max(1, interval_count), 1e-12)
        magnitude = 10.0 ** math.floor(math.log10(raw_step))
        normalized = raw_step / magnitude
        for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
            if normalized <= candidate:
                return candidate * magnitude
        return 10.0 * magnitude

    @classmethod
    def _y_axis_bounds(cls, values):
        data_min = min(min(values), 0.0)
        data_max = max(max(values), 0.0)
        span = data_max - data_min
        if span < 1e-9:
            padding = max(1.0, abs(data_max) * 0.1)
        else:
            padding = max(0.05, span * 0.12)
        lower = data_min - padding
        upper = data_max + padding
        step = cls._nice_step(upper - lower)
        lower = math.floor(lower / step) * step
        upper = math.ceil(upper / step) * step
        if upper <= lower:
            upper = lower + step
        return lower, upper, step

    @staticmethod
    def _format_tick(value: float) -> str:
        absolute = abs(value)
        if absolute >= 1000.0:
            return f"{value:.0f}"
        if absolute >= 10.0:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.INPUT_BG))

        plot_rect = self.rect().adjusted(72, 34, -22, -48)
        if plot_rect.width() <= 20 or plot_rect.height() <= 20:
            return

        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(plot_rect.left(), 22, "Live Force Signal")

        axis_pen = QPen(QColor(theme.BORDER))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)
        painter.drawRect(plot_rect)

        if not self._samples:
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(plot_rect, Qt.AlignCenter, "Waiting for force samples")
            self._draw_axis_titles(painter, plot_rect)
            return

        latest_time = self._samples[-1][0]
        earliest_time = latest_time - self._time_window_seconds
        visible = [
            (timestamp, raw_force - self._tare_newtons)
            for timestamp, raw_force in self._samples
            if timestamp >= earliest_time
        ]
        if not visible:
            return

        values = [force for _timestamp, force in visible]
        y_min, y_max, y_step = self._y_axis_bounds(values)
        y_span = y_max - y_min

        grid_pen = QPen(QColor(theme.BORDER_SUBTLE))
        grid_pen.setStyle(Qt.DashLine)
        painter.setPen(grid_pen)

        x_interval_count = 5
        for index in range(x_interval_count + 1):
            ratio = index / float(x_interval_count)
            x = plot_rect.left() + ratio * plot_rect.width()
            painter.drawLine(int(x), plot_rect.top(), int(x), plot_rect.bottom())
            relative_seconds = -self._time_window_seconds * (1.0 - ratio)
            label = "0" if index == x_interval_count else f"{relative_seconds:.0f}"
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(
                QRectF(x - 28, plot_rect.bottom() + 6, 56, 18),
                Qt.AlignHCenter | Qt.AlignTop,
                label,
            )
            painter.setPen(grid_pen)

        tick_count = int(round((y_max - y_min) / y_step))
        for index in range(tick_count + 1):
            value = y_min + index * y_step
            ratio = (value - y_min) / y_span
            y = plot_rect.bottom() - ratio * plot_rect.height()
            painter.drawLine(plot_rect.left(), int(y), plot_rect.right(), int(y))
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(
                QRectF(4, y - 9, plot_rect.left() - 10, 18),
                Qt.AlignRight | Qt.AlignVCenter,
                self._format_tick(value),
            )
            painter.setPen(grid_pen)

        zero_ratio = (0.0 - y_min) / y_span
        zero_y = plot_rect.bottom() - zero_ratio * plot_rect.height()
        zero_pen = QPen(QColor("#6F7C91"), 1)
        zero_pen.setStyle(Qt.DashLine)
        painter.setPen(zero_pen)
        painter.drawLine(plot_rect.left(), int(zero_y), plot_rect.right(), int(zero_y))

        path = QPainterPath()
        points = []
        for timestamp, force in visible:
            x_ratio = (timestamp - earliest_time) / self._time_window_seconds
            y_ratio = (force - y_min) / y_span
            x = plot_rect.left() + x_ratio * plot_rect.width()
            y = plot_rect.bottom() - y_ratio * plot_rect.height()
            points.append((x, y))

        if points:
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)

            signal_pen = QPen(QColor(theme.INFO), 2)
            painter.setPen(signal_pen)
            painter.drawPath(path)

            latest_x, latest_y = points[-1]
            painter.setPen(QPen(QColor(theme.TEXT_PRIMARY), 1))
            painter.setBrush(QColor(theme.SUCCESS))
            painter.drawEllipse(QRectF(latest_x - 4, latest_y - 4, 8, 8))

        painter.setBrush(Qt.NoBrush)
        self._draw_axis_titles(painter, plot_rect)

    def _draw_axis_titles(self, painter: QPainter, plot_rect):
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.drawText(
            QRectF(plot_rect.left(), self.height() - 22, plot_rect.width(), 18),
            Qt.AlignCenter,
            "Time from latest sample (s)",
        )
        painter.save()
        painter.translate(15, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRectF(-plot_rect.height() / 2, -9, plot_rect.height(), 18),
            Qt.AlignCenter,
            "Force (N)",
        )
        painter.restore()

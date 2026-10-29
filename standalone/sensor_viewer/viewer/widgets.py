"""Small, dependency-light Qt heatmap and live trace widgets."""

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QWidget, QToolTip

from .core import Grid

BG = "#101827"
SURFACE = "#172338"
TEXT = "#EDF4FF"
MUTED = "#A6B6CB"
LINE = "#30415A"
ACCENT = "#5EEAD4"
NEGATIVE = "#63C5FF"
POSITIVE = "#FFB76B"


class Heatmap(QWidget):
    taxel_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = Grid(5, 5)
        self.values = None
        self.selected = 12
        self.mode = "difference"
        self.scale = 100.0
        self.show_values = True
        self.setMinimumSize(280, 220)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Sensor taxel map; use arrow keys to select a taxel")

    def grid_rect(self):
        cell = min((self.width() - 32) / self.grid.columns,
                   (self.height() - 36) / self.grid.rows)
        return QRectF((self.width() - cell*self.grid.columns)/2, 10,
                      cell*self.grid.columns, cell*self.grid.rows), cell

    def select(self, index):
        index = min(max(0, int(index)), self.grid.size - 1)
        if index != self.selected:
            self.selected = index
            self.taxel_selected.emit(index)
        self.update()

    def index_at(self, point):
        area, cell = self.grid_rect()
        if not area.contains(point):
            return None
        column = min(self.grid.columns - 1, int((point.x() - area.x()) / cell))
        row = min(self.grid.rows - 1, int((point.y() - area.y()) / cell))
        return column * self.grid.rows + row

    def mousePressEvent(self, event):
        index = self.index_at(event.localPos())
        if index is not None and event.button() == Qt.LeftButton:
            self.setFocus()
            self.select(index)

    def mouseMoveEvent(self, event):
        index = self.index_at(event.localPos())
        if index is not None:
            value = "No data" if self.values is None else f"{self.values[index]:+.2f}"
            QToolTip.showText(event.globalPos(), f"Taxel {index} · row {index % self.grid.rows}, column {index // self.grid.rows}\n{value}", self)

    def keyPressEvent(self, event):
        row, column = self.selected % self.grid.rows, self.selected // self.grid.rows
        moves = {Qt.Key_Left: (row, column-1), Qt.Key_Right: (row, column+1),
                 Qt.Key_Up: (row-1, column), Qt.Key_Down: (row+1, column)}
        if event.key() in moves:
            row, column = moves[event.key()]
            if 0 <= row < self.grid.rows and 0 <= column < self.grid.columns:
                self.select(column*self.grid.rows + row)
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        area, cell = self.grid_rect()
        base = np.array([28, 43, 64], dtype=float)
        values = self.values
        sequential = self.mode in {"raw", "baseline"}
        low, high = 0., 1.
        if values is not None and sequential:
            low, high = float(np.nanmin(values)), float(np.nanmax(values))
        for i in range(self.grid.size):
            rect = QRectF(area.x() + (i//self.grid.rows)*cell + 1,
                          area.y() + (i%self.grid.rows)*cell + 1, cell-2, cell-2)
            value = values[i] if values is not None else np.nan
            amount = 0.
            target = np.array([39, 143, 153])
            if np.isfinite(value):
                if sequential:
                    amount = (value-low) / max(high-low, 1)
                else:
                    amount = min(abs(value) / self.scale, 1)
                    target = np.array([27, 104, 154] if value < 0 else [145, 77, 33])
            colour = base*(1-amount) + target*amount
            painter.setBrush(QColor(*[int(v) for v in colour]))
            painter.setPen(QPen(QColor(ACCENT if i == self.selected else LINE), 2.5 if i == self.selected else .7))
            painter.drawRoundedRect(rect, min(6, cell/8), min(6, cell/8))
            painter.setPen(QColor(TEXT))
            if cell >= 25:
                font = QFont(self.font())
                font.setPointSizeF(max(7, min(10, cell/5)))
                painter.setFont(font)
                painter.drawText(rect.adjusted(2, 1, -2, -1), Qt.AlignTop | Qt.AlignLeft, str(i))
                if self.show_values and cell >= 42 and np.isfinite(value):
                    font.setBold(True)
                    font.setPointSizeF(min(13, cell/5))
                    painter.setFont(font)
                    painter.drawText(rect, Qt.AlignCenter, f"{value:.0f}" if sequential else f"{value:+.0f}")
        painter.setFont(self.font())
        painter.setPen(QColor(MUTED))
        caption = "Auto range · sensor units" if sequential else f"Blue −  /  orange +   ·   colour range ±{self.scale:g}"
        painter.drawText(QRectF(4, self.height()-24, self.width()-8, 22), Qt.AlignCenter, caption)


class TraceChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.times = np.array([])
        self.values = np.array([])
        self.seconds = 30.0
        self.label = "Signed difference (sensor units)"
        self.hover = None
        self.setMouseTracking(True)
        self.setMinimumSize(320, 220)
        self.setAccessibleName("Selected taxel signal graph; hover to read values, wheel to change time range")

    def set_series(self, times, values, label):
        self.times = np.asarray(times, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.label = label
        self.update()

    def view_data(self):
        if not len(self.times):
            return self.times, self.values, (0., self.seconds), (-1., 1.)
        right = max(float(self.times[-1]), self.seconds)
        left = max(0., right-self.seconds)
        keep = (self.times >= left) & np.isfinite(self.values)
        times, values = self.times[keep], self.values[keep]
        if len(values):
            low, high = float(values.min()), float(values.max())
            if "difference" in self.label.lower():
                low, high = min(low, 0), max(high, 0)
            margin = max((high-low)*.12, 1.)
            bounds = (low-margin, high+margin)
        else:
            bounds = (-1., 1.)
        return times, values, (left, right), bounds

    def mouseMoveEvent(self, event):
        self.hover = event.localPos()
        self.update()

    def leaveEvent(self, event):
        self.hover = None
        self.update()

    def wheelEvent(self, event):
        self.seconds = float(np.clip(self.seconds * (.8 if event.angleDelta().y() > 0 else 1.25), 2, 120))
        self.update()
        event.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        area = QRectF(65, 32, max(20, self.width()-84), max(20, self.height()-70))
        p.fillRect(area, QColor(BG))
        times, values, (left, right), (low, high) = self.view_data()
        x = lambda t: area.left() + (t-left) / max(right-left, 1e-9)*area.width()
        y = lambda v: area.bottom() - (v-low) / max(high-low, 1e-9)*area.height()
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(65, 2, area.width(), 24), Qt.AlignLeft | Qt.AlignVCenter, self.label)
        for step in range(5):
            value = low + step*(high-low)/4
            t = left + step*(right-left)/4
            p.setPen(QPen(QColor(LINE), .7))
            p.drawLine(QPointF(area.left(), y(value)), QPointF(area.right(), y(value)))
            p.drawLine(QPointF(x(t), area.top()), QPointF(x(t), area.bottom()))
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(0, y(value)-10, 57, 20), Qt.AlignRight | Qt.AlignVCenter, f"{value:.1f}")
            p.drawText(QRectF(x(t)-30, area.bottom()+7, 60, 20), Qt.AlignCenter, f"{t:.1f}s")
        if not len(times):
            p.drawText(area, Qt.AlignCenter, "Connect a sensor or start Demo to see a live signal")
            return
        p.save()
        p.setClipRect(area)
        if low <= 0 <= high:
            p.setPen(QPen(QColor(MUTED), .8, Qt.DashLine))
            p.drawLine(QPointF(area.left(), y(0)), QPointF(area.right(), y(0)))
        path = QPainterPath()
        last_time = None
        for t, value in zip(times, values):
            point = QPointF(x(t), y(value))
            if last_time is None or t-last_time > .75:
                path.moveTo(point)  # Do not bridge a missing-data gap.
            else:
                path.lineTo(point)
            last_time = t
        p.setPen(QPen(QColor(ACCENT), 1.8))
        p.drawPath(path)
        p.restore()
        if self.hover is not None and area.contains(self.hover):
            t = left + (self.hover.x()-area.left())/area.width()*(right-left)
            index = int(np.argmin(np.abs(times-t)))
            px, py = x(times[index]), y(values[index])
            p.setPen(QPen(QColor(TEXT), 1, Qt.DashLine))
            p.drawLine(QPointF(px, area.top()), QPointF(px, area.bottom()))
            p.setBrush(QColor(ACCENT))
            p.drawEllipse(QPointF(px, py), 4, 4)
            box = QRectF(max(area.left(), min(px+10, area.right()-205)), area.top()+8, 200, 28)
            p.fillRect(box, QColor(SURFACE))
            p.setPen(QColor(TEXT))
            p.drawText(box, Qt.AlignCenter, f"{times[index]:.3f} s   |   {values[index]:+.3f}")

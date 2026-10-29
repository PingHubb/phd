"""Interactive software-rendered 3D sensor plotter; no GPU/VTK dependency.

The scene is a regular taxel grid, not a measurement of physical deformation.
Projection, depth sorting and picking use the same column-major taxel IDs as
the heatmap. All scaling is display-only; input arrays are never modified.
"""

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QToolTip, QVBoxLayout, QWidget,
)

from .core import Grid
from .widgets import Heatmap, ACCENT, BG, LINE, MUTED, TEXT


def scene_geometry(grid, values, signal_mode, scale, gain):
    """Return base/response points, colours and valid mask in taxel-ID order."""
    columns, rows = np.divmod(np.arange(grid.size), grid.rows)
    span = max(grid.columns - 1, grid.rows - 1, 1)
    base = np.column_stack(((columns-(grid.columns-1)/2)*2/span,
                            ((grid.rows-1)/2-rows)*2/span, np.zeros(grid.size)))
    data = np.full(grid.size, np.nan) if values is None else np.asarray(values, dtype=float)
    if data.shape != (grid.size,):
        raise ValueError("Plotter frame size does not match the grid")
    valid = np.isfinite(data)
    response = np.zeros(grid.size)
    if signal_mode in {"raw", "baseline"}:
        low, high = (float(data[valid].min()), float(data[valid].max())) if valid.any() else (0., 0.)
        if high > low:
            response[valid] = (data[valid]-low)/(high-low)
        target = np.tile([94., 234., 212.], (grid.size, 1))
    else:
        response[valid] = np.clip(data[valid]/max(float(scale), .01), -1, 1)
        target = np.where((response < 0)[:, None], [99., 197., 255.], [255., 183., 107.])
    neutral = np.array([100., 123., 148.])
    amount = np.abs(response)[:, None]
    colours = (neutral*(1-amount)+target*amount).astype(np.uint8)
    points = base.copy()
    points[:, 2] = response * .65 * gain
    return base, points, colours, valid


class SensorPlot3D(QWidget):
    taxel_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = Grid(5, 5)
        self.values = None
        self.signal_mode = "difference"
        self.style = "point_grid"
        self.scale = 100.
        self.gain = 1.
        self.selected = 12
        self.show_values = True
        self.yaw, self.elevation, self.zoom = -32., 48., 1.
        self.pan = np.zeros(2)
        self._press = self._last = None
        self._dragged = False
        self._panning = False
        self.setMinimumSize(280, 260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("3D sensor plotter; drag to rotate, wheel to zoom, click or arrow keys to select a taxel")

    def reset_view(self):
        self.yaw, self.elevation, self.zoom = -32., 48., 1.
        self.pan[:] = 0
        self.update()

    def top_view(self):
        self.yaw, self.elevation, self.zoom = 0., 90., 1.
        self.pan[:] = 0
        self.update()

    def project(self, points):
        yaw, elevation = np.radians([self.yaw, self.elevation])
        x = points[:, 0]*np.cos(yaw) - points[:, 1]*np.sin(yaw)
        horizontal_depth = points[:, 0]*np.sin(yaw) + points[:, 1]*np.cos(yaw)
        y = horizontal_depth*np.sin(elevation) + points[:, 2]*np.cos(elevation)
        depth = points[:, 2]*np.sin(elevation) - horizontal_depth*np.cos(elevation)
        pixels = min(self.width()/3.6, max(1, self.height()-62)/(2.7+.45*self.gain)) * self.zoom
        projected = np.column_stack((self.width()/2+x*pixels+self.pan[0],
                                     self.height()/2-y*pixels+self.pan[1]))
        return projected, depth

    def geometry(self):
        return scene_geometry(self.grid, self.values, self.signal_mode, self.scale, self.gain)

    def projected_taxels(self):
        base, points, _colours, _valid = self.geometry()
        return self.project(base if self.style == "heatmap_3d" else points)

    def index_at(self, point):
        points, depth = self.projected_taxels()
        distances = np.linalg.norm(points - [point.x(), point.y()], axis=1)
        nearest = float(distances.min())
        if nearest > 13:
            return None
        # Prefer the front-most point only when projections essentially overlap.
        candidates = np.flatnonzero(distances <= nearest+.5)
        return int(candidates[np.argmax(depth[candidates])])

    def select(self, index):
        index = min(max(int(index), 0), self.grid.size-1)
        if index != self.selected:
            self.selected = index
            self.taxel_selected.emit(index)
        self.update()

    def mousePressEvent(self, event):
        if event.button() in {Qt.LeftButton, Qt.MiddleButton, Qt.RightButton}:
            self.setFocus()
            self._press = self._last = event.localPos()
            self._dragged = False
            self._panning = event.button() != Qt.LeftButton or bool(event.modifiers() & Qt.ShiftModifier)

    def mouseMoveEvent(self, event):
        if self._press is not None:
            delta = event.localPos()-self._last
            self._last = event.localPos()
            if (event.localPos()-self._press).manhattanLength() > 4:
                self._dragged = True
            if self._dragged:
                QToolTip.hideText()
                if self._panning:
                    self.pan += [delta.x(), delta.y()]
                else:
                    self.yaw = (self.yaw+delta.x()*.45) % 360
                    self.elevation = float(np.clip(self.elevation+delta.y()*.35, 8, 90))
                self.update()
            return
        index = self.index_at(event.localPos())
        if index is not None:
            value = "No data" if self.values is None else f"{self.values[index]:+.3f}"
            QToolTip.showText(event.globalPos(), f"Taxel {index} · row {index % self.grid.rows}, column {index // self.grid.rows}\n{value}", self)
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, event):
        if self._press is not None and not self._dragged and event.button() == Qt.LeftButton:
            index = self.index_at(event.localPos())
            if index is not None:
                self.select(index)
        self._press = self._last = None

    def wheelEvent(self, event):
        self.zoom = float(np.clip(self.zoom * (1.12 if event.angleDelta().y() > 0 else 1/1.12), .4, 4))
        self.update()
        event.accept()

    def keyPressEvent(self, event):
        row, column = self.selected % self.grid.rows, self.selected // self.grid.rows
        moves = {Qt.Key_Left: (row, column-1), Qt.Key_Right: (row, column+1),
                 Qt.Key_Up: (row-1, column), Qt.Key_Down: (row+1, column)}
        if event.key() in moves:
            row, column = moves[event.key()]
            if 0 <= row < self.grid.rows and 0 <= column < self.grid.columns:
                self.select(column*self.grid.rows+row)
        elif event.key() == Qt.Key_R:
            self.reset_view()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG))
        base, points, colours, valid = self.geometry()
        screen, depth = self.project(base if self.style == "heatmap_3d" else points)
        base_screen, _ = self.project(base)
        edges = [(c*self.grid.rows+r, c*self.grid.rows+r+1)
                 for c in range(self.grid.columns) for r in range(self.grid.rows-1)]
        edges += [(c*self.grid.rows+r, (c+1)*self.grid.rows+r)
                  for c in range(self.grid.columns-1) for r in range(self.grid.rows)]
        # A quiet zero/reference plane makes positive and negative motion clear.
        p.setPen(QPen(QColor(LINE), .7, Qt.DotLine))
        for a, b in edges:
            p.drawLine(QPointF(*base_screen[a]), QPointF(*base_screen[b]))
        faces = []
        if self.style == "surface":
            for c in range(self.grid.columns-1):
                for r in range(self.grid.rows-1):
                    ids = np.array([c*self.grid.rows+r, (c+1)*self.grid.rows+r,
                                    (c+1)*self.grid.rows+r+1, c*self.grid.rows+r+1])
                    if valid[ids].all():
                        faces.append((float(depth[ids].mean()), screen[ids], colours[ids].mean(axis=0)))
        elif self.style == "heatmap_3d":
            half = .85/max(self.grid.rows-1, self.grid.columns-1, 1)
            offsets = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
            for index in range(self.grid.size):
                corners, z = self.project(base[index]+offsets)
                faces.append((float(z.mean()), corners, colours[index]*.7))
        for _z, polygon, colour in sorted(faces, key=lambda f: f[0]):
            p.setBrush(QColor(*map(int, colour)))
            p.setPen(QPen(QColor(LINE), .65))
            p.drawPolygon(QPolygonF([QPointF(*v) for v in polygon]))
        if self.style == "point_grid":
            for a, b in sorted(edges, key=lambda ab: (depth[ab[0]]+depth[ab[1]])/2):
                if valid[a] and valid[b] or self.values is None:
                    colour = (colours[a].astype(float)+colours[b])/2
                    p.setPen(QPen(QColor(*map(int, colour)), 1.4))
                    p.drawLine(QPointF(*screen[a]), QPointF(*screen[b]))
        radius = max(1.8, min(5., 24/max(self.grid.rows, self.grid.columns)))
        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        for index in np.argsort(depth):
            position = QPointF(*screen[index])
            if self.style != "heatmap_3d":
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(*map(int, colours[index])))
                if valid[index] or self.values is None:
                    p.drawEllipse(position, radius, radius)
            if self.show_values and self.grid.size <= 100:
                p.setPen(QColor(TEXT))
                p.drawText(position+QPointF(5, -6), str(index))
        selected = min(self.selected, self.grid.size-1)
        position = QPointF(*screen[selected])
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(ACCENT), 2))
        p.drawEllipse(position, 7, 7)
        if self.values is not None:
            p.setPen(QPen(QColor(ACCENT), 1, Qt.DashLine))
            p.drawLine(QPointF(*base_screen[selected]), position)
        p.setPen(QColor(MUTED))
        p.drawText(QRectF(8, 4, self.width()-16, 24), Qt.AlignLeft | Qt.AlignVCenter,
                   "Regular taxel grid · height shows signal, not physical deformation")
        value = "no data" if self.values is None else f"{self.values[selected]:+.2f}"
        p.setPen(QColor(ACCENT))
        p.drawText(QRectF(8, self.height()-26, self.width()-16, 22), Qt.AlignLeft | Qt.AlignVCenter,
                   f"Taxel {selected}  ·  {value}")
        # Axes in taxel grid coordinates, not millimetres.
        for index, label in [(0, "0"), ((self.grid.columns-1)*self.grid.rows, "column →"),
                             (self.grid.rows-1, "row ↓")]:
            p.setPen(QColor(MUTED))
            p.drawText(QPointF(*base_screen[index])+QPointF(-18, 20), label)


class PlotterPanel(QWidget):
    taxel_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.view_mode = QComboBox()
        for label, mode in [("3D Point Grid", "point_grid"), ("3D Surface", "surface"),
                            ("3D Heatmap", "heatmap_3d"), ("2D Heatmap", "heatmap")]:
            self.view_mode.addItem(label, mode)
        self.view_mode.setAccessibleName("Sensor plotter view")
        toolbar.addWidget(self.view_mode)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("Height"))
        self.gain = QDoubleSpinBox()
        self.gain.setRange(.2, 3.)
        self.gain.setSingleStep(.2)
        self.gain.setValue(1.)
        self.gain.setSuffix("×")
        self.gain.setToolTip("Display exaggeration only; does not change readings or recordings.")
        toolbar.addWidget(self.gain)
        self.reset_button = QPushButton("Reset view")
        self.top_button = QPushButton("Top")
        toolbar.addWidget(self.reset_button)
        toolbar.addWidget(self.top_button)
        layout.addLayout(toolbar)
        self.stack = QStackedWidget()
        self.plotter = SensorPlot3D()
        self.heatmap = Heatmap()
        self.stack.addWidget(self.plotter)
        self.stack.addWidget(self.heatmap)
        layout.addWidget(self.stack, 1)
        self.legend = QLabel("Blue − / orange + · signed height · display range ±100")
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet(f"color: {MUTED};")
        layout.addWidget(self.legend)
        self.hint = QLabel("Drag: rotate · wheel: zoom · Shift-drag: pan · click a node: select")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        layout.addWidget(self.hint)
        self.plotter.taxel_selected.connect(self.taxel_selected)
        self.heatmap.taxel_selected.connect(self.taxel_selected)
        self.view_mode.currentIndexChanged.connect(self._change_view)
        self.gain.valueChanged.connect(self._change_view)
        self.reset_button.clicked.connect(self.plotter.reset_view)
        self.top_button.clicked.connect(self.plotter.top_view)

    def _change_view(self, *_args):
        mode = self.view_mode.currentData()
        self.stack.setCurrentIndex(1 if mode == "heatmap" else 0)
        self.plotter.style = mode
        self.plotter.gain = self.gain.value()
        self.gain.setEnabled(mode in {"point_grid", "surface"})
        self.reset_button.setEnabled(mode != "heatmap")
        self.top_button.setEnabled(mode != "heatmap")
        self.hint.setText("Click a cell or use arrow keys to select a taxel." if mode == "heatmap"
                          else "Drag: rotate · wheel: zoom · Shift-drag: pan · click a node: select")
        self.plotter.update()

    def set_frame(self, grid, values, mode, scale, selected, show_values):
        for widget in (self.plotter, self.heatmap):
            widget.grid = grid
            widget.values = values
            widget.selected = min(selected, grid.size-1)
            widget.scale = scale
            widget.show_values = show_values
        self.heatmap.mode = self.plotter.signal_mode = mode
        if mode in {"raw", "baseline"}:
            finite = np.array([]) if values is None else values[np.isfinite(values)]
            limits = f"{finite.min():.1f} to {finite.max():.1f}" if len(finite) else "no data"
            self.legend.setText(f"Raw/baseline height & colour: relative auto range ({limits})")
        else:
            unit = "%" if mode == "percent" else "units"
            self.legend.setText(f"Blue − / orange + · zero/reference plane · display range ±{scale:g} {unit}")
        self.stack.currentWidget().update()

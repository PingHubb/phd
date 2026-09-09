"""Result plots for the force-versus-tactile calibration experiment."""

from __future__ import annotations

import math
from pathlib import Path

from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPainterPath
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from phd.ui import theme


SIGNAL_LABELS = {
    "raw": "Raw sensor value",
    "raw_ave": "Averaged raw value",
    "diff": "Sensor difference (counts)",
    "diff_ave": "Averaged difference (counts)",
    "diff_percent": "Relative change (%)",
    "diff_percent_ave": "Averaged relative change (%)",
}


class CalibrationCurveWidget(QWidget):
    """Draw aligned time traces or a force-versus-signal relationship."""

    def __init__(self, rows, signal_field, mode="time", parent=None):
        super().__init__(parent)
        self.rows = list(rows or [])
        self.signal_field = str(signal_field)
        self.mode = str(mode)
        self.setMinimumHeight(430)

    def set_data(self, rows, signal_field) -> None:
        self.rows = list(rows or [])
        self.signal_field = str(signal_field)
        self.update()

    @staticmethod
    def _finite_points(rows, x_field, y_field):
        points = []
        for row in rows:
            try:
                x_value = float(row.get(x_field))
                y_value = float(row.get(y_field))
            except (TypeError, ValueError):
                continue
            if math.isfinite(x_value) and math.isfinite(y_value):
                points.append((x_value, y_value))
        return points

    @staticmethod
    def _range(values, padding=0.08):
        minimum = min(values)
        maximum = max(values)
        if abs(maximum - minimum) < 1e-12:
            extra = max(1.0, abs(minimum) * 0.1)
            return minimum - extra, maximum + extra
        extra = (maximum - minimum) * float(padding)
        return minimum - extra, maximum + extra

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
        path = QPainterPath()
        first = cls._map_point(points[0][0], points[0][1], x_range, y_range, rect)
        path.moveTo(*first)
        for x_value, y_value in points[1:]:
            path.lineTo(*cls._map_point(x_value, y_value, x_range, y_range, rect))
        painter.setPen(QPen(QColor(color), 2))
        painter.drawPath(path)

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

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.INPUT_BG))
        plot_rect = QRectF(self.rect().adjusted(78, 48, -78, -58))
        if plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        signal_label = SIGNAL_LABELS.get(self.signal_field, self.signal_field)
        if self.mode == "relationship":
            points = self._finite_points(
                self.rows, "force_delta_abs_n", self.signal_field
            )
            if len(points) < 2:
                painter.setPen(QColor(theme.TEXT_MUTED))
                painter.drawText(plot_rect, Qt.AlignCenter, "Not enough samples")
                return
            x_range = self._range([point[0] for point in points])
            x_range = (0.0, max(x_range[1], 1e-6))
            y_range = self._range([point[1] for point in points])
            self._draw_grid(painter, plot_rect)
            self._draw_series(
                painter, points, x_range, y_range, plot_rect, "#ff8a65"
            )
            painter.setBrush(QColor("#ffd180"))
            painter.setPen(Qt.NoPen)
            for x_value, y_value in points[:: max(1, len(points) // 80)]:
                px, py = self._map_point(
                    x_value, y_value, x_range, y_range, plot_rect
                )
                painter.drawEllipse(int(px - 2), int(py - 2), 4, 4)
            painter.setPen(QColor(theme.TEXT_PRIMARY))
            painter.drawText(12, 24, "Sensor Response vs Applied Force")
            painter.drawText(
                int(plot_rect.center().x() - 55), self.height() - 16, "Force change (N)"
            )
            painter.drawText(8, int(plot_rect.top() - 12), signal_label)
            painter.drawText(
                int(plot_rect.left() - 14),
                int(plot_rect.bottom() + 20),
                f"{x_range[0]:.3f}",
            )
            painter.drawText(
                int(plot_rect.right() - 30),
                int(plot_rect.bottom() + 20),
                f"{x_range[1]:.3f}",
            )
            painter.drawText(8, int(plot_rect.top() + 5), f"{y_range[1]:.3g}")
            painter.drawText(8, int(plot_rect.bottom()), f"{y_range[0]:.3g}")
            return

        force_points = self._finite_points(
            self.rows, "elapsed_s", "force_delta_abs_n"
        )
        signal_points = self._finite_points(
            self.rows, "elapsed_s", self.signal_field
        )
        if len(force_points) < 2 or len(signal_points) < 2:
            painter.setPen(QColor(theme.TEXT_MUTED))
            painter.drawText(plot_rect, Qt.AlignCenter, "Not enough samples")
            return
        x_values = [point[0] for point in force_points + signal_points]
        x_range = self._range(x_values, padding=0.0)
        force_range = self._range([point[1] for point in force_points])
        force_range = (0.0, max(force_range[1], 1e-6))
        signal_range = self._range([point[1] for point in signal_points])
        self._draw_grid(painter, plot_rect)
        self._draw_series(
            painter, force_points, x_range, force_range, plot_rect, "#42a5f5"
        )
        self._draw_series(
            painter, signal_points, x_range, signal_range, plot_rect, "#ff8a65"
        )
        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(12, 24, "Synchronized Force and Sensor Response")
        painter.drawText(
            int(plot_rect.center().x() - 34), self.height() - 16, "Time (s)"
        )
        painter.setPen(QColor("#42a5f5"))
        painter.drawText(8, int(plot_rect.top() - 12), "Force change (N)")
        painter.drawText(8, int(plot_rect.top() + 5), f"{force_range[1]:.3f}")
        painter.drawText(8, int(plot_rect.bottom()), f"{force_range[0]:.3f}")
        painter.setPen(QColor("#ff8a65"))
        painter.drawText(
            QRectF(
                plot_rect.center().x(),
                plot_rect.top() - 30,
                plot_rect.width() / 2.0,
                22,
            ),
            Qt.AlignRight | Qt.AlignVCenter,
            signal_label,
        )
        painter.drawText(
            int(plot_rect.right() + 8), int(plot_rect.top() + 5), f"{signal_range[1]:.3g}"
        )
        painter.drawText(
            int(plot_rect.right() + 8), int(plot_rect.bottom()), f"{signal_range[0]:.3g}"
        )
        painter.setPen(QColor(theme.TEXT_MUTED))
        painter.drawText(
            int(plot_rect.left() - 10),
            int(plot_rect.bottom() + 20),
            f"{x_range[0]:.2f}",
        )
        painter.drawText(
            int(plot_rect.right() - 28),
            int(plot_rect.bottom() + 20),
            f"{x_range[1]:.2f}",
        )


class CalibrationResultDialog(QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = dict(result or {})
        trial_results = list(self.result.get("trial_results") or [])
        self.trial_results = trial_results or [self.result]
        self.exported_graph_paths: list[str] = []
        self.setWindowTitle("Calibration Force-Signal Result")
        self.resize(1040, 700)
        self.setStyleSheet(
            f"QDialog {{ background: {theme.WINDOW_BG}; color: {theme.TEXT_PRIMARY}; }}"
        )
        layout = QVBoxLayout(self)
        requested_trials = int(self.result.get("requested_trials", 1) or 1)
        title = QLabel(
            f"Taxel {int(self.result.get('taxel_index', 0))} "
            + (
                "Repeated Force-Signal Characterization"
                if requested_trials > 1
                else "Force-Signal Characterization"
            )
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
        self.time_chart = CalibrationCurveWidget([], "diff_percent_ave", mode="time")
        self.relationship_chart = CalibrationCurveWidget(
            [], "diff_percent_ave", mode="relationship"
        )
        self.tabs.addTab(self.time_chart, "Time Alignment")
        self.tabs.addTab(self.relationship_chart, "Force vs Signal")
        layout.addWidget(self.tabs, 1)

        self.export_status_label = QLabel()
        self.export_status_label.setWordWrap(True)
        self.export_status_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        layout.addWidget(self.export_status_label)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)

        if self.trial_selector is not None:
            self.trial_selector.currentIndexChanged.connect(self._select_trial)
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
        rows = list(result.get("rows") or [])
        signal_field = str(result.get("signal_field") or "diff_percent_ave")
        duration = max(
            (self._finite_float(row.get("elapsed_s")) for row in rows),
            default=0.0,
        )
        peak_force = max(
            (self._finite_float(row.get("force_delta_abs_n")) for row in rows),
            default=0.0,
        )
        self.summary_label.setText(
            f"Trial {int(result.get('trial_number', index + 1))}   |   "
            f"Stop reason: {result.get('stop_reason', 'unknown')}   |   "
            f"Samples: {len(rows)}   |   Duration: {duration:.2f} s   |   "
            f"Peak force change: {peak_force:.3f} N\n"
            f"CSV: {result.get('csv_path', 'not saved')}"
        )
        self.time_chart.set_data(rows, signal_field)
        self.relationship_chart.set_data(rows, signal_field)

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

    @classmethod
    def _export_trial_graph(cls, result, path: Path) -> bool:
        rows = list((result or {}).get("rows") or [])
        if not rows:
            return False
        signal_field = str(
            (result or {}).get("signal_field") or "diff_percent_ave"
        )
        time_image = cls._render_curve(rows, signal_field, "time")
        relationship_image = cls._render_curve(
            rows, signal_field, "relationship"
        )
        margin = 18
        header_height = 44
        canvas = QImage(
            time_image.width() + relationship_image.width() + margin * 3,
            max(time_image.height(), relationship_image.height())
            + header_height
            + margin,
            QImage.Format_ARGB32,
        )
        canvas.fill(QColor(theme.WINDOW_BG))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(theme.TEXT_PRIMARY))
        painter.drawText(
            margin,
            28,
            f"Calibration Trial {int((result or {}).get('trial_number', 0)):04d}",
        )
        painter.drawImage(margin, header_height, time_image)
        painter.drawImage(
            margin * 2 + time_image.width(),
            header_height,
            relationship_image,
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
        self.exported_graph_paths = exported
        if exported:
            message = (
                f"Saved {len(exported)} trial graph(s): {output_directory}"
            )
            if failures:
                message += f" ({failures} failed)"
        else:
            message = "Trial graphs could not be exported."
        self.export_status_label.setText(message)
        parent = self.parent()
        callback = getattr(parent, "_append_sidebar_control_message", None)
        if callable(callback):
            callback(message)

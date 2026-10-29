"""Standalone sensor UI: imports no research application or robot controller."""

from collections import deque
from datetime import datetime
import logging
from pathlib import Path
import time

import numpy as np
import serial.tools.list_ports
from PyQt5.QtCore import Qt, QTimer, QStandardPaths
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QMainWindow, QPlainTextEdit,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from . import VERSION
from .core import Grid, Recording, displayed_values
from .worker import SensorWorker
from .widgets import TraceChart, BG, SURFACE, TEXT, MUTED, LINE, ACCENT
from .plotter import PlotterPanel


STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QLabel#title {{ font-size: 25px; font-weight: 600; }}
QLabel#subtitle {{ color: {MUTED}; }}
QLabel#section {{ color: {ACCENT}; font-size: 12px; font-weight: 600; }}
QLabel#value {{ font-size: 24px; font-weight: 600; }}
QPushButton {{ background: #22334B; border: 1px solid #3C526C; border-radius: 7px; padding: 9px 12px; }}
QPushButton:hover {{ background: #2D4460; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #385775; }}
QPushButton:disabled {{ color: #8290A4; background: #1C293C; border-color: #28384D; }}
QPushButton#primary {{ background: #13796F; border-color: #28A99B; color: white; font-weight: 600; }}
QPushButton#primary:hover {{ background: #168E82; }}
QPushButton#primary:disabled {{ background: #24443F; color: #93A4A1; border-color: #36554E; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ background: {SURFACE}; border: 1px solid #435973; border-radius: 5px; padding: 6px; min-height: 21px; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPushButton:focus {{ border: 2px solid {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; selection-background-color: #27565B; }}
QPlainTextEdit {{ background: #111D2E; border: 1px solid {LINE}; border-radius: 6px; color: {MUTED}; font-size: 11px; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #3C526C; min-height: 30px; border-radius: 4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QSplitter::handle {{ background: {LINE}; }}
QCheckBox {{ spacing: 8px; padding: 4px 0; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid #6C809A; border-radius: 3px; background: {SURFACE}; }}
QCheckBox::indicator:checked {{ background: #2AAB9B; border-color: {ACCENT}; }}
QToolTip {{ background: #22334B; color: {TEXT}; border: 1px solid {ACCENT}; padding: 6px; }}
"""


def caption(text, name="subtitle"):
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(name not in {"title", "section", "value"})
    return label


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PingLab Sensor Viewer · {VERSION}")
        self.resize(1180, 900)
        self.setMinimumSize(920, 650)
        self.worker = None
        self.recording = None
        self.packet = None
        self.source = {}
        self.grid = Grid(5, 5)
        self.history = deque(maxlen=3000)
        self.started_at = None
        self.last_frame_at = None
        self._closing = False
        self._last_error = ""
        self._recording_drop_start = 0
        self._last_baseline_version = None
        self._calibrating = False
        self._stopping = False
        self.viewer_dialog = None
        self.plotter_dialog = None
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(14)
        title = QHBoxLayout()
        title.addWidget(caption("Sensor Viewer", "title"))
        title.addStretch()
        self.connection_label = caption("OFFLINE")
        title.addWidget(self.connection_label)
        layout.addLayout(title)
        layout.addWidget(caption("Connect. Explore. Record.   A sensor-only workspace — no robot, ROS or experiment controls."))
        body = QSplitter(Qt.Horizontal)
        layout.addWidget(body, 1)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_scroll.setMinimumWidth(255)
        side_scroll.setMaximumWidth(330)
        side = QWidget()
        side_scroll.setWidget(side)
        controls = QVBoxLayout(side)
        controls.setContentsMargins(0, 4, 14, 4)
        controls.setSpacing(12)
        controls.addWidget(caption("01 / CONNECTION", "section"))
        self.port = QComboBox()
        self.port.setAccessibleName("Sensor COM port")
        self.port.setToolTip("Only the selected port is opened. Close Arduino Serial Monitor before connecting.")
        controls.addWidget(self.port)
        self.refresh_button = QPushButton("Refresh ports")
        self.refresh_button.clicked.connect(self.refresh_ports)
        controls.addWidget(self.refresh_button)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.baud = QComboBox()
        for text, value in [("Auto · 115200 / 9600", 0), ("115200", 115200), ("9600", 9600), ("230400", 230400), ("460800", 460800), ("921600", 921600)]:
            self.baud.addItem(text, value)
        form.addRow("Baud rate", self.baud)
        self.shape = QComboBox()
        for text, value in [("Auto · firmware", None), ("5 × 5", (5, 5)), ("8 × 10", (8, 10)), ("20 × 20", (20, 20)), ("Custom", "custom")]:
            self.shape.addItem(text, value)
        form.addRow("Grid", self.shape)
        self.row_count = QSpinBox()
        self.column_count = QSpinBox()
        for spin in (self.row_count, self.column_count):
            spin.setRange(1, 64)
            spin.setValue(5)
        custom = QWidget()
        custom_layout = QHBoxLayout(custom)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.addWidget(self.row_count)
        custom_layout.addWidget(QLabel("×"))
        custom_layout.addWidget(self.column_count)
        form.addRow("Rows × cols", custom)
        self.shape.currentIndexChanged.connect(self._shape_changed)
        controls.addLayout(form)
        self.connect_button = QPushButton("Connect sensor")
        self.connect_button.setObjectName("primary")
        self.connect_button.clicked.connect(lambda: self.start(False))
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect)
        controls.addWidget(self.connect_button)
        controls.addWidget(self.disconnect_button)
        self.demo_button = QPushButton("Try Demo · no hardware")
        self.demo_button.clicked.connect(lambda: self.start(True))
        controls.addWidget(self.demo_button)
        controls.addWidget(caption("02 / BASELINE & DISPLAY", "section"))
        self.calibrate_button = QPushButton("Update Calibration")
        self.calibrate_button.setToolTip("Re-zero the sensor while untouched. Keeps the same firmware command as the research UI.")
        self.calibrate_button.clicked.connect(self.calibrate)
        controls.addWidget(self.calibrate_button)
        self.mode = QComboBox()
        for text, value in [("Signed difference", "difference"), ("Raw signal", "raw"), ("Baseline", "baseline"), ("Signed difference (%)", "percent")]:
            self.mode.addItem(text, value)
        self.mode.setToolTip("Signed difference = raw signal − baseline; negative values remain negative.")
        controls.addWidget(self.mode)
        self.mode.currentIndexChanged.connect(self.render_frame)
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(.01, 1000000)
        self.range_spin.setDecimals(2)
        self.range_spin.setValue(100)
        scale_form = QFormLayout()
        scale_form.addRow("Colour range ±", self.range_spin)
        controls.addLayout(scale_form)
        self.range_spin.valueChanged.connect(self.render_frame)
        self.numbers = QCheckBox("Show taxel IDs / cell values")
        self.numbers.setChecked(True)
        self.numbers.toggled.connect(self.render_frame)
        controls.addWidget(self.numbers)
        controls.addWidget(caption("03 / RECORDING", "section"))
        self.record_button = QPushButton("Record CSV…")
        self.record_button.clicked.connect(self.toggle_recording)
        controls.addWidget(self.record_button)
        controls.addWidget(caption("CSV saves every delivered frame for all taxels, plus baseline metadata. No robot data."))
        controls.addStretch()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(150)
        self.log.setMaximumHeight(100)
        self.log.setAccessibleName("Connection and recording log")
        controls.addWidget(self.log)
        body.addWidget(side_scroll)
        for combo in (self.port, self.baud, self.shape, self.mode):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(1)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        display = QWidget()
        # On short laptop screens the content scrolls instead of overlapping.
        display.setMinimumHeight(760)
        display_layout = QVBoxLayout(display)
        display_layout.setContentsMargins(12, 0, 0, 0)
        row = QHBoxLayout()
        self.grid_label = caption("SENSOR PLOTTER", "section")
        row.addWidget(self.grid_label)
        row.addStretch()
        self.selected = QSpinBox()
        self.selected.setRange(0, 24)
        self.selected.setValue(12)
        self.selected.setPrefix("Taxel ")
        row.addWidget(self.selected)
        self.expand_plotter_button = QPushButton("Open 3D Plotter")
        self.expand_plotter_button.clicked.connect(self.open_plotter)
        row.addWidget(self.expand_plotter_button)
        display_layout.addLayout(row)
        self.plot_panel = PlotterPanel()
        self.heatmap = self.plot_panel.heatmap
        self.plot_panel.taxel_selected.connect(self.selected.setValue)
        self.selected.valueChanged.connect(self.select_taxel)
        display_layout.addWidget(self.plot_panel, 3)
        values = QHBoxLayout()
        self.value_label = caption("—", "value")
        self.raw_label = caption("Raw —  ·  Baseline —")
        values.addWidget(self.value_label)
        values.addStretch()
        values.addWidget(self.raw_label)
        display_layout.addLayout(values)
        trace_toolbar = QHBoxLayout()
        trace_toolbar.addWidget(caption("SIGNAL HISTORY", "section"))
        trace_toolbar.addStretch()
        self.freeze = QCheckBox("Freeze graph")
        trace_toolbar.addWidget(self.freeze)
        self.pop_button = QPushButton("Open Signal Viewer")
        self.pop_button.clicked.connect(self.open_viewer)
        trace_toolbar.addWidget(self.pop_button)
        display_layout.addLayout(trace_toolbar)
        self.chart = TraceChart()
        display_layout.addWidget(self.chart, 2)
        display_layout.addWidget(caption("Click a taxel or use arrow keys. Hover over the graph for exact values; mouse wheel changes the time window."))
        display_scroll = QScrollArea()
        display_scroll.setWidgetResizable(True)
        display_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        display_scroll.setWidget(display)
        body.addWidget(display_scroll)
        body.setSizes([290, 850])
        self.status = caption("Ready. Select a COM port and connect, or explore Demo mode.")
        self.status.setMinimumHeight(38)
        layout.addWidget(self.status)
        self.refresh_ports()
        self._set_running(False)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.consume)
        self.timer.start(33)

    def note(self, message):
        self.status.setText(message)
        self.log.appendPlainText(f"{datetime.now():%H:%M:%S}  {message}")
        logging.info(message)

    def _shape_changed(self):
        enabled = self.worker is None and self.shape.currentData() == "custom"
        self.row_count.setEnabled(enabled)
        self.column_count.setEnabled(enabled)

    def refresh_ports(self):
        selected = self.port.currentData()
        self.port.clear()
        for item in sorted(serial.tools.list_ports.comports(), key=lambda p: p.device):
            self.port.addItem(f"{item.device} · {item.description}", item.device)
            self.port.setItemData(self.port.count()-1, f"USB serial: {item.serial_number or 'not reported'}", Qt.ToolTipRole)
        if not self.port.count():
            self.port.addItem("No serial ports found", None)
        index = self.port.findData(selected)
        if index >= 0:
            self.port.setCurrentIndex(index)
        self.connect_button.setEnabled(self.worker is None and self.port.currentData() is not None)

    def _set_running(self, running):
        for widget in (self.port, self.baud, self.shape, self.refresh_button, self.demo_button):
            widget.setEnabled(not running)
        self.connect_button.setEnabled(not running and self.port.currentData() is not None)
        self.disconnect_button.setEnabled(running)
        self.calibrate_button.setEnabled(running and self.packet is not None)
        self.record_button.setEnabled(running and self.packet is not None)
        self._shape_changed()

    def start(self, demo=False):
        if self.worker is not None:
            return
        try:
            choice = self.shape.currentData()
            grid = (Grid(self.row_count.value(), self.column_count.value()) if choice == "custom"
                    else Grid(*choice) if choice else None)
        except ValueError as exc:
            self.note(str(exc))
            return
        port = "DEMO" if demo else self.port.currentData()
        if not port:
            return
        self.packet = None
        self.history.clear()
        self.render_frame()
        self.value_label.setText("—")
        self.raw_label.setText("Raw —  ·  Baseline —")
        self.chart.set_series([], [], "Waiting for sensor")
        self.started_at = self.last_frame_at = None
        self._last_baseline_version = None
        self._last_error = ""
        self._calibrating = self._stopping = False
        self.freeze.setChecked(False)
        self.worker = SensorWorker(port, self.baud.currentData(), grid, demo, self)
        self.worker.status.connect(self.note)
        self.worker.failed.connect(self.failed)
        self.worker.connected.connect(self.on_connected)
        self.worker.calibration_finished.connect(self.calibration_done)
        self.worker.finished.connect(self.on_finished)
        self._set_running(True)
        self.connection_label.setText("CONNECTING…" if not demo else "DEMO · STARTING")
        self.sync_plotter_title()
        self.worker.start()

    def on_connected(self, grid, source):
        self.grid, self.source = grid, source
        self.plot_panel.plotter.grid = self.heatmap.grid = grid
        self.selected.setRange(0, grid.size-1)
        self.selected.setValue((grid.columns//2)*grid.rows + grid.rows//2)
        self.heatmap.select(self.selected.value())
        self.grid_label.setText(f"SENSOR PLOTTER · {grid.rows} × {grid.columns}")
        self.render_frame()
        self.note(f"{'DEMO · generated, not measured data' if source['demo'] else source['port']} · {grid.size} taxels")

    def failed(self, message):
        self._last_error = message
        self.note(message)

    def disconnect(self):
        if self.worker:
            self._stopping = True
            self.disconnect_button.setEnabled(False)
            self.calibrate_button.setEnabled(False)
            self.record_button.setEnabled(False)
            self.worker.stop()
            self.note("Disconnecting; waiting for the current serial transaction to finish…")

    def on_finished(self):
        self.consume()
        self.stop_recording()
        worker, self.worker = self.worker, None
        if worker:
            worker.deleteLater()
        self._set_running(False)
        self.connection_label.setText("OFFLINE · last data retained" if self.packet is not None else "OFFLINE")
        self.sync_plotter_title()
        self.note(self._last_error or "Disconnected. No serial port is held open.")
        if self._closing:
            QTimer.singleShot(0, self.close)

    def calibrate(self):
        if self.worker and not self._stopping:
            self._calibrating = True
            self.calibrate_button.setEnabled(False)
            self.worker.calibrate()

    def calibration_done(self, ok):
        self._calibrating = False
        self.calibrate_button.setEnabled(self.worker is not None and not self._stopping)

    def consume(self):
        if self.worker is None:
            return
        packets = self.worker.take_packets()
        for packet in packets:
            if self.started_at is None:
                self.started_at = packet["time"]
            if packet["baseline_version"] != self._last_baseline_version:
                self.history.clear()  # Do not join traces across a re-zero.
                self._last_baseline_version = packet["baseline_version"]
            self.history.append(packet)
            self.packet = packet
            self.last_frame_at = packet["time"]
            if self.recording:
                try:
                    self.recording.append(packet)
                except Exception as exc:
                    self.note(f"Recording stopped: {exc}")
                    self.stop_recording()
        if packets:
            self.calibrate_button.setEnabled(not self._calibrating and not self._stopping)
            self.record_button.setEnabled(not self._stopping)
            self.render_frame()
        if self.last_frame_at is not None:
            age = time.monotonic()-self.last_frame_at
            label = "DEMO · SYNTHETIC" if self.source.get("demo") else str(self.source.get("port", "Sensor"))
            if len(self.history) > 1:
                recent = list(self.history)[-100:]
                hz = (len(recent)-1) / max(recent[-1]["time"]-recent[0]["time"], .001)
            else:
                hz = 0
            state = f"STALE · {age:.1f}s" if age > 1.5 else f"{hz:.1f} frames/s"
            self.connection_label.setText(f"{label} · {state}" + (f" · dropped {self.worker.dropped}" if self.worker.dropped else ""))
            self.sync_plotter_title()

    def select_taxel(self, index):
        self.render_frame()

    def render_frame(self, *_args):
        mode = self.mode.currentData()
        self.range_spin.setEnabled(mode in {"difference", "percent"})
        values = (None if self.packet is None else
                  displayed_values(self.packet["raw"], self.packet["baseline"], mode))
        panels = [self.plot_panel]
        if self.plotter_dialog is not None:
            panels.append(self.plotter_dialog.panel)
        for panel in panels:
            panel.set_frame(self.grid, values, mode, self.range_spin.value(),
                            self.selected.value(), self.numbers.isChecked())
        if self.packet is None:
            return
        packet = self.packet
        index = self.selected.value()
        suffix = "%" if mode == "percent" else "units"
        value = f"{values[index]:+.2f}" if mode in {"difference", "percent"} else f"{values[index]:.2f}"
        self.value_label.setText(f"P{index}   {value} {suffix}")
        self.raw_label.setText(f"Raw {packet['raw'][index]:.1f}  ·  Baseline {packet['baseline'][index]:.1f}")
        if not self.freeze.isChecked():
            times = [p["time"]-self.started_at for p in self.history]
            series = displayed_values(np.fromiter((p["raw"][index] for p in self.history), float),
                                      np.fromiter((p["baseline"][index] for p in self.history), float), mode)
            names = dict(difference="Signed difference (sensor units)", raw="Raw signal (sensor units)",
                         baseline="Baseline (sensor units)", percent="Signed difference (%)")
            label = f"Taxel {index} · {names[mode]}"
            self.chart.set_series(times, series, label)
            if self.viewer_dialog is not None:
                self.viewer_dialog.chart.set_series(times, series, label)

    def sync_plotter_title(self):
        if self.plotter_dialog is not None:
            self.plotter_dialog.setWindowTitle(f"Sensor Plotter — {self.connection_label.text()}")

    def open_plotter(self):
        if self.plotter_dialog is None:
            dialog = QDialog(self)
            dialog.resize(1080, 760)
            dialog.setMinimumSize(600, 440)
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            layout = QVBoxLayout(dialog)
            dialog.panel = PlotterPanel()
            # Keep a separate camera so the main view remains undisturbed.
            view_index = self.plot_panel.view_mode.currentIndex()
            dialog.panel.view_mode.setCurrentIndex(view_index if view_index != 3 else 0)
            dialog.panel.gain.setValue(self.plot_panel.gain.value())
            dialog.panel.taxel_selected.connect(self.selected.setValue)
            layout.addWidget(dialog.panel, 1)
            save = QPushButton("Save plotter image…")
            save.clicked.connect(lambda: self.save_chart(dialog.panel))
            layout.addWidget(save)
            dialog.destroyed.connect(lambda: setattr(self, "plotter_dialog", None))
            self.plotter_dialog = dialog
        self.sync_plotter_title()
        self.plotter_dialog.show()
        self.plotter_dialog.raise_()
        self.render_frame()

    def open_viewer(self):
        if self.viewer_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Sensor Signal Viewer — selected taxel")
            dialog.resize(1000, 560)
            dialog.setAttribute(Qt.WA_DeleteOnClose)
            layout = QVBoxLayout(dialog)
            layout.addWidget(caption("Select any taxel in the sensor plotter or heatmap. Hover to inspect; scroll to zoom the time window."))
            dialog.chart = TraceChart()
            layout.addWidget(dialog.chart)
            save = QPushButton("Save graph image…")
            save.clicked.connect(lambda: self.save_chart(dialog.chart))
            layout.addWidget(save)
            dialog.destroyed.connect(lambda: setattr(self, "viewer_dialog", None))
            self.viewer_dialog = dialog
        self.viewer_dialog.show()
        self.viewer_dialog.raise_()
        self.render_frame()

    def save_chart(self, chart):
        path, _ = QFileDialog.getSaveFileName(self, "Save graph", "sensor_signal.png", "PNG image (*.png)")
        if path:
            if not chart.grab().save(path, "PNG"):
                self.note("Could not save the graph image. Check the folder permissions.")

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
            return
        if self.packet is None or self.worker is None:
            return
        base = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        name = f"{'DEMO_' if self.source.get('demo') else ''}sensor_{datetime.now():%Y%m%d_%H%M%S}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Record sensor data", str(Path(base)/name), "CSV (*.csv)")
        if not path:
            return
        try:
            self.recording = Recording(Path(path).with_suffix(".csv"), self.grid, self.source)
            self._recording_drop_start = self.worker.dropped
            self.record_button.setText("Stop recording")
            self.note(f"Recording all {self.grid.size} taxels to {self.recording.path}")
        except Exception as exc:
            self.note(f"Cannot start recording: {exc}")

    def stop_recording(self):
        if self.recording:
            recording, self.recording = self.recording, None
            try:
                recording.close((self.worker.dropped-self._recording_drop_start) if self.worker else 0)
                self.note(f"Saved {recording.count} frames: {recording.path}")
            except Exception as exc:
                self.note(f"Recording finalisation error: {exc}. Check the saved CSV and log.")
        self.record_button.setText("Record CSV…")

    def closeEvent(self, event):
        if self.worker is not None:
            self._closing = True
            self.disconnect()
            event.ignore()
            return
        self.stop_recording()
        event.accept()


def main(argv=None):
    app = QApplication(argv or [])
    app.setApplicationName("PingLab Sensor Viewer")
    app.setOrganizationName("PingLab")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    if argv and "--demo" in argv:
        QTimer.singleShot(0, lambda: window.start(True))
    return app.exec_()

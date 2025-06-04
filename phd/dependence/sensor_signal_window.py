import sys
import math
import serial
import serial.tools.list_ports
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QApplication,
    QHeaderView,
    QLabel,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QDialogButtonBox,
    QMessageBox,
)
from PyQt5.QtGui import QFont, QColor, QBrush

# Color‐logic constants:
threshold_offset = 30
max_value = 100
max_value_above = 1000


class SerialPortDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Serial Port")
        self.setModal(True)
        self.selected_port = None

        layout = QVBoxLayout(self)
        label = QLabel("Available serial ports:")
        layout.addWidget(label)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.populate_ports()

    def populate_ports(self):
        ports = list(serial.tools.list_ports.comports())
        all_port_names = [p.device for p in ports]

        acm_ports = sorted([name for name in all_port_names if "ttyACM" in name])
        other_ports = sorted([name for name in all_port_names if "ttyACM" not in name])
        sorted_ports = acm_ports + other_ports

        for port_name in sorted_ports:
            item = QListWidgetItem(port_name)
            self.list_widget.addItem(item)

        if not sorted_ports:
            placeholder = QListWidgetItem("— no serial ports found —")
            placeholder.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(placeholder)

        if sorted_ports:
            self.list_widget.setCurrentRow(0)

    def accept_selection(self):
        current = self.list_widget.currentItem()
        if current and current.flags() & Qt.ItemIsEnabled:
            self.selected_port = current.text()
            self.accept()
        else:
            self.reject()

    def get_selected_port(self):
        return self.selected_port


class SensorSignalWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, flags=Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

        # ── Make sure `self.timer` always exists, even if we return early. ──
        self.timer = None

        # ─── 0) Prompt the user to pick a serial port before anything else ───
        port = self._choose_serial_port()
        if not port:
            QMessageBox.information(
                None,
                "No Port Selected",
                "No serial port was selected. The sensor viewer will now close.",
            )
            self.close()
            return

        # ─── Now we know a port was chosen ───
        try:
            from phd.dependence.sensor_api import ArduinoCommander
            self.sensor_api = ArduinoCommander(serial_port=port, baud_rate=9600)
        except Exception as e:
            QMessageBox.critical(
                None,
                "Serial Connection Error",
                f"Failed to open serial port {port}:\n{e}",
            )
            self.close()
            return

        # ─── 1) Immediately query channel_check() ───
        try:
            data_list = self.sensor_api.channel_check()
            self.table_columns = data_list[0] - 1
            self.table_rows = data_list[1]
        except Exception as e:
            print(f"[SensorSignalWindow] channel_check() failed: {e}")
            self.table_columns = 1
            self.table_rows = 1

        # ─── Grab an initial calibration ───
        try:
            self.calibration_data = self.sensor_api.update_cal()
        except Exception as e:
            self.calibration_data = []
            print(f"[SensorSignalWindow] Failed to update calibration on init: {e}")

        # Flags for display mode and hiding numbers
        self.show_diff = False
        self.hide_numbers = True

        # For threshold logic: collect first 100 diffs per cell
        self.initial_diffs = []
        self.threshold_max = {}
        self.initial_threshold_calculated = {}
        self.cells_remaining_for_threshold = None

        # Store per‐cell intensity (positive=red, negative=blue)
        self.red_intensity_dict = {}

        self.setWindowTitle("Sensor Signal Viewer")
        self.resize(1600, 800)

        layout = QVBoxLayout(self)

        # ── 1) QTableWidget ──
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.table)

        # Hide headers, stretch cells
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide()
        h_header = self.table.horizontalHeader()
        v_header = self.table.verticalHeader()
        h_header.setSectionResizeMode(QHeaderView.Stretch)
        v_header.setSectionResizeMode(QHeaderView.Stretch)

        font = QFont()
        font.setPointSize(10)
        self.table.setFont(font)

        self.selected_index = None
        self.table.cellClicked.connect(self.on_cell_clicked)

        # ── 2) Info + Buttons ──
        info_container = QWidget()
        info_layout = QHBoxLayout(info_container)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(8)

        self.info_label = QLabel("Click any cell to see its index")
        self.info_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(self.info_label, stretch=1)

        self.update_cal_button = QPushButton("Update Calibration")
        self.update_cal_button.clicked.connect(self.on_update_calibration)
        info_layout.addWidget(self.update_cal_button)

        self.toggle_mode_button = QPushButton("Show Differences")
        self.toggle_mode_button.clicked.connect(self.on_toggle_mode)
        info_layout.addWidget(self.toggle_mode_button)

        self.action_button = QPushButton("Action")
        self.action_button.setEnabled(False)
        info_layout.addWidget(self.action_button)

        self.hide_button = QPushButton("Hide Numbers")
        self.hide_button.clicked.connect(self.on_toggle_hide)
        info_layout.addWidget(self.hide_button)

        layout.addWidget(info_container)

        # ── 3) Timer ──
        # Only now that we’ve successfully chosen a port and queried channel_check do we create the QTimer:
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)

        # Initialize the “cells_remaining_for_threshold” counter if calibration_data is valid:
        if self.calibration_data:
            self.cells_remaining_for_threshold = len(self.calibration_data)
        else:
            self.cells_remaining_for_threshold = None

    def _choose_serial_port(self):
        dialog = SerialPortDialog(self)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            return dialog.get_selected_port()
        else:
            return None

    def on_update_calibration(self):
        try:
            new_cal = self.sensor_api.update_cal()
            self.calibration_data = list(new_cal)
            self.info_label.setText(f"Calibration updated ({len(self.calibration_data)} values)")

            # Reset per-cell threshold state
            self.initial_diffs.clear()
            self.threshold_max.clear()
            self.initial_threshold_calculated.clear()

            # Reset the “remaining” counter to n cells
            self.cells_remaining_for_threshold = len(self.calibration_data)

            # Reset the button’s style to default (no special background)
            self.update_cal_button.setStyleSheet("")

            # print("Thresholds cleared; will recalculate once 100 diffs per cell have arrived.")
        except Exception as e:
            self.info_label.setText(f"Calibration error: {e}")

    def on_toggle_mode(self):
        self.show_diff = not self.show_diff
        self.toggle_mode_button.setText(
            "Show Raw Values" if self.show_diff else "Show Differences"
        )
        self.refresh_data()

    def on_toggle_hide(self):
        self.hide_numbers = not self.hide_numbers
        self.hide_button.setText(
            "Show Numbers" if self.hide_numbers else "Hide Numbers"
        )
        self.refresh_data()

    def refresh_data(self):
        try:
            raw_list = self.sensor_api.read_raw()
        except Exception as e:
            self.table.clear()
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            item = QTableWidgetItem(f"Error: {e}")
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(0, 0, item)
            self.info_label.setText("Error reading sensor")
            self.selected_index = None
            self.action_button.setEnabled(False)
            return

        n = len(raw_list)
        if n == 0:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.info_label.setText("No data available")
            self.selected_index = None
            self.action_button.setEnabled(False)
            return

        if self.cells_remaining_for_threshold is None:
            self.cells_remaining_for_threshold = n

        if len(self.calibration_data) == n:
            diff_list = [abs(raw_list[i] - self.calibration_data[i]) for i in range(n)]
        else:
            diff_list = [0] * n

        if not self.initial_diffs:
            self.initial_diffs = [[] for _ in range(n)]

        for idx in range(n):
            if len(self.initial_diffs[idx]) < 100:
                self.initial_diffs[idx].append(diff_list[idx])

                if len(self.initial_diffs[idx]) == 100:
                    data100 = self.initial_diffs[idx]
                    max_val = max(data100)
                    self.threshold_max[idx] = max_val + threshold_offset
                    self.initial_threshold_calculated[idx] = True

                    self.cells_remaining_for_threshold -= 1
                    # print(f"Cell {idx} threshold_max set → {self.threshold_max[idx]!r}")

                    if self.cells_remaining_for_threshold == 0:
                        # All thresholds initialized → turn button green
                        self.update_cal_button.setStyleSheet("background-color: lightgreen;")
                        # print("All thresholds initialized; Update Calibration button is now green.")

        if self.show_diff and len(self.calibration_data) == n:
            display_list = diff_list[:]
        else:
            display_list = raw_list[:]

        rows = self.table_rows
        columns = self.table_columns
        shape_changed = (
                self.table.rowCount() != rows or
                self.table.columnCount() != columns
        )
        if shape_changed:
            self.table.clear()
            self.table.setRowCount(rows)
            self.table.setColumnCount(columns)

        self.red_intensity_dict.clear()

        idx = 0
        for col in range(columns):
            for row in range(rows):
                if idx < n:
                    val = display_list[idx]
                    diff = diff_list[idx]
                    t_max = self.threshold_max.get(idx, float('inf'))

                    if diff <= t_max:
                        color = QColor(255, 255, 255)
                        intensity = 0
                    elif diff > max_value_above:
                        blue_intensity = int(
                            ((diff - max_value_above) / (diff if diff > 0 else 1)) * 255
                        )
                        blue_intensity = max(0, min(255, blue_intensity))
                        color = QColor(255 - blue_intensity, 255 - blue_intensity, 255)
                        intensity = -blue_intensity
                    else:
                        difference = diff - t_max
                        red_intensity = int(((difference - 0) / (max_value - 0)) * 255)
                        red_intensity = max(0, min(255, red_intensity))
                        color = QColor(255, 255 - red_intensity, 255 - red_intensity)
                        intensity = red_intensity

                    self.red_intensity_dict[(row, col)] = intensity
                    cell_text = "" if self.hide_numbers else str(val)

                    if shape_changed:
                        item = QTableWidgetItem(cell_text)
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setBackground(QBrush(color))
                        self.table.setItem(row, col, item)
                    else:
                        existing = self.table.item(row, col)
                        if existing is None:
                            item = QTableWidgetItem(cell_text)
                            item.setTextAlignment(Qt.AlignCenter)
                            item.setBackground(QBrush(color))
                            self.table.setItem(row, col, item)
                        else:
                            existing.setText(cell_text)
                            existing.setTextAlignment(Qt.AlignCenter)
                            existing.setBackground(QBrush(color))
                else:
                    if shape_changed:
                        empty = QTableWidgetItem("")
                        empty.setFlags(Qt.NoItemFlags)
                        self.table.setItem(row, col, empty)
                    else:
                        existing = self.table.item(row, col)
                        if existing:
                            existing.setText("")
                            existing.setFlags(Qt.NoItemFlags)

                idx += 1

        # ── 5) Restore previously‐selected cell if still valid ──
        if self.selected_index is not None and self.selected_index < n:
            sel_col = self.selected_index // rows
            sel_row = self.selected_index % rows
            QTimer.singleShot(
                0,
                lambda r=sel_row, c=sel_col: self.table.setCurrentCell(r, c)
            )
            self.info_label.setText(f"Clicked cell index: {self.selected_index}")
            self.action_button.setEnabled(True)
        else:
            self.selected_index = None
            self.table.clearSelection()
            self.info_label.setText("Click any cell to see its index")
            self.action_button.setEnabled(False)

    def on_cell_clicked(self, row, column):
        flat_index = column * self.table_rows + row

        if self.selected_index == flat_index:
            self.table.clearSelection()
            self.table.setCurrentCell(-1, -1)
            self.selected_index = None
            self.info_label.setText("Click any cell to see its index")
            self.action_button.setEnabled(False)
        else:
            self.selected_index = flat_index
            self.info_label.setText(f"Clicked cell index: {flat_index}")
            self.action_button.setEnabled(True)

    def showEvent(self, event):
        # Only start the timer if it exists and isn’t already running
        if getattr(self, "timer", None) and not self.timer.isActive():
            self.timer.start(0)
        super().showEvent(event)

    def hideEvent(self, event):
        # Only stop the timer if it exists and is active
        if getattr(self, "timer", None) and self.timer.isActive():
            self.timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event):
        # Only stop the timer if it exists and is active
        if getattr(self, "timer", None) and self.timer.isActive():
            self.timer.stop()
        parent = self.parent()
        if parent and hasattr(parent, "sensorWindow"):
            parent.sensorWindow = None
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    w = SensorSignalWindow(None)
    w.show()

    sys.exit(app.exec_())

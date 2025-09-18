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
    QStyledItemDelegate,
)
from PyQt5.QtGui import QFont, QColor, QBrush, QPainter

# --- Data Role for our custom delegate ---
CALIBRATION_ROLE = Qt.UserRole + 1

# Color‐logic constants:
threshold_offset = 30
max_value = 100


# --- Custom Delegate for drawing cell content ---
class CellDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window = parent

    def paint(self, painter: QPainter, option, index):
        painter.save()
        background_brush = index.data(Qt.BackgroundRole)
        if background_brush:
            painter.fillRect(option.rect, background_brush)

        if self.window and self.window.hide_numbers:
            painter.restore()
            return

        main_text = index.data(Qt.DisplayRole)
        if main_text is not None:
            painter.setFont(option.font)
            painter.setPen(QColor(Qt.black))
            painter.drawText(option.rect, Qt.AlignCenter, str(main_text))

        cal_text = index.data(CALIBRATION_ROLE)
        if cal_text is not None:
            cal_font = QFont(option.font)
            cal_font.setPointSize(7)
            painter.setFont(cal_font)
            painter.setPen(QColor(Qt.darkGray))
            text_rect = option.rect.adjusted(3, 3, -3, -3)
            painter.drawText(text_rect, Qt.AlignTop | Qt.AlignLeft, str(cal_text))

        painter.restore()


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
            self.list_widget.addItem(QListWidgetItem(port_name))
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
    def __init__(self, parent=None, sensor_functions_ref=None):
        super().__init__(parent, flags=Qt.Window | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.timer = None
        self.sensor_functions = sensor_functions_ref

        port = self._choose_serial_port()
        if not port:
            QMessageBox.information(None, "No Port Selected",
                                    "No serial port was selected. The sensor viewer will now close.")
            self.close()
            return
        try:
            # Note: This is a placeholder for your actual API
            from phd.dependence.sensor_api import ArduinoCommander
            self.sensor_api = ArduinoCommander(serial_port=port, baud_rate=9600)
        except Exception as e:
            QMessageBox.critical(None, "Serial Connection Error", f"Failed to open serial port {port}:\n{e}")
            self.close()
            return
        try:
            data_list = self.sensor_api.channel_check()
            self.table_columns = data_list[0] - 1
            self.table_rows = data_list[1]
        except Exception as e:
            print(f"[SensorSignalWindow] channel_check() failed: {e}")
            self.table_columns = 1
            self.table_rows = 1
        try:
            self.calibration_data = self.sensor_api.update_cal()
        except Exception as e:
            self.calibration_data = []
            print(f"[SensorSignalWindow] Failed to update calibration on init: {e}")

        self.display_mode = "raw"
        self.hide_numbers = True
        self.initial_diffs = []
        self.threshold_max = {}
        self.initial_threshold_calculated = {}
        self.cells_remaining_for_threshold = None
        self.red_intensity_dict = {}

        self.setWindowTitle("Sensor Signal Viewer")
        self.resize(1600, 800)
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.table)

        delegate = CellDelegate(self)
        self.table.setItemDelegate(delegate)

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
        self.hide_button = QPushButton("Show Numbers")
        self.hide_button.clicked.connect(self.on_toggle_hide)
        info_layout.addWidget(self.hide_button)
        layout.addWidget(info_container)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        if self.calibration_data:
            self.cells_remaining_for_threshold = len(self.calibration_data)

    def _choose_serial_port(self):
        dialog = SerialPortDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_selected_port()
        return None

    def on_update_calibration(self):
        try:
            new_cal = self.sensor_api.update_cal()
            self.calibration_data = list(new_cal)
            self.info_label.setText(f"Calibration updated ({len(self.calibration_data)} values)")
            self.initial_diffs.clear()
            self.threshold_max.clear()
            self.initial_threshold_calculated.clear()
            self.cells_remaining_for_threshold = len(self.calibration_data)
            self.update_cal_button.setStyleSheet("")

            # Safely call the update method on the injected reference
            if self.sensor_functions:
                self.sensor_functions.updateCal()

        except Exception as e:
            self.info_label.setText(f"Calibration error: {e}")

    def on_toggle_mode(self):
        if self.display_mode == "raw":
            self.display_mode = "diff"
            self.toggle_mode_button.setText("Show Calibration")
        elif self.display_mode == "diff":
            self.display_mode = "cal"
            self.toggle_mode_button.setText("Show Raw Values")
        else:
            self.display_mode = "raw"
            self.toggle_mode_button.setText("Show Differences")
        self.refresh_data()

    def on_toggle_hide(self):
        self.hide_numbers = not self.hide_numbers
        self.hide_button.setText("Show Numbers" if self.hide_numbers else "Hide Numbers")
        self.refresh_data()

    def refresh_data(self):
        try:
            raw_list = self.sensor_api.read_raw()
        except Exception as e:
            self.table.setRowCount(1)
            self.table.setColumnCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(f"Error: {e}"))
            return

        n = len(raw_list)
        if n == 0:
            self.table.setRowCount(0)
            return

        if len(self.calibration_data) == n:
            diff_list = [abs(raw_list[i] - self.calibration_data[i]) for i in range(n)]
            # diff_list = [raw_list[i] - self.calibration_data[i] for i in range(n)]

        else:
            diff_list = [0] * n

        if not self.initial_diffs:
            self.initial_diffs = [[] for _ in range(n)]

        for idx in range(n):
            if len(self.initial_diffs[idx]) < 10:
                self.initial_diffs[idx].append(diff_list[idx])
                if len(self.initial_diffs[idx]) == 10:
                    max_val = max(self.initial_diffs[idx])
                    self.threshold_max[idx] = max_val + threshold_offset
                    if self.cells_remaining_for_threshold is not None:
                        self.cells_remaining_for_threshold -= 1
                        if self.cells_remaining_for_threshold == 0:
                            self.update_cal_button.setStyleSheet("background-color: lightgreen;")

        if self.display_mode == 'diff':
            display_list = diff_list[:]
        elif self.display_mode == 'cal':
            display_list = self.calibration_data[:]
        else:
            display_list = raw_list[:]

        rows, columns = self.table_rows, self.table_columns
        if self.table.rowCount() != rows or self.table.columnCount() != columns:
            self.table.setRowCount(rows)
            self.table.setColumnCount(columns)

        idx = 0
        for col in range(columns):
            for row in range(rows):
                if idx < n:
                    diff = diff_list[idx]
                    t_max = self.threshold_max.get(idx, float('inf'))

                    if diff <= t_max:
                        color = QColor(255, 255, 255)
                    else:
                        difference = diff - t_max
                        intensity_ratio = difference / max_value
                        red_intensity = max(0, min(255, int(intensity_ratio * 255)))
                        color = QColor(255, 255 - red_intensity, 255 - red_intensity)
                        # Calculate the difference from the threshold

                        # difference = diff - t_max
                        # # Determine the intensity of the gray color based on the difference
                        # intensity_ratio = difference / max_value
                        # gray_intensity = max(0, min(255, int(intensity_ratio * 255)))
                        # # Calculate the color value for grayscale (from white to black)
                        # color_value = 255 - gray_intensity
                        # # Set all three channels (R, G, B) to the same value for grayscale
                        # color = QColor(color_value, color_value, color_value)

                    item = self.table.item(row, col)
                    if item is None:
                        item = QTableWidgetItem()
                        self.table.setItem(row, col, item)

                    item.setData(Qt.DisplayRole, display_list[idx])
                    if idx < len(self.calibration_data):
                        item.setData(CALIBRATION_ROLE, self.calibration_data[idx])
                    item.setBackground(QBrush(color))
                else:
                    self.table.setItem(row, col, QTableWidgetItem())
                idx += 1

    def on_cell_clicked(self, row, column):
        flat_index = column * self.table_rows + row
        if self.selected_index == flat_index:
            self.table.clearSelection()
            self.selected_index = None
            self.info_label.setText("Click any cell to see its index")
            self.action_button.setEnabled(False)
        else:
            self.selected_index = flat_index
            self.info_label.setText(f"Clicked cell index: {flat_index}")
            self.action_button.setEnabled(True)

    def showEvent(self, event):
        if self.timer and not self.timer.isActive():
            self.timer.start(0)
        super().showEvent(event)

    def hideEvent(self, event):
        if self.timer and self.timer.isActive():
            self.timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event):
        if self.timer and self.timer.isActive():
            self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    sys.exit(app.exec_())
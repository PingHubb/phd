import sys

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter


CALIBRATION_ROLE = Qt.UserRole + 1
THRESHOLD_OFFSET = 30
MAX_VALUE = 100
REFRESH_INTERVAL_MS = 30
THRESHOLD_SAMPLE_COUNT = 10


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
        layout.addWidget(QLabel("Available serial ports:"))

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.populate_ports()

    def _get_port_names(self):
        if serial is None:
            return []

        try:
            ports = list(serial.tools.list_ports.comports())
        except Exception:
            return []

        all_port_names = [p.device for p in ports]
        acm_ports = sorted(name for name in all_port_names if "ttyACM" in name)
        other_ports = sorted(name for name in all_port_names if "ttyACM" not in name)
        return acm_ports + other_ports

    def populate_ports(self):
        sorted_ports = self._get_port_names()

        for port_name in sorted_ports:
            self.list_widget.addItem(QListWidgetItem(port_name))

        if not sorted_ports:
            placeholder = QListWidgetItem("— no serial ports found —")
            placeholder.setFlags(Qt.NoItemFlags)
            self.list_widget.addItem(placeholder)
        else:
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

        self.sensor_functions = sensor_functions_ref
        self.sensor_api = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)

        self.table_rows = 1
        self.table_columns = 1
        self.calibration_data = []
        self.display_mode = "raw"
        self.hide_numbers = True
        self.initial_diffs = []
        self.threshold_max = {}
        self.cells_remaining_for_threshold = None
        self.selected_index = None

        self._build_ui()
        self._connect_sensor()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("Sensor Signal Viewer")
        self.resize(1600, 800)

        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setItemDelegate(CellDelegate(self))
        self.table.cellClicked.connect(self.on_cell_clicked)

        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        font = QFont()
        font.setPointSize(10)
        self.table.setFont(font)

        layout.addWidget(self.table)
        layout.addWidget(self._build_toolbar())

    def _build_toolbar(self):
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

        self.action_button = QPushButton("Show Selected")
        self.action_button.setEnabled(False)
        self.action_button.clicked.connect(self.on_action_clicked)
        info_layout.addWidget(self.action_button)

        self.hide_button = QPushButton("Show Numbers")
        self.hide_button.clicked.connect(self.on_toggle_hide)
        info_layout.addWidget(self.hide_button)

        return info_container

    # ------------------------------------------------------------------
    # Sensor connection
    # ------------------------------------------------------------------
    def _choose_serial_port(self):
        if serial is None:
            QMessageBox.critical(
                self,
                "pyserial Missing",
                "pyserial is not installed, so the sensor viewer cannot open a serial connection.",
            )
            return None

        dialog = SerialPortDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_selected_port()
        return None

    def _connect_sensor(self):
        port = self._choose_serial_port()
        if not port:
            QMessageBox.information(
                self,
                "No Port Selected",
                "No serial port was selected. The sensor viewer will now close.",
            )
            QTimer.singleShot(0, self.close)
            return

        try:
            from phd.dependence.sensor_api import ArduinoCommander
            self.sensor_api = ArduinoCommander(serial_port=port, baud_rate=9600)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Serial Connection Error",
                f"Failed to open serial port {port}:\n{exc}",
            )
            self.sensor_api = None
            QTimer.singleShot(0, self.close)
            return

        if hasattr(self.sensor_api, "is_connected") and not self.sensor_api.is_connected():
            QMessageBox.critical(
                self,
                "Serial Connection Error",
                f"Failed to open serial port {port}.",
            )
            self.sensor_api = None
            QTimer.singleShot(0, self.close)
            return

        self._initialize_sensor_dimensions()
        self._initialize_calibration()

    def _initialize_sensor_dimensions(self):
        try:
            data_list = self.sensor_api.channel_check() or []
            if len(data_list) >= 2:
                self.table_columns = max(1, int(data_list[0]) - 1)
                self.table_rows = max(1, int(data_list[1]))
            else:
                raise ValueError(f"Unexpected channel_check result: {data_list}")
        except Exception as exc:
            print(f"[SensorSignalWindow] channel_check() failed: {exc}")
            self.table_columns = 1
            self.table_rows = 1

    def _initialize_calibration(self):
        try:
            self.calibration_data = list(self.sensor_api.update_cal() or [])
        except Exception as exc:
            self.calibration_data = []
            print(f"[SensorSignalWindow] Failed to update calibration on init: {exc}")

        if self.calibration_data:
            self.cells_remaining_for_threshold = len(self.calibration_data)

    # ------------------------------------------------------------------
    # UI actions
    # ------------------------------------------------------------------
    def on_update_calibration(self):
        if not self.sensor_api:
            self.info_label.setText("Calibration unavailable: sensor not connected")
            return

        try:
            new_cal = self.sensor_api.update_cal() or []
            self.calibration_data = list(new_cal)
            self.info_label.setText(f"Calibration updated ({len(self.calibration_data)} values)")
            self.initial_diffs.clear()
            self.threshold_max.clear()
            self.cells_remaining_for_threshold = len(self.calibration_data) if self.calibration_data else None
            self.update_cal_button.setStyleSheet("")

            if self.sensor_functions:
                update_fn = getattr(self.sensor_functions, "updateCal", None)
                if callable(update_fn):
                    update_fn()
        except Exception as exc:
            self.info_label.setText(f"Calibration error: {exc}")

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
        self.table.viewport().update()

    def on_action_clicked(self):
        if self.selected_index is None:
            return
        QMessageBox.information(self, "Selected Cell", f"Selected flat index: {self.selected_index}")

    def on_cell_clicked(self, row, column):
        flat_index = column * self.table_rows + row
        if self.selected_index == flat_index:
            self.table.clearSelection()
            self.selected_index = None
            self.info_label.setText("Click any cell to see its index")
            self.action_button.setEnabled(False)
            return

        self.selected_index = flat_index
        self.info_label.setText(f"Clicked cell index: {flat_index}")
        self.action_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Data refresh / rendering
    # ------------------------------------------------------------------
    def refresh_data(self):
        if not self.sensor_api:
            return

        try:
            raw_list = self.sensor_api.read_raw() or []
        except Exception as exc:
            self._show_error_cell(f"Error: {exc}")
            return

        n = len(raw_list)
        if n == 0:
            self.table.setRowCount(0)
            return

        diff_list = self._build_diff_list(raw_list)
        self._update_thresholds(diff_list)

        if self.display_mode == "diff":
            display_list = diff_list
        elif self.display_mode == "cal":
            display_list = list(self.calibration_data)
        else:
            display_list = list(raw_list)

        self._render_table(display_list, diff_list, n)

    def _build_diff_list(self, raw_list):
        n = len(raw_list)
        if len(self.calibration_data) == n:
            return [abs(raw_list[i] - self.calibration_data[i]) for i in range(n)]
        return [0] * n

    def _update_thresholds(self, diff_list):
        n = len(diff_list)
        if len(self.initial_diffs) != n:
            self.initial_diffs = [[] for _ in range(n)]

        for idx, diff_value in enumerate(diff_list):
            if len(self.initial_diffs[idx]) >= THRESHOLD_SAMPLE_COUNT:
                continue

            self.initial_diffs[idx].append(diff_value)
            if len(self.initial_diffs[idx]) == THRESHOLD_SAMPLE_COUNT:
                max_val = max(self.initial_diffs[idx])
                self.threshold_max[idx] = max_val + THRESHOLD_OFFSET
                if self.cells_remaining_for_threshold is not None:
                    self.cells_remaining_for_threshold -= 1
                    if self.cells_remaining_for_threshold == 0:
                        self.update_cal_button.setStyleSheet("background-color: lightgreen;")

    def _render_table(self, display_list, diff_list, n):
        rows, columns = self.table_rows, self.table_columns
        if self.table.rowCount() != rows or self.table.columnCount() != columns:
            self.table.setRowCount(rows)
            self.table.setColumnCount(columns)

        idx = 0
        for col in range(columns):
            for row in range(rows):
                if idx < n:
                    item = self.table.item(row, col)
                    if item is None:
                        item = QTableWidgetItem()
                        self.table.setItem(row, col, item)

                    item.setData(Qt.DisplayRole, display_list[idx])
                    if idx < len(self.calibration_data):
                        item.setData(CALIBRATION_ROLE, self.calibration_data[idx])
                    else:
                        item.setData(CALIBRATION_ROLE, None)
                    item.setBackground(QBrush(self._color_for_diff(diff_list[idx], idx)))
                else:
                    self.table.setItem(row, col, QTableWidgetItem())
                idx += 1

        self.table.viewport().update()

    def _color_for_diff(self, diff, idx):
        threshold = self.threshold_max.get(idx, float("inf"))
        if diff <= threshold:
            return QColor(255, 255, 255)

        difference = diff - threshold
        intensity_ratio = difference / MAX_VALUE
        red_intensity = max(0, min(255, int(intensity_ratio * 255)))
        return QColor(255, 255 - red_intensity, 255 - red_intensity)

    def _show_error_cell(self, message):
        self.table.setRowCount(1)
        self.table.setColumnCount(1)
        self.table.setItem(0, 0, QTableWidgetItem(message))

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------
    def showEvent(self, event):
        if self.sensor_api and not self.timer.isActive():
            self.timer.start(REFRESH_INTERVAL_MS)
        super().showEvent(event)

    def hideEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()

        if self.sensor_api and hasattr(self.sensor_api, "shutdown"):
            try:
                self.sensor_api.shutdown()
            except Exception:
                pass

        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SensorSignalWindow()
    if not window.isHidden():
        window.show()
    sys.exit(app.exec_())

import os
import sys
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter
from phd.dependence.paths import resource_path


CALIBRATION_ROLE = Qt.UserRole + 1
THRESHOLD_OFFSET = 30
# Percentage-change at which the heatmap saturates to deep red. Slightly
# wider than the 3D plotter's default (~1.7 %) so the viewer feels less
# twitchy. Lower values = more sensitive (smaller change → deeper red).
DEFAULT_HEATMAP_SATURATION_PCT = 5.0
# Anything strictly below this % stays pure white (kills baseline noise).
DEFAULT_HEATMAP_NOISE_FLOOR_PCT = 0.5
# Slider granularity: we use integer ticks of 0.1 % for both sliders.
HEATMAP_SLIDER_SCALE = 10
HEATMAP_SLIDER_MIN = 1          # 0.1 %
HEATMAP_SLIDER_MAX = 200        # 20.0 %
HEATMAP_FLOOR_SLIDER_MIN = 0    # 0.0 %
HEATMAP_FLOOR_SLIDER_MAX = 100  # 10.0 %
REFRESH_INTERVAL_MS = 30
THRESHOLD_SAMPLE_COUNT = 10


class _SensorSignalReadWorker(QObject):
    raw_ready = pyqtSignal(int, list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        sensor_api,
        generation=0,
        interval_ms=REFRESH_INTERVAL_MS,
        response_timeout=0.5,
        idle_sleep_sec=0.002,
    ):
        super().__init__()
        self.sensor_api = sensor_api
        self.generation = int(generation)
        self.interval_sec = max(0.001, float(interval_ms) / 1000.0)
        self.response_timeout = max(0.05, float(response_timeout))
        self.idle_sleep_sec = max(0.0, float(idle_sleep_sec))
        self._running = False

    def stop(self):
        self._running = False

    @staticmethod
    def _parse_ints(text):
        values = []
        for token in str(text).split():
            try:
                values.append(int(token))
            except ValueError:
                continue
        return values

    def _read_raw(self):
        ser = getattr(self.sensor_api, "ser", None)
        if ser is None or not getattr(ser, "is_open", False):
            return None

        try:
            ser.write(b"readRaw\n")
        except Exception as exc:
            self.error.emit(f"write failed: {exc}")
            return None

        deadline = time.time() + self.response_timeout
        while self._running and time.time() < deadline:
            try:
                waiting = int(getattr(ser, "in_waiting", 0))
            except Exception as exc:
                self.error.emit(f"read failed: {exc}")
                return None

            if waiting <= 0:
                time.sleep(self.idle_sleep_sec)
                continue

            try:
                line = ser.readline().decode("utf-8", errors="ignore").rstrip()
            except Exception as exc:
                self.error.emit(f"line read failed: {exc}")
                return None

            if not line:
                continue

            data_list = self._parse_ints(line)
            if data_list:
                return data_list[2:-2] if len(data_list) >= 4 else data_list

        return None

    def run(self):
        self._running = True
        try:
            while self._running:
                started = time.perf_counter()
                raw_list = self._read_raw()
                if raw_list is not None:
                    self.raw_ready.emit(self.generation, list(raw_list))

                elapsed = time.perf_counter() - started
                sleep_sec = max(0.0, self.interval_sec - elapsed)
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
        finally:
            self.finished.emit()


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
        self._using_shared_sensor_data = False
        self._shared_sensor_functions_ref = None
        self._shared_calibration_overridden = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_data)
        self._reader_thread = None
        self._reader_worker = None
        self._reader_generation = 0
        self._latest_raw_list = []
        self._last_reader_error_log_time = 0.0

        self.table_rows = 1
        self.table_columns = 1
        self.calibration_data = []
        self.display_mode = "raw"
        self.hide_numbers = True
        self.initial_diffs = []
        self.threshold_max = {}
        self.cells_remaining_for_threshold = None
        self.selected_index = None
        self._heatmap_saturation_pct = float(DEFAULT_HEATMAP_SATURATION_PCT)
        self._heatmap_noise_floor_pct = float(DEFAULT_HEATMAP_NOISE_FLOOR_PCT)

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

        gain_row = QWidget()
        gain_layout = QHBoxLayout(gain_row)
        gain_layout.setContentsMargins(0, 4, 0, 0)
        gain_layout.setSpacing(10)
        gain_layout.addWidget(
            QLabel("Heatmap gain — % change for full red:")
        )
        self._heatmap_span_slider = QSlider(Qt.Horizontal)
        self._heatmap_span_slider.setMinimum(HEATMAP_SLIDER_MIN)
        self._heatmap_span_slider.setMaximum(HEATMAP_SLIDER_MAX)
        self._heatmap_span_slider.setSingleStep(1)
        self._heatmap_span_slider.setPageStep(5)
        self._heatmap_span_slider.setValue(
            int(round(DEFAULT_HEATMAP_SATURATION_PCT * HEATMAP_SLIDER_SCALE))
        )
        self._heatmap_span_slider.setToolTip(
            "Upper end of the heatmap. Cells whose |raw − cal|/|cal|×100\n"
            "reaches this value are full deep-red.  Lower value = saturate\n"
            "sooner (more sensitive).  Default 5.0 %."
        )
        self._heatmap_span_slider.valueChanged.connect(self._on_heatmap_span_changed)
        gain_layout.addWidget(self._heatmap_span_slider, stretch=1)
        self._heatmap_span_value_label = QLabel(
            f"{DEFAULT_HEATMAP_SATURATION_PCT:.1f} %"
        )
        self._heatmap_span_value_label.setMinimumWidth(48)
        self._heatmap_span_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        gain_layout.addWidget(self._heatmap_span_value_label)
        layout.addWidget(gain_row)

        floor_row = QWidget()
        floor_layout = QHBoxLayout(floor_row)
        floor_layout.setContentsMargins(0, 0, 0, 0)
        floor_layout.setSpacing(10)
        floor_layout.addWidget(
            QLabel("Noise floor — % change below this stays white:")
        )
        self._heatmap_floor_slider = QSlider(Qt.Horizontal)
        self._heatmap_floor_slider.setMinimum(HEATMAP_FLOOR_SLIDER_MIN)
        self._heatmap_floor_slider.setMaximum(HEATMAP_FLOOR_SLIDER_MAX)
        self._heatmap_floor_slider.setSingleStep(1)
        self._heatmap_floor_slider.setPageStep(5)
        self._heatmap_floor_slider.setValue(
            int(round(DEFAULT_HEATMAP_NOISE_FLOOR_PCT * HEATMAP_SLIDER_SCALE))
        )
        self._heatmap_floor_slider.setToolTip(
            "Lower edge of the heatmap.  Cells whose % change is below this\n"
            "value are forced to plain white (kills idle / breathing noise).\n"
            "Above this floor the colour grows from pale pink to deep red."
        )
        self._heatmap_floor_slider.valueChanged.connect(self._on_heatmap_floor_changed)
        floor_layout.addWidget(self._heatmap_floor_slider, stretch=1)
        self._heatmap_floor_value_label = QLabel(
            f"{DEFAULT_HEATMAP_NOISE_FLOOR_PCT:.1f} %"
        )
        self._heatmap_floor_value_label.setMinimumWidth(48)
        self._heatmap_floor_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        floor_layout.addWidget(self._heatmap_floor_value_label)
        layout.addWidget(floor_row)

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

        self.save_png_button = QPushButton("Save Heatmap PNG")
        self.save_png_button.setToolTip(
            "Capture the current heatmap as a PNG.\n"
            "Masked (grey) cells are written as fully transparent pixels so\n"
            "they cleanly drop out when you overlay the screenshot elsewhere."
        )
        self.save_png_button.clicked.connect(self.on_save_heatmap_png)
        info_layout.addWidget(self.save_png_button)

        return info_container

    def _on_heatmap_span_changed(self, value: int):
        pct = max(1e-6, float(value) / HEATMAP_SLIDER_SCALE)
        self._heatmap_saturation_pct = pct
        if hasattr(self, "_heatmap_span_value_label") and self._heatmap_span_value_label:
            self._heatmap_span_value_label.setText(f"{pct:.1f} %")
        if self._has_sensor_source():
            self.refresh_data()
        else:
            self.table.viewport().update()

    def _on_heatmap_floor_changed(self, value: int):
        pct = max(0.0, float(value) / HEATMAP_SLIDER_SCALE)
        self._heatmap_noise_floor_pct = pct
        if hasattr(self, "_heatmap_floor_value_label") and self._heatmap_floor_value_label:
            self._heatmap_floor_value_label.setText(f"{pct:.1f} %")
        if self._has_sensor_source():
            self.refresh_data()
        else:
            self.table.viewport().update()

    # ------------------------------------------------------------------
    # Screenshot export
    # ------------------------------------------------------------------
    def on_save_heatmap_png(self):
        """Render the current heatmap into a transparent-background PNG.

        Each cell becomes a solid-coloured square using the same gain/floor
        ramp as the live view.  Masked (grey) cells are written as fully
        transparent pixels so the export can be overlaid on top of e.g. a
        photo without the grey checkerboard bleeding through.

        The 30 ms redraw timer is paused while the save dialog is open so the
        table does not repaint under the capture flow.
        """
        if not self._has_sensor_source():
            self.info_label.setText("Save Heatmap PNG: sensor not connected.")
            print("[SensorSignalWindow] Save Heatmap PNG aborted — no sensor.")
            return

        timer_was_active = bool(self.timer.isActive())
        if timer_was_active:
            self.timer.stop()
        try:
            self._save_heatmap_png_impl()
        finally:
            if timer_was_active:
                self.timer.start(REFRESH_INTERVAL_MS)

    def _save_heatmap_png_impl(self):
        try:
            raw_list = list(self._current_raw_list() or [])
        except Exception as exc:
            self.info_label.setText(f"Save Heatmap PNG read failed: {exc}")
            print(f"[SensorSignalWindow] Save Heatmap PNG read failed: {exc}")
            return

        n = len(raw_list)
        if n == 0:
            self.info_label.setText("Save Heatmap PNG: sensor returned no data.")
            print("[SensorSignalWindow] Save Heatmap PNG: empty payload.")
            return

        rows = max(1, int(self.table_rows))
        columns = max(1, int(self.table_columns))
        percent_list = self._build_percent_list(raw_list)
        zero_mask = self._get_zero_mask_matrix(rows, columns)

        # Prefer a project-local screenshots folder so saved PNGs are easy to
        # find. We try the workspace path first, then fall back to ~/. The
        # most-recently-used directory is remembered across saves in this
        # session via ``self._last_screenshot_dir``.
        project_dir = resource_path("sensor_screenshots")
        last_dir = getattr(self, "_last_screenshot_dir", None)
        if last_dir and os.path.isdir(last_dir):
            default_dir = last_dir
        else:
            default_dir = project_dir
            try:
                os.makedirs(default_dir, exist_ok=True)
            except Exception:
                default_dir = os.path.expanduser("~")

        default_name = time.strftime("sensor_heatmap_%Y%m%d_%H%M%S.png")
        default_path = os.path.join(default_dir, default_name)
        print(f"[SensorSignalWindow] Save dialog default path: {default_path}")

        # Build the file dialog manually so we can keep it on top of the main
        # window. ``getSaveFileName`` sometimes spawns a native dialog that
        # gets reparented behind a transient parent on some desktop envs —
        # using a Qt dialog with explicit window flags is rock-solid.
        dialog = QFileDialog(self, "Save Heatmap PNG", default_dir)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("PNG image (*.png)")
        dialog.setDefaultSuffix("png")
        dialog.selectFile(default_name)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec_() != QFileDialog.Accepted:
            print("[SensorSignalWindow] Save Heatmap PNG cancelled (no path).")
            return
        selected = dialog.selectedFiles()
        path = selected[0] if selected else ""
        if not path:
            print("[SensorSignalWindow] Save Heatmap PNG cancelled (no path).")
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        # Make sure the destination dir exists (the user may have typed in a
        # new sub-folder).
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass
        self._last_screenshot_dir = os.path.dirname(path) or default_dir

        cell_px = 64
        width_px = max(1, columns * cell_px)
        height_px = max(1, rows * cell_px)
        image = QImage(width_px, height_px, QImage.Format_ARGB32)
        image.fill(QColor(0, 0, 0, 0))  # fully transparent canvas

        painter = QPainter(image)
        try:
            painter.setPen(Qt.NoPen)
            idx = 0
            for col in range(columns):
                for row in range(rows):
                    if idx >= n:
                        idx += 1
                        continue
                    is_masked = (
                        bool(zero_mask[row, col]) if zero_mask is not None else False
                    )
                    if not is_masked:
                        pct = percent_list[idx] if idx < len(percent_list) else 0.0
                        colour = self._color_for_cell(pct)
                        painter.fillRect(
                            col * cell_px,
                            row * cell_px,
                            cell_px,
                            cell_px,
                            colour,
                        )
                    idx += 1

            # Light grid overlay so cell boundaries stay visible without
            # competing with the heatmap colours. Drawn last so it sits on
            # top of every cell fill but stays subtle (semi-transparent grey).
            #
            # Masked (grey) cells are skipped entirely: any grid edge that
            # would touch a masked cell is omitted, so the transparent square
            # stays completely free of overlay marks.
            from PyQt5.QtGui import QPen
            grid_pen = QPen(QColor(70, 70, 70, 140))
            grid_pen.setWidth(1)
            grid_pen.setCosmetic(True)
            painter.setPen(grid_pen)

            def _is_real_non_masked(r: int, c: int) -> bool:
                """True iff (r, c) is an in-bounds, non-masked cell. Out-of-
                bounds positions count as "not a real cell" (False) so the
                outer border of a masked cell does not produce a stray edge."""
                if r < 0 or r >= rows or c < 0 or c >= columns:
                    return False
                if zero_mask is None:
                    return True
                return not bool(zero_mask[r, c])

            # Vertical lines: a segment at column boundary x runs the height
            # of one row. Draw the segment when AT LEAST ONE adjacent in-
            # bounds cell is non-masked — this keeps the edge between a non-
            # masked and a masked cell visible, while dropping lines that
            # would otherwise sit on the outside of an isolated masked cell.
            for c in range(columns + 1):
                x = min(c * cell_px, width_px - 1)
                for r in range(rows):
                    if not (_is_real_non_masked(r, c - 1) or _is_real_non_masked(r, c)):
                        continue
                    y0 = r * cell_px
                    y1 = (r + 1) * cell_px
                    painter.drawLine(x, y0, x, y1)

            # Horizontal lines: same idea, per column segment.
            for r in range(rows + 1):
                y = min(r * cell_px, height_px - 1)
                for c in range(columns):
                    if not (_is_real_non_masked(r - 1, c) or _is_real_non_masked(r, c)):
                        continue
                    x0 = c * cell_px
                    x1 = (c + 1) * cell_px
                    painter.drawLine(x0, y, x1, y)
        finally:
            painter.end()

        if not image.save(path, "PNG"):
            print(f"[SensorSignalWindow] image.save() returned False for {path}")
            self.info_label.setText(
                f"Save Heatmap PNG failed (write error): {path}"
            )
            return
        # Confirm to both UI + console so the user can always find it.
        print(f"[SensorSignalWindow] Heatmap saved -> {path}")
        self.info_label.setText(f"Heatmap saved → {path}")

    # ------------------------------------------------------------------
    # Sensor connection
    # ------------------------------------------------------------------
    def _try_connect_shared_sensor_source(self):
        sensor_functions = self._resolve_sensor_functions()
        if sensor_functions is None:
            return False

        data_obj = getattr(sensor_functions, "_data", None)
        n_row = int(getattr(sensor_functions, "n_row", 0) or 0)
        n_col = int(getattr(sensor_functions, "n_col", 0) or 0)
        is_connected = bool(getattr(sensor_functions, "is_connected", False))
        if not is_connected or data_obj is None or n_row <= 0 or n_col <= 0:
            return False

        self._using_shared_sensor_data = True
        self._shared_sensor_functions_ref = sensor_functions
        self._shared_calibration_overridden = False
        self.sensor_api = None
        self.table_rows = max(1, n_row)
        self.table_columns = max(1, n_col)
        self._sync_shared_calibration()
        self.info_label.setText("Using live sensor stream")
        return True

    def _has_sensor_source(self):
        return bool(self._using_shared_sensor_data or self.sensor_api)

    def _flatten_sensor_matrix_column_major(self, matrix, rows, columns):
        try:
            import numpy as _np
            arr = _np.asarray(matrix)
        except Exception:
            return []

        if arr.ndim != 2 or arr.shape[0] < rows or arr.shape[1] < columns:
            return []

        values = []
        for col in range(columns):
            for row in range(rows):
                value = arr[row, col]
                values.append(value.item() if hasattr(value, "item") else value)
        return values

    def _get_shared_sensor_functions(self):
        sensor_functions = self._resolve_sensor_functions()
        if sensor_functions is not None and getattr(sensor_functions, "_data", None) is not None:
            self._shared_sensor_functions_ref = sensor_functions
            return sensor_functions
        return self._shared_sensor_functions_ref

    def _sync_shared_calibration(self):
        if self._shared_calibration_overridden:
            return

        sensor_functions = self._get_shared_sensor_functions()
        data_obj = getattr(sensor_functions, "_data", None) if sensor_functions is not None else None
        if data_obj is None:
            return

        rows = max(1, int(getattr(sensor_functions, "n_row", self.table_rows) or self.table_rows))
        columns = max(1, int(getattr(sensor_functions, "n_col", self.table_columns) or self.table_columns))
        cal_matrix = getattr(data_obj, "calData", None)
        cal_values = self._flatten_sensor_matrix_column_major(cal_matrix, rows, columns)
        if cal_values and cal_values != self.calibration_data:
            self.calibration_data = cal_values
            self.cells_remaining_for_threshold = len(self.calibration_data)

    def _read_shared_raw_list(self):
        sensor_functions = self._get_shared_sensor_functions()
        data_obj = getattr(sensor_functions, "_data", None) if sensor_functions is not None else None
        if data_obj is None:
            return []

        rows = max(1, int(getattr(sensor_functions, "n_row", self.table_rows) or self.table_rows))
        columns = max(1, int(getattr(sensor_functions, "n_col", self.table_columns) or self.table_columns))
        self.table_rows = rows
        self.table_columns = columns
        self._sync_shared_calibration()

        raw_matrix = getattr(data_obj, "rawData", None)
        return self._flatten_sensor_matrix_column_major(raw_matrix, rows, columns)

    def _current_raw_list(self):
        if self._using_shared_sensor_data:
            raw_list = self._read_shared_raw_list()
            if raw_list:
                self._latest_raw_list = list(raw_list)
            return raw_list
        return list(self._latest_raw_list or [])

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
        if self._try_connect_shared_sensor_source():
            return

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
    # Background sensor reader
    # ------------------------------------------------------------------
    def _reader_is_running(self):
        thread = getattr(self, "_reader_thread", None)
        return bool(thread is not None and thread.isRunning())

    def _start_reader_worker(self):
        if not self.sensor_api or self._reader_is_running():
            return

        self._reader_generation += 1
        generation = self._reader_generation
        self._latest_raw_list = []

        thread = QThread(self)
        worker = _SensorSignalReadWorker(
            self.sensor_api,
            generation=generation,
            interval_ms=REFRESH_INTERVAL_MS,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.raw_ready.connect(self._on_reader_raw_ready)
        worker.error.connect(self._on_reader_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda thread=thread, worker=worker: self._on_reader_thread_finished(thread, worker)
        )

        self._reader_thread = thread
        self._reader_worker = worker
        thread.start()

    def _stop_reader_worker(self):
        self._reader_generation += 1

        worker = getattr(self, "_reader_worker", None)
        thread = getattr(self, "_reader_thread", None)

        if worker is not None:
            worker.stop()

        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(1000)

        self._reader_worker = None
        self._reader_thread = None

    def _on_reader_thread_finished(self, thread=None, worker=None):
        if thread is not None and self._reader_thread is not thread:
            return
        if worker is not None and self._reader_worker is not worker:
            return
        self._reader_worker = None
        self._reader_thread = None

    def _on_reader_raw_ready(self, generation, raw_list):
        if int(generation) != int(self._reader_generation):
            return
        self._latest_raw_list = list(raw_list or [])

    def _on_reader_error(self, message):
        now = time.monotonic()
        if now - self._last_reader_error_log_time < 2.0:
            return
        self._last_reader_error_log_time = now
        print(f"[SensorSignalWindow] Sensor read error: {message}")

    # ------------------------------------------------------------------
    # UI actions
    # ------------------------------------------------------------------
    def on_update_calibration(self):
        if not self._has_sensor_source():
            self.info_label.setText("Calibration unavailable: sensor not connected")
            return

        if self._using_shared_sensor_data:
            timer_was_active = bool(self.timer.isActive())
            if timer_was_active:
                self.timer.stop()
            try:
                raw_list = list(self._read_shared_raw_list() or [])
                if not raw_list:
                    self.info_label.setText("Calibration unavailable: no live sensor data")
                    return
                self.calibration_data = raw_list
                self._shared_calibration_overridden = True
                self.initial_diffs.clear()
                self.threshold_max.clear()
                self.cells_remaining_for_threshold = len(self.calibration_data)
                self.update_cal_button.setStyleSheet("")
                self.info_label.setText(f"Heatmap baseline updated ({len(self.calibration_data)} values)")
            except Exception as exc:
                self.info_label.setText(f"Calibration error: {exc}")
            finally:
                if timer_was_active:
                    self.timer.start(REFRESH_INTERVAL_MS)
            return

        reader_was_running = self._reader_is_running()
        self._stop_reader_worker()
        try:
            new_cal = self.sensor_api.update_cal() or []
            self.calibration_data = list(new_cal)
            if not self.calibration_data:
                self.info_label.setText("Calibration error: sensor returned no values")
                return

            self.info_label.setText(f"Calibration updated ({len(self.calibration_data)} values)")
            self.initial_diffs.clear()
            self.threshold_max.clear()
            self.cells_remaining_for_threshold = len(self.calibration_data) if self.calibration_data else None
            self.update_cal_button.setStyleSheet("")
        except Exception as exc:
            self.info_label.setText(f"Calibration error: {exc}")
        finally:
            self._latest_raw_list = []
            if reader_was_running:
                self._start_reader_worker()

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
        if not self._has_sensor_source():
            return

        raw_list = list(self._current_raw_list() or [])
        if not raw_list:
            return

        n = len(raw_list)

        diff_list = self._build_diff_list(raw_list)
        percent_list = self._build_percent_list(raw_list)
        self._update_thresholds(diff_list)

        if self.display_mode == "diff":
            display_list = diff_list
        elif self.display_mode == "cal":
            display_list = list(self.calibration_data)
        else:
            display_list = list(raw_list)

        self._render_table(display_list, diff_list, percent_list, n)

    def _build_diff_list(self, raw_list):
        n = len(raw_list)
        if len(self.calibration_data) == n:
            return [abs(raw_list[i] - self.calibration_data[i]) for i in range(n)]
        return [0] * n

    def _build_percent_list(self, raw_list):
        """Per-cell |raw − cal| / cal × 100, matching ``data.calDiffPer`` so the
        heatmap saturates the same way as the 3D sensor plotter. Returns 0 for
        cells where calibration is zero or unavailable."""
        n = len(raw_list)
        if len(self.calibration_data) != n:
            return [0.0] * n
        out = [0.0] * n
        for i in range(n):
            cal = self.calibration_data[i]
            try:
                cal_val = float(cal)
            except Exception:
                cal_val = 0.0
            if cal_val == 0.0:
                continue
            try:
                raw_val = float(raw_list[i])
            except Exception:
                raw_val = 0.0
            out[i] = abs(raw_val - cal_val) / abs(cal_val) * 100.0
        return out

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

    def _render_table(self, display_list, diff_list, percent_list, n):
        rows, columns = self.table_rows, self.table_columns
        if self.table.rowCount() != rows or self.table.columnCount() != columns:
            self.table.setRowCount(rows)
            self.table.setColumnCount(columns)
            self._last_cell_states = {}

        # Dirty-check cache: only touch QTableWidgetItems whose content
        # actually changed. This runs at ~33 Hz over the whole grid, and
        # unchanged cells are the common case, so skipping them removes
        # thousands of redundant item updates per second.
        if not hasattr(self, "_last_cell_states"):
            self._last_cell_states = {}

        zero_mask = self._get_zero_mask_matrix(rows, columns)
        masked_brush = QBrush(QColor(210, 210, 210))  # neutral light gray
        masked_positions: list = []
        changed_any = False

        idx = 0
        for col in range(columns):
            for row in range(rows):
                if idx < n:
                    is_masked = bool(zero_mask[row, col]) if zero_mask is not None else False
                    if is_masked:
                        masked_positions.append((row, col))
                        state = ("masked",)
                    else:
                        cal = (
                            self.calibration_data[idx]
                            if idx < len(self.calibration_data)
                            else None
                        )
                        pct = percent_list[idx] if idx < len(percent_list) else 0.0
                        # Quantize the heat percentage so imperceptible color
                        # changes do not force a repaint.
                        state = (display_list[idx], cal, round(pct, 1))

                    if self._last_cell_states.get((row, col)) == state:
                        idx += 1
                        continue
                    self._last_cell_states[(row, col)] = state
                    changed_any = True

                    item = self.table.item(row, col)
                    if item is None:
                        item = QTableWidgetItem()
                        self.table.setItem(row, col, item)

                    if is_masked:
                        # Keep cell visible but neutral: gray background, dash
                        # placeholder. This makes which cells are masked very
                        # obvious so any mask misalignment is easy to spot.
                        item.setData(Qt.DisplayRole, "—")
                        item.setData(CALIBRATION_ROLE, None)
                        item.setBackground(masked_brush)
                    else:
                        item.setData(Qt.DisplayRole, state[0])
                        item.setData(CALIBRATION_ROLE, state[1])
                        item.setBackground(QBrush(self._color_for_cell(state[2])))
                else:
                    if self._last_cell_states.get((row, col)) != ("empty",):
                        self._last_cell_states[(row, col)] = ("empty",)
                        self.table.setItem(row, col, QTableWidgetItem())
                        changed_any = True
                idx += 1

        # Diagnostic: warn (once per change) when the number of cells the
        # viewer greyed out does not match the live mask's True count.
        if zero_mask is not None:
            try:
                expected = int(zero_mask.sum())
            except Exception:
                expected = -1
            actual = len(masked_positions)
            last = getattr(self, "_last_mask_mismatch", None)
            if expected >= 0 and expected != actual and last != (expected, actual):
                self._last_mask_mismatch = (expected, actual)
                print(
                    f"[SensorSignalWindow] zero-mask mismatch: mask has "
                    f"{expected} True cell(s) but viewer marked {actual}. "
                    f"viewer dims=({rows},{columns}); masked={masked_positions}"
                )
            elif expected == actual:
                self._last_mask_mismatch = None

        if changed_any:
            self.table.viewport().update()

    def _get_zero_mask_matrix(self, rows: int, columns: int):
        """Pull the latest ``cell_zero_mask`` from the shared MySensor (the same
        object the Sensor Zero Mask dialog writes to) and return a 2-D bool
        array sized exactly ``(rows, columns)`` so it can be indexed as
        ``mask[row, col]`` to match the viewer's table.

        We re-look-up the sensor object every refresh in case the user built
        / rebuilt the sensor scene *after* opening the viewer (otherwise we
        could still be holding the original ``_FeatureDisabledProxy``).

        Returns ``None`` if the mask is missing, all-False, or cannot be
        reconciled with the viewer dimensions.
        """
        sensor_functions = self._resolve_sensor_functions()
        if sensor_functions is None:
            return None
        getter = getattr(sensor_functions, "get_cell_zero_mask", None)
        if not callable(getter):
            return None
        try:
            mask = getter()
        except Exception:
            return None
        try:
            import numpy as _np
            arr = _np.asarray(mask, dtype=bool)
        except Exception:
            return None
        if arr.ndim != 2 or arr.size == 0:
            return None
        if not arr.any():
            return None
        if arr.shape == (rows, columns):
            return arr
        if arr.shape == (columns, rows):
            return arr.T.copy()
        return None

    def _resolve_sensor_functions(self):
        """Re-look up the live MySensor each call so a build that happened
        after the viewer opened (or a swap to a different sensor) is honoured.

        Falls back to the originally-injected ``sensor_functions_ref``."""
        parent = self.parent()
        if parent is not None:
            ui_ros = getattr(parent, "ui_ros", None)
            if ui_ros is not None:
                sf = getattr(ui_ros, "sensor_functions", None)
                if sf is not None:
                    return sf
        return self.sensor_functions

    def _color_for_cell(self, percent: float) -> QColor:
        """Smooth pale-pink → deep-red ramp keyed to the *relative* change
        ``|raw−cal|/|cal|×100``.

        Two user knobs control the response:
          * noise-floor %  — anything below stays white (no colour at all),
          * gain %         — value at which the ramp reaches full deep red.
        Between the two the colour grows continuously, so small changes show
        up as light pink instead of jumping straight to red.
        """
        abs_pct = abs(float(percent))
        span_pct = max(
            1e-6,
            float(
                getattr(self, "_heatmap_saturation_pct", DEFAULT_HEATMAP_SATURATION_PCT)
            ),
        )
        floor_pct = max(
            0.0,
            float(
                getattr(
                    self,
                    "_heatmap_noise_floor_pct",
                    DEFAULT_HEATMAP_NOISE_FLOOR_PCT,
                )
            ),
        )
        # Stay pure white below the noise floor.
        if abs_pct <= floor_pct:
            return QColor(255, 255, 255)

        # If the user dragged the floor at or above the gain, treat anything
        # above the floor as full red — otherwise we'd divide by ~0.
        usable_span = max(1e-6, span_pct - floor_pct)
        # 0 at the floor → 1 at the gain (clamped).
        red_strength = max(0.0, min(1.0, (abs_pct - floor_pct) / usable_span))
        # Same colour family as the 3D plotter: red stays 255, green/blue drop.
        g = int(round(255.0 * (1.0 - red_strength)))
        return QColor(255, g, g)

    def _show_error_cell(self, message):
        self.table.setRowCount(1)
        self.table.setColumnCount(1)
        self.table.setItem(0, 0, QTableWidgetItem(message))

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------
    def showEvent(self, event):
        if self._using_shared_sensor_data:
            if not self.timer.isActive():
                self.timer.start(REFRESH_INTERVAL_MS)
        elif self.sensor_api:
            self._start_reader_worker()
            if not self.timer.isActive():
                self.timer.start(REFRESH_INTERVAL_MS)
        super().showEvent(event)

    def hideEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        self._stop_reader_worker()
        super().hideEvent(event)

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        self._stop_reader_worker()

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

import copy
import json
import os
import time
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, QSize, QEvent, QSettings
from PyQt5.QtWidgets import (
    QWidget, QAction, QSplitter, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QToolBar, QStatusBar, QComboBox,
    QFrame, QMessageBox, QStyle, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QGridLayout, QToolButton, QSizePolicy, QMenu, QDoubleSpinBox, QFormLayout, QSpinBox,
    QCheckBox, QApplication, QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox,
    QFileDialog, QScrollArea, QTabWidget, QTreeWidget, QTreeWidgetItem,
)
from PyQt5.QtGui import QIcon, QColor, QPainter, QPen, QPainterPath, QPixmap, QFont
import numpy as np
from pyvistaqt import QtInteractor, MainWindow
from phd.dependence.paths import icon_path, resource_path, stylesheet_path
from phd.ui import components, icons, theme
from phd.ui.ui_ping import UI
from phd.ui import experiment_tasks
from phd.ui.calibration_graph_analysis import (
    CalibrationGraphAnalysisDialog,
    load_calibration_recording,
)
from phd.ui.calibration_result_dialog import (
    CalibrationResultDialog,
    CalibrationTaxelMap,
    FullTaxelCalibrationResultDialog,
)
from phd.ui.taxel_position_navigation import MappedTaxelMoveTask, prepare_taxel_map
from phd.ui.camera_preview_window import CameraPreviewWindow
from phd.ui.plotter_video_recorder import PlotterVideoRecorder
from phd.ui.heatmap_signal_recording import (
    HeatmapSignalPlayback,
    HeatmapSignalRecorder,
)
from phd.dependence.sensor_signal_window import SensorSignalWindow


SIGRAPH_HEATMAP_RECORD_FPS = 60.0


class Resources:
    """
    A helper class to manage and provide access to application resources.
    This centralizes the hardcoded paths for icons and stylesheets,
    making it easy to update them in one place if they ever change.
    """
    STYLE_FILE = stylesheet_path("ui_style.qss")

    def get_icon(self, name: str) -> QIcon:
        """Loads a QIcon from the predefined icon directory."""
        path = icon_path(name)
        if not os.path.exists(path):
            return QIcon()
        return QIcon(path)

    def get_stylesheet(self) -> str:
        """Loads the content of the QSS stylesheet."""
        try:
            with open(self.STYLE_FILE, 'r', encoding='utf-8') as f:
                sheet = f.read()
        except FileNotFoundError:
            print(f"Warning: Stylesheet not found at '{self.STYLE_FILE}'.")
            # Return an empty string so the app can still run
            return ""

        generated_dir = icon_path("generated")
        icons.write_stylesheet_assets(generated_dir)
        try:
            return theme.render_stylesheet(
                sheet,
                ICON_DIR=generated_dir.replace(os.sep, "/"),
            )
        except ValueError as exc:
            print(f"Warning: Stylesheet token error: {exc}")
            return ""


class SimpleLineChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.series = []
        self.setMinimumHeight(320)

    def set_series(self, series):
        self.series = list(series or [])
        self.update()

    def _format_value(self, value: float) -> str:
        return f"{value:.3f}"

    def _build_linear_ticks(self, minimum: float, maximum: float, tick_count: int):
        if tick_count <= 1 or maximum <= minimum:
            return [minimum]
        step = (maximum - minimum) / float(tick_count - 1)
        return [minimum + step * idx for idx in range(tick_count)]

    def _build_smoothed_values(self, values, alpha=0.18):
        if not values:
            return []
        smoothed = [float(values[0])]
        for value in values[1:]:
            smoothed.append(alpha * float(value) + (1.0 - alpha) * smoothed[-1])
        return smoothed

    def _points_from_series(self, times, values, min_t, max_t, min_v, max_v, plot_rect):
        points = []
        for time_s, value in zip(times, values):
            x_ratio = (float(time_s) - min_t) / (max_t - min_t)
            y_ratio = (float(value) - min_v) / (max_v - min_v)
            x = plot_rect.left() + x_ratio * plot_rect.width()
            y = plot_rect.bottom() - y_ratio * plot_rect.height()
            points.append((x, y))
        return points

    def _path_from_points(self, points):
        path = QPainterPath()
        path.moveTo(points[0][0], points[0][1])
        for idx in range(1, len(points)):
            path.lineTo(points[idx][0], points[idx][1])
        return path

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme.INPUT_BG))

        left = 72
        right = 24
        top = 36
        bottom = 48
        plot_rect = self.rect().adjusted(left, top, -right, -bottom)
        if plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        axis_pen = QPen(QColor("#6f7682"))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)
        painter.drawRect(plot_rect)

        if len(self.series) < 2:
            painter.setPen(QColor("#cccccc"))
            painter.drawText(plot_rect, Qt.AlignCenter, "Not enough samples to draw.")
            return

        times = [point[0] for point in self.series]
        values = [point[1] for point in self.series]
        smooth_values = self._build_smoothed_values(values)
        min_t = float(min(times))
        max_t = float(max(times))
        min_v = float(min(min(values), min(smooth_values)))
        max_v = float(max(max(values), max(smooth_values)))
        if max_t <= min_t:
            max_t = min_t + 1.0
        if abs(max_v - min_v) <= 1e-12:
            max_v = min_v + 1.0

        y_padding = max((max_v - min_v) * 0.10, 1e-6)
        min_v -= y_padding
        max_v += y_padding

        x_ticks = self._build_linear_ticks(min_t, max_t, 6)
        y_ticks = self._build_linear_ticks(min_v, max_v, 5)

        grid_pen = QPen(QColor(theme.BORDER_SUBTLE))
        grid_pen.setWidth(1)
        grid_pen.setStyle(Qt.DashLine)
        painter.setPen(grid_pen)

        for tick in x_ticks:
            x_ratio = (tick - min_t) / (max_t - min_t)
            x = plot_rect.left() + x_ratio * plot_rect.width()
            painter.drawLine(int(x), plot_rect.top(), int(x), plot_rect.bottom())

        for tick in y_ticks:
            y_ratio = (tick - min_v) / (max_v - min_v)
            y = plot_rect.bottom() - y_ratio * plot_rect.height()
            painter.drawLine(plot_rect.left(), int(y), plot_rect.right(), int(y))

        points = self._points_from_series(times, values, min_t, max_t, min_v, max_v, plot_rect)
        smooth_points = self._points_from_series(
            times,
            smooth_values,
            min_t,
            max_t,
            min_v,
            max_v,
            plot_rect,
        )

        path = self._path_from_points(points)
        smooth_path = self._path_from_points(smooth_points)

        area_path = QPainterPath(path)
        area_path.lineTo(points[-1][0], plot_rect.bottom())
        area_path.lineTo(points[0][0], plot_rect.bottom())
        area_path.closeSubpath()
        painter.fillPath(area_path, QColor(79, 195, 247, 50))

        raw_pen = QPen(QColor("#4fc3f7"))
        raw_pen.setWidth(2)
        painter.setPen(raw_pen)
        painter.drawPath(path)

        smooth_pen = QPen(QColor("#ff8a65"))
        smooth_pen.setWidth(2)
        painter.setPen(smooth_pen)
        painter.drawPath(smooth_path)

        peak_order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
        top_peak_indices = []
        for idx in peak_order:
            if not top_peak_indices or all(abs(idx - existing) > 3 for existing in top_peak_indices):
                top_peak_indices.append(idx)
            if len(top_peak_indices) >= 3:
                break

        last_idx = len(values) - 1
        marker_pen = QPen(QColor("#ffffff"))
        marker_pen.setWidth(1)
        painter.setPen(marker_pen)

        peak_colors = [QColor("#ffb74d"), QColor("#ce93d8"), QColor("#90caf9")]
        for rank, peak_idx in enumerate(top_peak_indices):
            painter.setBrush(peak_colors[rank])
            painter.drawEllipse(int(points[peak_idx][0] - 4), int(points[peak_idx][1] - 4), 8, 8)

        painter.setBrush(QColor("#81c784"))
        painter.drawEllipse(int(points[last_idx][0] - 4), int(points[last_idx][1] - 4), 8, 8)

        painter.setPen(QColor("#dddddd"))
        painter.drawText(plot_rect.left(), 20, "Peak Sensor Change Over Time")

        legend_y = 18
        painter.setPen(QPen(QColor("#4fc3f7"), 2))
        painter.drawLine(plot_rect.right() - 220, legend_y, plot_rect.right() - 196, legend_y)
        painter.setPen(QColor("#d7dde5"))
        painter.drawText(plot_rect.right() - 190, legend_y + 5, "Raw")
        painter.setPen(QPen(QColor("#ff8a65"), 2))
        painter.drawLine(plot_rect.right() - 140, legend_y, plot_rect.right() - 116, legend_y)
        painter.setPen(QColor("#d7dde5"))
        painter.drawText(plot_rect.right() - 110, legend_y + 5, "Smoothed")

        painter.setPen(QColor("#aeb6c2"))
        for tick in x_ticks:
            x_ratio = (tick - min_t) / (max_t - min_t)
            x = plot_rect.left() + x_ratio * plot_rect.width()
            painter.drawText(int(x - 14), self.height() - 16, f"{tick:.1f}s")

        for tick in y_ticks:
            y_ratio = (tick - min_v) / (max_v - min_v)
            y = plot_rect.bottom() - y_ratio * plot_rect.height()
            painter.drawText(10, int(y + 4), self._format_value(tick))

        peak_label_colors = ["#ffcc80", "#e1bee7", "#bbdefb"]
        for rank, peak_idx in enumerate(top_peak_indices):
            painter.setPen(QColor(peak_label_colors[rank]))
            label_y = max(int(points[peak_idx][1] - 8 - (rank * 18)), plot_rect.top() + 18)
            painter.drawText(
                min(int(points[peak_idx][0] + 8), plot_rect.right() - 135),
                label_y,
                f"P{rank + 1} {self._format_value(values[peak_idx])}",
            )
        painter.setPen(QColor("#a5d6a7"))
        painter.drawText(
            min(int(points[last_idx][0] + 8), plot_rect.right() - 110),
            max(int(points[last_idx][1] - 8), plot_rect.top() + 34),
            f"Last {self._format_value(values[last_idx])}",
        )

        painter.setPen(QColor("#cfd8dc"))
        stats_text = (
            f"Samples: {len(values)}    "
            f"Mean: {self._format_value(float(np.mean(values)))}    "
            f"Std: {self._format_value(float(np.std(values)))}"
        )
        painter.drawText(plot_rect.left(), self.height() - 28, stats_text)


class SensorCaptureResultDialog(QDialog):
    def __init__(self, series, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sensor Peak Change (10s)")
        components.size_to_screen(self, 980, 620)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {theme.WINDOW_BG};
                color: {theme.TEXT_PRIMARY};
            }}
            QLabel#resultTitle {{
                font-size: 20px;
                font-weight: 600;
                color: {theme.TEXT_PRIMARY};
            }}
            QLabel#resultSubtitle {{
                font-size: 12px;
                color: {theme.TEXT_MUTED};
            }}
            QFrame#metricCard {{
                background-color: {theme.SURFACE_RAISED};
                border: 1px solid {theme.BORDER};
                border-radius: 10px;
            }}
            QLabel#metricName {{
                font-size: 11px;
                color: {theme.TEXT_MUTED};
            }}
            QLabel#metricValue {{
                font-size: 18px;
                font-weight: 600;
                color: {theme.TEXT_PRIMARY};
            }}
            QFrame#sectionCard {{
                background-color: {theme.SURFACE};
                border: 1px solid {theme.BORDER_SUBTLE};
                border-radius: 12px;
            }}
            QLabel#sectionTitle {{
                font-size: 13px;
                font-weight: 600;
                color: {theme.TEXT_PRIMARY};
            }}
            QLabel#sectionBody {{
                font-size: 12px;
                color: {theme.TEXT_MUTED};
            }}
            QPushButton {{
                min-width: 88px;
                min-height: 30px;
            }}
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        values = [point[1] for point in series] if series else []
        duration = float(series[-1][0]) if series else 0.0
        peak_value = max(values) if values else 0.0
        min_value = min(values) if values else 0.0
        mean_value = float(np.mean(values)) if values else 0.0
        std_value = float(np.std(values)) if values else 0.0

        title = QLabel("Sensor Peak Change Analysis")
        title.setObjectName("resultTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Signal used: max(abs(diffDataAve)) over the full sensor matrix at each sample."
        )
        subtitle.setObjectName("resultSubtitle")
        layout.addWidget(subtitle)

        def create_metric_card(name: str, value: str):
            card = QFrame()
            card.setObjectName("metricCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(2)

            name_label = QLabel(name)
            name_label.setObjectName("metricName")
            card_layout.addWidget(name_label)

            value_label = QLabel(value)
            value_label.setObjectName("metricValue")
            card_layout.addWidget(value_label)
            return card

        metrics_row = QGridLayout()
        metrics_row.setHorizontalSpacing(10)
        metrics_row.setVerticalSpacing(10)
        metric_cards = [
            ("Samples", str(len(values))),
            ("Duration", f"{duration:.2f} s"),
            ("Peak", f"{peak_value:.3f}"),
            ("Mean", f"{mean_value:.3f}"),
            ("Min", f"{min_value:.3f}"),
            ("Std", f"{std_value:.3f}"),
        ]
        for idx, (name, value) in enumerate(metric_cards):
            metrics_row.addWidget(create_metric_card(name, value), idx // 3, idx % 3)
        layout.addLayout(metrics_row)

        chart_card = QFrame()
        chart_card.setObjectName("sectionCard")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(12, 12, 12, 12)
        chart_layout.setSpacing(8)

        chart_title = QLabel("Time Series")
        chart_title.setObjectName("sectionTitle")
        chart_layout.addWidget(chart_title)

        chart = SimpleLineChartWidget(self)
        chart.set_series(series)
        chart_layout.addWidget(chart)
        layout.addWidget(chart_card, stretch=1)

        notes_card = QFrame()
        notes_card.setObjectName("sectionCard")
        notes_layout = QVBoxLayout(notes_card)
        notes_layout.setContentsMargins(12, 10, 12, 10)
        notes_layout.setSpacing(6)

        notes_title = QLabel("Interpretation")
        notes_title.setObjectName("sectionTitle")
        notes_layout.addWidget(notes_title)

        notes_body = QLabel(
            "Higher values mean stronger instantaneous capacitance-change activity somewhere on the sensor.\n"
            "The blue line is the raw peak-change signal, and the orange line is an EMA-smoothed trend."
        )
        notes_body.setWordWrap(True)
        notes_body.setObjectName("sectionBody")
        notes_layout.addWidget(notes_body)
        layout.addWidget(notes_card)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        button_box.accepted.connect(self.close)
        layout.addWidget(button_box)


class ExperimentControlsWindow(QDialog):
    """Reusable modeless home for the experiment controls panel."""

    _GEOMETRY_KEY = "windows/experiment_controls_geometry_wide"

    def __init__(self, controls: QWidget, parent=None):
        super().__init__(parent, Qt.Window)
        self._settings = QSettings("PingLab", "PingLab")
        self.setWindowTitle("Experiment Controls")
        self.setModal(False)
        self.setMinimumSize(860, 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(controls)

        saved_geometry = self._settings.value(self._GEOMETRY_KEY)
        if not saved_geometry or not self.restoreGeometry(saved_geometry):
            components.size_to_screen(self, 1080, 820)

    def remember_geometry(self) -> None:
        self._settings.setValue(self._GEOMETRY_KEY, self.saveGeometry())

    def reject(self) -> None:
        """Escape hides the utility without destroying its controls or state."""
        self.remember_geometry()
        self.hide()

    def closeEvent(self, event) -> None:
        """The window's close button hides it; experiments keep running."""
        self.remember_geometry()
        self.hide()
        event.ignore()


class MyMainWindow(MainWindow):
    """
    The main application window for PingLab.

    This class is responsible for setting up the main UI components,
    including the menu bar, toolbar, 3D view area, experiment window, and status bar.
    It manages the overall layout and handles core application events and state.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.resources = Resources()
        self._fps_timer = None
        self._sidebar_control_timer = None
        self._startup_requested = False
        self._sensor_scene_ready = False
        self._sensor_update_backend_enabled = True
        self.experiments_window = None
        self.camera_preview_window = None
        self._sidebar_active_control_id = None
        self._sensor_capture_result_dialog = None
        self._calibration_result_dialog = None
        self._full_taxel_calibration_result_dialog = None
        self._calibration_graph_analysis_dialog = None
        self._taxel_navigation_result = None
        self._taxel_mapping_anchor_updated_after_load = False
        self._taxel_navigation_source_path = None
        self._sidebar_active_task = None  # experiment_tasks.ExperimentTask, set while running
        self._sigraph_heatmap_recorder = None
        self._sigraph_heatmap_signal_recorder = None
        self._sigraph_heatmap_record_timer = None
        self._sigraph_heatmap_record_started_at = None
        self._sigraph_heatmap_playback = None
        self._sigraph_heatmap_playback_sensor = None
        self._sigraph_heatmap_playback_timer = None
        self._sigraph_heatmap_playback_started_at = None
        self._sigraph_heatmap_playback_last_index = -1
        self._sigraph_heatmap_playback_restore_state = None

        # --- Initialize Window Properties ---
        self.setWindowTitle("PingLab")
        self.setMinimumSize(960, 540)
        self.resize(1650, 900)
        self.setWindowIcon(self.resources.get_icon('logo0.png'))

        # --- Initialize Application State ---
        self.ping_mode_entered = False
        self.sensor_window = None
        self.ui_ros = None  # This will hold the main UI widget from ui_ping.py

        # --- Keyboard tool-frame velocity (toolbar toggle) ---
        # Multiple keys can be active at once (e.g. W+D = +Y and +X combined).
        # Press adds a direction; release removes it; Space clears all.
        self._keyboard_tool_velocity_enabled = False
        self._keyboard_vel_active_tokens: set[str] = set()
        self._keyboard_vel_timer = QTimer(self)
        self._keyboard_vel_timer.setInterval(50)
        self._keyboard_vel_timer.timeout.connect(self._keyboard_vel_timer_tick)

        # --- Build the User Interface ---
        self._setup_actions()
        self._setup_menu_and_toolbar()
        self.set_sensor_scene_ready(False)
        self._setup_main_layout()
        self._connect_signals_to_slots()

        # Apply the global stylesheet
        self.setStyleSheet(self.resources.get_stylesheet())

        # Delay heavy UI construction until the event loop starts so the window appears sooner.
        QTimer.singleShot(0, self.run_ping_mode)

    def _setup_actions(self):
        """Creates all QAction objects used in menus and toolbars."""
        self.action_ping_mode = QAction(icons.icon('play'), 'Ping', self)
        self.action_ping_mode.setToolTip('Enter Ping mode and build the main workspace')

        self.action_log = QAction(icons.icon('terminal'), 'Log', self)
        self.action_log.setToolTip('Open or close the Application Log window')

        self.action_sensor_signal = QAction(icons.icon('signal'), "Sensor Signal", self)
        self.action_sensor_signal.setToolTip('Open the standalone sensor signal viewer')
        self.action_ps5_controller_test = QAction(
            icons.icon('gamepad'),
            'PS5 Controller Test',
            self,
        )
        self.action_ps5_controller_test.setToolTip('Open the PS5 controller input test window')
        self.action_sensor_controller_test = QAction(
            icons.icon('target'),
            'Sensor Controller Test',
            self,
        )
        self.action_sensor_controller_test.setToolTip('Open the sensor-to-controller mapping test window')

        self.action_keyboard_tool_velocity = QAction(
            icons.icon('keyboard'),
            'Keyboard Tool Velocity',
            self,
        )
        self.action_keyboard_tool_velocity.setCheckable(True)
        self.action_keyboard_tool_velocity.setToolTip(
            "Keyboard tool-frame velocity (same panel as Robots → Send Tool Velocity).\n"
            "W/S = ±Y, A/D = ±X, O/P = ±Z, Q/E = ±Rz, R/T = ±Rx, Y/U = ±Ry.\n"
            "Hold or press multiple keys to combine (e.g. W+D = +Y and +X). "
            "Release a key to remove that component. Space stops all."
        )

        self.action_direct_finger_motion_params = QAction(
            icons.icon('hand'), 'Direct Finger Motion Parameters', self
        )
        self.action_direct_finger_motion_params.setToolTip('Open the Direct Finger Motion parameter editor')
        self.action_proximity_control_params = QAction(
            icons.icon('eye'),
            'Proximity Parameters',
            self,
        )
        self.action_proximity_control_params.setToolTip('Open the Proximity Control parameter editor')
        self.action_console_control_params = QAction(
            icons.icon('sliders'),
            'Console Control Parameters',
            self,
        )
        self.action_console_control_params.setToolTip('Open the Console Control parameter editor')
        self.action_sensor_params = QAction(
            icons.icon('sensor'),
            'Sensor Parameters',
            self,
        )
        self.action_sensor_params.setToolTip('Open the Sensor parameter editor')
        self.action_parameter_settings = QAction(
            icons.icon('settings'),
            'Parameter Settings',
            self,
        )
        self.action_parameter_settings.setToolTip('Open parameter settings menu')
        self.action_update_sensor = QAction(icons.icon('refresh'), 'Calibrate Sensor', self)
        self.action_update_sensor.setToolTip(
            'Calibrate the sensor baseline (Sensor Plotter and humanoid sensors)'
        )
        self.action_stop_all_motion = QAction(
            icons.icon('stop', color=theme.DANGER),
            'Stop All Motion',
            self,
        )
        self.action_stop_all_motion.setToolTip(
            'Software stop: cancel active controls and command all robot motion '
            'to stop. This does not replace the physical emergency stop.'
        )
        self.action_disconnect_sensor = QAction(
            icons.icon('x-circle', color=theme.DANGER),
            'Disconnect Sensor',
            self,
        )
        self.action_disconnect_sensor.setToolTip(
            'Stop the active sensor streams and clear the built sensor scene'
        )

        self.action_toggle_controls = QAction(icons.icon('flask'), 'Experiments', self)
        self.action_toggle_controls.setToolTip(
            'Open or focus the Experiment Controls window'
        )
        self.action_camera_preview = QAction(icons.icon('camera'), 'Camera', self)
        self.action_camera_preview.setToolTip(
            'Open the RealSense window for color, depth, infrared, or point cloud'
        )

        self.action_window_experiments = QAction(
            icons.icon('flask'), 'Experiments', self
        )
        self.action_window_log = QAction(
            icons.icon('terminal'), 'Application Log', self
        )
        self.action_window_sensor_signal = QAction(
            icons.icon('signal'), 'Sensor Signal', self
        )
        self.action_close_auxiliary_windows = QAction(
            icons.icon('x-circle'), 'Close Auxiliary Windows', self
        )

        self.action_presentation_mode = QAction('Presentation Mode', self)
        self.action_presentation_mode.setCheckable(True)
        self.action_presentation_mode.setShortcut('Ctrl+Shift+P')
        self.action_presentation_mode.setToolTip(
            'Increase control and row sizes for projected demonstrations or touch use'
        )

        self.action_exit = QAction(icons.icon('power', color=theme.DANGER), 'Exit', self)
        self.action_exit.setToolTip('Quit PingLab (Ctrl+Q)')
        self.action_exit.setShortcut('Ctrl+Q')

    def _setup_menu_and_toolbar(self):
        """Initializes the menu bar and the main toolbar using the predefined actions."""
        # --- Menu Bar ---
        menu = self.menuBar()
        mode_menu = menu.addMenu('Mode')
        mode_menu.addAction(self.action_ping_mode)

        view_menu = menu.addMenu('View')
        view_menu.addAction(self.action_presentation_mode)

        window_menu = menu.addMenu('Window')
        window_menu.addAction(self.action_window_experiments)
        window_menu.addAction(self.action_camera_preview)
        window_menu.addAction(self.action_window_log)
        window_menu.addAction(self.action_window_sensor_signal)
        window_menu.addSeparator()
        window_menu.addAction(self.action_close_auxiliary_windows)

        function_menu = menu.addMenu('Menu')
        function_menu.addAction(self.action_log)
        function_menu.addAction(self.action_sensor_signal)
        function_menu.addAction(self.action_ps5_controller_test)
        function_menu.addAction(self.action_sensor_controller_test)
        function_menu.addAction(self.action_keyboard_tool_velocity)
        parameter_menu = function_menu.addMenu('Parameter Settings')
        parameter_menu.addAction(self.action_direct_finger_motion_params)
        parameter_menu.addAction(self.action_proximity_control_params)
        parameter_menu.addAction(self.action_console_control_params)
        parameter_menu.addAction(self.action_sensor_params)
        function_menu.addSeparator()
        function_menu.addAction(self.action_toggle_controls)
        function_menu.addAction(self.action_camera_preview)
        function_menu.addSeparator()
        function_menu.addAction(self.action_exit)

        # --- Toolbar ---
        toolbar = QToolBar("Main")
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        toolbar.addAction(self.action_ping_mode)
        toolbar.addSeparator()
        toolbar.addAction(self.action_log)
        toolbar.addAction(self.action_sensor_signal)
        toolbar.addAction(self.action_ps5_controller_test)
        toolbar.addAction(self.action_sensor_controller_test)
        toolbar.addAction(self.action_keyboard_tool_velocity)
        self.toolbar_parameter_button = QToolButton()
        self.toolbar_parameter_button.setDefaultAction(self.action_parameter_settings)
        self.toolbar_parameter_button.setPopupMode(QToolButton.InstantPopup)
        parameter_popup_menu = QMenu(toolbar)
        parameter_popup_menu.addAction(self.action_direct_finger_motion_params)
        parameter_popup_menu.addAction(self.action_proximity_control_params)
        parameter_popup_menu.addAction(self.action_console_control_params)
        parameter_popup_menu.addAction(self.action_sensor_params)
        self.toolbar_parameter_button.setMenu(parameter_popup_menu)
        toolbar.addWidget(self.toolbar_parameter_button)
        toolbar.addAction(self.action_toggle_controls)
        toolbar.addAction(self.action_camera_preview)
        toolbar.addSeparator()
        toolbar.addAction(self.action_exit)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Without this the spacer picks up the default window background and
        # paints a pale slab across the middle of the toolbar.
        spacer.setStyleSheet("background: transparent;")
        toolbar.addWidget(spacer)
        self.toolbar_stop_all_motion_button = QToolButton()
        self.toolbar_stop_all_motion_button.setText("Stop All Motion")
        self.toolbar_stop_all_motion_button.setIcon(
            icons.icon('stop', color=theme.TEXT_ON_ACCENT, size=16)
        )
        self.toolbar_stop_all_motion_button.setIconSize(QSize(16, 16))
        self.toolbar_stop_all_motion_button.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.toolbar_stop_all_motion_button.setToolTip(
            self.action_stop_all_motion.toolTip()
        )
        self.toolbar_stop_all_motion_button.setCursor(Qt.PointingHandCursor)
        self.toolbar_stop_all_motion_button.setStyleSheet(
            "QToolButton {"
            f" border: 1px solid {theme.DANGER};"
            f" border-radius: {theme.RADIUS_MD}px;"
            " padding: 4px 12px;"
            " font-weight: 700;"
            f" color: {theme.TEXT_ON_ACCENT};"
            f" background-color: {theme.DANGER};"
            "}"
            "QToolButton:hover {"
            f" background-color: {theme.DANGER_HOVER};"
            f" border-color: {theme.DANGER_HOVER};"
            "}"
            "QToolButton:pressed {"
            f" background-color: {theme.DANGER_PRESSED};"
            f" border-color: {theme.DANGER_PRESSED};"
            "}"
        )
        self.toolbar_stop_all_motion_button.clicked.connect(
            self.action_stop_all_motion.trigger
        )
        toolbar.addWidget(self.toolbar_stop_all_motion_button)
        toolbar.addSeparator()
        self.toolbar_disconnect_sensor_button = QToolButton()
        self.toolbar_disconnect_sensor_button.setText("Disconnect Sensor")
        self.toolbar_disconnect_sensor_button.setIcon(
            icons.icon('x-circle', color=theme.DANGER, size=16)
        )
        self.toolbar_disconnect_sensor_button.setIconSize(QSize(16, 16))
        self.toolbar_disconnect_sensor_button.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.toolbar_disconnect_sensor_button.setToolTip(
            self.action_disconnect_sensor.toolTip()
        )
        self.toolbar_disconnect_sensor_button.setCursor(Qt.PointingHandCursor)
        self.toolbar_disconnect_sensor_button.setStyleSheet(
            "QToolButton {"
            f" border: 1px solid {theme.DANGER_BORDER};"
            f" border-radius: {theme.RADIUS_MD}px;"
            " padding: 4px 12px;"
            " font-weight: 600;"
            f" color: {theme.DANGER};"
            f" background-color: {theme.DANGER_SOFT};"
            "}"
            "QToolButton:hover {"
            f" color: {theme.TEXT_ON_ACCENT};"
            f" background-color: {theme.DANGER_HOVER};"
            f" border-color: {theme.DANGER_HOVER};"
            "}"
            "QToolButton:pressed {"
            f" color: {theme.TEXT_ON_ACCENT};"
            f" background-color: {theme.DANGER_PRESSED};"
            f" border-color: {theme.DANGER_PRESSED};"
            "}"
            "QToolButton:disabled {"
            f" background-color: {theme.SURFACE_RAISED};"
            f" border-color: {theme.BORDER};"
            f" color: {theme.TEXT_DISABLED};"
            "}"
        )
        self.toolbar_disconnect_sensor_button.clicked.connect(
            self.action_disconnect_sensor.trigger
        )
        toolbar.addWidget(self.toolbar_disconnect_sensor_button)
        # Calibration is the main repeated sensor action, so it keeps the
        # filled accent treatment beside the global safety controls.
        self.toolbar_update_sensor_button = QToolButton()
        self.toolbar_update_sensor_button.setText("Calibrate Sensor")
        self.toolbar_update_sensor_button.setIcon(
            icons.icon('refresh', color=theme.TEXT_ON_ACCENT, size=16)
        )
        self.toolbar_update_sensor_button.setIconSize(QSize(16, 16))
        self.toolbar_update_sensor_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toolbar_update_sensor_button.setToolTip(
            self.action_update_sensor.toolTip()
        )
        self.toolbar_update_sensor_button.setCursor(Qt.PointingHandCursor)
        self.toolbar_update_sensor_button.setStyleSheet(
            "QToolButton {"
            f" border: 1px solid {theme.ACCENT};"
            f" border-radius: {theme.RADIUS_MD}px;"
            " padding: 4px 12px;"
            " font-weight: 600;"
            f" color: {theme.TEXT_ON_ACCENT};"
            f" background-color: {theme.ACCENT};"
            "}"
            "QToolButton:hover {"
            f" background-color: {theme.ACCENT_HOVER};"
            f" border-color: {theme.ACCENT_HOVER};"
            "}"
            "QToolButton:pressed {"
            f" background-color: {theme.ACCENT_PRESSED};"
            f" border-color: {theme.ACCENT_PRESSED};"
            "}"
            "QToolButton:disabled {"
            f" background-color: {theme.SURFACE_RAISED};"
            f" border-color: {theme.BORDER};"
            f" color: {theme.TEXT_DISABLED};"
            "}"
        )
        self.toolbar_update_sensor_button.clicked.connect(self.action_update_sensor.trigger)
        toolbar.addWidget(self.toolbar_update_sensor_button)
        self.addToolBar(toolbar)

    def _setup_main_layout(self):
        """Constructs the central widget and the main horizontal layout."""
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        self.h_splitter = QSplitter(Qt.Horizontal)
        self.h_splitter.setHandleWidth(6)

        # --- Container for the 3D View ---
        self.view_container = QWidget()
        view_layout = QVBoxLayout(self.view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self.view_placeholder = QLabel("Entering Ping Mode...")
        self.view_placeholder.setAlignment(Qt.AlignCenter)
        view_layout.addWidget(self.view_placeholder)

        # The experiment controls live in a modeless utility window so they do
        # not continually take width away from the visualization workspace.
        self.sidebar = self._create_sidebar()
        self.experiments_window = ExperimentControlsWindow(
            self.sidebar,
            parent=self,
        )

        self.h_splitter.addWidget(self.view_container)

        main_layout.addWidget(self.h_splitter)

        # --- Status Bar ---
        self._setup_status_bar()

    def _create_sidebar(self) -> QFrame:
        """Creates the sidebar widget and all its contents."""
        sidebar = QFrame()
        sidebar.setObjectName("experimentPanel")
        sidebar.setFrameShape(QFrame.StyledPanel)
        sidebar.setStyleSheet(
            f"QFrame#experimentPanel {{ background-color: {theme.WINDOW_BG}; "
            "border: none; }"
        )
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.experiment_controls_splitter = QSplitter(Qt.Horizontal)
        self.experiment_controls_splitter.setChildrenCollapsible(False)
        self.experiment_controls_splitter.setHandleWidth(8)

        task_column = QWidget()
        task_column.setMinimumWidth(210)
        task_layout = QVBoxLayout(task_column)
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.setSpacing(8)
        tasks_header = QLabel("Available Tasks")
        font = tasks_header.font()
        font.setBold(True)
        tasks_header.setFont(font)
        task_layout.addWidget(tasks_header)

        self.sidebar_task_list = QTreeWidget()
        self.sidebar_task_list.setHeaderHidden(True)
        self.sidebar_task_list.setSelectionMode(QTreeWidget.SingleSelection)
        self.sidebar_task_list.setRootIsDecorated(True)
        self.sidebar_task_list.setIndentation(16)
        grouped_tasks: dict = {}
        for label_text, task_id, description, category in experiment_tasks.task_definitions():
            grouped_tasks.setdefault(category, []).append(
                (label_text, task_id, description)
            )
        first_task_item = None
        for category in experiment_tasks.SIDEBAR_CATEGORIES:
            tasks = grouped_tasks.get(category) or []
            if not tasks:
                continue
            header = QTreeWidgetItem([category])
            header.setFlags(Qt.ItemIsEnabled)
            header_font = header.font(0)
            header_font.setBold(True)
            header.setFont(0, header_font)
            self.sidebar_task_list.addTopLevelItem(header)
            for label_text, task_id, description in tasks:
                item = QTreeWidgetItem([label_text])
                item.setData(0, Qt.UserRole, task_id)
                item.setData(0, Qt.UserRole + 1, description)
                header.addChild(item)
                if first_task_item is None:
                    first_task_item = item
            header.setExpanded(
                category not in experiment_tasks.COLLAPSED_SIDEBAR_CATEGORIES
            )
        if first_task_item is not None:
            self.sidebar_task_list.setCurrentItem(first_task_item)
        task_layout.addWidget(self.sidebar_task_list, stretch=1)

        detail_column = QWidget()
        detail_column.setMinimumWidth(520)
        detail_layout = QVBoxLayout(detail_column)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(8)

        self.sidebar_task_info = QLabel()
        self.sidebar_task_info.setWordWrap(True)
        self.sidebar_task_info.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; background-color: {theme.SURFACE_RAISED}; "
            f"border: 1px solid {theme.BORDER_SUBTLE}; padding: 8px; border-radius: 6px;"
        )
        detail_layout.addWidget(self.sidebar_task_info)
        self._update_sidebar_task_info()

        # Each task may expose a custom widget under this container; only the
        # widget that matches the currently-selected task is shown.
        self._build_sidebar_task_param_widgets(detail_layout)
        self._update_sidebar_task_param_visibility()

        # Container for task execution and recording buttons
        btn_container = QWidget()
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        run_layout = QHBoxLayout()
        run_layout.addStretch()
        self.sidebar_btn_start = QPushButton("Start")
        self.sidebar_btn_start.setObjectName("btnStart")
        run_layout.addWidget(self.sidebar_btn_start)
        self.sigraph_scanning_reverse_button = QPushButton("Reverse Run")
        self.sigraph_scanning_reverse_button.setObjectName("btnReverse")
        self.sigraph_scanning_reverse_button.setVisible(False)
        run_layout.addWidget(self.sigraph_scanning_reverse_button)
        self.sidebar_btn_stop = QPushButton("Stop")
        self.sidebar_btn_stop.setObjectName("btnStop")
        run_layout.addWidget(self.sidebar_btn_stop)
        run_layout.addStretch()
        btn_layout.addLayout(run_layout)

        record_layout = QHBoxLayout()
        record_layout.addStretch()
        self.sigraph_scanning_run_record_button = QPushButton("Run + Record")
        self.sigraph_scanning_run_record_button.setObjectName("btnRunRecord")
        self.sigraph_scanning_run_record_button.setVisible(False)
        record_layout.addWidget(self.sigraph_scanning_run_record_button)
        self.sigraph_scanning_reverse_record_button = QPushButton(
            "Reverse Run + Record"
        )
        self.sigraph_scanning_reverse_record_button.setObjectName(
            "btnReverseRecord"
        )
        self.sigraph_scanning_reverse_record_button.setVisible(False)
        record_layout.addWidget(self.sigraph_scanning_reverse_record_button)
        record_layout.addStretch()
        btn_layout.addLayout(record_layout)
        detail_layout.addWidget(btn_container)
        for index in range(detail_layout.count()):
            item = detail_layout.itemAt(index)
            widget = None if item is None else item.widget()
            if widget is None or widget in {
                self.sidebar_task_info,
                btn_container,
            }:
                detail_layout.setStretch(index, 0)
                continue
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            detail_layout.setStretch(index, 1)
        self._update_sidebar_task_action_labels()

        self.experiment_controls_splitter.addWidget(task_column)
        self.experiment_controls_splitter.addWidget(detail_column)
        self.experiment_controls_splitter.setStretchFactor(0, 0)
        self.experiment_controls_splitter.setStretchFactor(1, 1)
        self.experiment_controls_splitter.setSizes([240, 800])
        layout.addWidget(self.experiment_controls_splitter, stretch=1)

        return sidebar

    def _setup_status_bar(self):
        """Build the compact status bar.

        Left side carries the last action message plus a warning slot; the
        right side carries always-on device state and live rates, so the
        operator can see connection, sensor rate, render rate and the active
        control mode without leaving the current workspace.
        """
        status = QStatusBar()
        status.setSizeGripEnabled(False)

        self.info_process = components.ElidedLabel("Ready")
        self.info_process.setObjectName("statusActivity")
        self.info_process.setToolTip("Ready")
        status.addWidget(self.info_process, 1)

        self.status_experiment = components.StatusDot("Experiment", "active")
        self.status_experiment.setVisible(False)
        status.addWidget(self.status_experiment)

        self.status_calibration_travel = QLabel(
            "Z travel: — mm | Max: — mm"
        )
        self.status_calibration_travel.setObjectName("statusMetric")
        self.status_calibration_travel.setToolTip(
            "Measured from live robot TCP Z feedback relative to the current "
            "calibration trial's starting height. This is displacement, not "
            "cumulative round-trip path length."
        )
        self.status_calibration_travel.setVisible(False)
        status.addWidget(self.status_calibration_travel)

        self.status_experiment_stop_button = QPushButton("Stop")
        self.status_experiment_stop_button.setObjectName("btnStop")
        self.status_experiment_stop_button.setToolTip(
            "Stop the currently running experiment"
        )
        self.status_experiment_stop_button.setVisible(False)
        status.addWidget(self.status_experiment_stop_button)

        self.status_warning = QLabel("")
        self.status_warning.setStyleSheet(
            f"color: {theme.WARNING}; font-size: {theme.FONT_SIZE_SMALL};"
            f" font-weight: {theme.WEIGHT_MEDIUM};"
        )
        self.status_warning.setVisible(False)
        status.addWidget(self.status_warning)

        self.status_sensor = components.StatusDot("Sensor", "idle")
        self.status_robot = components.StatusDot("Robot", "idle")
        self.status_hand = components.StatusDot("Hand", "idle")

        self.status_mode = QLabel("Idle")
        self.status_mode.setObjectName("statusMetric")
        self.status_mode.setToolTip("Active control mode")

        self.status_sensor_hz = QLabel("— Hz")
        self.status_sensor_hz.setObjectName("statusMetric")
        self.status_sensor_hz.setToolTip("Sensor acquisition rate")

        self.info_fps = QLabel("0 fps")
        self.info_fps.setObjectName("statusMetric")
        self.info_fps.setToolTip("3D viewport render rate")

        for widget in (
            self.status_sensor,
            self.status_robot,
            self.status_hand,
            self._status_separator(),
            self.status_mode,
            self._status_separator(),
            self.status_sensor_hz,
            self.info_fps,
        ):
            status.addPermanentWidget(widget)

        self.setStatusBar(status)

    @staticmethod
    def _status_separator() -> QLabel:
        separator = QLabel("│")
        separator.setObjectName("statusSeparator")
        return separator

    def set_status_warning(self, message: str = "") -> None:
        """Show (or clear) the amber warning slot in the status bar."""
        self.status_warning.setText(str(message or ""))
        self.status_warning.setVisible(bool(message))

    def set_activity_message(self, message: str = "") -> None:
        """Show the newest log line without letting it widen the status bar."""
        message = " ".join(str(message or "").splitlines()).strip()
        message = message or "Application log is empty"
        self.info_process.setText(message)
        self.info_process.setToolTip(message)

    def _refresh_sensor_update_control(self) -> None:
        scene_ready = bool(self._sensor_scene_ready)
        backend_idle = bool(self._sensor_update_backend_enabled)
        enabled = scene_ready and backend_idle
        self.action_update_sensor.setEnabled(enabled)
        self.toolbar_update_sensor_button.setEnabled(enabled)
        self.action_disconnect_sensor.setEnabled(enabled)
        self.toolbar_disconnect_sensor_button.setEnabled(enabled)
        update_tooltip = (
            "Wait for the current sensor operation to finish."
            if scene_ready and not backend_idle
            else (
                "Calibrate the sensor baseline (Sensor Plotter and humanoid sensors)"
                if scene_ready
                else "Connect a sensor scene before calibrating it."
            )
        )
        disconnect_tooltip = (
            "Wait for the current sensor operation to finish."
            if scene_ready and not backend_idle
            else (
                "Stop the active sensor streams and clear the built sensor scene"
                if scene_ready
                else "No built sensor scene is connected."
            )
        )
        self.action_update_sensor.setToolTip(update_tooltip)
        self.toolbar_update_sensor_button.setToolTip(update_tooltip)
        self.action_disconnect_sensor.setToolTip(disconnect_tooltip)
        self.toolbar_disconnect_sensor_button.setToolTip(disconnect_tooltip)

    def set_sensor_scene_ready(self, ready: bool = True) -> None:
        """Record whether a valid sensor scene exists for recalibration."""
        self._sensor_scene_ready = bool(ready)
        self._refresh_sensor_update_control()

    def set_sensor_update_enabled(self, enabled: bool = True) -> None:
        """Gate recalibration while preserving the scene-ready requirement."""
        self._sensor_update_backend_enabled = bool(enabled)
        self._refresh_sensor_update_control()

    def set_log_unread_count(self, count: int = 0) -> None:
        """Mark the Log action when lines arrived while the panel was hidden.

        This replaces the old embedded log forcing itself open on every message. The
        toolbar is icon-only, so the count goes in the menu label and tooltip
        and the icon switches to the accent colour -- noticeable on a glance,
        but it never takes width from the 3D viewport.
        """
        count = max(int(count or 0), 0)
        if count == getattr(self, "_log_unread_shown", None):
            return
        self._log_unread_shown = count
        self.action_log.setText(f"Log ({count})" if count else "Log")
        self.action_log.setToolTip(
            f"Open the Application Log window — {count} new "
            f"{'message' if count == 1 else 'messages'}"
            if count
            else "Open or close the Application Log window"
        )
        self.action_log.setIcon(
            icons.icon("terminal", color=theme.ACCENT) if count
            else icons.icon("terminal")
        )

    # Helper attribute -> label shown in the status bar's "active mode" slot.
    # Ordered by precedence: the first running helper wins.
    _MODE_INDICATORS = (
        ("ai_direct_finger_motion_execution_class", "AI DFM (exec)"),
        ("ai_direct_finger_motion_class", "AI DFM (record)"),
        ("direct_finger_motion_class", "Rule-based DFM"),
        ("proximity_control_class", "Proximity"),
        ("console_control_class", "Console"),
        ("threelevel_hierarchical_transformer_class", "3-Level"),
        ("record_gesture_class", "Recording"),
    )

    def _refresh_status_indicators(self):
        """Repoll device state for the status bar. Called once per second.

        Everything read here is a cached flag or counter -- no serial or ROS
        traffic is generated, so this stays off the critical path.
        """
        ui = self.ui_ros
        if ui is None:
            return

        features = getattr(ui, "features", {}) or {}
        sensor_functions = getattr(ui, "sensor_functions", None)

        # --- Sensor ------------------------------------------------------
        sensor_api = getattr(ui, "sensor_api", None)
        connected = False
        try:
            connected = bool(sensor_api.is_connected())
        except Exception:
            connected = False
        # Normal serial scenes are owned by MySensor rather than the lazy
        # top-level API instance, so either backend can prove the stream live.
        connected = connected or bool(
            getattr(sensor_functions, "is_connected", False)
        )
        sensor_hz = float(getattr(sensor_functions, "_sensor_update_hz", 0.0) or 0.0)
        if connected and sensor_hz > 0.5:
            self.status_sensor.set_state("active", "Sensor")
        elif connected:
            self.status_sensor.set_state("warning", "Sensor")
        else:
            self.status_sensor.set_state("idle", "Sensor")
        self.status_sensor_hz.setText(
            f"{sensor_hz:5.1f} Hz" if sensor_hz > 0 else "  — Hz"
        )

        # --- Robot -------------------------------------------------------
        robot_ready = bool(features.get("robot_ready", False))
        velocity_active = False
        try:
            velocity_active = bool(ui._robot_velocity_mode_active())
        except Exception:
            velocity_active = False
        if velocity_active:
            self.status_robot.set_state("warning", "Robot moving")
        elif robot_ready:
            self.status_robot.set_state("active", "Robot")
        else:
            self.status_robot.set_state("idle", "Robot")

        # --- Dexterous hand ---------------------------------------------
        self.status_hand.set_state(
            "active" if features.get("hand_ready", False) else "idle", "Hand"
        )

        # --- Active control mode ----------------------------------------
        active_mode = self._active_mode_label(sensor_functions)
        ai_proximity_button = getattr(ui, "ai_proximity_detection_button", None)
        if ai_proximity_button is not None and ai_proximity_button.isChecked():
            admittance_button = getattr(ui, "ai_proximity_admittance_button", None)
            active_mode = (
                "AI Proximity + Admittance"
                if admittance_button is not None and admittance_button.isChecked()
                else "AI Proximity"
            )
        self.status_mode.setText(active_mode)
        self._refresh_session_strip(ui, active_mode, connected, sensor_hz)
        self._sync_taxel_maps_to_streaming_sensor()

    @staticmethod
    def _selected_sensor_summary(
        ui,
        connected: bool,
        sensor_hz: float,
    ) -> tuple[str, str]:
        source = None
        sensor_functions = getattr(ui, "sensor_functions", None)
        if connected:
            describe_source = getattr(
                sensor_functions,
                "describe_live_sensor_source",
                None,
            )
            if callable(describe_source):
                try:
                    source = describe_source()
                except Exception:
                    source = None

        if source:
            port_text = str(source.get("port_label", "") or "").strip()
            rows = int(source.get("n_row", 0) or 0)
            cols = int(source.get("n_col", 0) or 0)
        elif connected:
            # A highlighted row is only a setup choice; it does not identify
            # the stream that is already running. Prefer backend-owned port
            # state, and avoid claiming that the selected row is live when
            # older/fallback backends cannot provide source metadata.
            primary_port = str(
                getattr(sensor_functions, "_primary_sensor_port", "") or ""
            ).strip()
            sensor_api = getattr(ui, "sensor_api", None)
            active_port = primary_port or str(
                getattr(sensor_api, "serial_port", "") or ""
            ).strip()
            port_text = os.path.basename(active_port) if active_port else "Active sensor"
            rows = int(getattr(sensor_functions, "n_row", 0) or 0)
            cols = int(getattr(sensor_functions, "n_col", 0) or 0)
        else:
            ports = getattr(ui, "serial_channel", None)
            current = ports.currentItem() if ports is not None else None
            port_text = current.text().strip() if current is not None else "No sensor"
            rows = getattr(
                getattr(ui, "grid_rows_spin", None),
                "value",
                lambda: 0,
            )()
            cols = getattr(
                getattr(ui, "grid_cols_spin", None),
                "value",
                lambda: 0,
            )()

        port_text = port_text or "Active sensor"
        shape = f"{int(rows)}x{int(cols)}" if rows and cols else ""
        state = "Streaming" if connected else "Offline"
        details = [state, port_text]
        if shape and shape.lower() not in port_text.lower():
            details.append(shape)
        if connected and sensor_hz > 0:
            details.append(f"{sensor_hz:.0f} Hz")
        value = " · ".join(details)
        return value, value

    def _refresh_session_strip(
        self,
        ui,
        active_mode: str,
        connected: bool,
        sensor_hz: float,
    ) -> None:
        """Update the operator summary from cached labels and flags only."""
        strip = getattr(ui, "session_strip", None)
        if strip is None:
            return

        sensor_value, sensor_tooltip = self._selected_sensor_summary(
            ui,
            connected,
            sensor_hz,
        )
        strip.set_value(
            "sensor",
            sensor_value,
            tooltip=sensor_tooltip,
            state="active" if connected else "idle",
        )
        strip.set_value(
            "control",
            active_mode,
            state="active" if active_mode != "Idle" else "idle",
        )

        if active_mode == "AI DFM (exec)":
            model_label = getattr(ui, "ai_direct_execution_model_status", None)
        elif active_mode.startswith("AI Proximity"):
            model_label = getattr(ui, "ai_proximity_model_status", None)
        else:
            model_label = None
        model_text = model_label.text().strip() if model_label is not None else "Not active"
        strip.set_value(
            "model",
            model_text,
            tooltip=model_label.toolTip() if model_label is not None else model_text,
            state="active" if model_label is not None else "idle",
        )

        force_label = getattr(ui, "force_meter_value_label", None)
        force_button = getattr(ui, "force_meter_connect_button", None)
        force_live = force_button is not None and force_button.isChecked()
        force_text = force_label.text().strip() if force_label is not None else "— N"
        force_value = f"Live · {force_text}" if force_live else "Offline"
        strip.set_value(
            "force",
            force_value,
            tooltip=f"Force meter: {force_value}",
            state="active" if force_live else "idle",
        )

    def _active_mode_label(self, sensor_functions) -> str:
        """Return the highest-precedence running control mode, or ``Idle``.

        A helper that failed to initialize is replaced by a stub whose
        ``__getattr__`` hands back a no-op *function* for every name, so a
        truthiness test on ``is_running`` would report a mode that is not
        running. Hence the identity check against ``True`` and the falsiness
        guard -- both stub flavours define ``__bool__`` as ``False``.
        """
        if sensor_functions is None:
            return "Idle"
        getter = getattr(sensor_functions, "get_initialized_helper", None)
        for attribute, label in self._MODE_INDICATORS:
            if callable(getter):
                helper = getter(attribute)
            else:
                helper = getattr(sensor_functions, attribute, None)
            if not helper:
                continue
            if getattr(helper, "is_running", False) is True:
                return label
        return "Idle"

    def _connect_signals_to_slots(self):
        """Connects all QAction and widget signals to their corresponding methods (slots)."""
        self.action_exit.triggered.connect(self.close)
        self.action_ping_mode.triggered.connect(self.run_ping_mode)
        self.action_log.triggered.connect(self._toggle_log_view)
        self.action_toggle_controls.triggered.connect(
            self.open_experiments_window
        )
        self.action_camera_preview.triggered.connect(
            self.open_camera_preview_window
        )
        self.action_window_experiments.triggered.connect(
            self.open_experiments_window
        )
        self.action_window_log.triggered.connect(
            self.open_application_log_window
        )
        self.action_window_sensor_signal.triggered.connect(
            self.open_sensor_signal_window
        )
        self.action_close_auxiliary_windows.triggered.connect(
            self.close_auxiliary_windows
        )
        self.action_sensor_signal.triggered.connect(self.open_sensor_signal_window)
        self.action_ps5_controller_test.triggered.connect(self.open_ps5_controller_test_window)
        self.action_sensor_controller_test.triggered.connect(self.open_sensor_controller_test_window)
        self.action_keyboard_tool_velocity.toggled.connect(self._on_keyboard_tool_velocity_toggled)
        self.action_direct_finger_motion_params.triggered.connect(self.open_direct_finger_motion_params_window)
        self.action_proximity_control_params.triggered.connect(self.open_proximity_control_params_window)
        self.action_console_control_params.triggered.connect(self.open_console_control_params_window)
        self.action_sensor_params.triggered.connect(self.open_sensor_params_window)
        self.action_presentation_mode.toggled.connect(
            self._set_presentation_mode
        )
        self.action_stop_all_motion.triggered.connect(
            self._trigger_global_stop_all_motion
        )
        self.action_disconnect_sensor.triggered.connect(
            self._trigger_global_sensor_disconnect
        )
        self.action_update_sensor.triggered.connect(self._trigger_global_sensor_update)
        self.sidebar_btn_start.clicked.connect(self._start_sidebar_control)
        self.sigraph_scanning_reverse_button.clicked.connect(
            self._start_sigraph_scanning_reverse
        )
        self.sigraph_scanning_run_record_button.clicked.connect(
            self._start_sigraph_scanning_run_and_record
        )
        self.sigraph_scanning_reverse_record_button.clicked.connect(
            self._start_sigraph_scanning_reverse_and_record
        )
        self.sidebar_btn_stop.clicked.connect(
            lambda: self._stop_sidebar_control()
        )
        self.status_experiment_stop_button.clicked.connect(
            lambda: self._stop_sidebar_control()
        )
        self.sidebar_task_list.currentItemChanged.connect(self._on_sidebar_task_changed)

    def _set_presentation_mode(self, enabled: bool) -> None:
        """Apply larger hit targets without changing the normal lab layout."""
        stylesheet = self.resources.get_stylesheet()
        if enabled:
            stylesheet += theme.presentation_override_stylesheet()
        self.setStyleSheet(stylesheet)

    def _trigger_global_sensor_update(self):
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before calibrating the sensor."
        ):
            return
        # Legacy embedded callers do not expose the new state attribute; the
        # real main window always initializes it explicitly to False.
        if not bool(getattr(self, "_sensor_scene_ready", True)):
            message = "Connect a sensor scene before calibrating"
            self.info_process.setText(message)
            return
        try:
            humanoid_update_requested = False
            humanoid_viewer = getattr(
                self.ui_ros,
                "humanoid_viewer",
                None,
            )
            humanoid_update_requested = bool(
                humanoid_viewer is not None
                and humanoid_viewer.request_sensor_update()
            )
            if hasattr(self.ui_ros, "_on_sensor_update"):
                self.ui_ros._on_sensor_update()
            elif hasattr(self.ui_ros, "sensor_functions") and hasattr(self.ui_ros.sensor_functions, "updateCal"):
                self.ui_ros.sensor_functions.updateCal()
            self.info_process.setText(
                "Calibrating Sensor Plotter and humanoid sensors..."
                if humanoid_update_requested
                else "Calibrating Sensor Plotter..."
            )
        except Exception as exc:
            self.info_process.setText("Sensor calibration failed")
            if self.ui_ros is not None and hasattr(self.ui_ros, "log_display"):
                self.ui_ros.log_display.append(f"Sensor calibration failed: {exc}")

    def _trigger_global_sensor_disconnect(self):
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before disconnecting the sensor."
        ):
            return
        if not bool(getattr(self, "_sensor_scene_ready", True)):
            message = "No built sensor scene is connected"
            self.info_process.setText(message)
            return

        disconnect = getattr(self.ui_ros, "disconnect_sensor_scene", None)
        if not callable(disconnect):
            sensor = getattr(self.ui_ros, "sensor_functions", None)
            disconnect = getattr(sensor, "disconnect_sensor_scene", None)
        if not callable(disconnect):
            message = "Sensor disconnect is unavailable"
            self.info_process.setText(message)
            return

        try:
            disconnected = bool(disconnect())
        except Exception as exc:
            disconnected = False
            sensor = getattr(self.ui_ros, "sensor_functions", None)
            set_error = getattr(sensor, "_set_sensor_stream_error", None)
            if callable(set_error):
                set_error(f"Sensor disconnect failed: {exc}")

        if disconnected:
            self.set_sensor_scene_ready(False)
            self.set_sensor_update_enabled(True)
            self.info_process.setText("Sensor disconnected")
            return

        sensor = getattr(self.ui_ros, "sensor_functions", None)
        get_error = getattr(sensor, "get_last_sensor_stream_error", None)
        error = str(get_error() or "") if callable(get_error) else ""
        message = error or "Sensor disconnect failed"
        self.info_process.setText(message)
        log_display = getattr(self.ui_ros, "log_display", None)
        if log_display is not None and hasattr(log_display, "append"):
            log_display.append(message)

    def _trigger_global_stop_all_motion(self):
        """Best-effort software stop without disconnecting sensor feedback."""
        issues = []
        try:
            self._stop_sidebar_control(
                show_result=False,
                message="Experiment stopped by Stop All Motion",
            )
        except Exception as exc:
            issues.append(f"experiment: {exc}")

        try:
            self._disable_keyboard_tool_velocity_internal()
        except Exception as exc:
            issues.append(f"keyboard velocity: {exc}")

        stop_motion = getattr(self.ui_ros, "stop_all_motion", None)
        if callable(stop_motion):
            try:
                result = stop_motion()
                if isinstance(result, (list, tuple)):
                    issues.extend(str(item) for item in result if item)
            except Exception as exc:
                issues.append(f"embedded controls: {exc}")
        elif self.ui_ros is not None:
            issues.append("embedded motion stop is unavailable")

        if issues:
            message = "Software stop sent with warnings: " + "; ".join(issues)
        else:
            message = "All software motion stopped; sensor streaming remains active"
        self.info_process.setText(message)
        log_display = getattr(self.ui_ros, "log_display", None)
        if log_display is not None and hasattr(log_display, "append"):
            log_display.append(message)

    def _set_sidebar_visible(self, visible: bool):
        """Compatibility helper for showing or hiding the experiment window."""
        window = getattr(self, "experiments_window", None)
        if window is None:
            return
        if visible:
            window.show()
            self._focus_window(window)
        else:
            window.remember_geometry()
            window.hide()

    def _require_ui_ros(self, message: str) -> bool:
        """Shows a warning when Ping Mode is not ready yet."""
        if self.ui_ros is not None:
            return True

        QMessageBox.warning(self, "Warning", message)
        return False

    def _focus_window(self, window: QWidget):
        """Brings an already-open child window to the foreground."""
        window.raise_()
        window.activateWindow()

    def open_experiments_window(self):
        """Open or focus the reusable modeless experiment-controls window."""
        self._set_sidebar_visible(True)

    def open_camera_preview_window(self):
        """Open or focus the live RealSense color window."""
        camera_ui = getattr(self, "ui_ros", None)
        worker = getattr(camera_ui, "yolo_worker", None)
        if worker is not None and worker.isRunning():
            camera_ui._stop_auto_centering()
            camera_ui._stop_yolo_worker()
            camera_ui._close_camera_window()
            camera_ui._set_live_camera_ui_state(False)
            log_display = getattr(camera_ui, "log_display", None)
            if log_display is not None:
                log_display.append(
                    "Object detection stopped so the RealSense preview can "
                    "use the camera."
                )
        window = getattr(self, "camera_preview_window", None)
        if window is not None:
            try:
                visible = window.isVisible()
            except RuntimeError:
                window = None
            else:
                if visible:
                    window.show()
                    self._focus_window(window)
                    return
        window = CameraPreviewWindow(self)
        window.destroyed.connect(self._clear_camera_preview_window)
        self.camera_preview_window = window
        window.show()
        self._focus_window(window)

    def _clear_camera_preview_window(self, *_args):
        """Drop the preview handle after Qt deletes the closed window."""
        self.camera_preview_window = None

    def open_application_log_window(self):
        """Open or focus the modeless application-log window."""
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before opening the Application Log."
        ):
            return

        console = getattr(self.ui_ros, "log_console_window", None)
        if console is None:
            log_display = getattr(self.ui_ros, "log_display", None)
            if log_display is not None:
                log_display.setVisible(True)
            return
        console.show()
        self._focus_window(console)
        clear_unread = getattr(self.ui_ros, "_clear_log_unread_count", None)
        if callable(clear_unread):
            clear_unread()

    def close_auxiliary_windows(self):
        """Hide the three routine auxiliary windows without ending the session."""
        self._set_sidebar_visible(False)

        sensor_window = getattr(self, "sensor_window", None)
        if sensor_window is not None:
            sensor_window.close()

        ui = getattr(self, "ui_ros", None)
        console = getattr(ui, "log_console_window", None) if ui is not None else None
        if console is not None:
            remember = getattr(console, "remember_geometry", None)
            if callable(remember):
                remember()
            console.hide()
        elif ui is not None:
            log_display = getattr(ui, "log_display", None)
            if log_display is not None:
                log_display.setVisible(False)

    def toggle_sidebar(self):
        """Backward-compatible alias for opening the experiment window."""
        self.open_experiments_window()

    def open_sensor_signal_window(self):
        """
        Opens the sensor signal viewer window.
        Prevents opening if the main UI is not ready or if a window instance already exists.
        """
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before opening the Sensor Signal Viewer."
        ):
            return

        if self.sensor_window is not None and self.sensor_window.isVisible():
            self._focus_window(self.sensor_window)
            return

        self.sensor_window = SensorSignalWindow(
            parent=self,
            sensor_functions_ref=self.ui_ros.sensor_functions
        )
        self.sensor_window.show()


    def open_ps5_controller_test_window(self):
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before opening the PS5 Controller Test."
        ):
            return

        if hasattr(self.ui_ros, 'open_ps5_controller_test_dialog'):
            self.ui_ros.open_ps5_controller_test_dialog()
        else:
            QMessageBox.information(self, "Info", "PS5 Controller Test is not available in the current UI.")

    def open_sensor_controller_test_window(self):
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before opening the Sensor Controller Test."
        ):
            return

        if hasattr(self.ui_ros, 'open_sensor_controller_test_dialog'):
            self.ui_ros.open_sensor_controller_test_dialog()
        else:
            QMessageBox.information(self, "Info", "Sensor Controller Test is not available in the current UI.")


    def open_direct_finger_motion_params_window(self):
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before opening the DFM parameter editor."
        ):
            return

        if hasattr(self.ui_ros, 'open_direct_finger_motion_settings_dialog'):
            self.ui_ros.open_direct_finger_motion_settings_dialog()
        else:
            QMessageBox.information(self, "Info", "DFM parameter editor is not available in the current UI.")

    def open_proximity_control_params_window(self):
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before opening the Proximity parameter editor."
        ):
            return

        if hasattr(self.ui_ros, 'open_proximity_settings_dialog'):
            self.ui_ros.open_proximity_settings_dialog()
        else:
            QMessageBox.information(self, "Info", "Proximity parameter editor is not available in the current UI.")

    def open_console_control_params_window(self):
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before opening the Console Control parameter editor."
        ):
            return

        if hasattr(self.ui_ros, 'open_console_control_settings_dialog'):
            self.ui_ros.open_console_control_settings_dialog()
        else:
            QMessageBox.information(self, "Info", "Console Control parameter editor is not available in the current UI.")

    def open_sensor_params_window(self):
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before opening the Sensor parameter editor."
        ):
            return

        if hasattr(self.ui_ros, 'open_sensor_parameters_dialog'):
            self.ui_ros.open_sensor_parameters_dialog()
        else:
            QMessageBox.information(self, "Info", "Sensor parameter editor is not available in the current UI.")

    def run_ping_mode(self):
        """Initializes and displays the main 'Ping Mode' UI, replacing the placeholder."""
        if self.ping_mode_entered or self._startup_requested:
            return

        self._startup_requested = True
        self.ping_mode_entered = True
        self.action_ping_mode.setEnabled(False)  # Disable the action to prevent re-entry

        # Remove the placeholder and add the actual complex UI
        self.view_placeholder.deleteLater()
        self.ui_ros = UI(Qt.Horizontal)
        self.view_container.layout().addWidget(self.ui_ros)
        self.ui_ros.reLayout()  # Call the UI's own layout adjustment

        self._start_fps_counter()
        self.info_process.setText("PING Mode Activated")

    def _start_fps_counter(self):
        """Finds the 3D renderer and sets up a timer to track and display FPS."""
        self._frame_count = 0

        iren = self.ui_ros.findChild(QtInteractor)
        if iren is None:
            QMessageBox.critical(self, "Fatal Error", "The QtInteractor for the 3D view could not be found.")
            self.close()
            return

        iren.render_signal.connect(self._on_frame_rendered)
        self._fps_render_interactor = iren

        if self._fps_timer is None:
            self._fps_timer = QTimer(self)
            self._fps_timer.timeout.connect(self._update_fps_display)
        self._fps_timer.start(1000)  # Update FPS count every second

    def _on_frame_rendered(self):
        """This slot is called every time a frame is rendered in the 3D view."""
        self._frame_count += 1

    def _update_fps_display(self):
        """Updates the FPS label and device indicators in the status bar."""
        self.info_fps.setText(f"{self._frame_count:3d} fps")
        self._frame_count = 0
        try:
            self._refresh_status_indicators()
        except Exception:
            # The status bar is cosmetic: never let it interrupt the session.
            pass

    def _toggle_log_view(self):
        """Toggle the modeless log console without resizing the workspace."""
        if self.ui_ros:
            self.ui_ros.toggle_plotter_visibility()
        else:
            QMessageBox.information(self, "Info", "Log is only available after Ping Mode has started.")

    def _append_sidebar_control_message(self, message: str):
        # The status bar carries the message, so the log stays as the user
        # left it. Experiment tasks tick once a second; forcing the log open
        # on each one used to shrink the 3D view mid-run.
        self.info_process.setText(message)
        if self.ui_ros is not None and hasattr(self.ui_ros, "log_display"):
            try:
                self.ui_ros.log_display.append(message)
                return
            except Exception:
                pass
        print(message)

    def _refresh_experiment_status(self) -> None:
        """Keep a compact stop affordance visible when the popup is hidden."""
        indicator = getattr(self, "status_experiment", None)
        stop_button = getattr(self, "status_experiment_stop_button", None)
        travel_label = getattr(self, "status_calibration_travel", None)
        if indicator is None or stop_button is None:
            return

        task = getattr(self, "_sidebar_active_task", None)
        running = task is not None
        if running:
            task_label = str(
                getattr(task, "label", None)
                or getattr(task, "id", None)
                or "Experiment"
            )
            indicator.set_state("active", f"Experiment: {task_label}")
            indicator.setToolTip(f"Running experiment: {task_label}")
        indicator.setVisible(running)
        stop_button.setVisible(running)
        stop_button.setEnabled(running)
        if travel_label is not None:
            travel_label.setVisible(
                running
                and str(getattr(task, "id", ""))
                in {"calibration", "full_taxel_calibration"}
            )
        self._refresh_taxel_navigation_controls()

    def _current_sidebar_task_item(self):
        tree = getattr(self, "sidebar_task_list", None)
        return tree.currentItem() if tree is not None else None

    def _sidebar_item_value(self, item, role):
        if item is None:
            return None
        return item.data(0, role)

    def select_sidebar_task(self, task_id: str) -> bool:
        """Select a task by id, expanding its category when needed."""
        tree = getattr(self, "sidebar_task_list", None)
        if tree is None:
            return False
        for category_index in range(tree.topLevelItemCount()):
            category = tree.topLevelItem(category_index)
            for child_index in range(category.childCount()):
                item = category.child(child_index)
                if self._sidebar_item_value(item, Qt.UserRole) != task_id:
                    continue
                category.setExpanded(True)
                tree.setCurrentItem(item)
                return True
        return False

    def _current_sidebar_control_id(self):
        item = self._current_sidebar_task_item()
        task_id = self._sidebar_item_value(item, Qt.UserRole)
        if task_id == "calibration" and self._calibration_mode_is_full_taxel():
            return "full_taxel_calibration"
        return task_id

    def _calibration_mode_is_full_taxel(self) -> bool:
        tabs = getattr(self, "calibration_mode_tabs", None)
        full_index = getattr(self, "_calibration_full_taxel_tab_index", None)
        return (
            tabs is not None
            and full_index is not None
            and tabs.currentIndex() == int(full_index)
        )

    def _on_calibration_mode_changed(self, _index):
        if self._current_sidebar_task_item() is None:
            return
        item = self._current_sidebar_task_item()
        if self._sidebar_item_value(item, Qt.UserRole) != "calibration":
            return
        self._on_sidebar_task_changed()

    def _current_sidebar_control_label(self):
        item = self._current_sidebar_task_item()
        return item.text(0) if item is not None else "No Task"

    def _update_sidebar_task_info(self):
        label = getattr(self, "sidebar_task_info", None)
        item = self._current_sidebar_task_item()
        if label is None:
            return
        if item is None:
            label.setText("Select a task from the list.")
            return
        description = self._sidebar_item_value(item, Qt.UserRole + 1) or ""
        label.setText(f"Task: {item.text(0)}\n\n{description}")

    def _on_sidebar_task_changed(self, _current=None, _previous=None):
        self._update_sidebar_task_info()
        self._update_sidebar_task_param_visibility()
        self._update_sidebar_task_action_labels()
        self._refresh_calibration_controls()
        self._refresh_full_taxel_calibration_controls()
        self._refresh_taxel_mapping_controls()

    def _update_sidebar_task_action_labels(self):
        start_button = getattr(self, "sidebar_btn_start", None)
        if start_button is None:
            return
        task = experiment_tasks.get_task(self._current_sidebar_control_id())
        start_button.setText(
            str(getattr(task, "start_button_label", "Start") or "Start")
        )
        # Reset stale enablement inherited from the previously selected task.
        # Hardware tasks apply their stricter prerequisites immediately after
        # this method; one-shot offline tasks such as Graph Analysis remain
        # available without any connected device.
        start_button.setEnabled(
            task is not None
            and getattr(self, "_sidebar_active_task", None) is None
        )
        reverse_button = getattr(self, "sigraph_scanning_reverse_button", None)
        if reverse_button is not None:
            is_sigraph = getattr(task, "id", None) == "sigraph2026_scanning"
            reverse_button.setVisible(is_sigraph)
            reverse_button.setEnabled(
                is_sigraph
                and bool(getattr(task, "recorded_points", []))
                and getattr(task, "_running_index", None) is None
            )
        is_sigraph = getattr(task, "id", None) == "sigraph2026_scanning"
        for button_name in (
            "sigraph_scanning_run_record_button",
            "sigraph_scanning_reverse_record_button",
        ):
            record_button = getattr(self, button_name, None)
            if record_button is None:
                continue
            record_button.setVisible(is_sigraph)
            record_button.setEnabled(
                is_sigraph
                and bool(getattr(task, "recorded_points", []))
                and getattr(task, "_running_index", None) is None
            )

    # --- Per-task parameter widgets in the sidebar ----------------------------
    def _build_sidebar_task_param_widgets(self, parent_layout):
        """Build the optional per-task parameter editors shown in the sidebar.

        Each entry maps a ``task_id`` to a small QWidget that mutates the task
        instance directly when the user adjusts a value. Only the widget for
        the currently-selected task is visible at a time.
        """
        self._task_param_widgets: dict = {}

        task = experiment_tasks.get_task("sigraph2026_scanning")
        if task is None:
            return
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.setSpacing(8)

        set_row = QHBoxLayout()
        set_row.addWidget(QLabel("Point set"))
        self.sigraph_scanning_set_combo = QComboBox()
        self.sigraph_scanning_set_combo.setToolTip(
            "Select which saved group of scanning points to edit or run"
        )
        self.sigraph_scanning_new_set_button = QPushButton("New Set")
        self.sigraph_scanning_new_set_button.setToolTip(
            "Create and switch to an empty point set; existing sets are kept"
        )
        set_row.addWidget(self.sigraph_scanning_set_combo, 1)
        set_row.addWidget(self.sigraph_scanning_new_set_button)
        panel_layout.addLayout(set_row)

        self.sigraph_scanning_count_label = QLabel("Recorded points: 0")
        panel_layout.addWidget(self.sigraph_scanning_count_label)
        self.sigraph_scanning_point_list = QListWidget()
        self.sigraph_scanning_point_list.setSelectionMode(
            QListWidget.SingleSelection
        )
        self.sigraph_scanning_point_list.setMinimumHeight(150)
        panel_layout.addWidget(self.sigraph_scanning_point_list)

        record_row = QHBoxLayout()
        self.sigraph_scanning_record_button = QPushButton("Record Current Point")
        self.sigraph_scanning_copy_button = QPushButton("Copy")
        self.sigraph_scanning_copy_button.setToolTip(
            "Duplicate the selected point and append it as the latest point"
        )
        self.sigraph_scanning_remove_button = QPushButton("Remove")
        self.sigraph_scanning_clear_button = QPushButton("Clear")
        components.apply_variant(self.sigraph_scanning_clear_button, "danger")
        record_row.addWidget(self.sigraph_scanning_record_button, 1)
        record_row.addWidget(self.sigraph_scanning_copy_button)
        record_row.addWidget(self.sigraph_scanning_remove_button)
        record_row.addWidget(self.sigraph_scanning_clear_button)
        panel_layout.addLayout(record_row)

        order_row = QHBoxLayout()
        self.sigraph_scanning_go_button = QPushButton("Go to Selected")
        self.sigraph_scanning_up_button = QToolButton()
        self.sigraph_scanning_up_button.setIcon(
            self.style().standardIcon(QStyle.SP_ArrowUp)
        )
        self.sigraph_scanning_up_button.setToolTip("Move selected point earlier")
        self.sigraph_scanning_down_button = QToolButton()
        self.sigraph_scanning_down_button.setIcon(
            self.style().standardIcon(QStyle.SP_ArrowDown)
        )
        self.sigraph_scanning_down_button.setToolTip("Move selected point later")
        order_row.addWidget(self.sigraph_scanning_go_button, 1)
        order_row.addStretch()
        order_row.addWidget(self.sigraph_scanning_up_button)
        order_row.addWidget(self.sigraph_scanning_down_button)
        panel_layout.addLayout(order_row)

        settings_form = QFormLayout()
        self.sigraph_scanning_velocity_spin = QDoubleSpinBox()
        self.sigraph_scanning_velocity_spin.setRange(0.01, 1.0)
        self.sigraph_scanning_velocity_spin.setDecimals(2)
        self.sigraph_scanning_velocity_spin.setSingleStep(0.05)
        self.sigraph_scanning_velocity_spin.setValue(
            float(task.velocity_rad_s)
        )
        self.sigraph_scanning_velocity_spin.setSuffix(" rad/s")
        settings_form.addRow("Joint velocity", self.sigraph_scanning_velocity_spin)
        self.sigraph_scanning_dwell_spin = QDoubleSpinBox()
        self.sigraph_scanning_dwell_spin.setRange(0.0, 10.0)
        self.sigraph_scanning_dwell_spin.setDecimals(2)
        self.sigraph_scanning_dwell_spin.setSingleStep(0.1)
        self.sigraph_scanning_dwell_spin.setValue(float(task.dwell_sec))
        self.sigraph_scanning_dwell_spin.setSuffix(" s")
        settings_form.addRow("Point dwell", self.sigraph_scanning_dwell_spin)
        panel_layout.addLayout(settings_form)

        self.sigraph_scanning_replay_button = QPushButton(
            "Replay Heatmap Recording"
        )
        self.sigraph_scanning_replay_button.setObjectName("btnReplayHeatmap")
        self.sigraph_scanning_replay_button.setIcon(
            self.style().standardIcon(QStyle.SP_MediaPlay)
        )
        self.sigraph_scanning_replay_button.setToolTip(
            "Replay saved tactile frames in the 3D heatmap without moving the robot"
        )
        panel_layout.addWidget(self.sigraph_scanning_replay_button)

        self.sigraph_scanning_record_button.clicked.connect(
            self._record_sigraph_scanning_point
        )
        self.sigraph_scanning_set_combo.currentIndexChanged.connect(
            self._select_sigraph_scanning_point_set
        )
        self.sigraph_scanning_new_set_button.clicked.connect(
            self._create_sigraph_scanning_point_set
        )
        self.sigraph_scanning_copy_button.clicked.connect(
            self._copy_sigraph_scanning_point
        )
        self.sigraph_scanning_remove_button.clicked.connect(
            self._remove_sigraph_scanning_point
        )
        self.sigraph_scanning_clear_button.clicked.connect(
            self._clear_sigraph_scanning_points
        )
        self.sigraph_scanning_go_button.clicked.connect(
            self._go_to_selected_sigraph_scanning_point
        )
        self.sigraph_scanning_up_button.clicked.connect(
            lambda: self._move_sigraph_scanning_point(-1)
        )
        self.sigraph_scanning_down_button.clicked.connect(
            lambda: self._move_sigraph_scanning_point(1)
        )
        self.sigraph_scanning_velocity_spin.valueChanged.connect(
            lambda value: setattr(task, "velocity_rad_s", float(value))
        )
        self.sigraph_scanning_dwell_spin.valueChanged.connect(
            lambda value: setattr(task, "dwell_sec", float(value))
        )
        self.sigraph_scanning_replay_button.clicked.connect(
            self._choose_sigraph_heatmap_recording_for_playback
        )
        self._task_param_widgets[task.id] = panel
        parent_layout.addWidget(panel)
        self._refresh_sigraph_scanning_points()

        calibration_task = experiment_tasks.get_task("calibration")
        if calibration_task is None:
            return
        calibration_panel = QFrame()
        calibration_panel.setFrameShape(QFrame.StyledPanel)
        calibration_outer = QVBoxLayout(calibration_panel)
        calibration_outer.setContentsMargins(0, 0, 0, 0)
        calibration_outer.setSpacing(0)
        self.calibration_mode_tabs = QTabWidget()
        self.calibration_mode_tabs.setDocumentMode(True)
        single_page = QWidget()
        calibration_layout = QVBoxLayout(single_page)
        calibration_layout.setContentsMargins(8, 8, 8, 8)
        calibration_layout.setSpacing(8)

        self.calibration_initial_position_label = QLabel(
            "Load a taxel position map, then select a taxel."
        )
        self.calibration_initial_position_label.setWordWrap(True)
        calibration_layout.addWidget(self.calibration_initial_position_label)
        self.calibration_taxel_map = CalibrationTaxelMap(rows=8, columns=10)
        self.calibration_taxel_map.set_taxels([])
        self.calibration_taxel_map.taxelSelected.connect(
            self._select_single_calibration_taxel
        )
        calibration_layout.addWidget(
            self.calibration_taxel_map, alignment=Qt.AlignHCenter
        )
        self.calibration_load_map_button = QPushButton(
            "Load Taxel Position Map…"
        )
        self.calibration_load_map_button.setToolTip(
            "Use a taxel_position_map.json from Taxel Position Mapping. "
            "The selected taxel's saved start pose is the calibration "
            "initial position."
        )
        self.calibration_load_map_button.clicked.connect(
            self._load_single_calibration_map
        )
        calibration_layout.addWidget(self.calibration_load_map_button)

        calibration_form = QFormLayout()
        self.calibration_taxel_spin = QSpinBox()
        self.calibration_taxel_spin.setRange(0, 79)
        self.calibration_taxel_spin.setValue(int(calibration_task.taxel_index))
        self.calibration_taxel_spin.setToolTip(
            "Physical taxel under the indenter and the initially displayed "
            "node. Every taxel signal is recorded for later graph selection. "
            "Indices follow the Sensor Signal Viewer: top-down, column-major."
        )
        self.calibration_taxel_label = QLabel("Physical target taxel (8x10)")
        calibration_form.addRow(
            self.calibration_taxel_label,
            self.calibration_taxel_spin,
        )

        self.calibration_signal_combo = QComboBox()
        self.calibration_signal_combo.addItem(
            "Sensor signal change", "diff_ave"
        )
        self.calibration_signal_combo.addItem(
            "Relative change (%)", "diff_percent_ave"
        )
        self.calibration_signal_combo.addItem("Raw value", "raw_ave")
        self.calibration_signal_combo.setToolTip(
            "Sensor signal change is the signed response: raw value minus "
            "calibration value. Positive and negative values are kept."
        )
        signal_index = self.calibration_signal_combo.findData(
            calibration_task.signal_field
        )
        if signal_index >= 0:
            self.calibration_signal_combo.setCurrentIndex(signal_index)
        calibration_form.addRow("Plotted signal", self.calibration_signal_combo)

        self.calibration_speed_spin = QDoubleSpinBox()
        self.calibration_speed_spin.setRange(0.01, 2.0)
        self.calibration_speed_spin.setDecimals(2)
        self.calibration_speed_spin.setSingleStep(0.01)
        self.calibration_speed_spin.setSuffix(" mm/s")
        self.calibration_speed_spin.setValue(
            float(calibration_task.approach_speed_m_s) * 1000.0
        )
        calibration_form.addRow("Base -Z speed", self.calibration_speed_spin)

        self.calibration_force_threshold_spin = QDoubleSpinBox()
        self.calibration_force_threshold_spin.setRange(
            0.1,
            float(calibration_task.MAX_CONTACT_THRESHOLD_G),
        )
        self.calibration_force_threshold_spin.setDecimals(1)
        self.calibration_force_threshold_spin.setSingleStep(1.0)
        self.calibration_force_threshold_spin.setSuffix(" g")
        self.calibration_force_threshold_spin.setValue(
            float(calibration_task.contact_threshold_g)
        )
        self.calibration_force_threshold_spin.setToolTip(
            "Stop base -Z motion when the absolute DYLY force change from the "
            "stationary baseline reaches this value."
        )
        calibration_form.addRow(
            "Max contact threshold", self.calibration_force_threshold_spin
        )

        self.calibration_max_travel_spin = QDoubleSpinBox()
        self.calibration_max_travel_spin.setRange(0.0, 200.0)
        self.calibration_max_travel_spin.setDecimals(1)
        self.calibration_max_travel_spin.setSingleStep(0.5)
        self.calibration_max_travel_spin.setSuffix(" mm")
        self.calibration_max_travel_spin.setSpecialValueText("No limit")
        self.calibration_max_travel_spin.setToolTip(
            "Maximum downward travel for one contact trial. Set to No limit "
            "to disable the distance guard; force freshness, contact threshold, "
            "and Stop remain active."
        )
        self.calibration_max_travel_spin.setValue(
            float(calibration_task.max_travel_m) * 1000.0
        )
        calibration_form.addRow("Maximum travel", self.calibration_max_travel_spin)

        self.calibration_timeout_spin = QDoubleSpinBox()
        self.calibration_timeout_spin.setRange(0.0, 1800.0)
        self.calibration_timeout_spin.setDecimals(1)
        self.calibration_timeout_spin.setSingleStep(10.0)
        self.calibration_timeout_spin.setSuffix(" s")
        self.calibration_timeout_spin.setSpecialValueText("No limit")
        self.calibration_timeout_spin.setToolTip(
            "Maximum time for one contact trial. Set to No limit to disable "
            "the timer; force freshness, maximum travel, and Stop remain active."
        )
        self.calibration_timeout_spin.setValue(float(calibration_task.timeout_sec))
        calibration_form.addRow("Timeout", self.calibration_timeout_spin)

        self.calibration_repeat_spin = QSpinBox()
        self.calibration_repeat_spin.setRange(1, 1000)
        self.calibration_repeat_spin.setValue(int(calibration_task.repeat_count))
        self.calibration_repeat_spin.setToolTip(
            "After each contact, return to the remembered pose, recalibrate, "
            "and repeat this many times"
        )
        calibration_form.addRow("Repetitions", self.calibration_repeat_spin)
        calibration_layout.addLayout(calibration_form)

        calibration_note = QLabel(
            "Select one mapped taxel. Run moves to that taxel's saved start "
            "pose, measures the DYLY baseline, and moves in base -Z until the "
            "gram threshold is reached. It holds for 1 s, returns in base +Z "
            "at the same speed, and completes the trial only at that start "
            "height. The tactile sensor recalibrates between repetitions."
        )
        calibration_note.setWordWrap(True)
        calibration_note.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        calibration_layout.addWidget(calibration_note)

        self.calibration_progress_label = QLabel("Ready")
        self.calibration_progress_label.setWordWrap(True)
        self.calibration_progress_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-weight: 600;"
        )
        calibration_layout.addWidget(self.calibration_progress_label)
        self.calibration_live_force_label = QLabel(
            "Force: -- g   |   Contact change: -- g"
        )
        self.calibration_live_force_label.setStyleSheet(
            theme.readout_style("caption")
        )
        calibration_layout.addWidget(self.calibration_live_force_label)

        def _remember_calibration_taxel(value):
            if getattr(calibration_task, "position_map", None) is not None:
                self._select_single_calibration_taxel(int(value))
                return
            calibration_task.taxel_index = int(value)
            sensor_key = str(
                getattr(calibration_task, "active_sensor_key_name", "") or ""
            )
            if sensor_key:
                calibration_task._taxel_by_sensor[sensor_key] = int(value)

        self.calibration_taxel_spin.valueChanged.connect(
            _remember_calibration_taxel
        )
        self.calibration_signal_combo.currentIndexChanged.connect(
            lambda _index: setattr(
                calibration_task,
                "signal_field",
                str(self.calibration_signal_combo.currentData()),
            )
        )
        self.calibration_speed_spin.valueChanged.connect(
            lambda value: setattr(
                calibration_task, "approach_speed_m_s", float(value) / 1000.0
            )
        )
        self.calibration_force_threshold_spin.valueChanged.connect(
            lambda value: setattr(
                calibration_task, "contact_threshold_g", float(value)
            )
        )
        self.calibration_max_travel_spin.valueChanged.connect(
            lambda value: setattr(
                calibration_task, "max_travel_m", float(value) / 1000.0
            )
        )
        self.calibration_timeout_spin.valueChanged.connect(
            lambda value: setattr(calibration_task, "timeout_sec", float(value))
        )
        self.calibration_repeat_spin.valueChanged.connect(
            lambda value: setattr(calibration_task, "repeat_count", int(value))
        )
        self.calibration_repeat_spin.valueChanged.connect(
            lambda _value: self._refresh_calibration_progress()
        )
        self._calibration_single_tab_index = self.calibration_mode_tabs.addTab(
            single_page, "Single taxel"
        )
        self._refresh_calibration_controls()
        self._refresh_calibration_progress()

        full_calibration_task = experiment_tasks.get_task(
            "full_taxel_calibration"
        )
        if full_calibration_task is None:
            return
        full_panel = QFrame()
        full_panel.setFrameShape(QFrame.StyledPanel)
        full_layout = QVBoxLayout(full_panel)
        full_layout.setContentsMargins(8, 8, 8, 8)
        full_layout.setSpacing(8)

        self.full_taxel_initial_position_label = QLabel(
            "Initial position: not remembered"
        )
        self.full_taxel_initial_position_label.setWordWrap(True)
        full_layout.addWidget(self.full_taxel_initial_position_label)
        self.full_taxel_remember_position_button = QPushButton(
            "Remember Initial Position"
        )
        self.full_taxel_remember_position_button.setToolTip(
            "Save the robot's current joint and TCP pose as the start and "
            "return position for Full Taxel Calibration. This pose is kept "
            "separate from Single taxel calibration, and each sensor keeps "
            "its own pose."
        )
        self.full_taxel_remember_position_button.clicked.connect(
            self._remember_full_taxel_initial_position
        )
        full_layout.addWidget(self.full_taxel_remember_position_button)

        self.full_taxel_calibration_map_info = QLabel(
            "Position map: not loaded"
        )
        self.full_taxel_calibration_map_info.setWordWrap(True)
        full_layout.addWidget(self.full_taxel_calibration_map_info)
        self.full_taxel_calibration_map = CalibrationTaxelMap(
            rows=8, columns=10
        )
        self.full_taxel_calibration_map.set_taxels([])
        self.full_taxel_calibration_map.set_interaction_enabled(False)
        full_layout.addWidget(
            self.full_taxel_calibration_map,
            alignment=Qt.AlignHCenter,
        )
        self.full_taxel_calibration_load_map_button = QPushButton(
            "Load Taxel Position Map…"
        )
        self.full_taxel_calibration_load_map_button.setToolTip(
            "Load the taxel_position_map.json created by Taxel Position Mapping. "
            "Loading a map does not move the robot."
        )
        self.full_taxel_calibration_load_map_button.clicked.connect(
            self._load_full_taxel_calibration_map
        )
        full_layout.addWidget(self.full_taxel_calibration_load_map_button)

        signal_form = QFormLayout()
        self.full_taxel_calibration_signal_combo = QComboBox()
        self.full_taxel_calibration_signal_combo.addItem(
            "Sensor signal change", "diff_ave"
        )
        self.full_taxel_calibration_signal_combo.addItem(
            "Relative change (%)", "diff_percent_ave"
        )
        self.full_taxel_calibration_signal_combo.addItem(
            "Raw value", "raw_ave"
        )
        signal_index = self.full_taxel_calibration_signal_combo.findData(
            full_calibration_task.signal_field
        )
        if signal_index >= 0:
            self.full_taxel_calibration_signal_combo.setCurrentIndex(
                signal_index
            )
        self.full_taxel_calibration_signal_combo.setToolTip(
            "Each taxel folder keeps only that physical target's graph signal. "
            "Sensor signal change remains signed."
        )
        signal_form.addRow(
            "Plotted signal", self.full_taxel_calibration_signal_combo
        )
        full_layout.insertLayout(0, signal_form)

        full_form = QFormLayout()

        self.full_taxel_calibration_speed_spin = QDoubleSpinBox()
        self.full_taxel_calibration_speed_spin.setRange(0.01, 2.0)
        self.full_taxel_calibration_speed_spin.setDecimals(2)
        self.full_taxel_calibration_speed_spin.setSingleStep(0.01)
        self.full_taxel_calibration_speed_spin.setSuffix(" mm/s")
        self.full_taxel_calibration_speed_spin.setValue(
            float(full_calibration_task.approach_speed_m_s) * 1000.0
        )
        full_form.addRow(
            "Base -Z speed", self.full_taxel_calibration_speed_spin
        )

        self.full_taxel_calibration_route_speed_spin = QDoubleSpinBox()
        self.full_taxel_calibration_route_speed_spin.setRange(1.0, 100.0)
        self.full_taxel_calibration_route_speed_spin.setDecimals(1)
        self.full_taxel_calibration_route_speed_spin.setSingleStep(1.0)
        self.full_taxel_calibration_route_speed_spin.setSuffix(" mm/s")
        self.full_taxel_calibration_route_speed_spin.setValue(
            float(full_calibration_task.line_speed_m_s) * 1000.0
        )
        self.full_taxel_calibration_route_speed_spin.setToolTip(
            "Straight-line speed used only for saved clearance and non-contact "
            "routing moves between taxels."
        )
        full_form.addRow(
            "Safe routing speed",
            self.full_taxel_calibration_route_speed_spin,
        )

        self.full_taxel_calibration_force_threshold_spin = QDoubleSpinBox()
        self.full_taxel_calibration_force_threshold_spin.setRange(
            0.1, float(full_calibration_task.MAX_CONTACT_THRESHOLD_G)
        )
        self.full_taxel_calibration_force_threshold_spin.setDecimals(1)
        self.full_taxel_calibration_force_threshold_spin.setSingleStep(1.0)
        self.full_taxel_calibration_force_threshold_spin.setSuffix(" g")
        self.full_taxel_calibration_force_threshold_spin.setValue(
            float(full_calibration_task.contact_threshold_g)
        )
        full_form.addRow(
            "Max contact threshold",
            self.full_taxel_calibration_force_threshold_spin,
        )

        self.full_taxel_calibration_route_guard_spin = QDoubleSpinBox()
        self.full_taxel_calibration_route_guard_spin.setRange(1.0, 2000.0)
        self.full_taxel_calibration_route_guard_spin.setDecimals(1)
        self.full_taxel_calibration_route_guard_spin.setSingleStep(1.0)
        self.full_taxel_calibration_route_guard_spin.setSuffix(" g")
        self.full_taxel_calibration_route_guard_spin.setValue(
            float(full_calibration_task.route_force_guard_g)
        )
        self.full_taxel_calibration_route_guard_spin.setToolTip(
            "Stop if force changes unexpectedly during a clearance routing move."
        )
        full_form.addRow(
            "Routing force guard",
            self.full_taxel_calibration_route_guard_spin,
        )

        self.full_taxel_calibration_max_travel_spin = QDoubleSpinBox()
        self.full_taxel_calibration_max_travel_spin.setRange(0.0, 200.0)
        self.full_taxel_calibration_max_travel_spin.setDecimals(1)
        self.full_taxel_calibration_max_travel_spin.setSingleStep(0.5)
        self.full_taxel_calibration_max_travel_spin.setSuffix(" mm")
        self.full_taxel_calibration_max_travel_spin.setSpecialValueText(
            "No limit"
        )
        self.full_taxel_calibration_max_travel_spin.setValue(
            float(full_calibration_task.max_travel_m) * 1000.0
        )
        full_form.addRow(
            "Maximum travel", self.full_taxel_calibration_max_travel_spin
        )

        self.full_taxel_calibration_timeout_spin = QDoubleSpinBox()
        self.full_taxel_calibration_timeout_spin.setRange(0.0, 1800.0)
        self.full_taxel_calibration_timeout_spin.setDecimals(1)
        self.full_taxel_calibration_timeout_spin.setSingleStep(10.0)
        self.full_taxel_calibration_timeout_spin.setSuffix(" s")
        self.full_taxel_calibration_timeout_spin.setSpecialValueText(
            "No limit"
        )
        self.full_taxel_calibration_timeout_spin.setValue(
            float(full_calibration_task.timeout_sec)
        )
        full_form.addRow(
            "Timeout", self.full_taxel_calibration_timeout_spin
        )

        self.full_taxel_calibration_repeat_spin = QSpinBox()
        self.full_taxel_calibration_repeat_spin.setRange(1, 1000)
        self.full_taxel_calibration_repeat_spin.setValue(
            int(full_calibration_task.repeat_count)
        )
        self.full_taxel_calibration_repeat_spin.setToolTip(
            "Number of times to scan every mapped taxel. Each scan visits all "
            "taxels once before the next scan starts."
        )
        full_form.addRow(
            "Full scans",
            self.full_taxel_calibration_repeat_spin,
        )
        full_layout.addLayout(full_form)

        full_note = QLabel(
            "Each full scan visits every mapped taxel once: route at the saved "
            "clearance height, descend to its non-contact start pose, measure "
            "force baseline, approach in base -Z, return in base +Z, and save "
            "only that taxel's signal. The sensor recalibrates at clearance "
            "before the next taxel. The next full scan starts after the whole "
            "set has been visited."
        )
        full_note.setWordWrap(True)
        full_note.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        full_layout.addWidget(full_note)

        self.full_taxel_calibration_progress_label = QLabel("Ready")
        self.full_taxel_calibration_progress_label.setWordWrap(True)
        self.full_taxel_calibration_progress_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-weight: 600;"
        )
        full_layout.addWidget(self.full_taxel_calibration_progress_label)
        self.full_taxel_calibration_live_force_label = QLabel(
            "Force: -- g   |   Contact change: -- g"
        )
        self.full_taxel_calibration_live_force_label.setStyleSheet(
            theme.readout_style("caption")
        )
        full_layout.addWidget(self.full_taxel_calibration_live_force_label)

        self.full_taxel_calibration_signal_combo.currentIndexChanged.connect(
            lambda _index: setattr(
                full_calibration_task,
                "signal_field",
                str(self.full_taxel_calibration_signal_combo.currentData()),
            )
        )
        self.full_taxel_calibration_speed_spin.valueChanged.connect(
            lambda value: setattr(
                full_calibration_task,
                "approach_speed_m_s",
                float(value) / 1000.0,
            )
        )
        self.full_taxel_calibration_route_speed_spin.valueChanged.connect(
            lambda value: setattr(
                full_calibration_task,
                "line_speed_m_s",
                float(value) / 1000.0,
            )
        )
        self.full_taxel_calibration_force_threshold_spin.valueChanged.connect(
            lambda value: setattr(
                full_calibration_task,
                "contact_threshold_g",
                float(value),
            )
        )
        self.full_taxel_calibration_route_guard_spin.valueChanged.connect(
            lambda value: setattr(
                full_calibration_task,
                "route_force_guard_g",
                float(value),
            )
        )
        self.full_taxel_calibration_max_travel_spin.valueChanged.connect(
            lambda value: setattr(
                full_calibration_task,
                "max_travel_m",
                float(value) / 1000.0,
            )
        )
        self.full_taxel_calibration_timeout_spin.valueChanged.connect(
            lambda value: setattr(
                full_calibration_task, "timeout_sec", float(value)
            )
        )
        self.full_taxel_calibration_repeat_spin.valueChanged.connect(
            lambda value: setattr(
                full_calibration_task, "repeat_count", int(value)
            )
        )
        self.full_taxel_calibration_repeat_spin.valueChanged.connect(
            lambda _value: self._refresh_full_taxel_calibration_controls()
        )

        full_scroll = QScrollArea()
        full_scroll.setFrameShape(QFrame.NoFrame)
        full_scroll.setWidgetResizable(True)
        full_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        full_scroll.setWidget(full_panel)
        self._calibration_full_taxel_tab_index = (
            self.calibration_mode_tabs.addTab(full_scroll, "Full taxel")
        )
        self.calibration_mode_tabs.currentChanged.connect(
            self._on_calibration_mode_changed
        )
        calibration_outer.addWidget(self.calibration_mode_tabs)
        self._task_param_widgets[calibration_task.id] = calibration_panel
        self._task_param_widgets[full_calibration_task.id] = calibration_panel
        parent_layout.addWidget(calibration_panel)
        self._refresh_full_taxel_calibration_controls()
        self._refresh_full_taxel_calibration_progress()

        mapping_task = experiment_tasks.get_task("taxel_position_mapping")
        if mapping_task is None:
            return
        mapping_panel = QFrame()
        mapping_panel.setFrameShape(QFrame.StyledPanel)
        mapping_layout = QVBoxLayout(mapping_panel)
        mapping_layout.setContentsMargins(8, 8, 8, 8)
        mapping_layout.setSpacing(8)

        navigation_title = QLabel("Click a mapped taxel to move above it")
        navigation_title.setWordWrap(True)
        mapping_layout.addWidget(navigation_title)
        self.taxel_mapping_map_info = QLabel("No position map yet. Run mapping or load a saved map.")
        self.taxel_mapping_map_info.setWordWrap(True)
        mapping_layout.addWidget(self.taxel_mapping_map_info)
        self.taxel_mapping_map = CalibrationTaxelMap(rows=8, columns=10)
        self.taxel_mapping_map.set_taxels([])
        self.taxel_mapping_map.taxelSelected.connect(self._move_above_mapped_taxel)
        mapping_layout.addWidget(self.taxel_mapping_map, alignment=Qt.AlignHCenter)
        self.taxel_mapping_load_map_button = QPushButton("Load Saved Map…")
        self.taxel_mapping_load_map_button.clicked.connect(self._load_taxel_position_map)
        mapping_layout.addWidget(self.taxel_mapping_load_map_button)
        self.taxel_mapping_resume_button = QPushButton("Resume Loaded Map")
        self.taxel_mapping_resume_button.setEnabled(False)
        self.taxel_mapping_resume_button.setToolTip(
            "Load an incomplete taxel_position_map.json first. Resume keeps "
            "its measured taxels and verifies Taxel 35 before moving on."
        )
        self.taxel_mapping_resume_button.clicked.connect(
            self._resume_loaded_taxel_map
        )
        mapping_layout.addWidget(self.taxel_mapping_resume_button)
        self.taxel_mapping_relocate_button = QPushButton(
            "Relocate Loaded Map from New Anchor"
        )
        self.taxel_mapping_relocate_button.setEnabled(False)
        self.taxel_mapping_relocate_button.setToolTip(
            "Load a completed map, move the sensor, position the robot safely "
            "and approximately above its centre taxel, remember the new anchor, "
            "then use this button. A new map is saved; the original is kept."
        )
        self.taxel_mapping_relocate_button.clicked.connect(
            self._relocate_loaded_taxel_map
        )
        mapping_layout.addWidget(self.taxel_mapping_relocate_button)
        self.taxel_mapping_move_status_label = QLabel(
            "Moves use the saved clearance Z height. Grey cells are unmapped. "
            "Use saved maps only with the same sensor placement, robot base and tool."
        )
        self.taxel_mapping_move_status_label.setWordWrap(True)
        self.taxel_mapping_move_status_label.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        mapping_layout.addWidget(self.taxel_mapping_move_status_label)

        self.taxel_mapping_anchor_label = QLabel(
            "Taxel 35 anchor: not remembered"
        )
        self.taxel_mapping_anchor_label.setWordWrap(True)
        mapping_layout.addWidget(self.taxel_mapping_anchor_label)
        self.taxel_mapping_anchor_button = QPushButton(
            "Remember Current Pose Above Taxel 35"
        )
        self.taxel_mapping_anchor_button.setToolTip(
            "Save the current TCP pose above Taxel 35. Its height is the safe "
            "travel height for the whole run and may be well above the skin."
        )
        mapping_layout.addWidget(self.taxel_mapping_anchor_button)

        mapping_form = QFormLayout()
        self.taxel_mapping_detection_combo = QComboBox()
        self.taxel_mapping_detection_combo.addItem(
            "Robust downward signal change (recommended)", "robust_shift"
        )
        self.taxel_mapping_detection_combo.addItem(
            "Lower baseline envelope (legacy)", "adaptive_range"
        )
        self.taxel_mapping_detection_combo.addItem(
            "Fixed downward signal change", "fixed"
        )
        detection_index = self.taxel_mapping_detection_combo.findData(
            str(mapping_task.proximity_detection_mode)
        )
        self.taxel_mapping_detection_combo.setCurrentIndex(
            max(0, detection_index)
        )
        self.taxel_mapping_detection_combo.setToolTip(
            "Robust mode learns the normal variation of rolling medians at "
            "the safe anchor and detects a persistent downward distribution shift."
        )
        mapping_form.addRow(
            "Proximity detection", self.taxel_mapping_detection_combo
        )

        self.taxel_mapping_signal_threshold_spin = QDoubleSpinBox()
        self.taxel_mapping_signal_threshold_spin.setRange(1.0, 1000000.0)
        self.taxel_mapping_signal_threshold_spin.setDecimals(1)
        self.taxel_mapping_signal_threshold_spin.setSingleStep(10.0)
        self.taxel_mapping_signal_threshold_spin.setSuffix(" sensor units")
        self.taxel_mapping_signal_threshold_spin.setValue(
            float(mapping_task.signal_threshold_counts)
        )
        self.taxel_mapping_signal_threshold_spin.setToolTip(
            "Required decrease in the signed sensor signal when Fixed "
            "downward signal change is selected."
        )
        mapping_form.addRow(
            "Fixed signal change", self.taxel_mapping_signal_threshold_spin
        )

        self.taxel_mapping_adaptive_percent_spin = QDoubleSpinBox()
        self.taxel_mapping_adaptive_percent_spin.setRange(0.0, 500.0)
        self.taxel_mapping_adaptive_percent_spin.setDecimals(1)
        self.taxel_mapping_adaptive_percent_spin.setSingleStep(5.0)
        self.taxel_mapping_adaptive_percent_spin.setSuffix(" %")
        self.taxel_mapping_adaptive_percent_spin.setValue(
            float(mapping_task.adaptive_margin_fraction) * 100.0
        )
        self.taxel_mapping_adaptive_percent_spin.setToolTip(
            "Extra margin outside the recorded baseline range. With a -5 to "
            "+5 range, 30% places the legacy downward trigger at -8."
        )
        mapping_form.addRow(
            "Legacy envelope margin",
            self.taxel_mapping_adaptive_percent_spin,
        )

        self.taxel_mapping_adaptive_min_spin = QDoubleSpinBox()
        self.taxel_mapping_adaptive_min_spin.setRange(0.1, 1000.0)
        self.taxel_mapping_adaptive_min_spin.setDecimals(1)
        self.taxel_mapping_adaptive_min_spin.setSingleStep(0.5)
        self.taxel_mapping_adaptive_min_spin.setSuffix(" sensor units")
        self.taxel_mapping_adaptive_min_spin.setValue(
            float(mapping_task.adaptive_min_margin_counts)
        )
        self.taxel_mapping_adaptive_min_spin.setToolTip(
            "Minimum persistent downward sensor change in robust mode, or minimum "
            "margin below the baseline minimum in legacy-envelope mode."
        )
        mapping_form.addRow(
            "Minimum downward signal change",
            self.taxel_mapping_adaptive_min_spin,
        )

        self.taxel_mapping_baseline_frames_spin = QSpinBox()
        self.taxel_mapping_baseline_frames_spin.setRange(3, 200)
        self.taxel_mapping_baseline_frames_spin.setValue(
            int(mapping_task.baseline_frame_count)
        )
        self.taxel_mapping_baseline_frames_spin.setToolTip(
            "Number of stationary fresh frames used to learn the safe-anchor distribution."
        )
        mapping_form.addRow(
            "Baseline frames", self.taxel_mapping_baseline_frames_spin
        )

        self.taxel_mapping_probe_window_spin = QSpinBox()
        self.taxel_mapping_probe_window_spin.setRange(3, 51)
        self.taxel_mapping_probe_window_spin.setSingleStep(2)
        self.taxel_mapping_probe_window_spin.setValue(
            int(mapping_task.probe_window_frames)
        )
        self.taxel_mapping_probe_window_spin.setToolTip(
            "Window used to learn normal rolling-median noise from the "
            "stationary baseline."
        )
        mapping_form.addRow(
            "Baseline smoothing window", self.taxel_mapping_probe_window_spin
        )

        self.taxel_mapping_probe_sample_frames_spin = QSpinBox()
        self.taxel_mapping_probe_sample_frames_spin.setRange(3, 100)
        self.taxel_mapping_probe_sample_frames_spin.setValue(
            int(mapping_task.probe_sample_frames)
        )
        self.taxel_mapping_probe_sample_frames_spin.setToolTip(
            "At every downward Z level, the robot stops and takes this many "
            "fresh sensor frames. Their median is used for proximity detection."
        )
        mapping_form.addRow(
            "Frames per probe level",
            self.taxel_mapping_probe_sample_frames_spin,
        )

        self.taxel_mapping_probe_step_spin = QDoubleSpinBox()
        self.taxel_mapping_probe_step_spin.setRange(0.1, 5.0)
        self.taxel_mapping_probe_step_spin.setDecimals(2)
        self.taxel_mapping_probe_step_spin.setSingleStep(0.1)
        self.taxel_mapping_probe_step_spin.setSuffix(" mm")
        self.taxel_mapping_probe_step_spin.setValue(
            float(mapping_task.probe_step_m) * 1000.0
        )
        self.taxel_mapping_probe_step_spin.setToolTip(
            "Vertical distance between stationary sampling levels. A smaller "
            "step gives a more precise onset position but takes longer."
        )
        mapping_form.addRow(
            "Probe depth step", self.taxel_mapping_probe_step_spin
        )

        self.taxel_mapping_scan_strength_spin = QDoubleSpinBox()
        self.taxel_mapping_scan_strength_spin.setRange(1.0, 10.0)
        self.taxel_mapping_scan_strength_spin.setDecimals(2)
        self.taxel_mapping_scan_strength_spin.setSingleStep(0.25)
        self.taxel_mapping_scan_strength_spin.setSuffix(" × onset")
        self.taxel_mapping_scan_strength_spin.setValue(
            float(mapping_task.centre_scan_signal_multiplier)
        )
        self.taxel_mapping_scan_strength_spin.setToolTip(
            "After remembering the first confirmed proximity height, descend "
            "until the downward signal change reaches this multiple of the "
            "learned onset threshold. The stronger level is used for XY scans."
        )
        mapping_form.addRow(
            "Centre-scan signal strength",
            self.taxel_mapping_scan_strength_spin,
        )

        self.taxel_mapping_scan_extra_descent_spin = QDoubleSpinBox()
        self.taxel_mapping_scan_extra_descent_spin.setRange(0.5, 20.0)
        self.taxel_mapping_scan_extra_descent_spin.setDecimals(1)
        self.taxel_mapping_scan_extra_descent_spin.setSingleStep(0.5)
        self.taxel_mapping_scan_extra_descent_spin.setSuffix(" mm")
        self.taxel_mapping_scan_extra_descent_spin.setValue(
            float(mapping_task.centre_scan_max_extra_descent_m) * 1000.0
        )
        self.taxel_mapping_scan_extra_descent_spin.setToolTip(
            "Safety limit after proximity onset. If the requested signal "
            "strength is not reached within this distance, use the strongest "
            "safe sampled level instead of descending farther."
        )
        mapping_form.addRow(
            "Maximum descent after onset",
            self.taxel_mapping_scan_extra_descent_spin,
        )

        self.taxel_mapping_reference_speed_spin = QDoubleSpinBox()
        self.taxel_mapping_reference_speed_spin.setRange(
            0.01, float(mapping_task.MAX_PROBE_SPEED_M_S) * 1000.0
        )
        self.taxel_mapping_reference_speed_spin.setDecimals(2)
        self.taxel_mapping_reference_speed_spin.setSingleStep(0.01)
        self.taxel_mapping_reference_speed_spin.setSuffix(" mm/s")
        self.taxel_mapping_reference_speed_spin.setValue(
            float(mapping_task.reference_probe_speed_m_s) * 1000.0
        )
        self.taxel_mapping_reference_speed_spin.setToolTip(
            "Speed of the single long probe from the anchor that learns the "
            "sensor height at Taxel 35. Higher speeds increase stopping distance."
        )
        mapping_form.addRow(
            "First probe speed", self.taxel_mapping_reference_speed_spin
        )

        self.taxel_mapping_reference_descent_spin = QDoubleSpinBox()
        self.taxel_mapping_reference_descent_spin.setRange(
            1.0, float(mapping_task.MAX_PROBE_DESCENT_M) * 1000.0
        )
        self.taxel_mapping_reference_descent_spin.setDecimals(1)
        self.taxel_mapping_reference_descent_spin.setSingleStep(1.0)
        self.taxel_mapping_reference_descent_spin.setSuffix(" mm")
        self.taxel_mapping_reference_descent_spin.setValue(
            float(mapping_task.reference_max_descent_m) * 1000.0
        )
        self.taxel_mapping_reference_descent_spin.setToolTip(
            "Furthest the first probe may descend below the anchor. Keep it "
            "shorter than the anchor-to-skin distance; the force stop is the "
            "backup if it is not."
        )
        mapping_form.addRow(
            "First probe max descent", self.taxel_mapping_reference_descent_spin
        )

        self.taxel_mapping_start_margin_spin = QDoubleSpinBox()
        self.taxel_mapping_start_margin_spin.setRange(0.0, 10.0)
        self.taxel_mapping_start_margin_spin.setDecimals(1)
        self.taxel_mapping_start_margin_spin.setSingleStep(0.5)
        self.taxel_mapping_start_margin_spin.setSuffix(" mm")
        self.taxel_mapping_start_margin_spin.setValue(
            float(mapping_task.probe_start_margin_m) * 1000.0
        )
        self.taxel_mapping_start_margin_spin.setToolTip(
            "Every later probe starts this far above the height where the "
            "first probe detected Taxel 35."
        )
        mapping_form.addRow(
            "Probe start above learned height",
            self.taxel_mapping_start_margin_spin,
        )

        self.taxel_mapping_probe_speed_spin = QDoubleSpinBox()
        self.taxel_mapping_probe_speed_spin.setRange(
            0.01, float(mapping_task.MAX_PROBE_SPEED_M_S) * 1000.0
        )
        self.taxel_mapping_probe_speed_spin.setDecimals(2)
        self.taxel_mapping_probe_speed_spin.setSingleStep(0.01)
        self.taxel_mapping_probe_speed_spin.setSuffix(" mm/s")
        self.taxel_mapping_probe_speed_spin.setValue(
            float(mapping_task.probe_speed_m_s) * 1000.0
        )
        self.taxel_mapping_probe_speed_spin.setToolTip(
            "Vertical speed between stationary sampling levels. The task "
            "stops at every level before reading the sensor."
        )
        mapping_form.addRow("Probe speed", self.taxel_mapping_probe_speed_spin)

        self.taxel_mapping_confirmation_speed_spin = QDoubleSpinBox()
        self.taxel_mapping_confirmation_speed_spin.setRange(
            0.01, float(mapping_task.MAX_PROBE_SPEED_M_S) * 1000.0
        )
        self.taxel_mapping_confirmation_speed_spin.setDecimals(2)
        self.taxel_mapping_confirmation_speed_spin.setSingleStep(0.05)
        self.taxel_mapping_confirmation_speed_spin.setSuffix(" mm/s")
        self.taxel_mapping_confirmation_speed_spin.setValue(
            float(mapping_task.probe_confirmation_speed_m_s) * 1000.0
        )
        self.taxel_mapping_confirmation_speed_spin.setToolTip(
            "Slower speed used to move to the next sampling level after the "
            "first candidate proximity response."
        )
        mapping_form.addRow(
            "Confirmation speed",
            self.taxel_mapping_confirmation_speed_spin,
        )

        self.taxel_mapping_descent_spin = QDoubleSpinBox()
        self.taxel_mapping_descent_spin.setRange(
            0.1, float(mapping_task.MAX_PROBE_DESCENT_M) * 1000.0
        )
        self.taxel_mapping_descent_spin.setDecimals(2)
        self.taxel_mapping_descent_spin.setSingleStep(0.5)
        self.taxel_mapping_descent_spin.setSuffix(" mm")
        self.taxel_mapping_descent_spin.setValue(
            float(mapping_task.max_probe_descent_m) * 1000.0
        )
        self.taxel_mapping_descent_spin.setToolTip(
            "Furthest each later probe may descend from its start height. "
            "Use large values only when the path is clear and the live force "
            "safety guard is available."
        )
        mapping_form.addRow("Probe max descent", self.taxel_mapping_descent_spin)

        self.taxel_mapping_force_guard_spin = QDoubleSpinBox()
        self.taxel_mapping_force_guard_spin.setRange(1.0, 2000.0)
        self.taxel_mapping_force_guard_spin.setDecimals(1)
        self.taxel_mapping_force_guard_spin.setSingleStep(10.0)
        self.taxel_mapping_force_guard_spin.setSuffix(" g")
        self.taxel_mapping_force_guard_spin.setValue(
            float(mapping_task.force_guard_g)
        )
        self.taxel_mapping_force_guard_spin.setToolTip(
            "Stop the complete task if force changes this much from the "
            "safe-height baseline."
        )
        mapping_form.addRow("Unexpected-force stop", self.taxel_mapping_force_guard_spin)

        self.taxel_mapping_coarse_radius_spin = QDoubleSpinBox()
        self.taxel_mapping_coarse_radius_spin.setRange(1.0, 30.0)
        self.taxel_mapping_coarse_radius_spin.setDecimals(1)
        self.taxel_mapping_coarse_radius_spin.setSingleStep(0.5)
        self.taxel_mapping_coarse_radius_spin.setSuffix(" mm")
        self.taxel_mapping_coarse_radius_spin.setValue(
            float(mapping_task.coarse_radius_m) * 1000.0
        )
        mapping_form.addRow("Neighbour-search radius", self.taxel_mapping_coarse_radius_spin)

        self.taxel_mapping_coarse_step_spin = QDoubleSpinBox()
        self.taxel_mapping_coarse_step_spin.setRange(0.1, 5.0)
        self.taxel_mapping_coarse_step_spin.setDecimals(2)
        self.taxel_mapping_coarse_step_spin.setSingleStep(0.1)
        self.taxel_mapping_coarse_step_spin.setSuffix(" mm")
        self.taxel_mapping_coarse_step_spin.setValue(
            float(mapping_task.coarse_step_m) * 1000.0
        )
        mapping_form.addRow("Neighbour-search step", self.taxel_mapping_coarse_step_spin)

        self.taxel_mapping_local_radius_spin = QDoubleSpinBox()
        self.taxel_mapping_local_radius_spin.setRange(0.5, 10.0)
        self.taxel_mapping_local_radius_spin.setDecimals(1)
        self.taxel_mapping_local_radius_spin.setSingleStep(0.5)
        self.taxel_mapping_local_radius_spin.setSuffix(" mm")
        self.taxel_mapping_local_radius_spin.setValue(
            float(mapping_task.local_radius_m) * 1000.0
        )
        self.taxel_mapping_local_radius_spin.setToolTip(
            "Half-length of the row and column line scans through each taxel. "
            "It is capped at 0.6 of the measured taxel pitch."
        )
        mapping_form.addRow("Local scan half-width", self.taxel_mapping_local_radius_spin)

        self.taxel_mapping_local_step_spin = QDoubleSpinBox()
        self.taxel_mapping_local_step_spin.setRange(0.1, 2.0)
        self.taxel_mapping_local_step_spin.setDecimals(2)
        self.taxel_mapping_local_step_spin.setSingleStep(0.1)
        self.taxel_mapping_local_step_spin.setSuffix(" mm")
        self.taxel_mapping_local_step_spin.setValue(
            float(mapping_task.local_step_m) * 1000.0
        )
        mapping_form.addRow("Local scan step", self.taxel_mapping_local_step_spin)

        self.taxel_mapping_line_speed_spin = QSpinBox()
        self.taxel_mapping_line_speed_spin.setRange(1, 100)
        self.taxel_mapping_line_speed_spin.setSuffix(" mm/s")
        self.taxel_mapping_line_speed_spin.setValue(
            int(round(float(mapping_task.line_speed_m_s) * 1000.0))
        )
        self.taxel_mapping_line_speed_spin.setToolTip(
            "Speed of every straight-line tool move. The robot driver only "
            "accepts whole mm/s."
        )
        mapping_form.addRow("Tool line speed", self.taxel_mapping_line_speed_spin)

        self.taxel_mapping_xy_limit_spin = QDoubleSpinBox()
        self.taxel_mapping_xy_limit_spin.setRange(10.0, 200.0)
        self.taxel_mapping_xy_limit_spin.setDecimals(1)
        self.taxel_mapping_xy_limit_spin.setSingleStep(5.0)
        self.taxel_mapping_xy_limit_spin.setSuffix(" mm")
        self.taxel_mapping_xy_limit_spin.setValue(
            float(mapping_task.max_xy_offset_m) * 1000.0
        )
        self.taxel_mapping_xy_limit_spin.setToolTip(
            "Reject any fitted taxel position farther than this XY distance "
            "from the remembered Taxel 35 anchor."
        )
        mapping_form.addRow("Maximum XY offset", self.taxel_mapping_xy_limit_spin)
        mapping_layout.addLayout(mapping_form)

        self.taxel_mapping_note = QLabel(
            "The default robust detector learns 100 safe-anchor frames, then "
            "moves downward in small steps. At every level the robot stops, "
            "collects a fresh frame batch, and evaluates its median. It ignores "
            "positive changes and records the first confirmed proximity Z-position. "
            "It then continues downward at confirmation speed until the signal "
            "reaches the configured centre-scan strength (2× by default), giving "
            "the XY scan a clearer peak. The "
            "first probe from the anchor "
            "learns the sensor height at Taxel 35. "
            "After that, every taxel is approached at the anchor height, "
            "probed from just above the learned height, centred with row and "
            "column line scans, and verified using a 20-frame centre sample. "
            "All moves are straight lines. A missing taxel "
            "returns to Taxel 35 and retries once; a second failure returns "
            "to Taxel 35 and stops. The force transmitter connects automatically "
            "and robot motion waits for a fresh safety reading."
        )
        self.taxel_mapping_note.setWordWrap(True)
        self.taxel_mapping_note.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        mapping_layout.addWidget(self.taxel_mapping_note)
        self.taxel_mapping_progress_label = QLabel("Ready | 0/80 located")
        self.taxel_mapping_progress_label.setWordWrap(True)
        self.taxel_mapping_progress_label.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; font-weight: 600;"
        )
        mapping_layout.addWidget(self.taxel_mapping_progress_label)

        self.taxel_mapping_anchor_button.clicked.connect(
            self._remember_taxel_mapping_anchor
        )
        self.taxel_mapping_detection_combo.currentIndexChanged.connect(
            lambda _index: setattr(
                mapping_task,
                "proximity_detection_mode",
                str(self.taxel_mapping_detection_combo.currentData()),
            )
        )
        self.taxel_mapping_signal_threshold_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "signal_threshold_counts", float(value))
        )
        self.taxel_mapping_adaptive_percent_spin.valueChanged.connect(
            lambda value: setattr(
                mapping_task, "adaptive_margin_fraction", float(value) / 100.0
            )
        )
        self.taxel_mapping_adaptive_min_spin.valueChanged.connect(
            lambda value: setattr(
                mapping_task, "adaptive_min_margin_counts", float(value)
            )
        )
        self.taxel_mapping_baseline_frames_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "baseline_frame_count", int(value))
        )
        self.taxel_mapping_probe_window_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "probe_window_frames", int(value))
        )
        self.taxel_mapping_probe_sample_frames_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "probe_sample_frames", int(value))
        )
        self.taxel_mapping_probe_step_spin.valueChanged.connect(
            lambda value: setattr(
                mapping_task, "probe_step_m", float(value) / 1000.0
            )
        )
        self.taxel_mapping_scan_strength_spin.valueChanged.connect(
            lambda value: setattr(
                mapping_task,
                "centre_scan_signal_multiplier",
                float(value),
            )
        )
        self.taxel_mapping_scan_extra_descent_spin.valueChanged.connect(
            lambda value: setattr(
                mapping_task,
                "centre_scan_max_extra_descent_m",
                float(value) / 1000.0,
            )
        )
        self.taxel_mapping_reference_speed_spin.valueChanged.connect(
            lambda value: setattr(
                mapping_task, "reference_probe_speed_m_s", float(value) / 1000.0
            )
        )
        self.taxel_mapping_reference_descent_spin.valueChanged.connect(
            lambda value: setattr(
                mapping_task, "reference_max_descent_m", float(value) / 1000.0
            )
        )
        self.taxel_mapping_start_margin_spin.valueChanged.connect(
            lambda value: setattr(
                mapping_task, "probe_start_margin_m", float(value) / 1000.0
            )
        )
        self.taxel_mapping_probe_speed_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "probe_speed_m_s", float(value) / 1000.0)
        )
        self.taxel_mapping_confirmation_speed_spin.valueChanged.connect(
            lambda value: setattr(
                mapping_task,
                "probe_confirmation_speed_m_s",
                float(value) / 1000.0,
            )
        )
        self.taxel_mapping_descent_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "max_probe_descent_m", float(value) / 1000.0)
        )
        self.taxel_mapping_force_guard_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "force_guard_g", float(value))
        )
        self.taxel_mapping_coarse_radius_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "coarse_radius_m", float(value) / 1000.0)
        )
        self.taxel_mapping_coarse_step_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "coarse_step_m", float(value) / 1000.0)
        )
        self.taxel_mapping_local_radius_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "local_radius_m", float(value) / 1000.0)
        )
        self.taxel_mapping_local_step_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "local_step_m", float(value) / 1000.0)
        )
        self.taxel_mapping_line_speed_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "line_speed_m_s", float(value) / 1000.0)
        )
        self.taxel_mapping_xy_limit_spin.valueChanged.connect(
            lambda value: setattr(mapping_task, "max_xy_offset_m", float(value) / 1000.0)
        )
        mapping_scroll = QScrollArea()
        mapping_scroll.setFrameShape(QFrame.NoFrame)
        mapping_scroll.setWidgetResizable(True)
        mapping_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mapping_scroll.setWidget(mapping_panel)
        self._task_param_widgets[mapping_task.id] = mapping_scroll
        parent_layout.addWidget(mapping_scroll)
        self._refresh_taxel_mapping_controls()
        self._refresh_taxel_mapping_progress()

    def _streaming_sensor_shape(self) -> tuple[int, int]:
        sensor = getattr(getattr(self, "ui_ros", None), "sensor_functions", None)
        if sensor is None or not bool(getattr(sensor, "is_connected", False)):
            return 0, 0
        try:
            rows = int(getattr(sensor, "n_row", 0) or 0)
            columns = int(getattr(sensor, "n_col", 0) or 0)
        except (TypeError, ValueError):
            return 0, 0
        if rows <= 0 or columns <= 0:
            return 0, 0
        return rows, columns

    def _sync_taxel_maps_to_streaming_sensor(self) -> None:
        """Keep both taxel maps on the grid size of the live sensor stream."""
        if getattr(self, "_sidebar_active_task", None) is not None:
            return
        rows, columns = self._streaming_sensor_shape()
        if rows <= 0 or columns <= 0:
            return
        mapping_grid = getattr(self, "taxel_mapping_map", None)
        if mapping_grid is not None and (mapping_grid.rows, mapping_grid.columns) != (
            rows,
            columns,
        ):
            mapping_grid.set_grid_shape(rows, columns)
            self._paint_loaded_taxel_navigation_map()
        calibration_grid = getattr(self, "calibration_taxel_map", None)
        if calibration_grid is not None and (
            calibration_grid.rows,
            calibration_grid.columns,
        ) != (rows, columns):
            self._refresh_calibration_controls()

    def _paint_loaded_taxel_navigation_map(self) -> None:
        prepared = getattr(self, "_taxel_navigation_result", None)
        grid = getattr(self, "taxel_mapping_map", None)
        info = getattr(self, "taxel_mapping_map_info", None)
        if prepared is None or grid is None:
            return
        settings = prepared.get("settings") or {}
        try:
            file_rows = int(settings["sensor_rows"])
            file_columns = int(settings["sensor_columns"])
        except (KeyError, TypeError, ValueError):
            file_rows, file_columns = 8, 10
        if (grid.rows, grid.columns) != (file_rows, file_columns):
            grid.set_taxels([])
            if info is not None:
                info.setText(
                    f"Streaming sensor is {grid.rows}x{grid.columns}. "
                    f"The loaded map is {file_rows}x{file_columns}."
                )
            return
        records = list(prepared.get("positions") or [])
        skip_border = bool(settings.get("skip_outer_border", False))
        skipped = set()
        if skip_border:
            skipped = {
                index
                for index in range(file_rows * file_columns)
                if index % file_rows in (0, file_rows - 1)
                or index // file_rows in (0, file_columns - 1)
            }
        grid.set_taxels(
            (record["taxel_index"] for record in records),
            skipped=skipped,
        )
        for index, button in grid.buttons.items():
            if index in skipped:
                button.setToolTip(
                    f"Taxel {index}: skipped because it is on the outer border"
                )
            else:
                button.setToolTip(
                    f"Taxel {index}: no measured position in this map"
                )
        for record in records:
            index = int(record["taxel_index"])
            if index not in grid.buttons:
                continue
            xyz = ", ".join(
                f"{value * 1000.0:.2f}"
                for value in record["safe_approach_position_m"]
            )
            grid.buttons[index].setToolTip(
                f"Click to move above Taxel {index}\n"
                f"Saved clearance TCP: ({xyz}) mm"
            )
        if info is None:
            return
        source = str(
            getattr(self, "_taxel_navigation_source_path", None)
            or prepared.get("json_path")
            or "Current mapping result"
        )
        state = "Completed map" if prepared.get("completed") else "Partial map"
        target_total = file_rows * file_columns - len(skipped)
        mapped_target = sum(
            int(record["taxel_index"]) not in skipped for record in records
        )
        region = " · outer border skipped" if skipped else ""
        info.setText(
            f"{state} · {mapped_target}/{target_total} target taxels available"
            f"{region}\n"
            f"{prepared.get('started_at') or prepared.get('saved_at') or ''}"
        )
        info.setToolTip(source)

    def _show_taxel_position_map(self, result, source_path=None):
        """Display a completed/partial map; loading it never moves the robot."""
        prepared = prepare_taxel_map(result)
        self._taxel_navigation_result = prepared
        self._taxel_mapping_anchor_updated_after_load = False
        self._taxel_navigation_source_path = str(
            source_path or prepared.get("json_path") or ""
        ) or None
        settings = prepared.get("settings") or {}
        try:
            file_rows = int(settings["sensor_rows"])
            file_columns = int(settings["sensor_columns"])
        except (KeyError, TypeError, ValueError):
            file_rows, file_columns = 8, 10
        live_rows, live_columns = self._streaming_sensor_shape()
        if live_rows > 0 and live_columns > 0:
            rows, columns = live_rows, live_columns
            if (rows, columns) != (file_rows, file_columns):
                self._append_sidebar_control_message(
                    f"The loaded map is {file_rows}x{file_columns}, but the "
                    f"streaming sensor is {rows}x{columns}. The taxel map uses "
                    "the streaming sensor size."
                )
        else:
            rows, columns = file_rows, file_columns
        grid = getattr(self, "taxel_mapping_map", None)
        if grid is not None:
            grid.set_grid_shape(rows, columns)
        self._paint_loaded_taxel_navigation_map()
        self._refresh_taxel_navigation_status(
            "Click a numbered cell to move above that taxel at the saved clearance height. "
            "The sensor placement, robot base and tool must match this map."
        )
        self._refresh_taxel_navigation_controls()
        # The same validated map is immediately available to Full Taxel
        # Calibration. Loading remains a read-only action; no motion starts.
        try:
            self._show_full_taxel_calibration_map(
                prepared,
                source_path=source_path,
            )
        except ValueError:
            # Older navigation-only maps can still be used to move above a
            # taxel even when they do not contain the probe-start height that
            # Full Taxel Calibration requires.
            pass
        scroll = self._task_param_widgets.get("taxel_position_mapping")
        if isinstance(scroll, QScrollArea):
            scroll.verticalScrollBar().setValue(0)

    def _load_taxel_position_map(self):
        if self._sidebar_active_task is not None:
            self._append_sidebar_control_message("Stop the current experiment before loading a position map.")
            return
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Load taxel position map",
            resource_path("taxel_position_maps"),
            "Taxel position map (taxel_position_map.json);;JSON files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as stream:
                result = json.load(stream)
            self._show_taxel_position_map(result, source_path=path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._append_sidebar_control_message(f"Could not load the taxel map: {exc}")
            return
        self._append_sidebar_control_message(f"Taxel position map loaded: {path}")

    def _refresh_taxel_navigation_status(self, detail):
        label = getattr(self, "taxel_mapping_move_status_label", None)
        if label is not None:
            label.setText(str(detail))

    def _refresh_taxel_mapping_live_taxels(
        self,
        *,
        measured,
        skipped,
        target_total,
    ):
        """Paint newly saved taxels immediately while motion stays locked."""
        measured = {int(index) for index in measured}
        skipped = {int(index) for index in skipped}
        measured_targets = measured - skipped
        grid = getattr(self, "taxel_mapping_map", None)
        if grid is None:
            return
        grid.set_taxels(measured, skipped=skipped)
        grid.set_interaction_enabled(False)
        for index, button in grid.buttons.items():
            if index in measured:
                button.setToolTip(
                    f"Taxel {index}: position saved during this mapping run"
                )
            elif index in skipped:
                button.setToolTip(
                    f"Taxel {index}: skipped because it is on the outer border"
                )
            else:
                button.setToolTip(f"Taxel {index}: waiting to be mapped")
        region = " · outer border skipped" if skipped else ""
        self.taxel_mapping_map_info.setText(
            f"Mapping live · {len(measured_targets)}/{int(target_total)} target "
            f"taxels saved{region}\nGreen cells are already saved."
        )

    def _refresh_taxel_navigation_controls(self):
        active = getattr(self, "_sidebar_active_task", None)
        busy = active is not None
        grid = getattr(self, "taxel_mapping_map", None)
        if grid is not None:
            interaction_setter = getattr(
                grid, "set_interaction_enabled", None
            )
            if callable(interaction_setter):
                interaction_setter(not busy)
            else:
                grid.setEnabled(not busy)
        loader = getattr(self, "taxel_mapping_load_map_button", None)
        if loader is not None:
            loader.setEnabled(not busy)
        resume = getattr(self, "taxel_mapping_resume_button", None)
        if resume is not None:
            task = self._taxel_mapping_task()
            valid, message = (
                task.validate_resume_result(self._taxel_navigation_result)
                if task is not None
                else (False, "Taxel Position Mapping is unavailable.")
            )
            resume.setEnabled(not busy and valid)
            resume.setToolTip(
                message
                if valid
                else "Resume unavailable: " + str(message)
            )
        relocate = getattr(self, "taxel_mapping_relocate_button", None)
        if relocate is not None:
            task = self._taxel_mapping_task()
            valid, message = (
                task.validate_relocation_result(self._taxel_navigation_result)
                if task is not None
                else (False, "Taxel Position Mapping is unavailable.")
            )
            anchor_ready = bool(getattr(task, "anchor_pose", None)) and bool(
                getattr(self, "_taxel_mapping_anchor_updated_after_load", False)
            )
            relocate.setEnabled(not busy and valid and anchor_ready)
            if valid and not anchor_ready:
                message = "Remember a new safe anchor above the centre taxel first."
            relocate.setToolTip(
                message
                if valid and anchor_ready
                else "Relocation unavailable: " + str(message)
            )
        keyboard = getattr(self, "action_keyboard_tool_velocity", None)
        if keyboard is not None:
            keyboard.setEnabled(getattr(active, "id", None) != "mapped_taxel_move")

    def _resume_loaded_taxel_map(self):
        if self._sidebar_active_task is not None:
            self._append_sidebar_control_message(
                "Stop the current experiment before resuming a position map."
            )
            return
        if self._keyboard_tool_velocity_enabled:
            self._append_sidebar_control_message(
                "Turn off keyboard tool velocity before resuming taxel mapping."
            )
            return
        task = self._taxel_mapping_task()
        if task is None or self._current_sidebar_control_id() != task.id:
            self._append_sidebar_control_message(
                "Select Taxel Position Mapping before resuming the loaded map."
            )
            return
        accepted, message = task.request_resume(
            self._taxel_navigation_result,
            self._taxel_navigation_source_path,
        )
        if not accepted:
            self._append_sidebar_control_message(
                f"Taxel mapping cannot resume: {message}"
            )
            self._refresh_taxel_navigation_controls()
            return
        self._append_sidebar_control_message(message)
        self._start_sidebar_control()
        if self._sidebar_active_task is not task:
            task._resume_request = None
            task._resume_source_path = None
            self._refresh_taxel_mapping_controls()

    def _relocate_loaded_taxel_map(self):
        if self._sidebar_active_task is not None:
            self._append_sidebar_control_message(
                "Stop the current experiment before relocating a position map."
            )
            return
        if self._keyboard_tool_velocity_enabled:
            self._append_sidebar_control_message(
                "Turn off keyboard tool velocity before relocating the map."
            )
            return
        task = self._taxel_mapping_task()
        if task is None or self._current_sidebar_control_id() != task.id:
            self._append_sidebar_control_message(
                "Select Taxel Position Mapping before relocating the loaded map."
            )
            return
        if task.anchor_pose is None:
            self._append_sidebar_control_message(
                "Move the robot to a safe pose approximately above the moved "
                f"sensor's Taxel {task.anchor_taxel}, then remember that anchor."
            )
            return
        accepted, message = task.request_relocation(
            self._taxel_navigation_result,
            self._taxel_navigation_source_path,
        )
        if not accepted:
            self._append_sidebar_control_message(
                f"Taxel map cannot be relocated: {message}"
            )
            self._refresh_taxel_navigation_controls()
            return
        self._append_sidebar_control_message(message)
        self._start_sidebar_control()
        if self._sidebar_active_task is not task:
            task._relocation_request = None
            task._relocation_source_path = None
            self._refresh_taxel_mapping_controls()

    def _move_above_mapped_taxel(self, index, result=None):
        if self._sidebar_active_task is not None:
            self._append_sidebar_control_message("Stop the current experiment before selecting another taxel.")
            return
        if self._keyboard_tool_velocity_enabled:
            self._append_sidebar_control_message("Turn off keyboard tool velocity before selecting a mapped taxel.")
            return
        mapping_task = self._taxel_mapping_task()
        try:
            task = MappedTaxelMoveTask(
                result if result is not None else self._taxel_navigation_result,
                int(index),
                line_speed_m_s=mapping_task.line_speed_m_s,
                force_guard_g=mapping_task.force_guard_g,
            )
        except (ValueError, TypeError, KeyError) as exc:
            self._append_sidebar_control_message(f"Cannot move to Taxel {index}: {exc}")
            return
        if self._sidebar_control_timer is None:
            self._sidebar_control_timer = QTimer(self)
            self._sidebar_control_timer.timeout.connect(self._run_sidebar_control_action)
        self._sidebar_active_task = task
        self._sidebar_active_control_id = task.id
        try:
            started = task.on_start(self)
        except Exception as exc:
            self._stop_sidebar_control(show_result=False, message=f"Taxel move failed: {exc}")
            return
        if not started:
            self._sidebar_active_task = None
            self._sidebar_active_control_id = None
            self._refresh_taxel_mapping_controls()
            self._refresh_experiment_status()
            return
        self._sidebar_control_timer.start(task.tick_interval_ms)
        self._refresh_experiment_status()

    def _calibration_task(self):
        return experiment_tasks.get_task("calibration")

    def _refresh_calibration_controls(self):
        task = self._calibration_task()
        label = getattr(self, "calibration_initial_position_label", None)
        if task is None:
            return
        if hasattr(task, "use_connected_sensor"):
            task.use_connected_sensor(self)
        sensor_label = str(
            getattr(task, "active_sensor_label", "") or "this sensor"
        )
        taxel_spin = getattr(self, "calibration_taxel_spin", None)
        taxel_label = getattr(self, "calibration_taxel_label", None)
        if taxel_spin is not None and not bool(getattr(task, "_test_active", False)):
            count = max(1, int(task.sensor_rows) * int(task.sensor_columns))
            taxel_spin.blockSignals(True)
            taxel_spin.setRange(0, count - 1)
            if taxel_spin.value() != int(task.taxel_index):
                taxel_spin.setValue(int(task.taxel_index))
            taxel_spin.blockSignals(False)
        if taxel_label is not None:
            taxel_label.setText(
                f"Physical target taxel ({int(task.sensor_rows)}x{int(task.sensor_columns)})"
            )
        initial_position = getattr(task, "initial_position", None) or {}
        mapped = initial_position.get("source") == "taxel_position_map"
        tool_position = task._finite_values(
            initial_position.get("tool_position_m"), 3
        )
        if label is not None:
            if not mapped or tool_position is None:
                label.setText(
                    "Load a taxel position map, then select a taxel. "
                    "Its saved start pose is the initial position."
                )
            else:
                tcp_mm = np.asarray(tool_position, dtype=float) * 1000.0
                label.setText(
                    f"Taxel {int(task.taxel_index)} on {sensor_label}\n"
                    f"Start TCP: ({tcp_mm[0]:.2f}, {tcp_mm[1]:.2f}, "
                    f"{tcp_mm[2]:.2f}) mm"
                )
        grid = getattr(self, "calibration_taxel_map", None)
        if grid is not None and not bool(getattr(task, "_test_active", False)):
            prepared = getattr(task, "position_map", None)
            settings = (prepared or {}).get("settings") or {}
            try:
                file_rows = int(settings["sensor_rows"])
                file_columns = int(settings["sensor_columns"])
            except (KeyError, TypeError, ValueError):
                file_rows = int(task.sensor_rows)
                file_columns = int(task.sensor_columns)
            live_rows, live_columns = self._streaming_sensor_shape()
            if live_rows > 0 and live_columns > 0:
                rows, columns = live_rows, live_columns
            else:
                rows, columns = file_rows, file_columns
            grid.set_grid_shape(rows, columns)
            if prepared is not None and (file_rows, file_columns) == (rows, columns):
                records = list(prepared.get("positions") or [])
                selected = int(task.taxel_index) if mapped else None
                grid.set_taxels(
                    (record["taxel_index"] for record in records),
                    selected=selected,
                )
            elif prepared is not None:
                grid.set_taxels([])
            grid.set_interaction_enabled(True)
        start_button = getattr(self, "sidebar_btn_start", None)
        if (
            start_button is not None
            and self._current_sidebar_control_id() == task.id
        ):
            start_button.setEnabled(
                mapped
                and tool_position is not None
                and not bool(getattr(task, "_test_active", False))
            )

    def _refresh_calibration_progress(
        self, current=None, total=None, phase=None, detail=""
    ):
        task = self._calibration_task()
        label = getattr(self, "calibration_progress_label", None)
        if task is None or label is None:
            return
        current = int(
            getattr(task, "_trial_number", 0) if current is None else current
        )
        total = int(
            getattr(task, "repeat_count", 1) if total is None else total
        )
        phase = str(getattr(task, "_phase", "idle") if phase is None else phase)
        phase_labels = {
            "idle": "Ready",
            "connecting_force_meter": "Connecting force transmitter",
            "force_baseline": "Force baseline",
            "approaching": "Approaching",
            "contact_hold": "Contact reached; holding",
            "returning": "Returning in base +Z",
            "recalibrating": "Updating sensor calibration",
            "recovering_force_meter": "Recovering force transmitter",
        }
        status = str(detail or phase_labels.get(phase, phase.replace("_", " ").title()))
        if current > 0:
            label.setText(f"Trial {current}/{max(1, total)} | {status}")
        else:
            label.setText(f"Ready | {max(1, total)} repetition(s)")

    def _set_calibration_controls_running(self, running):
        active_task = getattr(self, "_sidebar_active_task", None)
        if str(getattr(active_task, "id", "")) == "full_taxel_calibration":
            self._set_full_taxel_calibration_controls_running(running)
            return
        running = bool(running)
        for name in (
            "calibration_taxel_spin",
            "calibration_signal_combo",
            "calibration_speed_spin",
            "calibration_force_threshold_spin",
            "calibration_max_travel_spin",
            "calibration_timeout_spin",
            "calibration_repeat_spin",
            "calibration_load_map_button",
            "calibration_mode_tabs",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not running)
        start_button = getattr(self, "sidebar_btn_start", None)
        if start_button is not None:
            start_button.setEnabled(not running)
        grid = getattr(self, "calibration_taxel_map", None)
        if grid is not None:
            grid.set_interaction_enabled(not running)
        if not running:
            self._refresh_calibration_controls()

    def _refresh_calibration_live_force(
        self,
        force_g=None,
        force_delta_g=None,
    ):
        active_task = getattr(self, "_sidebar_active_task", None)
        if str(getattr(active_task, "id", "")) == "full_taxel_calibration":
            self._refresh_full_taxel_calibration_live_force(
                force_g, force_delta_g
            )
            return
        label = getattr(self, "calibration_live_force_label", None)
        if label is None:
            return
        force_text = "--" if force_g is None else f"{float(force_g):+.1f}"
        delta_text = (
            "--"
            if force_delta_g is None
            else f"{float(force_delta_g):+.1f}"
        )
        label.setText(
            f"Force: {force_text} g   |   Contact change: {delta_text} g"
        )

    def _refresh_calibration_travel(
        self,
        current_mm=None,
        maximum_mm=None,
    ):
        """Update the bottom-bar TCP-Z displacement for Calibration."""
        label = getattr(self, "status_calibration_travel", None)
        if label is None:
            return
        try:
            current = float(current_mm)
            maximum = float(maximum_mm)
        except (TypeError, ValueError):
            current = maximum = None
        if (
            current is None
            or maximum is None
            or not np.isfinite(current)
            or not np.isfinite(maximum)
        ):
            label.setText("Z travel: — mm | Max: — mm")
            return
        current = max(0.0, current)
        maximum = max(current, maximum)
        label.setText(
            f"Z travel: {current:.2f} mm | Max: {maximum:.2f} mm"
        )

    def _load_single_calibration_map(self):
        if self._sidebar_active_task is not None:
            self._append_sidebar_control_message(
                "Stop the current experiment before loading a position map."
            )
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load map for single-taxel calibration",
            resource_path("taxel_position_maps"),
            "Taxel position map (taxel_position_map.json);;JSON files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as stream:
                result = json.load(stream)
            prepared = experiment_tasks.FullTaxelCalibrationTask.prepare_position_map(
                result
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._append_sidebar_control_message(
                f"Could not load the calibration taxel map: {exc}"
            )
            return
        task = self._calibration_task()
        if task is None:
            return
        task.position_map = prepared
        if (task.initial_position or {}).get("source") == "taxel_position_map":
            task.initial_position = None
        self._refresh_calibration_controls()
        self._append_sidebar_control_message(
            f"Single-taxel calibration map loaded: {path}"
        )

    def _select_single_calibration_taxel(self, index):
        task = self._calibration_task()
        if task is None or bool(getattr(task, "_test_active", False)):
            return
        prepared = getattr(task, "position_map", None)
        records = {
            int(record["taxel_index"]): record
            for record in list((prepared or {}).get("positions") or [])
            if isinstance(record, dict)
        }
        record = records.get(int(index))
        if record is None:
            self._append_sidebar_control_message(
                f"Taxel {int(index)} has no measured position in the loaded map."
            )
            return
        settings = (prepared or {}).get("settings") or {}
        try:
            rows = int(settings["sensor_rows"])
            columns = int(settings["sensor_columns"])
            task.use_mapped_taxel(record, rows=rows, columns=columns)
        except (KeyError, TypeError, ValueError) as exc:
            self._append_sidebar_control_message(
                f"Cannot use Taxel {int(index)} as the calibration start: {exc}"
            )
            return
        self._refresh_calibration_controls()
        start = task.initial_position["tool_position_m"]
        self._append_sidebar_control_message(
            f"Single-taxel calibration will start at Taxel {int(index)} "
            f"TCP {[round(float(value) * 1000.0, 2) for value in start]} mm."
        )
        self._move_above_mapped_taxel(int(index), prepared)

    def _remember_full_taxel_initial_position(self):
        task = self._full_taxel_calibration_task()
        if task is None or self._current_sidebar_control_id() != task.id:
            return
        task.capture_initial_position(self)

    def _full_taxel_calibration_task(self):
        return experiment_tasks.get_task("full_taxel_calibration")

    @staticmethod
    def _full_taxel_map_skipped(result):
        settings = (result or {}).get("settings") or {}
        if not bool(settings.get("skip_outer_border", False)):
            return set()
        try:
            rows = int(settings.get("sensor_rows") or 8)
            columns = int(settings.get("sensor_columns") or 10)
        except (TypeError, ValueError):
            rows, columns = 8, 10
        if rows <= 0 or columns <= 0:
            return set()
        return {
            index
            for index in range(rows * columns)
            if index % rows in (0, rows - 1)
            or index // rows in (0, columns - 1)
        }

    def _show_full_taxel_calibration_map(self, result, source_path=None):
        task = self._full_taxel_calibration_task()
        if task is None:
            return False
        loaded, message = task.load_position_map(result, source_path)
        if not loaded:
            raise ValueError(message)
        records = task.position_map["positions"]
        skipped = self._full_taxel_map_skipped(task.position_map)
        grid = getattr(self, "full_taxel_calibration_map", None)
        settings = task.position_map.get("settings") or {}
        if grid is not None and hasattr(grid, "set_grid_shape"):
            try:
                grid.set_grid_shape(
                    int(settings.get("sensor_rows") or grid.rows),
                    int(settings.get("sensor_columns") or grid.columns),
                )
            except (TypeError, ValueError):
                pass
        if grid is not None:
            grid.set_taxels(
                (record["taxel_index"] for record in records),
                skipped=skipped,
            )
            grid.set_interaction_enabled(False)
        label = getattr(self, "full_taxel_calibration_map_info", None)
        if label is not None:
            state = "Completed map" if task.position_map.get("completed") else "Partial map"
            source = str(source_path or task.position_map.get("json_path") or "")
            try:
                grid_text = (
                    f"{int(settings.get('sensor_rows'))}x"
                    f"{int(settings.get('sensor_columns'))}"
                )
            except (TypeError, ValueError):
                grid_text = ""
            label.setText(
                f"{state} · {grid_text + ' · ' if grid_text else ''}"
                f"{len(records)} measured taxels · "
                f"{len(records) * max(1, int(task.repeat_count))} total trial(s)"
            )
            label.setToolTip(source)
        self._refresh_full_taxel_calibration_controls()
        self._refresh_full_taxel_calibration_progress()
        return True

    def _load_full_taxel_calibration_map(self):
        if self._sidebar_active_task is not None:
            self._append_sidebar_control_message(
                "Stop the current experiment before loading a position map."
            )
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load map for Full Taxel Calibration",
            resource_path("taxel_position_maps"),
            "Taxel position map (taxel_position_map.json);;JSON files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as stream:
                result = json.load(stream)
            prepared = prepare_taxel_map(result)
            self._show_full_taxel_calibration_map(
                prepared, source_path=path
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._append_sidebar_control_message(
                f"Could not load the full-taxel calibration map: {exc}"
            )
            return
        self._append_sidebar_control_message(
            f"Full Taxel Calibration map loaded: {path}"
        )

    def _refresh_full_taxel_calibration_controls(self):
        task = self._full_taxel_calibration_task()
        if task is None:
            return
        result = getattr(task, "position_map", None)
        records = list((result or {}).get("positions") or [])
        label = getattr(self, "full_taxel_calibration_map_info", None)
        if label is not None and result is None:
            label.setText("Position map: not loaded")
        elif label is not None and not bool(getattr(task, "_test_active", False)):
            state = "Completed map" if result.get("completed") else "Partial map"
            map_settings = result.get("settings") or {}
            try:
                grid_text = (
                    f"{int(map_settings.get('sensor_rows'))}x"
                    f"{int(map_settings.get('sensor_columns'))}"
                )
            except (TypeError, ValueError):
                grid_text = ""
            label.setText(
                f"{state} · {grid_text + ' · ' if grid_text else ''}"
                f"{len(records)} measured taxels · "
                f"{len(records) * max(1, int(task.repeat_count))} total trial(s)"
            )
        if not bool(getattr(task, "_test_active", False)):
            task.use_connected_sensor(self)
        home = task._finite_values(
            (getattr(task, "initial_position", None) or {}).get("tool_position_m"),
            3,
        )
        home_label = getattr(self, "full_taxel_initial_position_label", None)
        sensor_label = str(getattr(task, "active_sensor_label", "") or "this sensor")
        if home_label is not None:
            pose = getattr(task, "initial_position", None)
            if home is None or pose is None:
                home_label.setText(
                    f"Initial position for {sensor_label}: not remembered"
                )
            else:
                joints_deg = np.degrees(
                    np.asarray(pose.get("joints_rad", []), dtype=float)
                )
                tcp = np.asarray(home, dtype=float) * 1000.0
                home_label.setText(
                    f"Initial position for {sensor_label} remembered\n"
                    f"J: {[round(value, 2) for value in joints_deg]} deg\n"
                    f"TCP: ({tcp[0]:.2f}, {tcp[1]:.2f}, {tcp[2]:.2f}) mm"
                )
        remember_button = getattr(self, "full_taxel_remember_position_button", None)
        if remember_button is not None:
            remember_button.setEnabled(not bool(getattr(task, "_test_active", False)))
        start_button = getattr(self, "sidebar_btn_start", None)
        if (
            start_button is not None
            and self._current_sidebar_control_id() == task.id
        ):
            start_button.setEnabled(
                result is not None
                and bool(records)
                and home is not None
                and not bool(getattr(task, "_test_active", False))
            )

    def _refresh_full_taxel_calibration_progress(
        self,
        *,
        current_taxel=None,
        taxel_number=None,
        taxel_total=None,
        trial=None,
        trials=None,
        phase=None,
        detail="",
        completed=None,
    ):
        task = self._full_taxel_calibration_task()
        label = getattr(self, "full_taxel_calibration_progress_label", None)
        if task is None or label is None:
            return
        targets = list(getattr(task, "_targets", []) or [])
        if not targets:
            targets = list((getattr(task, "position_map", None) or {}).get("positions") or [])
        total = len(targets) if taxel_total is None else int(taxel_total)
        number = max(0, int(getattr(task, "_target_cursor", -1)) + 1) if taxel_number is None else int(taxel_number)
        trial = int(getattr(task, "_trial_number", 0)) if trial is None else int(trial)
        trials = max(1, int(getattr(task, "repeat_count", 1))) if trials is None else max(1, int(trials))
        phase = str(getattr(task, "_phase", "idle") if phase is None else phase)
        current_taxel = (
            getattr(task, "taxel_index", None)
            if current_taxel is None and number > 0
            else current_taxel
        )
        completed = list(
            getattr(task, "_completed_taxels", [])
            if completed is None
            else completed
        )
        phase_labels = {
            "idle": "Ready",
            "connecting_force_meter": "Connecting force transmitter",
            "full_moving": "Safe routing move",
            "force_baseline": "Force baseline",
            "approaching": "Approaching",
            "contact_hold": "Contact reached; holding",
            "returning": "Returning in base +Z",
            "recalibrating": "Recalibrating before repetition",
            "full_recalibrating": "Recalibrating at clearance",
            "recovering_force_meter": "Recovering force transmitter",
            "paused_force_meter": "Paused for force transmitter",
        }
        status = str(
            detail or phase_labels.get(phase, phase.replace("_", " ").title())
        )
        if number > 0 and current_taxel is not None:
            label.setText(
                f"Taxel {int(current_taxel)} · {number}/{max(1, total)} | "
                f"Pass {max(1, trial)}/{trials} | {status}"
            )
        elif total > 0:
            label.setText(
                f"Ready | {total} taxels × {trials} full scan(s) = "
                f"{total * trials} trials"
            )
        else:
            label.setText("Ready | Load a taxel position map")
        grid = getattr(self, "full_taxel_calibration_map", None)
        if grid is not None:
            skipped = self._full_taxel_map_skipped(
                getattr(task, "position_map", None)
            )
            active = (
                None
                if phase == "idle" or current_taxel is None
                else int(current_taxel)
            )
            if bool(getattr(task, "_test_active", False)):
                available = set(int(index) for index in completed)
                if active is not None:
                    available.add(active)
            else:
                available = {
                    int(record["taxel_index"])
                    for record in targets
                }
            grid.set_taxels(available, selected=active, skipped=skipped)
            grid.set_interaction_enabled(False)

    def _set_full_taxel_calibration_controls_running(self, running):
        running = bool(running)
        for name in (
            "full_taxel_remember_position_button",
            "full_taxel_calibration_load_map_button",
            "full_taxel_calibration_signal_combo",
            "full_taxel_calibration_speed_spin",
            "full_taxel_calibration_route_speed_spin",
            "full_taxel_calibration_force_threshold_spin",
            "full_taxel_calibration_route_guard_spin",
            "full_taxel_calibration_max_travel_spin",
            "full_taxel_calibration_timeout_spin",
            "full_taxel_calibration_repeat_spin",
            "calibration_mode_tabs",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not running)
        start_button = getattr(self, "sidebar_btn_start", None)
        if start_button is not None:
            start_button.setEnabled(not running)
        if not running:
            self._refresh_full_taxel_calibration_controls()

    def _refresh_full_taxel_calibration_live_force(
        self, force_g=None, force_delta_g=None
    ):
        label = getattr(
            self, "full_taxel_calibration_live_force_label", None
        )
        if label is None:
            return
        force_text = "--" if force_g is None else f"{float(force_g):+.1f}"
        delta_text = (
            "--"
            if force_delta_g is None
            else f"{float(force_delta_g):+.1f}"
        )
        label.setText(
            f"Force: {force_text} g   |   Contact change: {delta_text} g"
        )

    def _taxel_mapping_task(self):
        return experiment_tasks.get_task("taxel_position_mapping")

    def _sync_taxel_mapping_settings(self):
        """Show the settings restored from a loaded partial-map JSON."""
        task = self._taxel_mapping_task()
        if task is None:
            return
        values = {
            "taxel_mapping_signal_threshold_spin": task.signal_threshold_counts,
            "taxel_mapping_adaptive_percent_spin": task.adaptive_margin_fraction * 100.0,
            "taxel_mapping_adaptive_min_spin": task.adaptive_min_margin_counts,
            "taxel_mapping_baseline_frames_spin": task.baseline_frame_count,
            "taxel_mapping_probe_window_spin": task.probe_window_frames,
            "taxel_mapping_probe_sample_frames_spin": task.probe_sample_frames,
            "taxel_mapping_probe_step_spin": task.probe_step_m * 1000.0,
            "taxel_mapping_scan_strength_spin": task.centre_scan_signal_multiplier,
            "taxel_mapping_scan_extra_descent_spin": (
                task.centre_scan_max_extra_descent_m * 1000.0
            ),
            "taxel_mapping_reference_speed_spin": task.reference_probe_speed_m_s * 1000.0,
            "taxel_mapping_reference_descent_spin": task.reference_max_descent_m * 1000.0,
            "taxel_mapping_start_margin_spin": task.probe_start_margin_m * 1000.0,
            "taxel_mapping_probe_speed_spin": task.probe_speed_m_s * 1000.0,
            "taxel_mapping_confirmation_speed_spin": task.probe_confirmation_speed_m_s * 1000.0,
            "taxel_mapping_descent_spin": task.max_probe_descent_m * 1000.0,
            "taxel_mapping_force_guard_spin": task.force_guard_g,
            "taxel_mapping_coarse_radius_spin": task.coarse_radius_m * 1000.0,
            "taxel_mapping_coarse_step_spin": task.coarse_step_m * 1000.0,
            "taxel_mapping_local_radius_spin": task.local_radius_m * 1000.0,
            "taxel_mapping_local_step_spin": task.local_step_m * 1000.0,
            "taxel_mapping_line_speed_spin": round(task.line_speed_m_s * 1000.0),
            "taxel_mapping_xy_limit_spin": task.max_xy_offset_m * 1000.0,
        }
        for name, value in values.items():
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setValue(value)
        detection_combo = getattr(
            self, "taxel_mapping_detection_combo", None
        )
        if detection_combo is not None:
            index = detection_combo.findData(
                str(task.proximity_detection_mode)
            )
            if index >= 0:
                detection_combo.setCurrentIndex(index)

    def _refresh_taxel_mapping_controls(self):
        task = self._taxel_mapping_task()
        if task is None:
            return
        if hasattr(task, "use_connected_sensor") and not bool(
            getattr(task, "_active", False)
        ):
            task.use_connected_sensor(self)
        anchor_taxel = int(getattr(task, "anchor_taxel", 35))
        grid_name = f"{int(task.sensor_rows)}x{int(task.sensor_columns)}"
        sensor_label = str(getattr(task, "active_sensor_label", "") or grid_name)
        anchor = getattr(task, "anchor_pose", None)
        label = getattr(self, "taxel_mapping_anchor_label", None)
        if label is not None:
            if anchor is None:
                label.setText(
                    f"Taxel {anchor_taxel} anchor for {sensor_label}: not remembered"
                )
            else:
                position = np.asarray(anchor.get("tool_position_m", []), dtype=float)
                if position.size >= 3:
                    position_mm = position[:3] * 1000.0
                    label.setText(
                        f"Taxel {anchor_taxel} safe anchor for {sensor_label} remembered\n"
                        f"TCP: ({position_mm[0]:.2f}, {position_mm[1]:.2f}, "
                        f"{position_mm[2]:.2f}) mm"
                    )
                else:
                    label.setText(
                        f"Taxel {anchor_taxel} anchor for {sensor_label}: saved pose is invalid"
                    )
        anchor_button = getattr(self, "taxel_mapping_anchor_button", None)
        if anchor_button is not None:
            anchor_button.setText(
                f"Remember Current Pose Above Taxel {anchor_taxel}"
            )
        note = getattr(self, "taxel_mapping_note", None)
        if note is not None:
            note.setText(
                f"One step-and-sample probe from the anchor learns the sensor "
                f"height at Taxel {anchor_taxel} on this {grid_name} sensor. "
                "At each downward level the robot stops and collects a fresh "
                "batch of sensor frames before deciding whether proximity began. "
                "After that, every taxel is approached at the anchor height, "
                "probed from just above the learned height, centred with row and "
                "column line scans, and verified from a 20-frame sample at its "
                "centre. All moves are straight lines. A missing taxel "
                f"returns to Taxel {anchor_taxel} and retries once; a second failure "
                f"returns to Taxel {anchor_taxel} and stops. A live force transmitter "
                "is mandatory."
            )
        progress = getattr(self, "taxel_mapping_progress_label", None)
        if progress is not None and not bool(getattr(task, "_active", False)):
            total = int(task.sensor_rows) * int(task.sensor_columns)
            progress.setText(f"Ready | 0/{total} located")
        grid = getattr(self, "taxel_mapping_map", None)
        if grid is not None and hasattr(grid, "set_grid_shape") and not bool(
            getattr(task, "_active", False)
        ):
            live_rows, live_columns = self._streaming_sensor_shape()
            loaded = getattr(self, "_taxel_navigation_result", None)
            loaded_settings = (loaded or {}).get("settings") or {}
            try:
                file_rows = int(loaded_settings["sensor_rows"])
                file_columns = int(loaded_settings["sensor_columns"])
            except (KeyError, TypeError, ValueError):
                file_rows = int(task.sensor_rows)
                file_columns = int(task.sensor_columns)
            if live_rows > 0 and live_columns > 0:
                rows, columns = live_rows, live_columns
            else:
                rows, columns = file_rows, file_columns
            grid.set_grid_shape(rows, columns)
            if loaded is not None and (file_rows, file_columns) == (rows, columns):
                self._paint_loaded_taxel_navigation_map()
        start_button = getattr(self, "sidebar_btn_start", None)
        if (
            start_button is not None
            and self._current_sidebar_control_id() == task.id
            and not bool(getattr(task, "_active", False))
        ):
            start_button.setText(str(task.start_button_label))
        running = bool(getattr(task, "_active", False)) or (
            getattr(self, "_sidebar_active_task", None) is not None
        )
        anchor_button = getattr(self, "taxel_mapping_anchor_button", None)
        if anchor_button is not None:
            anchor_button.setEnabled(not running)
        start_button = getattr(self, "sidebar_btn_start", None)
        if (
            start_button is not None
            and self._current_sidebar_control_id() == task.id
        ):
            start_button.setEnabled(anchor is not None and not running)
        self._refresh_taxel_navigation_controls()

    def _refresh_taxel_mapping_progress(
        self,
        mapped=None,
        total=80,
        phase=None,
        detail="",
        current_taxel=None,
        attempt=0,
        max_attempts=2,
    ):
        task = self._taxel_mapping_task()
        label = getattr(self, "taxel_mapping_progress_label", None)
        if task is None or label is None:
            return
        mapped = len(getattr(task, "_positions", {})) if mapped is None else int(mapped)
        total = int(total)
        phase = str(getattr(task, "_phase", "idle") if phase is None else phase)
        phase_text = {
            "idle": "Ready",
            "connecting_force_meter": "Connecting force transmitter",
            "moving": "Robot moving",
            "sensor_calibrating": "Updating full sensor calibration",
            "collecting": "Checking sensor stability / collecting frames",
            "probing": "Searching in base -Z",
        }.get(phase, phase.replace("_", " ").title())
        status = str(detail or phase_text)
        context = []
        if current_taxel is not None:
            context.append(f"Taxel {int(current_taxel)}")
        try:
            attempt = int(attempt)
            max_attempts = max(1, int(max_attempts))
        except (TypeError, ValueError):
            attempt = 0
            max_attempts = 2
        if attempt > 0:
            context.append(
                f"attempt {min(attempt, max_attempts)}/{max_attempts}"
            )
        context_text = f" | {' | '.join(context)}" if context else ""
        label.setText(
            f"{mapped}/{max(1, total)} located{context_text} | {status}"
        )

    def _set_taxel_mapping_controls_running(self, running):
        running = bool(running)
        for name in (
            "sidebar_task_list",
            "taxel_mapping_anchor_button",
            "taxel_mapping_detection_combo",
            "taxel_mapping_signal_threshold_spin",
            "taxel_mapping_adaptive_percent_spin",
            "taxel_mapping_adaptive_min_spin",
            "taxel_mapping_baseline_frames_spin",
            "taxel_mapping_probe_window_spin",
            "taxel_mapping_probe_sample_frames_spin",
            "taxel_mapping_probe_step_spin",
            "taxel_mapping_scan_strength_spin",
            "taxel_mapping_scan_extra_descent_spin",
            "taxel_mapping_reference_speed_spin",
            "taxel_mapping_reference_descent_spin",
            "taxel_mapping_start_margin_spin",
            "taxel_mapping_probe_speed_spin",
            "taxel_mapping_confirmation_speed_spin",
            "taxel_mapping_descent_spin",
            "taxel_mapping_force_guard_spin",
            "taxel_mapping_coarse_radius_spin",
            "taxel_mapping_coarse_step_spin",
            "taxel_mapping_local_radius_spin",
            "taxel_mapping_local_step_spin",
            "taxel_mapping_line_speed_spin",
            "taxel_mapping_xy_limit_spin",
            "taxel_mapping_load_map_button",
            "taxel_mapping_resume_button",
            "taxel_mapping_relocate_button",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not running)
        start_button = getattr(self, "sidebar_btn_start", None)
        if start_button is not None:
            start_button.setEnabled(not running)
        if not running:
            self._refresh_taxel_mapping_controls()
        else:
            active = getattr(self, "_sidebar_active_task", None)
            grid = getattr(self, "taxel_mapping_map", None)
            if (
                getattr(active, "id", None) == "taxel_position_mapping"
                and grid is not None
                and hasattr(grid, "set_grid_shape")
            ):
                grid.set_grid_shape(
                    int(active.sensor_rows), int(active.sensor_columns)
                )
            self._refresh_taxel_navigation_controls()

    def _remember_taxel_mapping_anchor(self):
        task = self._taxel_mapping_task()
        if task is None or self._current_sidebar_control_id() != task.id:
            return
        saved = task.capture_anchor_pose(self)
        if saved:
            self._taxel_mapping_anchor_updated_after_load = True
            self._refresh_taxel_navigation_controls()

    def _show_calibration_result(self, result):
        dialog = getattr(self, "_calibration_result_dialog", None)
        if dialog is not None and dialog.isVisible():
            dialog.close()
        self._calibration_result_dialog = CalibrationResultDialog(
            result, parent=self
        )
        self._calibration_result_dialog.show()

    def _show_full_taxel_calibration_result(self, result):
        dialog = getattr(
            self, "_full_taxel_calibration_result_dialog", None
        )
        if dialog is not None and dialog.isVisible():
            dialog.close()
        self._full_taxel_calibration_result_dialog = (
            FullTaxelCalibrationResultDialog(result, parent=self)
        )
        self._full_taxel_calibration_result_dialog.show()

    def _open_calibration_graph_analysis(self):
        """Select saved calibration data and open its offline analysis."""
        start_directory = resource_path("calibration_recordings")
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select calibration recording or graph",
            start_directory,
            "Calibration data or graph (*.csv *.json *.png)",
        )
        if not selected_path:
            return
        try:
            loaded = load_calibration_recording(selected_path)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(
                self,
                "Graph Analysis",
                f"The calibration recording could not be loaded.\n\n{exc}",
            )
            self._append_sidebar_control_message(
                f"Graph Analysis could not load {selected_path}: {exc}"
            )
            return

        dialog = getattr(self, "_calibration_graph_analysis_dialog", None)
        if dialog is not None:
            dialog.close()
        dialog = CalibrationGraphAnalysisDialog(loaded, parent=self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(
            lambda _object=None, dialog=dialog: self._graph_analysis_closed(
                dialog
            )
        )
        self._calibration_graph_analysis_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._append_sidebar_control_message(
            f"Graph Analysis loaded {len(loaded['datasets'])} trial(s)."
        )

    def _graph_analysis_closed(self, dialog):
        if self._calibration_graph_analysis_dialog is dialog:
            self._calibration_graph_analysis_dialog = None

    def _sigraph_scanning_task(self):
        return experiment_tasks.get_task("sigraph2026_scanning")

    def _refresh_sigraph_scanning_points(self, active_index=None):
        point_list = getattr(self, "sigraph_scanning_point_list", None)
        count_label = getattr(self, "sigraph_scanning_count_label", None)
        set_combo = getattr(self, "sigraph_scanning_set_combo", None)
        task = self._sigraph_scanning_task()
        if point_list is None or count_label is None or task is None:
            return
        if set_combo is not None:
            set_combo.blockSignals(True)
            set_combo.clear()
            set_combo.addItems(task.point_set_names())
            if set_combo.count():
                set_combo.setCurrentIndex(int(task.active_set_index))
            set_combo.blockSignals(False)
        previous_row = point_list.currentRow()
        point_list.clear()
        for index, point in enumerate(task.recorded_points):
            joints_deg = np.degrees(
                np.asarray(point.get("joints", [0.0] * 6), dtype=float)
            )
            position = point.get("tool_position")
            position_text = ""
            if position is not None and len(position) >= 3:
                position_mm = np.asarray(position[:3], dtype=float) * 1000.0
                position_text = (
                    f" | TCP=({position_mm[0]:.1f}, {position_mm[1]:.1f}, "
                    f"{position_mm[2]:.1f}) mm"
                )
            text = (
                f"P{index + 1:03d} | "
                f"J={[round(value, 1) for value in joints_deg]} deg"
                f"{position_text}"
            )
            item = QListWidgetItem(text)
            item.setToolTip(text)
            point_list.addItem(item)
        count_label.setText(
            f"Recorded points in {task.active_set_name}: "
            f"{len(task.recorded_points)}"
        )
        if active_index is not None and 0 <= int(active_index) < point_list.count():
            point_list.setCurrentRow(int(active_index))
        elif 0 <= previous_row < point_list.count():
            point_list.setCurrentRow(previous_row)
        elif point_list.count():
            point_list.setCurrentRow(0)
        go_button = getattr(self, "sigraph_scanning_go_button", None)
        if go_button is not None:
            go_button.setEnabled(
                point_list.count() > 0 and task._running_index is None
            )
        copy_button = getattr(self, "sigraph_scanning_copy_button", None)
        if copy_button is not None:
            copy_button.setEnabled(
                point_list.count() > 0 and task._running_index is None
            )
        reverse_button = getattr(self, "sigraph_scanning_reverse_button", None)
        if reverse_button is not None:
            reverse_button.setEnabled(
                point_list.count() > 0 and task._running_index is None
            )
        for button_name in (
            "sigraph_scanning_run_record_button",
            "sigraph_scanning_reverse_record_button",
        ):
            record_button = getattr(self, button_name, None)
            if record_button is not None:
                record_button.setEnabled(
                    point_list.count() > 0 and task._running_index is None
                )

    def _select_sigraph_scanning_point_set(self, index):
        task = self._sigraph_scanning_task()
        if task is None or not task.select_point_set(int(index)):
            return
        self._append_sidebar_control_message(
            f"Selected scanning point set: {task.active_set_name}."
        )
        self._refresh_sigraph_scanning_points()
        self._update_sidebar_task_action_labels()

    def _create_sigraph_scanning_point_set(self):
        task = self._sigraph_scanning_task()
        if task is None:
            return
        new_index = task.create_point_set()
        if new_index < 0:
            self._append_sidebar_control_message(
                "Stop the scanning replay before creating a new point set."
            )
            return
        self._refresh_sigraph_scanning_points()
        self._update_sidebar_task_action_labels()
        self._append_sidebar_control_message(
            f"Created {task.active_set_name}. Record points into this new set; "
            "the previous sets are still saved."
        )

    def _record_sigraph_scanning_point(self):
        task = self._sigraph_scanning_task()
        if task is not None:
            task.capture_current_point(self)

    def _go_to_selected_sigraph_scanning_point(self):
        task = self._sigraph_scanning_task()
        point_list = getattr(self, "sigraph_scanning_point_list", None)
        if task is None or point_list is None:
            return
        task.go_to_point(self, point_list.currentRow())

    def _start_sigraph_scanning_reverse(self):
        task = self._sigraph_scanning_task()
        point_list = getattr(self, "sigraph_scanning_point_list", None)
        if (
            task is None
            or point_list is None
            or self._current_sidebar_control_id() != task.id
        ):
            return
        task.request_reverse_run(point_list.currentRow())
        self._start_sidebar_control()

    def _sigraph_heatmap_plotter_for_recording(self):
        ui_ros = getattr(self, "ui_ros", None)
        if ui_ros is None:
            self._append_sidebar_control_message(
                "Start Ping Mode before recording the 3D heatmap."
            )
            return None
        sensor = getattr(ui_ros, "sensor_functions", None)
        if sensor is None or not bool(getattr(sensor, "is_connected", False)):
            self._append_sidebar_control_message(
                "Build and update the sensor before recording the 3D heatmap."
            )
            return None

        combo = getattr(ui_ros, "sensor_visualization_mode_combo", None)
        if combo is not None:
            heatmap_index = combo.findData("heatmap_3d")
            if heatmap_index >= 0:
                combo.setCurrentIndex(heatmap_index)
        if hasattr(sensor, "set_main_visualization_enabled"):
            sensor.set_main_visualization_enabled(True, render=False)
        if hasattr(sensor, "set_sensor_visualization_mode"):
            sensor.set_sensor_visualization_mode("heatmap_3d")
        if getattr(sensor, "heatmapPoly", None) is None:
            self._append_sidebar_control_message(
                "The 3D heatmap is not ready; build and update the sensor first."
            )
            return None

        plotter = getattr(ui_ros, "plotter_2", None)
        if plotter is None:
            self._append_sidebar_control_message(
                "The 3D sensor plotter is unavailable."
            )
            return None
        try:
            plotter.render()
        except Exception:
            pass
        return plotter

    def _sigraph_heatmap_record_elapsed(self):
        started_at = getattr(self, "_sigraph_heatmap_record_started_at", None)
        if started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - float(started_at))

    def _capture_sigraph_heatmap_frame(
        self, elapsed_s=None, stop_on_error=True
    ):
        recorder = getattr(self, "_sigraph_heatmap_recorder", None)
        if recorder is None or not recorder.is_recording:
            return False
        signal_recorder = getattr(
            self, "_sigraph_heatmap_signal_recorder", None
        )
        elapsed = (
            self._sigraph_heatmap_record_elapsed()
            if elapsed_s is None
            else max(0.0, float(elapsed_s))
        )
        frames_due = recorder.frames_due_for_elapsed(elapsed)
        if frames_due <= 0:
            return True

        sensor = getattr(getattr(self, "ui_ros", None), "sensor_functions", None)
        snapshot_getter = getattr(sensor, "get_heatmap_recording_snapshot", None)
        snapshot = snapshot_getter() if callable(snapshot_getter) else None
        target_frame_count = recorder.frame_count + frames_due
        if (
            signal_recorder is None
            or snapshot is None
            or not signal_recorder.capture_to_frame_count(
                snapshot, target_frame_count
            )
        ):
            error = (
                signal_recorder.last_error
                if signal_recorder is not None and signal_recorder.last_error
                else "Could not read a tactile frame for the heatmap recording."
            )
            if stop_on_error:
                self._stop_sigraph_heatmap_recording(
                    announce=True, capture_final=False
                )
            self._append_sidebar_control_message(error)
            return False
        if not recorder.capture_frame(repeat_count=frames_due):
            signal_recorder.discard_last_frames(frames_due)
            error = recorder.last_error
            if stop_on_error:
                self._stop_sigraph_heatmap_recording(
                    announce=True, capture_final=False
                )
            self._append_sidebar_control_message(error)
            return False
        return True

    def _stop_sigraph_heatmap_recording(
        self,
        announce=True,
        discard=False,
        capture_final=True,
        elapsed_s=None,
    ):
        timer = getattr(self, "_sigraph_heatmap_record_timer", None)
        if timer is not None:
            timer.stop()
        recorder = getattr(self, "_sigraph_heatmap_recorder", None)
        signal_recorder = getattr(
            self, "_sigraph_heatmap_signal_recorder", None
        )
        if recorder is None:
            self._sigraph_heatmap_record_started_at = None
            return None

        duration_s = (
            self._sigraph_heatmap_record_elapsed()
            if elapsed_s is None
            else max(0.0, float(elapsed_s))
        )
        if bool(capture_final) and not discard and signal_recorder is not None:
            if not self._capture_sigraph_heatmap_frame(
                elapsed_s=duration_s, stop_on_error=False
            ):
                self._append_sidebar_control_message(
                    "The final synchronized heatmap frame could not be captured."
                )

        self._sigraph_heatmap_recorder = None
        self._sigraph_heatmap_signal_recorder = None
        self._sigraph_heatmap_record_started_at = None

        output_path = recorder.stop(capture_final=False)
        if signal_recorder is not None:
            while signal_recorder.frame_count > recorder.frame_count:
                signal_recorder.discard_last_frame()
            signal_recorder.metadata["run_duration_s"] = float(duration_s)
            signal_recorder.metadata["encoded_duration_s"] = float(
                recorder.frame_count / recorder.fps
            )
            signal_path = signal_recorder.stop(discard=discard)
        else:
            signal_path = None
        if discard and output_path is not None:
            try:
                os.remove(output_path)
            except OSError:
                pass
            return None
        if announce and output_path is not None:
            self._append_sidebar_control_message(
                f"3D heatmap recording saved: {output_path} "
                f"({recorder.frame_count} frames, "
                f"run {duration_s:.3f} s, "
                f"video {recorder.frame_count / recorder.fps:.3f} s)"
            )
            if signal_path is not None:
                self._append_sidebar_control_message(
                    f"Tactile signal recording saved: {signal_path} "
                    f"({signal_recorder.frame_count} frames)"
                )
            elif signal_recorder is not None and signal_recorder.last_error:
                self._append_sidebar_control_message(signal_recorder.last_error)
        return output_path

    def _start_sigraph_scanning_run_and_record(self):
        self._start_sigraph_scanning_recorded_run(reverse=False)

    def _start_sigraph_scanning_reverse_and_record(self):
        self._start_sigraph_scanning_recorded_run(reverse=True)

    def _start_sigraph_scanning_recorded_run(self, reverse=False):
        task = self._sigraph_scanning_task()
        point_list = getattr(self, "sigraph_scanning_point_list", None)
        if (
            task is None
            or point_list is None
            or self._current_sidebar_control_id() != task.id
        ):
            return
        if self._sidebar_active_task is not None:
            self._stop_sidebar_control(
                show_result=False, message="Control restarted"
            )
        selected_index = point_list.currentRow()
        if not 0 <= selected_index < len(task.recorded_points):
            self._append_sidebar_control_message(
                "Select a recorded scanning point before running and recording."
            )
            return
        plotter = self._sigraph_heatmap_plotter_for_recording()
        if plotter is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_directory = resource_path("sigraph2026_scanning_recordings")
        direction = "reverse" if reverse else "forward"
        recording_stem = f"sigraph2026_{direction}_heatmap_{timestamp}"
        output_path = os.path.join(output_directory, f"{recording_stem}.mp4")
        signal_output_path = os.path.join(
            output_directory, f"{recording_stem}_sensor.npz"
        )
        sensor = getattr(getattr(self, "ui_ros", None), "sensor_functions", None)
        snapshot_getter = getattr(sensor, "get_heatmap_recording_snapshot", None)
        metadata_getter = getattr(sensor, "get_heatmap_recording_metadata", None)
        snapshot = snapshot_getter() if callable(snapshot_getter) else None
        if snapshot is None:
            self._append_sidebar_control_message(
                "The tactile signal is unavailable; update the sensor before recording."
            )
            return
        recorder = PlotterVideoRecorder(fps=SIGRAPH_HEATMAP_RECORD_FPS)
        if not recorder.start(plotter, output_path):
            self._append_sidebar_control_message(recorder.last_error)
            return
        signal_recorder = HeatmapSignalRecorder(
            fps=SIGRAPH_HEATMAP_RECORD_FPS
        )
        metadata = metadata_getter() if callable(metadata_getter) else {}
        metadata = dict(metadata or {})
        metadata.update({
            "experiment": "Sigraph2026 Scanning",
            "created_at": datetime.now().astimezone().isoformat(),
            "point_set": task.active_set_name,
            "run_direction": direction,
            "start_point_index": selected_index,
            "start_point_number": selected_index + 1,
            "timeline": "monotonic_elapsed_frame_pacing",
        })
        if not signal_recorder.start(
            signal_output_path,
            snapshot,
            metadata=metadata,
            video_path=output_path,
        ):
            recorder.stop(capture_final=False)
            try:
                os.remove(output_path)
            except OSError:
                pass
            self._append_sidebar_control_message(signal_recorder.last_error)
            return
        self._sigraph_heatmap_recorder = recorder
        self._sigraph_heatmap_signal_recorder = signal_recorder
        if self._sigraph_heatmap_record_timer is None:
            self._sigraph_heatmap_record_timer = QTimer(self)
            self._sigraph_heatmap_record_timer.setTimerType(Qt.PreciseTimer)
            self._sigraph_heatmap_record_timer.timeout.connect(
                self._capture_sigraph_heatmap_frame
            )
        self._append_sidebar_control_message(
            f"3D heatmap and tactile {direction} recording started from "
            f"point {selected_index + 1}: {output_path}"
        )

        task.request_run(selected_index, reverse=bool(reverse))
        self._sigraph_heatmap_record_started_at = time.perf_counter()
        signal_recorder.align_timeline_start(
            self._sigraph_heatmap_record_started_at
        )
        self._sigraph_heatmap_record_timer.start(
            max(1, round(1000.0 / SIGRAPH_HEATMAP_RECORD_FPS))
        )
        self._start_sidebar_control()
        if self._sidebar_active_task is not task:
            self._stop_sigraph_heatmap_recording(
                announce=False, discard=True
            )
            self._append_sidebar_control_message(
                "Scanning did not start; 3D heatmap recording was discarded."
            )

    def _choose_sigraph_heatmap_recording_for_playback(self):
        start_directory = resource_path("sigraph2026_scanning_recordings")
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Replay 3D Heatmap Recording",
            start_directory,
            "Tactile Heatmap Recordings (*_sensor.npz *.npz)",
        )
        if file_path:
            self._start_sigraph_heatmap_playback(file_path)

    def _start_sigraph_heatmap_playback(self, file_path):
        try:
            playback = HeatmapSignalPlayback.load(file_path)
        except Exception as exc:
            self._append_sidebar_control_message(
                f"Could not load tactile recording: {exc}"
            )
            return False

        ui_ros = getattr(self, "ui_ros", None)
        sensor = getattr(ui_ros, "sensor_functions", None) if ui_ros else None
        if sensor is None:
            self._append_sidebar_control_message(
                "Start Ping Mode and build the recorded sensor before replaying."
            )
            return False
        current_shape = (
            int(getattr(sensor, "n_row", 0) or 0),
            int(getattr(sensor, "n_col", 0) or 0),
        )
        if current_shape != playback.frame_shape:
            self._append_sidebar_control_message(
                "Build the matching sensor before replay: recording is "
                f"{playback.frame_shape[0]} x {playback.frame_shape[1]}, "
                f"current scene is {current_shape[0]} x {current_shape[1]}."
            )
            return False
        recorded_model = str(playback.metadata.get("model", "") or "")
        current_model = str(getattr(sensor, "current_model_name", "") or "")
        if recorded_model and recorded_model != current_model:
            self._append_sidebar_control_message(
                f"Build the recorded sensor model '{recorded_model}' before replaying."
            )
            return False

        if self._sidebar_active_task is not None:
            self._stop_sidebar_control(
                show_result=False, message="Control stopped for heatmap replay"
            )
        self._stop_sigraph_heatmap_playback(announce=False)

        self._sigraph_heatmap_playback_restore_state = {
            "mode": str(
                getattr(sensor, "sensor_visualization_mode", "point_grid")
            ),
            "heatmap_settings": copy.deepcopy(
                sensor.get_heatmap_settings()
                if hasattr(sensor, "get_heatmap_settings")
                else None
            ),
            "geometry": copy.deepcopy(
                getattr(sensor, "current_sensor_geometry_config", None)
            ),
            "zero_mask": np.array(
                sensor.get_cell_zero_mask(), dtype=bool, copy=True
            ),
        }
        if hasattr(sensor, "set_main_visualization_enabled"):
            sensor.set_main_visualization_enabled(True, render=False)
        if hasattr(sensor, "set_heatmap_playback_active"):
            sensor.set_heatmap_playback_active(True)
        if hasattr(sensor, "set_sensor_visualization_mode"):
            sensor.set_sensor_visualization_mode("heatmap_3d")

        recorded_geometry = playback.metadata.get("geometry")
        if isinstance(recorded_geometry, dict) and hasattr(
            sensor, "set_sensor_geometry_config"
        ):
            sensor.set_sensor_geometry_config(
                recorded_geometry, save_current_sensor=False, render=False
            )
        recorded_settings = playback.metadata.get("heatmap_settings")
        if isinstance(recorded_settings, dict) and hasattr(
            sensor, "set_heatmap_settings"
        ):
            recorded_settings = dict(recorded_settings)
            current_settings = self._sigraph_heatmap_playback_restore_state.get(
                "heatmap_settings"
            )
            if isinstance(current_settings, dict):
                recorded_settings["palette_3d"] = current_settings.get(
                    "palette_3d", recorded_settings.get("palette_3d")
                )
            sensor.set_heatmap_settings(
                recorded_settings, save_current_sensor=False
            )
        recorded_mask = np.asarray(
            playback.metadata.get("zero_mask", []), dtype=bool
        )
        if recorded_mask.shape == playback.frame_shape and hasattr(
            sensor, "set_cell_zero_mask"
        ):
            sensor.set_cell_zero_mask(recorded_mask)

        combo = getattr(ui_ros, "sensor_visualization_mode_combo", None)
        if combo is not None:
            heatmap_index = combo.findData("heatmap_3d")
            if heatmap_index >= 0:
                combo.setCurrentIndex(heatmap_index)

        self._sigraph_heatmap_playback = playback
        self._sigraph_heatmap_playback_sensor = sensor
        self._sigraph_heatmap_playback_started_at = time.perf_counter()
        self._sigraph_heatmap_playback_last_index = -1
        if self._sigraph_heatmap_playback_timer is None:
            self._sigraph_heatmap_playback_timer = QTimer(self)
            self._sigraph_heatmap_playback_timer.setTimerType(Qt.PreciseTimer)
            self._sigraph_heatmap_playback_timer.timeout.connect(
                self._advance_sigraph_heatmap_playback
            )
        self._sigraph_heatmap_playback_timer.start(
            max(1, round(1000.0 / playback.fps))
        )
        self._set_sigraph_scanning_controls_running(True)
        self._advance_sigraph_heatmap_playback()
        self._append_sidebar_control_message(
            f"Replaying tactile heatmap: {playback.path} "
            f"({playback.frame_count} frames, {playback.fps:.1f} FPS). "
            "The robot will not move."
        )
        return True

    def _advance_sigraph_heatmap_playback(self):
        playback = getattr(self, "_sigraph_heatmap_playback", None)
        sensor = getattr(self, "_sigraph_heatmap_playback_sensor", None)
        started_at = getattr(self, "_sigraph_heatmap_playback_started_at", None)
        if playback is None or sensor is None or started_at is None:
            return
        elapsed = max(0.0, time.perf_counter() - float(started_at))
        frame_index = playback.frame_index_at(elapsed)
        if frame_index != self._sigraph_heatmap_playback_last_index:
            rendered = sensor.render_recorded_heatmap_frame(
                playback.heatmap_frames[frame_index]
            )
            if not rendered:
                self._append_sidebar_control_message(
                    "Heatmap replay stopped because the recorded frame could not be rendered."
                )
                self._stop_sigraph_heatmap_playback(announce=False)
                return
            self._sigraph_heatmap_playback_last_index = frame_index
        if elapsed >= playback.duration_s + (1.0 / playback.fps):
            self._stop_sigraph_heatmap_playback(announce=True, completed=True)

    def _stop_sigraph_heatmap_playback(self, announce=True, completed=False):
        timer = getattr(self, "_sigraph_heatmap_playback_timer", None)
        if timer is not None:
            timer.stop()
        playback = getattr(self, "_sigraph_heatmap_playback", None)
        sensor = getattr(self, "_sigraph_heatmap_playback_sensor", None)
        restore = getattr(self, "_sigraph_heatmap_playback_restore_state", None)
        self._sigraph_heatmap_playback = None
        self._sigraph_heatmap_playback_sensor = None
        self._sigraph_heatmap_playback_started_at = None
        self._sigraph_heatmap_playback_last_index = -1
        self._sigraph_heatmap_playback_restore_state = None
        if playback is None:
            return False

        if sensor is not None and isinstance(restore, dict):
            geometry = restore.get("geometry")
            if isinstance(geometry, dict) and hasattr(
                sensor, "set_sensor_geometry_config"
            ):
                sensor.set_sensor_geometry_config(
                    geometry, save_current_sensor=False, render=False
                )
            settings = restore.get("heatmap_settings")
            if isinstance(settings, dict) and hasattr(
                sensor, "set_heatmap_settings"
            ):
                sensor.set_heatmap_settings(settings, save_current_sensor=False)
            zero_mask = restore.get("zero_mask")
            if zero_mask is not None and hasattr(sensor, "set_cell_zero_mask"):
                sensor.set_cell_zero_mask(zero_mask)
            previous_mode = str(restore.get("mode", "point_grid"))
            if hasattr(sensor, "set_sensor_visualization_mode"):
                sensor.set_sensor_visualization_mode(previous_mode)
            combo = getattr(
                getattr(self, "ui_ros", None),
                "sensor_visualization_mode_combo",
                None,
            )
            if combo is not None:
                previous_index = combo.findData(previous_mode)
                if previous_index >= 0:
                    combo.setCurrentIndex(previous_index)
        if sensor is not None and hasattr(sensor, "set_heatmap_playback_active"):
            sensor.set_heatmap_playback_active(False)

        self._set_sigraph_scanning_controls_running(False)
        self._refresh_sigraph_scanning_points()
        if announce:
            status = "complete" if completed else "stopped"
            self._append_sidebar_control_message(
                f"Tactile heatmap replay {status}: {playback.path}"
            )
        return True

    def _copy_sigraph_scanning_point(self):
        task = self._sigraph_scanning_task()
        point_list = getattr(self, "sigraph_scanning_point_list", None)
        if task is None or point_list is None:
            return
        source_row = point_list.currentRow()
        copied_row = task.copy_point(source_row)
        if copied_row < 0:
            self._append_sidebar_control_message(
                "Select a scanning point to copy."
            )
            return
        self._append_sidebar_control_message(
            f"Copied {task.active_set_name} point {source_row + 1} "
            f"to new point {copied_row + 1}."
        )
        self._refresh_sigraph_scanning_points(active_index=copied_row)

    def _remove_sigraph_scanning_point(self):
        task = self._sigraph_scanning_task()
        point_list = getattr(self, "sigraph_scanning_point_list", None)
        if task is None or point_list is None:
            return
        row = point_list.currentRow()
        if task.remove_point(row):
            self._append_sidebar_control_message(
                f"Removed {task.active_set_name} point {row + 1}."
            )
            self._refresh_sigraph_scanning_points()

    def _clear_sigraph_scanning_points(self):
        task = self._sigraph_scanning_task()
        if task is None:
            return
        point_count = len(task.recorded_points)
        if point_count == 0:
            self._append_sidebar_control_message(
                f"{task.active_set_name} is already empty."
            )
            return
        reply = QMessageBox.warning(
            self,
            f"Clear {task.active_set_name}?",
            f"This will permanently remove all {point_count} recorded points "
            f"from {task.active_set_name}.\n\nDo you want to continue?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            self._append_sidebar_control_message(
                f"Clear cancelled; {task.active_set_name} was not changed."
            )
            return
        if task.clear_points():
            self._append_sidebar_control_message(
                f"Cleared all scanning points in {task.active_set_name}."
            )
            self._refresh_sigraph_scanning_points()

    def _move_sigraph_scanning_point(self, offset):
        task = self._sigraph_scanning_task()
        point_list = getattr(self, "sigraph_scanning_point_list", None)
        if task is None or point_list is None:
            return
        row = point_list.currentRow()
        new_row = task.move_point(row, int(offset))
        self._refresh_sigraph_scanning_points()
        if 0 <= new_row < point_list.count():
            point_list.setCurrentRow(new_row)

    def _set_sigraph_scanning_controls_running(self, running):
        running = bool(running)
        for name in (
            "sidebar_task_list",
            "sigraph_scanning_set_combo",
            "sigraph_scanning_new_set_button",
            "sigraph_scanning_point_list",
            "sigraph_scanning_record_button",
            "sigraph_scanning_copy_button",
            "sigraph_scanning_remove_button",
            "sigraph_scanning_clear_button",
            "sigraph_scanning_go_button",
            "sigraph_scanning_reverse_button",
            "sigraph_scanning_run_record_button",
            "sigraph_scanning_reverse_record_button",
            "sigraph_scanning_replay_button",
            "sigraph_scanning_up_button",
            "sigraph_scanning_down_button",
            "sigraph_scanning_velocity_spin",
            "sigraph_scanning_dwell_spin",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not running)
        start_button = getattr(self, "sidebar_btn_start", None)
        if start_button is not None:
            start_button.setEnabled(not running)

    def _update_sidebar_task_param_visibility(self):
        widgets = getattr(self, "_task_param_widgets", None)
        if not widgets:
            return
        current_id = self._current_sidebar_control_id()
        grouped = {
            "calibration": "calibration",
            "full_taxel_calibration": "calibration",
        }
        current_group = grouped.get(current_id, current_id)
        seen = set()
        for task_id, widget in widgets.items():
            marker = id(widget)
            if marker in seen:
                continue
            seen.add(marker)
            widget.setVisible(grouped.get(task_id, task_id) == current_group)

    def _show_sensor_capture_result(self, series):
        """Open the capture-result dialog using a series passed in by a task."""
        if not series:
            self._append_sidebar_control_message("Sensor capture finished, but no samples were collected.")
            return

        if self._sensor_capture_result_dialog is not None and self._sensor_capture_result_dialog.isVisible():
            self._sensor_capture_result_dialog.close()

        self._sensor_capture_result_dialog = SensorCaptureResultDialog(
            list(series),
            parent=self,
        )
        self._sensor_capture_result_dialog.show()

    # --- Experiments-sidebar lifecycle dispatchers --------------------------------
    #
    # Concrete tasks live in ``phd.ui.experiment_tasks``. The methods below only
    # route events between the QTimer / sidebar buttons and the active task object.

    def _run_sidebar_control_action(self):
        task = self._sidebar_active_task
        if task is None:
            return
        try:
            task.on_tick(self)
        except Exception as exc:
            self._append_sidebar_control_message(
                f"Task '{getattr(task, 'label', task.id)}' tick error: {exc}"
            )
            self._stop_sidebar_control(show_result=False, message="Control stopped (tick error)")

    def _disconnect_force_meter_after_task(self, task) -> bool:
        """Release the force reader after an opted-in experiment ends."""
        if not bool(
            getattr(task, "disconnect_force_meter_on_stop", False)
        ):
            return False
        ui_ros = getattr(self, "ui_ros", None)
        disconnect = getattr(ui_ros, "_stop_force_meter", None)
        if not callable(disconnect):
            return False
        try:
            return bool(disconnect())
        except Exception as exc:
            self._append_sidebar_control_message(
                "Force transmitter could not be disconnected after the "
                f"task: {exc}"
            )
            return False

    def _start_sidebar_control(self):
        if self._sidebar_control_timer is None:
            self._sidebar_control_timer = QTimer(self)
            self._sidebar_control_timer.timeout.connect(self._run_sidebar_control_action)

        if self._sidebar_control_timer.isActive() or self._sidebar_active_task is not None:
            self._stop_sidebar_control(show_result=False, message="Control restarted")

        action_id = self._current_sidebar_control_id()
        task = experiment_tasks.get_task(action_id)
        if task is None:
            self._append_sidebar_control_message(
                f"No experiment task is registered for id '{action_id}'."
            )
            return
        if (
            getattr(task, "id", None) == "sigraph2026_scanning"
            and not bool(getattr(task, "_run_request_pending", False))
        ):
            point_list = getattr(self, "sigraph_scanning_point_list", None)
            selected_index = (
                point_list.currentRow() if point_list is not None else -1
            )
            task.request_run(selected_index, reverse=False)

        self._sidebar_active_task = task
        self._sidebar_active_control_id = action_id

        try:
            started = bool(task.on_start(self))
        except Exception as exc:
            self._append_sidebar_control_message(
                f"Task '{task.label}' failed to start: {exc}"
            )
            self._sidebar_active_task = None
            self._sidebar_active_control_id = None
            return

        if not started:
            # One-shot tasks (or tasks that aborted) never engage the timer.
            self._sidebar_active_task = None
            self._sidebar_active_control_id = None
            return

        interval_ms = task.tick_interval_ms
        if interval_ms is None:
            self._sidebar_active_task = None
            self._sidebar_active_control_id = None
            return

        self._sidebar_control_timer.start(int(interval_ms))
        self.info_process.setText(f"Control started: {task.label}")
        self._refresh_experiment_status()

    def _stop_sidebar_control(self, show_result=True, message="Control stopped"):
        recording_elapsed_s = self._sigraph_heatmap_record_elapsed()
        if self._sidebar_control_timer is not None:
            self._sidebar_control_timer.stop()
        task = self._sidebar_active_task
        self._sidebar_active_task = None
        self._sidebar_active_control_id = None
        self.info_process.setText(message)
        self._refresh_experiment_status()

        if task is not None:
            try:
                task.on_stop(self, show_result=show_result)
            except Exception as exc:
                self._append_sidebar_control_message(
                    f"Task '{getattr(task, 'label', task.id)}' stop error: {exc}"
                )
            self._disconnect_force_meter_after_task(task)
        self._stop_sigraph_heatmap_recording(
            announce=True, elapsed_s=recording_elapsed_s
        )
        self._stop_sigraph_heatmap_playback(announce=True)

        self._refresh_taxel_mapping_controls()

    # --- Keyboard tool-frame velocity (WASD + O/P + Q/E + R/T + Y/U) ------

    def _keyboard_velocity_should_capture(self) -> bool:
        """Return False when the user is typing in a text / numeric field."""
        w = QApplication.focusWidget()
        if w is None:
            return True
        if isinstance(w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            return False
        if isinstance(w, QComboBox) and w.isEditable():
            return False
        return True

    def _keyboard_vel_key_token(self, key: int) -> Optional[str]:
        mapping = {
            Qt.Key_W: "w",
            Qt.Key_S: "s",
            Qt.Key_A: "a",
            Qt.Key_D: "d",
            Qt.Key_Q: "q",
            Qt.Key_E: "e",
            Qt.Key_O: "o",
            Qt.Key_P: "p",
            Qt.Key_R: "r",
            Qt.Key_T: "t",
            Qt.Key_Y: "y",
            Qt.Key_U: "u",
        }
        return mapping.get(key)

    def _keyboard_vel_widget(self):
        if self.ui_ros is None:
            return None
        return getattr(self.ui_ros, "position_toolframe_widget", None)

    def _keyboard_vel_speeds(self):
        """Linear (m/s) and angular (rad/s) from the Robots tool-velocity panel."""
        w = self._keyboard_vel_widget()
        if w is not None:
            try:
                return float(w.linear_speed), float(w.angular_speed)
            except Exception:
                pass
        return 0.02, 0.0005

    def _keyboard_vel_stop_robot(self):
        """Send zero velocity (keeps velocity mode armed for the next press)."""
        w = self._keyboard_vel_widget()
        if w is None:
            return
        try:
            w.stop_all_velocity()
        except Exception:
            pass

    def _keyboard_vel_full_stop(self):
        """Zero velocity AND tear down velocity mode (Suspend → Stop)."""
        w = self._keyboard_vel_widget()
        if w is None:
            return
        try:
            w.stop_all_velocity()
        except Exception:
            pass
        try:
            w.stop_velocity_mode()
        except Exception:
            pass

    def _keyboard_vel_vector_for_token(self, token: str):
        """Return (v_lin, v_rot) contribution for a single key."""
        lin_spd, ang_spd = self._keyboard_vel_speeds()
        vx = vy = vz = rx = ry = rz = 0.0
        if token == "d":
            vx = lin_spd
        elif token == "a":
            vx = -lin_spd
        elif token == "w":
            vy = lin_spd
        elif token == "s":
            vy = -lin_spd
        elif token == "o":
            vz = lin_spd
        elif token == "p":
            vz = -lin_spd
        elif token == "r":
            rx = -ang_spd
        elif token == "t":
            rx = ang_spd
        elif token == "y":
            ry = -ang_spd
        elif token == "u":
            ry = ang_spd
        elif token == "q":
            rz = -ang_spd
        elif token == "e":
            rz = ang_spd
        return [vx, vy, vz], [rx, ry, rz]

    def _keyboard_vel_vector_for_tokens(self, tokens):
        """Sum velocity contributions from all active keys."""
        vx = vy = vz = rx = ry = rz = 0.0
        for token in tokens:
            v_lin, v_rot = self._keyboard_vel_vector_for_token(token)
            vx += v_lin[0]
            vy += v_lin[1]
            vz += v_lin[2]
            rx += v_rot[0]
            ry += v_rot[1]
            rz += v_rot[2]
        return [vx, vy, vz], [rx, ry, rz]

    def _keyboard_vel_token_label(self, token: str) -> str:
        labels = {
            "w": "+Y", "s": "−Y",
            "a": "−X", "d": "+X",
            "o": "+Z", "p": "−Z",
            "r": "−Rx", "t": "+Rx",
            "y": "−Ry", "u": "+Ry",
            "q": "−Rz", "e": "+Rz",
        }
        return labels.get(token, token)

    def _keyboard_vel_update_status_text(self):
        tokens = self._keyboard_vel_active_tokens
        if not tokens:
            self.info_process.setText("Keyboard tool velocity: stopped")
            return
        parts = [self._keyboard_vel_token_label(t) for t in sorted(tokens)]
        self.info_process.setText(f"Keyboard tool velocity: {' + '.join(parts)}")

    def _keyboard_vel_timer_tick(self):
        """Re-send the combined velocity from all active keys.

        TM's ContinueVLine needs periodic refreshes — if no command arrives for
        a while, the driver will treat it as a watchdog timeout. 50 ms is well
        below the driver's typical timeout (~200 ms) so the latch feels stable.
        """
        if not self._keyboard_tool_velocity_enabled:
            return
        if not self._keyboard_vel_active_tokens:
            return
        w = self._keyboard_vel_widget()
        if w is None or not hasattr(w, "_send_velocity"):
            return
        v_lin, v_rot = self._keyboard_vel_vector_for_tokens(self._keyboard_vel_active_tokens)
        try:
            w._send_velocity(v_lin, v_rot)
        except Exception:
            pass

    def _keyboard_vel_apply_active_tokens(self):
        if not self._keyboard_vel_active_tokens:
            self._keyboard_vel_stop_robot()
            self._keyboard_vel_update_status_text()
            return
        self._keyboard_vel_timer_tick()
        self._keyboard_vel_update_status_text()

    def _on_keyboard_tool_velocity_toggled(self, checked: bool):
        if checked:
            if not self.ping_mode_entered or self.ui_ros is None:
                QMessageBox.information(
                    self,
                    "Keyboard Tool Velocity",
                    "Please wait for Ping Mode to finish loading, then try again.",
                )
                self.action_keyboard_tool_velocity.blockSignals(True)
                self.action_keyboard_tool_velocity.setChecked(False)
                self.action_keyboard_tool_velocity.blockSignals(False)
                return
            if self._keyboard_vel_widget() is None:
                QMessageBox.warning(
                    self,
                    "Keyboard Tool Velocity",
                    "Tool velocity panel is not available on the embedded UI.",
                )
                self.action_keyboard_tool_velocity.blockSignals(True)
                self.action_keyboard_tool_velocity.setChecked(False)
                self.action_keyboard_tool_velocity.blockSignals(False)
                return
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self)
            self._keyboard_tool_velocity_enabled = True
            self._keyboard_vel_active_tokens = set()
            self._keyboard_vel_timer.start()
            self.info_process.setText(
                "Keyboard tool velocity: ON (W/S=±Y, A/D=±X, O/P=±Z, R/T=±Rx, Y/U=±Ry, Q/E=±Rz, Space=stop, keys combine)"
            )
        else:
            self._disable_keyboard_tool_velocity_internal()

    def _disable_keyboard_tool_velocity_internal(self):
        self._keyboard_tool_velocity_enabled = False
        self._keyboard_vel_active_tokens = set()
        self._keyboard_vel_timer.stop()
        app = QApplication.instance()
        if app is not None:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass
        self._keyboard_vel_full_stop()
        if hasattr(self, "action_keyboard_tool_velocity"):
            self.action_keyboard_tool_velocity.blockSignals(True)
            self.action_keyboard_tool_velocity.setChecked(False)
            self.action_keyboard_tool_velocity.blockSignals(False)
        self.info_process.setText("Keyboard tool velocity: OFF")

    def eventFilter(self, watched, event):
        if self._keyboard_tool_velocity_enabled:
            et = event.type()
            if et == QEvent.ShortcutOverride:
                # Block other QShortcuts (e.g. Space → AI direct finger motion
                # toggle in ui_ping_ai_controls) from swallowing our control
                # keys, so the KeyPress can reach this filter normally.
                ev = event
                if self._keyboard_velocity_should_capture():
                    key = ev.key()
                    if key == Qt.Key_Space or self._keyboard_vel_key_token(key) is not None:
                        ev.accept()
                        return True
            elif et == QEvent.KeyPress:
                ev = event
                if self._keyboard_velocity_should_capture():
                    key = ev.key()
                    if key == Qt.Key_Space:
                        if not ev.isAutoRepeat():
                            self._keyboard_vel_active_tokens.clear()
                            self._keyboard_vel_stop_robot()
                            self.info_process.setText(
                                "Keyboard tool velocity: stopped (Space)"
                            )
                        return True
                    tok = self._keyboard_vel_key_token(key)
                    if tok is not None:
                        # Auto-repeat events must still be consumed so VTK does
                        # not interpret W/P/R as wireframe, picking, or camera
                        # reset shortcuts. Motion is already sustained by the
                        # 50 ms velocity refresh timer.
                        if not ev.isAutoRepeat():
                            self._keyboard_vel_active_tokens.add(tok)
                            self._keyboard_vel_apply_active_tokens()
                        return True
            elif et == QEvent.KeyRelease:
                ev = event
                if self._keyboard_velocity_should_capture():
                    tok = self._keyboard_vel_key_token(ev.key())
                    if tok is not None:
                        if not ev.isAutoRepeat():
                            self._keyboard_vel_active_tokens.discard(tok)
                            self._keyboard_vel_apply_active_tokens()
                        return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        """Handles global key press events."""
        if not self.ping_mode_entered and event.key() == Qt.Key_1:
            self.run_ping_mode()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Executes when the main window is closed to ensure clean shutdown."""
        if self._keyboard_tool_velocity_enabled or (
            hasattr(self, "action_keyboard_tool_velocity")
            and self.action_keyboard_tool_velocity.isChecked()
        ):
            self._disable_keyboard_tool_velocity_internal()
        if self._fps_timer is not None:
            self._fps_timer.stop()
        iren = getattr(self, "_fps_render_interactor", None)
        if iren is not None:
            try:
                iren.render_signal.disconnect(self._on_frame_rendered)
            except Exception:
                pass
            self._fps_render_interactor = None
        self._stop_sidebar_control(
            show_result=False,
            message="Control stopped (window closing)",
        )

        # Explicitly close any child windows to avoid orphaned processes
        experiments_window = getattr(self, "experiments_window", None)
        if experiments_window is not None:
            experiments_window.remember_geometry()
            experiments_window.hide()
        camera_window = getattr(self, "camera_preview_window", None)
        if camera_window is not None:
            try:
                camera_window.close()
            except RuntimeError:
                self.camera_preview_window = None
        if self.sensor_window:
            self.sensor_window.close()

        if self.ui_ros is not None and hasattr(self.ui_ros, "shutdown"):
            self.ui_ros.shutdown()

        event.accept()

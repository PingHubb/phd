import copy
import os
import time
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, QSize, QEvent
from PyQt5.QtWidgets import (
    QWidget, QAction, QSplitter, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QToolBar, QStatusBar, QComboBox,
    QFrame, QMessageBox, QStyle, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QGridLayout, QToolButton, QSizePolicy, QMenu, QDoubleSpinBox, QFormLayout, QSpinBox,
    QCheckBox, QApplication, QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox,
    QFileDialog,
)
from PyQt5.QtGui import QIcon, QColor, QPainter, QPen, QPainterPath, QPixmap, QFont
import numpy as np
from pyvistaqt import QtInteractor, MainWindow
from phd.dependence.paths import icon_path, resource_path, stylesheet_path
from phd.ui import theme
from phd.ui.ui_ping import UI
from phd.ui import experiment_tasks
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
                return f.read()
        except FileNotFoundError:
            print(f"Warning: Stylesheet not found at '{self.STYLE_FILE}'.")
            # Return an empty string so the app can still run
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
        self.resize(980, 620)
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {theme.WINDOW_BG};
                color: {theme.TEXT_PRIMARY};
            }}
            QLabel#resultTitle {{
                font-size: 20px;
                font-weight: 600;
                color: #FFFFFF;
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


class MyMainWindow(MainWindow):
    """
    The main application window for PingLab.

    This class is responsible for setting up the main UI components,
    including the menu bar, toolbar, 3D view area, sidebar, and status bar.
    It manages the overall layout and handles core application events and state.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.resources = Resources()
        self._fps_timer = None
        self._sidebar_control_timer = None
        self._startup_requested = False
        self._sidebar_active_control_id = None
        self._sensor_capture_result_dialog = None
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
        self._setup_main_layout()
        self._connect_signals_to_slots()

        # Apply the global stylesheet
        self.setStyleSheet(self.resources.get_stylesheet())

        # Delay heavy UI construction until the event loop starts so the window appears sooner.
        QTimer.singleShot(0, self.run_ping_mode)

    def _setup_actions(self):
        """Creates all QAction objects used in menus and toolbars."""
        self.action_ping_mode = QAction(self.resources.get_icon('logo0.png'), 'Ping', self)

        self.action_log = QAction(self.resources.get_icon('logo2.png'), 'Log', self)

        self.action_sensor_signal = QAction(self.resources.get_icon('logo3.png'), "Sensor Signal", self)
        self.action_ps5_controller_test = QAction(
            self._emoji_toolbar_icon("🎮"),
            'PS5 Controller Test',
            self,
        )
        self.action_ps5_controller_test.setToolTip('Open the PS5 controller input test window')
        self.action_sensor_controller_test = QAction(
            self._emoji_toolbar_icon("👆"),
            'Sensor Controller Test',
            self,
        )
        self.action_sensor_controller_test.setToolTip('Open the sensor-to-controller mapping test window')

        self.action_keyboard_tool_velocity = QAction(
            self._emoji_toolbar_icon("⌨"),
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

        self.action_direct_finger_motion_params = QAction(self.style().standardIcon(QStyle.SP_CommandLink), '🖐 DFM Parameters', self)
        self.action_direct_finger_motion_params.setToolTip('Open the Direct Finger Motion parameter editor')
        self.action_proximity_control_params = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            'Proximity Parameters',
            self,
        )
        self.action_proximity_control_params.setToolTip('Open the Proximity Control parameter editor')
        self.action_console_control_params = QAction(
            self.style().standardIcon(QStyle.SP_ComputerIcon),
            'Console Control Parameters',
            self,
        )
        self.action_console_control_params.setToolTip('Open the Console Control parameter editor')
        self.action_sensor_params = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogListView),
            'Sensor Parameters',
            self,
        )
        self.action_sensor_params.setToolTip('Open the Sensor parameter editor')
        self.action_parameter_settings = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            'Parameter Settings',
            self,
        )
        self.action_parameter_settings.setToolTip('Open parameter settings menu')
        self.action_update_sensor = QAction('Update Sensor', self)
        self.action_update_sensor.setToolTip('Update sensor values (same action used by AI/Sensor tabs before)')

        self.action_toggle_controls = QAction(self.resources.get_icon('logo5.png'), 'Show Experiments', self)
        self.action_toggle_controls.setToolTip('Show the experiments sidebar')

        self.action_exit = QAction(self.resources.get_icon('logo4.png'), 'Exit', self)
        self.action_exit.setShortcut('Ctrl+Q')

    def _emoji_toolbar_icon(self, emoji: str, size: int = 28) -> QIcon:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setFont(QFont("Noto Color Emoji", max(12, int(size * 0.7))))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, str(emoji))
        finally:
            painter.end()
        return QIcon(pixmap)

    def _setup_menu_and_toolbar(self):
        """Initializes the menu bar and the main toolbar using the predefined actions."""
        # --- Menu Bar ---
        menu = self.menuBar()
        mode_menu = menu.addMenu('Mode')
        mode_menu.addAction(self.action_ping_mode)

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
        function_menu.addSeparator()
        function_menu.addAction(self.action_exit)

        # --- Toolbar ---
        toolbar = QToolBar("Main")
        toolbar.setIconSize(QSize(24, 24))
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
        toolbar.addSeparator()
        toolbar.addAction(self.action_exit)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        self.toolbar_update_sensor_button = QToolButton()
        self.toolbar_update_sensor_button.setText("Update Sensor")
        self.toolbar_update_sensor_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.toolbar_update_sensor_button.setAutoRaise(True)
        self.toolbar_update_sensor_button.setStyleSheet(
            "QToolButton {"
            f" border: 1px solid {theme.ACCENT};"
            " border-radius: 6px;"
            " padding: 4px 10px;"
            " font-weight: 600;"
            f" color: {theme.TEXT_PRIMARY};"
            " background-color: rgba(61, 130, 240, 0.16);"
            "}"
            "QToolButton:hover {"
            " background-color: rgba(61, 130, 240, 0.30);"
            "}"
            "QToolButton:pressed {"
            " background-color: rgba(61, 130, 240, 0.45);"
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

        # --- Sidebar (initially hidden) ---
        self.sidebar = self._create_sidebar()

        self.h_splitter.addWidget(self.view_container)
        self.h_splitter.addWidget(self.sidebar)
        self.h_splitter.setCollapsible(1, True)
        self._set_sidebar_visible(False)

        main_layout.addWidget(self.h_splitter)

        # --- Status Bar ---
        self._setup_status_bar()

    def _create_sidebar(self) -> QFrame:
        """Creates the sidebar widget and all its contents."""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFrameShape(QFrame.StyledPanel)
        # Wider than the legacy 250 px so longer task descriptions / labels
        # (e.g. "Save per-point sensor CSV", "Per-point dwell:", J=[...]° log lines)
        # are not clipped by the splitter handle.
        sidebar.setMinimumWidth(330)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Available tasks first — this is what the user picks on every run, so keep
        # it pinned to the top of the sidebar.
        tasks_header = QLabel("Available Tasks")
        font = tasks_header.font()
        font.setBold(True)
        tasks_header.setFont(font)
        layout.addWidget(tasks_header)

        self.sidebar_task_list = QListWidget()
        self.sidebar_task_list.setSelectionMode(QListWidget.SingleSelection)
        for label_text, task_id, description in experiment_tasks.task_definitions():
            item = QListWidgetItem(label_text)
            item.setData(Qt.UserRole, task_id)
            item.setData(Qt.UserRole + 1, description)
            self.sidebar_task_list.addItem(item)
        self.sidebar_task_list.setCurrentRow(0)
        layout.addWidget(self.sidebar_task_list, stretch=1)

        self.sidebar_task_info = QLabel()
        self.sidebar_task_info.setWordWrap(True)
        self.sidebar_task_info.setStyleSheet(
            f"color: {theme.TEXT_MUTED}; background-color: {theme.SURFACE_RAISED}; "
            f"border: 1px solid {theme.BORDER_SUBTLE}; padding: 8px; border-radius: 6px;"
        )
        layout.addWidget(self.sidebar_task_info)
        self._update_sidebar_task_info()

        # ─── Task-specific parameter controls ────────────────────────────────
        # Each task may expose a custom widget under this container; only the
        # widget that matches the currently-selected task is shown. New tasks
        # can register their own group by extending ``_task_param_widgets``.
        self._build_sidebar_task_param_widgets(layout)
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
        layout.addWidget(btn_container)
        self._update_sidebar_task_action_labels()

        return sidebar

    def _setup_status_bar(self):
        """Initializes the status bar with informational labels."""
        status = QStatusBar()
        self.info_process = QLabel("Ready")
        self.info_fps = QLabel("FPS: 0")
        status.addWidget(self.info_process)
        status.addPermanentWidget(self.info_fps)
        self.setStatusBar(status)

    def _connect_signals_to_slots(self):
        """Connects all QAction and widget signals to their corresponding methods (slots)."""
        self.action_exit.triggered.connect(self.close)
        self.action_ping_mode.triggered.connect(self.run_ping_mode)
        self.action_log.triggered.connect(self._toggle_log_view)
        self.action_toggle_controls.triggered.connect(self.toggle_sidebar)
        self.action_sensor_signal.triggered.connect(self.open_sensor_signal_window)
        self.action_ps5_controller_test.triggered.connect(self.open_ps5_controller_test_window)
        self.action_sensor_controller_test.triggered.connect(self.open_sensor_controller_test_window)
        self.action_keyboard_tool_velocity.toggled.connect(self._on_keyboard_tool_velocity_toggled)
        self.action_direct_finger_motion_params.triggered.connect(self.open_direct_finger_motion_params_window)
        self.action_proximity_control_params.triggered.connect(self.open_proximity_control_params_window)
        self.action_console_control_params.triggered.connect(self.open_console_control_params_window)
        self.action_sensor_params.triggered.connect(self.open_sensor_params_window)
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
        self.sidebar_btn_stop.clicked.connect(self._stop_sidebar_control)
        self.sidebar_task_list.currentItemChanged.connect(self._on_sidebar_task_changed)

    def _trigger_global_sensor_update(self):
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before updating the sensor."
        ):
            return
        try:
            if hasattr(self.ui_ros, "_on_sensor_update"):
                self.ui_ros._on_sensor_update()
            elif hasattr(self.ui_ros, "sensor_functions") and hasattr(self.ui_ros.sensor_functions, "updateCal"):
                self.ui_ros.sensor_functions.updateCal()
            self.info_process.setText("Sensor updated")
        except Exception as exc:
            self.info_process.setText("Sensor update failed")
            if self.ui_ros is not None and hasattr(self.ui_ros, "log_display"):
                self.ui_ros.log_display.append(f"Sensor update failed: {exc}")

    def _set_sidebar_visible(self, visible: bool):
        """Updates sidebar visibility and keeps the toggle action text in sync."""
        # Keep this in sync with ``_create_sidebar()``'s setMinimumWidth so
        # the splitter never starts narrower than the sidebar can render
        # without text clipping.
        sidebar_w = max(330, self.sidebar.minimumWidth())
        if visible:
            self.sidebar.show()
            self.h_splitter.setSizes([max(self.width() - sidebar_w, 0), sidebar_w])
            self.action_toggle_controls.setText('Hide Experiments')
            self.action_toggle_controls.setToolTip('Hide the experiments sidebar')
        else:
            self.sidebar.hide()
            self.h_splitter.setSizes([self.width(), 0])
            self.action_toggle_controls.setText('Show Experiments')
            self.action_toggle_controls.setToolTip('Show the experiments sidebar')

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

    def toggle_sidebar(self):
        """Shows or hides the control sidebar."""
        self._set_sidebar_visible(self.sidebar.isHidden())

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
        """Updates the FPS label in the status bar."""
        self.info_fps.setText(f"FPS: {self._frame_count}")
        self._frame_count = 0

    def _toggle_log_view(self):
        """Toggles the visibility of the log panel inside the main UI widget."""
        if self.ui_ros:
            self.ui_ros.toggle_plotter_visibility()
        else:
            QMessageBox.information(self, "Info", "Log is only available after Ping Mode has started.")

    def _append_sidebar_control_message(self, message: str):
        self.info_process.setText(message)
        if self.ui_ros is not None and hasattr(self.ui_ros, "log_display"):
            try:
                if not self.ui_ros.log_display.isVisible():
                    self.ui_ros.log_display.setVisible(True)
                    if hasattr(self.ui_ros, "adjust_splitter_sizes"):
                        self.ui_ros.adjust_splitter_sizes()
                self.ui_ros.log_display.append(message)
                return
            except Exception:
                pass
        print(message)

    def _current_sidebar_task_item(self):
        return getattr(self, "sidebar_task_list", None).currentItem() if hasattr(self, "sidebar_task_list") else None

    def _current_sidebar_control_id(self):
        item = self._current_sidebar_task_item()
        return item.data(Qt.UserRole) if item is not None else None

    def _current_sidebar_control_label(self):
        item = self._current_sidebar_task_item()
        return item.text() if item is not None else "No Task"

    def _update_sidebar_task_info(self):
        label = getattr(self, "sidebar_task_info", None)
        item = self._current_sidebar_task_item()
        if label is None:
            return
        if item is None:
            label.setText("Select a task from the list.")
            return
        description = item.data(Qt.UserRole + 1) or ""
        label.setText(f"Task: {item.text()}\n\n{description}")

    def _on_sidebar_task_changed(self, _current=None, _previous=None):
        self._update_sidebar_task_info()
        self._update_sidebar_task_param_visibility()
        self._update_sidebar_task_action_labels()

    def _update_sidebar_task_action_labels(self):
        start_button = getattr(self, "sidebar_btn_start", None)
        if start_button is None:
            return
        task = experiment_tasks.get_task(self._current_sidebar_control_id())
        start_button.setText(
            str(getattr(task, "start_button_label", "Start") or "Start")
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
        for task_id, widget in widgets.items():
            widget.setVisible(task_id == current_id)

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

    def _stop_sidebar_control(self, show_result=True, message="Control stopped"):
        recording_elapsed_s = self._sigraph_heatmap_record_elapsed()
        if self._sidebar_control_timer is not None:
            self._sidebar_control_timer.stop()
        task = self._sidebar_active_task
        self._sidebar_active_task = None
        self._sidebar_active_control_id = None
        self.info_process.setText(message)

        if task is not None:
            try:
                task.on_stop(self, show_result=show_result)
            except Exception as exc:
                self._append_sidebar_control_message(
                    f"Task '{getattr(task, 'label', task.id)}' stop error: {exc}"
                )
        self._stop_sigraph_heatmap_recording(
            announce=True, elapsed_s=recording_elapsed_s
        )
        self._stop_sigraph_heatmap_playback(announce=True)

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
                if not ev.isAutoRepeat() and self._keyboard_velocity_should_capture():
                    key = ev.key()
                    if key == Qt.Key_Space or self._keyboard_vel_key_token(key) is not None:
                        ev.accept()
                        return True
            elif et == QEvent.KeyPress:
                ev = event
                if not ev.isAutoRepeat() and self._keyboard_velocity_should_capture():
                    key = ev.key()
                    if key == Qt.Key_Space:
                        self._keyboard_vel_active_tokens.clear()
                        self._keyboard_vel_stop_robot()
                        self.info_process.setText("Keyboard tool velocity: stopped (Space)")
                        return True
                    tok = self._keyboard_vel_key_token(key)
                    if tok is not None:
                        self._keyboard_vel_active_tokens.add(tok)
                        self._keyboard_vel_apply_active_tokens()
                        return True
            elif et == QEvent.KeyRelease:
                ev = event
                if not ev.isAutoRepeat() and self._keyboard_velocity_should_capture():
                    tok = self._keyboard_vel_key_token(ev.key())
                    if tok is not None:
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
        if self.sensor_window:
            self.sensor_window.close()

        if self.ui_ros is not None and hasattr(self.ui_ros, "shutdown"):
            self.ui_ros.shutdown()

        event.accept()

import os
from typing import Optional

from PyQt5.QtCore import Qt, QTimer, QSize, QEvent
from PyQt5.QtWidgets import (
    QWidget, QAction, QSplitter, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QToolBar, QStatusBar, QComboBox,
    QFrame, QMessageBox, QStyle, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QGridLayout, QToolButton, QSizePolicy, QMenu, QDoubleSpinBox, QFormLayout, QSpinBox,
    QCheckBox, QApplication, QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox,
)
from PyQt5.QtGui import QIcon, QColor, QPainter, QPen, QPainterPath, QPixmap, QFont
import numpy as np
from pyvistaqt import QtInteractor, MainWindow
from phd.dependence.paths import icon_path, stylesheet_path
from phd.ui.ui_ping import UI
from phd.ui import experiment_tasks
from phd.dependence.sensor_signal_window import SensorSignalWindow
from phd.ui.sensor_zero_mask_window import SensorZeroMaskDialog


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
        painter.fillRect(self.rect(), QColor("#16181c"))

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

        grid_pen = QPen(QColor("#2b3138"))
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
            """
            QDialog {
                background-color: #15181d;
                color: #dce3ea;
            }
            QLabel#resultTitle {
                font-size: 20px;
                font-weight: 600;
                color: #f3f6f9;
            }
            QLabel#resultSubtitle {
                font-size: 12px;
                color: #9fb0bf;
            }
            QFrame#metricCard {
                background-color: #1d232b;
                border: 1px solid #2c3642;
                border-radius: 10px;
            }
            QLabel#metricName {
                font-size: 11px;
                color: #8fa1b3;
            }
            QLabel#metricValue {
                font-size: 18px;
                font-weight: 600;
                color: #f0f4f8;
            }
            QFrame#sectionCard {
                background-color: #1b2027;
                border: 1px solid #2a333e;
                border-radius: 12px;
            }
            QLabel#sectionTitle {
                font-size: 13px;
                font-weight: 600;
                color: #e6edf3;
            }
            QLabel#sectionBody {
                font-size: 12px;
                color: #b8c4cf;
            }
            QPushButton {
                min-width: 88px;
                min-height: 30px;
            }
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
            "W/S = ±Y, A/D = ±X, O/P = ±Z, Q/E = ±Rz, T/Y = ±Rx, G/H = ±Ry.\n"
            "Hold or press multiple keys to combine (e.g. W+D = +Y and +X). "
            "Release a key to remove that component. Space stops all."
        )

        self.action_direct_finger_motion_params = QAction(self.style().standardIcon(QStyle.SP_CommandLink), '🖐 DFM Parameters', self)
        self.action_direct_finger_motion_params.setToolTip('Open the Direct Finger Motion parameter editor')
        self.action_direct_finger_motion_v2_params = QAction(
            self.style().standardIcon(QStyle.SP_MediaSeekForward),
            'DFM V2 Parameters',
            self,
        )
        self.action_direct_finger_motion_v2_params.setToolTip('Open the Direct Finger Motion V2 parameter editor')
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
        self.action_parameter_settings = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            'Parameter Settings',
            self,
        )
        self.action_parameter_settings.setToolTip('Open parameter settings menu')
        self.action_update_sensor = QAction('Update Sensor', self)
        self.action_update_sensor.setToolTip('Update sensor values (same action used by AI/Sensor tabs before)')

        self.action_sensor_zero_mask = QAction(
            self.style().standardIcon(QStyle.SP_DialogResetButton),
            'Sensor Zero Mask',
            self,
        )
        self.action_sensor_zero_mask.setToolTip(
            'Pick sensor cells that should always read 0 (after Build Scene + Update Sensor)'
        )

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
        function_menu.addAction(self.action_sensor_zero_mask)
        function_menu.addAction(self.action_ps5_controller_test)
        function_menu.addAction(self.action_sensor_controller_test)
        function_menu.addAction(self.action_keyboard_tool_velocity)
        parameter_menu = function_menu.addMenu('Parameter Settings')
        parameter_menu.addAction(self.action_direct_finger_motion_params)
        parameter_menu.addAction(self.action_direct_finger_motion_v2_params)
        parameter_menu.addAction(self.action_proximity_control_params)
        parameter_menu.addAction(self.action_console_control_params)
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
        toolbar.addAction(self.action_sensor_zero_mask)
        toolbar.addAction(self.action_ps5_controller_test)
        toolbar.addAction(self.action_sensor_controller_test)
        toolbar.addAction(self.action_keyboard_tool_velocity)
        self.toolbar_parameter_button = QToolButton()
        self.toolbar_parameter_button.setDefaultAction(self.action_parameter_settings)
        self.toolbar_parameter_button.setPopupMode(QToolButton.InstantPopup)
        parameter_popup_menu = QMenu(toolbar)
        parameter_popup_menu.addAction(self.action_direct_finger_motion_params)
        parameter_popup_menu.addAction(self.action_direct_finger_motion_v2_params)
        parameter_popup_menu.addAction(self.action_proximity_control_params)
        parameter_popup_menu.addAction(self.action_console_control_params)
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
            " border: 1px solid #4fc3f7;"
            " border-radius: 6px;"
            " padding: 4px 10px;"
            " font-weight: 600;"
            " color: #e8f6ff;"
            " background-color: rgba(79, 195, 247, 0.14);"
            "}"
            "QToolButton:hover {"
            " background-color: rgba(79, 195, 247, 0.26);"
            "}"
            "QToolButton:pressed {"
            " background-color: rgba(79, 195, 247, 0.38);"
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
            "color: #d6dbe0; background-color: rgba(255,255,255,0.05); "
            "border: 1px solid rgba(255,255,255,0.08); padding: 8px; border-radius: 6px;"
        )
        layout.addWidget(self.sidebar_task_info)
        self._update_sidebar_task_info()

        # ─── Task-specific parameter controls ────────────────────────────────
        # Each task may expose a custom widget under this container; only the
        # widget that matches the currently-selected task is shown. New tasks
        # can register their own group by extending ``_task_param_widgets``.
        self._build_sidebar_task_param_widgets(layout)
        self._update_sidebar_task_param_visibility()

        # Container for Start/Stop buttons
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.addStretch()
        self.sidebar_btn_start = QPushButton("Start")
        self.sidebar_btn_start.setObjectName("btnStart")
        btn_layout.addWidget(self.sidebar_btn_start)
        self.sidebar_btn_stop = QPushButton("Stop")
        self.sidebar_btn_stop.setObjectName("btnStop")
        btn_layout.addWidget(self.sidebar_btn_stop)
        btn_layout.addStretch()
        layout.addWidget(btn_container)

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
        self.action_direct_finger_motion_v2_params.triggered.connect(self.open_direct_finger_motion_v2_params_window)
        self.action_proximity_control_params.triggered.connect(self.open_proximity_control_params_window)
        self.action_console_control_params.triggered.connect(self.open_console_control_params_window)
        self.action_update_sensor.triggered.connect(self._trigger_global_sensor_update)
        self.action_sensor_zero_mask.triggered.connect(self._open_sensor_zero_mask_window)
        self.sidebar_btn_start.clicked.connect(self._start_sidebar_control)
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

    def _open_sensor_zero_mask_window(self):
        """Open the dialog that lets the user zero-out specific sensor cells."""
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before configuring the zero mask."
        ):
            return

        sensor_functions = getattr(self.ui_ros, "sensor_functions", None)
        if sensor_functions is None:
            QMessageBox.warning(
                self,
                "Sensor Zero Mask",
                "Sensor functions are not available.",
            )
            return

        n_row = int(getattr(sensor_functions, "n_row", 0) or 0)
        n_col = int(getattr(sensor_functions, "n_col", 0) or 0)
        if n_row <= 0 or n_col <= 0:
            QMessageBox.information(
                self,
                "Sensor Zero Mask",
                "Build the sensor scene first.\n\n"
                "Steps: Sensor tab → 'Build Scene' → 'Update Sensor',\n"
                "then re-open this window to choose cells to force to 0.",
            )
            return

        if not getattr(sensor_functions, "is_connected", False):
            reply = QMessageBox.question(
                self,
                "Sensor Zero Mask",
                "The sensor has not been calibrated yet (Update Sensor was not run).\n"
                "You can still edit the mask, but it only takes effect once data starts flowing.\n\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            sensor_key = sensor_functions.get_zero_mask_key()
        except Exception:
            sensor_key = "(unknown)"

        dialog = SensorZeroMaskDialog(
            self, sensor_functions=sensor_functions, sensor_key=sensor_key
        )
        dialog.exec_()

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

    def open_direct_finger_motion_v2_params_window(self):
        if not self._require_ui_ros(
            "Please wait for Ping Mode to load before opening the DFM V2 parameter editor."
        ):
            return

        if hasattr(self.ui_ros, 'open_direct_finger_motion_v2_settings_dialog'):
            self.ui_ros.open_direct_finger_motion_v2_settings_dialog()
        else:
            QMessageBox.information(self, "Info", "DFM V2 parameter editor is not available in the current UI.")

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

    # --- Per-task parameter widgets in the sidebar ----------------------------
    def _build_sidebar_task_param_widgets(self, parent_layout):
        """Build the optional per-task parameter editors shown in the sidebar.

        Each entry maps a ``task_id`` to a small QWidget that mutates the task
        instance directly when the user adjusts a value. Only the widget for
        the currently-selected task is visible at a time.
        """
        self._task_param_widgets: dict = {}

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

    # --- Keyboard tool-frame velocity (WASD + O/P + Q/E + T/Y + G/H) ------

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
            Qt.Key_T: "t",
            Qt.Key_Y: "y",
            Qt.Key_G: "g",
            Qt.Key_H: "h",
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
        elif token == "t":
            rx = -ang_spd
        elif token == "y":
            rx = ang_spd
        elif token == "g":
            ry = -ang_spd
        elif token == "h":
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
            "t": "−Rx", "y": "+Rx",
            "g": "−Ry", "h": "+Ry",
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
                "Keyboard tool velocity: ON (W/S=±Y, A/D=±X, O/P=±Z, T/Y=±Rx, G/H=±Ry, Q/E=±Rz, Space=stop, keys combine)"
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
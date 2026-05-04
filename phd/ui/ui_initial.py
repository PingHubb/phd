import os
import time
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtWidgets import (
    QWidget, QAction, QSplitter, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTreeWidget, QTreeWidgetItem, QToolBar, QStatusBar, QComboBox,
    QFrame, QMessageBox, QStyle, QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QGridLayout
)
from PyQt5.QtGui import QIcon, QColor, QPainter, QPen, QPainterPath
import numpy as np
from pyvistaqt import QtInteractor, MainWindow
from phd.ui.ui_ping import UI
from phd.dependence.sensor_signal_window import SensorSignalWindow


class Resources:
    """
    A helper class to manage and provide access to application resources.
    This centralizes the hardcoded paths for icons and stylesheets,
    making it easy to update them in one place if they ever change.
    """
    ICON_DIR = '/home/ping2/ros2_ws/src/phd/phd/resource/icon/'
    STYLE_FILE = '/home/ping2/ros2_ws/src/phd/phd/resource/stylesheets/ui_style.qss'

    def get_icon(self, name: str) -> QIcon:
        """Loads a QIcon from the predefined icon directory."""
        icon_path = self.ICON_DIR + name
        if not os.path.exists(icon_path):
            return QIcon()
        return QIcon(icon_path)

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
        self._sensor_capture_started_at = None
        self._sensor_capture_series = []
        self._sensor_capture_duration_sec = 10.0
        self._sensor_capture_result_dialog = None

        # --- Initialize Window Properties ---
        self.setWindowTitle("PingLab")
        self.setMinimumSize(960, 540)
        self.resize(1650, 900)
        self.setWindowIcon(self.resources.get_icon('logo0.png'))

        # --- Initialize Application State ---
        self.ping_mode_entered = False
        self.sensor_window = None
        self.ui_ros = None  # This will hold the main UI widget from ui_ping.py

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
            self.style().standardIcon(QStyle.SP_DriveNetIcon),
            'PS5 Controller Test',
            self,
        )
        self.action_ps5_controller_test.setToolTip('Open the PS5 controller input test window')
        self.action_sensor_controller_test = QAction(
            self.style().standardIcon(QStyle.SP_DesktopIcon),
            'Sensor Controller Test',
            self,
        )
        self.action_sensor_controller_test.setToolTip('Open the sensor-to-controller mapping test window')

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

        self.action_toggle_controls = QAction(self.resources.get_icon('logo5.png'), 'Show Controls', self)
        self.action_toggle_controls.setToolTip('Show the control sidebar')

        self.action_exit = QAction(self.resources.get_icon('logo4.png'), 'Exit', self)
        self.action_exit.setShortcut('Ctrl+Q')

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
        toolbar.addAction(self.action_ps5_controller_test)
        toolbar.addAction(self.action_sensor_controller_test)
        toolbar.addAction(self.action_direct_finger_motion_params)
        toolbar.addAction(self.action_direct_finger_motion_v2_params)
        toolbar.addAction(self.action_proximity_control_params)
        toolbar.addAction(self.action_console_control_params)
        toolbar.addAction(self.action_toggle_controls)
        toolbar.addSeparator()
        toolbar.addAction(self.action_exit)
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
        sidebar.setMinimumWidth(250)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        root = QTreeWidgetItem(tree, ['Controls'])
        font = root.font(0)
        font.setBold(True)
        root.setFont(0, font)
        layout.addWidget(tree)

        layout.addWidget(QLabel("Available Tasks"))

        self.sidebar_task_list = QListWidget()
        self.sidebar_task_list.setSelectionMode(QListWidget.SingleSelection)
        task_definitions = [
            (
                "Hello World",
                "hello_world",
                "Every second, print 'hello world' to the log until you stop it.",
            ),
            (
                "Sensor Peak Change (10s)",
                "sensor_peak_change",
                "Capture 10 seconds of sensor data and visualize the strongest diffDataAve change over time.",
            ),
            (
                "Move Robot To Preset Joint",
                "robot_preset_joint",
                "Send the robot once to the preset joint-angle target used for quick positioning.",
            ),
        ]
        for label_text, task_id, description in task_definitions:
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
        self.action_direct_finger_motion_params.triggered.connect(self.open_direct_finger_motion_params_window)
        self.action_direct_finger_motion_v2_params.triggered.connect(self.open_direct_finger_motion_v2_params_window)
        self.action_proximity_control_params.triggered.connect(self.open_proximity_control_params_window)
        self.action_console_control_params.triggered.connect(self.open_console_control_params_window)
        self.sidebar_btn_start.clicked.connect(self._start_sidebar_control)
        self.sidebar_btn_stop.clicked.connect(self._stop_sidebar_control)
        self.sidebar_task_list.currentItemChanged.connect(self._on_sidebar_task_changed)

    def _set_sidebar_visible(self, visible: bool):
        """Updates sidebar visibility and keeps the toggle action text in sync."""
        if visible:
            self.sidebar.show()
            self.h_splitter.setSizes([self.width() - 250, 250])
            self.action_toggle_controls.setText('Hide Controls')
            self.action_toggle_controls.setToolTip('Hide the control sidebar')
        else:
            self.sidebar.hide()
            self.h_splitter.setSizes([self.width(), 0])
            self.action_toggle_controls.setText('Show Controls')
            self.action_toggle_controls.setToolTip('Show the control sidebar')

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

    def _show_sensor_capture_result(self):
        if not self._sensor_capture_series:
            self._append_sidebar_control_message("Sensor capture finished, but no samples were collected.")
            return

        if self._sensor_capture_result_dialog is not None and self._sensor_capture_result_dialog.isVisible():
            self._sensor_capture_result_dialog.close()

        self._sensor_capture_result_dialog = SensorCaptureResultDialog(
            self._sensor_capture_series,
            parent=self,
        )
        self._sensor_capture_result_dialog.show()

    def _read_sensor_peak_change_value(self):
        if self.ui_ros is None:
            return None

        sensor_functions = getattr(self.ui_ros, "sensor_functions", None)
        data_obj = getattr(sensor_functions, "_data", None) if sensor_functions is not None else None
        diff_data_ave = getattr(data_obj, "diffDataAve", None) if data_obj is not None else None
        if diff_data_ave is None:
            return None

        values = np.asarray(diff_data_ave, dtype=float)
        if values.size == 0:
            return None
        return float(np.max(np.abs(values)))

    def _send_sidebar_preset_joint_target(self):
        if self.ui_ros is None:
            self._append_sidebar_control_message("Ping UI is not ready yet.")
            return False

        robot_api = getattr(self.ui_ros, "robot_api", None)
        if robot_api is None or not hasattr(robot_api, "send_positions_joint_angle"):
            self._append_sidebar_control_message("Robot API is unavailable.")
            return False

        target_positions = [
            -0.743667827275479,
            -0.2904200138495811,
            -1.557570536189345,
            0.2835772211657043,
            -1.571901671805334,
            -0.024454002334083434,
        ]

        try:
            success = bool(robot_api.send_positions_joint_angle(target_positions))
        except Exception as exc:
            self._append_sidebar_control_message(f"Failed to send preset joint target: {exc}")
            return False

        if success:
            self._append_sidebar_control_message(
                "Preset joint target sent: "
                "[-0.7437, -0.2904, -1.5576, 0.2836, -1.5719, -0.0245]"
            )
            return True

        self._append_sidebar_control_message("Robot command was not accepted.")
        return False

    def _run_sidebar_control_action(self):
        action_id = self._sidebar_active_control_id or self._current_sidebar_control_id()
        if action_id == "hello_world":
            self._append_sidebar_control_message("hello world")
            return

        if action_id == "sensor_peak_change":
            peak_value = self._read_sensor_peak_change_value()
            if peak_value is None:
                self._append_sidebar_control_message("Sensor data is not ready for capture.")
                self._stop_sidebar_control(show_result=False, message="Control stopped")
                return

            elapsed = 0.0
            if self._sensor_capture_started_at is not None:
                elapsed = time.time() - self._sensor_capture_started_at
            self._sensor_capture_series.append((elapsed, peak_value))

            if elapsed >= self._sensor_capture_duration_sec:
                self._stop_sidebar_control(
                    show_result=True,
                    message=f"Sensor capture finished ({len(self._sensor_capture_series)} samples)",
                )
            return

    def _start_sidebar_control(self):
        if self._sidebar_control_timer is None:
            self._sidebar_control_timer = QTimer(self)
            self._sidebar_control_timer.timeout.connect(self._run_sidebar_control_action)

        if self._sidebar_control_timer.isActive():
            self._stop_sidebar_control(show_result=False, message="Control restarted")

        action_id = self._current_sidebar_control_id()
        self._sidebar_active_control_id = action_id

        if action_id == "hello_world":
            self._run_sidebar_control_action()
            self._sidebar_control_timer.start(1000)
            self.info_process.setText(
                f"Control started: {self._current_sidebar_control_label()}"
            )
            return

        if action_id == "sensor_peak_change":
            peak_value = self._read_sensor_peak_change_value()
            if peak_value is None:
                self._append_sidebar_control_message("Sensor data is not ready. Please build/update the sensor first.")
                self._sidebar_active_control_id = None
                return

            self._sensor_capture_started_at = time.time()
            self._sensor_capture_series = [(0.0, peak_value)]
            self._sidebar_control_timer.start(50)
            self.info_process.setText("Control started: Sensor Peak Change (10s)")
            return

        if action_id == "robot_preset_joint":
            self._send_sidebar_preset_joint_target()
            self._sidebar_active_control_id = None
            return

    def _stop_sidebar_control(self, show_result=True, message="Control stopped"):
        if self._sidebar_control_timer is not None:
            self._sidebar_control_timer.stop()
        was_sensor_capture = self._sidebar_active_control_id == "sensor_peak_change"
        self._sidebar_active_control_id = None
        self._sensor_capture_started_at = None
        self.info_process.setText(message)

        if was_sensor_capture and show_result:
            self._show_sensor_capture_result()

    def keyPressEvent(self, event):
        """Handles global key press events."""
        if not self.ping_mode_entered and event.key() == Qt.Key_1:
            self.run_ping_mode()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Executes when the main window is closed to ensure clean shutdown."""
        if self._fps_timer is not None:
            self._fps_timer.stop()
        if self._sidebar_control_timer is not None:
            self._sidebar_control_timer.stop()

        # Explicitly close any child windows to avoid orphaned processes
        if self.sensor_window:
            self.sensor_window.close()

        if self.ui_ros is not None and hasattr(self.ui_ros, "shutdown"):
            self.ui_ros.shutdown()

        event.accept()
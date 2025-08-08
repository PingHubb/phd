import sys
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtWidgets import (
    QWidget, QAction, QSplitter, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTreeWidget, QTreeWidgetItem, QToolBar, QStatusBar,
    QFrame, QMessageBox
)
from PyQt5.QtGui import QIcon
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
        # Return an empty icon if the file doesn't exist to prevent a crash
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

        # Automatically start the main functionality on launch
        self.run_ping_mode()

    def _setup_actions(self):
        """Creates all QAction objects used in menus and toolbars."""
        self.action_ping_mode = QAction(self.resources.get_icon('logo0.png'), 'Ping', self)

        self.action_log = QAction(self.resources.get_icon('logo2.png'), 'Log', self)

        self.action_sensor_signal = QAction(self.resources.get_icon('logo3.png'), "Sensor Signal", self)

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
        self.sidebar.hide()

        self.h_splitter.addWidget(self.view_container)
        self.h_splitter.addWidget(self.sidebar)
        self.h_splitter.setSizes([self.width(), 0])  # Sidebar is collapsed initially
        self.h_splitter.setCollapsible(1, True)

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

        # Container for Start/Stop buttons
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.addStretch()
        btn_start = QPushButton("Start")
        btn_start.setObjectName("btnStart")
        btn_layout.addWidget(btn_start)
        btn_stop = QPushButton("Stop")
        btn_stop.setObjectName("btnStop")
        btn_layout.addWidget(btn_stop)
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

    def toggle_sidebar(self):
        """Shows or hides the control sidebar."""
        if self.sidebar.isHidden():
            self.sidebar.show()
            # Restore to a reasonable size, e.g., 250px
            self.h_splitter.setSizes([self.width() - 250, 250])
            self.action_toggle_controls.setText('Hide Controls')
            self.action_toggle_controls.setToolTip('Hide the control sidebar')
        else:
            self.sidebar.hide()
            # Give all available space back to the main view
            self.h_splitter.setSizes([self.width(), 0])
            self.action_toggle_controls.setText('Show Controls')
            self.action_toggle_controls.setToolTip('Show the control sidebar')

    def open_sensor_signal_window(self):
        """
        Opens the sensor signal viewer window.
        Prevents opening if the main UI is not ready or if a window instance already exists.
        """
        if self.ui_ros is None:
            QMessageBox.warning(self, "Warning",
                                "Please wait for Ping Mode to load before opening the Sensor Signal Viewer.")
            return

        if self.sensor_window is not None and self.sensor_window.isVisible():
            # If window is already open, just bring it to the front
            self.sensor_window.raise_()
            self.sensor_window.activateWindow()
            return

        self.sensor_window = SensorSignalWindow(
            parent=self,
            sensor_functions_ref=self.ui_ros.sensor_functions
        )
        self.sensor_window.show()

    def run_ping_mode(self):
        """Initializes and displays the main 'Ping Mode' UI, replacing the placeholder."""
        if self.ping_mode_entered:
            return

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

    def keyPressEvent(self, event):
        """Handles global key press events."""
        if not self.ping_mode_entered and event.key() == Qt.Key_1:
            self.run_ping_mode()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Executes when the main window is closed to ensure clean shutdown."""
        # Explicitly close any child windows to avoid orphaned processes
        if self.sensor_window:
            self.sensor_window.close()
        event.accept()
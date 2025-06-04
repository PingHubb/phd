import time
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtWidgets import (
    QWidget, QAction, QSplitter, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QToolBar, QStatusBar, QFrame
)
from PyQt5.QtGui import QIcon, QColor
from pyvistaqt import QtInteractor, MainWindow
from phd.ui.ui_ping import UI
from phd.dependence.sensor_signal_window import SensorSignalWindow


class MyMainWindow(MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PingLab")
        self.setMinimumSize(960, 540)
        self.resize(1650, 900)

        self.setWindowIcon(QIcon('/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo0.png'))
        self.ping_mode_entered = False
        self.sensorWindow = None

        # Menus & toolbar
        self.setup_menu_and_toolbar()

        # Central layout
        self.setup_main_layout()

        # Signals
        self.connectFunction()

        # Styling (external)
        style_file = '/home/ping2/ros2_ws/src/phd/phd/resource/stylesheets/ui_style.qss'
        with open(style_file, 'r', encoding='utf-8') as f:
            self.setStyleSheet(f.read())

        # Start in Ping mode
        self.runPingMode()

    def setup_menu_and_toolbar(self):
        # Menu bar styling
        menu = self.menuBar()
        menu.setStyleSheet(
            "QMenuBar {"
            "  background: qlineargradient(spread:pad, x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #555555, stop:1 #333333);"
            "  color: white;"
            "}"
            "QMenuBar::item {"
            "  background: transparent;"
            "  padding: 4px 12px;"
            "}"
            "QMenuBar::item:selected {"
            "  background: qlineargradient(spread:pad, x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #777777, stop:1 #555555);"
            "}"
        )

        # Mode menu
        modeMenu = menu.addMenu('Mode')
        self.modePING = QAction('Ping', self)
        self.modePING.setIcon(QIcon('/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo0.png'))
        modeMenu.addAction(self.modePING)

        # Function Menu
        functionMenu = menu.addMenu('Menu')
        self.log = QAction('Log', self)
        self.log.setIcon(QIcon('/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo2.png'))
        functionMenu.addAction(self.log)

        # Sensor Signal action (will open a separate window)
        self.sensorsignal = QAction("Sensor Signal", self)
        self.sensorsignal.setIcon(QIcon('/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo3.png'))
        functionMenu.addAction(self.sensorsignal)

        # Toggle sidebar action
        self.toggleControls = QAction('Show controls', self)
        self.toggleControls.setIcon(QIcon('/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo5.png'))
        functionMenu.addAction(self.toggleControls)

        # Exit action
        self.exit = QAction('Exit', self)
        self.exit.setIcon(QIcon('/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo4.png'))
        self.exit.setShortcut('Ctrl+Q')
        functionMenu.addAction(self.exit)

        # Toolbar with gradient, accent hover/checked, separators
        tb = QToolBar("Main")
        tb.setIconSize(QSize(24, 24))
        tb.setStyleSheet("""
            QToolBar {
                background: qlineargradient(spread:pad, x1:0,y1:0,x2:1,y2:0,
                                            stop:0 #444444, stop:1 #222222);
                border: none;
            }
            QToolBar::separator {
                background: #555555; width: 1px; margin: 0 6px;
            }
            QToolButton {
                color: white; padding: 4px;
            }
            QToolButton:hover, QToolButton:checked {
                color: #1abc9c;
                background: qlineargradient(spread:pad, x1:0,y1:0,x2:1,y2:0,
                                            stop:0 #2a2a2a, stop:1 #3b3b3b);
            }
        """)

        tb.addAction(self.modePING)
        tb.addSeparator()
        tb.addAction(self.log)
        tb.addAction(self.sensorsignal)
        tb.addAction(self.toggleControls)
        tb.addSeparator()
        tb.addAction(self.exit)
        self.addToolBar(tb)

    def setup_main_layout(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        central.setStyleSheet("""
            QWidget#centralWidget {
                background: qlineargradient(spread:pad, x1:0,y1:0,x2:0,y2:1,
                                             stop:0 #444444, stop:1 #222222);
            }
            QLabel, QTreeWidget, QPushButton,
            QLineEdit, QComboBox, QTextEdit {
                color: white;
            }
            QGroupBox {
                border: 1px solid #555555;
                margin-top: 1.5em;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px; padding: 0 3px;
                color: #bd7666; font-weight: bold;
            }
            /* TAB BAR BACKGROUND */
            QTabBar {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #1a1a1a, stop:0.5 #262626, stop:1 #333333);
                padding: 2px;
                border-bottom: 2px solid #1abc9c; font-weight: bold;
            }
            /* unified tab styling */
            QTabBar::tab {
                background: #2b2b2b;
                padding: 12px 20px;
                color: white;
            }
            QTabBar::tab:selected {
                border: 1px solid #3f9ee4;
                color: #3f9ee4;
            }
            QListWidget {
                color: #9959ca;
            }
            QLineEdit, QTextEdit {
                background: #2b2b2b;
            }
            QPushButton#btnStart {
                background-color: #27ae60; color: white; font-weight: bold;
            }
            QPushButton#btnStop  {
                background-color: #c0392b; color: white; font-weight: bold;
            }
        """)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Horizontal splitter for 3D view and sidebar
        self.hsplit = QSplitter(Qt.Horizontal)
        self.hsplit.setHandleWidth(6)

        # 3D view pane
        self.view_container = QWidget()
        view_layout = QVBoxLayout(self.view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)
        self.view_placeholder = QLabel("Enter Ping Mode to view 3D")
        self.view_placeholder.setAlignment(Qt.AlignCenter)
        view_layout.addWidget(self.view_placeholder)
        self.hsplit.addWidget(self.view_container)

        # Sidebar pane (collapsed by default)
        self.sidebar = QFrame()
        self.sidebar.setFrameShape(QFrame.StyledPanel)
        self.sidebar.setMinimumWidth(200)
        self.sidebar.setStyleSheet("""
            QFrame {
                background: qlineargradient(spread:pad, x1:1,y1:0,x2:0,y2:1,
                                            stop:0 #333333, stop:1 #2a2a2a);
            }
        """)
        sb_layout = QVBoxLayout(self.sidebar)
        sb_layout.setContentsMargins(8, 8, 8, 8)
        sb_layout.setSpacing(12)

        # Controls tree
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        root = QTreeWidgetItem(self.tree, ['Controls'])
        font_root = root.font(0)
        font_root.setBold(True)
        root.setFont(0, font_root)
        root.setForeground(0, QColor('#1abc9c'))
        for i in range(root.childCount()):
            child = root.child(i)
            font_c = child.font(0)
            font_c.setItalic(True)
            child.setFont(0, font_c)
        sb_layout.addWidget(self.tree)

        # Start/Stop buttons
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setSpacing(8)
        btn_layout.addStretch()
        btn_start = QPushButton("Start")
        btn_start.setObjectName("btnStart")
        btn_layout.addWidget(btn_start)
        btn_stop = QPushButton("Stop")
        btn_stop.setObjectName("btnStop")
        btn_layout.addWidget(btn_stop)
        btn_layout.addStretch()
        sb_layout.addWidget(btn_container)

        self.hsplit.addWidget(self.sidebar)
        self.hsplit.setCollapsible(1, True)
        self.hsplit.setSizes([1, 0])
        main_layout.addWidget(self.hsplit)

        # Status bar
        status = QStatusBar()
        status.setStyleSheet("""
            QStatusBar {
                background: qlineargradient(spread:pad, x1:0,y1:0,x2:1,y2:0,
                                            stop:0 #222222, stop:1 #444444);
            }
            QStatusBar QLabel { color: white; }
        """)
        status.setContentsMargins(2, 0, 2, 0)

        self.info_process = QLabel("Ready")
        font_info = self.info_process.font()
        font_info.setBold(True)
        self.info_process.setFont(font_info)
        status.addWidget(self.info_process)

        self.info_FPS = QLabel("FPS: 0")
        font_fps = self.info_FPS.font()
        font_fps.setItalic(True)
        self.info_FPS.setFont(font_fps)
        status.addPermanentWidget(self.info_FPS)

        self.setStatusBar(status)

    def toggle_sidebar(self):
        sizes = self.hsplit.sizes()
        handle = self.hsplit.handleWidth()
        if sizes[1] == 0:
            min_w = self.sidebar.minimumWidth()
            total = self.hsplit.width() - handle
            left_w = max(total - min_w, 0)
            self.hsplit.setSizes([left_w, min_w])
            self.toggleControls.setToolTip('Hide controls')
        else:
            self.hsplit.setSizes([self.hsplit.width() - handle, 0])
            self.toggleControls.setToolTip('Show controls')

    def connectFunction(self):
        self.exit.triggered.connect(self.close)
        self.modePING.triggered.connect(self.runPingMode)
        self.log.triggered.connect(lambda:
            getattr(self, 'UIROS', None) and self.UIROS.toggle_plotter_visibility()
        )
        self.toggleControls.triggered.connect(self.toggle_sidebar)
        self.sensorsignal.triggered.connect(self.openSensorSignalWindow)

    def openSensorSignalWindow(self):
        if self.sensorWindow is not None:
            self.sensorWindow.raise_()
            self.sensorWindow.activateWindow()
            return

        # Create a new SensorSignalWindow with `self` as parent.
        # self.sensorWindow = SensorSignalWindow(parent=self, sensor_api=self.UIROS.sensor_api)
        self.sensorWindow = SensorSignalWindow(parent=self)
        self.sensorWindow.show()

    def runPingMode(self):
        if self.ping_mode_entered:
            return
        self.ping_mode_entered = True
        self.modePING.setDisabled(True)

        self.view_container.layout().removeWidget(self.view_placeholder)
        self.view_placeholder.deleteLater()
        self.UIROS = UI(Qt.Horizontal)
        self.view_container.layout().addWidget(self.UIROS)
        self.UIROS.reLayout()

        self._frame_count = 0
        iren = self.UIROS.findChild(QtInteractor)
        if iren is None:
            raise RuntimeError("QtInteractor not found in UI")
        iren.render_signal.connect(self.on_frame_rendered)
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self.update_fps)
        self._fps_timer.start(1000)
        self.info_process.setText("PING Mode Activated")

    def on_frame_rendered(self):
        self._frame_count += 1

    def update_fps(self):
        self.info_FPS.setText(f"FPS: {self._frame_count}")
        self._frame_count = 0

    def keyPressEvent(self, event):
        if not self.ping_mode_entered and event.key() == Qt.Key_1:
            self.runPingMode()
        else:
            super().keyPressEvent(event)

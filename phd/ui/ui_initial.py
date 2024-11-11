import time

from PyQt5.QtCore import pyqtSignal, Qt, QRect, QEvent, QSize
from PyQt5.QtWidgets import QWidget, QAction, QGridLayout, QSplitter, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QTreeWidget, QTreeWidgetItem, QComboBox
from pyvistaqt import QtInteractor, MainWindow
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon, QMouseEvent, QPixmap, QCursor, QColor, QBrush, QFont
from phd.ui.ui_meshlab import MeshLabSplitter
from phd.ui.ui_sensor import SensorSplitter
from phd.ui.ui_ping import RosSplitter


class MyMainWindow(MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PingLab")
        self.setMinimumSize(192 * 5, 108 * 5)
        self.resize(192 * 8, 108 * 8)
        self.setWindowIcon(QIcon('/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo0.png'))

        # Flag to track if Ping mode has been entered
        self.ping_mode_entered = False

        # 创建菜单栏并设置样式
        self.setup_menu()

        # 创建主布局并添加组件
        self.setup_main_layout()

        self.connectFunction()

        style_file = '/home/ping2/ros2_ws/src/phd/phd/resource/stylesheets/ui_style.qss'
        with open(style_file, 'r', encoding='UTF-8') as file:
            self.style_sheet = file.read()
        self.setObjectName("windowMain")
        self.setStyleSheet(self.style_sheet)

    def setup_menu(self):
        # Change MenuBar below------------------------------------------------------------------------------------------
        # 创建菜单栏并设置其样式
        mainMenu = self.menuBar()
        mainMenu.setObjectName("menuBar")

        # Add pages for menubar below-----------------------------------------------------------------------------
        # 添加Mode菜单
        modeMenu = mainMenu.addMenu('Mode')
        modeMenu.setObjectName("menuMode")

        self.modeMeshLab = QAction('MeshLab', self)
        self.modeMeshLab.setIcon(QIcon("/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo1.png"))
        modeMenu.addAction(self.modeMeshLab)

        self.modeSensor = QAction('Sensor', self)
        self.modeSensor.setIcon(QIcon("/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo2.png"))
        modeMenu.addAction(self.modeSensor)

        self.modePING = QAction('Ping', self)
        self.modePING.setIcon(QIcon("/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo3.png"))
        modeMenu.addAction(self.modePING)

        # 添加Mesh菜单
        fileMenu = mainMenu.addMenu('Help')
        fileMenu.setObjectName("menuFile")

        self.about = QAction('About', self)
        self.about.setIcon(QIcon("/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo4.png"))
        fileMenu.addAction(self.about)

        self.exit = QAction('Exit', self)
        self.exit.setIcon(QIcon("/home/ping2/ros2_ws/src/phd/phd/resource/icon/logo5.png"))
        self.exit.setShortcut('Ctrl+Q')
        fileMenu.addAction(self.exit)

    def setup_main_layout(self):
        # Change MainLayout below---------------------------------------------------------------------------------------
        # 创建主布局
        main_layout = QGridLayout()

        # 创建上下分割器0(上方显示区+树+功能，下方信息区)
        self.splitter_0 = QSplitter(Qt.Vertical, self)
        self.splitter_0.setHandleWidth(3)
        widget_void = QWidget()

        # 添加信息栏部分
        self.widget_info = QWidget()
        self.widget_info.setObjectName("widgetInfo")
        self.widget_info.setFixedHeight(20)

        layout = QHBoxLayout(self.widget_info)
        self.info_process = QLabel("Hello, Ping!")
        self.info_FPS = QLabel("Hello, Ping!")
        self.info_FPS.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.info_process.resize(100, 20)
        self.info_FPS.resize(100, 20)
        layout.addWidget(self.info_process)
        layout.addWidget(self.info_FPS)
        layout.setContentsMargins(5, 0, 5, 0)
        self.splitter_0.addWidget(widget_void)
        self.splitter_0.addWidget(self.widget_info)
        self.splitter_0.setCollapsible(1, False)

        # 添加分割器到主布局
        main_layout.addWidget(self.splitter_0)
        main_layout.setContentsMargins(3, 3, 3, 3)

        # 设置主窗口的中心布局
        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(main_layout)
        self.centralWidget().setObjectName("widgetMain")

    def connectFunction(self):
        self.exit.triggered.connect(self.close)
        self.modeMeshLab.triggered.connect(self.runMeshLab)
        self.modeSensor.triggered.connect(self.runSensor)
        self.modePING.triggered.connect(self.runPingMode)

    def runMeshLab(self):
        self.modeMeshLab.setDisabled(True)
        self.modeSensor.setEnabled(True)
        self.modePING.setEnabled(True)
        self.UIMeshLab = MeshLabSplitter(Qt.Horizontal)
        self.splitter_0.replaceWidget(0, self.UIMeshLab)
        self.UIMeshLab.reLayout()

    def runSensor(self):
        self.modeMeshLab.setEnabled(True)
        self.modeSensor.setDisabled(True)
        self.modePING.setEnabled(True)
        self.UISensor = SensorSplitter(Qt.Horizontal)
        self.splitter_0.replaceWidget(0, self.UISensor)
        self.UISensor.reLayout()

    def runPingMode(self):
        self.modeMeshLab.setEnabled(True)
        self.modeSensor.setEnabled(True)
        self.modePING.setDisabled(True)  # Current mode, disable button to prevent reactivation

        # Update the label to show the ROS mode message
        self.info_process.setText("Hello, PING Mode!")
        self.info_FPS.setText("FPS Display for PING")

        # Create a new ROS mode widget
        self.UIROS = RosSplitter(Qt.Horizontal)
        self.splitter_0.replaceWidget(0, self.UIROS)
        self.UIROS.reLayout()

        # Set the flag to indicate Ping mode has been entered
        self.ping_mode_entered = True

    def keyPressEvent(self, event):
        if not self.ping_mode_entered and event.key() == Qt.Key_1:
            self.runPingMode()
        else:
            super().keyPressEvent(event)

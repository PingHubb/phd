# ui_design.py

from PyQt5.QtCore import pyqtSignal, Qt,QRect, QEvent,QSize
from PyQt5.QtWidgets import QWidget, QAction,QGridLayout, QSplitter, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton,QTreeWidget,QTreeWidgetItem,QComboBox
from pyvistaqt import QtInteractor, MainWindow
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon, QMouseEvent, QPixmap, QCursor, QColor, QBrush, QFont
from phd.ui.ui_meshlab import MeshLabSplitter


class PlotterWidget(QWidget):
    filesDropped = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        file_paths = []
        for url in event.mimeData().urls():
            file_paths.append(str(url.toLocalFile()))
        self.filesDropped.emit(file_paths)  # 发射带有文件路径列表参数的信号


class TreeWidgetItem(QTreeWidgetItem):
    def __init__(self,parent,name:str,level:int,type:int):
        super().__init__(parent)
        self.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)  # 第一列左对齐
        self.iconVisible = QIcon("./resource/visible.png")
        self.iconUnvisible = QIcon("./resource/unvisible.png")
        self.iconVisibleSelected = QIcon("./resource/visible_selected.png")
        self.iconUnvisibleSelected = QIcon("./resource/unvisible_selected.png")
        self.level = level
        self._type = type
        self.visible = True
        self.setIcon(0, self.iconVisible)
        self.setText(0,name)
        if not self.level:
            if self._type == 0:
                childVertex = TreeWidgetItem(self,"Vertex",1,0)
                childEdges = TreeWidgetItem(self,"Edges",1,1)
                childFaces = TreeWidgetItem(self,"Faces",1,2)
                childNVertex = TreeWidgetItem(self,"N_Vertex",1,3)
                childNEdges = TreeWidgetItem(self,"N_Edges",1,4)
                childNFaces = TreeWidgetItem(self,"N_Faces",1,5)


class TreeWidget(QTreeWidget):
    icon_clicked = pyqtSignal(TreeWidgetItem)

    def __init__(self) -> None:
        super().__init__()
        self.setAlternatingRowColors(True)
        self.setHeaderHidden(True)
        self.itemClicked.connect(self.handle_item_clicked)
        
    def handle_item_clicked(self, item:TreeWidgetItem):
        if item.level:
            item.parent().setSelected(True)
            item.parent().setForeground(0,QColor(255,0,0))
        item_rect = self.visualItemRect(item)
        icon_rect = QRect(0,0,20,20)
            
        if icon_rect.contains(self.viewport().mapFromGlobal(QCursor.pos()) - item_rect.topLeft()):
            if item.visible:
                item.setIcon(0, item.iconUnvisible)
                item.visible = False
                if not item.level:
                    for i in range(item.childCount()):
                        item.child(i).visible = False
                        item.child(i).setIcon(0,item.iconUnvisible)
                else:
                    any_show = False
                    for i in range(item.parent().childCount()):
                        if item.parent().child(i).visible:
                            any_show = True
                    if not any_show:
                        item.parent().visible = False
                        item.parent().setIcon(0,item.iconUnvisible)
            else:
                item.setIcon(0, item.iconVisible)
                item.visible = True
                if not item.level:
                    for i in range(item.childCount()):
                        item.child(i).visible = True
                        item.child(i).setIcon(0,item.iconVisible)
                else:
                    any_show = False
                    for i in range(item.parent().childCount()):
                        if item.parent().child(i).visible:
                            any_show = True
                        print(any_show)
                    if any_show:
                        item.parent().visible = True
                        item.parent().setIcon(0,item.iconVisible)

            self.icon_clicked.emit(item)

    def viewportEvent(self, event):
        if event.type() == QEvent.HoverMove:
            pos = event.pos()
            item = self.itemAt(pos)
            if item:
                item_rect = self.visualItemRect(item)
                icon_rect = QRect(2,2,16,16)
                if icon_rect.contains(self.viewport().mapFromGlobal(QCursor.pos()) - item_rect.topLeft()): 
                    if item.visible:
                        item.setIcon(0, item.iconVisibleSelected)
                    else:
                        item.setIcon(0, item.iconUnvisibleSelected)
                else:
                    if item.visible:
                        item.setIcon(0, item.iconVisible)
                    else:
                        item.setIcon(0, item.iconUnvisible)
        return super().viewportEvent(event)


class MyMainWindow(MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PingLab")
        self.setMinimumSize(192*5, 108*5)
        self.resize(192*8, 108*8)
        self.setWindowIcon(QIcon('./resource/logo0.png'))

        # 创建菜单栏并设置样式
        self.setup_menu()

        # 创建主布局并添加组件
        self.setup_main_layout()

        self.connectFunction()
        
        style_file = '/home/ping2/ros2_ws/src/phd/phd/stylesheets/ui_style.qss'
        with open(style_file, 'r',  encoding='UTF-8') as file:
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
        self.modeMeshLab.setIcon(QIcon("./resource/logo1.png"))
        modeMenu.addAction(self.modeMeshLab)
        
        self.modeROS = QAction('ROS', self)
        self.modeROS.setIcon(QIcon("./resource/logo3.png"))
        modeMenu.addAction(self.modeROS)

        # 添加Mesh菜单
        fileMenu = mainMenu.addMenu('Help')
        fileMenu.setObjectName("menuFile")

        self.about = QAction('About', self)
        self.about.setIcon(QIcon("./resource/logo4.png"))
        fileMenu.addAction(self.about)
        
        self.exit = QAction('Exit', self)
        self.exit.setIcon(QIcon("./resource/logo5.png"))
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
        self.info_process.resize(100,20)
        self.info_FPS.resize(100,20)
        layout.addWidget(self.info_process)
        layout.addWidget(self.info_FPS)
        layout.setContentsMargins(5,0,5,0)
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
        self.modeROS.triggered.connect(self.runROSMode)

    def runMeshLab(self):
        self.modeMeshLab.setDisabled(True)
        self.modeROS.setEnabled(True)
        self.UIMeshLab = MeshLabSplitter(Qt.Horizontal)
        self.splitter_0.replaceWidget(0,self.UIMeshLab)
        self.UIMeshLab.reLayout()

    def runROSMode(self):
        # Enable MeshLab mode button and disable ROS mode button
        self.modeMeshLab.setEnabled(True)  # MeshLab mode can be activated again
        self.modeROS.setDisabled(True)  # Current mode, disable button to prevent reactivation

        # Update the label to show the ROS mode message
        self.info_process.setText("Hello, ROS Mode!")
        self.info_FPS.setText("FPS Display for ROS")

        # Optionally, you can reset the layout to show something specific for ROS mode
        # This is just a placeholder for any specific GUI elements you want to add for ROS mode
        # Example of resetting the layout for ROS mode
        ui_ros = QWidget()  # Create a new widget for ROS mode
        layout_ros = QHBoxLayout(ui_ros)
        layout_ros.addWidget(QLabel("ROS Mode Active"))
        self.splitter_0.replaceWidget(0, ui_ros)  # Replace the current widget in splitter with ROS mode widget




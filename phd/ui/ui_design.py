# ui_design.py

from PyQt5.QtCore import pyqtSignal, Qt, QRect, QEvent, QSize
from PyQt5.QtWidgets import QWidget, QAction, QGridLayout, QSplitter, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton, QTreeWidget, QTreeWidgetItem, QComboBox
from pyvistaqt import QtInteractor, MainWindow
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon, QMouseEvent, QPixmap, QCursor,QColor, QBrush, QFont


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

        self.setWindowTitle("SingaLab")
        self.setMinimumSize(192*5, 108*5)
        self.resize(192*8, 108*8)
        self.setWindowIcon(QIcon('./resource/logo0.png'))
        style_file = './stylesheets/ui_style.qss'
        with open(style_file, 'r',  encoding='UTF-8') as file:
            self.style_sheet = file.read()
        self.setObjectName("windowMain")
        self.setStyleSheet(self.style_sheet)

        # 创建菜单栏并设置样式
        self.setup_menu()

        # 创建主布局并添加组件
        self.setup_main_layout()
        
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
        
        exitButton = QAction('Exit', self)
        exitButton.setIcon(QIcon("./resource/logo5.png"))
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)


    def setup_main_layout(self):
        # Change MainLayout below---------------------------------------------------------------------------------------
        # 创建主布局
        main_layout = QGridLayout()
        
        # 创建上下分割器0(上方显示区+树+功能，下方信息区)
        self.splitter_0 = QSplitter(Qt.Vertical, self)
        self.splitter_0.setObjectName("splitter0")
        self.splitter_0.setHandleWidth(3)
        # 创建左右分割器1(左方显示区，右方树+功能)
        self.splitter_1 = QSplitter(Qt.Horizontal, self)
        self.splitter_1.setObjectName("splitter1")
        self.splitter_1.setHandleWidth(3)

        # 创建Plotter部分
        self.widget_plotter = PlotterWidget()
        self.widget_plotter.setObjectName("widgetPlotter")
        self.layout_plotter = QGridLayout(self.widget_plotter)
        self.layout_plotter.setContentsMargins(1,1,1,1)
        self.plotter = QtInteractor(self)
        
        self.plotter.background_color = '#303030'
        self.widget_plotter.setAcceptDrops(True)
        self.layout_plotter.addWidget(self.plotter.interactor)
        self.widget_plotter.setLayout(self.layout_plotter)
        
        # 创建上下分割器2(上方树，下方功能)
        self.splitter_2 = QSplitter(Qt.Vertical, self)
        self.splitter_2.setObjectName("splitter2")
        self.splitter_2.setHandleWidth(3)

        # 创建Tree部分
        self.widget_tree = TreeWidget()
        self.widget_tree.setObjectName("widgetTree")
        
        # 创建Function部分
        self.widget_func = QWidget()
        self.widget_func.setObjectName("widgetFunc")
        self.layout_func = QVBoxLayout()
        self.widget_func.setLayout(self.layout_func)

        # 添加组件到分割器2
        self.splitter_2.addWidget(self.widget_tree)
        self.splitter_2.addWidget(self.widget_func)
        self.splitter_2.setStretchFactor(0, 1)
        self.splitter_2.setStretchFactor(1, 2)

        # 添加组件到分割器1
        self.splitter_1.addWidget(self.widget_plotter)
        self.splitter_1.addWidget(self.splitter_2)
        self.splitter_1.setStretchFactor(0, 9)
        self.splitter_1.setStretchFactor(1, 3)

        # 添加信息栏部分
        self.widget_info = QWidget()
        self.widget_info.setObjectName("widgetInfo")
        self.widget_info.setFixedHeight(20)

        layout = QHBoxLayout(self.widget_info)
        self.label_info = QLabel("Hello, Singa!")
        
        self.label_info.resize(100,20)
        layout.addWidget(self.label_info)
        layout.setContentsMargins(5,0,0,0)
        self.splitter_0.addWidget(self.splitter_1)
        self.splitter_0.addWidget(self.widget_info)
        self.splitter_0.setCollapsible(1, False)

        # 添加分割器到主布局
        main_layout.addWidget(self.splitter_0)
        main_layout.setContentsMargins(3, 3, 3, 3)

        # 设置主窗口的中心布局
        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(main_layout)
        self.centralWidget().setObjectName("widgetMain")
        # Add controllers for Tree below-------------------------------------------------------------------------

        # Add controllers for Function below---------------------------------------------------------------------
        self.add_sphere_button = QPushButton("Add Sphere",self.widget_func)
        self.layout_func.addWidget(self.add_sphere_button)
        
        self.add_robot_button = QPushButton("addRobot",self.widget_func)
        self.layout_func.addWidget(self.add_robot_button)

        self.hand_construction = QPushButton("handConstruction",self.widget_func)
        self.layout_func.addWidget(self.hand_construction)

        self.serial_channel = QComboBox(self.widget_func)
        self.layout_func.addWidget(self.serial_channel)

        self.serial_connection = QPushButton("serialConnection",self.widget_func)
        self.layout_func.addWidget(self.serial_connection)

        self.show_edges_checkbox = QCheckBox("Show Edges",self.widget_func)
        self.layout_func.addWidget(self.show_edges_checkbox)


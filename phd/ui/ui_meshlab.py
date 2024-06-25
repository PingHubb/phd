from PyQt5 import QtCore
from PyQt5.QtGui import QDragEnterEvent,QDropEvent,QIcon,QColor,QCursor
from PyQt5.QtCore import pyqtSignal,Qt,QRect,QEvent
from PyQt5.QtWidgets import QSplitter,QWidget,QGridLayout,QTreeWidgetItem,QTreeWidget,QVBoxLayout,QPushButton,QCheckBox,QComboBox
from pyvistaqt import QtInteractor
from phd.dependence.func_meshLab import MyMeshLab

class PlotterWidget(QWidget):
    filesDropped = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
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


class MeshLabSplitter(QSplitter):
    def __init__(self, orientation: QtCore.Qt.Orientation) -> None:
        super().__init__(orientation)
        self.setHandleWidth(3)
        self.setup_layout()
        self.functions = MyMeshLab(self)
        self.connect_function()

    def setup_layout(self):
        # 创建Plotter部分
        self.widget_plotter = PlotterWidget()
        self.widget_plotter.setObjectName("widgetPlotter")

        self.layout_plotter = QGridLayout(self.widget_plotter)
        self.layout_plotter.setContentsMargins(0,0,0,0)
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

        # 添加组件到分割器1
        self.addWidget(self.widget_plotter)
        self.addWidget(self.splitter_2)
        
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
        
    def reLayout(self):
        self.setSizes([round((self.width()-3)*6/7),round((self.width()-3)/7)])
        self.splitter_2.setSizes([round((self.height()-3)/2),self.height()-3-round((self.height()-3)/2)])

    def connect_function(self):
        self.widget_plotter.filesDropped.connect(lambda file_paths: self.functions.loadMesh(file_paths))
        self.add_sphere_button.pressed.connect(lambda: self.functions.add_sphere(self.show_edges_checkbox.isChecked()))
        self.add_robot_button.pressed.connect(lambda: self.functions.addRobot())
        self.hand_construction.pressed.connect(lambda: self.functions.Init_handConstruction())
        self.serial_connection.pressed.connect(lambda: self.functions.serialConnection())
        self.widget_tree.icon_clicked.connect(lambda item: self.functions.changeVisibility(item))

        
from PyQt5 import QtCore
from PyQt5.QtGui import QDragEnterEvent,QDropEvent,QIcon,QColor,QCursor, QMouseEvent,QWheelEvent,QPainter,QPen,QPixmap
from PyQt5.QtCore import pyqtSignal,Qt,QRect,QEvent,QLine, QTimer
from PyQt5.QtWidgets import QSplitter,QWidget,QGridLayout,QTreeWidgetItem,QTreeWidget,QVBoxLayout,QPushButton,QCheckBox,QComboBox,QScrollArea,QLabel,QSizePolicy
from pyvistaqt import QtInteractor
from dependence.func_knitPaint import MyKnitPaint
import time

class ImageLabel(QLabel):
    scrollSignal = pyqtSignal(int,int)
    scrollScallSignal = pyqtSignal(int,int)

    def __init__(self):
        super().__init__()
        self.setObjectName("mapKnitting")
        self.setMouseTracking(True)
        # self.setCursor(Qt.CrossCursor)
        self.start_pos = None
        self.end_pos = None
        self.current_pos = None
        self.is_shift = False
        self.grid_size = 20
        self.original_scale_factor = 1
        self.current_scale_factor = 1
        self.scale_factor_inter = 0.1
        self.original_pixmap = None
        self.min_scale_factor = 0.2
        self.max_scale_factor = 2
        self.gridShow_scale_factor = 0.3
        self.setAlignment(Qt.AlignCenter)

        
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.start_pos = event.pos()
            self.is_shift = True
    
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MiddleButton:
            self.end_pos = event.pos()
            offset = (self.start_pos - self.end_pos)
            self.scrollSignal.emit(offset.x(),offset.y())
            
            # pixmap_pos = self.mapToParent(self.pixmap().rect().bottomLeft())
            # print(pixmap_pos)

        self.current_pos = event.pos()
        self.update()
    
    def leaveEvent(self,event):
        self.current_pos = None
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        
        print(event.pos(),self.mapToParent(self.rect().bottomLeft()),self.pixmap().rect(),self.size(),self.parent().size())


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.start_pos = None
            self.end_pos = None
            self.is_shift = False

    # def centerScale(self):
    #     image_center = QPointF(self.pixmap().width() / 2, self.pixmap().height() / 2)
    #     diff = QPointF(image_center.x() - self.parent().width() / 2,image_center.y() - self.parent().height() / 2)
    #     self.scrollScallSignal.emit(int(diff.x()),int(diff.y()))

    def wheelEvent(self,event):
        if self.original_pixmap ==None:
            return
        # 获取滚轮的滚动方向和角度
        angle = event.angleDelta().y()
        label_center = self.rect().center()
        diff_0 = event.pos()-label_center
        label_pos = self.mapToParent(event.pos())

        # 根据滚轮滚动方向进行缩放
        scale_factor_inter = 1
        if angle > 0:
            scale_factor_inter = 1 + self.scale_factor_inter
        else:
            scale_factor_inter = 1 - self.scale_factor_inter

        if self.current_scale_factor * scale_factor_inter > self.max_scale_factor:
            scale_factor_inter = self.max_scale_factor/self.current_scale_factor
            self.current_scale_factor = self.max_scale_factor
        elif self.current_scale_factor * scale_factor_inter < self.min_scale_factor:
            scale_factor_inter = self.min_scale_factor/self.current_scale_factor
            self.current_scale_factor = self.min_scale_factor
        else:
            self.current_scale_factor *= scale_factor_inter  

        diff_1 = diff_0 * scale_factor_inter
        label_center *= scale_factor_inter
        x = label_center.x() + diff_1.x() - label_pos.x()
        y = label_center.y() + diff_1.y() - label_pos.y()
        # print(self.rect().center(),self.pixmap().rect().center())

        # 根据缩放因子调整 QPixmap 大小
        scaled_pixmap = self.original_pixmap.scaled(
            int(self.original_pixmap.width() * self.current_scale_factor),
            int(self.original_pixmap.height() * self.current_scale_factor),
            Qt.AspectRatioMode.IgnoreAspectRatio
        )
        # 在 QLabel 中显示缩放后的 QPixmap
        self.setPixmap(scaled_pixmap)
        self.scrollScallSignal.emit(int(x),int(y))

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.original_pixmap != None:
            if self.current_scale_factor > self.gridShow_scale_factor:
                painter = QPainter(self)
                painter.setPen(QPen(Qt.gray, 2, Qt.SolidLine))
                LineList = []
                if self.width() == self.parent().width():
                    offset_x = ((self.parent().width() - self.pixmap().width())/2)%(20*self.current_scale_factor)
                    for i in range(round(self.width()/(20*self.current_scale_factor)) + 1):
                        ver_line = QLine((20*self.current_scale_factor)*i + offset_x,0,(20*self.current_scale_factor)*i + offset_x,self.height())
                        LineList.append(ver_line)
                else:
                    for i in range(round(self.width()/(20*self.current_scale_factor)) + 1):
                        ver_line = QLine((20*self.current_scale_factor)*i,0,(20*self.current_scale_factor)*i,self.height())
                        LineList.append(ver_line)
                if self.height() == self.parent().height():
                    offset_y = ((self.parent().height() - self.pixmap().height())/2)%(20*self.current_scale_factor)
                    for i in range(round(self.height()/(20*self.current_scale_factor)) + 1):
                        hor_line = QLine(0,(20*self.current_scale_factor)*i + offset_y,self.width(),(20*self.current_scale_factor)*i + offset_y)
                        LineList.append(hor_line)
                else:
                    for i in range(round(self.height()/(20*self.current_scale_factor)) + 1):
                        hor_line = QLine(0,(20*self.current_scale_factor)*i,self.width(),(20*self.current_scale_factor)*i)
                        LineList.append(hor_line)
                painter.drawLines(LineList)
                painter.end()
            if self.current_pos != None:
                global_pos = QCursor.pos()
                label_pos = self.mapFromGlobal(global_pos)
                painter = QPainter(self)
                painter.setPen(QPen(Qt.green, 1, Qt.DashLine))
                hor_line = QLine(0,label_pos.y(),self.width(),label_pos.y())
                ver_line = QLine(label_pos.x(),0,label_pos.x(),self.height())
                painter.drawLines(hor_line,ver_line)
                painter.end()

class KnittingMap(QScrollArea):
    def __init__(self):
        super().__init__()
        self.image = ImageLabel()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.putItCenter)
        self.setup()
        
    
    def setup(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("widgetKnitting")
        self.setWidget(self.image)
        self.image.scrollSignal.connect(self.scrollImageLabel)
        self.image.scrollScallSignal.connect(self.scrollImageLabel_scall)
        self.setWidgetResizable(True)
        
    def setImage(self, image:QPixmap):
        self.image.original_pixmap = image
        self.image.setPixmap(self.image.original_pixmap)
        
        self.timer.start(0)
    def wheelEvent(self, event: QWheelEvent):
        event.ignore()  # 忽略滚轮事件，阻止滚动区域的滚动

    def scrollImageLabel(self,x,y):
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + x)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + y)

    def scrollImageLabel_scall(self, x, y):
        self.horizontalScrollBar().setMaximum(2*x)
        self.verticalScrollBar().setMaximum(2*y)
        self.horizontalScrollBar().setValue(x)
        self.verticalScrollBar().setValue(y)

    def putItCenter(self):
        if self.image.pixmap().height()>self.height():
            self.verticalScrollBar().setValue((self.image.pixmap().height()-self.height()+10)/2)
        if self.image.pixmap().width()>self.width():
            self.horizontalScrollBar().setValue((self.image.pixmap().width()-self.width()+10)/2)
        print(self.image.width(),self.image.height(),self.width()-10,self.height()-10)
        print((self.image.width()-self.width()+10)/2)

class KnitPaintSplitter(QSplitter):
    def __init__(self, orientation: QtCore.Qt.Orientation) -> None:
        super().__init__(orientation)
        self.map_packaged:QPixmap = None
        self.map_unpackaged:QPixmap = None
        self.map_pattern_packaged:QPixmap = None
        self.map_pattern_unpackaged:QPixmap = None
        self.widget_packaged = KnittingMap()
        self.widget_unpackaged = KnittingMap()
        self.splitter_2 = QSplitter(Qt.Vertical, self)
        self.widget_pattern_packaged = KnittingMap()
        self.widget_pattern_unpackaged = KnittingMap()
        self.widget_func = QWidget()
        self.layout_func = QVBoxLayout()
        self.setObjectName("knitting")
        self.add_sphere_button = QPushButton("Add Sphere",self.widget_func)
        self.add_robot_button = QPushButton("addRobot",self.widget_func)
        self.hand_construction = QPushButton("handConstruction",self.widget_func)
        self.serial_connection = QPushButton("serialConnection",self.widget_func)
        self.show_edges_checkbox = QCheckBox("Show Edges",self.widget_func)
        self.serial_channel = QComboBox(self.widget_func)
        self.setHandleWidth(3)

        self.setup_layout()
        # self.functions = MyKnitPaint(self)
        self.connect_function()

    def reLayout(self):
        self.setSizes([round((self.width()-6)/3),round((self.width()-6)/3),self.width()-6-round((self.width()-6)/3)-round((self.width()-6)/3)])
        self.splitter_2.setSizes([round((self.height()-3)/4),round((self.height()-3)/4),self.height()-3-round((self.height()-3)/4)-round((self.height()-3)/4)])

    def setup_layout(self):
        self.addWidget(self.widget_packaged)

        self.addWidget(self.widget_unpackaged)

        self.splitter_2.setHandleWidth(3)
        self.splitter_2.setObjectName("knitting")
        self.addWidget(self.splitter_2)

        self.splitter_2.addWidget(self.widget_pattern_packaged)

        self.splitter_2.addWidget(self.widget_pattern_unpackaged)

        self.widget_func.setObjectName("widgetFunc")
        self.widget_func.setLayout(self.layout_func)
        self.splitter_2.addWidget(self.widget_func)
        
        self.layout_func.addWidget(self.add_sphere_button)
        
        self.layout_func.addWidget(self.add_robot_button)

        self.layout_func.addWidget(self.hand_construction)

        self.layout_func.addWidget(self.serial_channel)

        self.layout_func.addWidget(self.serial_connection)

        self.layout_func.addWidget(self.show_edges_checkbox)
    
    def connect_function(self):
        self.add_sphere_button.pressed.connect(lambda: self.test())


    def test(self):
        self.map_packaged:QPixmap = QPixmap("./resource/1.png")
        self.widget_packaged.setImage(self.map_packaged)
        self.map_unpackaged:QPixmap = QPixmap("./resource/logo1.png")
        self.widget_unpackaged.setImage(self.map_unpackaged)
        self.map_pattern_packaged:QPixmap = QPixmap("./resource/logo2.png")
        self.widget_pattern_packaged.setImage(self.map_pattern_packaged)
        self.map_pattern_unpackaged:QPixmap = QPixmap("./resource/logo3.png")
        self.widget_pattern_unpackaged.setImage(self.map_pattern_unpackaged)


        
# main.py
import sys
from PyQt5 import QtWidgets,QtGui
from ui.ui_initial import MyMainWindow
from dependence.functionality import MyFunction
from dependence.func_knitPaint import MyKnitPaint
from dependence.func_ROS import MyROS

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont('Calibri', 12))
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
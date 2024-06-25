# phd_ui.py
import sys
from PyQt5 import QtWidgets, QtGui
from phd.ui.ui_initial import MyMainWindow
import rclpy


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont('Calibri', 12))
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
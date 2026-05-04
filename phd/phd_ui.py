import sys
from PyQt5 import QtWidgets, QtGui
from phd.ui.ui_initial import MyMainWindow


def main() -> int:
    print("Starting PingLab...")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Only apply Calibri when it is available on the host system.
    if "Calibri" in QtGui.QFontDatabase().families():
        app.setFont(QtGui.QFont("Calibri", 12))

    window = MyMainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())


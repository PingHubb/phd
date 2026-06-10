import os
import sys
import tempfile


_DEFAULT_MPLCONFIGDIR = os.path.join(tempfile.gettempdir(), "pinglab-matplotlib")
os.makedirs(_DEFAULT_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _DEFAULT_MPLCONFIGDIR)

from PyQt5 import QtWidgets, QtGui  # noqa: E402
from phd.ui.ui_initial import MyMainWindow  # noqa: E402


def main() -> int:
    print("Starting PingLab...")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Only apply Calibri when it is available on the host system.
    if "Calibri" in QtGui.QFontDatabase().families():
        app.setFont(QtGui.QFont("Calibri", 12))

    window = MyMainWindow()
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

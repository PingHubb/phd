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

    # Prefer the design-system font (Inter); fall back to Calibri if present.
    families = QtGui.QFontDatabase().families()
    for family in ("Inter", "Calibri"):
        if family in families:
            app.setFont(QtGui.QFont(family, 11))
            break

    window = MyMainWindow()
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

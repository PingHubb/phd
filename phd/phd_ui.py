import os
import sys
import tempfile


_DEFAULT_MPLCONFIGDIR = os.path.join(tempfile.gettempdir(), "pinglab-matplotlib")
os.makedirs(_DEFAULT_MPLCONFIGDIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _DEFAULT_MPLCONFIGDIR)

from PyQt5 import QtWidgets, QtGui  # noqa: E402
from phd.ui import theme  # noqa: E402
from phd.ui.ui_initial import MyMainWindow  # noqa: E402


def main() -> int:
    print("Starting PingLab...")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Family and size both come from the design tokens, so the application
    # font matches what the stylesheet paints with.
    families = set(QtGui.QFontDatabase().families())
    for family in theme.FONT_FAMILY_PREFERENCE:
        if family in families:
            app.setFont(QtGui.QFont(family, theme.FONT_POINT_SIZE_BASE))
            break

    window = MyMainWindow()
    window.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

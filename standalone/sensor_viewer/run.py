"""GUI entry point used by both Python and the Windows native launcher."""

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import traceback

HERE = Path(__file__).resolve().parent
# Source checkout support. A packaged copy contains just two sensor helpers,
# not the research UI, robot drivers or their dependencies.
checkout = HERE.parents[1]
if (checkout / "phd" / "dependence" / "sensor_api.py").is_file():
    sys.path.insert(0, str(checkout))
sys.path.insert(0, str(HERE))

log_folder = Path(os.environ.get("LOCALAPPDATA", str(Path.home()/".local"/"state"))) / "PingLabSensorViewer"
log_folder.mkdir(parents=True, exist_ok=True)
handler = RotatingFileHandler(log_folder/"viewer.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8")
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(message)s")


class LogWriter:
    def write(self, text):
        if text.strip():
            logging.info(text.strip())
        return len(text)

    def flush(self):
        pass


if sys.stdout is None:
    sys.stdout = LogWriter()
if sys.stderr is None:
    sys.stderr = LogWriter()

_dll_directories = []
if sys.platform == "win32":
    runtime = Path(sys.executable).parent
    qt = runtime / "Lib" / "site-packages" / "PyQt5" / "Qt5"
    if qt.is_dir():
        _dll_directories.append(os.add_dll_directory(str(qt/"bin")))
        os.environ["QT_PLUGIN_PATH"] = str(qt/"plugins")
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt/"plugins"/"platforms")


def report_exception(kind, error, trace):
    logging.error("Uncaught error", exc_info=(kind, error, trace))
    # Do not let a failed Qt callback silently disappear.
    from PyQt5.QtWidgets import QApplication, QMessageBox
    if QApplication.instance() is not None:
        QMessageBox.critical(None, "Sensor Viewer error", f"{error}\n\nDetails: {log_folder / 'viewer.log'}")


if __name__ == "__main__":
    try:
        if "--diagnostics" in sys.argv:
            from PyQt5.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
            import numpy
            import serial
            print(f"Python {sys.version}\nQt {QT_VERSION_STR}\nPyQt {PYQT_VERSION_STR}\nNumPy {numpy.__version__}\npyserial {serial.__version__}\nLog: {log_folder / 'viewer.log'}")
            from viewer.app import MainWindow
            print("App import OK. No hardware was opened.")
        else:
            from viewer.app import main
            sys.excepthook = report_exception
            raise SystemExit(main(sys.argv))
    except Exception:
        logging.exception("Startup failed")
        print(traceback.format_exc())
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, f"Sensor Viewer could not start.\n\nSee {log_folder / 'viewer.log'}\n\nExtract the entire ZIP before launching; keep runtime and app beside the EXE.", "PingLab Sensor Viewer", 0x10)
        raise SystemExit(1)

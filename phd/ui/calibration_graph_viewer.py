"""Independent saved-file viewer: no robot, sensor, ROS node or task registry.

The experiment UI launches this process without waiting. File selection, CSV
loading, analysis and plotting therefore never run on its control/timer thread.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox


def launch_graph_analysis(start_directory):
    """Start an independent viewer, with no IPC or access to the live host.

    Use the same Python/package as the caller, including source-tree launches
    that added their package root to sys.path but not PYTHONPATH. Limit numeric
    worker threads in the child to reduce competition with acquisition.
    """
    process = QProcess()
    process.setProgram(sys.executable)
    process.setArguments([
        "-m", "phd.ui.calibration_graph_viewer", "--start-directory", str(start_directory),
    ])
    environment = QProcessEnvironment.systemEnvironment()
    package_root = str(Path(__file__).resolve().parents[2])
    environment.insert("PYTHONPATH", os.pathsep.join(filter(None, (
        package_root, environment.value("PYTHONPATH"),
    ))))
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment.insert(key, "1")
    process.setProcessEnvironment(environment)
    process.setWorkingDirectory(package_root)
    # Detached: closing either UI cannot send a stop/kill to the other process.
    started, _pid = process.startDetached()
    return bool(started)


def open_recording(selected_path):
    """Load only saved measurements; errors are handled inside this process."""
    from phd.ui.calibration_graph_analysis import (
        CalibrationGraphAnalysisDialog, load_calibration_recording,
    )
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        loaded = load_calibration_recording(selected_path, browse_full_run=True)
        dialog = CalibrationGraphAnalysisDialog(loaded)
    finally:
        QApplication.restoreOverrideCursor()
    dialog.setWindowTitle("Graph Analysis — Offline Viewer")
    dialog.status_label.setText(
        "Independent offline viewer · saved files only · no robot or sensor connection. "
        "A file still being recorded may be incomplete; reopen it after the trial finishes."
    )
    return dialog


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-directory", default="")
    parser.add_argument("--recording", help="Optional saved calibration CSV, JSON or graph")
    args = parser.parse_args(argv)
    if hasattr(os, "nice"):
        try:
            os.nice(5)  # Only this offline process; favour the acquisition UI.
        except OSError:
            pass
    app = QApplication([sys.argv[0]])
    app.setApplicationName("PingLab Graph Analysis")
    # The startup file picker is the only window until a recording is loaded.
    app.setQuitOnLastWindowClosed(False)
    from phd.dependence.paths import experiment_path, icon_path, stylesheet_path
    from phd.ui import theme
    try:
        template = Path(stylesheet_path("ui_style.qss")).read_text(encoding="utf-8")
        app.setStyleSheet(theme.render_stylesheet(
            template, ICON_DIR=icon_path("generated").replace(os.sep, "/"),
        ))
    except (OSError, ValueError):
        pass  # The viewer also works without optional theme assets.

    def select_and_open():
        selected = args.recording
        if not selected:
            selected, _filter = QFileDialog.getOpenFileName(
                None, "Graph Analysis — select a saved recording (offline)",
                args.start_directory or experiment_path("calibration_recordings"),
                "Calibration data or graph (*.csv *.json *.png)",
            )
        if not selected:
            app.quit()
            return
        try:
            dialog = open_recording(selected)
        except Exception as exc:
            QMessageBox.warning(None, "Graph Analysis", f"Could not load the saved recording.\n\n{exc}")
            app.exit(1)
            return
        app.analysis_dialog = dialog  # Retain the Python wrapper until close.
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        app.setQuitOnLastWindowClosed(True)

    QTimer.singleShot(0, select_and_open)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

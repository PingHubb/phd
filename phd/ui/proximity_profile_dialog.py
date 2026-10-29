"""Result window for one Calibration proximity-profile trial."""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase, QPixmap
from PyQt5.QtWidgets import QDialog, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout

from phd.ui.proximity_profile import report_lines


class ProximityProfileResultDialog(QDialog):
    """Graph and plain-text report; reads only the saved trial result."""

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Proximity profile result")
        layout = QVBoxLayout(self)
        files = result.get("proximity_profile_files") or {}
        png = files.get("png")
        if png and Path(png).exists():
            graph = QLabel()
            graph.setObjectName("proximityProfileGraph")
            graph.setPixmap(QPixmap(png).scaledToWidth(1100, Qt.SmoothTransformation))
            layout.addWidget(graph)
        report = QPlainTextEdit("\n".join(report_lines(result.get("proximity_profile") or {})))
        report.setObjectName("proximityProfileReport")
        report.setReadOnly(True)
        report.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        report.setMinimumHeight(260)
        layout.addWidget(report)
        if files.get("json"):
            location = QLabel(f"Saved: {Path(files['json']).parent}")
            location.setTextInteractionFlags(Qt.TextSelectableByMouse)
            location.setWordWrap(True)
            layout.addWidget(location)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        layout.addWidget(close, alignment=Qt.AlignRight)
        self.resize(1140, 820)

"""Read-only comparison of independently repeated response-centre maps."""
import json
from pathlib import Path
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QFileDialog, QMessageBox
from phd.ui import components
from phd.ui.experiment_quality import compare_position_maps


def open_mapping_repeatability(parent):
    paths, _ = QFileDialog.getOpenFileNames(
        parent, "Select independently repeated maps from the SAME unchanged setup",
        "", "Position maps (*.json)"
    )
    if not paths:
        return
    try:
        maps = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
        result = compare_position_maps(maps)
        result["sources"] = paths
    except (OSError, ValueError, KeyError, TypeError) as exc:
        QMessageBox.warning(parent, "Cannot compare maps", str(exc))
        return
    dialog = QDialog(parent)
    dialog.setWindowTitle("Mapping Repeatability — not absolute accuracy")
    components.size_to_screen(dialog, 800, 600)
    layout = QVBoxLayout(dialog)
    text = QTextEdit()
    text.setReadOnly(True)
    text.setPlainText(json.dumps(result, indent=2))
    layout.addWidget(text)
    export = QPushButton("Export comparison JSON…")

    def save():
        path, _ = QFileDialog.getSaveFileName(dialog, "Export map repeatability", "mapping_repeatability.json", "JSON (*.json)")
        if path:
            try:
                Path(path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                QMessageBox.warning(dialog, "Export failed", str(exc))

    export.clicked.connect(save)
    layout.addWidget(export)
    dialog.exec_()

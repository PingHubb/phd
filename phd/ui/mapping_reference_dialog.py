"""Independent XY validation, deliberately separate from fast/dense benchmarks."""
import csv
import json
from pathlib import Path
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QFileDialog, QMessageBox

from phd.dependence.characterisation_study import compare_map_reference
from phd.ui import components


def open_mapping_reference(parent):
    map_path, _ = QFileDialog.getOpenFileName(parent, "Position map to validate", "", "Position maps (*.json)")
    if not map_path:
        return
    reference_path, _ = QFileDialog.getOpenFileName(parent, "Independent robot-base references: taxel_index,x_mm,y_mm", "", "CSV (*.csv)")
    if not reference_path:
        return
    confirmation = QMessageBox.question(
        parent, "Independent reference and coordinate frame",
        "Were these XY positions independently measured in the SAME robot-base frame, with the same specimen placement and probe/tool? Do not use predictions or positions copied from the map being tested. No alignment is fitted to minimise the error. Only XY is compared, not proximity-plane Z.",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
    )
    if confirmation != QMessageBox.Yes:
        return
    try:
        result = json.loads(Path(map_path).read_text(encoding="utf-8"))
        with Path(reference_path).open(newline="", encoding="utf-8-sig") as stream:
            references = list(csv.DictReader(stream))
        report = compare_map_reference(result, references, same_frame_confirmed=True)
        report.update(map_source=map_path, reference_source=reference_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        QMessageBox.warning(parent, "Cannot validate map", str(exc))
        return
    dialog = QDialog(parent)
    dialog.setWindowTitle("Mapping: independent XY reference error")
    components.size_to_screen(dialog, 800, 600)
    layout = QVBoxLayout(dialog)
    text = QTextEdit()
    text.setReadOnly(True)
    text.setPlainText(json.dumps(report, indent=2))
    layout.addWidget(text)
    export = QPushButton("Export reference-validation JSON…")
    def save():
        path, _ = QFileDialog.getSaveFileName(dialog, "Export map validation", "mapping_reference_validation.json", "JSON (*.json)")
        if path:
            try:
                Path(path).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
            except OSError as exc:
                QMessageBox.warning(dialog, "Export failed", str(exc))
    export.clicked.connect(save)
    layout.addWidget(export)
    dialog.exec_()

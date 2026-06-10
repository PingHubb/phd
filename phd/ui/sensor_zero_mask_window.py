"""Modal dialog used from the menu bar to choose sensor cells that should
always read 0. The dialog edits the live `MySensor.cell_zero_mask` so users
can see the effect immediately, and the same mask can be saved per-sensor
into ``phd/resource/config/sensor_zero_masks.json``.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class SensorZeroMaskDialog(QDialog):
    """Pick which sensor cells should always be forced to 0."""

    CELL_STYLE = (
        "QToolButton {"
        " border: 1px solid #555;"
        " border-radius: 3px;"
        " color: #dddddd;"
        " background: #2b2b2b;"
        " font-size: 10px;"
        " padding: 0px;"
        "}"
        "QToolButton:checked {"
        " background: #d04545;"
        " color: white;"
        " border: 1px solid #ff7676;"
        " font-weight: 600;"
        "}"
        "QToolButton:hover {"
        " border: 1px solid #4fc3f7;"
        "}"
    )

    def __init__(self, parent=None, *, sensor_functions=None, sensor_key="(unknown)"):
        super().__init__(parent)
        self.setWindowTitle("Sensor Zero Mask")
        self.setMinimumWidth(380)
        self._sensor = sensor_functions
        self._sensor_key = sensor_key

        n_row = int(getattr(sensor_functions, "n_row", 0) or 0)
        n_col = int(getattr(sensor_functions, "n_col", 0) or 0)
        self._n_row = n_row
        self._n_col = n_col

        self._cell_buttons: list[list[QToolButton]] = []
        self._status_label: QLabel | None = None

        self._build_ui()
        self._refresh_from_sensor()

    # --------------------------------------------------------------
    # UI construction
    # --------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        if self._n_row <= 0 or self._n_col <= 0:
            label = QLabel(
                "Build a sensor scene first.\n\n"
                "Steps: Sensor tab → 'Build Scene' → 'Update Sensor', then\n"
                "re-open this window to choose cells that should be forced to 0."
            )
            label.setAlignment(Qt.AlignCenter)
            root.addWidget(label)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.accept)
            root.addWidget(close_btn, alignment=Qt.AlignCenter)
            return

        info = QLabel(
            f"Sensor key: <b>{self._sensor_key}</b> &nbsp;&nbsp;"
            f"Grid: {self._n_row} × {self._n_col}<br>"
            "<span style='color:#bbbbbb'>Click a cell to mark it as always 0. "
            "Click again to release.<br>'Save to File' stores the mask under the "
            "sensor key so it auto-loads next time you build the same sensor.</span>"
        )
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        root.addWidget(info)

        # Grid of toggle buttons (one per cell).
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(2)
        for r in range(self._n_row):
            row_buttons: list[QToolButton] = []
            for c in range(self._n_col):
                btn = QToolButton()
                btn.setCheckable(True)
                btn.setText(f"{r},{c}")
                btn.setMinimumSize(36, 28)
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                btn.setStyleSheet(self.CELL_STYLE)
                btn.toggled.connect(self._on_cell_toggled)
                grid.addWidget(btn, r, c)
                row_buttons.append(btn)
            self._cell_buttons.append(row_buttons)
        root.addWidget(grid_widget, stretch=1)

        # Action buttons row.
        action_row = QHBoxLayout()
        self._btn_clear = QPushButton("Clear All")
        self._btn_clear.clicked.connect(self._on_clear)
        action_row.addWidget(self._btn_clear)

        self._btn_invert = QPushButton("Invert")
        self._btn_invert.clicked.connect(self._on_invert)
        action_row.addWidget(self._btn_invert)

        self._btn_reload = QPushButton("Reload from File")
        self._btn_reload.clicked.connect(self._on_reload)
        action_row.addWidget(self._btn_reload)

        self._btn_save = QPushButton("Save to File")
        self._btn_save.setStyleSheet("QPushButton { font-weight: bold; }")
        self._btn_save.clicked.connect(self._on_save)
        action_row.addWidget(self._btn_save)
        root.addLayout(action_row)

        # Footer: status text + close button.
        footer = QHBoxLayout()
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #4fc3f7;")
        footer.addWidget(self._status_label, stretch=1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    # --------------------------------------------------------------
    # Mask <-> UI sync
    # --------------------------------------------------------------
    def _refresh_from_sensor(self):
        if self._sensor is None or not self._cell_buttons:
            return
        try:
            mask = self._sensor.get_cell_zero_mask()
        except Exception:
            return
        for r, row in enumerate(self._cell_buttons):
            for c, btn in enumerate(row):
                btn.blockSignals(True)
                btn.setChecked(bool(mask[r, c]))
                btn.blockSignals(False)
        self._update_count_label()

    def _current_mask(self) -> np.ndarray:
        mask = np.zeros((self._n_row, self._n_col), dtype=bool)
        for r, row in enumerate(self._cell_buttons):
            for c, btn in enumerate(row):
                mask[r, c] = btn.isChecked()
        return mask

    def _push_mask(self, mask: np.ndarray):
        if self._sensor is None:
            return
        try:
            self._sensor.set_cell_zero_mask(mask)
        except Exception as exc:
            self._set_status(f"Failed to apply: {exc}")
        self._update_count_label()

    def _update_count_label(self):
        if not self._status_label or not self._cell_buttons:
            return
        n_zero = sum(1 for row in self._cell_buttons for btn in row if btn.isChecked())
        self._set_status(f"{n_zero} cell(s) forced to 0")

    def _set_status(self, message: str):
        if self._status_label is not None:
            self._status_label.setText(str(message))

    # --------------------------------------------------------------
    # Slots
    # --------------------------------------------------------------
    def _on_cell_toggled(self, _checked: bool):
        self._push_mask(self._current_mask())

    def _on_clear(self):
        for row in self._cell_buttons:
            for btn in row:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
        self._push_mask(np.zeros((self._n_row, self._n_col), dtype=bool))

    def _on_invert(self):
        for row in self._cell_buttons:
            for btn in row:
                btn.blockSignals(True)
                btn.setChecked(not btn.isChecked())
                btn.blockSignals(False)
        self._push_mask(self._current_mask())

    def _on_reload(self):
        if self._sensor is None:
            return
        try:
            loaded = self._sensor.load_cell_zero_mask_from_disk()
        except Exception as exc:
            QMessageBox.warning(self, "Sensor Zero Mask", f"Reload failed: {exc}")
            return
        if loaded:
            self._refresh_from_sensor()
            self._set_status("Loaded mask from file.")
        else:
            QMessageBox.information(
                self,
                "Sensor Zero Mask",
                f"No saved mask for '{self._sensor_key}'.\n"
                "Adjust the grid and press 'Save to File' to create one.",
            )

    def _on_save(self):
        if self._sensor is None:
            return
        try:
            ok = self._sensor.save_cell_zero_mask_to_disk()
        except Exception as exc:
            QMessageBox.warning(self, "Sensor Zero Mask", f"Save failed: {exc}")
            return
        if ok:
            self._set_status(f"Saved as '{self._sensor_key}'.")
        else:
            QMessageBox.warning(
                self,
                "Sensor Zero Mask",
                "Failed to save mask file. Check the log for details.",
            )

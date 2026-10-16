"""Make multi-choice popups tall enough to show every current choice."""

from __future__ import annotations

from PyQt5.QtWidgets import QApplication, QComboBox


def install_full_choice_popups() -> None:
    """Show every combo-box choice, and scroll the rest only in whole rows.

    Qt sizes a styled popup to the screen edge. That height is not a multiple
    of the row height, so the final choice stays partly hidden no matter how
    far the list is scrolled.
    """
    if getattr(QComboBox.showPopup, "_pinglab_full_popup", False):
        return

    original_show_popup = QComboBox.showPopup

    def show_popup(self):
        count = int(self.count())
        if count > 0:
            self.setMaxVisibleItems(count)
        original_show_popup(self)
        _fit_popup_to_choices(self)

    show_popup._pinglab_full_popup = True
    QComboBox.showPopup = show_popup


def _fit_popup_to_choices(combo: QComboBox) -> None:
    view = combo.view()
    count = int(combo.count())
    if view is None or count <= 0:
        return
    popup = view.window()
    if popup is None or popup is combo.window():
        return

    row_height = int(view.sizeHintForRow(0))
    if row_height < 1:
        return
    frame = int(popup.height()) - int(view.viewport().height())
    if frame < 0:
        frame = 0

    screen = QApplication.screenAt(popup.geometry().center())
    if screen is None:
        screen = QApplication.primaryScreen()
    available_height = (
        int(screen.availableGeometry().height()) - 12
        if screen is not None
        else int(popup.height())
    )
    visible_rows = min(count, max(1, (available_height - frame) // row_height))
    height = visible_rows * row_height + frame
    width = max(int(popup.width()), int(combo.width()))
    popup.resize(width, height)

    if screen is None:
        return
    available = screen.availableGeometry()
    if popup.y() < available.top():
        popup.move(popup.x(), available.top())
    overflow = popup.y() + popup.height() - available.bottom()
    if overflow > 0:
        popup.move(popup.x(), max(available.top(), popup.y() - overflow))

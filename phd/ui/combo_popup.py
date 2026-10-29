"""Make multi-choice popups tall enough to show every current choice."""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, QPoint, Qt
from PyQt5.QtWidgets import QApplication, QComboBox, QWidget


_SCROLLER_CLASS = "QComboBoxPrivateScroller"


def install_full_choice_popups() -> None:
    """Show every combo-box choice that fits, without a covering scroll arrow.

    Qt adds its own up/down arrow inside the popup. The first scroll leaves
    that arrow on top of the last row; only the next scroll removes it.
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
        _hide_popup_scroll_arrows(self)

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
    frame = max(0, int(popup.height()) - int(view.viewport().height()))
    screen = QApplication.screenAt(combo.mapToGlobal(QPoint(0, 0)))
    if screen is None:
        screen = QApplication.primaryScreen()
    available = None if screen is None else screen.availableGeometry()
    anchor = combo.mapToGlobal(QPoint(0, 0))
    combo_top = int(anchor.y())
    combo_bottom = combo_top + int(combo.height())
    if available is None:
        space_below = row_height * count + frame
        space_above = space_below
        limit_top = combo_top
        limit_bottom = combo_bottom + space_below
    else:
        space_below = int(available.bottom()) - combo_bottom
        space_above = combo_top - int(available.top())
        limit_top = int(available.top())
        limit_bottom = int(available.bottom())

    def rows_in(space: int) -> int:
        return max(0, (int(space) - frame) // row_height)

    below = rows_in(space_below)
    above = rows_in(space_above)
    open_below = below >= above
    visible_rows = max(1, min(count, below if open_below else above))
    height = visible_rows * row_height + frame
    width = max(int(popup.width()), int(combo.width()))
    if open_below:
        y = combo_bottom
        if y + height > limit_bottom:
            y = max(limit_top, limit_bottom - height)
    else:
        y = combo_top - height
        if y < limit_top:
            y = limit_top
    popup.resize(width, height)
    popup.move(int(popup.x()), int(y))


def _hide_popup_scroll_arrows(combo: QComboBox) -> None:
    view = combo.view()
    if view is None:
        return
    popup = view.window()
    if popup is None or popup is combo.window():
        return
    view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    if getattr(popup, "_pinglab_arrow_guard", None) is None:
        popup._pinglab_arrow_guard = _PopupArrowGuard(popup, view)
    popup._pinglab_arrow_guard.suppress()


class _PopupArrowGuard(QObject):
    def __init__(self, popup, view):
        super().__init__(popup)
        self._popup = popup
        self._view = view
        self._busy = False
        popup.installEventFilter(self)

    def eventFilter(self, watched, event):
        if (
            watched is self._popup
            and not self._busy
            and event.type() in (QEvent.Resize, QEvent.LayoutRequest, QEvent.Show)
        ):
            self.suppress()
        return False

    def suppress(self):
        if self._busy:
            return
        self._busy = True
        try:
            for _ in range(4):
                if not self._suppress_once():
                    break
        finally:
            self._busy = False

    def _suppress_once(self) -> bool:
        popup = self._popup
        view = self._view
        changed = False
        for child in popup.children():
            if not isinstance(child, QWidget):
                continue
            if child.metaObject().className() != _SCROLLER_CLASS:
                continue
            if child.isVisible() or child.height() != 0:
                child.setFixedHeight(0)
                child.hide()
                changed = True
        row_height = int(view.sizeHintForRow(0))
        hidden = _hidden_tail_pixels(view)
        if row_height < 1 or hidden <= 0:
            return changed
        screen = QApplication.screenAt(popup.geometry().center())
        if screen is None:
            screen = QApplication.primaryScreen()
        room = hidden
        if screen is not None:
            room = int(screen.availableGeometry().bottom()) - (
                int(popup.y()) + int(popup.height())
            )
        if hidden > room:
            return False
        popup.resize(popup.width(), popup.height() + hidden)
        return True


def _hidden_tail_pixels(view) -> int:
    """Pixels needed to reveal rows cut off before the user scrolls."""
    scrollbar = view.verticalScrollBar()
    model = view.model()
    if scrollbar is None or model is None or scrollbar.value() != 0:
        return 0
    viewport_height = int(view.viewport().height())
    hidden = 0
    for row in range(model.rowCount()):
        rect = view.visualRect(model.index(row, 0))
        if rect.isValid() and int(rect.bottom()) > viewport_height:
            hidden = max(hidden, int(rect.bottom()) - viewport_height + 2)
    return hidden

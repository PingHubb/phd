"""Top navigation bar for the main workspace switcher.

The app's top-level areas (Sensor, Robots, AI, ...) sit in a single horizontal
bar above the sub-tabs, so the control panel spends all of its width on
controls rather than on a permanent navigation column.

Rather than replacing ``QTabWidget`` with a hand-rolled stack, this module only
swaps in a custom ``QTabBar``. That matters: ``addTab``, ``setTabEnabled``,
``currentIndex``, ``currentChanged`` and -- critically -- the **tab index
numbers** keep working exactly as before, so the index-dependent logic that
drives the humanoid 3D viewport lifecycle is untouched.

The custom bar exists because Qt cannot draw a recoloured icon beside a label
in a tab; painting it here also lets the selected entry use a soft accent pill
that reads as a level above the segmented sub-tabs below it.

Usage::

    from phd.ui import nav_tabs
    nav_tabs.install(self.tab_widget, {"Sensor": "sensor", "Robots": "robot"})
"""

from __future__ import annotations

from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt5.QtWidgets import QStyle, QStyleOptionTab, QTabBar, QTabWidget

from phd.ui import icons, theme

TAB_HEIGHT = 34
ICON_SIZE = 17

# Horizontal insets used both for painting and for the width calculation:
# gap between entries, inset inside an entry, and icon-to-text gap.
PAD_OUTER = 3
PAD_INNER = 10
PAD_ICON_GAP = 7

# Label width kept when the bar is compressed to its minimum. Enough for a
# few characters plus the ellipsis, so a squeezed entry stays identifiable
# next to its icon rather than collapsing to the icon alone.
MIN_LABEL_WIDTH = 26


class NavigationTabBar(QTabBar):
    """Horizontal tab bar that paints a recoloured icon beside each label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("navBar")
        self._hovered = -1
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setUsesScrollButtons(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.setCursor(Qt.PointingHandCursor)

    # -- geometry ---------------------------------------------------------
    def _tab_width(self, index: int) -> int:
        """Width of one entry, measured with the weight it is painted at.

        The selected entry is drawn semi-bold. Measuring every entry at that
        weight costs a few pixels on the unselected ones and guarantees the
        current one never paints wider than the rect Qt gave it -- the usual
        cause of clipped tab labels.
        """
        bold = QFont(self.font())
        bold.setWeight(QFont.DemiBold)
        text = QFontMetrics(bold).horizontalAdvance(self.tabText(index))
        icon = (ICON_SIZE + PAD_ICON_GAP) if self.tabData(index) else 0
        return PAD_OUTER * 2 + PAD_INNER * 2 + icon + text

    def tabSizeHint(self, index):  # noqa: N802 - Qt override
        return QSize(self._tab_width(index), TAB_HEIGHT)

    def minimumTabSizeHint(self, index):  # noqa: N802 - Qt override
        """Deliberately smaller than :meth:`tabSizeHint`.

        A tab bar's minimum becomes a floor on the whole control panel, and
        the sum of five full-width entries was wide enough to push the 3D
        viewport below its own minimum on a small display. Reporting a
        compressible minimum lets Qt narrow the entries when space is tight;
        :meth:`_paint_tab` already elides the label to whatever it is given.
        """
        glyph = (ICON_SIZE + PAD_ICON_GAP) if self.tabData(index) else 0
        floor = PAD_OUTER * 2 + PAD_INNER * 2 + glyph + MIN_LABEL_WIDTH
        return QSize(min(floor, self._tab_width(index)), TAB_HEIGHT)

    def tabInserted(self, index):  # noqa: N802 - Qt override
        super().tabInserted(index)
        self.updateGeometry()

    def tabRemoved(self, index):  # noqa: N802 - Qt override
        super().tabRemoved(index)
        self.updateGeometry()

    # -- hover tracking ---------------------------------------------------
    def mouseMoveEvent(self, event):
        index = self.tabAt(event.pos())
        if index != self._hovered:
            self._hovered = index
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hovered != -1:
            self._hovered = -1
            self.update()
        super().leaveEvent(event)

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(theme.WINDOW_BG))

        current = self.currentIndex()
        for index in range(self.count()):
            rect = self.tabRect(index)
            if not rect.isValid():
                continue

            option = QStyleOptionTab()
            self.initStyleOption(option, index)
            enabled = bool(option.state & QStyle.State_Enabled)
            selected = index == current
            hovered = index == self._hovered and enabled and not selected

            self._paint_tab(painter, rect, index, enabled, selected, hovered)
        painter.end()

    def _paint_tab(self, painter, rect, index, enabled, selected, hovered):
        body = rect.adjusted(PAD_OUTER, 3, -PAD_OUTER, -3)

        if selected:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.ACCENT_SOFT))
            painter.drawRoundedRect(body, theme.RADIUS_MD, theme.RADIUS_MD)
        elif hovered:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.SURFACE_SUNKEN))
            painter.drawRoundedRect(body, theme.RADIUS_MD, theme.RADIUS_MD)

        if not enabled:
            color = theme.TEXT_DISABLED
        elif selected:
            color = theme.ACCENT
        else:
            color = theme.TEXT_SECONDARY

        text_left = body.left() + PAD_INNER
        glyph = str(self.tabData(index) or "")
        if glyph:
            pixmap = icons.pixmap(
                glyph,
                color=color,
                size=ICON_SIZE,
                ratio=float(self.devicePixelRatioF() or 1.0),
            )
            icon_rect = QRect(
                body.left() + PAD_INNER,
                body.top() + (body.height() - ICON_SIZE) // 2,
                ICON_SIZE,
                ICON_SIZE,
            )
            painter.drawPixmap(icon_rect, pixmap)
            text_left = icon_rect.right() + PAD_ICON_GAP

        font = QFont(painter.font())
        font.setWeight(QFont.DemiBold if selected else QFont.Normal)
        painter.setFont(font)
        painter.setPen(QColor(color))

        text_rect = QRect(
            text_left,
            body.top(),
            max(0, body.right() - text_left - PAD_INNER + PAD_OUTER),
            body.height(),
        )
        label = painter.fontMetrics().elidedText(
            self.tabText(index), Qt.ElideRight, text_rect.width()
        )
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, label)


def install(
    tab_widget: QTabWidget,
    glyphs: dict[str, str] | None = None,
) -> NavigationTabBar:
    """Convert ``tab_widget`` into the top-level navigation bar.

    ``glyphs`` maps tab text to an icon name from :mod:`phd.ui.icons`. It is
    matched case-insensitively so renaming a tab's capitalisation does not
    silently drop its icon.
    """
    bar = NavigationTabBar(tab_widget)
    tab_widget.setObjectName("navTabs")
    tab_widget.setTabBar(bar)
    tab_widget.setTabPosition(QTabWidget.North)
    tab_widget.setDocumentMode(True)
    tab_widget.setElideMode(Qt.ElideNone)
    set_glyphs(tab_widget, glyphs or {})
    return bar


def set_glyphs(tab_widget: QTabWidget, glyphs: dict[str, str]) -> None:
    """Attach icon names to tabs by label, matched case-insensitively.

    Tab data lives on the ``QTabBar``, not on the ``QTabWidget``.
    """
    bar = tab_widget.tabBar()
    if bar is None:
        return
    lookup = {key.strip().lower(): value for key, value in glyphs.items()}
    for index in range(bar.count()):
        label = bar.tabText(index).strip()
        glyph = lookup.get(label.lower())
        if glyph:
            bar.setTabData(index, glyph)
        if not bar.tabToolTip(index):
            bar.setTabToolTip(index, label)
    bar.updateGeometry()

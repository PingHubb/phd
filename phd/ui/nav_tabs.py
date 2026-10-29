"""Top navigation bar for the main workspace switcher.

The app's top-level areas (Sensor, Robots, AI, ...) sit in a single horizontal
bar outside the adaptive split, so navigation never moves with the controls.

Rather than replacing ``QTabWidget`` with a hand-rolled stack, this module only
uses a custom ``QTabBar`` and synchronizes an external navigator with it.
That matters: ``addTab``, ``setTabEnabled``,
``currentIndex``, ``currentChanged`` and -- critically -- the **tab index
numbers** keep working exactly as before, so the index-dependent logic that
drives the humanoid 3D viewport lifecycle is untouched.

The custom bar exists because Qt cannot draw a recoloured icon beside a label
in a tab; painting it here also lets the selected entry use a soft accent pill
that reads as a level above the underlined sub-tabs below it.

Usage::

    from phd.ui import nav_tabs
    nav_tabs.install(self.tab_widget, {"Sensor": "sensor", "Robots": "robot"})
"""

from __future__ import annotations

from PyQt5.QtCore import QRect, QRectF, QSize, Qt, QSignalBlocker, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QStyle, QStyleOptionTab, QTabBar, QTabWidget

from phd.ui import icons, motion, theme

TAB_HEIGHT = 40
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
        self._indicator = QRectF()
        self._indicator_target = QRectF()
        self._indicator_motion = motion.animation(self, motion.NAV_MS, self._set_indicator)
        self.currentChanged.connect(self.sync_indicator)
        motion.preferences().changed.connect(self._snap_indicator)

    def _selection_rect(self):
        index = self.currentIndex()
        if index < 0 or not self.isTabVisible(index):
            return QRectF()
        return QRectF(self.tabRect(index).adjusted(PAD_OUTER, 3, -PAD_OUTER, -3))

    def _set_indicator(self, rect):
        self._indicator = QRectF(rect)
        self.update()

    def _snap_indicator(self, *_args):
        self._indicator_motion.stop()
        self._indicator_target = self._selection_rect()
        self._set_indicator(self._indicator_target)

    def sync_indicator(self, *_args):
        """Selection is already committed; this only retargets its paint."""
        target = self._selection_rect()
        if target == self._indicator_target:
            return
        self._indicator_motion.stop()
        self._indicator_target = target
        if (motion.preferences().reduced or not self.isVisible() or
                not self._indicator.isValid() or not target.isValid()):
            self._set_indicator(target)
            return
        self._indicator_motion.setStartValue(self._indicator)
        self._indicator_motion.setEndValue(target)
        self._indicator_motion.start()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._snap_indicator()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._snap_indicator()

    def hideEvent(self, event):  # noqa: N802
        self._snap_indicator()
        super().hideEvent(event)

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
        painter.fillRect(self.rect(), QColor(theme.SURFACE))

        # Label/visibility changes can alter tab geometry without a resize.
        if self._selection_rect() != self._indicator_target:
            self._snap_indicator()
        if 0 <= self._hovered < self.count() and self.isTabEnabled(self._hovered):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.SURFACE_SUNKEN))
            painter.drawRoundedRect(
                self.tabRect(self._hovered).adjusted(PAD_OUTER, 3, -PAD_OUTER, -3),
                theme.RADIUS_MD, theme.RADIUS_MD,
            )
        indicator_path = QPainterPath()
        indicator_path.addRoundedRect(self._indicator, theme.RADIUS_MD, theme.RADIUS_MD)
        painter.fillPath(indicator_path, QColor(theme.ACCENT))

        current = self.currentIndex()
        for index in range(self.count()):
            rect = self.tabRect(index)
            if not rect.isValid() or not self.isTabVisible(index):
                continue

            option = QStyleOptionTab()
            self.initStyleOption(option, index)
            enabled = bool(option.state & QStyle.State_Enabled)
            selected = index == current
            self._paint_tab(painter, rect, index, enabled, selected, on_accent=False)
            # White glyphs only where the moving pill actually covers them.
            # This avoids unreadable white text at the destination mid-flight.
            if self._indicator.intersects(QRectF(rect)):
                painter.save()
                painter.setClipPath(indicator_path)
                self._paint_tab(painter, rect, index, enabled, selected, on_accent=True)
                painter.restore()
        painter.end()

    def _paint_tab(self, painter, rect, index, enabled, selected, on_accent):
        body = rect.adjusted(PAD_OUTER, 3, -PAD_OUTER, -3)
        compact = self.width() < sum(self._tab_width(i) for i in range(self.count()))
        inset = 6 if compact else PAD_INNER

        if not enabled:
            color = theme.TEXT_DISABLED
        elif on_accent:
            color = theme.TEXT_ON_ACCENT
        else:
            color = theme.TEXT_SECONDARY

        # On a narrow panel, readable workspace names are more useful than
        # retaining every icon and reducing each label to one character.
        text_left = body.left() + inset
        glyph = "" if compact else str(self.tabData(index) or "")
        if glyph:
            pixmap = icons.pixmap(
                glyph,
                color=color,
                size=ICON_SIZE,
                ratio=float(self.devicePixelRatioF() or 1.0),
            )
            icon_rect = QRect(
                body.left() + inset,
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
            max(0, body.right() - text_left - inset + PAD_OUTER),
            body.height(),
        )
        label = painter.fontMetrics().elidedText(
            self.tabText(index), Qt.ElideRight, text_rect.width()
        )
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, label)
        if selected and self.hasFocus():
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(theme.TEXT_ON_ACCENT if on_accent else theme.ACCENT), 1, Qt.DotLine))
            painter.drawRoundedRect(body.adjusted(3, 3, -3, -3), 5, 5)


class WorkspaceTabs(QTabWidget):
    """Keep page ownership in QTabWidget, with a synchronized external navigator.

    The native tab indices, signals and enablement API remain authoritative.
    No page or renderer is moved when the external navigation is attached.
    """

    navigation_activated = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.external_bar = None
        self.currentChanged.connect(self._sync_navigation)

    def create_external_navigation(self):
        if self.external_bar is None:
            self.external_bar = NavigationTabBar()
            self.external_bar.currentChanged.connect(self._activate_navigation)
            self.external_bar.tabBarClicked.connect(self._activate_navigation)
        self._sync_navigation()
        self.tabBar().hide()
        return self.external_bar

    def _activate_navigation(self, index):
        if 0 <= index < self.count() and self.isTabEnabled(index):
            self.setCurrentIndex(index)
            self.navigation_activated.emit(index)

    def _sync_navigation(self, *_args):
        bar = self.external_bar
        if bar is None:
            return
        blocker = QSignalBlocker(bar)
        while bar.count() > self.count():
            bar.removeTab(bar.count() - 1)
        for index in range(self.count()):
            if index >= bar.count():
                bar.addTab(self.tabText(index))
            # Runtime readiness checks call this repeatedly. Reassigning the
            # same text invalidates QTabBar geometry and schedules needless
            # paints during a workspace transition.
            if bar.tabText(index) != self.tabText(index):
                bar.setTabText(index, self.tabText(index))
            if bar.tabData(index) != self.tabBar().tabData(index):
                bar.setTabData(index, self.tabBar().tabData(index))
            if bar.tabToolTip(index) != self.tabToolTip(index):
                bar.setTabToolTip(index, self.tabToolTip(index))
            if bar.isTabEnabled(index) != self.isTabEnabled(index):
                bar.setTabEnabled(index, self.isTabEnabled(index))
            if bar.isTabVisible(index) != self.isTabVisible(index):
                bar.setTabVisible(index, self.isTabVisible(index))
        if bar.currentIndex() != self.currentIndex():
            bar.setCurrentIndex(self.currentIndex())
        del blocker
        bar.sync_indicator()

    def tabInserted(self, index):  # noqa: N802
        super().tabInserted(index)
        self._sync_navigation()

    def tabRemoved(self, index):  # noqa: N802
        super().tabRemoved(index)
        self._sync_navigation()

    def setTabEnabled(self, index, enabled):  # noqa: N802
        super().setTabEnabled(index, enabled)
        self._sync_navigation()

    def setTabVisible(self, index, visible):  # noqa: N802
        super().setTabVisible(index, visible)
        self._sync_navigation()

    def setTabText(self, index, text):  # noqa: N802
        super().setTabText(index, text)
        self._sync_navigation()

    def setTabToolTip(self, index, text):  # noqa: N802
        super().setTabToolTip(index, text)
        self._sync_navigation()


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

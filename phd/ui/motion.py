"""Small, interruptible presentation effects; never schedule device actions.

Only opted-in controls use these helpers. No graphics effects, screenshots or
animation of the workspace splitter are involved, so native/VTK rendering and
button signals keep their normal lifecycle.
"""

from PyQt5.QtCore import (
    QEasingCurve, QEvent, QObject, QSettings, Qt, QTimer, QVariantAnimation,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QApplication, QWidget

from phd.ui import theme

PAGE_MS = 140
NAV_MS = 140
HOVER_MS = 100
DISCLOSURE_MS = 150


class MotionPreferences(QObject):
    changed = pyqtSignal(bool)

    def __init__(self, parent):
        super().__init__(parent)
        self.settings = QSettings("PingLab", "PingLab")
        self.reduced = self.settings.value("appearance/reduce_motion", False, type=bool)

    def set_reduced(self, reduced):
        reduced = bool(reduced)
        if self.reduced == reduced:
            return
        self.reduced = reduced
        self.settings.setValue("appearance/reduce_motion", reduced)
        self.changed.emit(reduced)


def preferences():
    app = QApplication.instance()
    if not hasattr(app, "_pinglab_motion"):
        app._pinglab_motion = MotionPreferences(app)
    return app._pinglab_motion


def animation(parent, duration, update):
    tween = QVariantAnimation(parent)
    tween.setDuration(duration)
    tween.setEasingCurve(QEasingCurve.OutCubic)
    tween.valueChanged.connect(update)
    return tween


class _PaintLayer(QWidget):
    """Input-transparent paint only: never participates in a layout."""

    def __init__(self, parent, *, border=False):
        super().__init__(parent)
        self.border = border
        self.amount = 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.NoFocus)
        self.hide()

    def set_amount(self, amount):
        self.amount = float(amount)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        color = QColor(theme.ACCENT if self.border else theme.SURFACE)
        color.setAlphaF(max(0.0, min(1.0, self.amount)))
        if self.border:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(color)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2),
                                    theme.RADIUS_MD - 1, theme.RADIUS_MD - 1)
        else:
            painter.fillRect(self.rect(), color)


class PageReveal(QObject):
    """A light 140 ms reveal on ordinary task controls, not a frozen page.

    At least 88% of the real controls remain visible from the first frame.
    Unlike an opacity effect this does not force a live renderer offscreen.
    The overlay receives no input and is removed as soon as the tween ends.
    """

    def __init__(self, area):
        super().__init__(area)
        self.area = area
        self.viewport = area.viewport()
        self.layer = _PaintLayer(self.viewport)
        self.tween = animation(self, PAGE_MS, self.layer.set_amount)
        self.tween.finished.connect(self.layer.hide)
        self.pending = QTimer(self)
        self.pending.setSingleShot(True)
        self.pending.timeout.connect(self.start)
        area.installEventFilter(self)
        self.viewport.installEventFilter(self)
        preferences().changed.connect(self.finish)

    def start(self):
        self.finish()
        if preferences().reduced or not self.area.isVisible():
            return
        # Defensive exclusion for future task pages containing native views.
        for child in self.area.findChildren(QWidget):
            name = child.metaObject().className().lower()
            if (child.property("motionExclude") or child.property("variant") == "danger" or "vtk" in name or
                    "opengl" in name or child.testAttribute(Qt.WA_NativeWindow)):
                return
        self.layer.setGeometry(self.viewport.rect())
        self.layer.set_amount(0.12)
        self.layer.show()
        self.layer.raise_()
        self.tween.setStartValue(0.12)
        self.tween.setEndValue(0.0)
        self.tween.start()

    def finish(self, *_args):
        self.pending.stop()
        self.tween.stop()
        self.layer.hide()

    def eventFilter(self, watched, event):  # noqa: N802
        kind = event.type()
        if watched is self.area and kind == QEvent.Show:
            self.pending.start(0)
        elif watched is self.area and kind == QEvent.Hide:
            self.finish()
        elif kind == QEvent.Resize and watched is self.viewport:
            self.layer.setGeometry(self.viewport.rect())
        return False


class HoverFeedback(QObject):
    """Ease in a slim accent outline; native press/focus semantics are intact."""

    def __init__(self, button):
        super().__init__(button)
        self.button = button
        self.layer = _PaintLayer(button, border=True)
        self.tween = animation(self, HOVER_MS, self.layer.set_amount)
        self.tween.finished.connect(self._finished)
        button.installEventFilter(self)
        button.pressed.connect(self.clear)
        preferences().changed.connect(self.clear)

    def clear(self, *_args):
        self.tween.stop()
        self.layer.set_amount(0.0)
        self.layer.hide()

    def _finished(self):
        if self.layer.amount == 0:
            self.layer.hide()

    def eventFilter(self, watched, event):  # noqa: N802
        kind = event.type()
        if kind in (QEvent.Hide, QEvent.EnabledChange, QEvent.DynamicPropertyChange):
            self.clear()
        elif kind == QEvent.Resize:
            self.layer.setGeometry(self.button.rect())
        elif kind in (QEvent.Enter, QEvent.Leave):
            if (preferences().reduced or not self.button.isEnabled() or
                    self.button.isDown() or self.button.property("variant") == "danger"):
                self.clear()
                return False
            self.tween.stop()
            self.layer.setGeometry(self.button.rect())
            self.layer.show()
            self.layer.raise_()
            self.tween.setStartValue(self.layer.amount)
            self.tween.setEndValue(0.65 if kind == QEvent.Enter else 0.0)
            self.tween.start()
        return False


def add_hover(button):
    if not hasattr(button, "_hover_feedback"):
        button._hover_feedback = HoverFeedback(button)


class DisclosureMotion(QObject):
    """Animate only a disclosure's maximum height, restoring normal sizing.

    QWidget's default size hint ignores maximumHeight; the content subclass
    clamps its hints during the tween so surrounding scroll pages shrink too.
    """

    def __init__(self, group):
        super().__init__(group)
        self.group = group
        self.content = group.content
        self.tween = animation(self, DISCLOSURE_MS, self._height)
        self.tween.finished.connect(self.finish)
        group.installEventFilter(self)
        preferences().changed.connect(self.finish)

    def _height(self, value):
        self.content.setMaximumHeight(int(value))
        self.content.updateGeometry()

    def set_expanded(self, expanded):
        was_visible = self.content.isVisible()
        start = self.content.height() if was_visible else 0
        self.tween.stop()
        if preferences().reduced or not self.group.isVisible():
            self.finish()
            return
        if not expanded:
            focus = QApplication.focusWidget()
            if focus is not None and self.content.isAncestorOf(focus):
                self.group.toggle.setFocus(Qt.OtherFocusReason)
        self.content._revealing = True
        self.content.setVisible(True)
        self.content.setAttribute(Qt.WA_TransparentForMouseEvents, not expanded)
        self._height(start)
        end = self.content.layout().totalSizeHint().height() if expanded else 0
        self.tween.setStartValue(start)
        self.tween.setEndValue(end)
        self.tween.start()

    def finish(self, *_args):
        self.tween.stop()
        self.content._revealing = False
        self.content.setVisible(self.group.toggle.isChecked())
        self.content.setMaximumHeight(16777215)
        self.content.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.content.updateGeometry()

    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() == QEvent.Hide:
            self.finish()
        return False


class DisclosureBody(QWidget):
    _revealing = False

    def sizeHint(self):  # noqa: N802
        hint = super().sizeHint()
        if self._revealing:
            hint.setHeight(min(hint.height(), self.maximumHeight()))
        return hint

    def minimumSizeHint(self):  # noqa: N802
        hint = super().minimumSizeHint()
        if self._revealing:
            hint.setHeight(min(hint.height(), self.maximumHeight()))
        return hint

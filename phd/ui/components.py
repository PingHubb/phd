"""Reusable presentation widgets for the PingLab UI.

These are *visual* primitives only -- they hold no sensor, robot, AI or
experiment logic. Feature code builds one of these, drops its own controls
inside, and connects its own signals, which keeps the layout language
consistent without coupling the look to any subsystem.

The vocabulary:

``SectionHeader``
    Title (+ optional subtitle and right-aligned actions) above a block.
``CardGroup``
    The card primitive -- a ``QGroupBox`` with the house margins.
``CollapsibleGroup``
    A card that hides advanced parameters behind a disclosure arrow.
``ParameterGrid``
    Aligned label / control / hint rows so every settings block lines up.
``StatusPill`` / ``StatusDot``
    Named-state badges (idle, active, warning, error, info).
``IconButton`` / ``Toolbar``
    Borderless icon actions with mandatory tooltips.
``SegmentedControl``
    Mutually exclusive choices rendered as one control.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from phd.ui import icons, theme


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# Fraction of the screen a dialog may occupy before it is clamped. Leaves room
# for the panel/taskbar and for the main window to stay recognisable behind it.
DIALOG_SCREEN_FRACTION = 0.9

# Floor for a clamped dialog. Below this nothing inside is usable anyway, so a
# very small screen gets a dialog that overflows rather than one that is blank.
MIN_DIALOG_WIDTH = 320
MIN_DIALOG_HEIGHT = 240


def clamp_to_available(
    preferred_width: int,
    preferred_height: int,
    available: tuple[int, int] | None,
    *,
    fraction: float = DIALOG_SCREEN_FRACTION,
) -> tuple[int, int]:
    """Fit a preferred dialog size into ``available`` screen space.

    Split out from :func:`size_to_screen` so the arithmetic can be exercised
    without a live QApplication. ``available`` of ``None`` means "screen
    unknown", in which case the preferred size is honoured.
    """
    width, height = int(preferred_width), int(preferred_height)
    if available is not None:
        avail_w, avail_h = available
        width = min(width, int(avail_w * fraction))
        height = min(height, int(avail_h * fraction))
    return max(width, MIN_DIALOG_WIDTH), max(height, MIN_DIALOG_HEIGHT)


def size_to_screen(
    dialog: QWidget,
    preferred_width: int,
    preferred_height: int,
    *,
    fraction: float = DIALOG_SCREEN_FRACTION,
) -> QWidget:
    """Resize ``dialog`` to its preferred size, capped to the actual screen.

    Several analysis and viewer dialogs ask for sizes up to 1400x1040, which
    does not fit a 1366x768 laptop: the window opens with its buttons
    off-screen and no way to reach them. Asking for the preferred size but
    clamping to the screen keeps the roomy layout on a large display without
    stranding controls on a small one.
    """
    dialog.resize(
        *clamp_to_available(
            preferred_width,
            preferred_height,
            _available_screen_size(dialog),
            fraction=fraction,
        )
    )
    return dialog


def _available_screen_size(dialog: QWidget) -> tuple[int, int] | None:
    """Usable geometry of the screen ``dialog`` will appear on, if knowable."""
    app = QApplication.instance()
    if app is None:
        return None
    handle = dialog.windowHandle()
    screen = handle.screen() if handle is not None else None
    if screen is None:
        screen = app.screenAt(dialog.pos()) or app.primaryScreen()
    if screen is None:
        return None
    available = screen.availableGeometry()
    return available.width(), available.height()


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def apply_variant(widget: QWidget, variant: str | None) -> QWidget:
    """Tag ``widget`` with a QSS ``variant`` (primary/success/warning/danger/ghost).

    Qt does not re-evaluate property selectors on its own, so the widget is
    unpolished and repolished to force the stylesheet to reapply.
    """
    widget.setProperty("variant", variant or "")
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()
    return widget


def muted(label: QLabel) -> QLabel:
    """Render ``label`` as secondary/helper text."""
    label.setStyleSheet(theme.MUTED_LABEL_STYLE)
    return label


class ElidedLabel(QLabel):
    """Label that shows a trailing ellipsis instead of being cut off.

    For values whose length is not under our control -- checkpoint filenames,
    device paths -- so a long value gives up width gracefully rather than
    pushing the controls next to it out of the panel. ``text()`` still returns
    the full string, and the full value is mirrored into the tooltip.
    """

    def __init__(self, text: str = "", parent=None, *, mode=Qt.ElideRight):
        super().__init__(text, parent)
        self._elide_mode = mode
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def setText(self, text):  # noqa: N802 - Qt override
        super().setText(text)
        if not self.toolTip():
            self.setToolTip(text)

    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        metrics = self.fontMetrics()
        elided = metrics.elidedText(self.text(), self._elide_mode, self.width())
        painter.drawText(self.rect(), int(self.alignment()), elided)


def divider(orientation: Qt.Orientation = Qt.Horizontal) -> QFrame:
    """A one-pixel hairline separator."""
    line = QFrame()
    line.setFrameShape(
        QFrame.HLine if orientation == Qt.Horizontal else QFrame.VLine
    )
    line.setFrameShadow(QFrame.Plain)
    line.setStyleSheet(f"background: {theme.BORDER_SUBTLE}; border: none;")
    if orientation == Qt.Horizontal:
        line.setFixedHeight(1)
    else:
        line.setFixedWidth(1)
    return line


def segmented(tabs: QTabWidget) -> QTabWidget:
    """Restyle an existing ``QTabWidget`` as a segmented control.

    Used on the sub-tab bars so related tools read as one control instead of a
    second row of page tabs competing with the main navigation.
    """
    tabs.setObjectName("segmentedTabs")
    tabs.setDocumentMode(True)
    # Segments size to their label; eliding them would produce "Data Trainin…"
    # in a bar that has plenty of room.
    tabs.setElideMode(Qt.ElideNone)
    bar = tabs.tabBar()
    if bar is not None:
        # Named on the bar itself so the QSS padding also feeds tabSizeHint.
        bar.setObjectName("segmentedBar")
        bar.setDrawBase(False)
        bar.setExpanding(False)
        bar.setUsesScrollButtons(False)
        bar.setElideMode(Qt.ElideNone)
        bar.setCursor(Qt.PointingHandCursor)
    return tabs


# ---------------------------------------------------------------------------
# Headers and containers
# ---------------------------------------------------------------------------

class SectionHeader(QWidget):
    """Title, optional subtitle, and optional right-aligned action widgets."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_SM)

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(1)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionHeaderTitle")
        text_column.addWidget(self.title_label)

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("sectionHeaderSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        text_column.addWidget(self.subtitle_label)

        layout.addLayout(text_column, 1)
        self._layout = layout

    def set_subtitle(self, text: str) -> None:
        self.subtitle_label.setText(text)
        self.subtitle_label.setVisible(bool(text))

    def add_action(self, widget: QWidget) -> QWidget:
        """Place ``widget`` at the trailing edge of the header."""
        self._layout.addWidget(widget, 0, Qt.AlignRight | Qt.AlignVCenter)
        return widget


class CardGroup(QGroupBox):
    """A titled card with the house margins and spacing."""

    def __init__(self, title: str = "", parent=None, *, danger: bool = False):
        super().__init__(title, parent)
        if danger:
            self.setObjectName("dangerZone")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(
            theme.SPACE_MD, theme.SPACE_SM, theme.SPACE_MD, theme.SPACE_MD
        )
        self.body.setSpacing(theme.SPACE_SM)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.body.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout):
        self.body.addLayout(layout)
        return layout


class SettingsRow(QWidget):
    """One row of an :class:`InsetList`: ``label [+ hint] | control``.

    The label column is left-aligned and the control trailing, which is what
    makes a column of rows scannable -- you read down the labels, and the
    controls line up on the right regardless of their individual widths.
    """

    def __init__(
        self,
        label: str = "",
        control: QWidget | None = None,
        *,
        hint: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("settingsRow")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(
            theme.INSET_ROW_PADDING_H,
            theme.INSET_ROW_PADDING_V,
            theme.INSET_ROW_PADDING_H,
            theme.INSET_ROW_PADDING_V,
        )
        layout.setSpacing(theme.SPACE_MD)
        self.setMinimumHeight(theme.INSET_ROW_MIN_HEIGHT)

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(0)

        self.label = QLabel(label)
        self.label.setObjectName("settingsRowLabel")
        self.label.setVisible(bool(label))
        # Wraps so a narrow panel costs the row a second line instead of
        # forcing the whole page to scroll sideways.
        self.label.setWordWrap(True)
        text_column.addWidget(self.label)

        self.hint = QLabel(hint)
        self.hint.setObjectName("settingsRowHint")
        self.hint.setWordWrap(True)
        self.hint.setVisible(bool(hint))
        text_column.addWidget(self.hint)

        layout.addLayout(text_column, 1)
        self._layout = layout

        self.control = control
        if control is not None:
            layout.addWidget(control, 0, Qt.AlignRight | Qt.AlignVCenter)

    def set_hint(self, text: str) -> None:
        self.hint.setText(text)
        self.hint.setVisible(bool(text))


class InsetList(QWidget):
    """A rounded surface whose children are separated by hairlines.

    The macOS System Settings pattern. One of these replaces a run of
    individually bordered group boxes: the section keeps a single outline, and
    the rows inside are divided by 1 px rules instead of by more borders. The
    section *title* belongs outside and above, in a :class:`SectionHeader`.

    Rows can be a :class:`SettingsRow` or any widget -- a button, a grid of
    buttons, a chart -- so a section does not have to be uniform to read as
    one block.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("insetList")
        # A plain QWidget ignores a stylesheet background and border unless it
        # is told to draw a styled background, so without this the rounded
        # surface simply does not appear.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._rows = []

    def add(self, widget: QWidget) -> QWidget:
        """Append ``widget`` as a row, inserting a separator above it."""
        if not isinstance(widget, SettingsRow):
            widget = self._pad(widget)
        return self._append(widget)

    def _append(self, row: QWidget) -> QWidget:
        """Append an already-padded row, preceded by a separator."""
        if self._rows:
            separator = divider()
            separator.setObjectName("insetSeparator")
            # divider() applies its own inline sheet, which would outrank the
            # #insetSeparator rule; clear it so the QSS colour applies.
            separator.setStyleSheet("")
            self._layout.addWidget(separator)
        self._layout.addWidget(row)
        self._rows.append(row)
        return row

    def add_row(self, label: str, control: QWidget, hint: str = "") -> QWidget:
        """Convenience for the common ``label | control`` row."""
        self.add(SettingsRow(label, control, hint=hint))
        return control

    def add_stacked(
        self, label: str, control: QWidget, hint: str = ""
    ) -> QWidget:
        """A row with the control on its own line, under the label.

        For controls that cannot compress -- a strip of five or six buttons,
        or two label/combo pairs. Beside a label these set a minimum width
        wider than the control panel gets on a small display, which forces the
        whole page to scroll sideways; given the full row they simply fit.
        """
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(
            theme.INSET_ROW_PADDING_H,
            theme.INSET_ROW_PADDING_V,
            theme.INSET_ROW_PADDING_H,
            theme.INSET_ROW_PADDING_V,
        )
        column.setSpacing(theme.SPACE_XS)

        caption = QLabel(label)
        caption.setObjectName("settingsRowLabel")
        caption.setWordWrap(True)
        column.addWidget(caption)
        if hint:
            note = QLabel(hint)
            note.setObjectName("settingsRowHint")
            note.setWordWrap(True)
            column.addWidget(note)
        column.addWidget(control)

        self._append(holder)
        return control

    @staticmethod
    def _pad(widget: QWidget) -> QWidget:
        """Wrap a bare widget so it gets the same insets as a settings row."""
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(
            theme.INSET_ROW_PADDING_H,
            theme.INSET_ROW_PADDING_V,
            theme.INSET_ROW_PADDING_H,
            theme.INSET_ROW_PADDING_V,
        )
        layout.setSpacing(theme.SPACE_SM)
        layout.addWidget(widget)
        return holder


class EmptyState(QWidget):
    """Centred glyph, one line, and an optional action.

    For the states a research rig spends real time in -- nothing plugged in,
    driver missing, no data yet. An empty area with no explanation reads as a
    bug; this says which one it is and what to do about it.
    """

    def __init__(
        self,
        message: str,
        *,
        glyph: str = "plug",
        hint: str = "",
        action: QWidget | None = None,
        parent=None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_LG, theme.SPACE_LG
        )
        layout.setSpacing(theme.SPACE_XS)
        layout.addStretch(1)

        icon = QLabel()
        icon.setPixmap(icons.pixmap(glyph, color=theme.TEXT_DISABLED, size=28))
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        self.message = QLabel(message)
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setWordWrap(True)
        self.message.setStyleSheet(theme.type_style("callout", color=theme.TEXT_MUTED))
        layout.addWidget(self.message)

        self.hint = QLabel(hint)
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(theme.type_style("caption"))
        self.hint.setVisible(bool(hint))
        layout.addWidget(self.hint)

        if action is not None:
            row = QHBoxLayout()
            row.setContentsMargins(0, theme.SPACE_SM, 0, 0)
            row.addStretch(1)
            row.addWidget(action)
            row.addStretch(1)
            layout.addLayout(row)

        layout.addStretch(1)

    def set_message(self, text: str) -> None:
        self.message.setText(text)


class EmptyStateStack(QStackedWidget):
    """Shows ``content``, or an :class:`EmptyState` while it has nothing to show.

    Keeps the decision in one place, so callers only say *when* they are empty.
    """

    def __init__(self, content: QWidget, empty: EmptyState, parent=None):
        super().__init__(parent)
        self.content = content
        self.empty = empty
        self.addWidget(empty)
        self.addWidget(content)

    def set_empty(self, is_empty: bool) -> None:
        self.setCurrentWidget(self.empty if is_empty else self.content)


def is_visible_to_parent(widget: QWidget | None) -> bool:
    """Whether ``widget`` is shown, ignoring whether its ancestors are.

    ``isVisible()`` is False for anything inside a hidden page, so it cannot
    answer "is this panel open?" for a widget that lives inside a workspace
    the user has navigated away from. ``isVisibleTo`` asks only about the
    widget's own visibility flag.
    """
    if widget is None:
        return False
    parent = widget.parentWidget()
    return widget.isVisibleTo(parent) if parent is not None else widget.isVisible()


def list_row_height(view: QWidget) -> int:
    """Height of one row, valid before the stylesheet has been applied.

    ``sizeHintForRow`` reports ~16 px while a page is being built and ~27 px
    once the stylesheet lands, so sizing a list from it at construction time
    clips the list. Font metrics are correct from the start, and the item
    padding is a token shared with the QSS, so this agrees with what Qt will
    eventually paint.
    """
    return view.fontMetrics().height() + 2 * theme.LIST_ITEM_PADDING_V


def fit_list_height(view: QWidget, *, min_rows: int = 1, max_rows: int = 6) -> QWidget:
    """Size a list to its rows, scrolling past ``max_rows``.

    A QListWidget otherwise reserves a fixed 192 px box whatever it holds,
    which leaves a slab of dead space under a five-item list and makes the
    section look unfinished.
    """
    rows = max(min_rows, min(view.count() or min_rows, max_rows))
    # A couple of pixels of slack: too tall is invisible, too short clips.
    view.setFixedHeight(rows * list_row_height(view) + 2 * view.frameWidth() + 6)
    return view


def fit_table_height(table: QWidget, rows: int) -> QWidget:
    """Size a table to ``rows`` data rows plus its header.

    Same reasoning as :func:`fit_list_height`: derived from font metrics so it
    is correct before the stylesheet is applied, which replaces the magic
    pixel heights these tables used to carry.
    """
    row_height = list_row_height(table)
    header = table.horizontalHeader()
    header_height = header.height() if header.isVisible() else 0
    table.setFixedHeight(
        rows * row_height + max(header_height, row_height) + 6
    )
    return table


def section(title: str, subtitle: str = "") -> tuple[QWidget, InsetList]:
    """A titled section: header above, one inset list below.

    Returns ``(container, list)`` -- add rows to the list, add the container to
    the page. This is the default way to build a page; reach for
    :class:`CardGroup` only when a block genuinely needs its own titled border.
    """
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.SPACE_XS)
    layout.addWidget(SectionHeader(title, subtitle))
    body = InsetList()
    layout.addWidget(body)
    return container, body


class CollapsibleGroup(QWidget):
    """Disclosure container for advanced parameters.

    All sections start open so each tab exposes its complete set of controls.
    The disclosure arrow still lets an operator collapse sections they do not
    need during a session.
    """

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, parent=None, *, expanded: bool = True):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_XS)

        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle.setStyleSheet(
            "QToolButton { border: none; background: transparent;"
            f" color: {theme.TEXT_SECONDARY}; font-weight: 600;"
            f" font-size: {theme.FONT_SIZE_SMALL}; padding: 3px 2px; text-align: left; }}"
            f"QToolButton:hover {{ color: {theme.TEXT_PRIMARY}; }}"
        )
        layout.addWidget(self.toggle)

        self.content = QWidget()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(theme.SPACE_SM, 0, 0, theme.SPACE_XS)
        self.body.setSpacing(theme.SPACE_SM)
        self.content.setVisible(expanded)
        layout.addWidget(self.content)

        self.toggle.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        self.content.setVisible(checked)
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.toggled.emit(checked)

    def add(self, widget: QWidget) -> QWidget:
        self.body.addWidget(widget)
        return widget

    def add_layout(self, layout):
        self.body.addLayout(layout)
        return layout


class ParameterGrid(QWidget):
    """Aligned ``label | control | hint`` rows for settings blocks."""

    def __init__(self, parent=None, *, label_width: int = 0):
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(theme.SPACE_MD)
        self.grid.setVerticalSpacing(theme.SPACE_SM)
        self.grid.setColumnStretch(1, 1)
        self._label_width = label_width
        self._row = 0

    def add_row(self, label: str, widget: QWidget, hint: str = "") -> QWidget:
        name = QLabel(label)
        name.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if self._label_width:
            name.setMinimumWidth(self._label_width)
        self.grid.addWidget(name, self._row, 0)
        self.grid.addWidget(widget, self._row, 1)
        self._row += 1

        if hint:
            hint_label = muted(QLabel(hint))
            hint_label.setWordWrap(True)
            hint_label.setStyleSheet(
                f"color: {theme.TEXT_MUTED}; font-size: {theme.FONT_SIZE_SMALL};"
            )
            self.grid.addWidget(hint_label, self._row, 1)
            self._row += 1
        return widget

    def add_full_width(self, widget: QWidget) -> QWidget:
        self.grid.addWidget(widget, self._row, 0, 1, 2)
        self._row += 1
        return widget


# ---------------------------------------------------------------------------
# State indicators
# ---------------------------------------------------------------------------

class StatusPill(QLabel):
    """Soft-filled badge for a named state.

    States come from :data:`phd.ui.theme.STATE_COLORS`: ``idle``, ``active``,
    ``warning``, ``error``, ``info``.
    """

    def __init__(self, text: str = "Idle", state: str = "idle", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self._state = state
        self.set_state(state, text)

    def set_state(self, state: str, text: str | None = None) -> None:
        self._state = state
        if text is not None:
            self.setText(text)
        self.setStyleSheet(theme.state_pill_style(state))

    def state(self) -> str:
        return self._state


class StatusDot(QWidget):
    """A coloured dot plus a label -- the compact form used in the status bar."""

    _DOT = {
        "idle": theme.TEXT_DISABLED,
        "active": theme.SUCCESS,
        "warning": theme.WARNING,
        "error": theme.DANGER,
        "info": theme.ACCENT,
    }

    def __init__(self, text: str = "", state: str = "idle", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.dot = QLabel()
        self.dot.setFixedSize(8, 8)
        layout.addWidget(self.dot, 0, Qt.AlignVCenter)

        self.label = QLabel(text)
        layout.addWidget(self.label, 0, Qt.AlignVCenter)

        self._state = state
        self.set_state(state, text)

    def set_state(self, state: str, text: str | None = None) -> None:
        self._state = state
        if text is not None:
            self.label.setText(text)
        color = self._DOT.get(state, theme.TEXT_DISABLED)
        self.dot.setStyleSheet(
            f"background-color: {color}; border-radius: 4px;"
        )
        self.label.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: {theme.FONT_SIZE_SMALL};"
        )
        self.setToolTip(f"{self.label.text()} — {state}")

    def state(self) -> str:
        return self._state


class SessionStrip(QFrame):
    """Compact, persistent summary of the current operating session.

    It deliberately contains only cached UI state. Updating this strip must
    never read a serial device or call ROS because it refreshes once per
    second while control loops are active.
    """

    _FIELD_ORDER = ("sensor", "control", "model", "force")
    _FIELD_TITLES = {
        "sensor": "SENSOR",
        "control": "CONTROL",
        "model": "MODEL",
        "force": "FORCE",
    }
    _FIELD_STRETCH = {
        "sensor": 4,
        "control": 3,
        "model": 3,
        "force": 2,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sessionStrip")
        self.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(
            theme.SPACE_MD,
            theme.SPACE_XS,
            theme.SPACE_MD,
            theme.SPACE_XS,
        )
        layout.setSpacing(theme.SPACE_SM)
        self._values = {}
        self._states = {}

        for index, key in enumerate(self._FIELD_ORDER):
            if index:
                separator = divider(Qt.Vertical)
                separator.setObjectName("sessionSeparator")
                separator.setStyleSheet("")
                layout.addWidget(separator)

            field = QWidget()
            field.setObjectName("sessionField")
            field_layout = QVBoxLayout(field)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.setSpacing(0)

            title = QLabel(self._FIELD_TITLES[key])
            title.setObjectName("sessionFieldTitle")
            value = ElidedLabel("—")
            value.setObjectName("sessionFieldValue")
            value.setMinimumWidth(0)
            value.setStyleSheet(theme.state_text_style("idle"))
            field_layout.addWidget(title)
            field_layout.addWidget(value)
            layout.addWidget(field, self._FIELD_STRETCH[key])
            self._values[key] = value
            self._states[key] = "idle"

    def set_value(
        self,
        key: str,
        value: str,
        *,
        tooltip: str = "",
        state: str | None = None,
    ) -> None:
        label = self._values.get(key)
        if label is None:
            raise KeyError(f"Unknown session field: {key}")
        text = str(value or "—")
        label.setText(text)
        label.setToolTip(tooltip or text)
        if state is not None:
            self._states[key] = state
            label.setStyleSheet(theme.state_text_style(state))

    def value(self, key: str) -> str:
        label = self._values.get(key)
        if label is None:
            raise KeyError(f"Unknown session field: {key}")
        return label.text()

    def state(self, key: str) -> str:
        if key not in self._values:
            raise KeyError(f"Unknown session field: {key}")
        return self._states[key]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def IconButton(  # noqa: N802 - factory reads as a widget constructor
    glyph: str,
    tooltip: str,
    *,
    text: str = "",
    variant: str = "ghost",
    color: str | None = None,
    size: int = 16,
    parent=None,
) -> QPushButton:
    """A compact icon action. ``tooltip`` is required so icons stay discoverable."""
    button = QPushButton(text, parent)
    button.setIcon(icons.icon(glyph, color=color, size=size))
    button.setIconSize(icons.icon_size(size))
    button.setToolTip(tooltip)
    button.setCursor(Qt.PointingHandCursor)
    if not text:
        button.setFixedSize(theme.CONTROL_HEIGHT, theme.CONTROL_HEIGHT)
    apply_variant(button, variant)
    return button


class Toolbar(QWidget):
    """A horizontal strip of icon actions, optionally led by a title."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(theme.SPACE_XS)
        if title:
            label = QLabel(title)
            label.setObjectName("sectionHeaderTitle")
            self._layout.addWidget(label)
        self._layout.addStretch(1)

    def add(self, widget: QWidget) -> QWidget:
        """Append ``widget`` after the existing actions (before the stretch)."""
        self._layout.insertWidget(self._layout.count(), widget)
        return widget

    def add_action(
        self, glyph: str, tooltip: str, slot=None, **kwargs
    ) -> QPushButton:
        button = IconButton(glyph, tooltip, **kwargs)
        if slot is not None:
            button.clicked.connect(slot)
        return self.add(button)

    def add_separator(self) -> QFrame:
        return self.add(divider(Qt.Vertical))


class SegmentedControl(QWidget):
    """Mutually exclusive options rendered as a single joined control.

    Use for control *modes* that cannot be active at the same time, so the
    exclusivity is obvious from the shape rather than only from behaviour.
    """

    changed = pyqtSignal(str)

    def __init__(self, options: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        self.setStyleSheet(
            f"SegmentedControl {{ background-color: {theme.SURFACE_SUNKEN};"
            f" border-radius: {theme.RADIUS_MD + 1}px; }}"
        )

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        for index, (key, label) in enumerate(options):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet(
                "QPushButton { background: transparent; border: 1px solid transparent;"
                f" border-radius: {theme.RADIUS_SM}px; padding: 4px 14px;"
                f" color: {theme.TEXT_SECONDARY}; font-weight: 500; }}"
                f"QPushButton:hover {{ color: {theme.TEXT_PRIMARY}; }}"
                f"QPushButton:checked {{ background-color: {theme.SURFACE};"
                f" border-color: {theme.BORDER}; color: {theme.TEXT_PRIMARY};"
                " font-weight: 600; }"
            )
            self._group.addButton(button, index)
            self._buttons[key] = button
            layout.addWidget(button)
            button.clicked.connect(
                lambda _checked, k=key: self.changed.emit(k)
            )

        if options:
            self._buttons[options[0][0]].setChecked(True)

    def set_current(self, key: str) -> None:
        button = self._buttons.get(key)
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def current(self) -> str | None:
        for key, button in self._buttons.items():
            if button.isChecked():
                return key
        return None

    def button(self, key: str) -> QPushButton | None:
        return self._buttons.get(key)


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------

def confirm(
    parent,
    title: str,
    message: str,
    *,
    detail: str = "",
    confirm_text: str = "Continue",
    cancel_text: str = "Cancel",
    destructive: bool = True,
) -> bool:
    """Ask before an irreversible action. Returns ``True`` to proceed.

    ``Cancel`` is the default button so an accidental Return keypress never
    triggers the destructive path.
    """
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Warning if destructive else QMessageBox.Question)
    box.setText(message)
    if detail:
        box.setInformativeText(detail)

    proceed = box.addButton(confirm_text, QMessageBox.AcceptRole)
    cancel = box.addButton(cancel_text, QMessageBox.RejectRole)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    if destructive:
        apply_variant(proceed, "danger")
    else:
        apply_variant(proceed, "primary")

    box.exec_()
    return box.clickedButton() is proceed

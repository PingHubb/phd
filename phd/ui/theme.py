"""Central design tokens for the PingLab UI.

This module is the single source of truth for the visual language. The global
stylesheet ``resource/stylesheets/ui_style.qss`` is a tokenized template which
is rendered from the values below at startup. Changing a token here therefore
updates both Python-built widgets and QSS-built widgets.

Design language: a restrained, light, "professional instrument" look. Neutral
grey chrome, white working surfaces, dark readable text, hairline borders and
one saturated accent. Colour carries meaning and nothing else:

===========  ==========================================================
Blue         primary action / current selection
Green        active, running, connected, healthy
Amber        warning, degraded, needs attention
Red          stop, danger, disconnected, emergency
Grey         everything else
===========  ==========================================================

Modules should import the tokens rather than hard-coding hex values, so a
future palette change stays a one-file edit::

    from phd.ui import theme
    label.setStyleSheet(theme.MUTED_LABEL_STYLE)
"""

# ---------------------------------------------------------------------------
# Neutral surfaces
# ---------------------------------------------------------------------------
# WINDOW_BG is the app chrome (menu bar, toolbar, status bar, gutters).
# SURFACE is a working surface that content sits on. SURFACE_RAISED is used for
# secondary fills that need to separate from SURFACE without a border.
WINDOW_BG = "#F4F4F6"
SURFACE = "#FFFFFF"
SURFACE_RAISED = "#F0F0F3"
SURFACE_SUNKEN = "#E9E9ED"
INPUT_BG = "#FFFFFF"

BORDER = "#D0D0D6"
BORDER_SUBTLE = "#E3E3E8"
BORDER_STRONG = "#B8B8C0"

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
TEXT_PRIMARY = "#1C1C1E"
TEXT_SECONDARY = "#48484A"
TEXT_MUTED = "#6B6B70"
TEXT_DISABLED = "#A8A8AE"
TEXT_ON_ACCENT = "#FFFFFF"

# ---------------------------------------------------------------------------
# Semantic colours
# ---------------------------------------------------------------------------
# Every filled variant below is dark enough to carry white text at >= 4.5:1,
# so a coloured button never becomes unreadable.
ACCENT = "#0B6BD3"
ACCENT_HOVER = "#1A7BE5"
ACCENT_PRESSED = "#0A5AB0"
ACCENT_SOFT = "#E8F1FC"
ACCENT_BORDER = "#A8CBF2"
INFO = "#0B6BD3"

SUCCESS = "#1B8A3F"
SUCCESS_HOVER = "#22A34C"
SUCCESS_PRESSED = "#166F33"
SUCCESS_SOFT = "#E6F4EA"
SUCCESS_BORDER = "#A6D9B5"

WARNING = "#B45309"
WARNING_HOVER = "#D2690F"
WARNING_SOFT = "#FDF3E3"
WARNING_BORDER = "#EBCB96"

DANGER = "#C42B1C"
DANGER_HOVER = "#D93A29"
DANGER_PRESSED = "#A32316"
DANGER_SOFT = "#FDECEA"
DANGER_BORDER = "#F0B4AC"

# Neutral "idle / unknown" state fill for indicators.
NEUTRAL_SOFT = "#EFEFF2"
NEUTRAL_BORDER = "#D8D8DE"

# 3D viewport background (VTK/PyVista). Deliberately dark even though the
# chrome is light: the tactile heatmap encodes "no contact" as pure white
# (see sensor_heatmap.heatmap_rgb), so a light canvas would hide every idle
# taxel. Treated as a content canvas inside light chrome, the way pro media
# tools do, rather than as another surface.
VIEWPORT_BG = "#2B2B30"

# Secondary accents are reserved for the few places where two concurrent
# non-status actions must remain distinguishable (for example forward and
# reverse experiment recording). They are design tokens too, not local widget
# colours.
PURPLE = "#6D28D9"
PURPLE_HOVER = "#7C3AED"
TEAL = "#0F766E"
TEAL_HOVER = "#0D9488"

# Fine-grained neutral fills used by tables, scrollbars and inset surfaces.
TEXT_FAINT = "#C7C7CC"
VIEWPORT_EDGE = "#33333A"
SURFACE_BRIGHT = "#FCFCFD"
SURFACE_QUIET = "#F8F8FA"
TABLE_ALTERNATE = "#FAFAFC"
TABLE_GRID = "#EDEDF1"

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_FAMILY = (
    '"Inter", "Noto Sans", "Segoe UI", "Cantarell", '
    '"DejaVu Sans", "Helvetica Neue", sans-serif'
)
FONT_FAMILY_MONO = (
    '"JetBrains Mono", "Cascadia Mono", "Noto Sans Mono", '
    '"DejaVu Sans Mono", monospace'
)

FONT_SIZE_SMALL = "9pt"
FONT_SIZE_BASE = "10pt"
FONT_SIZE_MEDIUM = "11pt"
FONT_SIZE_TITLE = "13pt"

# The same base size as a number, plus the family stack in preference order,
# for setting QApplication's font. Qt sizes some widgets (notably QTabBar) with
# the application font but paints them with the stylesheet font, so the two
# must agree or labels get clipped.
FONT_POINT_SIZE_BASE = 10
FONT_FAMILY_PREFERENCE = (
    "Inter",
    "Noto Sans",
    "Cantarell",
    "DejaVu Sans",
)

WEIGHT_REGULAR = 400
WEIGHT_MEDIUM = 500
WEIGHT_SEMIBOLD = 600

# ---------------------------------------------------------------------------
# Type roles
# ---------------------------------------------------------------------------
# Pick a *role*, never a size. Five roles cover the whole app, so adding a
# sixth should feel like a decision -- and a literal font-size in feature code
# is a bug. A role carries colour as well as size, because that is what makes
# a caption recede: it is small *and* muted, not just small.
#
#   title     dialog and window headers
#   headline  section titles above a group of controls
#   body      default -- control labels, list items, button text
#   callout   an emphasised inline value that is not a heading
#   caption   helper text, hints, units, secondary status
TYPE_ROLES = {
    "title": (FONT_SIZE_TITLE, WEIGHT_SEMIBOLD, TEXT_PRIMARY),
    "headline": (FONT_SIZE_BASE, WEIGHT_SEMIBOLD, TEXT_PRIMARY),
    "body": (FONT_SIZE_BASE, WEIGHT_REGULAR, TEXT_PRIMARY),
    "callout": (FONT_SIZE_BASE, WEIGHT_MEDIUM, TEXT_PRIMARY),
    "caption": (FONT_SIZE_SMALL, WEIGHT_REGULAR, TEXT_MUTED),
}


def type_style(role: str, *, color: str | None = None) -> str:
    """Style snippet for a named role in :data:`TYPE_ROLES`.

    ``color`` overrides only the colour, so a status label can be amber while
    still sitting on the caption size and weight.
    """
    size, weight, default_color = TYPE_ROLES.get(role, TYPE_ROLES["body"])
    return (
        f"font-size: {size}; font-weight: {weight}; "
        f"color: {color or default_color};"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# A single 4 px rhythm keeps every gap and inset on the same grid.
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16
SPACE_XL = 24

RADIUS_SM = 4
RADIUS_MD = 6
RADIUS_LG = 8

CONTROL_HEIGHT = 28
CONTROL_HEIGHT_SM = 24
CONTROL_HEIGHT_LG = 34

# Presentation mode is intentionally opt-in. It increases hit targets and
# row rhythm for projected demos or touch use while preserving the compact
# mouse-and-keyboard layout used during daily lab work.
PRESENTATION_CONTROL_HEIGHT = 36
PRESENTATION_ROW_MIN_HEIGHT = 46

# Grouped inset list (the macOS System Settings pattern): one rounded surface
# per section, hairline separators between its rows, section title outside and
# above. ROW_PADDING_V is generous on purpose -- the rhythm of the rows is what
# replaces 36 competing group-box borders.
INSET_ROW_PADDING_H = SPACE_MD
INSET_ROW_PADDING_V = SPACE_SM
INSET_ROW_MIN_HEIGHT = 36

# Vertical padding QSS gives each QListWidget/QListView item. Duplicated here
# because a list's row height has to be known in Python to size the list to
# its contents, and Qt only reports the real value after the stylesheet has
# been applied -- too late for code that runs while the page is built. Keep in
# step with the `QListWidget::item` padding in ui_style.qss.
LIST_ITEM_PADDING_V = 5

# Width restored when the control panel is un-collapsed.
CONTROL_PANEL_WIDTH = 700
# Share of the window given to the control panel before its real size hint is
# known. Expressed as a ratio because QSplitter.setSizes normalises to the real
# width, which is not yet available when the splitter is first populated.
CONTROL_PANEL_RATIO = 0.36
# Bounds applied once the panel can be measured against its content. The panel
# asks for exactly the width its widest control row needs -- that number does
# not grow with the window, so on a large display the viewport takes all the
# extra space. VIEWPORT_MIN_WIDTH is the floor the 3D view keeps even on a
# small laptop, where the panel would otherwise swallow the whole window.
CONTROL_PANEL_MIN_WIDTH = 480
VIEWPORT_MIN_WIDTH = 460


# ---------------------------------------------------------------------------
# Reusable style snippets
# ---------------------------------------------------------------------------

# Muted helper/description text under titles and parameters.
MUTED_LABEL_STYLE = f"color: {TEXT_MUTED};"

# Informational (accent) status text.
INFO_LABEL_STYLE = f"color: {INFO};"

# Dialog headers.
TITLE_LABEL_STYLE = (
    f"font-size: {FONT_SIZE_TITLE}; font-weight: {WEIGHT_SEMIBOLD}; "
    f"color: {TEXT_PRIMARY};"
)
SUBTITLE_LABEL_STYLE = f"color: {TEXT_MUTED};"

# Monospace numeric readouts (status bar, force meter, Hz counters). A
# monospaced family is how we get tabular figures in Qt: every digit has the
# same advance, so a value updating at 60 Hz does not shuffle the widgets
# beside it as it crosses 9 -> 10 or gains a minus sign.
MONO_VALUE_STYLE = f"font-family: {FONT_FAMILY_MONO};"


def readout_style(role: str = "body", *, color: str | None = None) -> str:
    """Fixed-width numeric readout at a named type role.

    For any value that changes while the user is watching it.
    """
    return f"{MONO_VALUE_STYLE} {type_style(role, color=color)}"


def active_button_style() -> str:
    """Style for a toggled-on (running) action button.

    Green means "this is running right now". Used by ``_set_button_active`` for
    every mode toggle in the app, so it must stay visually unmistakable.
    """
    return (
        f"QPushButton {{ background-color: {SUCCESS}; border: 1px solid {SUCCESS}; "
        f"color: {TEXT_ON_ACCENT}; font-weight: {WEIGHT_SEMIBOLD}; }}"
        f"QPushButton:hover {{ background-color: {SUCCESS_HOVER}; "
        f"border-color: {SUCCESS_HOVER}; }}"
        f"QPushButton:pressed {{ background-color: {SUCCESS_PRESSED}; "
        f"border-color: {SUCCESS_PRESSED}; }}"
    )


def danger_button_style() -> str:
    """Style for a stop/destructive action button."""
    return (
        f"QPushButton {{ background-color: {DANGER}; border: 1px solid {DANGER}; "
        f"color: {TEXT_ON_ACCENT}; font-weight: {WEIGHT_SEMIBOLD}; }}"
        f"QPushButton:hover {{ background-color: {DANGER_HOVER}; "
        f"border-color: {DANGER_HOVER}; }}"
        f"QPushButton:pressed {{ background-color: {DANGER_PRESSED}; "
        f"border-color: {DANGER_PRESSED}; }}"
    )


def checkable_button_style(color: str, *, padding: str = "4px 12px") -> str:
    """Style a checkable button so its *checked* state fills with ``color``.

    Pass a semantic token, not a literal: :data:`SUCCESS` for "this mode is
    running", :data:`ACCENT` for a view/selection toggle, :data:`WARNING` for a
    mode that commands real robot motion.
    """
    return (
        f"QPushButton {{ padding: {padding}; }}"
        f"QPushButton:checked {{ background-color: {color}; border-color: {color};"
        f" color: {TEXT_ON_ACCENT}; font-weight: {WEIGHT_SEMIBOLD}; }}"
    )


def indicator_style(active: bool, *, center: bool = False) -> str:
    """Pill-shaped state indicator (PS5 / sensor mapping test dialogs)."""
    if not active:
        return (
            f"background-color: {NEUTRAL_SOFT}; color: {TEXT_MUTED}; "
            f"border: 1px solid {NEUTRAL_BORDER}; border-radius: {RADIUS_MD}px;"
        )
    if center:
        return (
            f"background-color: {ACCENT}; color: {TEXT_ON_ACCENT}; "
            f"border: 1px solid {ACCENT}; border-radius: {RADIUS_MD}px; "
            f"font-weight: {WEIGHT_SEMIBOLD};"
        )
    return (
        f"background-color: {SUCCESS}; color: {TEXT_ON_ACCENT}; "
        f"border: 1px solid {SUCCESS}; border-radius: {RADIUS_MD}px; "
        f"font-weight: {WEIGHT_SEMIBOLD};"
    )


# State name -> (text colour, soft fill, border). Used by status pills so every
# "connected / running / warning / error / idle" badge in the app agrees.
STATE_COLORS = {
    "idle": (TEXT_MUTED, NEUTRAL_SOFT, NEUTRAL_BORDER),
    "active": (SUCCESS, SUCCESS_SOFT, SUCCESS_BORDER),
    "warning": (WARNING, WARNING_SOFT, WARNING_BORDER),
    "error": (DANGER, DANGER_SOFT, DANGER_BORDER),
    "info": (ACCENT, ACCENT_SOFT, ACCENT_BORDER),
}


def state_pill_style(state: str) -> str:
    """Soft-filled badge for a named state in :data:`STATE_COLORS`."""
    fg, bg, border = STATE_COLORS.get(state, STATE_COLORS["idle"])
    return (
        f"color: {fg}; background-color: {bg}; border: 1px solid {border}; "
        f"border-radius: {RADIUS_SM}px; padding: 1px 6px; "
        f"font-weight: {WEIGHT_MEDIUM};"
    )


def state_text_style(state: str) -> str:
    """Plain coloured text for a named state (no fill)."""
    fg, _bg, _border = STATE_COLORS.get(state, STATE_COLORS["idle"])
    return f"color: {fg};"


# ---------------------------------------------------------------------------
# QSS template rendering
# ---------------------------------------------------------------------------

QSS_TOKENS = {
    "WINDOW_BG": WINDOW_BG,
    "SURFACE": SURFACE,
    "SURFACE_RAISED": SURFACE_RAISED,
    "SURFACE_SUNKEN": SURFACE_SUNKEN,
    "INPUT_BG": INPUT_BG,
    "BORDER": BORDER,
    "BORDER_SUBTLE": BORDER_SUBTLE,
    "BORDER_STRONG": BORDER_STRONG,
    "TEXT_PRIMARY": TEXT_PRIMARY,
    "TEXT_SECONDARY": TEXT_SECONDARY,
    "TEXT_MUTED": TEXT_MUTED,
    "TEXT_DISABLED": TEXT_DISABLED,
    "TEXT_ON_ACCENT": TEXT_ON_ACCENT,
    "ACCENT": ACCENT,
    "ACCENT_HOVER": ACCENT_HOVER,
    "ACCENT_PRESSED": ACCENT_PRESSED,
    "ACCENT_SOFT": ACCENT_SOFT,
    "ACCENT_BORDER": ACCENT_BORDER,
    "SUCCESS": SUCCESS,
    "SUCCESS_HOVER": SUCCESS_HOVER,
    "SUCCESS_PRESSED": SUCCESS_PRESSED,
    "SUCCESS_SOFT": SUCCESS_SOFT,
    "SUCCESS_BORDER": SUCCESS_BORDER,
    "WARNING": WARNING,
    "WARNING_HOVER": WARNING_HOVER,
    "WARNING_SOFT": WARNING_SOFT,
    "WARNING_BORDER": WARNING_BORDER,
    "DANGER": DANGER,
    "DANGER_HOVER": DANGER_HOVER,
    "DANGER_PRESSED": DANGER_PRESSED,
    "DANGER_SOFT": DANGER_SOFT,
    "DANGER_BORDER": DANGER_BORDER,
    "NEUTRAL_SOFT": NEUTRAL_SOFT,
    "NEUTRAL_BORDER": NEUTRAL_BORDER,
    "VIEWPORT_BG": VIEWPORT_BG,
    "PURPLE": PURPLE,
    "PURPLE_HOVER": PURPLE_HOVER,
    "TEAL": TEAL,
    "TEAL_HOVER": TEAL_HOVER,
    "TEXT_FAINT": TEXT_FAINT,
    "VIEWPORT_EDGE": VIEWPORT_EDGE,
    "SURFACE_BRIGHT": SURFACE_BRIGHT,
    "SURFACE_QUIET": SURFACE_QUIET,
    "TABLE_ALTERNATE": TABLE_ALTERNATE,
    "TABLE_GRID": TABLE_GRID,
    "FONT_FAMILY": FONT_FAMILY,
    "FONT_FAMILY_MONO": FONT_FAMILY_MONO,
    "FONT_SIZE_SMALL": FONT_SIZE_SMALL,
    "FONT_SIZE_BASE": FONT_SIZE_BASE,
    "FONT_SIZE_MEDIUM": FONT_SIZE_MEDIUM,
    "FONT_SIZE_TITLE": FONT_SIZE_TITLE,
    "CONTROL_HEIGHT": f"{CONTROL_HEIGHT}px",
    "PRESENTATION_CONTROL_HEIGHT": f"{PRESENTATION_CONTROL_HEIGHT}px",
    "PRESENTATION_ROW_MIN_HEIGHT": f"{PRESENTATION_ROW_MIN_HEIGHT}px",
}


def render_stylesheet(template: str, **extra_tokens: str) -> str:
    """Render a ``@TOKEN@`` QSS template from the central design tokens.

    Unknown placeholders are rejected. A misspelled token should fail loudly
    during startup instead of leaving one panel with partially applied styles.
    """
    import re

    values = dict(QSS_TOKENS)
    values.update({name: str(value) for name, value in extra_tokens.items()})
    rendered = str(template)
    for name, value in values.items():
        rendered = rendered.replace(f"@{name}@", str(value))

    unresolved = sorted(set(re.findall(r"@([A-Z][A-Z0-9_]*)@", rendered)))
    if unresolved:
        raise ValueError(
            "Unresolved QSS design tokens: " + ", ".join(unresolved)
        )
    return rendered


def presentation_override_stylesheet() -> str:
    """QSS appended while the operator enables presentation/touch mode."""
    return f"""
QPushButton, QToolButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: {PRESENTATION_CONTROL_HEIGHT}px;
}}
QWidget#settingsRow {{
    min-height: {PRESENTATION_ROW_MIN_HEIGHT}px;
}}
QTabBar::tab {{
    min-height: {PRESENTATION_CONTROL_HEIGHT}px;
}}
QListWidget::item, QListView::item {{
    padding-top: {SPACE_SM}px;
    padding-bottom: {SPACE_SM}px;
}}
"""

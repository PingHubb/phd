"""Central design tokens for the PingLab UI.

Every inline ``setStyleSheet`` call in the app should take its colors from
here so the palette stays consistent with ``resource/stylesheets/ui_style.qss``
(the global QSS is the source of truth; this module mirrors it for the few
places that need programmatic styling, e.g. state-dependent indicators).
"""

# --- Core palette (mirrors ui_style.qss header table) ---------------------
WINDOW_BG = "#0E1320"
SURFACE = "#151B28"
SURFACE_RAISED = "#1C2434"
INPUT_BG = "#0B101B"
BORDER = "#2A3447"
BORDER_SUBTLE = "#212A3B"

TEXT_PRIMARY = "#E6EBF5"
TEXT_MUTED = "#97A3B8"
TEXT_DISABLED = "#5B6779"

ACCENT = "#3D82F0"
ACCENT_HOVER = "#5493F5"
ACCENT_PRESSED = "#2F6BD0"
INFO = "#6EA8FE"

SUCCESS = "#16A34A"
SUCCESS_HOVER = "#1FBF5C"
DANGER = "#C0392B"
DANGER_HOVER = "#E74C3C"
WARNING = "#D97706"

# 3D viewport background (VTK/PyVista) — slightly lighter than the window so
# meshes and axes remain readable, but still clearly part of the dark theme.
VIEWPORT_BG = "#10161F"

# --- Reusable style snippets ----------------------------------------------

# Muted helper/description text under titles and parameters.
MUTED_LABEL_STYLE = f"color: {TEXT_MUTED};"

# Informational (light blue) status text.
INFO_LABEL_STYLE = f"color: {INFO};"

# Dialog headers.
TITLE_LABEL_STYLE = "font-size: 15px; font-weight: 600;"
SUBTITLE_LABEL_STYLE = f"color: {TEXT_MUTED};"


def active_button_style() -> str:
    """Style for a toggled-on (running) action button."""
    return (
        f"QPushButton {{ background-color: {SUCCESS}; border-color: {SUCCESS}; "
        f"color: #FFFFFF; font-weight: 600; }}"
        f"QPushButton:hover {{ background-color: {SUCCESS_HOVER}; }}"
    )


def indicator_style(active: bool, *, center: bool = False) -> str:
    """Pill-shaped state indicator (PS5 / sensor mapping test dialogs)."""
    if not active:
        return (
            f"background-color: {SURFACE_RAISED}; color: {TEXT_MUTED}; "
            f"border: 1px solid {BORDER}; border-radius: 6px;"
        )
    if center:
        return (
            f"background-color: {ACCENT}; color: #FFFFFF; "
            f"border: 1px solid {ACCENT_HOVER}; border-radius: 6px; font-weight: 600;"
        )
    return (
        f"background-color: {SUCCESS}; color: #FFFFFF; "
        f"border: 1px solid {SUCCESS_HOVER}; border-radius: 6px; font-weight: 600;"
    )

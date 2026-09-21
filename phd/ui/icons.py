"""Recolourable vector icon set for the PingLab UI.

The glyphs are stored as SVG markup in this module rather than as files so a
colour can be substituted at render time (``currentColor``) and so the icon set
travels with the Python package -- no extra ``data_files`` entry and no
resource-path resolution at import time.

Usage::

    from phd.ui import icons, theme

    button.setIcon(icons.icon("refresh"))
    button.setIcon(icons.icon("stop", color=theme.DANGER))

All glyphs are drawn on a 24x24 grid with a 1.7 px stroke so they stay
optically consistent at the 16-20 px sizes the UI uses.
"""

from __future__ import annotations

from PyQt5.QtCore import QByteArray, QRectF, QSize, Qt
from PyQt5.QtGui import QIcon, QImage, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer

from phd.ui import theme

# Stroke-based glyphs: rendered with ``fill="none"`` and a shared stroke style.
_STROKE_GLYPHS = {
    "sensor": (
        '<rect x="3.25" y="3.25" width="17.5" height="17.5" rx="3.5"/>'
        '<path d="M9.08 3.25v17.5M14.92 3.25v17.5M3.25 9.08h17.5M3.25 14.92h17.5"/>'
    ),
    "robot": (
        '<rect x="3.75" y="8" width="16.5" height="11.5" rx="3.5"/>'
        '<path d="M12 4.6V8"/>'
        '<circle cx="12" cy="3.2" r="1.5"/>'
        '<path d="M8.9 12.6v1.8M15.1 12.6v1.8"/>'
        '<path d="M1.9 12.4v2.6M22.1 12.4v2.6"/>'
    ),
    "ai": (
        '<path d="M11 2.9l1.75 4.6 4.6 1.75-4.6 1.75L11 15.6 9.25 11 4.65 9.25 9.25 7.5z"/>'
        '<path d="M18.1 14.4l.85 2.25 2.25.85-2.25.85-.85 2.25-.85-2.25-2.25-.85 2.25-.85z"/>'
    ),
    # Arc commands are written with every flag space-separated. Qt's SVG
    # parser rejects the compact form ("0 011.9-1.9") that most icon sets use.
    "hand": (
        '<path d="M8.1 11.4V5.6a1.55 1.55 0 0 1 3.1 0v5.2"/>'
        '<path d="M11.2 10.8V4.6a1.55 1.55 0 0 1 3.1 0v6.4"/>'
        '<path d="M14.3 11V7.4a1.55 1.55 0 0 1 3.1 0v6.4'
        'a6.2 6.2 0 0 1 -6.2 6.2h-1a6.2 6.2 0 0 1 -6.2 -6.2'
        'v-2.2a1.55 1.55 0 0 1 3.1 0v1.6"/>'
    ),
    "tools": (
        '<path d="M3.5 7.2h9.2M18.4 7.2h2.1"/>'
        '<path d="M3.5 16.8h2.1M11.3 16.8h9.2"/>'
        '<circle cx="15.6" cy="7.2" r="2.3"/>'
        '<circle cx="8.4" cy="16.8" r="2.3"/>'
    ),
    "play": '<path d="M7.5 4.6l11.4 7.4-11.4 7.4z"/>',
    "stop": '<rect x="6.4" y="6.4" width="11.2" height="11.2" rx="2.2"/>',
    "terminal": (
        '<rect x="2.9" y="4.1" width="18.2" height="15.8" rx="3.2"/>'
        '<path d="M7.3 9.6l2.6 2.5-2.6 2.5M12.6 14.9h4.1"/>'
    ),
    "signal": '<path d="M1.8 12h3.1l2.5-7.2 3.1 14.4 3-10.2 2.4 5.2h6.3"/>',
    "gamepad": (
        '<rect x="2.4" y="7.1" width="19.2" height="9.8" rx="4.9"/>'
        '<path d="M7.1 10.1v3.8M5.2 12h3.8"/>'
        '<path d="M15.9 10.9v.01M18.3 13.3v.01"/>'
    ),
    "keyboard": (
        '<rect x="2.4" y="6.1" width="19.2" height="11.8" rx="2.6"/>'
        '<path d="M6.1 9.6h.01M9.5 9.6h.01M12.9 9.6h.01M16.3 9.6h.01"/>'
        '<path d="M6.1 12.8h.01M9.5 12.8h.01M12.9 12.8h.01M16.3 12.8h.01"/>'
        '<path d="M7.9 15.6h8.2"/>'
    ),
    "settings": (
        '<circle cx="12" cy="12" r="3.1"/>'
        '<path d="M12 2.6v2.3M12 19.1v2.3M5 5l1.65 1.65M17.35 17.35L19 19'
        'M2.6 12h2.3M19.1 12h2.3M5 19l1.65-1.65M17.35 6.65L19 5"/>'
    ),
    "sliders": (
        '<path d="M5.4 4.2v15.6M12 4.2v15.6M18.6 4.2v15.6"/>'
        '<circle cx="5.4" cy="8.6" r="2.1"/>'
        '<circle cx="12" cy="14.4" r="2.1"/>'
        '<circle cx="18.6" cy="7.4" r="2.1"/>'
    ),
    "flask": (
        '<path d="M9.2 2.9h5.6"/>'
        '<path d="M10.1 2.9v5.7L5.2 16.9a1.9 1.9 0 0 0 1.65 2.85h10.3'
        'a1.9 1.9 0 0 0 1.65 -2.85L13.9 8.6V2.9"/>'
        '<path d="M7.4 14.2h9.2"/>'
    ),
    "power": (
        '<path d="M12 3.2v7.4"/>'
        '<path d="M6.9 7a7.6 7.6 0 1 0 10.2 0"/>'
    ),
    "refresh": (
        '<path d="M3.4 12a8.6 8.6 0 0 1 14.7 -6.05'
        'M20.6 12a8.6 8.6 0 0 1 -14.7 6.05"/>'
        '<path d="M18.3 2.2v3.9h-3.9M5.7 21.8v-3.9h3.9"/>'
    ),
    "folder": (
        '<path d="M3.2 6.4a1.9 1.9 0 0 1 1.9 -1.9h3.6l2 2.4h8.2'
        'a1.9 1.9 0 0 1 1.9 1.9v9.1a1.9 1.9 0 0 1 -1.9 1.9H5.1'
        'a1.9 1.9 0 0 1 -1.9 -1.9z"/>'
    ),
    "save": (
        '<path d="M12 3.4v10.9M7.7 10.2l4.3 4.1 4.3 -4.1"/>'
        '<path d="M4.1 17.3v2.1a1.6 1.6 0 0 0 1.6 1.6h12.6'
        'a1.6 1.6 0 0 0 1.6 -1.6v-2.1"/>'
    ),
    "copy": (
        '<rect x="8.2" y="8.2" width="11.2" height="11.2" rx="1.8"/>'
        '<path d="M15.8 8.2V5.9a1.8 1.8 0 0 0-1.8-1.8H5.9'
        'a1.8 1.8 0 0 0-1.8 1.8V14a1.8 1.8 0 0 0 1.8 1.8h2.3"/>'
    ),
    "search": (
        '<circle cx="10.7" cy="10.7" r="6.5"/>'
        '<path d="M15.4 15.4L20.2 20.2"/>'
    ),
    "trash": (
        '<path d="M3.9 6.8h16.2M9.2 6.8V4.3h5.6v2.5"/>'
        '<path d="M5.9 6.8l.95 13.1h10.3l.95-13.1"/>'
        '<path d="M10.2 10.7v5.5M13.8 10.7v5.5"/>'
    ),
    "camera": (
        '<path d="M3.4 9.1a2.1 2.1 0 0 1 2.1 -2.1h1.7l1.25 -2.1h7.1L16.8 7h1.7'
        'a2.1 2.1 0 0 1 2.1 2.1v8.5a2.1 2.1 0 0 1 -2.1 2.1H5.5'
        'a2.1 2.1 0 0 1 -2.1 -2.1z"/>'
        '<circle cx="12" cy="13.3" r="3.4"/>'
    ),
    "cube": (
        '<path d="M12 2.7l8.4 4.65v9.3L12 21.3 3.6 16.65v-9.3z"/>'
        '<path d="M3.6 7.35L12 12l8.4-4.65M12 12v9.3"/>'
    ),
    "target": (
        '<circle cx="12" cy="12" r="7.4"/>'
        '<circle cx="12" cy="12" r="2.6"/>'
        '<path d="M12 1.6v3.4M12 19v3.4M1.6 12H5M19 12h3.4"/>'
    ),
    "chart": (
        '<path d="M4.1 3.6v16.3h16.3"/>'
        '<path d="M8.2 16.6v-4.7M12.4 16.6V7.6M16.6 16.6v-6.8"/>'
    ),
    "plug": (
        '<path d="M9.1 2.8v5.1M14.9 2.8v5.1"/>'
        '<path d="M6.3 7.9h11.4v3.7a5.7 5.7 0 0 1 -11.4 0z"/>'
        '<path d="M12 17.3v4"/>'
    ),
    "check-circle": (
        '<circle cx="12" cy="12" r="8.6"/>'
        '<path d="M8.2 12.3l2.7 2.7 5.1-5.5"/>'
    ),
    "alert": (
        '<path d="M12 3.7l8.7 15.2H3.3z"/>'
        '<path d="M12 9.2v4.1M12 16.1v.01"/>'
    ),
    "x-circle": (
        '<circle cx="12" cy="12" r="8.6"/>'
        '<path d="M9.1 9.1l5.8 5.8M14.9 9.1l-5.8 5.8"/>'
    ),
    "eye": (
        '<path d="M1.9 12S5.6 5.4 12 5.4 22.1 12 22.1 12 18.4 18.6 12 18.6 1.9 12 1.9 12z"/>'
        '<circle cx="12" cy="12" r="3.1"/>'
    ),
    "chevron-left": '<path d="M14.9 5.4L8.3 12l6.6 6.6"/>',
    "chevron-right": '<path d="M9.1 5.4L15.7 12l-6.6 6.6"/>',
    "check": '<path d="M4.8 12.6l4.7 4.7L19.2 7.1"/>',
    # "transmit this one value" -- used by the per-actuator send buttons.
    "send": (
        '<path d="M3.2 12h9.4M3.2 12l17.6-7.4-3.9 7.4 3.9 7.4z"/>'
    ),
}

# Solid glyphs: rendered with ``fill`` and no stroke.
_FILL_GLYPHS = {
    "dot": '<circle cx="12" cy="12" r="5"/>',
}

_STROKE_TEMPLATE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="{color}" stroke-width="{width}" '
    'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
)

_FILL_TEMPLATE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="{color}" stroke="none">{body}</svg>'
)

DEFAULT_SIZE = 18
# Stroke width in the 24-unit viewBox. At an 18 px render this lands just under
# 1.5 device pixels, which stays crisp without looking hairline-faint.
DEFAULT_STROKE_WIDTH = 1.9

_pixmap_cache: dict[tuple, QPixmap] = {}
_icon_cache: dict[tuple, QIcon] = {}


def available() -> list[str]:
    """Return every glyph name this module can render."""
    return sorted({*_STROKE_GLYPHS, *_FILL_GLYPHS})


def svg_markup(
    name: str,
    color: str,
    *,
    stroke_width: float = DEFAULT_STROKE_WIDTH,
) -> str:
    """Return the full SVG document for ``name`` in ``color``."""
    if name in _FILL_GLYPHS:
        return _FILL_TEMPLATE.format(color=color, body=_FILL_GLYPHS[name])
    body = _STROKE_GLYPHS.get(name)
    if body is None:
        raise KeyError(f"Unknown icon {name!r}. Available: {available()}")
    return _STROKE_TEMPLATE.format(color=color, width=stroke_width, body=body)


def pixmap(
    name: str,
    *,
    color: str | None = None,
    size: int = DEFAULT_SIZE,
    ratio: float = 1.0,
    stroke_width: float = DEFAULT_STROKE_WIDTH,
) -> QPixmap:
    """Render ``name`` to a pixmap, cached per (name, colour, size, ratio)."""
    color = color or theme.TEXT_SECONDARY
    key = (name, color, size, round(ratio, 3), stroke_width)
    cached = _pixmap_cache.get(key)
    if cached is not None:
        return cached

    px = max(1, int(round(size * ratio)))
    image = QImage(px, px, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)

    renderer = QSvgRenderer(
        QByteArray(svg_markup(name, color, stroke_width=stroke_width).encode("utf-8"))
    )
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, px, px))
    painter.end()

    result = QPixmap.fromImage(image)
    result.setDevicePixelRatio(ratio)
    _pixmap_cache[key] = result
    return result


def icon(
    name: str,
    *,
    color: str | None = None,
    size: int = DEFAULT_SIZE,
    disabled_color: str | None = None,
    active_color: str | None = None,
) -> QIcon:
    """Return a ``QIcon`` for ``name`` with proper disabled/active variants.

    Supplying the disabled variant explicitly avoids Qt's default 50 % alpha
    fade, which is too faint to read on a light background.
    """
    color = color or theme.TEXT_SECONDARY
    disabled_color = disabled_color or theme.TEXT_DISABLED
    key = (name, color, size, disabled_color, active_color)
    cached = _icon_cache.get(key)
    if cached is not None:
        return cached

    result = QIcon()
    for ratio in (1.0, 2.0):
        result.addPixmap(
            pixmap(name, color=color, size=size, ratio=ratio),
            QIcon.Normal,
            QIcon.Off,
        )
        result.addPixmap(
            pixmap(name, color=disabled_color, size=size, ratio=ratio),
            QIcon.Disabled,
            QIcon.Off,
        )
        if active_color:
            for mode in (QIcon.Active, QIcon.Selected):
                result.addPixmap(
                    pixmap(name, color=active_color, size=size, ratio=ratio),
                    mode,
                    QIcon.Off,
                )
    _icon_cache[key] = result
    return result


def icon_size(size: int = DEFAULT_SIZE) -> QSize:
    """Convenience ``QSize`` for ``setIconSize``."""
    return QSize(size, size)


def write_stylesheet_assets(target_dir) -> None:
    """Write the few glyphs the QSS references by ``url()`` to ``target_dir``.

    Qt style sheets cannot take inline SVG, so the check-mark used by
    ``QCheckBox::indicator:checked`` has to exist as a file. Generating it here
    keeps its shape and colour tied to this module instead of a stale asset.
    """
    from pathlib import Path

    directory = Path(target_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assets = {
        "check-white.svg": svg_markup("check", "#FFFFFF", stroke_width=2.6),
    }
    for filename, markup in assets.items():
        path = directory / filename
        try:
            if path.exists() and path.read_text(encoding="utf-8") == markup:
                continue
            path.write_text(markup, encoding="utf-8")
        except OSError:
            # A read-only install prefix is not fatal: the checkbox simply
            # falls back to a plain filled indicator.
            continue

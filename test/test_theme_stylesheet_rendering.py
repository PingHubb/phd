"""The QSS must stay generated from the Python design tokens."""

from pathlib import Path

import pytest

from phd.ui import theme


QSS = (
    Path(__file__).resolve().parents[1]
    / "phd"
    / "resource"
    / "stylesheets"
    / "ui_style.qss"
)


def test_qss_template_does_not_duplicate_palette_hex_values():
    template = QSS.read_text()
    duplicated = {
        value
        for value in theme.QSS_TOKENS.values()
        if isinstance(value, str) and value.startswith("#") and value in template
    }
    assert not duplicated, f"palette values duplicated in QSS: {sorted(duplicated)}"


def test_qss_template_renders_without_placeholders():
    rendered = theme.render_stylesheet(
        QSS.read_text(),
        ICON_DIR="/tmp/pinglab-icons",
    )
    assert "@" not in rendered
    assert theme.ACCENT in rendered
    assert theme.WINDOW_BG in rendered


def test_unknown_qss_token_fails_loudly():
    with pytest.raises(ValueError, match="NOT_A_TOKEN"):
        theme.render_stylesheet("QWidget { color: @NOT_A_TOKEN@; }")


def test_presentation_override_increases_hit_targets():
    override = theme.presentation_override_stylesheet()
    assert f"min-height: {theme.PRESENTATION_CONTROL_HEIGHT}px" in override
    assert f"min-height: {theme.PRESENTATION_ROW_MIN_HEIGHT}px" in override

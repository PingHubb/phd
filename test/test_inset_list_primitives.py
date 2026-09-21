"""Guards on the grouped-inset-list primitives and their shared tokens.

`theme.LIST_ITEM_PADDING_V` is deliberately duplicated between Python and the
QSS: a list's row height must be known while a page is being built, but Qt only
reports the real value after the stylesheet has been applied. The duplication
is the point of failure, so it is pinned here.
"""

import re
from pathlib import Path

from phd.ui import theme

QSS = (
    Path(__file__).resolve().parents[1]
    / "phd" / "resource" / "stylesheets" / "ui_style.qss"
)


def _qss():
    return QSS.read_text()


def test_list_item_padding_token_matches_the_stylesheet():
    """If the QSS padding changes, lists sized in Python would clip."""
    block = re.search(
        r"QListWidget::item, QListView::item \{(.*?)\}", _qss(), re.S
    )
    assert block, "QListWidget::item rule not found"
    padding = re.search(r"padding:\s*(\d+)px\s+(\d+)px", block.group(1))
    assert padding, "expected a two-value padding on list items"
    assert int(padding.group(1)) == theme.LIST_ITEM_PADDING_V


def test_inset_list_surface_is_styled():
    qss = _qss()
    assert "QWidget#insetList {" in qss
    assert "border-radius" in qss.split("QWidget#insetList {")[1][:200]


def test_inset_separator_rule_outranks_the_transparent_children_rule():
    """Row children are transparent; the separator must still paint.

    The separator rule has to be scoped through #insetList, otherwise the
    broader `QWidget#insetList > QWidget { background: transparent }` wins and
    the hairlines vanish.
    """
    qss = _qss()
    assert "QWidget#insetList > QFrame#insetSeparator {" in qss


def test_nested_frames_are_removed_inside_an_inset_list():
    """No bordered box inside a bordered box."""
    qss = _qss()
    block_start = qss.find("QWidget#insetList QListWidget,")
    assert block_start != -1, "expected the nested-frame reset rule"
    block = qss[block_start:block_start + 500]
    assert "border: none" in block


def test_focus_rings_keep_the_control_box_the_same_size():
    """A 2px focus border must be paired with 1px less padding.

    Otherwise taking focus grows the control and nudges its neighbours.
    """
    qss = _qss()
    for selector, base_selector in (
        ("QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus",
         "QLineEdit, QTextEdit, QPlainTextEdit"),
        ("QComboBox:focus, QComboBox:on", "QComboBox"),
        ("QSpinBox:focus, QDoubleSpinBox:focus", "QSpinBox, QDoubleSpinBox"),
    ):
        focus = re.search(
            re.escape(selector) + r" \{(.*?)\}", qss, re.S
        )
        base = re.search(
            r"(?:^|\n)" + re.escape(base_selector) + r" \{(.*?)\}", qss, re.S
        )
        assert focus and base, selector

        focus_pad = re.search(r"padding:\s*(\d+)px\s+(\d+)px", focus.group(1))
        base_pad = re.search(r"padding:\s*(\d+)px\s+(\d+)px", base.group(1))
        assert focus_pad and base_pad, f"{selector} needs a padding pair"

        focus_border = re.search(r"border:\s*(\d+)px", focus.group(1))
        base_border = re.search(r"border:\s*(\d+)px", base.group(1))
        assert focus_border and base_border, f"{selector} needs a border width"

        grew = int(focus_border.group(1)) - int(base_border.group(1))
        for axis in (1, 2):
            shrank = int(base_pad.group(axis)) - int(focus_pad.group(axis))
            assert shrank == grew, (
                f"{selector}: border grew by {grew}px but padding shrank by "
                f"{shrank}px on axis {axis}"
            )


def test_type_roles_cover_the_documented_set():
    assert set(theme.TYPE_ROLES) == {
        "title", "headline", "body", "callout", "caption",
    }


def test_type_style_emits_size_weight_and_colour():
    style = theme.type_style("caption")
    assert "font-size" in style and "font-weight" in style and "color" in style


def test_type_style_colour_override_keeps_the_role_metrics():
    caption = theme.type_style("caption")
    amber = theme.type_style("caption", color=theme.WARNING)
    assert theme.WARNING in amber
    assert amber.split("color:")[0] == caption.split("color:")[0]


def test_unknown_role_falls_back_to_body():
    assert theme.type_style("nonsense") == theme.type_style("body")


def test_readout_style_is_fixed_width():
    """Live values must not shuffle their neighbours as digits change."""
    assert theme.FONT_FAMILY_MONO in theme.readout_style()

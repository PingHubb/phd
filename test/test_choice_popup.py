"""Choice controls show every option and use recognizable arrow icons."""

import os
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QComboBox

from phd.dependence.paths import icon_path, stylesheet_path
from phd.ui import icons, theme
from phd.ui.combo_popup import install_full_choice_popups


app = QApplication.instance() or QApplication([])


def test_down_chevron_has_both_arms():
    """The previous path retraced its left arm, so the right side never drew."""
    down = icons.svg_markup("chevron-down", "#000000")
    assert "l6.6 -6.6" in down
    assert "l-6.6 -6.6" not in down


def test_choice_arrows_are_real_chevron_icons(tmp_path):
    icons.write_stylesheet_assets(tmp_path)
    stylesheet = (
        Path(__file__).resolve().parents[1]
        / "phd"
        / "resource"
        / "stylesheets"
        / "ui_style.qss"
    ).read_text(encoding="utf-8")

    for name in (
        "chevron-up.svg",
        "chevron-down.svg",
        "chevron-up-disabled.svg",
        "chevron-down-disabled.svg",
    ):
        assert (tmp_path / name).is_file()
        assert f"@ICON_DIR@/{name}" in stylesheet
    assert "border-top: 5px solid" not in stylesheet
    assert "border-bottom: 4px solid" not in stylesheet


def test_combo_popup_requests_every_current_choice():
    install_full_choice_popups()
    combo = QComboBox()
    combo.addItems(["one", "two", "three", "four"])

    combo.showPopup()

    assert combo.maxVisibleItems() == 4
    combo.hidePopup()


def test_combo_popup_can_reach_the_last_choice():
    install_full_choice_popups()
    icon_dir = icon_path("generated")
    icons.write_stylesheet_assets(icon_dir)
    sheet = Path(stylesheet_path("ui_style.qss")).read_text(encoding="utf-8")
    app.setStyleSheet(
        theme.render_stylesheet(
            sheet,
            ICON_DIR=icon_dir.replace(os.sep, "/"),
        )
    )
    combo = QComboBox()
    combo.addItems(
        [f"choice {index} with a longer label" for index in range(30)]
    )
    combo.resize(260, 32)
    combo.show()

    combo.showPopup()
    view = combo.view()
    last = view.model().index(combo.count() - 1, 0)
    view.scrollTo(last)
    visible = view.visualRect(last)
    viewport_height = view.viewport().height()

    assert visible.isValid()
    assert visible.height() == view.sizeHintForRow(0)
    assert visible.top() >= 0
    assert visible.bottom() <= viewport_height
    combo.hidePopup()

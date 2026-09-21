"""Dialogs must fit the screen they open on.

Several analysis and viewer dialogs asked for up to 1400x1040, which does not
fit a 1366x768 laptop: the window opened with its buttons off-screen and no
way to reach them.
"""

from phd.ui.components import (
    DIALOG_SCREEN_FRACTION,
    MIN_DIALOG_HEIGHT,
    MIN_DIALOG_WIDTH,
    clamp_to_available,
)

LAPTOP = (1366, 768)
SMALL = (960, 540)
LARGE = (3840, 2160)


def test_widest_dialog_fits_a_laptop_screen():
    """1400x1040 is the largest size any dialog in the app asks for."""
    width, height = clamp_to_available(1400, 1040, LAPTOP)

    assert width <= LAPTOP[0]
    assert height <= LAPTOP[1]


def test_dialog_that_already_fits_is_left_alone():
    assert clamp_to_available(760, 420, LAPTOP) == (760, 420)


def test_large_display_keeps_the_roomy_layout():
    assert clamp_to_available(1400, 1040, LARGE) == (1400, 1040)


def test_clamped_dialog_leaves_a_margin_around_itself():
    width, height = clamp_to_available(9999, 9999, LAPTOP)

    assert width == int(LAPTOP[0] * DIALOG_SCREEN_FRACTION)
    assert height == int(LAPTOP[1] * DIALOG_SCREEN_FRACTION)


def test_minimum_window_size_still_yields_a_usable_dialog():
    width, height = clamp_to_available(1600, 800, SMALL)

    assert width <= SMALL[0]
    assert height <= SMALL[1]
    assert width >= MIN_DIALOG_WIDTH
    assert height >= MIN_DIALOG_HEIGHT


def test_tiny_request_is_floored_not_shrunk_further():
    assert clamp_to_available(10, 10, LAPTOP) == (
        MIN_DIALOG_WIDTH,
        MIN_DIALOG_HEIGHT,
    )


def test_unknown_screen_honours_the_preferred_size():
    assert clamp_to_available(1400, 1040, None) == (1400, 1040)


def test_every_dialog_size_used_in_the_app_fits_a_laptop():
    """The sizes passed to size_to_screen across the app, as a set."""
    requested = [
        (980, 620), (760, 420), (760, 520), (900, 720), (720, 520),
        (1400, 1040), (1320, 920), (900, 700), (860, 760), (820, 560),
        (920, 720), (900, 760), (520, 760), (1000, 520), (1600, 800),
        (1040, 700), (1180, 760),
    ]
    for width, height in requested:
        fitted_w, fitted_h = clamp_to_available(width, height, LAPTOP)
        assert fitted_w <= LAPTOP[0], (width, height)
        assert fitted_h <= LAPTOP[1], (width, height)

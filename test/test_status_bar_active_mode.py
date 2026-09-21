"""The status bar's active-mode readout must only name a mode that is running."""

from phd.dependence.func_sensor import (
    MySensor,
    _FeatureDisabledProxy,
    _LazyFeatureProxy,
)
from phd.ui.ui_initial import MyMainWindow


class _Helper:
    """Stand-in for a control helper such as DirectFingerMotion."""

    def __init__(self, running=False):
        self.is_running = running


class _SensorFunctions:
    """Minimal MySensor surface: helpers in ``__dict__``, production getter.

    Borrowing the real ``get_initialized_helper`` keeps the lazy-proxy
    filtering that ``_active_mode_label`` depends on under test, instead of
    re-implementing it in a stub that could drift.
    """

    get_initialized_helper = MySensor.get_initialized_helper

    def __init__(self, **helpers):
        self.__dict__.update(helpers)


class _Window:
    """Just enough of the window to read the mode precedence table."""

    _MODE_INDICATORS = MyMainWindow._MODE_INDICATORS


def _label(**helpers):
    return MyMainWindow._active_mode_label(_Window(), _SensorFunctions(**helpers))


def test_idle_when_no_helper_is_running():
    assert _label(direct_finger_motion_class=_Helper()) == "Idle"


def test_running_helper_is_named():
    assert (
        _label(direct_finger_motion_class=_Helper(running=True)) == "Rule-based DFM"
    )


def test_ai_execution_outranks_rule_based():
    assert (
        _label(
            direct_finger_motion_class=_Helper(running=True),
            ai_direct_finger_motion_execution_class=_Helper(running=True),
        )
        == "AI DFM (exec)"
    )


def test_helper_that_failed_to_initialize_does_not_report_a_mode():
    """Regression: the disabled stub answers every attribute with a function.

    ``bool(<function>)`` is True, so a truthiness test on ``is_running`` used
    to make the status bar claim AI DFM was executing on a machine where the
    helper could not even be constructed.
    """
    stub = _FeatureDisabledProxy("AI_DirectFingerMotion_execution", "Unavailable")
    assert stub.is_running() is None  # the stub really does hand back a callable
    assert _label(ai_direct_finger_motion_execution_class=stub) == "Idle"


def test_unresolved_lazy_helper_does_not_report_a_mode():
    """A helper nobody has touched yet is idle, and must not be constructed."""
    resolved = []
    proxy = _LazyFeatureProxy(
        object(), "attr", "Feature", lambda: resolved.append(True)
    )
    functions = _SensorFunctions(ai_direct_finger_motion_execution_class=proxy)
    # get_initialized_helper filters the proxy out; assert it is never resolved.
    assert MyMainWindow._active_mode_label(_Window(), functions) == "Idle"
    assert resolved == []


def test_missing_sensor_functions_is_idle():
    assert MyMainWindow._active_mode_label(_Window(), None) == "Idle"

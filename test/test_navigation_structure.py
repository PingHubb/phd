"""Guards on the workspace navigation, which had no test coverage at all.

The control panel's tabs are wired to real side effects: enabling a workspace
when its hardware appears, and starting/stopping the humanoid 3D viewport.
Those used to be driven partly by literal indices (`setTabEnabled(0, ...)`,
`setTabEnabled(2, ...)`), so reordering or inserting a workspace would have
silently disabled the wrong one. Icons are matched on tab *text*, so a rename
would silently drop them.

These checks read the source rather than building a window: the rest of this
suite deliberately avoids constructing Qt widgets, and the coupling being
guarded is structural.
"""

import ast
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[1] / "phd" / "ui"
UI_PING = UI_DIR / "ui_ping.py"

# Workspace order as the user sees it, left to right.
EXPECTED_WORKSPACES = [
    "Sensor",
    "Robots",
    "Control",
    "AI",
    "Hand",
    "Tools",
]


def _module(path):
    return ast.parse(path.read_text(), filename=str(path))


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _addtab_calls(node, receiver):
    """Yield (assigned_attr_or_None, label) for `self.<receiver>.addTab(...)`."""
    for child in ast.walk(node):
        call = None
        assigned = None
        if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call):
            call = child.value
            target = child.targets[0]
            if isinstance(target, ast.Attribute):
                assigned = target.attr
        elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
            call = child.value
        if call is None or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "addTab":
            continue
        owner = call.func.value
        if not (isinstance(owner, ast.Attribute) and owner.attr == receiver):
            continue
        label = None
        if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
            label = call.args[1].value
        yield assigned, label


def _nav_glyph_keys():
    tree = _module(UI_PING)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "NAV_GLYPHS":
            return [k.value for k in node.value.keys]
    raise AssertionError("NAV_GLYPHS not found")


def _collapsible_titles(node):
    titles = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not child.args:
            continue
        function = child.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "CollapsibleGroup"
        ):
            continue
        title = child.args[0]
        if isinstance(title, ast.Constant):
            titles.append(title.value)
    return titles


def test_workspace_order_is_explicit():
    calls = list(_addtab_calls(_function(_module(UI_PING), "setup_tabs"), "tab_widget"))
    assert [label for _assigned, label in calls] == EXPECTED_WORKSPACES


def test_every_workspace_index_is_captured_in_a_named_attribute():
    """A bare addTab means some later code has to guess the position."""
    calls = list(_addtab_calls(_function(_module(UI_PING), "setup_tabs"), "tab_widget"))
    unnamed = [label for assigned, label in calls if assigned is None]
    assert not unnamed, f"these workspaces have no *_tab_index attribute: {unnamed}"


def test_each_workspace_keeps_its_expected_index_attribute():
    """Pins label -> attribute, so a rename cannot orphan the callbacks."""
    expected = {
        "Sensor": "sensor_tab_index",
        "Robots": "robots_tab_index",
        "Control": "control_tab_index",
        "AI": "ai_tab_index",
        "Hand": "hand_tab_index",
        "Tools": "tools_tab_index",
    }
    calls = {
        label: assigned
        for assigned, label in _addtab_calls(
            _function(_module(UI_PING), "setup_tabs"), "tab_widget"
        )
    }
    assert calls == expected


def test_nav_glyphs_covers_every_workspace():
    """Icons are looked up by tab text, so a rename must update NAV_GLYPHS."""
    missing = set(EXPECTED_WORKSPACES) - set(_nav_glyph_keys())
    assert not missing, f"workspaces with no icon: {sorted(missing)}"


def test_nav_glyphs_has_no_stale_entries():
    stale = set(_nav_glyph_keys()) - set(EXPECTED_WORKSPACES)
    assert not stale, f"NAV_GLYPHS keys matching no workspace: {sorted(stale)}"


def test_no_literal_index_is_passed_to_set_tab_enabled():
    """Regression: `setTabEnabled(0, ...)` / `setTabEnabled(2, ...)`.

    Those two disabled the Sensor and AI workspaces when the sensor module was
    missing. Inserting Control at index 2 would have made the second one
    disable Control instead, with no error and no visible cause.
    """
    offenders = []
    for path in sorted(UI_DIR.glob("*.py")):
        for node in ast.walk(_module(path)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "setTabEnabled" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, int):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"literal tab index passed to setTabEnabled at {offenders}"


def test_sensor_subtabs_are_named_by_task():
    calls = list(_addtab_calls(_function(_module(UI_PING), "setup_tab1"), "sensor_sub_tabs"))
    assert [label for _a, label in calls] == ["Setup", "Diagnostics"]


def test_ai_subtabs_separate_running_from_data():
    calls = list(_addtab_calls(_function(_module(UI_PING), "setup_tab3"), "ai_sub_tabs"))
    assert [label for _a, label in calls] == ["Run", "Data"]


def test_dense_robot_and_ai_pages_use_basic_advanced_disclosure():
    tree = _module(UI_PING)
    assert _collapsible_titles(_function(tree, "setup_tab2")) == [
        "Basic controls",
        "Advanced commands",
    ]
    ai_titles = _collapsible_titles(_function(tree, "setup_tab3"))
    assert all(title in ai_titles for title in [
        "Basic controls",
        "Advanced setup",
        "Basic recording",
        "Advanced labels",
    ])


def test_data_subtab_index_is_captured_for_the_record_shortcut():
    """Space/Esc only toggle recording while the Data subtab is showing."""
    calls = dict(
        (label, assigned)
        for assigned, label in _addtab_calls(
            _function(_module(UI_PING), "setup_tab3"), "ai_sub_tabs"
        )
    )
    assert calls["Data"] == "ai_data_training_tab_index"


def test_robots_subtab_index_is_captured_for_the_arm_gate():
    calls = dict(
        (label, assigned)
        for assigned, label in _addtab_calls(
            _function(_module(UI_PING), "setup_tabs"), "robots_sub_tabs"
        )
    )
    assert calls["TM Robot"] == "tm_robot_tab_index"
    assert calls["Humanoid"] == "humanoid_tab_index"

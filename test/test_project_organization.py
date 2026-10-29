"""Compatibility and discoverability checks for the grouped project layout."""

import ast
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import pytest

from phd.script.catalog import TOOLS, tool_module


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", sorted(name for name, (group, _) in TOOLS.items() if group != "hardware"))
def test_legacy_import_is_the_canonical_module_not_a_copy(name):
    legacy = importlib.import_module(f"phd.script.{name}")
    canonical = importlib.import_module(tool_module(name))
    assert legacy is canonical
    assert legacy.main is canonical.main
    assert Path(canonical.__file__).parent.name == TOOLS[name][0]
    # Private helpers and subsequent monkeypatching must share module state.
    marker = object()
    canonical._organization_test_marker = marker
    try:
        assert legacy._organization_test_marker is marker
    finally:
        del canonical._organization_test_marker


@pytest.mark.parametrize("module", [
    "analysis.check_ai_direct_finger_motion_dataset",
    "analysis.benchmark_ai_direct_finger_motion_robustness",
    "training.train_ai_direct_finger_motion",
    "training.train_tactile_proximity",
])
def test_moving_tools_does_not_change_default_package_resource_root(module):
    tool = importlib.import_module(f"phd.script.{module}")
    from phd.dependence.paths import PACKAGE_ROOT

    assert tool.PACKAGE_ROOT == PACKAGE_ROOT


@pytest.mark.parametrize("name,group", [
    ("train_touch_tracking", "training"),
    ("train_intention", "training"),
    ("analyse_height_profile", "analysis"),
    ("analyse_proximity_profile", "analysis"),
    ("iges_xyz_to_csv", "conversion"),
    ("predict_robot_dog_gesture", "inference"),
])
@pytest.mark.parametrize("legacy", [True, False])
def test_direct_script_help_from_unrelated_working_directory(tmp_path, name, group, legacy):
    path = ROOT / "phd" / "script"
    path = path / f"{name}.py" if legacy else path / group / f"{name}.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)  # exercise the source-tree bootstrap, not test-runner paths
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run([sys.executable, str(path), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("command", ["list", "paths"])
def test_catalogue_is_cold_no_ml_qt_or_hardware_imports(command, capsys):
    # A fresh interpreter lets us prove the cold-start import boundary even
    # when other collected tests have already imported Qt or torch.
    script = f"""
import sys
from phd.script.__main__ import main
assert main([{command!r}]) == 0
for prefix in ('torch', 'PyQt5', 'rclpy', 'serial', 'phd.ui', 'phd.dependence.robot_api'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert ("TRAINING" if command == "list" else "Experiments:") in result.stdout


@pytest.mark.parametrize("name", ["rh56f1_hand_menu", "setup_goodix_usb_permissions"])
def test_catalogue_hardware_help_cannot_start_device_or_permission_actions(name, monkeypatch, capsys):
    from phd.script import __main__ as catalogue

    monkeypatch.setattr(catalogue.runpy, "run_module", lambda *_a, **_kw: pytest.fail("hardware executed for help"))
    assert catalogue.main([name, "--help"]) == 0
    assert "intentionally want" in capsys.readouterr().out


def test_catalogue_dispatch_keeps_arguments_and_restores_process_argv(monkeypatch):
    from phd.script import __main__ as catalogue

    observed = []
    previous = sys.argv
    monkeypatch.setattr(catalogue.runpy, "run_module", lambda module, **kw: observed.append((module, list(sys.argv), kw)))
    assert catalogue.main(["train_touch_tracking", "check", "/a folder/trial", "--supervision", "camera_free"]) == 0
    assert observed == [("phd.script.training.train_touch_tracking",
                         ["train_touch_tracking", "check", "/a folder/trial", "--supervision", "camera_free"],
                         {"run_name": "__main__"})]
    assert sys.argv is previous


def test_existing_ros_command_names_remain_and_targets_exist():
    tree = ast.parse((ROOT / "setup.py").read_text())
    setup = next(node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)).value
    entries = ast.literal_eval(next(kw.value for kw in setup.keywords if kw.arg == "entry_points"))["console_scripts"]
    mapping = dict(item.split(" = ") for item in entries)
    assert {"phd_ui", "check_ai_dfm_dataset", "train_ai_dfm_model", "train_tactile_proximity",
            "train_touch_tracking", "setup_goodix_usb_permissions", "analyse_height_profile",
            "analyse_proximity_profile"} <= mapping.keys()
    for destination in mapping.values():
        module, function = destination.split(":")
        path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
        assert path.is_file()
        assert any(isinstance(node, ast.FunctionDef) and node.name == function
                   for node in ast.parse(path.read_text()).body)


@pytest.mark.parametrize("folder", ["calibration_recordings", "taxel_position_maps", "sigraph2026_scanning_recordings"])
def test_recordings_live_only_under_experiment(folder):
    legacy = ROOT / "phd" / "resource" / folder
    assert not legacy.exists() and not legacy.is_symlink(), (
        f"{legacy} splits recordings away from resource/experiment/{folder}"
    )
    assert (ROOT / "phd" / "resource" / "experiment" / folder).is_dir()


def test_local_links_in_project_indexes_and_guides_exist():
    paths = [ROOT/"README.md", *ROOT.glob("docs/**/*.md"),
             ROOT/"phd/README.md", ROOT/"phd/INTENTION_LEARNING.md",
             ROOT/"phd/ROBOT_DOG_GESTURES.md", ROOT/"phd/SENSOR_CHARACTERISATION.md",
             *ROOT.glob("phd/**/README.md"), ROOT/"archive/README.md"]
    missing = []
    for path in paths:
        for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            target = raw.strip().strip("<>")
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not (path.parent / target).exists():
                missing.append((str(path.relative_to(ROOT)), target))
    assert not missing

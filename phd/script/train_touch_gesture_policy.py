"""Entry point for the grouped six-gesture policy trainer."""
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from runpy import run_module
    run_module("phd.script.training.train_touch_gesture_policy", run_name="__main__")
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("phd.script.training.train_touch_gesture_policy")

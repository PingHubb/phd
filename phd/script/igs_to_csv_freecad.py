"""Compatibility entry point; edit :mod:`phd.script.conversion.igs_to_csv_freecad` instead."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from runpy import run_module

    run_module("phd.script.conversion.igs_to_csv_freecad", run_name="__main__")
else:
    from importlib import import_module

    # Alias the module itself, not just public exports: private helpers,
    # dataclasses, monkeypatching and existing imports keep one shared state.
    sys.modules[__name__] = import_module("phd.script.conversion.igs_to_csv_freecad")

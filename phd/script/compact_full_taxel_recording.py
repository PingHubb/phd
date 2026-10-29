"""Entry point; edit phd.script.conversion.compact_full_taxel_recording."""
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from runpy import run_module
    run_module("phd.script.conversion.compact_full_taxel_recording", run_name="__main__")
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("phd.script.conversion.compact_full_taxel_recording")

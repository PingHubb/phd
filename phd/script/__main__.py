"""Browse tools or run one explicit command; listing never connects hardware."""

import argparse
import runpy
import sys

from phd.script.catalog import TOOLS, tool_module


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    if not args or args[0] in {"list", "--help", "-h"}:
        print("Usage: python3 -m phd.script [list | paths | TOOL [arguments...]]")
        for group in ("training", "analysis", "inference", "conversion", "hardware"):
            print(f"\n{group.upper()}")
            for name, (category, description) in TOOLS.items():
                if category == group:
                    print(f"  {name}\n    {description}")
        print("\nHardware utilities act only when explicitly run; listing performs no device actions.")
        return 0
    if args[0] == "paths":
        if len(args) != 1:
            parser.error("paths does not accept extra arguments")
        from phd.dependence.paths import (
            AI_RESOURCE_DIR, EXPERIMENT_RESOURCE_DIR, PACKAGE_ROOT, RESOURCE_ROOT,
        )

        for label, path in (
            ("Imported Python package", PACKAGE_ROOT), ("Resources", RESOURCE_ROOT),
            ("Experiments", EXPERIMENT_RESOURCE_DIR), ("AI resources", AI_RESOURCE_DIR),
            ("AI datasets", AI_RESOURCE_DIR / "data"), ("AI models", AI_RESOURCE_DIR / "models"),
        ):
            print(f"{label}: {path}")
        return 0
    name, extra = args[0], args[1:]
    if name not in TOOLS:
        parser.error(f"unknown tool {name!r}; run 'python3 -m phd.script list'")
    group, description = TOOLS[name]
    # These original utilities do not implement argparse help. Never treat
    # an innocent --help request as permission to start hand control or udev.
    if group == "hardware" and extra:
        if extra in (["--help"], ["-h"]):
            print(f"{name}: {description}\nModule: {tool_module(name)}")
            print("Run without arguments only when you intentionally want this hardware/system action.")
            return 0
        parser.error(f"{name} does not accept command-line options")
    previous = sys.argv
    try:
        sys.argv = [name, *extra]
        runpy.run_module(tool_module(name), run_name="__main__")
    finally:
        sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

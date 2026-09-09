"""Install the udev rule required for non-root Goodix USB access."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RULE_PATH = Path("/etc/udev/rules.d/70-pinglab-goodix-touch.rules")
RULE_TEXT = (
    'SUBSYSTEM=="usb", ATTR{idVendor}=="27c6", ATTR{idProduct}=="0528", '
    'MODE="0660", GROUP="plugdev", TAG+="uaccess"\n'
)


def main():
    """Write and reload the PingLab Goodix USB permission rule."""
    if os.geteuid() != 0:
        print(
            "Administrator permission is required. Run:\n"
            f"  sudo python3 {Path(__file__).resolve()}"
        )
        return 1

    RULE_PATH.write_text(RULE_TEXT, encoding="ascii")
    subprocess.run(
        ["udevadm", "control", "--reload-rules"],
        check=True,
    )
    subprocess.run(
        [
            "udevadm",
            "trigger",
            "--subsystem-match=usb",
            "--attr-match=idVendor=27c6",
            "--attr-match=idProduct=0528",
        ],
        check=True,
    )
    print(f"Installed {RULE_PATH}")
    print("Reconnect the Goodix sensor, then restart phd_ui.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

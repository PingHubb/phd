"""Central path helpers for the PingLab project.

Keep runtime code independent from a specific user's workspace path.  These
helpers resolve paths relative to the installed/source ``phd`` Python package,
so the UI can run from a moved workspace without hard-coded ``/home/...``
strings.
"""

from __future__ import annotations

import os
from pathlib import Path


AI_RESOURCE_ROOT_ENV = "PINGLAB_AI_RESOURCE_ROOT"
RESOURCE_ROOT_ENV = "PINGLAB_RESOURCE_ROOT"

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def _resolve_resource_root() -> Path:
    """Return the shared resource root, allowing source-tree overrides."""
    external_resource_root = _env_path(RESOURCE_ROOT_ENV)
    if external_resource_root is not None:
        return external_resource_root
    return PACKAGE_ROOT / "resource"


def _resolve_ai_resource_dir() -> Path:
    """Return the AI asset root, allowing large models/data to live externally.

    Resolution order:
    1. ``PINGLAB_AI_RESOURCE_ROOT`` points directly at an ``ai`` directory.
    2. ``PINGLAB_RESOURCE_ROOT`` points at a resource root containing ``ai``.
    3. Fall back to the source/package ``phd/resource/ai`` directory.
    """
    explicit_ai_root = _env_path(AI_RESOURCE_ROOT_ENV)
    if explicit_ai_root is not None:
        return explicit_ai_root

    external_resource_root = _env_path(RESOURCE_ROOT_ENV)
    if external_resource_root is not None:
        return external_resource_root / "ai"

    return RESOURCE_ROOT / "ai"


RESOURCE_ROOT = _resolve_resource_root()
AI_RESOURCE_DIR = _resolve_ai_resource_dir()

ICON_DIR = RESOURCE_ROOT / "icon"
STYLESHEET_DIR = RESOURCE_ROOT / "stylesheets"
SENSOR_RESOURCE_DIR = RESOURCE_ROOT / "sensor"
ROBOT_RESOURCE_DIR = RESOURCE_ROOT / "robot"


def resource_path(*parts: str) -> str:
    """Return an absolute path under ``phd/resource`` as ``str``.

    Existing code and third-party APIs mostly expect strings, so this helper
    keeps call sites simple while centralising the actual base directory.
    With no ``parts``, returns the resource root directory itself.
    """
    if not parts:
        return str(RESOURCE_ROOT)
    return str(RESOURCE_ROOT.joinpath(*parts))


def icon_path(name: str) -> str:
    return str(ICON_DIR / name)


def stylesheet_path(name: str) -> str:
    return str(STYLESHEET_DIR / name)


def robot_resource_path(*parts: str) -> str:
    if not parts:
        return str(ROBOT_RESOURCE_DIR)
    return str(ROBOT_RESOURCE_DIR.joinpath(*parts))


def sensor_resource_path(*parts: str) -> str:
    if not parts:
        return str(SENSOR_RESOURCE_DIR)
    return str(SENSOR_RESOURCE_DIR.joinpath(*parts))


def ai_resource_path(*parts: str) -> str:
    if not parts:
        return str(AI_RESOURCE_DIR)
    return str(AI_RESOURCE_DIR.joinpath(*parts))

"""Shared, portable names for calibration recordings and their migrations."""

from __future__ import annotations

import hashlib


PROTOCOL_TOKENS = {
    "standard": "loading-unloading",
    "hold_recovery": "hold-recovery",
    "speed_sweep": "speed-comparison",
    "force_sweep": "force-level-sweep",
    "drift": "drift-repeated-cycling",
    "sliding": "spatial-sliding",
    "proximity": "proximity-approach",
    "proximity_profile": "proximity-profile",
}


def _token(value: str) -> str:
    result = "".join(
        c if c.isalnum() or c in "-_" else "-" for c in str(value).strip()
    )
    while "--" in result:
        result = result.replace("--", "-")
    return result.strip("-_") or "unknown"


def sensor_name_for_recording(label="", identity="", shape=(0, 0)) -> str:
    """Prefer the recorded display name, then a known stable identity."""
    if str(label or "").strip():
        return str(label).strip()
    identity = str(identity or "").strip()
    if identity and not identity.startswith("layout:"):
        from phd.dependence.humanoid_sensor_registry import HUMANOID_PART_DISPLAY

        profile = HUMANOID_PART_DISPLAY.get(identity)
        return str(profile[0]) if profile else identity
    rows, columns = shape
    return f"Unknown {int(rows)}x{int(columns)} Sensor"


def calibration_folder_name(kind, timestamp, settings, *, sensor_name, protocol):
    """Keep the settings intact and bound long sensor names to 255 UTF-8 bytes."""
    prefix = f"{kind}_{_token(timestamp)}_sensor-"
    protocol_token = PROTOCOL_TOKENS.get(str(protocol), _token(protocol))
    suffix = f"_protocol-{protocol_token}_{settings}"
    budget = 255 - len((prefix + suffix).encode("utf-8"))
    if budget < 10:
        raise ValueError("Calibration settings leave too little room for the sensor name.")
    token = _token(sensor_name)
    if len(token.encode("utf-8")) > budget:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
        token = token.encode("utf-8")[:budget - 9].decode("utf-8", errors="ignore").rstrip("-_")
        token = f"{token}-{digest}"
    return prefix + token + suffix

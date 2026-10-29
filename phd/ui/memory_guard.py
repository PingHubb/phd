"""Free-memory checks so long experiments stop cleanly instead of being OOM-killed."""

from __future__ import annotations

from typing import Optional

GIB = 1024 ** 3


def _proc_field_bytes(path: str, key: str) -> Optional[int]:
    try:
        with open(path, encoding="ascii") as stream:
            for line in stream:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def available_memory_bytes() -> Optional[int]:
    """Memory the kernel can still hand out (MemAvailable); None when unknown."""
    return _proc_field_bytes("/proc/meminfo", "MemAvailable:")


def process_rss_bytes() -> Optional[int]:
    """Resident memory of this process; None when unknown."""
    return _proc_field_bytes("/proc/self/status", "VmRSS:")

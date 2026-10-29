"""Find saved position maps by stable sensor identity, without device access."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from phd.ui.taxel_position_navigation import prepare_taxel_map


@dataclass
class SavedTaxelMap:
    path: Path
    fingerprint: tuple
    saved_time: float
    result: dict


class TaxelMapCatalog:
    """Cache parsed maps; inspect file stats to notice new/resumed recordings."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self._cache = {}
        self.errors = {}

    def latest(self, sensor_identity, shape):
        # Grid size and USB/ACM number cannot identify a physical sensor.
        identity = str(sensor_identity or "").strip()
        if not identity or identity.startswith("layout:"):
            return None
        found = set()
        candidates = []
        for path in self.directory.rglob("taxel_position_map.json"):
            try:
                stat = path.stat()
                fingerprint = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
                found.add(path)
                cached = self._cache.get(path)
                if cached is None or cached[0] != fingerprint:
                    entry = None
                    try:
                        with path.open(encoding="utf-8") as stream:
                            result = prepare_taxel_map(json.load(stream))
                        # Acquisition time survives copying/restoring a folder.
                        saved_time = stat.st_mtime
                        try:
                            saved = datetime.fromisoformat(str(result.get("saved_at", "")).replace("Z", "+00:00"))
                            if saved.tzinfo is None:
                                saved = saved.replace(tzinfo=timezone.utc)
                            saved_time = saved.timestamp()
                        except (ValueError, TypeError, OverflowError):
                            pass
                        entry = SavedTaxelMap(path, fingerprint, saved_time, result)
                        self.errors.pop(path, None)
                    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                        self.errors[path] = str(exc)
                    self._cache[path] = (fingerprint, entry)
                entry = self._cache[path][1]
                if entry is None or entry.result.get("sensor_identity") != identity:
                    continue
                settings = entry.result["settings"]
                if (int(settings["sensor_rows"]), int(settings["sensor_columns"])) == tuple(shape):
                    candidates.append(entry)
            except OSError:
                continue
        self._cache = {path: item for path, item in self._cache.items() if path in found}
        self.errors = {path: error for path, error in self.errors.items() if path in found}
        return max(candidates, key=lambda item: (item.saved_time, str(item.path)), default=None)

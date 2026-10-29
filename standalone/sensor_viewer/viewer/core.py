"""Hardware-independent frame validation, layout and recording."""

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Grid:
    rows: int
    columns: int

    def __post_init__(self):
        if not (1 <= self.rows <= 64 and 1 <= self.columns <= 64
                and self.rows * self.columns <= 1600):
            raise ValueError("Choose a grid with 1–64 rows/columns and at most 1,600 taxels.")

    @property
    def size(self):
        return self.rows * self.columns

    def extract(self, values):
        data = np.asarray(values, dtype=float)
        allowed = {self.size, self.rows * (self.columns + 1)}
        if (self.rows, self.columns) == (10, 9):
            allowed.add(110)  # Legacy extra reference block, as in the lab UI.
        if data.ndim != 1 or data.size not in allowed or not np.isfinite(data).all():
            raise ValueError(
                f"Frame has {data.size} values; {self.rows}×{self.columns} expects "
                f"{self.size} (or {self.rows * (self.columns + 1)} with a reference column). "
                "Select the correct grid, then reconnect."
            )
        return data[:self.size].copy()

    @classmethod
    def from_channels(cls, values):
        # Existing sensor firmware reports drive columns INCLUDING its spare
        # reference column, followed by the number of sense rows.
        if len(values) != 2:
            raise ValueError("No valid channelCheck response. Select your grid manually.")
        drive, sense = values
        if int(drive) != drive or int(sense) != sense:
            raise ValueError("Invalid channel dimensions.")
        return cls(int(sense), int(drive) - 1)


def displayed_values(raw, baseline, mode):
    raw = np.asarray(raw, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if mode == "raw":
        return raw
    if mode == "baseline":
        return baseline
    if mode == "percent":
        result = np.full(raw.shape, np.nan)
        np.divide(100 * (raw - baseline), baseline, out=result, where=baseline != 0)
        return result
    return raw - baseline  # Never absolute value.


class Recording:
    """Record every delivered frame, independent of the selected display taxel."""

    def __init__(self, path, grid, source):
        self.path = Path(path)
        self.metadata_path = self.path.with_suffix(".json")
        if self.path.exists() or self.metadata_path.exists():
            raise FileExistsError("That CSV or its JSON already exists. Choose a new filename.")
        self.grid = grid
        self.started = None
        self.count = 0
        self.closed = False
        self.metadata = dict(
            format="pinglab_sensor_viewer_v1", rows=grid.rows, columns=grid.columns,
            indexing="index = column * rows + row; row 0 is the top row",
            signal="signed difference = raw - baseline; raw sensor units, not force",
            source=source, started_at=datetime.now(timezone.utc).isoformat(),
            baseline_history=[], frames=0, dropped_frames=0,
        )
        # Reserve the sidecar first. Do not overwrite existing recordings.
        with self.metadata_path.open("x", encoding="utf-8") as stream:
            json.dump(self.metadata, stream, indent=2)
        self.stream = self.path.open("x", newline="", encoding="utf-8")
        self.writer = csv.writer(self.stream)
        self.writer.writerow(["timestamp_utc", "elapsed_s", "sequence", "baseline_version"]
                             + [f"taxel_{i:03d}_raw" for i in range(grid.size)]
                             + [f"taxel_{i:03d}_difference" for i in range(grid.size)])
        self.stream.flush()

    def append(self, packet):
        if self.started is None:
            self.started = packet["time"]
        version = packet["baseline_version"]
        history = self.metadata["baseline_history"]
        if not history or history[-1]["version"] != version:
            history.append(dict(version=version, sequence=packet["sequence"],
                                values=packet["baseline"].tolist()))
        self.writer.writerow([packet["utc"], f"{packet['time'] - self.started:.6f}",
                              packet["sequence"], version]
                             + packet["raw"].tolist()
                             + (packet["raw"] - packet["baseline"]).tolist())
        self.count += 1
        if self.count % 50 == 0:
            self.stream.flush()

    def close(self, dropped_frames=0):
        if self.closed:
            return
        self.closed = True
        self.stream.close()
        self.metadata.update(frames=self.count, dropped_frames=int(dropped_frames),
                             ended_at=datetime.now(timezone.utc).isoformat())
        with self.metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(self.metadata, stream, indent=2)

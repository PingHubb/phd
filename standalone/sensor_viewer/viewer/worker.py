"""A single serial owner: acquisition and calibration never race each other."""

from collections import deque
from datetime import datetime, timezone
from queue import Empty, Queue
import threading
import time

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from phd.dependence.sensor_api import ArduinoCommander
from phd.dependence.sensor_protocol import discard_pending_serial_input
from .core import Grid


class SensorWorker(QThread):
    status = pyqtSignal(str)
    connected = pyqtSignal(object, object)
    failed = pyqtSignal(str)
    calibration_finished = pyqtSignal(bool)

    def __init__(self, port, baud, grid=None, demo=False, parent=None, api_factory=None):
        super().__init__(parent)
        self.port, self.baud, self.grid, self.demo = port, baud, grid, demo
        self.api_factory = api_factory or ArduinoCommander
        self.commands = Queue()
        self.packets = deque(maxlen=256)
        self.lock = threading.Lock()
        self.stop_requested = threading.Event()
        self.dropped = 0

    def stop(self):
        self.stop_requested.set()
        self.requestInterruption()

    def stopped(self):
        # Also covers cancellation just before QThread starts executing.
        return self.stop_requested.is_set() or self.isInterruptionRequested()

    def calibrate(self):
        self.commands.put("calibrate")

    def take_packets(self):
        with self.lock:
            packets = list(self.packets)
            self.packets.clear()
        return packets

    def _publish(self, sequence, raw, baseline, version):
        packet = dict(sequence=sequence, time=time.monotonic(),
                      utc=datetime.now(timezone.utc).isoformat(),
                      raw=raw, baseline=baseline, baseline_version=version)
        with self.lock:
            if len(self.packets) == self.packets.maxlen:
                self.dropped += 1
            self.packets.append(packet)

    def _sleep(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.stopped() and time.monotonic() < deadline:
            self.msleep(20)

    def _connect_sensor(self):
        rates = [115200, 9600] if self.baud == 0 else [self.baud]
        errors = []
        for baud in rates:
            if self.stopped():
                return None
            self.status.emit(f"Opening {self.port} at {baud:,} baud…")
            api = self.api_factory(self.port, baud_rate=baud, connect_immediately=False)
            if not api.reconnect():
                api.close()
                errors.append(f"Cannot open {self.port}; close other apps using it.")
                continue
            try:
                # Some ESP32 boards reset when their COM port opens.
                self._sleep(2.0)
                discard_pending_serial_input(api.ser)
                grid = self.grid
                if grid is None:
                    for _attempt in range(3):
                        if self.stopped():
                            api.close()
                            return None
                        try:
                            grid = Grid.from_channels(api.channel_check() or [])
                            break
                        except ValueError:
                            self._sleep(.15)
                    if grid is None:
                        raise ValueError("Grid detection failed. Select 5×5, 8×10, 20×20 or Custom, then reconnect.")
                raw = None
                for _attempt in range(3):
                    if self.stopped():
                        api.close()
                        return None
                    values = api.read_raw()
                    if values is not None:
                        raw = grid.extract(values)
                        break
                if raw is None:
                    raise ValueError("No complete sensor frames. Check baud rate, firmware and USB cable.")
                # Exact packet length is learned from real data, including any
                # reference column. Truncated packets must never become frames.
                api.expected_payload_values = len(values)
                self.grid, self.baud = grid, baud
                calibration = api.read_cal()
                if calibration is None:
                    self.status.emit("No stored baseline. Keep the sensor untouched; collecting 20 baseline frames…")
                    frames = [raw]
                    for _ in range(19):
                        if self.stopped():
                            api.close()
                            return None
                        sample = api.read_raw()
                        if sample is None:
                            raise ValueError("Baseline acquisition failed. Check the sensor connection.")
                        frames.append(grid.extract(sample))
                    baseline = np.median(frames, axis=0)
                    origin = "20-frame software baseline"
                else:
                    baseline = grid.extract(calibration)
                    origin = "stored sensor baseline"
                return api, raw, baseline, origin
            except Exception as exc:
                api.close()
                errors.append(str(exc))
        if self.stopped():
            return None
        raise RuntimeError(" / ".join(errors))

    def run(self):
        api = None
        try:
            if self.demo:
                self.grid = self.grid or Grid(5, 5)
                baseline = np.full(self.grid.size, 1000.0)
                raw = baseline.copy()
                origin = "synthetic baseline"
            else:
                connected = self._connect_sensor()
                if connected is None:
                    return
                api, raw, baseline, origin = connected
            source = dict(port=self.port, baud=self.baud, demo=self.demo, baseline_source=origin)
            self.connected.emit(self.grid, source)
            self.status.emit(f"{'DEMO · synthetic data' if self.demo else self.port + ' · connected'} · {origin}")
            version, sequence, misses = 1, 0, 0
            started = time.monotonic()
            rng = np.random.default_rng(19)
            while not self.stopped():
                try:
                    command = self.commands.get_nowait()
                except Empty:
                    command = None
                if command == "calibrate":
                    self.status.emit("Updating baseline — keep the sensor untouched…")
                    try:
                        if self.demo:
                            new_baseline = np.full(self.grid.size, 1000.0)
                        else:
                            response = api.update_cal()
                            if response is None:
                                response = api.read_cal()
                            if response is None:
                                raise ValueError("No calibration response; previous baseline retained.")
                            new_baseline = self.grid.extract(response)
                        baseline = new_baseline
                        version += 1
                        self.status.emit("Baseline updated. Signed difference = raw − baseline.")
                        self.calibration_finished.emit(True)
                    except Exception as exc:
                        self.status.emit(f"Calibration failed: {exc}")
                        self.calibration_finished.emit(False)
                if self.demo:
                    t = time.monotonic() - started
                    columns, rows = np.divmod(np.arange(self.grid.size), self.grid.rows)
                    centre_r = (self.grid.rows - 1) * (.5 + .35 * np.sin(t * .7))
                    centre_c = (self.grid.columns - 1) * (.5 + .35 * np.cos(t * .5))
                    width = max(1., min(self.grid.rows, self.grid.columns) / 6)
                    blob = np.exp(-((rows-centre_r)**2 + (columns-centre_c)**2) / (2*width**2))
                    raw = 1000 + 160 * np.sin(t) * blob + rng.normal(0, 2, self.grid.size)
                else:
                    values = api.read_raw()
                    if values is None:
                        misses += 1
                        if misses == 1:
                            self.status.emit("Waiting for a complete frame…")
                        if misses >= 5:
                            raise RuntimeError("Sensor stopped replying. Disconnected; check the cable/port, then reconnect.")
                        self._sleep(.1)
                        continue
                    raw = self.grid.extract(values)
                    if misses:
                        self.status.emit(f"{self.port} · stream recovered")
                    misses = 0
                sequence += 1
                self._publish(sequence, raw, baseline, version)
                self._sleep(.02 if self.demo else .01)
        except Exception as exc:
            if not self.stopped():
                self.failed.emit(str(exc))
        finally:
            if api is not None:
                api.close()

import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from standalone.sensor_viewer.viewer.core import Grid, Recording, displayed_values
from standalone.sensor_viewer.viewer.worker import SensorWorker
from standalone.sensor_viewer.viewer.app import MainWindow, STYLE


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    instance.setStyleSheet(STYLE)
    return instance


def pump(app, predicate, seconds=3):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    return False


@pytest.mark.parametrize("rows,columns", [(5, 5), (8, 10), (20, 20), (1, 1)])
def test_grid_formats(rows, columns):
    grid = Grid.from_channels([columns+1, rows])
    assert grid == Grid(rows, columns)
    assert grid.extract(range(rows*columns)).tolist() == list(range(rows*columns))
    assert len(grid.extract(range(rows*(columns+1)))) == rows*columns
    with pytest.raises(ValueError):
        grid.extract(range(rows*columns-1))


@pytest.mark.parametrize("channels", [[], [25], [0, 5], [400, 400], [6, 5, 12], [2.2, 3]])
def test_grid_rejects_invalid_dimensions(channels):
    with pytest.raises(ValueError):
        Grid.from_channels(channels)


def test_difference_signs_and_zero_percentage():
    assert displayed_values([95, 110], [100, 100], "difference").tolist() == [-5, 10]
    result = displayed_values([95, 5], [100, 0], "percent")
    assert result[0] == -5
    assert np.isnan(result[1])
    with pytest.raises(ValueError):
        Grid(2, 2).extract([1, 2, np.nan, 4])


def packet(sequence=1, version=1):
    return dict(time=float(sequence), utc="2026-10-26T00:00:00+00:00", sequence=sequence,
                baseline_version=version, raw=np.array([95., 110.]), baseline=np.array([100., 100.]))


def test_recording_preserves_signs_versions_and_never_overwrites(tmp_path):
    path = tmp_path / "trial.csv"
    recorder = Recording(path, Grid(2, 1), dict(demo=True, port="DEMO"))
    recorder.append(packet())
    recorder.append(packet(2, 2))
    recorder.close(3)
    recorder.close()
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["taxel_000_difference"] == "-5.0"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["frames"] == 2 and metadata["dropped_frames"] == 3
    assert len(metadata["baseline_history"]) == 2
    assert metadata["source"]["demo"]
    with pytest.raises(FileExistsError):
        Recording(path, Grid(2, 1), {})


class FakeAPI:
    def __init__(self, *args, **kwargs):
        self.ser = SimpleNamespace(reset_input_buffer=lambda: None)
        self.closed = False
        self.reads = 0
        self.events = []

    def reconnect(self):
        return True

    def channel_check(self):
        return [6, 5]

    def read_raw(self):
        self.events.append("raw")
        self.reads += 1
        if self.reads > 5:
            self.worker.stop()
        return list(range(100, 130))

    def read_cal(self):
        self.events.append("read_cal")
        return [100]*30

    def update_cal(self):
        self.events.append("update_cal")
        return [101]*30

    def close(self):
        self.closed = True


def test_worker_uses_shared_protocol_serialises_calibration_and_closes(app):
    api = FakeAPI()
    worker = SensorWorker("COM6", 115200, api_factory=lambda *a, **kw: api)
    api.worker = worker
    worker._sleep = lambda *_: None
    worker.calibrate()
    worker.run()
    packets = worker.take_packets()
    assert len(packets) > 0 and api.closed
    assert worker.grid == Grid(5, 5)
    assert api.expected_payload_values == 30
    assert packets[0]["baseline_version"] == 2
    assert packets[0]["raw"][0] - packets[0]["baseline"][0] == -1
    assert api.events.index("read_cal") < api.events.index("update_cal")


def test_worker_reports_missing_frames_and_closes(app):
    api = FakeAPI()
    original = api.read_raw
    api.read_raw = lambda: original() if api.reads == 0 else None
    worker = SensorWorker("COM5", 115200, Grid(5, 5), api_factory=lambda *a, **kw: api)
    worker._sleep = lambda *_: None
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert api.closed
    assert any("stopped replying" in text for text in errors)


def test_worker_queue_is_bounded_and_counts_loss(app):
    worker = SensorWorker("DEMO", 0, demo=True)
    for i in range(300):
        worker._publish(i, np.zeros(25), np.zeros(25), 1)
    packets = worker.take_packets()
    assert len(packets) == 256 and worker.dropped == 44
    assert packets[-1]["sequence"] == 299


def test_cancel_before_start_never_opens_port(app):
    def unexpected_open(*args, **kwargs):
        pytest.fail("Cancelled worker must not open a port")
    worker = SensorWorker("COM8", 115200, api_factory=unexpected_open)
    worker.stop()
    worker.run()
    assert not worker.take_packets()


def test_failed_calibration_keeps_old_baseline(app):
    api = FakeAPI()
    api.update_cal = lambda: [1, 2, 3]  # Invalid length must not corrupt baseline.
    worker = SensorWorker("COM6", 115200, api_factory=lambda *a, **kw: api)
    api.worker = worker
    worker._sleep = lambda *_: None
    results = []
    worker.calibration_finished.connect(results.append)
    worker.calibrate()
    worker.run()
    assert results == [False]
    assert all(p["baseline_version"] == 1 for p in worker.take_packets())
    assert api.closed


def test_demo_ui_grid_selection_rezero_record_popout_and_shutdown(app, tmp_path):
    window = MainWindow()
    window.shape.setCurrentIndex(3)  # 20×20
    window.show()
    window.start(True)
    assert pump(app, lambda: window.packet is not None)
    assert window.grid == Grid(20, 20)
    assert window.source["demo"] and "DEMO" in window.connection_label.text()
    assert not window.port.isEnabled()
    window.plot_panel.view_mode.setCurrentIndex(3)  # Existing 2D map is still available.
    app.processEvents()
    area, cell = window.heatmap.grid_rect()
    QTest.mouseClick(window.heatmap, Qt.LeftButton,
                     pos=QPoint(int(area.x()+3.5*cell), int(area.y()+4.5*cell)))
    assert window.selected.value() == 64
    window.open_viewer()
    assert window.viewer_dialog.isVisible()
    window.recording = Recording(tmp_path/"demo.csv", window.grid, window.source)
    assert pump(app, lambda: window.recording.count >= 5)
    window.calibrate()
    assert not window.calibrate_button.isEnabled()
    assert pump(app, lambda: window.packet["baseline_version"] == 2)
    window.mode.setCurrentIndex(1)
    assert "Raw signal" in window.chart.label
    window.freeze.setChecked(True)
    old = window.chart.times.copy()
    old_seq = window.packet["sequence"]
    assert pump(app, lambda: window.packet["sequence"] > old_seq+3)
    np.testing.assert_array_equal(window.chart.times, old)
    window.close()
    assert pump(app, lambda: window.worker is None and not window.isVisible())
    assert json.loads((tmp_path/"demo.json").read_text())["frames"] >= 5


def test_imports_do_not_load_robot_or_ros():
    forbidden = [name for name in sys.modules if name == "rclpy" or name.startswith("phd.ui.")
                 or name.startswith("phd.dependence.robot_api") or name.startswith("pyvista")]
    assert not forbidden

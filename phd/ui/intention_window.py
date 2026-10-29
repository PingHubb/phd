"""Human-declared A/B/C intent workbench. Opening it has no hardware effects."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import queue
import sys
import time

import numpy as np
from PyQt5.QtCore import Qt, QProcess, QTimer, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from phd.dependence.intention_data import CLASSES, LAYOUTS, IntentionRecorder
from phd.dependence.paths import ai_resource_path
from phd.ui import components, theme


class GoalMap(QWidget):
    """Static candidate context, never the training label or inferred position."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.goals = np.array(LAYOUTS[0])
        self.shape = (8, 10)
        self.setMinimumHeight(210)
        self.setAccessibleName("Candidate targets in native sensor coordinates; not a live finger tracker")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.VIEWPORT_BG))
        pad, width, height = 38, self.width()-76, self.height()-76
        p.setPen(QPen(QColor(theme.VIEWPORT_GRID), 1))
        rows, cols = self.shape
        for c in range(cols):
            x = pad+round(c*width/(cols-1))
            p.drawLine(x, pad, x, pad+height)
        for r in range(rows):
            y = pad+round(r*height/(rows-1))
            p.drawLine(pad, y, pad+width, y)
        p.setPen(QColor(theme.VIEWPORT_MUTED))
        p.drawText(10, 18, "Native row 0 / column 0 at top left · x → columns · y ↓ rows")
        p.drawText(pad, self.height()-10, "Candidate locations only — no measured finger position")
        for label, (x, y) in zip("ABC", self.goals):
            px, py = pad+round(x*width), pad+round(y*height)
            p.setPen(QPen(QColor("#ffcb58"), 2))
            p.setBrush(QColor(theme.VIEWPORT_BG))
            p.drawEllipse(px-16, py-16, 32, 32)
            p.drawText(px-16, py-16, 32, 32, Qt.AlignCenter, label)


class IntentionWindow(QWidget):
    def __init__(self, owner):
        super().__init__(owner, Qt.Window)
        self.owner = owner
        self.setWindowTitle("Intention Learning — surface targets A / B / C")
        components.size_to_screen(self, 980, 820)
        self._closing = False
        self._close_requested = False
        self.mode = "idle"
        self.recorder = self._sensor_data = self._sensor_ref = None
        self._source = None
        self._stream_error = None
        self._queue = queue.Queue(maxsize=128)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="intention-workbench")
        self._future = self._future_kind = self._predictor = None
        self._process = None
        self._process_text = ""
        self._last_report = None
        self._active_label = "Unknown / transition"

        layout = QVBoxLayout(self)
        header = components.WorkspaceHeader(
            "Intention learning", "Predict a declared surface target from touch history and candidate positions. No robot commands.",
            eyebrow="PINGLAB / RESEARCH PROTOTYPE")
        header.set_compact(True)
        layout.addWidget(header)

        settings, settings_list = components.section("Sensor & task context", "Use the same physical sensor ID in collection and preview. All three targets are model inputs; your chosen target is not.")
        self.sensor_id = QLineEdit()
        self.sensor_id.setPlaceholderText("e.g. woven_5x5_unit1 (not its tty port)")
        settings_list.add(self._field_row("Sensor ID", self.sensor_id))
        self.layout_combo = QComboBox()
        for i in range(len(LAYOUTS)):
            self.layout_combo.addItem(f"Target layout {i+1} / 6")
        settings_list.add(components.SettingsRow("A / B / C positions", self.layout_combo))
        self.orientation = QCheckBox("I verified native row/column orientation and marked these target positions on the sensor.")
        self.orientation.setToolTip("Use Sensor signal to identify native taxels. This diagram is not automatically registered to the physical fabric.")
        settings_list.add_stacked("Physical registration", self.orientation)
        layout.addWidget(settings)
        self.shared_settings = settings

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_collect()
        self._build_train()
        self._build_preview()
        self.status = QLabel("Ready. Opening this window does not connect devices or move the robot.")
        self.status.setWordWrap(True)
        self.status.setObjectName("workspaceStatus")
        layout.addWidget(self.status)
        self.layout_combo.currentIndexChanged.connect(self._layout_changed)
        self._layout_changed()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)
        self._refresh()

    @staticmethod
    def _text(text):
        label = QLabel(text)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _field_row(title, control):
        # Paths and identifiers need usable editing width, unlike small toggles.
        row = components.SettingsRow(title, control)
        row._layout.setStretch(0, 1)
        row._layout.setStretch(1, 3)
        row._layout.setAlignment(control, Qt.AlignVCenter)
        return row

    def _page(self, title):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(theme.SPACE_MD)
        self.tabs.addTab(components.scroll_page(page), title)
        return layout

    def _path_row(self, layout, title, initial, *, checkpoint=False):
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        edit, button = QLineEdit(str(initial)), QPushButton("Browse…")
        row.addWidget(edit, 1)
        row.addWidget(button)
        def choose():
            if checkpoint:
                value, _ = QFileDialog.getOpenFileName(self, title, edit.text(), "Intention checkpoint (*.pt)")
            else:
                value = QFileDialog.getExistingDirectory(self, title, edit.text())
            if value:
                edit.setText(value)
        button.clicked.connect(choose)
        layout.addWidget(self._field_row(title, widget))
        return edit

    def _build_collect(self):
        layout = self._page("1 · Collect")
        layout.addWidget(components.SectionHeader("Declare the goal, then guide your finger", "These labels come from the person, never from a threshold or existing controller."))
        self.collect_settings = QWidget()
        form = QVBoxLayout(self.collect_settings)
        form.setContentsMargins(0, 0, 0, 0)
        self.participant = QLineEdit()
        self.participant.setPlaceholderText("Anonymous participant ID, e.g. P01")
        self.session = QLineEdit(datetime.now().strftime("%Y%m%d_session1"))
        form.addWidget(self._field_row("Participant", self.participant))
        form.addWidget(self._field_row("Session", self.session))
        self.data_root = self._path_row(form, "Recording folder", ai_resource_path("data", "intention_learning"))
        layout.addWidget(self.collect_settings)
        self.map = GoalMap()
        layout.addWidget(self.map)
        self.coordinates = self._text("")
        layout.addWidget(self.coordinates)
        layout.addWidget(self._text("1. Calibrate the streaming sensor with your hand away. 2. Start a trial. 3. Declare No current goal, A, B or C before the corresponding movement. Use Unknown during uncertain transitions. An assistant can mark changes while you move. Vary starts, paths, speed and pauses; do not follow a fixed timed cycle. Rotate layouts between trials."))
        self.record_button = QPushButton("Start intention trial")
        components.apply_variant(self.record_button, "primary")
        self.record_button.clicked.connect(self._toggle_record)
        layout.addWidget(self.record_button)
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        self.label_buttons = []
        for goal, name in [(-1, "Unknown"), (0, "No goal"), (1, "A"), (2, "B"), (3, "C")]:
            button = QPushButton(name)
            button.setCheckable(True)
            button.setAccessibleName("Declare " + name)
            button.clicked.connect(lambda checked=False, value=goal: self._declare(value))
            self.label_buttons.append((goal, button))
            row.addWidget(button)
        layout.addWidget(buttons)
        self.record_status = self._text("No trial recording. A declaration is approximate intention ground truth, not finger-position or contact ground truth.")
        layout.addWidget(self.record_status)
        layout.addStretch(1)

    def _build_train(self):
        layout = self._page("2 · Train && evaluate")
        layout.addWidget(components.SectionHeader("Test on people the model has not trained on", "Compare touch + context with touch-only and context-only baselines. No force meter or camera required."))
        self.train_settings = QWidget()
        form = QVBoxLayout(self.train_settings)
        form.setContentsMargins(0, 0, 0, 0)
        self.train_root = self._path_row(form, "Dataset folder", ai_resource_path("data", "intention_learning"))
        self.output_root = self._path_row(form, "Model runs", ai_resource_path("models", "intention_learning"))
        self.split_group = QComboBox()
        self.split_group.addItem("Hold out participants (recommended)", "participant")
        self.split_group.addItem("Hold out sessions (same-person pilot only)", "session")
        form.addWidget(components.SettingsRow("Evaluation split", self.split_group))
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 200)
        self.epochs.setValue(30)
        form.addWidget(components.SettingsRow("Maximum epochs", self.epochs))
        layout.addWidget(self.train_settings)
        layout.addWidget(self._text("Record all four known states in every participant/session group, across varied target layouts. At least three independent groups are required to run; this minimum is not enough to establish research performance. Copies or overlapping windows must never be split between training and test."))
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        self.check_button, self.train_button, self.cancel_button = QPushButton("Check dataset"), QPushButton("Train and compare"), QPushButton("Cancel job")
        self.check_button.clicked.connect(lambda: self._run_job(False))
        self.train_button.clicked.connect(lambda: self._run_job(True))
        self.cancel_button.clicked.connect(self._cancel_job)
        for b in (self.check_button, self.train_button, self.cancel_button):
            row.addWidget(b)
        layout.addWidget(row_widget)
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumBlockCount(3000)
        self.training_log.setMinimumHeight(180)
        layout.addWidget(self.training_log, 1)
        self.report_button = QPushButton("Open latest evaluation folder")
        self.report_button.clicked.connect(self._open_report)
        layout.addWidget(self.report_button)

    def _build_preview(self):
        layout = self._page("3 · Live preview")
        layout.addWidget(components.SectionHeader("See the predicted goal — without robot movement", "Probabilities are empirical model scores, not guaranteed confidence. Uncertain is different from No current goal."))
        self.checkpoint = self._path_row(layout, "Model checkpoint", "", checkpoint=True)
        self.preview_map = GoalMap()
        layout.addWidget(self.preview_map)
        self.preview_button = QPushButton("Start sensor-only intention preview")
        components.apply_variant(self.preview_button, "primary")
        self.preview_button.clicked.connect(self._toggle_preview)
        layout.addWidget(self.preview_button)
        self.prediction = self._text("Preview stopped. No target is predicted.")
        self.prediction.setAccessibleName("Predicted intention")
        layout.addWidget(self.prediction)
        self.probabilities = self._text("No current goal: —     A: —     B: —     C: —")
        layout.addWidget(self.probabilities)
        layout.addWidget(self._text("Use the layout shown above, but do not enter your chosen target. Only the sensor stream and all candidate positions go to the model. A trained model is required; old CIC and finger-tracking checkpoints are intentionally incompatible. This preview does not enable or disable other robot controllers."))
        layout.addStretch(1)

    def _layout_changed(self):
        self.goals = np.asarray(LAYOUTS[self.layout_combo.currentIndex()], np.float32)
        for widget in (self.map, self.preview_map):
            widget.goals = self.goals.copy()
            widget.update()
        self.coordinates.setText("Normalized native coordinates: " + " · ".join(f"{name}: x={x:.2f}, y={y:.2f}" for name, (x, y) in zip("ABC", self.goals)))
        # A changed layout must be physically checked again.
        self.orientation.setChecked(False)

    def _message(self, text):
        self.status.setText(str(text))

    def _refresh(self):
        idle = self.mode == "idle" and self._future is None and self._process is None and not self._close_requested
        self.shared_settings.setEnabled(idle)
        self.collect_settings.setEnabled(idle)
        self.train_settings.setEnabled(idle)
        self.record_button.setEnabled(idle or self.mode == "record")
        self.record_button.setText("Stop and save trial" if self.mode == "record" else "Start intention trial")
        for _, button in self.label_buttons:
            button.setEnabled(self.mode == "record")
        self.check_button.setEnabled(idle)
        self.train_button.setEnabled(idle)
        self.cancel_button.setEnabled(self._process is not None)
        self.preview_button.setEnabled(idle or self.mode in ("preview", "loading"))
        self.preview_button.setText("Stop intention preview" if self.mode in ("preview", "loading") else "Start sensor-only intention preview")
        self.checkpoint.setEnabled(idle)
        self.report_button.setEnabled(self._last_report is not None)

    def _motion_active(self):
        flags = ("_direct_finger_active", "_ai_direct_finger_execution_active",
                 "_console_control_active", "_console_control_sensor_active", "_console_control_sensor_v2_active",
                 "_ai_proximity_robot_motion_active", "_admittance_control_active", "centering_active", "_proximity_control_active")
        return any(bool(getattr(self.owner, key, False)) for key in flags) or getattr(self.owner.window(), "_sidebar_active_task", None) is not None

    def _sensor(self):
        if self._motion_active():
            raise ValueError("Stop active robot control / experiments first. This workbench does not send stop commands.")
        sensor = getattr(self.owner, "sensor_functions", None)
        data = getattr(sensor, "_data", None)
        if data is None or not getattr(sensor, "is_connected", False):
            raise ValueError("Build, calibrate and stream the sensor in Sensor first.")
        t = getattr(data, "frame_monotonic_sec", None)
        if t is None or not np.isfinite(t) or not 0 <= time.monotonic()-t < .35:
            raise ValueError("Sensor stream is stale. Waiting for a fresh stream is required.")
        base = np.asarray(data.calData)
        if not np.isfinite(base).all() or np.count_nonzero(base) < base.size/2:
            raise ValueError("Calibrate the sensor with your hand away first.")
        return sensor, data

    @staticmethod
    def _signature(sensor):
        return tuple(str(getattr(sensor, key, "")) for key in ("_primary_sensor_port", "current_model_name", "current_reorder_mode"))

    @staticmethod
    def _mask(sensor, shape):
        mask = getattr(sensor, "cell_zero_mask", None)
        return ~np.flipud(np.asarray(mask, bool)) if np.shape(mask) == shape else np.ones(shape, bool)

    def _attach(self, sensor, data):
        self._sensor_ref, self._sensor_data = sensor, data
        self._source = self._signature(sensor)
        self._stream_error = None
        self._last_frame = time.monotonic()
        self._queue = queue.Queue(maxsize=128)
        data.add_frame_listener(self._on_frame)
        for widget in (self.map, self.preview_map):
            widget.shape = data.rawData.shape
            widget.update()

    def _detach(self):
        if self._sensor_data is not None:
            self._sensor_data.remove_frame_listener(self._on_frame)
        self._sensor_ref = self._sensor_data = None

    def _on_frame(self, data):
        # Acquisition callback: no widgets and no heavy inference/disk work.
        try:
            frame = (float(data.frame_monotonic_sec), int(data.frame_sequence),
                     np.array(data.rawData, np.float32, copy=True), np.array(data.calData, np.float32, copy=True),
                     self._mask(self._sensor_ref, data.rawData.shape).copy())
            self._last_frame = frame[0]
            if self.recorder is not None:
                self.recorder.sensor_frame(*frame)
            elif self.mode == "preview":
                self._queue.put_nowait(frame)
        except queue.Full:
            self._stream_error = "Preview worker fell behind. Stopped rather than silently discard motion history."
        except Exception as exc:
            self._stream_error = str(exc)

    def _toggle_record(self):
        if self.mode == "record":
            self._stop_record()
            return
        if self.mode != "idle" or self._future is not None or self._process is not None:
            return
        try:
            sensor, data = self._sensor()
            source, model, order = self._signature(sensor)
            metadata = {"participant_id": self.participant.text(), "session_id": self.session.text(),
                        "sensor_id": self.sensor_id.text(), "sensor_shape": list(data.rawData.shape),
                        "sensor_model": model, "sensor_source": source, "reorder_mode": order,
                        "orientation_confirmed": self.orientation.isChecked(),
                        "raw_note": "Native buffer; acquisition filtering/masking may already apply. No physical position reference."}
            self.recorder = IntentionRecorder(Path(self.data_root.text()).expanduser(), metadata, self.goals)
            self.mode = "record"
            self._active_label = "Unknown / transition"
            for goal, button in self.label_buttons:
                button.setChecked(goal == -1)
            self._attach(sensor, data)
            self._message("Recording: declare your intended goal before moving. " + str(self.recorder.path))
        except Exception as exc:
            if self.recorder is not None:
                self._stop_record(str(exc))
            self._message(exc)
        self._refresh()

    def _declare(self, goal):
        if self.recorder is None:
            return
        try:
            self.recorder.declare(goal)
            self._active_label = CLASSES[goal] if goal >= 0 else "Unknown / transition"
            for value, button in self.label_buttons:
                button.setChecked(value == goal)
        except Exception as exc:
            self._stop_record(str(exc))
            self._message(exc)

    def _stop_record(self, reason=None):
        if self.recorder is None:
            return
        self._detach()
        recorder, self.recorder = self.recorder, None
        self.mode = "finalizing"
        def finish():
            path = recorder.close(reason)
            return path, recorder.error
        self._future, self._future_kind = self._executor.submit(finish), "save"
        self._message("Finishing disk writes…" + (" " + str(reason) if reason else ""))
        self._refresh()

    def _toggle_preview(self):
        if self.mode in ("preview", "loading"):
            self._stop_preview()
            return
        if self.mode != "idle" or self._future is not None or self._process is not None:
            return
        try:
            self._sensor()
            if not self.orientation.isChecked() or not self.sensor_id.text().strip():
                raise ValueError("Enter the physical sensor ID and confirm target orientation first.")
            path = Path(self.checkpoint.text()).expanduser()
            if not path.is_file():
                raise ValueError("Select a trained intention_preview.pt checkpoint first.")
            self.mode = "loading"
            def load():
                from phd.dependence.intention_model import IntentionPreview
                return IntentionPreview(path)
            self._future, self._future_kind = self._executor.submit(load), "load"
            self._message("Loading model in the background. No robot connection is made.")
        except Exception as exc:
            self._message(exc)
        self._refresh()

    def _stop_preview(self, reason="Preview stopped."):
        self._detach()
        self.mode = "idle"
        self._predictor = None
        self.prediction.setText(str(reason))
        self.probabilities.setText("No current goal: —     A: —     B: —     C: —")
        self._message(reason)
        self._refresh()

    def _tick(self):
        if self._closing:
            return
        if self.mode == "idle":
            # Read an already-created buffer only; never instantiate a sensor.
            data = getattr(getattr(self.owner, "sensor_functions", None), "_data", None)
            shape = np.shape(getattr(data, "rawData", None))
            if len(shape) == 2 and min(shape) >= 2:
                for widget in (self.map, self.preview_map):
                    if widget.shape != shape:
                        widget.shape = shape
                        widget.update()
                        self.orientation.setChecked(False)
        if self._sensor_data is not None:
            sensor = getattr(self.owner, "sensor_functions", None)
            reason = self._stream_error
            if (sensor is not self._sensor_ref or getattr(sensor, "_data", None) is not self._sensor_data
                    or self._signature(sensor) != self._source or not getattr(sensor, "is_connected", False)):
                reason = "Sensor source/model/order changed. Start a new trial or preview."
            elif time.monotonic()-self._last_frame > .35:
                reason = "Sensor stream became stale. Partial trials are retained but not trained."
            elif self._motion_active():
                reason = "A robot-control mode/experiment became active. Intention workbench stopped."
            if self.recorder is not None and self.recorder.error:
                reason = self.recorder.error
            if reason:
                if self.mode == "record":
                    self._stop_record(reason)
                else:
                    self._stop_preview(reason)
        if self.recorder is not None:
            self.record_status.setText(f"Recording · {time.monotonic()-self.recorder.started:.1f} s · {self.recorder.sensor_count} frames · Declared: {self._active_label}")
        if self._future is not None and self._future.done():
            future, kind = self._future, self._future_kind
            self._future = self._future_kind = None
            try:
                result = future.result()
                if kind == "save":
                    self.mode = "idle"
                    path, error = result
                    self._message(("Partial/failed trial retained: " + str(error) if error else "Trial saved") + " — " + str(path))
                    self.record_status.setText("Recording stopped. " + str(path))
                elif kind == "load" and self.mode == "loading":
                    sensor, data = self._sensor()
                    self._predictor = result
                    self.mode = "preview"
                    self._attach(sensor, data)
                    self._message("Sensor-only preview. No chosen-target labels are supplied to the model.")
                    self.prediction.setText("Warming up on fresh sensor frames…")
                elif kind == "infer" and self.mode == "preview":
                    if result["valid"]:
                        # Never display a delayed prediction as live.
                        if time.monotonic()-result["timestamp"] > .35:
                            self._stop_preview("Prediction became stale; restart when acquisition is responsive.")
                        else:
                            self.prediction.setText("Predicted goal: " + result["label"])
                            self.probabilities.setText("    ".join(f"{name}: {p:.1%}" for name, p in zip(CLASSES, result["probabilities"])))
                    else:
                        self.prediction.setText(result["reason"])
                        self.probabilities.setText("No current goal: —     A: —     B: —     C: —")
            except Exception as exc:
                if self.mode in ("preview", "loading"):
                    self._stop_preview(str(exc))
                else:
                    self.mode = "idle"
                    self._message(exc)
            self._refresh()
        if self.mode == "preview" and self._future is None:
            frames = []
            while not self._queue.empty():
                try:
                    frames.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            if frames:
                self._future = self._executor.submit(self._predictor.update_many, frames,
                    sensor_id=self.sensor_id.text(), sensor_model=self._source[1], reorder_mode=self._source[2], goals=self.goals.copy())
                self._future_kind = "infer"

    def _run_job(self, train):
        if self.mode != "idle" or self._future is not None or self._process is not None:
            return
        script = Path(__file__).resolve().parents[1]/"script"/"train_intention.py"
        args = [str(script), "train" if train else "check", str(Path(self.train_root.text()).expanduser())]
        if train:
            args += ["--output", str(Path(self.output_root.text()).expanduser()), "--group-by", self.split_group.currentData(), "--epochs", str(self.epochs.value())]
        self._process_text = ""
        self.training_log.clear()
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_process)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        self._process = process
        process.start(sys.executable, ["-u", *args])
        self._message("Checking dataset…" if not train else "Training three comparison models in a separate process…")
        self._refresh()

    def _read_process(self):
        if self._process is not None:
            text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
            self._process_text += text
            self.training_log.insertPlainText(text)
            self.training_log.verticalScrollBar().setValue(self.training_log.verticalScrollBar().maximum())

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self._process_finished(-1, QProcess.CrashExit)

    def _process_finished(self, code, status):
        self._read_process()
        process, self._process = self._process, None
        if process is not None:
            process.deleteLater()
        if code == 0 and status == QProcess.NormalExit:
            for line in self._process_text.splitlines():
                if line.startswith("COMPLETED="):
                    self._last_report = Path(line.partition("=")[2])
                    self.checkpoint.setText(str(self._last_report/"intention_preview.pt"))
            self._message("Job completed. Read the log and evaluation reports before drawing conclusions.")
        else:
            self._message("Job stopped or refused invalid data. See the log; previous recordings/models were preserved.")
        self._refresh()

    def _cancel_job(self):
        if self._process is not None:
            process = self._process
            process.terminate()
            def kill():
                if self._process is process and process.state() != QProcess.NotRunning:
                    process.kill()
            QTimer.singleShot(2000, kill)

    def _open_report(self):
        if self._last_report is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_report)))

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        self._close_requested = True
        if self.recorder is not None:
            self._stop_record()
        if self.mode in ("preview", "loading"):
            self._stop_preview()
        self._cancel_job()
        if self._future is not None or self._process is not None:
            event.ignore()
            self._message("Finishing writes / worker shutdown before closing…")
            QTimer.singleShot(100, self.close)
            return
        self._closing = True
        self.timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        event.accept()

    def shutdown(self):
        if self._closing:
            return
        if self.recorder is not None:
            self._stop_record()
        if self.mode in ("preview", "loading"):
            self._stop_preview()
        self._cancel_job()
        if self._process is not None:
            process = self._process
            if not process.waitForFinished(1000):
                process.kill()
                process.waitForFinished(1000)
        if self._future is not None:
            try:
                self._future.result(timeout=10)
            except Exception as exc:
                print(f"[Intention] Shutdown: completed raw chunks retained; {exc}")
        self._closing = True
        self.timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.close()

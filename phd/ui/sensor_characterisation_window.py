"""Fixture-guided research workbench. Never commands or stops a robot."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import json
from pathlib import Path
import time

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from phd.dependence.paths import experiment_path
from phd.dependence.touch_tracking_data import save_json
from phd.dependence.characterisation_study import (
    PROTOCOLS, CharacterisationRecorder, analyse_study, compare_studies,
    evaluate_prediction_csv, export_study_report,
    import_event_csv, finite, study_plot_data,
)
from phd.ui import components


class SensorCharacterisationWindow(QDialog):
    """Recording attaches to completed native frames, not a GUI polling loop."""
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.setWindowTitle("Sensor Characterisation — fixture-guided, no robot commands")
        components.size_to_screen(self, 1050, 800)
        self.recorder = None
        self._sensor = self._data = self._signature = None
        self._stream_error = None
        self._future = None
        self._future_kind = ""
        self._closing = False
        self._last_path = None
        self._report = None
        self._canvas = None
        self._plot_packet = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="characterisation")
        layout = QVBoxLayout(self)
        self.status = QLabel("Ready. Offline analysis needs no sensor, robot or force meter.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.setup = QWidget()
        form = components.SettingsForm()
        setup_layout = QVBoxLayout(self.setup)
        setup_layout.addLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.setup)
        self.tabs.addTab(scroll, "1 · Setup")
        self.protocol = QComboBox()
        self.protocol.setObjectName("characterisationStudyProtocol")
        for key, (label, _description) in PROTOCOLS.items():
            self.protocol.addItem(label, key)
        form.addRow("Experiment", self.protocol)
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        setup_layout.addWidget(self.instructions)
        self.protocol.currentIndexChanged.connect(self._protocol_changed)
        self.specimen = QLineEdit()
        self.specimen.setObjectName("studySpecimenId")
        self.specimen.setPlaceholderText("Stable physical specimen ID, not ttyACM number")
        self.session = QLineEdit(datetime.now().strftime("%Y%m%d_session1"))
        self.mounting = QLineEdit()
        self.mounting.setPlaceholderText("e.g. flat_mount1 or cylinder_R50_remount2")
        self.participant = QLineEdit("operator")
        self.robot_state = QComboBox()
        self.robot_state.addItems(["Off / disconnected", "Powered, stationary", "Externally moving, supervised"])
        for name, widget in (("Physical specimen ID", self.specimen), ("Session / day ID", self.session), ("Mounting / remount ID", self.mounting), ("Anonymous participant ID", self.participant), ("Declared robot condition", self.robot_state)):
            form.addRow(name, widget)
        self.baseline = QDoubleSpinBox()
        self.baseline.setRange(2, 60)
        self.baseline.setValue(5)
        self.baseline.setSuffix(" s")
        form.addRow("Initial untouched baseline", self.baseline)
        self.duration = QSpinBox()
        self.duration.setRange(10, 300)
        self.duration.setValue(60)
        self.duration.setSuffix(" s")
        form.addRow("Recording duration, including baseline", self.duration)
        self.orientation = QCheckBox("I checked Sensor signal grid orientation and taxel numbering")
        self.orientation.setObjectName("studyOrientationConfirmed")
        self.clear_baseline = QCheckBox("The hand/probe will remain clear throughout the initial baseline")
        form.addRow("Orientation", self.orientation)
        form.addRow("Baseline preflight", self.clear_baseline)
        self.root = QLineEdit(experiment_path("sensor_characterisation"))
        browse = QPushButton("Choose output folder…")
        browse.clicked.connect(self._choose_root)
        form.addRow("All related files under one trial folder", self.root)
        form.addRow("", browse)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("Material, grounding, temperature/humidity, mounting tension, reference apparatus…")
        form.addRow("Setup notes", self.notes)
        self._protocol_changed()

        self.capture = QWidget()
        capture_layout = QVBoxLayout(self.capture)
        capture_scroll = QScrollArea()
        capture_scroll.setWidgetResizable(True)
        capture_scroll.setWidget(self.capture)
        self.tabs.addTab(capture_scroll, "2 · Record conditions")
        caution = QLabel("Position and speed fields are independent declarations, not sensor predictions. Move/load the probe using your existing reviewed setup; this workbench never sends robot motion. One unchanged mounting per trial.")
        caution.setWordWrap(True)
        capture_layout.addWidget(caution)
        conditions = components.SettingsForm()
        capture_layout.addLayout(conditions)
        self.contact_count = QSpinBox()
        self.contact_count.setRange(0, 5)
        self.contact_count.setValue(1)
        self.gesture = QLineEdit()
        self.gesture.setPlaceholderText("Exact class label for optional prediction evaluation")
        self.diameter = QLineEdit()
        self.speed = QLineEdit()
        self.diameter.setPlaceholderText("Unknown if blank")
        self.speed.setPlaceholderText("Nominal/declarative; blank = unknown, 0 = stationary")
        self.position_source = QComboBox()
        for label, value in (("Unknown", "unknown"), ("Operator estimate (exploratory)", "operator_estimate"), ("Independent measured fixture", "measured_fixture"), ("Stationary robot TCP (flat specimen)", "stationary_robot_tcp")):
            self.position_source.addItem(label, value)
        self.xy = [QLineEdit() for _ in range(4)]
        for field in self.xy:
            field.setPlaceholderText("Unknown if blank; SAME robot-base reference frame")
        for label, widget in (("Declared simultaneous contacts", self.contact_count), ("Gesture label", self.gesture), ("Contact/probe diameter (mm)", self.diameter), ("Nominal tangential speed (mm/s)", self.speed), ("XY reference source", self.position_source), ("First contact X (mm)", self.xy[0]), ("First contact Y (mm)", self.xy[1]), ("Second contact X (mm)", self.xy[2]), ("Second contact Y (mm)", self.xy[3])):
            conditions.addRow(label, widget)
        tcp = QPushButton("Read current stationary TCP XY — no movement")
        tcp.clicked.connect(self._copy_tcp)
        conditions.addRow("Single probe reference", tcp)
        self.tcp_confirmed = QCheckBox("Probe is stationary; TCP is registered to this flat specimen's contact position")
        conditions.addRow("Robot XY reference preflight", self.tcp_confirmed)
        buttons = QHBoxLayout()
        self.start = QPushButton("Start native recording")
        self.start.setObjectName("startCharacterisationRecording")
        self.stop = QPushButton("Stop and save")
        self.start.clicked.connect(self._start)
        self.stop.clicked.connect(self._stop)
        buttons.addWidget(self.start)
        buttons.addWidget(self.stop)
        capture_layout.addLayout(buttons)
        self.mark_buttons = []
        for number, (label, kind) in enumerate((("Begin contact / gesture", "condition_start"), ("Release / no contact", "release"), ("Annotate contact onset", "contact_reference"), ("Annotate command", "command_reference"), ("Annotate motion", "motion_reference"))):
            if number in (0, 2):
                events = QHBoxLayout()
                capture_layout.addLayout(events)
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, event=kind: self._mark(event))
            self.mark_buttons.append(button)
            events.addWidget(button)
        self.capture_info = QLabel("First record the untouched baseline, then mark each condition. Manual onset buttons do NOT measure physical system latency.")
        self.capture_info.setWordWrap(True)
        capture_layout.addWidget(self.capture_info)
        capture_layout.addStretch()

        analysis = QWidget()
        analysis_layout = QVBoxLayout(analysis)
        self.tabs.addTab(analysis, "3 · Analyse saved data")
        bar = QHBoxLayout()
        for label, callback in (("Open saved trial…", self._open_trial), ("Compare trials…", self._compare), ("Import independent event CSV…", self._import_events), ("Evaluate prediction CSV…", self._evaluate_predictions)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            bar.addWidget(button)
        analysis_layout.addLayout(bar)
        self.analysis_buttons = [bar.itemAt(i).widget() for i in range(bar.count())]
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlainText("Open a saved trial without connecting devices. Analysis JSON, condition CSV and plots are exported into that same trial folder.\n\nEvent CSV: t_s,kind,uncertainty_ms (independently aligned recording-relative host seconds).\nPrediction CSV: t_s,predicted_label. The import cannot verify training/test provenance.")
        analysis_layout.addWidget(self.result, 1)
        self.plot_page = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_page)
        self.plot_condition = QComboBox()
        self.plot_condition.setAccessibleName("Condition shown in spatial response graph")
        self.plot_condition.currentIndexChanged.connect(self._redraw_condition)
        self.plot_layout.addWidget(self.plot_condition)
        self.tabs.addTab(self.plot_page, "4 · Graphs")
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self._update_controls()

    def _protocol_changed(self):
        self.instructions.setText(PROTOCOLS[self.protocol.currentData()][1])

    def _choose_root(self):
        path = QFileDialog.getExistingDirectory(self, "Parent folder for characterisation trials", self.root.text())
        if path:
            self.root.setText(path)

    def _motion_active(self):
        if getattr(self.host, "_sidebar_active_task", None) is not None:
            return True
        owner = getattr(self.host, "ui_ros", None)
        return any(bool(getattr(owner, flag, False)) for flag in (
            "_direct_finger_active", "_console_control_active",
            "_ai_proximity_robot_motion_active", "_admittance_control_active", "centering_active",
        ))

    @staticmethod
    def _sensor_signature(sensor, data):
        return (id(data), *(str(getattr(sensor, key, "")) for key in ("_primary_sensor_port", "current_model_name", "current_reorder_mode")))

    def _start(self):
        if self.recorder is not None or self._future is not None:
            return
        try:
            if self._motion_active():
                raise ValueError("Stop the active experiment/controller first. Opening this window never interrupts it.")
            owner = getattr(self.host, "ui_ros", None)
            sensor = getattr(owner, "sensor_functions", None)
            data = getattr(sensor, "_data", None)
            stamp = finite(getattr(data, "frame_monotonic_sec", None))
            if not getattr(sensor, "is_connected", False) or stamp is None or not 0 <= time.monotonic()-stamp < .5:
                raise ValueError("Build/calibrate and stream a fresh tactile sensor first. A force meter is optional for sensor-only recordings.")
            if self.duration.value() <= self.baseline.value()+2:
                raise ValueError("Recording duration must exceed the baseline by more than two seconds.")
            self._signature = self._sensor_signature(sensor, data)
            metadata = {
                "protocol": self.protocol.currentData(), "specimen_id": self.specimen.text().strip(),
                "session_id": self.session.text().strip(), "mounting_id": self.mounting.text().strip(),
                "participant_id": self.participant.text().strip(), "sensor_shape": list(data.rawData.shape),
                "orientation_confirmed": self.orientation.isChecked(), "baseline_confirmed": self.clear_baseline.isChecked(),
                "sensor_source": self._signature[1], "sensor_model": self._signature[2], "reorder_mode": self._signature[3],
                "robot_condition": self.robot_state.currentText(), "notes": self.notes.text(),
            }
            if not str(self.root.text()).strip():
                raise ValueError("Choose an output parent folder.")
            self.recorder = CharacterisationRecorder(Path(self.root.text()).expanduser(), metadata, baseline_sec=self.baseline.value())
            self._clear_analysis()
            self._sensor, self._data = sensor, data
            self._stream_error = None
            data.add_frame_listener(self._on_frame)
            self.tabs.setCurrentIndex(1)
            self.status.setText(f"Recording untouched baseline → {self.recorder.path}")
        except Exception as exc:
            if self.recorder is not None:
                self._stop(reason=f"Recording could not attach: {exc}")
            self._error(exc)
        self._update_controls()

    def _on_frame(self, data):
        # This callback may run on the acquisition thread. Never touch Qt here.
        try:
            recorder = self.recorder
            if recorder is None:
                return
            if self._sensor_signature(self._sensor, data) != self._signature:
                self._stream_error = "Sensor source, model, ordering or buffer changed; partial recording retained."
                return
            snapshot = getattr(data, "completed_frame_snapshot", None)
            if snapshot is None:
                self._stream_error = "Completed atomic sensor snapshots are required."
                return
            mask = getattr(self._sensor, "cell_zero_mask", None)
            valid = ~np.flipud(np.asarray(mask, bool)) if np.shape(mask) == snapshot.rawData.shape else np.ones(snapshot.rawData.shape, bool)
            recorder.sensor_frame(
                float(snapshot.frame_monotonic_sec), int(snapshot.frame_sequence), snapshot.rawData,
                snapshot.calData, valid, filtered_signed=np.flipud(snapshot.diffDataAve),
            )
        except Exception as exc:
            self._stream_error = str(exc)

    def _condition(self):
        if self.position_source.currentData() == "stationary_robot_tcp" and not self.tcp_confirmed.isChecked():
            raise ValueError("Confirm the stationary, registered flat-specimen TCP reference, or choose a different reference source.")
        return {
            "contact_count": self.contact_count.value(), "gesture_label": self.gesture.text().strip(),
            "mounting_id": self.mounting.text().strip(), "probe_diameter_mm": self.diameter.text().strip(),
            "nominal_speed_mm_s": self.speed.text().strip(), "position_source": self.position_source.currentData(),
            **dict(zip(("x_mm", "y_mm", "second_x_mm", "second_y_mm"), [field.text().strip() for field in self.xy])),
        }

    def _mark(self, kind):
        try:
            if self.recorder is not None:
                event = self.recorder.mark(kind, self._condition())
                self.capture_info.setText(f"{kind} at {event['t']:.3f} s. Operator annotation; not independently measured onset.")
        except Exception as exc:
            self._error(exc)

    def _copy_tcp(self):
        try:
            if self._motion_active() or not self.tcp_confirmed.isChecked():
                raise ValueError("Stop automated motion and confirm the stationary, registered flat-specimen TCP reference first.")
            api = getattr(getattr(self.host, "ui_ros", None), "robot_api", None)
            stamp = finite(getattr(api, "current_feedback_state_time", None))
            if stamp is None or not 0 <= time.monotonic()-stamp <= 1:
                raise ValueError("Fresh robot feedback is required. No motion command is sent.")
            position, _quaternion = api.get_current_tool_position()
            if len(position) != 3 or not np.isfinite(position).all():
                raise ValueError("Invalid TCP position.")
            self.xy[0].setText(f"{position[0]*1000:.6f}")
            self.xy[1].setText(f"{position[1]*1000:.6f}")
            self.position_source.setCurrentIndex(self.position_source.findData("stationary_robot_tcp"))
            self.capture_info.setText("Current TCP XY copied. Confirm the probe is stationary and the flat specimen/reference frame is registered; this is not a measured finger location.")
        except Exception as exc:
            self._error(exc)

    def _detach(self):
        if self._data is not None:
            self._data.remove_frame_listener(self._on_frame)
        self._data = self._sensor = None

    def _stop(self, _checked=False, *, reason=None):
        if self.recorder is None:
            return
        self._detach()
        recorder, self.recorder = self.recorder, None
        self._last_path = recorder.path

        def finish():
            recorder.close(reason)
            if recorder.error:
                raise ValueError(f"Partial recording retained at {recorder.path}: {recorder.error}")
            report = analyse_study(recorder.path)
            export_study_report(recorder.path, report)
            return recorder.path, report, study_plot_data(recorder.path, report)

        self._submit("analyse", finish)
        self.status.setText("Saving native chunks and analysing in the background…")

    def _submit(self, kind, function):
        if self._future is not None:
            raise ValueError("Wait for the current analysis/save to finish.")
        self._future_kind = kind
        self._future = self._executor.submit(function)
        self._update_controls()

    def _open_trial(self):
        path = QFileDialog.getExistingDirectory(self, "Select a completed Sensor Characterisation trial", self.root.text())
        if path:
            self._analyse_path(Path(path))

    def _analyse_path(self, path):
        self._clear_analysis()
        def run():
            report = analyse_study(path)
            export_study_report(path, report)
            return path, report, study_plot_data(path, report)
        self._submit("analyse", run)

    def _clear_analysis(self):
        # Never export a previous recording's plot into a new/failed trial.
        self._last_path = self._report = self._plot_packet = None
        self.plot_condition.clear()
        while self.plot_layout.count() > 1:
            widget = self.plot_layout.takeAt(1).widget()
            if widget is not None:
                widget.deleteLater()
        self._canvas = None

    def _compare(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select characterisation.json from separate trials of the SAME specimen", self.root.text(), "Trial manifests (characterisation.json)")
        if paths:
            unique = list(dict.fromkeys(str(Path(path).parent.resolve()) for path in paths))
            self._submit("compare", lambda: compare_studies([analyse_study(path) for path in unique]))

    def _import_events(self):
        if self._last_path is None:
            self._error("Open a completed trial first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "External events: t_s,kind,uncertainty_ms", "", "CSV (*.csv)")
        if not path:
            return
        choice = QMessageBox.question(self, "Independent timing reference", "Have these external device events been independently aligned to this recording's relative host-monotonic clock? Manual button times, unaligned device/video time or inference runtime alone cannot measure end-to-end latency.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if choice != QMessageBox.Yes:
            return
        target = self._last_path
        def run():
            import_event_csv(target, path, clock_alignment_confirmed=True)
            report = analyse_study(target)
            export_study_report(target, report)
            return target, report, study_plot_data(target, report)
        self._submit("analyse", run)

    def _evaluate_predictions(self):
        if self._last_path is None:
            self._error("Open a completed, independently labelled trial first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Predictions: t_s,predicted_label", "", "CSV (*.csv)")
        if path:
            target = self._last_path
            def run():
                report = evaluate_prediction_csv(target, path)
                save_json(target / "prediction_evaluation.json", report)
                return report
            self._submit("predictions", run)

    def _poll(self):
        recorder = self.recorder
        if recorder is not None:
            stamp = finite(getattr(self._data, "frame_monotonic_sec", None))
            reason = self._stream_error or recorder.error
            if self._motion_active():
                reason = reason or "Another automated experiment/controller became active; recording stopped without sending robot commands."
            if stamp is None or time.monotonic()-stamp > 1:
                reason = reason or "Tactile stream became stale."
            elapsed = time.monotonic()-recorder.started
            if reason or elapsed >= self.duration.value():
                self._stop(reason=reason)
                return
            owner = getattr(self.host, "ui_ros", None)
            api = getattr(owner, "robot_api", None)
            reader = getattr(owner, "_force_meter_thread", None)
            reader_running = reader is not None and callable(getattr(reader, "isRunning", None)) and reader.isRunning()
            reference = {
                "force_n": getattr(owner, "_force_meter_last_raw_newtons", None) if reader_running else None,
                "force_source_monotonic": getattr(owner, "_force_meter_last_sample_monotonic", None),
                "robot_source_monotonic": getattr(api, "current_feedback_state_time", None),
            }
            try:
                position, _quaternion = api.get_current_tool_position()
                reference.update(zip(("x_m", "y_m", "z_m"), position))
            except (AttributeError, TypeError, ValueError):
                pass
            recorder.reference(reference)
            phase = "untouched baseline" if elapsed < recorder.baseline_sec else "mark your conditions"
            self.status.setText(f"{elapsed:.1f} s · {recorder.sensor_count} native frames · {phase} · {recorder.path.name}")
            self._update_controls()
        if self._future is not None and self._future.done():
            future, kind = self._future, self._future_kind
            self._future = None
            try:
                result = future.result()
                if kind == "analyse":
                    self._last_path, self._report, self._plot_packet = result
                    self.result.setPlainText(json.dumps(result[1], indent=2, ensure_ascii=False))
                    self.plot_condition.blockSignals(True)
                    self.plot_condition.clear()
                    for index, segment in enumerate(self._report["segments"]):
                        label = segment["condition"].get("gesture_label") or "contact / condition"
                        self.plot_condition.addItem(f"{index+1}: {label} · {segment['start_s']:.1f}–{segment['end_s']:.1f} s")
                    self.plot_condition.setEnabled(bool(self._report["segments"]))
                    self.plot_condition.blockSignals(False)
                    self._draw(result[0], result[1], self._plot_packet)
                    self.status.setText(f"Saved/analysed: {result[0]}")
                    self.tabs.setCurrentIndex(2)
                else:
                    self.result.setPlainText(json.dumps(result, indent=2, ensure_ascii=False))
                    self.tabs.setCurrentIndex(2)
            except Exception as exc:
                self._error(exc)
            self._update_controls()
            if self._closing:
                self.close()

    def _redraw_condition(self, _index):
        if self._plot_packet is not None and self._report is not None:
            self._draw(self._last_path, self._report, self._plot_packet)

    def _draw(self, path, report, packet):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        figure = Figure(figsize=(9, 6), tight_layout=True)
        timeline = figure.add_subplot(211)
        timeline.plot(packet["t"], packet["peak"], label="Instantaneous signed difference: peak absolute baseline change")
        if "filtered_peak" in packet:
            timeline.plot(packet["t"], packet["filtered_peak"], label="Acquisition moving average: peak change")
        timeline.set(xlabel="Host-buffer time since recording start (s)", ylabel="Peak change (sensor units)", title="Recorded sensor response — display-decimated; full native samples retained")
        timeline.legend(fontsize=7)
        spatial = figure.add_subplot(212)
        if report["segments"]:
            segment_index = max(0, self.plot_condition.currentIndex())
            segment = report["segments"][segment_index]
            timeline.axvspan(segment["start_s"], segment["end_s"], alpha=.12, color="green")
            field = np.array(segment["signed_response_by_taxel"]).reshape(packet["shape"][1], packet["shape"][0]).T
            valid = np.array(report["valid_taxels"]).reshape(packet["shape"][1], packet["shape"][0]).T
            field = np.ma.masked_where(~valid, field)
            largest = max(float(np.max(np.abs(field))), 1e-9)
            picture = spatial.imshow(field, cmap="coolwarm", vmin=-largest, vmax=largest, origin="upper")
            figure.colorbar(picture, ax=spatial, label="Signed baseline-relative change (sensor units)")
            spatial.set(xlabel="Display column", ylabel="Display row", title=f"Condition {segment_index+1}: signed spatial response (not pressure)")
        else:
            spatial.hist(np.asarray(report["baseline_robust_sigma_units"]), bins=25)
            spatial.set(xlabel="Robust baseline noise (sensor units)", ylabel="Taxel count", title="Untouched baseline noise across taxels")
        while self.plot_layout.count() > 1:
            widget = self.plot_layout.takeAt(1).widget()
            if widget is not None:
                widget.deleteLater()
        self._canvas = FigureCanvasQTAgg(figure)
        self.plot_layout.addWidget(NavigationToolbar2QT(self._canvas, self))
        self.plot_layout.addWidget(self._canvas, 1)
        self._canvas.draw_idle()
        suffix = f"condition_{max(0, self.plot_condition.currentIndex())+1:03d}" if report["segments"] else "baseline"
        figure.savefig(Path(path)/f"sensor_response_{suffix}.png", dpi=160)

    def _update_controls(self):
        busy = self.recorder is not None or self._future is not None
        self.setup.setEnabled(not busy)
        self.start.setEnabled(not busy)
        self.stop.setEnabled(self.recorder is not None)
        ready = self.recorder is not None and time.monotonic()-self.recorder.started >= self.recorder.baseline_sec and self.recorder.sensor_count >= 20
        for button in self.mark_buttons:
            button.setEnabled(ready)
        for button in self.analysis_buttons:
            button.setEnabled(not busy)

    def _error(self, error):
        self.status.setText(str(error))
        self.result.setPlainText(str(error))

    def closeEvent(self, event):
        if self.recorder is not None:
            self._closing = True
            self._stop(reason="Window closed before trial completion; partial data retained.")
            event.ignore()
            return
        if self._future is not None:
            self._closing = True
            self.status.setText("Finishing the background save before closing…")
            event.ignore()
            return
        self._detach()
        self.timer.stop()
        self._executor.shutdown(wait=False)
        event.accept()

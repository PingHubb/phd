"""Sensor-only, whole-gesture recognition. No robot APIs or command mappings."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import time

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from phd.dependence.live_gesture_recognition import GestureTrialSegmenter
from phd.dependence.paths import ai_resource_path
from phd.ui import components, theme
from phd.ui.gesture_feedback import GestureResultCard, gesture_label


def default_gesture_checkpoint():
    return ai_resource_path("models", "robot_dog_gestures", "20261020_10class_v1",
                            "best_gesture_model.pt")


def _load_predictor(path):
    # Lazy import and CPU loading keep page construction and the GUI responsive.
    from phd.dependence.robot_dog_gesture_model import GesturePredictor, PREPROCESS
    predictor = GesturePredictor(path, device="cpu")
    if predictor.bundle["preprocess"] != PREPROCESS:
        raise ValueError("This checkpoint needs a different live preprocessing version.")
    return predictor


def _predict_saved(path, trial):
    predictor = _load_predictor(path)
    return predictor, predictor.predict_file(trial)


class GestureRecognitionPanel(QWidget):
    """Subscribe only while explicitly started; classify off the GUI thread.

    Sensor callbacks copy frames to a bounded queue. A GUI timer consumes them
    and polls one worker future. Stopping invalidates results without destroying
    a running Qt thread or allowing a late result to restart acquisition.
    """

    def __init__(self, sensor_getter, parent=None):
        super().__init__(parent)
        self.sensor_getter = sensor_getter
        self.checkpoint = default_gesture_checkpoint()
        self.predictor = None
        self.active = False
        self._closed = False
        self._sensor = self._data = None
        self._future = None
        self._operation = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gesture-preview")
        self._queue = queue.Queue(maxsize=128)
        self._stream_error = None
        self.last_result = None
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def _label(self, text, role="body"):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.PlainText)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        label.setStyleSheet(theme.type_style(role))
        return label

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_MD)
        layout.addWidget(components.SectionHeader(
            "Touch gesture recognition",
            "Touch, draw, then lift your finger. Sensor only — no robot commands.",
        ))
        self.model_label = self._label("10 gestures · CNN-TCN · 8 × 8 sensor", "headline")
        layout.addWidget(self.model_label)
        layout.addWidget(self._label(
            "Use the same sensor orientation and averaging as collection. "
            "Calibrate with your hand away before starting.", "caption"))
        controls = QHBoxLayout()
        self.start_button = QPushButton("Start recognition")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        components.apply_variant(self.start_button, "primary")
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        controls.addWidget(self.start_button, 1)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)
        self.status = components.StatusPill("Stopped")
        layout.addWidget(self.status, 0, Qt.AlignLeft)
        self.status_detail = self._label("Start the sensor stream, then press Start recognition.")
        layout.addWidget(self.status_detail)
        self.source_label = self._label("Source: not listening", "caption")
        layout.addWidget(self.source_label)

        self.result_card = GestureResultCard("Last completed gesture", "Results appear after release, not during the motion.")
        self.result_label = self.result_card.result_label
        self.confidence_label = self.result_card.confidence_label
        self.confidence_bar = self.result_card.confidence_bar
        layout.addWidget(self.result_card)

        details = components.CollapsibleGroup("Details · scores, recent gestures and model")
        self.scores_label = self._label("No scores yet.", "caption")
        details.add(self.scores_label)
        self.history = QListWidget()
        self.history.setAccessibleName("Recent gesture predictions")
        self.history.setMaximumHeight(130)
        self.history.setMinimumWidth(0)
        self.history.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        details.add(self.history)
        self.path_label = self._label(self.checkpoint, "caption")
        details.add(self.path_label)
        model_controls = QWidget()
        row = QHBoxLayout(model_controls)
        row.setContentsMargins(0, 0, 0, 0)
        self.browse_button = QPushButton("Choose model…")
        self.default_button = QPushButton("Use trained model")
        self.replay_button = QPushButton("Test saved trial…")
        for button in (self.browse_button, self.default_button):
            row.addWidget(button)
        self.browse_button.clicked.connect(self._browse_model)
        self.default_button.clicked.connect(lambda: self.set_checkpoint(default_gesture_checkpoint()))
        self.replay_button.clicked.connect(self._browse_trial)
        details.add(model_controls)
        details.add(self.replay_button)
        details.add(self._label(
            "A result needs ≥80% confidence and a ≥15-point lead; otherwise it is "
            "Uncertain. This is not validated detection of unknown gestures. "
            "Release confirmation: 3 fresh frames over ≥0.12 s. "
            "Gestures longer than 30 s / 2,000 frames are discarded. "
            "Test saved trial works offline without a sensor, camera or robot.", "caption"))
        layout.addWidget(details)
        self.details = details

    def _set_status(self, title, detail, state="idle"):
        if self.status.text() != title or self.status.state() != state:
            self.status.set_state(state, title)
        if self.status_detail.text() != detail:
            self.status_detail.setText(detail)

    def _buttons(self, running):
        self.start_button.setEnabled(not running and not self._closed)
        self.stop_button.setEnabled(running)
        for button in (self.browse_button, self.default_button, self.replay_button):
            button.setEnabled(not running and not self._closed)

    def set_checkpoint(self, path):
        if self.active or self._future is not None:
            return
        self.checkpoint = str(path)
        self.predictor = None
        self.path_label.setText(self.checkpoint)
        self.model_label.setText("Trained 10-gesture model" if self.checkpoint == default_gesture_checkpoint()
                                 else "Selected model · validated when started")
        self._clear_result()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose gesture checkpoint", str(Path(self.checkpoint).parent), "Gesture checkpoint (*.pt)")
        if path:
            self.set_checkpoint(path)

    def _browse_trial(self):
        path, _ = QFileDialog.getOpenFileName(self, "Test a saved touch trial", ai_resource_path("data", "tactile_recordings"), "Touch recording (*.npz)")
        if path:
            self.test_saved_trial(path)

    def _clear_result(self):
        self.last_result = None
        self.result_label.setStyleSheet(theme.type_style("title"))
        self.result_label.setText("Waiting for a gesture")
        self.confidence_label.setText("Results appear after release, not during the motion.")
        self.confidence_bar.setValue(0)
        self.scores_label.setText("No scores yet.")

    def test_saved_trial(self, path):
        if self.active or self._future is not None or self._closed:
            return
        self._clear_result()
        self._buttons(True)
        self._operation = "replay"
        self.source_label.setText("Saved-trial replay: " + Path(path).name)
        self._set_status("Analysing", "Testing a saved recording offline. No hardware connected by this page.", "info")
        self._future = self._executor.submit(_predict_saved, self.checkpoint, str(path))
        self.timer.start()

    @staticmethod
    def _signature(sensor, data):
        return (id(sensor), id(data), getattr(sensor, "_primary_sensor_port", None),
                getattr(sensor, "current_effective_reorder_logic", None),
                getattr(data, "windowSize", None))

    def _live_sensor(self):
        sensor = self.sensor_getter()
        if sensor is None or not getattr(sensor, "is_connected", False):
            raise ValueError("Start the sensor stream in Sensor first. No robot or force meter is needed.")
        if getattr(sensor, "_sensor_calibration_in_progress", False):
            raise ValueError("Wait for sensor calibration to finish, then restart recognition.")
        data = getattr(sensor, "_data", None)
        stamp = getattr(data, "frame_monotonic_sec", None)
        if stamp is None or not 0 <= time.monotonic() - stamp <= GestureTrialSegmenter.MAX_GAP_SECONDS:
            raise ValueError("No fresh sensor frames. Check streaming, then restart recognition.")
        if not callable(getattr(data, "add_frame_listener", None)):
            raise ValueError("This sensor source does not support live frame capture.")
        return sensor, data

    def start(self):
        if self.active or self._future is not None or self._closed:
            return
        try:
            sensor, data = self._live_sensor()
            # Bind before loading: a different stream must never be silently adopted.
            self._binding = self._signature(sensor, data)
            self._calibration = np.array(data.calData, copy=True)
        except Exception as exc:
            self._set_status("Cannot start", str(exc), "error")
            return
        self._clear_result()
        self._buttons(True)
        self._operation = "load"
        self._set_status("Loading model", "Preparing CPU inference; the UI remains available.", "info")
        self._future = self._executor.submit(_load_predictor, self.checkpoint)
        self.timer.start()

    def _attach(self):
        sensor, data = self._live_sensor()
        if self._signature(sensor, data) != self._binding or not np.array_equal(data.calData, self._calibration):
            raise ValueError("Sensor or calibration changed. Restart recognition with the intended sensor.")
        shape = tuple(self.predictor.bundle["sensor_shape"])
        if np.shape(data.diffPerData) != shape or np.shape(data.diffPerDataAve) != shape:
            raise ValueError(f"This model needs a {shape[0]} × {shape[1]} sensor; the live grid is {np.shape(data.diffPerData)}.")
        config = self.predictor.bundle["preprocess"]
        self.segmenter = GestureTrialSegmenter(shape, config["touch_threshold_percent"], config["minimum_touch_frames"])
        self._sensor, self._data = sensor, data
        self._queue = queue.Queue(maxsize=128)
        self._stream_error = None
        self.active = True
        data.add_frame_listener(self._on_frame)
        self.source_label.setText(f"Live source: {getattr(sensor, '_primary_sensor_port', None) or 'primary sensor'} · {shape[0]} × {shape[1]}")
        self._capture_status()

    def _on_frame(self, data):
        # This callback must not run inference or touch widgets in acquisition.
        if not self.active or data is not self._data:
            return
        try:
            self._queue.put_nowait((int(data.frame_sequence), float(data.frame_monotonic_sec),
                                    np.array(data.diffPerData, dtype=np.float32, copy=True),
                                    np.array(data.diffPerDataAve, dtype=np.float32, copy=True)))
        except queue.Full:
            self._stream_error = "Frame queue filled; gesture discarded. Restart recognition."
        except Exception as exc:
            self._stream_error = str(exc)

    def _validate_live(self):
        sensor, data = self._live_sensor()
        if (self._signature(sensor, data) != self._binding
                or not np.array_equal(data.calData, self._calibration)):
            raise ValueError("Sensor, layout, averaging or calibration changed. Restart recognition.")
        if self._stream_error:
            raise ValueError(self._stream_error)

    def _capture_status(self):
        if self._future is not None:
            self._set_status("Analysing", "Finishing the last gesture; lift your finger before the next one.", "info")
        elif self.segmenter.state == "recording":
            self._set_status("Recording gesture", "Follow your gesture, then lift your finger to recognise it.", "active")
        elif self.segmenter.state == "release":
            self._set_status("Lift finger to arm", "Keep your hand away briefly. An incomplete or overlong gesture is not classified.", "warning")
        else:
            self._set_status("Listening", "Touch to start. Lift to finish. Ready for the next gesture.", "active")

    def _show_result(self, result, replay=False):
        self.last_result = result
        self.result_label.setText(gesture_label(result["gesture"]))
        score = result["confidence"] * 100
        lead = result["margin"] * 100
        prefix = "Saved trial · " if replay else "Last gesture · "
        detail = f"{prefix}{score:.2f}% confidence · {lead:.2f}-point lead"
        if not result["accepted"]:
            detail += "\nClosest match: " + gesture_label(result["top_prediction"]) + ". Try a clearer gesture."
        self.confidence_label.setText(detail)
        self.confidence_bar.setValue(round(score * 10))
        self.result_label.setStyleSheet(theme.type_style("title", color=theme.TEXT_PRIMARY if result["accepted"] else theme.WARNING))
        self.scores_label.setText("\n".join(f"{gesture_label(name)}: {value:.1%}" for name, value in
                                           sorted(result["probabilities"].items(), key=lambda item: -item[1])))
        self.history.insertItem(0, f"{'Replay' if replay else time.strftime('%H:%M:%S')} · {gesture_label(result['gesture'])} · {score:.1f}%")
        while self.history.count() > 10:
            self.history.takeItem(self.history.count() - 1)

    def _tick(self):
        try:
            if self.active:
                self._validate_live()  # Never display a late result after a stream change.
            if self._future is not None and self._future.done():
                future, operation = self._future, self._operation
                self._future = self._operation = None
                value = future.result()
                if operation == "load":
                    self.predictor = value
                    self._model_loaded()
                    self._attach()
                elif operation == "replay":
                    self.predictor, result = value
                    self._model_loaded()
                    self._show_result(result, replay=True)
                    self._buttons(False)
                    self._set_status("Replay complete", "Saved recording tested. Start recognition to try the live sensor.", "info")
                    self.timer.stop()
                    return
                elif operation == "predict":
                    self._show_result(value)
            if not self.active:
                return
            for _ in range(128):
                try:
                    frame = self._queue.get_nowait()
                except queue.Empty:
                    break
                episode = self.segmenter.feed(*frame)
                if episode is not None:
                    if self._future is not None:
                        raise ValueError("Recognition fell behind. Incomplete results discarded; restart recognition.")
                    self._operation = "predict"
                    self._future = self._executor.submit(self.predictor.predict_arrays, *episode)
            self._capture_status()
        except Exception as exc:
            self.stop()
            self._set_status("Stopped — check sensor/model", str(exc), "error")

    def _model_loaded(self):
        bundle = self.predictor.bundle
        names = sorted({member["config"]["architecture"].replace("_", "-").upper()
                        for member in bundle["members"]})
        rows, cols = bundle["sensor_shape"]
        self.model_label.setText(f"{len(self.predictor.class_names)} gestures · {' + '.join(names)} · {rows} × {cols} sensor")

    def stop(self):
        self.active = False
        if self._data is not None:
            self._data.remove_frame_listener(self._on_frame)
        self._sensor = self._data = None
        if self._future is not None:
            self._future.cancel()
        self._future = self._operation = None
        # Retire the old queue so stop/start never reuses an incomplete gesture.
        self._queue = queue.Queue(maxsize=128)
        self.timer.stop()
        self._buttons(False)
        self._set_status("Stopped", "Recognition stopped. No robot commands were sent.")
        self.source_label.setText("Source: not listening")

    def shutdown(self):
        if not self._closed:
            self._closed = True
            self.stop()
            self._executor.shutdown(wait=False, cancel_futures=True)

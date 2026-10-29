"""Touch tracking: camera-free/independent capture, train, sensor-only preview.

Opening this window neither opens hardware nor commands a robot. Torch loads
only when training or preview is explicitly requested. Camera access is shared
with the existing preview window, never through a second RealSense pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
from PyQt5.QtCore import Qt, QProcess, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSlider,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from phd.dependence.paths import ai_resource_path
from phd.dependence.touch_tracking_data import (
    LAYOUT, SCHEMA, TrialRecorder, camera_index, homography_from_corners,
    import_reference_csv, project_position, save_reference, save_json, safe_name,
)
from phd.dependence.touch_tracking_camera_free import (
    CameraFreePreview, NOISE_SECONDS, PROTOCOL_SCHEMA, SUPERVISION,
)
from phd.ui import components, theme
from phd.ui.camera_preview_window import image_pixel_at


class ReferenceCanvas(QWidget):
    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.points = []
        self.marker = None
        self.setMinimumSize(440, 280)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.VIEWPORT_BG))
        if self.image.isNull():
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a captured trial to annotate its camera frames")
            return
        w, h = self.image.width(), self.image.height()
        scale = min(self.width()/w, self.height()/h)
        painter.translate((self.width()-w*scale)/2, (self.height()-h*scale)/2)
        painter.scale(scale, scale)
        painter.drawImage(0, 0, self.image)
        painter.setPen(QPen(QColor("#39df98"), 2/scale))
        for i, (x, y) in enumerate(self.points):
            painter.drawEllipse(int(x)-5, int(y)-5, 10, 10)
            painter.drawText(int(x)+7, int(y)-7, str(i+1))
        if self.marker:
            x, y = self.marker
            painter.setPen(QPen(QColor("#ff5b65"), 2/scale))
            painter.drawEllipse(int(x)-7, int(y)-7, 14, 14)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.image.isNull():
            return
        pixel = image_pixel_at(event.x(), event.y(), self.image.width(), self.image.height(), self.width(), self.height())
        if pixel is not None:
            self.clicked.emit(*pixel)


class TrackingMap(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shape = (8, 10)
        self.trail = []
        self.point = None
        self.setMinimumSize(360, 280)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.VIEWPORT_BG))
        pad = 25
        w, h = self.width()-2*pad, self.height()-2*pad
        p.setPen(QPen(QColor(theme.VIEWPORT_GRID), 1))
        rows, cols = self.shape
        for r in range(rows+1):
            y = pad + round(r*h/rows)
            p.drawLine(pad, y, pad+w, y)
        for c in range(cols+1):
            x = pad + round(c*w/cols)
            p.drawLine(x, pad, x, pad+h)
        if self.point is None:
            p.setPen(QColor(theme.VIEWPORT_MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "Waiting for a valid finger position")
        p.setPen(QPen(QColor("#36d296"), 2))
        for a, b in zip(self.trail, self.trail[1:]):
            p.drawLine(pad+round(a[0]*w), pad+round(a[1]*h), pad+round(b[0]*w), pad+round(b[1]*h))
        if self.point is not None:
            p.setBrush(QColor("#ffcb58"))
            x, y = pad+round(self.point[0]*w), pad+round(self.point[1]*h)
            p.drawEllipse(x-6, y-6, 12, 12)


class TouchTrackingWindow(QWidget):
    def __init__(self, owner):
        super().__init__(owner, Qt.Window)
        self.owner = owner
        self.setWindowTitle("Touch Tracking — camera-free or independent-reference training")
        components.size_to_screen(self, 1040, 820)
        self.recorder = None
        self.record_data = None
        self.record_sensor = None
        self.record_source = None
        self._finalizing = None
        self._last_camera = -float("inf")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="touch-workbench")
        self._pending = None
        self._predictor = None
        self._preview_active = False
        self._preview_generation = 0
        self._preview_sequence = None
        self._preview_data = None
        self._preview_sensor = None
        self._preview_queue = deque()
        self._preview_overflow = False
        self._prediction_submitted = None
        self._process = None
        self._process_text = ""
        self._last_output = None
        self._closing = False
        self._annotation_path = None
        self._camera_frames = []
        self._annotations = {}
        self._corners = []
        self._homography = None
        self._calibrating = False
        self._reference_source = "manual_camera"
        self._annotation_dirty = False
        self._candidate = None
        self._range_start = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        header = components.WorkspaceHeader(
            "Touch tracking", "Collect clean trials, train a model and follow one finger. No robot commands.",
            eyebrow="PINGLAB / LEARNING WORKSPACE",
        )
        header.set_compact(True)
        layout.addWidget(header)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_record_page()
        self._build_label_page()
        self._build_train_page()
        self._build_preview_page()
        self._record_mode_changed()
        self.status = QLabel("Ready. Existing recordings and robot policies remain unchanged.")
        self.status.setObjectName("workspaceStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _button(self, text, slot, layout):
        b = QPushButton(text)
        b.clicked.connect(slot)
        layout.addWidget(b)
        return b

    def _message(self, text):
        self.status.setText(str(text))

    def _build_record_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        setup_page = QWidget()
        setup_layout = QVBoxLayout(setup_page)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        form = components.SettingsForm()
        self.record_mode = QComboBox()
        self.record_mode.addItem("Camera-free — guided sensor collection", "camera_free")
        self.record_mode.addItem("Independent reference — camera or measured CSV", "independent")
        form.addRow("Collection mode", self.record_mode)
        self.data_root = QLineEdit(ai_resource_path("data", "touch_tracking"))
        self.participant = QLineEdit("p01")
        self.session = QLineEdit(datetime.now().strftime("%Y%m%d") + "_a")
        self.sensor_id = QLineEdit("sensor_01")
        identity_form = components.SettingsForm()
        for label, field in (("Dataset folder", self.data_root), ("Participant ID", self.participant),
                             ("Session ID (same sitting = same ID)", self.session), ("Physical sensor ID", self.sensor_id)):
            (form if field is self.data_root else identity_form).addRow(label, field)
        dimensions_form = components.SettingsForm()
        self.width_mm, self.height_mm = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (self.width_mm, self.height_mm):
            spin.setRange(0, 2000)
            spin.setSpecialValueText("Unknown — normalized units only")
            spin.setSuffix(" mm")
        dimensions_form.addRow("Sensor width (column direction)", self.width_mm)
        dimensions_form.addRow("Sensor height (row direction)", self.height_mm)
        self.duration = QSpinBox()
        self.duration.setRange(10, 600)
        self.duration.setValue(120)
        self.duration.setSuffix(" s maximum")
        form.addRow("Trial duration", self.duration)
        self.capture_camera = QCheckBox("Save reference camera frames (recommended)")
        self.capture_camera.setChecked(True)
        form.addRow(self.capture_camera)
        setup = components.CardGroup("Trial setup")
        setup.add_layout(form)
        setup_layout.addWidget(setup)
        identity = components.CardGroup("Participant and session")
        identity.add_layout(identity_form)
        setup_layout.addWidget(identity)
        self.sensor_dimensions = components.CollapsibleGroup("Sensor dimensions · optional")
        self.sensor_dimensions.add_layout(dimensions_form)
        setup_layout.addWidget(self.sensor_dimensions)
        hint = QLabel("Before recording: keep the sensor fixed, update calibration with no nearby hand, and identify its row/column directions.\n"
                      "Record slow/fast lines, curves, holds, reversals, lift-off and no-touch intervals. Do not recalibrate during a trial.\n"
                      "Camera labels are surface-contact positions, not an elevated fingernail marker. Use a flat sensor for four-corner calibration.")
        hint.setWordWrap(True)
        self.collection_help = components.CollapsibleGroup("Collection checklist")
        self.collection_help.add(hint)
        setup_layout.addWidget(self.collection_help)
        setup_layout.addStretch()
        self.record_setup_scroll = components.scroll_page(setup_page)
        layout.addWidget(self.record_setup_scroll, 1)
        self.record_hint = hint
        self.camera_free_instructions = QLabel(
            "Camera-free: keep your hand AWAY for the first 6 seconds. Then select Touch/move and keep one finger touching. "
            "Select Uncertain while lifting/repositioning; select No touch once clear. Position targets are sensor estimates, not measured truth.")
        self.camera_free_instructions.setWordWrap(True)
        self.camera_free_instructions.setObjectName("instructionNote")
        layout.addWidget(self.camera_free_instructions)
        self.protocol_confirm = QCheckBox("I will follow the hands-away start and mark touch/no-touch honestly (regular taxel grid).")
        layout.addWidget(self.protocol_confirm)
        state_buttons = QHBoxLayout()
        self.protocol_buttons = []
        for text, state, hint in (
            ("No touch\nHand away", "no_touch", "Hand completely clear of the sensing area."),
            ("Touch / move\nKeep contact", "touch", "One finger touching, including while holding still."),
            ("Uncertain\nReposition", "unknown", "Use while approaching, lifting or repositioning your finger."),
        ):
            button = self._button(text, lambda _checked=False, s=state: self._mark_protocol(s), state_buttons)
            button.setToolTip(hint)
            button.setCheckable(True)
            button.setEnabled(False)
            self.protocol_buttons.append(button)
        layout.addLayout(state_buttons)
        buttons = QHBoxLayout()
        self.open_camera_button = self._button("Open existing camera preview", self._open_camera, buttons)
        self.record_button = self._button("Start trial — no robot", self._toggle_record, buttons)
        components.apply_variant(self.record_button, "primary")
        self._button("Open dataset folder", lambda: self._open_folder(self.data_root.text()), buttons)
        layout.addLayout(buttons)
        self.record_status = QLabel("No trial running")
        self.record_status.setWordWrap(True)
        layout.addWidget(self.record_status)
        # State marking and stop/save stay visible while only setup scrolls.
        self.tabs.addTab(page, "1 · Record")
        self.record_mode.currentIndexChanged.connect(self._record_mode_changed)

    def _record_mode_changed(self, *_args):
        camera_free = self.record_mode.currentData() == "camera_free"
        self.capture_camera.setChecked(not camera_free)
        self.capture_camera.setEnabled(not camera_free)
        self.capture_camera.setVisible(not camera_free)
        self.open_camera_button.setVisible(not camera_free)
        self.protocol_confirm.setVisible(camera_free)
        self.camera_free_instructions.setVisible(camera_free)
        for button in self.protocol_buttons:
            button.setVisible(camera_free)
        if hasattr(self, "training_supervision"):
            self.training_supervision.setCurrentIndex(0 if camera_free else 1)
            namespace = "touch_tracking_camera_free" if camera_free else "touch_tracking"
            self.data_root.setText(ai_resource_path("data", namespace))
            self.train_root.setText(self.data_root.text())
            self.output_root.setText(ai_resource_path("models", namespace))
            self.tabs.setTabText(1, "2 · References (optional)" if camera_free else "2 · Reference labels")
            self.tabs.setTabToolTip(1, "Skip this step for camera-free collection." if camera_free else "Review independent position measurements.")
            self.record_hint.setText(
                "Keep the sensor fixed and calibrate with no nearby hand before recording. Record lines, curves, holds and reversals across the grid. "
                + ("No camera or image annotation is needed. Use the state buttons below." if camera_free else
                   "Camera annotation requires visible contact positions on a flat, fixed sensor, or import independently measured coordinates."))

    def _mark_protocol(self, state):
        if self.recorder is None or self._protocol is None:
            return
        elapsed = time.monotonic()-self.recorder.started
        if elapsed < NOISE_SECONDS + 1.0:
            return
        if self._protocol["events"][-1]["state"] != state:
            self._protocol["events"].append({"t": elapsed, "state": state})
            try:
                save_json(self.recorder.path / "camera_free_protocol.json", self._protocol)
            except Exception as exc:
                self._stop_record(f"Could not save collection protocol: {exc}")

    def _build_label_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        bar = QHBoxLayout()
        self._button("Open trial folder", self._open_annotation, bar)
        self._button("Select four sensor corners", self._begin_calibration, bar)
        self._button("Import measured CSV…", self._import_csv, bar)
        layout.addLayout(bar)
        self.label_hint = QLabel("Corners: (x=0,y=0) → (1,0) → (1,1) → (0,1). Keep this coordinate system consistent with sensor rows/columns.")
        self.label_hint.setWordWrap(True)
        layout.addWidget(self.label_hint)
        self.canvas = ReferenceCanvas()
        self.canvas.clicked.connect(self._image_clicked)
        layout.addWidget(self.canvas, 1)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self._show_annotation_frame)
        layout.addWidget(self.frame_slider)
        self.frame_status = QLabel("No reference images loaded")
        layout.addWidget(self.frame_status)
        nav = QHBoxLayout()
        self._button("Previous", lambda: self._step_frame(-1), nav)
        self._button("Next", lambda: self._step_frame(1), nav)
        self._button("No touch", lambda: self._label_contact(0), nav)
        layout.addLayout(nav)
        nav = QHBoxLayout()
        self._button("Unknown / occluded", lambda: self._label_contact(-1), nav)
        self._button("Suggest next position", self._suggest_next, nav)
        self._button("Accept visible suggestion", self._accept_candidate, nav)
        layout.addLayout(nav)
        ranges = QHBoxLayout()
        self._button("Set interval start", self._set_range_start, ranges)
        self._button("Mark reviewed interval: no touch", lambda: self._label_range(0), ranges)
        self._button("Mark interval: unknown", lambda: self._label_range(-1), ranges)
        layout.addLayout(ranges)
        self.reference_offset = QDoubleSpinBox()
        self.reference_offset.setRange(-10., 10.)
        self.reference_offset.setDecimals(4)
        self.reference_offset.setSingleStep(.01)
        self.reference_offset.setSuffix(" s reference → sensor time offset")
        self.reference_offset.setToolTip("Added to reference timestamps. Estimate independently, never by maximizing model test accuracy.")
        self.reference_offset.valueChanged.connect(self._annotation_changed)
        layout.addWidget(self.reference_offset)
        self.reference_verified = QCheckBox("I reviewed the contact points, no-touch labels, plane calibration and timing; uncertain frames remain unknown.")
        layout.addWidget(self.reference_verified)
        self._button("Save verified reference (keep previous versions)", self._save_annotations, layout)
        self.tabs.addTab(components.scroll_page(page), "2 · Reference labels")

    def _build_train_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        setup = QWidget()
        setup_layout = QVBoxLayout(setup)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Requires at least three independent sessions (or people), including movement and no touch.\n"
                      "Camera-free training skips image annotation: positions are generated from sensor signals. Its scores measure target agreement, NOT physical accuracy.")
        hint.setWordWrap(True)
        setup_layout.addWidget(hint)
        form = components.SettingsForm()
        self.training_supervision = QComboBox()
        self.training_supervision.addItem("Camera-free — operator states + estimated positions", "camera_free")
        self.training_supervision.addItem("Independent measured references", "independent")
        form.addRow("Training targets", self.training_supervision)
        self.train_root = QLineEdit(ai_resource_path("data", "touch_tracking"))
        self.output_root = QLineEdit(ai_resource_path("models", "touch_tracking"))
        form.addRow("Reviewed dataset folder", self.train_root)
        form.addRow("New model runs folder", self.output_root)
        self.split_group = QComboBox()
        self.split_group.addItem("Unseen sessions / days", "session")
        self.split_group.addItem("Unseen people", "participant")
        form.addRow("Evaluation split", self.split_group)
        self.architecture = QComboBox()
        self.architecture.addItem("Compare CNN–GRU and CNN–TCN", "gru,tcn")
        self.architecture.addItem("CNN–GRU", "gru")
        self.architecture.addItem("CNN–TCN", "tcn")
        form.addRow("Model", self.architecture)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(60)
        advanced_form = components.SettingsForm()
        advanced_form.addRow("Maximum epochs (early stopping enabled)", self.epochs)
        self.history_frames = QSpinBox()
        self.history_frames.setRange(4, 100)
        self.history_frames.setValue(16)
        advanced_form.addRow("History frames at 50 Hz", self.history_frames)
        self.seed = QSpinBox()
        self.seed.setRange(0, 1000000)
        self.seed.setValue(42)
        advanced_form.addRow("Reproducible split / training seed", self.seed)
        setup_layout.addLayout(form)
        self.training_advanced = components.CollapsibleGroup("Training parameters · epochs, history and seed")
        self.training_advanced.add_layout(advanced_form)
        setup_layout.addWidget(self.training_advanced)
        setup_layout.addStretch()
        self.train_setup_scroll = components.scroll_page(setup)
        layout.addWidget(self.train_setup_scroll, 3)
        bar = QHBoxLayout()
        self.check_button = self._button("Check dataset", lambda: self._run_training(False), bar)
        self.train_button = self._button("Train + evaluate", lambda: self._run_training(True), bar)
        components.apply_variant(self.train_button, "primary")
        self.cancel_button = self._button("Cancel job", self._cancel_job, bar)
        components.apply_variant(self.cancel_button, "danger")
        self.cancel_button.setEnabled(False)
        self._button("Open results", lambda: self._open_folder(self._last_output or self.output_root.text()), bar)
        layout.addLayout(bar)
        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumBlockCount(3000)
        self.training_log.setAccessibleName("Dataset check and training output")
        self.training_log.setPlaceholderText("Dataset checks and training output will appear here.")
        self.training_log.setMinimumHeight(90)
        layout.addWidget(self.training_log, 2)
        self.tabs.addTab(page, "3 · Train and evaluate")

    def _build_preview_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(components.SectionHeader(
            "Follow your finger", "For a first check, use the camera-free tracker below. No trained model is needed.",
        ))
        self.preview_backend = QComboBox()
        self.preview_backend.addItem("Camera-free robust tracker — no model/training needed", "direct")
        self.preview_backend.addItem("Trained neural model — select checkpoint below", "neural")
        layout.addWidget(self.preview_backend)
        self.preview_hands_away = QCheckBox("My hand is away; keep it away during the 5-second camera-free noise calibration.")
        layout.addWidget(self.preview_hands_away)
        self.checkpoint = QLineEdit()
        self.checkpoint.setPlaceholderText("Select a new Touch Tracking selected_model.pt")
        layout.addWidget(self.checkpoint)
        bar = QHBoxLayout()
        self.choose_model_button = self._button("Select model", self._choose_model, bar)
        self.preview_button = self._button("Start sensor-only preview", self._toggle_preview, bar)
        components.apply_variant(self.preview_button, "primary")
        layout.addLayout(bar)
        self.preview_map = TrackingMap()
        layout.addWidget(self.preview_map, 1)
        self.preview_status = QLabel("Direct tracking needs no trained model. Keep your hand away during startup; no robot commands are sent.")
        self.preview_status.setWordWrap(True)
        layout.addWidget(self.preview_status)
        self.tabs.addTab(components.scroll_page(page), "4 · Live preview")
        self.preview_backend.currentIndexChanged.connect(self._preview_backend_changed)
        self._preview_backend_changed()

    def _preview_backend_changed(self, *_args):
        neural = self.preview_backend.currentData() == "neural"
        self.checkpoint.setVisible(neural)
        self.choose_model_button.setVisible(neural)

    def _open_folder(self, path):
        path = Path(path).expanduser()
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        else:
            self._message("Folder does not exist yet: " + str(path))

    def _camera_window(self):
        return getattr(self.owner.window(), "camera_preview_window", None)

    def _open_camera(self):
        if self.record_mode.currentData() == "camera_free":
            self._message("Camera-free mode does not use or open a camera. Use the guided sensor collection below.")
            return
        main = self.owner.window()
        worker = getattr(self.owner, "yolo_worker", None)
        if worker is not None and worker.isRunning():
            self._message("Stop live object detection/auto-centering first; this workbench will not interrupt an active robot feature.")
            return
        opener = getattr(main, "open_camera_preview_window", None)
        if callable(opener):
            opener()
        else:
            self._message("Open the main application's camera preview first.")

    def _sensor(self):
        sensor = getattr(self.owner, "sensor_functions", None)
        data = getattr(sensor, "_data", None)
        if data is None or not getattr(sensor, "is_connected", False):
            raise ValueError("Build, calibrate and stream the sensor first.")
        t = getattr(data, "frame_monotonic_sec", None)
        if t is None or time.monotonic()-t > .3:
            raise ValueError("Sensor stream is stale. No prediction or recording will start.")
        if not np.isfinite(data.calData).all() or np.count_nonzero(data.calData) < data.calData.size/2:
            raise ValueError("Sensor baseline is missing/invalid. Calibrate with no nearby hand first.")
        return sensor, data

    @staticmethod
    def _taxel_mask(sensor, shape):
        zero = getattr(sensor, "cell_zero_mask", None)
        if zero is None or np.shape(zero) != shape:
            return np.ones(shape, bool)
        # The saved display mask is vertically flipped relative to native raw.
        return ~np.flipud(np.asarray(zero, bool))

    def _toggle_record(self):
        if self.recorder is not None:
            self._stop_record()
            return
        if self._finalizing is not None:
            self._message("Finishing the previous trial; please wait.")
            return
        try:
            active_modes = (
                "_direct_finger_active",
                "_ai_direct_finger_execution_active", "_console_control_active",
                "_ai_proximity_robot_motion_active", "centering_active",
            )
            if any(bool(getattr(self.owner, name, False)) for name in active_modes):
                raise ValueError("Stop active robot-control modes before collecting independent reference data.")
            if self._preview_active or self._pending is not None:
                raise ValueError("Stop tracking preview before recording a new dataset.")
            if self._process is not None:
                raise ValueError("Finish the training/check job before recording.")
            sensor, data = self._sensor()
            camera_free = self.record_mode.currentData() == "camera_free"
            if camera_free and not self.protocol_confirm.isChecked():
                raise ValueError("Confirm the camera-free hands-away start and touch/no-touch instructions first.")
            camera = None
            if not camera_free and self.capture_camera.isChecked():
                camera = self._camera_window()
                snapshot = camera.color_snapshot() if camera is not None else None
                if snapshot is None or time.monotonic()-snapshot[0] > .3:
                    raise ValueError("Open a fresh camera preview first, or explicitly disable camera capture to use an external reference CSV later.")
            metadata = {"participant_id": self.participant.text(), "session_id": self.session.text(),
                        "sensor_id": self.sensor_id.text(), "sensor_shape": list(data.rawData.shape),
                        "surface_size_mm": [self.width_mm.value(), self.height_mm.value()],
                        "sensor_source": str(getattr(sensor, "_primary_sensor_port", "")),
                        "sensor_model": str(getattr(sensor, "current_model_name", "")),
                        "reorder_mode": str(getattr(sensor, "current_reorder_mode", "")),
                        "capture_camera": self.capture_camera.isChecked(),
                        "supervision": SUPERVISION if camera_free else "independent",
                        "raw_note": "Observed native sensor buffer; existing device filtering/disabled-cell masking may already apply."}
            recorder = TrialRecorder(Path(self.data_root.text()).expanduser(), metadata)
            self.sensor_id.setText(recorder.manifest["sensor_id"])
            self.recorder, self.record_data, self.record_sensor = recorder, data, sensor
            self._protocol = ({"schema": PROTOCOL_SCHEMA, "complete": False, "operator_confirmed": True,
                               "events": [{"t": 0., "state": "no_touch"}],
                               "position_source": "sensor_estimate_not_independent_ground_truth"} if camera_free else None)
            if self._protocol is not None:
                save_json(recorder.path / "camera_free_protocol.json", self._protocol)
            self.record_mode.setEnabled(False)
            self.record_source = metadata["sensor_source"]
            self._record_reorder = metadata["reorder_mode"]
            self._record_camera = self.capture_camera.isChecked()
            self._record_camera_window = camera
            self._record_camera_grabber = getattr(camera, "_grabber", None)
            self._record_limit = self.duration.value()
            self._record_last_sensor = time.monotonic()
            self._last_camera = recorder.started
            data.add_frame_listener(self._record_sensor_frame)
            self.record_button.setText("Stop and save trial")
            self._message(("Camera-free recording: keep hand away, then mark your action. " if camera_free else "Recording independently of robot commands: ") + str(recorder.path))
        except Exception as exc:
            if self.recorder is not None:
                self._stop_record(str(exc))
            self._message(exc)

    def _record_sensor_frame(self, data):
        recorder = self.recorder
        if recorder is None:
            return
        try:
            recorder.sensor_frame(data.frame_monotonic_sec, data.frame_sequence, data.rawData, data.calData,
                                  self._taxel_mask(self.record_sensor, data.rawData.shape))
            self._record_last_sensor = data.frame_monotonic_sec
        except Exception as exc:
            recorder.error = str(exc)

    def _stop_record(self, reason=None):
        recorder = self.recorder
        if recorder is None:
            return
        self.record_data.remove_frame_listener(self._record_sensor_frame)
        self.recorder, self.record_data = None, None
        protocol = self._protocol
        for button in self.protocol_buttons:
            button.setEnabled(False)
            button.setChecked(False)
        def finish():
            path = recorder.close(reason)
            if protocol is not None:
                protocol["complete"] = recorder.error is None
                protocol["duration_sec"] = recorder.manifest["duration_sec"]
                save_json(path / "camera_free_protocol.json", protocol)
            return path
        self._finalizing = (self._executor.submit(finish), recorder)
        self.record_button.setEnabled(False)
        self.record_button.setText("Finishing disk writes…")

    def _record_tick(self):
        recorder = self.recorder
        if recorder is None:
            return
        sensor = getattr(self.owner, "sensor_functions", None)
        if (sensor is not self.record_sensor or getattr(sensor, "_data", None) is not self.record_data
                or str(getattr(sensor, "_primary_sensor_port", "")) != self.record_source
                or str(getattr(sensor, "current_reorder_mode", "")) != self._record_reorder):
            self._stop_record("Sensor source or taxel ordering changed during recording.")
            return
        if recorder.error:
            self._stop_record(recorder.error)
            return
        if time.monotonic()-self._record_last_sensor > .5:
            self._stop_record("Sensor stopped producing fresh frames; partial trial retained.")
            return
        if self._record_camera:
            camera = self._camera_window()
            if camera is not self._record_camera_window or getattr(camera, "_grabber", None) is not self._record_camera_grabber:
                self._stop_record("Reference camera was closed/reconnected; a new calibration/trial is required.")
                return
            snapshot = camera.color_snapshot(self._last_camera) if camera is not None else None
            if snapshot is not None:
                self._last_camera = snapshot[0]
                recorder.camera_frame(*snapshot)
            if time.monotonic()-self._last_camera > 1.0:
                self._stop_record("Reference camera stopped; partial trial retained.")
                return
        elapsed = time.monotonic()-recorder.started
        phase = ""
        if self._protocol is not None:
            for button, state in zip(self.protocol_buttons, ("no_touch", "touch", "unknown")):
                button.setEnabled(elapsed >= NOISE_SECONDS + 1.0)
                button.setChecked(self._protocol["events"][-1]["state"] == state)
            phase = (" | KEEP HAND AWAY" if elapsed < NOISE_SECONDS + 1.0 else
                     " | Declared: " + self._protocol["events"][-1]["state"])
        self.record_status.setText(f"{elapsed:.1f} s | Sensor: {recorder.sensor_count} frames | Camera: {recorder.camera_count} frames{phase}\n{recorder.path}")
        if elapsed >= self._record_limit:
            self._stop_record()

    def _annotation_changed(self, *_args):
        if self._annotation_path is not None:
            self._annotation_dirty = True
            self.reference_verified.setChecked(False)

    def _open_annotation(self):
        if self._annotation_dirty and QMessageBox.question(self, "Unsaved annotations", "Discard unsaved annotation edits?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        path = QFileDialog.getExistingDirectory(self, "Select one recorded trial", self.data_root.text())
        if path:
            self.load_annotation(path)

    def load_annotation(self, path):
        try:
            path = Path(path)
            meta = json.loads((path / "manifest.json").read_text())
            if meta.get("schema") != SCHEMA or meta.get("layout") != LAYOUT:
                raise ValueError("Not a new Touch Tracking trial.")
            frames = camera_index(path)
            self._annotation_path, self._camera_frames = path, frames
            self._annotations, self._corners, self._homography = {}, [], None
            self._reference_source = "manual_camera"
            self._range_start, self._candidate = None, None
            self._calibrating = False
            self.reference_offset.setValue(0)
            ref_path = path / "reference.json"
            if ref_path.exists():
                ref = json.loads(ref_path.read_text())
                self._reference_source = ref["source"]
                self._annotations = {float(s["t"]): s for s in ref["samples"]}
                self._corners = ref.get("camera_corners", [])
                self._homography = homography_from_corners(self._corners) if self._corners else None
                self.reference_offset.setValue(ref.get("time_offset_sec", 0))
            self._annotation_dirty = False
            self.reference_verified.setChecked(False)
            self.frame_slider.setRange(0, max(0, len(frames)-1))
            self.frame_slider.setValue(0)
            self._show_annotation_frame()
            self._message(f"Loaded {path.name}. Click the actual contact point to label touch; use Unknown when it is not visible.")
        except Exception as exc:
            self._message(exc)

    def _begin_calibration(self):
        if not self._camera_frames:
            self._message("Load a trial with camera frames first.")
            return
        if self._annotations and QMessageBox.question(self, "Change coordinates?", "Changing corners clears current annotations to avoid inconsistent coordinates. Saved references remain on disk. Continue?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._annotations, self._corners, self._homography = {}, [], None
        self._calibrating = True
        self._annotation_changed()
        self.canvas.points, self.canvas.marker = [], None
        self.canvas.update()
        self.label_hint.setText("Click corner 1: sensor (x=0, y=0), then (1,0), (1,1), (0,1). This is planar calibration; keep sensor and camera fixed.")

    def _image_clicked(self, x, y):
        if not self._camera_frames:
            return
        try:
            if self._calibrating:
                self._corners.append([x, y])
                self.canvas.points = self._corners
                if len(self._corners) == 4:
                    self._homography = homography_from_corners(self._corners)
                    self._calibrating = False
                    self.label_hint.setText("Calibration selected. Check centre/edges against known positions. Click visible surface contact to label touch; occluded or lifted finger positions are not contact ground truth.")
                self.canvas.update()
            else:
                if self._homography is None:
                    raise ValueError("Select the four sensor corners first.")
                xy = project_position(self._homography, (x, y))
                t = self._camera_frames[self.frame_slider.value()][0]
                self._annotations[t] = {"t": t, "contact": 1, "x": xy[0], "y": xy[1], "quality": 1.,
                                        "reviewed": True, "pixel": [x, y]}
                self._reference_source = "manual_camera"
                self._candidate = None
                self._show_annotation_frame()
            self._annotation_changed()
        except Exception as exc:
            if self._calibrating and len(self._corners) >= 4:
                self._corners = []
                self.canvas.points = []
                self.canvas.update()
            self._message(exc)

    def _label_contact(self, state):
        if not self._camera_frames:
            return
        t = self._camera_frames[self.frame_slider.value()][0]
        self._annotations[t] = {"t": t, "contact": state, "x": None, "y": None, "reviewed": True, "quality": 1.}
        self._reference_source = "manual_camera"
        self._candidate = None
        self._annotation_changed()
        self._show_annotation_frame()

    def _set_range_start(self):
        self._range_start = self.frame_slider.value()
        self._message(f"Interval starts at image {self._range_start}. Move to its end after reviewing the images.")

    def _label_range(self, state):
        if self._range_start is None or not self._camera_frames:
            self._message("Set interval start first.")
            return
        a, b = sorted([self._range_start, self.frame_slider.value()])
        if state == 0 and QMessageBox.question(self, "Confirm reference labels", f"Have you checked all {b-a+1} images in this interval and confirmed no physical touch?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        for i in range(a, b+1):
            t = self._camera_frames[i][0]
            self._annotations[t] = {"t": t, "contact": state, "x": None, "y": None, "quality": 1., "reviewed": True}
        self._reference_source = "manual_camera"
        self._annotation_changed()
        self._show_annotation_frame()

    def _step_frame(self, amount):
        self.frame_slider.setValue(self.frame_slider.value()+amount)

    def _show_annotation_frame(self, *_args):
        self.canvas.marker = None
        self.canvas.points = self._corners
        if not self._camera_frames:
            self.canvas.image = QImage()
            self.frame_status.setText("No camera images; import an independent measurement CSV for this trial.")
        else:
            i = self.frame_slider.value()
            t, path = self._camera_frames[i]
            self.canvas.image = QImage(str(path))
            label = self._annotations.get(t)
            state = "unlabelled" if label is None else {1: "touch", 0: "no touch", -1: "unknown"}[label["contact"]]
            self.canvas.marker = label.get("pixel") if label else None
            if self._candidate is not None and self._candidate[0] == i:
                self.canvas.marker = self._candidate[1]
                state += " — suggestion only, NOT a training label"
            self.frame_status.setText(f"Image {i+1}/{len(self._camera_frames)} | {t:.3f} s | {state} | {len(self._annotations)} labels")
        self.canvas.update()

    def _suggest_next(self):
        try:
            import cv2
            i = self.frame_slider.value()
            if i+1 >= len(self._camera_frames):
                return
            t, path = self._camera_frames[i]
            sample = self._annotations.get(t, {})
            pixel = sample.get("pixel")
            if sample.get("contact") != 1 or pixel is None:
                raise ValueError("Manually label a visible contact point first.")
            a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(str(self._camera_frames[i+1][1]), cv2.IMREAD_GRAYSCALE)
            if a is None or b is None or a.shape != b.shape:
                raise ValueError("Camera frames cannot be compared.")
            p = np.array([[pixel]], np.float32)
            q, status, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None)
            if q is None or not bool(status[0, 0]):
                raise ValueError("Visual tracking failed; label the next frame manually.")
            back, ok, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None)
            if back is None or not bool(ok[0, 0]) or np.linalg.norm(back-p) > 1.5:
                raise ValueError("Visual tracking is inconsistent; no label was created.")
            project_position(self._homography, q[0, 0])
            self._candidate = (i+1, q[0, 0].tolist())
            self.frame_slider.setValue(i+1)
            self._message("Inspect the red candidate. Accept only if it is the actual surface contact point; otherwise click the correct point or mark Unknown.")
        except Exception as exc:
            self._message(exc)

    def _accept_candidate(self):
        if self._candidate is None or self._candidate[0] != self.frame_slider.value():
            return
        self._image_clicked(*self._candidate[1])

    def _save_annotations(self):
        try:
            if self._annotation_path is None or not self.reference_verified.isChecked():
                raise ValueError("Review the reference and tick the verification checkbox first.")
            if any(s["contact"] == 1 for s in self._annotations.values()) and self._reference_source == "manual_camera" and self._homography is None:
                raise ValueError("Camera touch labels need a plane calibration.")
            reference = {"schema": SCHEMA, "source": self._reference_source, "verified": True,
                         "time_offset_sec": self.reference_offset.value(), "camera_corners": self._corners,
                         "samples": [self._annotations[t] for t in sorted(self._annotations)],
                         "review_note": "User verified contact coordinates and timing. Occluded/unreviewed intervals excluded."}
            save_reference(self._annotation_path, reference)
            self._annotation_dirty = False
            self._message("Verified reference saved. Previous versions are retained in reference_history/.")
        except Exception as exc:
            self._message(exc)

    def _import_csv(self):
        if self._annotation_path is None:
            self._message("Open a trial folder first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Measured reference: t,contact,x,y[,quality,reviewed]", "", "CSV (*.csv)")
        if not path:
            return
        answer = QMessageBox.question(self, "Verify independent measurement", "Confirm this CSV contains independently measured surface positions (x,y in 0..1), contact labels, and timestamps relative to trial start—not rule-generated robot commands. Timing and coordinate alignment must already be checked.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        try:
            import_reference_csv(self._annotation_path, path, time_offset_sec=self.reference_offset.value())
            self.load_annotation(self._annotation_path)
        except Exception as exc:
            self._message(exc)

    def _run_training(self, train):
        if self._process is not None or self.recorder is not None or self._finalizing is not None or self._preview_active:
            self._message("Stop recording/preview or the current job first.")
            return
        script = Path(__file__).resolve().parents[1] / "script" / "train_touch_tracking.py"
        args = [str(script), "train" if train else "check", self.train_root.text(), "--seq-len", str(self.history_frames.value())]
        args += ["--supervision", self.training_supervision.currentData()]
        if train:
            args += ["--output", self.output_root.text(), "--group-by", self.split_group.currentData(),
                     "--architectures", self.architecture.currentData(), "--epochs", str(self.epochs.value()), "--seed", str(self.seed.value())]
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_process)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        self._process = process
        self._process_text = ""
        self.training_log.clear()
        self.check_button.setEnabled(False)
        self.train_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        process.start(sys.executable, ["-u", *args])

    def _read_process(self):
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._process_text += text
        self.training_log.insertPlainText(text)
        self.training_log.verticalScrollBar().setValue(self.training_log.verticalScrollBar().maximum())
        for line in self._process_text.splitlines():
            if line.startswith("OUTPUT_DIR="):
                self._last_output = line.partition("=")[2]
            if line.startswith("COMPLETED="):
                self.checkpoint.setText(str(Path(line.partition("=")[2]) / "selected_model.pt"))
                self.preview_backend.setCurrentIndex(1)

    def _process_error(self, error):
        if error == QProcess.FailedToStart:
            self._message("Training Python process could not start.")
            self._process_finished(-1, QProcess.CrashExit)

    def _process_finished(self, code, _status):
        self._read_process()
        process, self._process = self._process, None
        if process is not None:
            process.deleteLater()
        self.check_button.setEnabled(True)
        self.train_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._message("Job completed; inspect the evaluation report before live preview." if code == 0 else "Job stopped or refused invalid data. See the log; existing data/models were not changed.")

    def _cancel_job(self):
        if self._process is not None:
            process = self._process
            process.terminate()
            def kill_if_running():
                if self._process is process and process.state() != QProcess.NotRunning:
                    process.kill()
            QTimer.singleShot(2000, kill_if_running)

    def _choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Touch Tracking model", self.output_root.text(), "PyTorch checkpoint (*.pt)")
        if path:
            self.checkpoint.setText(path)

    def _toggle_preview(self):
        if self._preview_active:
            self._stop_preview()
            return
        if self.recorder is not None or self._finalizing is not None or self._pending is not None or self._process is not None:
            self._message("Finish recording/training or wait for the previous preview operation first.")
            return
        try:
            sensor, data = self._sensor()
            direct = self.preview_backend.currentData() == "direct"
            path = self.checkpoint.text().strip()
            if not direct and not Path(path).is_file():
                raise ValueError("Select a trained Touch Tracking checkpoint first.")
            if not self.preview_hands_away.isChecked():
                raise ValueError("Remove your hand and confirm the hands-away preview start first.")
            self._preview_active = True
            self._preview_generation += 1
            self._preview_sequence = None
            self._preview_data = data
            self._preview_sensor = sensor
            self._preview_source = str(getattr(sensor, "_primary_sensor_port", ""))
            self._preview_reorder = str(getattr(sensor, "current_reorder_mode", ""))
            self._preview_queue.clear()
            self._preview_overflow = False
            self.preview_map.shape = data.rawData.shape
            self.preview_map.trail, self.preview_map.point = [], None
            self.preview_button.setText("Stop preview")
            self.preview_backend.setEnabled(False)
            self.record_mode.setEnabled(False)
            self.preview_status.setText("Preparing tracker — keep hand away…")
            shape, sensor_id, ordering = tuple(data.rawData.shape), safe_name(self.sensor_id.text()), self._preview_reorder
            self.sensor_id.setText(sensor_id)
            self._preview_sensor_id = sensor_id
            def load():
                if direct:
                    return CameraFreePreview(shape, sensor_id, ordering)
                from phd.dependence.touch_tracking_model import TrackingPreview
                return TrackingPreview(path)
            self._pending = ("load", self._preview_generation, self._executor.submit(load))
        except Exception as exc:
            self._stop_preview()
            self._message(exc)

    def _stop_preview(self):
        if self._preview_data is not None:
            self._preview_data.remove_frame_listener(self._preview_sensor_frame)
        self._preview_data = None
        self._preview_queue.clear()
        self._preview_active = False
        self._preview_generation += 1
        self._predictor = None
        self.preview_button.setText("Start sensor-only preview")
        self.preview_backend.setEnabled(True)
        if self.recorder is None and self._finalizing is None:
            self.record_mode.setEnabled(True)
        self.preview_map.point, self.preview_map.trail = None, []
        self.preview_map.update()
        self.preview_status.setText("Preview stopped. No robot commands were sent.")

    def _preview_sensor_frame(self, data):
        if len(self._preview_queue) >= 128:
            self._preview_overflow = True
            return
        self._preview_queue.append((data.frame_monotonic_sec, data.frame_sequence,
                                    np.array(data.rawData, copy=True), np.array(data.calData, copy=True),
                                    self._taxel_mask(self._preview_sensor, data.rawData.shape)))

    def _preview_tick(self):
        if self._pending is not None:
            kind, generation, future = self._pending
            if future.done():
                self._pending = None
                try:
                    value = future.result()
                    if self._preview_active and generation == self._preview_generation:
                        if kind == "load":
                            expected = value.config.get("reorder_mode", "")
                            if expected != self._preview_reorder:
                                raise ValueError("Taxel ordering does not match the model. Restore its recorded sensor ordering first.")
                            self._predictor = value
                            self._preview_data.add_frame_listener(self._preview_sensor_frame)
                        elif time.monotonic()-value.get("timestamp", time.monotonic()) <= .2:
                            self._display_prediction(value)
                        else:
                            self._display_prediction({"state": "Prediction too old — discarded", "valid": False})
                except Exception as exc:
                    self._stop_preview()
                    self._message(exc)
        if not self._preview_active:
            return
        if self._preview_overflow:
            self._stop_preview()
            self._message("Tracking worker could not keep up; preview stopped rather than silently dropping frames.")
            return
        try:
            sensor, data = self._sensor()
            if (sensor is not self._preview_sensor or data is not self._preview_data
                    or self.sensor_id.text().strip() != self._preview_sensor_id
                    or str(getattr(sensor, "_primary_sensor_port", "")) != self._preview_source
                    or str(getattr(sensor, "current_reorder_mode", "")) != self._preview_reorder):
                raise ValueError("Sensor source or taxel ordering changed. Restart preview with the correct sensor ID and ordering.")
        except Exception as exc:
            self._stop_preview()
            self._message(exc)
            return
        if self._pending is not None and self._pending[0] == "predict" and self._prediction_submitted is not None and time.monotonic()-self._prediction_submitted > .2:
            self._display_prediction({"state": "Waiting for a fresh prediction", "valid": False})
        if self._predictor is None or self._pending is not None or not self._preview_queue:
            return
        frames = list(self._preview_queue)
        self._preview_queue.clear()
        self._prediction_submitted = time.monotonic()
        self._pending = ("predict", self._preview_generation, self._executor.submit(
            self._predictor.update_many, frames, sensor_id=self.sensor_id.text().strip()))

    def _display_prediction(self, result):
        point = result.get("xy") if result.get("valid") else None
        self.preview_map.point = point
        if point is None:
            self.preview_map.trail = []
        else:
            self.preview_map.trail = (self.preview_map.trail + [point])[-120:]
        self.preview_map.update()
        text = result["state"]
        if result.get("valid"):
            velocity = result.get("velocity", [0, 0])
            if "contact_probability" in result:
                text += f" | Model contact probability {result['contact_probability']:.3f} (not a safety guarantee)"
            elif "signal_quality" in result:
                text += f" | Signal quality {result['signal_quality']:.2f} (NOT contact probability)"
            if point is not None:
                text += f"\nx={point[0]:.3f}, y={point[1]:.3f} | vx={velocity[0]:+.3f}, vy={velocity[1]:+.3f} surface units/s"
            text += f" | inference {result['inference_ms']:.1f} ms"
        self.preview_status.setText(text + "\nSensor-only preview; robot output is disabled by design.")

    def _tick(self):
        if self._closing:
            return
        self._record_tick()
        if self._finalizing is not None and self._finalizing[0].done():
            future, recorder = self._finalizing
            self._finalizing = None
            try:
                path = future.result()
                self._message(("Partial trial retained: " + recorder.error if recorder.error else "Saved trial: " + str(path)))
            except Exception as exc:
                self._message(exc)
            self.record_button.setEnabled(True)
            self.record_mode.setEnabled(True)
            self.record_button.setText("Start trial — no robot")
            self.record_status.setText("Stopped. " + str(recorder.path))
        self._preview_tick()

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        if not getattr(self.owner, "_is_shutting_down", False) and self._annotation_dirty:
            if QMessageBox.question(self, "Unsaved reference edits", "Close without saving reference edits? Saved recordings remain intact.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                event.ignore()
                return
        self._stop_record()
        self._stop_preview()
        if self._finalizing is not None or self._pending is not None or (self._process is not None and self._process.state() != QProcess.NotRunning):
            self._cancel_job()
            event.ignore()
            self._message("Finishing recording/worker shutdown before closing…")
            QTimer.singleShot(100, self.close)
            self._annotation_dirty = False
            return
        self._closing = True
        self.timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        event.accept()

    def shutdown(self):
        """Called before the application's sensor/camera objects are destroyed."""
        if self._closing:
            return
        self._stop_record()
        self._stop_preview()
        self._cancel_job()
        if self._process is not None:
            process = self._process
            if not process.waitForFinished(1000):
                process.kill()
                process.waitForFinished(1000)
        if self._finalizing is not None:
            try:
                self._finalizing[0].result(timeout=10)
            except Exception as exc:
                print(f"[TouchTracking] Final save incomplete; completed chunks retained: {exc}")
        self._closing = True
        self.timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.close()

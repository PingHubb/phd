"""Modeless live view for the Intel RealSense camera.

The window can show color, depth, either infrared camera, color with depth
drawn on top, or a point-cloud preview. Exposure, white balance, and the
stereo emitter are adjustable when the RealSense SDK stream is open.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from phd.ui import components, theme


VIEW_MODES = (
    ("color", "Color"),
    ("depth", "Depth"),
    ("infrared1", "Infrared 1"),
    ("infrared2", "Infrared 2"),
    ("overlay", "Color + depth"),
    ("points", "Point cloud"),
)

_CAPTURE_DIR = Path.home() / "Pictures" / "PingLab"


def realsense_color_candidates(directory: str = "/dev/v4l/by-id") -> list[str]:
    """
    Return stable RealSense video paths, lowest index first.

    ``by-id`` names survive replugs. On the D555 the color stream is
    ``video-index0``. Depth and infrared nodes are rejected after a probe.
    """
    if not os.path.isdir(directory):
        return []
    names = [
        name
        for name in os.listdir(directory)
        if "realsense" in name.lower() and "video-index" in name
    ]
    return [os.path.join(directory, name) for name in sorted(names)]


DEPTH_NEAR_M = 0.15
DEPTH_FAR_M = 3.0
DEPTH_READOUT_RADIUS = 2
DISTANCE_MODES = {"color", "depth", "infrared1", "overlay"}


def depth_meters(depth_z16, depth_scale):
    """Convert a raw depth frame into meters. Missing pixels stay at 0."""
    depth = np.asarray(depth_z16)
    return depth.astype(np.float32) * float(depth_scale)


def available_view_modes(stream_modes) -> list[str]:
    """Expand physical streams into the UI views they can generate."""
    available = set(stream_modes or ())
    if {"color", "depth"}.issubset(available):
        available.update(("overlay", "points"))
    return [mode for mode, _label in VIEW_MODES if mode in available]


def format_distance(meters) -> str:
    """Write one depth sample as meters and centimeters."""
    if meters is None or not np.isfinite(meters) or meters <= 0:
        return "no depth"
    return "%.2f m  (%.0f cm)" % (float(meters), float(meters) * 100.0)


def depth_sample_meters(depth, column, row, radius=DEPTH_READOUT_RADIUS):
    """Return the local median depth, ignoring holes and invalid samples."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return None
    column = int(column)
    row = int(row)
    if not (0 <= column < values.shape[1] and 0 <= row < values.shape[0]):
        return None
    radius = max(0, int(radius))
    patch = values[
        max(0, row - radius) : min(values.shape[0], row + radius + 1),
        max(0, column - radius) : min(values.shape[1], column + radius + 1),
    ]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def image_pixel_at(x, y, image_w, image_h, view_w, view_h):
    """Map a point in the centered preview to an image pixel, or None."""
    if image_w <= 0 or image_h <= 0 or view_w <= 0 or view_h <= 0:
        return None
    scale = min(view_w / float(image_w), view_h / float(image_h))
    drawn_w = image_w * scale
    drawn_h = image_h * scale
    offset_x = (view_w - drawn_w) / 2.0
    offset_y = (view_h - drawn_h) / 2.0
    column = (float(x) - offset_x) / scale
    row = (float(y) - offset_y) / scale
    if column < 0 or row < 0 or column >= image_w or row >= image_h:
        return None
    return int(column), int(row)


def depth_to_rgb(depth_z16, depth_scale, near_m=DEPTH_NEAR_M, far_m=DEPTH_FAR_M):
    """Color a depth frame. Near values are red and far values are blue."""
    depth = np.asarray(depth_z16)
    meters = depth.astype(np.float32) * float(depth_scale)
    valid = np.isfinite(meters) & (meters > 0)
    span = max(float(far_m) - float(near_m), 1e-6)
    safe_meters = np.where(valid, meters, float(near_m))
    amount = np.clip((safe_meters - float(near_m)) / span, 0.0, 1.0)
    rgb = np.zeros(depth.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = np.where(valid, (255 * (1.0 - amount)).astype(np.uint8), 0)
    rgb[..., 1] = np.where(
        valid, (255 * (1.0 - np.abs(amount - 0.5) * 2.0)).astype(np.uint8), 0
    )
    rgb[..., 2] = np.where(valid, (255 * amount).astype(np.uint8), 0)
    return rgb


def gray_to_rgb(frame):
    """Repeat a one-channel infrared frame into red, green, and blue."""
    gray = np.asarray(frame)
    if gray.ndim == 3:
        gray = gray[..., 0]
    return np.repeat(gray[..., None], 3, axis=2)


def bgr_to_rgb(frame):
    """Swap a BGR camera frame into RGB."""
    image = np.asarray(frame)
    return np.ascontiguousarray(image[..., ::-1])


def overlay_color_and_depth(color_bgr, depth_z16, depth_scale):
    """Draw the depth coloring over pixels that have a measured distance."""
    color = bgr_to_rgb(color_bgr)
    depth_rgb = depth_to_rgb(depth_z16, depth_scale)
    if depth_rgb.shape[:2] != color.shape[:2]:
        return color
    measured = np.asarray(depth_z16) > 0
    blended = color.copy()
    blended[measured] = (
        0.55 * color[measured] + 0.45 * depth_rgb[measured]
    ).astype(np.uint8)
    return blended


def point_cloud_to_rgb(vertices, width=640, height=360, near_m=0.15, far_m=3.0):
    """Project camera-frame points into a small image, colored by distance."""
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    step = max(1, points.shape[0] // 120000)
    points = points[::step]
    if points.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05)
    points = points[finite]
    if points.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    radius = float(np.percentile(np.abs(points[:, :2]), 98))
    radius = max(radius, 1e-3)
    scale = 0.45 * min(width, height) / radius
    columns = np.clip(
        (width * 0.5 + points[:, 0] * scale).astype(np.int32), 0, width - 1
    )
    rows = np.clip(
        (height * 0.5 + points[:, 1] * scale).astype(np.int32), 0, height - 1
    )
    span = max(float(far_m) - float(near_m), 1e-6)
    amount = np.clip((points[:, 2] - float(near_m)) / span, 0.0, 1.0)
    colors = np.stack(
        [
            (255 * (1.0 - amount)).astype(np.uint8),
            (255 * (1.0 - np.abs(amount - 0.5) * 2.0)).astype(np.uint8),
            (255 * amount).astype(np.uint8),
        ],
        axis=1,
    )
    image = np.zeros((height, width, 3), dtype=np.uint8)
    order = np.argsort(points[:, 2])[::-1]
    image[rows[order], columns[order]] = colors[order]
    return image


def vertices_from_points(raw):
    """Turn a RealSense vertex buffer into an ``N x 3`` array."""
    points = np.asanyarray(raw)
    names = getattr(points.dtype, "names", None) or ()
    if {"x", "y", "z"}.issubset(names):
        return np.stack((points["x"], points["y"], points["z"]), axis=1)
    if len(names) >= 3:
        return np.stack([points[name] for name in names[:3]], axis=1)
    return points.reshape(-1, 3)


def qimage_from_rgb(rgb) -> QImage:
    """Copy an RGB array into a ``QImage`` the window can keep."""
    image = np.ascontiguousarray(rgb)
    if image.ndim != 3:
        raise ValueError("expected a height x width x channels image")
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("expected an RGB image")
    wrapped = QImage(
        image.data,
        width,
        height,
        width * 3,
        QImage.Format_RGB888,
    )
    return wrapped.copy()


class _FrameGrabber(QThread):
    """Shared latest-frame gate so the window is not flooded."""

    frame_ready = pyqtSignal(QImage, object)
    failed = pyqtSignal(str)
    warning = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False
        self._lock = threading.Lock()
        self._frame_pending = False
        self._failure_reported = False
        self._color_snapshot = None

    def _store_color_frame(self, rgb, timestamp=None):
        """Reference capture is independent of depth view and display throttling."""
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        with self._lock:
            self._color_snapshot = (timestamp, np.array(rgb, dtype=np.uint8, copy=True))

    def color_snapshot(self, after=-float("inf")):
        with self._lock:
            if self._color_snapshot is None or self._color_snapshot[0] <= after:
                return None
            return self._color_snapshot[0], self._color_snapshot[1].copy()

    def stop(self) -> None:
        """Ask the grabber to release the camera and exit."""
        self._stop = True

    def mark_frame_shown(self) -> None:
        """Allow the next frame through after the window has painted."""
        with self._lock:
            self._frame_pending = False

    def _emit_latest(self, image: QImage, depth=None) -> None:
        with self._lock:
            if self._frame_pending:
                return
            self._frame_pending = True
        self.frame_ready.emit(image, depth)

    def _report_failure(self, message: str) -> None:
        """Emit at most one terminal stream error."""
        if self._failure_reported or self._stop:
            return
        self._failure_reported = True
        self.failed.emit(str(message))


class _CameraGrabber(_FrameGrabber):
    """Read one V4L2 color node when the RealSense SDK stream cannot start."""

    def __init__(self, device: str, parent=None):
        super().__init__(parent)
        self._device = device

    def run(self) -> None:
        """Open one V4L2 node and emit color frames until stopped."""
        try:
            import cv2
        except ImportError:
            self._report_failure(
                "OpenCV is not installed, so the camera cannot be shown."
            )
            return

        capture = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            self._report_failure("Could not open %s." % self._device)
            return
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, 30)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        frame = self._read_color_frame(capture)
        if frame is None:
            capture.release()
            self._report_failure("%s is not a color stream." % self._device)
            return

        while not self._stop:
            self._store_color_frame(bgr_to_rgb(frame))
            self._emit_latest(qimage_from_rgb(bgr_to_rgb(frame)), None)
            if self._stop:
                break
            ok, frame = capture.read()
            if (
                not ok
                or frame is None
                or getattr(frame, "ndim", 0) != 3
                or frame.shape[2] != 3
            ):
                self._report_failure("The camera stopped sending frames.")
                break
        capture.release()

    def _read_color_frame(self, capture):
        for _attempt in range(15):
            if self._stop:
                return None
            ok, frame = capture.read()
            if (
                ok
                and frame is not None
                and getattr(frame, "ndim", 0) == 3
                and frame.shape[2] == 3
            ):
                return frame
        return None


class _RealSenseGrabber(_FrameGrabber):
    """Read color, depth, and infrared from one RealSense pipeline."""

    controls_ready = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "color"
        self._options = {}
        self._pipeline = None
        self._pipeline_lock = threading.Lock()

    def set_mode(self, mode: str) -> None:
        """Switch the picture on the next frame without reopening the camera."""
        if mode not in {item[0] for item in VIEW_MODES}:
            return
        with self._lock:
            self._mode = mode

    def request_option(self, sensor_key: str, option_name: str, value: float) -> None:
        """Queue a sensor change to be applied on the camera thread."""
        with self._lock:
            self._options[(sensor_key, option_name)] = float(value)

    def stop(self) -> None:
        """Unblock ``wait_for_frames`` and leave the read loop."""
        super().stop()
        with self._pipeline_lock:
            pipeline = self._pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def run(self) -> None:
        """Start the widest stream set the camera accepts, then emit frames."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            self._report_failure(
                "The RealSense SDK is not installed, so depth and infrared "
                "cannot be shown."
            )
            return

        started = self._start_widest(rs)
        if started is None:
            return
        pipeline, modes = started
        with self._pipeline_lock:
            self._pipeline = pipeline

        try:
            profile = pipeline.get_active_profile()
            depth_scale = self._depth_scale(profile)
            sensors = self._sensors(rs, profile)
            self.controls_ready.emit(
                self._control_snapshot(
                    rs,
                    sensors,
                    available_view_modes(modes),
                    profile,
                )
            )
            align = rs.align(rs.stream.color) if "depth" in modes else None
            point_cloud = rs.pointcloud()
            while not self._stop:
                self._apply_options(rs, sensors)
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=2000)
                    received_at = time.monotonic()
                except Exception as exc:
                    if self._stop:
                        break
                    self._report_failure(
                        "The camera stopped sending frames: %s" % exc
                    )
                    break
                try:
                    color_reference = frames.get_color_frame()
                    if color_reference:
                        self._store_color_frame(
                            bgr_to_rgb(np.asanyarray(color_reference.get_data())),
                            received_at,
                        )
                    image = self._render_frame(
                        frames, align, point_cloud, depth_scale
                    )
                except Exception as exc:
                    image = None
                    text = str(exc)
                    if text != getattr(self, "_render_error", None):
                        self._render_error = text
                        self.warning.emit("Could not draw this view: %s" % text)
                if image is not None:
                    self._emit_latest(image[0], image[1])
        finally:
            with self._pipeline_lock:
                self._pipeline = None
            try:
                pipeline.stop()
            except Exception:
                pass

    def _start_widest(self, rs):
        attempts = (
            (
                ["color", "depth", "infrared1", "infrared2"],
                self._enable_all,
            ),
            (["color", "depth"], self._enable_color_depth),
            (["color"], self._enable_color),
        )
        last_error = "Could not open the RealSense camera."
        for modes, enable in attempts:
            pipeline = rs.pipeline()
            config = rs.config()
            enable(rs, config)
            try:
                pipeline.start(config)
                return pipeline, modes
            except Exception as exc:
                last_error = str(exc)
                try:
                    pipeline.stop()
                except Exception:
                    pass
        self._report_failure(last_error)
        return None

    @staticmethod
    def _enable_all(rs, config) -> None:
        _RealSenseGrabber._enable_color_depth(rs, config)
        config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

    @staticmethod
    def _enable_color_depth(rs, config) -> None:
        _RealSenseGrabber._enable_color(rs, config)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    @staticmethod
    def _enable_color(rs, config) -> None:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    @staticmethod
    def _depth_scale(profile) -> float:
        for sensor in profile.get_device().query_sensors():
            try:
                return float(sensor.as_depth_sensor().get_depth_scale())
            except Exception:
                continue
        return 0.001

    @staticmethod
    def _sensors(rs, profile) -> dict:
        found = {}
        for sensor in profile.get_device().query_sensors():
            try:
                name = sensor.get_info(rs.camera_info.name).lower()
            except Exception:
                continue
            if "rgb" in name or "color" in name:
                found["color"] = sensor
            elif "stereo" in name or "depth" in name:
                found["stereo"] = sensor
        return found

    @staticmethod
    def _option_spec(rs, sensor, option_name: str):
        option = getattr(rs.option, option_name, None)
        if option is None:
            return None
        try:
            if not sensor.supports(option):
                return None
            limits = sensor.get_option_range(option)
            return {
                "min": float(limits.min),
                "max": float(limits.max),
                "step": float(limits.step),
                "value": float(sensor.get_option(option)),
            }
        except Exception:
            return None

    def _control_snapshot(self, rs, sensors, modes, profile) -> dict:
        snapshot = {"modes": list(modes)}
        try:
            device = profile.get_device()
            snapshot["device_name"] = device.get_info(rs.camera_info.name)
            snapshot["serial_number"] = device.get_info(
                rs.camera_info.serial_number
            )
        except Exception:
            pass
        color = sensors.get("color")
        stereo = sensors.get("stereo")
        if color is not None:
            exposure = self._option_spec(rs, color, "exposure")
            auto_exposure = self._option_spec(rs, color, "enable_auto_exposure")
            balance = self._option_spec(rs, color, "white_balance")
            auto_balance = self._option_spec(
                rs, color, "enable_auto_white_balance"
            )
            if exposure is not None:
                exposure["auto"] = bool(
                    auto_exposure and auto_exposure["value"] >= 0.5
                )
                snapshot["color_exposure"] = exposure
            if balance is not None:
                balance["auto"] = bool(
                    auto_balance and auto_balance["value"] >= 0.5
                )
                snapshot["color_white_balance"] = balance
        if stereo is not None:
            laser = self._option_spec(rs, stereo, "laser_power")
            emitter = self._option_spec(rs, stereo, "emitter_enabled")
            if laser is not None:
                laser["emitter"] = bool(emitter and emitter["value"] >= 0.5)
                snapshot["laser_power"] = laser
        return snapshot

    def _apply_options(self, rs, sensors) -> None:
        with self._lock:
            pending = self._options
            self._options = {}
            mode = self._mode
        self._mode_now = mode
        for (sensor_key, option_name), value in pending.items():
            sensor = sensors.get(sensor_key)
            if sensor is None:
                continue
            option = getattr(rs.option, option_name, None)
            if option is None or not sensor.supports(option):
                continue
            try:
                sensor.set_option(option, value)
            except Exception:
                continue

    def _render_frame(self, frames, align, point_cloud, depth_scale):
        mode = getattr(self, "_mode_now", "color")
        if mode == "color":
            return self._color_view(frames, align, depth_scale)
        if mode == "depth":
            depth = frames.get_depth_frame()
            if not depth:
                return None
            raw = np.asanyarray(depth.get_data())
            return (
                qimage_from_rgb(depth_to_rgb(raw, depth_scale)),
                depth_meters(raw, depth_scale),
            )
        if mode in ("infrared1", "infrared2"):
            index = 1 if mode == "infrared1" else 2
            infrared = frames.get_infrared_frame(index)
            if not infrared:
                return None
            picture = gray_to_rgb(np.asanyarray(infrared.get_data()))
            meters = None
            if index == 1:
                depth = frames.get_depth_frame()
                if depth:
                    raw = np.asanyarray(depth.get_data())
                    if raw.shape[:2] == picture.shape[:2]:
                        meters = depth_meters(raw, depth_scale)
            return qimage_from_rgb(picture), meters
        if mode == "points":
            depth = frames.get_depth_frame()
            if not depth:
                return None
            points = point_cloud.calculate(depth)
            vertices = vertices_from_points(points.get_vertices())
            return qimage_from_rgb(point_cloud_to_rgb(vertices)), None
        if align is None:
            return None
        aligned = align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            return None
        color_bgr = np.asanyarray(color.get_data())
        depth_z16 = np.asanyarray(depth.get_data())
        meters = None
        if depth_z16.shape[:2] == color_bgr.shape[:2]:
            meters = depth_meters(depth_z16, depth_scale)
        if mode == "overlay":
            return (
                qimage_from_rgb(
                    overlay_color_and_depth(color_bgr, depth_z16, depth_scale)
                ),
                meters,
            )
        return None

    def _color_view(self, frames, align, depth_scale):
        depth_values = None
        if align is not None:
            aligned = align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if color and depth:
                raw_depth = np.asanyarray(depth.get_data())
                depth_values = depth_meters(raw_depth, depth_scale)
            else:
                color = frames.get_color_frame()
        else:
            color = frames.get_color_frame()
        if not color:
            return None
        return (
            qimage_from_rgb(bgr_to_rgb(np.asanyarray(color.get_data()))),
            depth_values,
        )


class CameraPreviewWindow(QWidget):
    """Live RealSense view. Closing the window releases the camera."""

    def color_snapshot(self, after=-float("inf")):
        grabber = self._grabber
        if grabber is None or not grabber.isRunning():
            return None
        return grabber.color_snapshot(after)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("RealSense Camera")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        components.size_to_screen(self, 980, 720)
        self.setStyleSheet(
            "QWidget { background: %s; color: %s; }"
            % (theme.WINDOW_BG, theme.TEXT_PRIMARY)
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addLayout(self._build_toolbar())
        self.view = QLabel("Looking for the RealSense camera...")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setStyleSheet(
            "background: %s; color: %s;"
            % (theme.VIEWPORT_BG, theme.TEXT_ON_ACCENT)
        )
        self.view.setMouseTracking(True)
        self.view.installEventFilter(self)
        layout.addWidget(self.view, 1)
        self.distance_label = QLabel(
            "Distance: move the cursor over Color, Depth, Color + depth, "
            "or Infrared 1"
        )
        self.distance_label.setWordWrap(True)
        self.distance_label.hide()
        layout.addWidget(self.distance_label)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self._grabber = None
        self._last_image = None
        self._last_depth = None
        self._cursor_pos = None
        self._closing = False
        self._using_realsense = False
        self._saw_frame = False
        self._devices = realsense_color_candidates()
        self._start_realsense()

    def _build_toolbar(self):
        bar = QHBoxLayout()
        bar.addWidget(QLabel("View"))
        self.view_combo = QComboBox()
        for mode, label in VIEW_MODES:
            self.view_combo.addItem(label, mode)
        self.view_combo.currentIndexChanged.connect(self._change_view)
        bar.addWidget(self.view_combo, 1)

        self.save_button = QPushButton("Save frame")
        self.save_button.clicked.connect(self._save_frame)
        bar.addWidget(self.save_button)
        self.reconnect_button = QPushButton("Reconnect")
        self.reconnect_button.setToolTip(
            "Release the camera and reconnect after a USB or stream error."
        )
        self.reconnect_button.clicked.connect(self._restart_stream)
        bar.addWidget(self.reconnect_button)

        controls = QVBoxLayout()
        controls.addLayout(bar)
        self.auto_exposure = QCheckBox("Auto exposure")
        self.exposure_slider = self._slider()
        self.exposure_value = QLabel("")
        controls.addLayout(
            self._slider_row(
                self.auto_exposure, self.exposure_slider, self.exposure_value
            )
        )
        self.auto_white_balance = QCheckBox("Auto white balance")
        self.white_balance_slider = self._slider()
        self.white_balance_value = QLabel("")
        controls.addLayout(
            self._slider_row(
                self.auto_white_balance,
                self.white_balance_slider,
                self.white_balance_value,
            )
        )
        self.emitter = QCheckBox("Emitter")
        self.laser_slider = self._slider()
        self.laser_value = QLabel("")
        controls.addLayout(
            self._slider_row(self.emitter, self.laser_slider, self.laser_value)
        )

        self.auto_exposure.toggled.connect(self._toggle_auto_exposure)
        self.exposure_slider.sliderReleased.connect(self._set_exposure)
        self.exposure_slider.valueChanged.connect(
            lambda value: self.exposure_value.setText(str(value))
        )
        self.auto_white_balance.toggled.connect(self._toggle_auto_white_balance)
        self.white_balance_slider.sliderReleased.connect(self._set_white_balance)
        self.white_balance_slider.valueChanged.connect(
            lambda value: self.white_balance_value.setText(str(value))
        )
        self.emitter.toggled.connect(self._toggle_emitter)
        self.laser_slider.sliderReleased.connect(self._set_laser)
        self.laser_slider.valueChanged.connect(
            lambda value: self.laser_value.setText(str(value))
        )
        self._set_controls_enabled(False)
        return controls

    @staticmethod
    def _slider() -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setEnabled(False)
        return slider

    @staticmethod
    def _slider_row(check, slider, value_label) -> QHBoxLayout:
        row = QHBoxLayout()
        check.setMinimumWidth(170)
        row.addWidget(check)
        row.addWidget(slider, 1)
        value_label.setMinimumWidth(48)
        row.addWidget(value_label)
        return row

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.auto_exposure,
            self.auto_white_balance,
            self.emitter,
            self.exposure_slider,
            self.white_balance_slider,
            self.laser_slider,
        ):
            widget.setEnabled(enabled)

    def _start_realsense(self) -> None:
        self._saw_frame = False
        self._using_realsense = True
        self._set_controls_enabled(False)
        self.status.setText("Connecting to the RealSense camera...")
        grabber = _RealSenseGrabber(self)
        grabber.frame_ready.connect(self._show_frame)
        grabber.failed.connect(self._realsense_failed)
        grabber.warning.connect(self._show_warning)
        grabber.controls_ready.connect(self._apply_control_ranges)
        self._grabber = grabber
        grabber.start()

    def _realsense_failed(self, message: str) -> None:
        if self._closing:
            return
        saw_frame = self._saw_frame
        self._stop_grabber(2500)
        self._using_realsense = False
        self._set_controls_enabled(False)
        self.reconnect_button.setEnabled(True)
        if saw_frame:
            self.status.setText(
                "%s Reconnect the camera when it is available." % message
            )
            return
        self._limit_modes(["color"])
        self.status.setText(
            "Depth, infrared, and point cloud need the RealSense stream. "
            "Showing color only."
        )
        self._try_next_device()

    def _show_warning(self, message: str) -> None:
        if not self._closing:
            self.status.setText(message)

    def _restart_stream(self) -> None:
        """Release any old handle and retry SDK discovery from scratch."""
        if self._closing:
            return
        self._stop_grabber(2500)
        self._last_image = None
        self._last_depth = None
        self._cursor_pos = None
        self._devices = realsense_color_candidates()
        self._limit_modes([mode for mode, _label in VIEW_MODES])
        self.view.clear()
        self.view.setText("Looking for the RealSense camera...")
        self.distance_label.hide()
        self._start_realsense()

    def _limit_modes(self, modes) -> None:
        allowed = set(modes)
        for index in range(self.view_combo.count()):
            mode = self.view_combo.itemData(index)
            enabled = mode in allowed
            item = self.view_combo.model().item(index)
            if item is not None:
                item.setEnabled(enabled)
        if self.view_combo.currentData() not in allowed:
            self.view_combo.setCurrentIndex(0)

    def _apply_control_ranges(self, snapshot: dict) -> None:
        if self._closing:
            return
        self._limit_modes(snapshot.get("modes") or ["color"])
        device_name = snapshot.get("device_name") or "Intel RealSense"
        serial_number = snapshot.get("serial_number")
        identity = (
            "%s (serial %s)" % (device_name, serial_number)
            if serial_number
            else str(device_name)
        )
        self.status.setText("Connected to %s." % identity)
        self._fill_color_control(
            snapshot.get("color_exposure"),
            self.auto_exposure,
            self.exposure_slider,
            self.exposure_value,
        )
        self._fill_color_control(
            snapshot.get("color_white_balance"),
            self.auto_white_balance,
            self.white_balance_slider,
            self.white_balance_value,
        )
        laser = snapshot.get("laser_power")
        if laser is None:
            self.emitter.setEnabled(False)
            self.laser_slider.setEnabled(False)
            return
        self._assign_slider(self.laser_slider, self.laser_value, laser)
        self.emitter.setEnabled(True)
        self.emitter.blockSignals(True)
        self.emitter.setChecked(bool(laser.get("emitter", True)))
        self.emitter.blockSignals(False)
        self.laser_slider.setEnabled(self.emitter.isChecked())

    def _fill_color_control(self, spec, check, slider, value_label) -> None:
        if spec is None:
            check.setEnabled(False)
            slider.setEnabled(False)
            return
        self._assign_slider(slider, value_label, spec)
        check.setEnabled(True)
        check.blockSignals(True)
        check.setChecked(bool(spec.get("auto", True)))
        check.blockSignals(False)
        slider.setEnabled(not check.isChecked())

    @staticmethod
    def _assign_slider(slider, value_label, spec) -> None:
        slider.blockSignals(True)
        slider.setMinimum(int(spec["min"]))
        slider.setMaximum(max(int(spec["min"]) + 1, int(spec["max"])))
        slider.setSingleStep(max(1, int(spec["step"])))
        slider.setValue(int(np.clip(spec["value"], spec["min"], spec["max"])))
        slider.blockSignals(False)
        value_label.setText(str(slider.value()))

    def _change_view(self, _index: int) -> None:
        self._last_depth = None
        mode = self.view_combo.currentData()
        self.distance_label.setVisible(mode in DISTANCE_MODES)
        self._update_distance_readout()
        grabber = self._grabber
        if isinstance(grabber, _RealSenseGrabber):
            grabber.set_mode(mode)

    def _request(self, sensor_key: str, option_name: str, value: float) -> None:
        grabber = self._grabber
        if isinstance(grabber, _RealSenseGrabber):
            grabber.request_option(sensor_key, option_name, value)

    def _toggle_auto_exposure(self, checked: bool) -> None:
        self.exposure_slider.setEnabled(not checked)
        self._request("color", "enable_auto_exposure", 1 if checked else 0)
        if not checked:
            self._set_exposure()

    def _set_exposure(self) -> None:
        if not self.auto_exposure.isChecked():
            self._request("color", "exposure", self.exposure_slider.value())

    def _toggle_auto_white_balance(self, checked: bool) -> None:
        self.white_balance_slider.setEnabled(not checked)
        self._request(
            "color", "enable_auto_white_balance", 1 if checked else 0
        )
        if not checked:
            self._set_white_balance()

    def _set_white_balance(self) -> None:
        if not self.auto_white_balance.isChecked():
            self._request(
                "color", "white_balance", self.white_balance_slider.value()
            )

    def _toggle_emitter(self, checked: bool) -> None:
        self.laser_slider.setEnabled(bool(checked))
        self._request("stereo", "emitter_enabled", 1 if checked else 0)

    def _set_laser(self) -> None:
        self._request("stereo", "laser_power", self.laser_slider.value())

    def _save_frame(self) -> None:
        if self._last_image is None or self._last_image.isNull():
            self.status.setText("No frame to save yet.")
            return
        _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        mode = self.view_combo.currentData() or "color"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = _CAPTURE_DIR / ("realsense_%s_%s.png" % (mode, stamp))
        if self._last_image.save(str(path)):
            self.status.setText("Saved %s" % path)
        else:
            self.status.setText("Could not save %s" % path)

    def _try_next_device(self) -> None:
        if not self._devices:
            self.view.setText(
                "No Intel RealSense color camera is connected."
            )
            self.reconnect_button.setEnabled(True)
            return
        device = self._devices.pop(0)
        self.status.setText("Trying color-only fallback: %s" % device)
        grabber = _CameraGrabber(device, self)
        grabber.frame_ready.connect(self._show_frame)
        grabber.failed.connect(self._show_error)
        grabber.warning.connect(self._show_warning)
        self._grabber = grabber
        grabber.start()

    def _show_frame(self, image: QImage, depth=None) -> None:
        if self._closing:
            return
        self._saw_frame = True
        if not self._using_realsense:
            self.status.setText("Connected in color-only fallback mode.")
        self._last_image = image
        if depth is None:
            self._last_depth = None
        else:
            self._last_depth = np.array(depth, dtype=np.float32, copy=True)
        self._fit_frame()
        self._update_distance_readout()
        grabber = self._grabber
        if grabber is not None:
            grabber.mark_frame_shown()

    def eventFilter(self, watched, event):
        """Read the depth under the cursor while it moves over the picture."""
        if watched is self.view and event.type() == QEvent.MouseMove:
            self._cursor_pos = event.pos()
            self._update_distance_readout()
        elif watched is self.view and event.type() == QEvent.Leave:
            self._cursor_pos = None
            self._update_distance_readout()
        return super().eventFilter(watched, event)

    def _update_distance_readout(self) -> None:
        depth = self._last_depth
        image = self._last_image
        if self.view_combo.currentData() not in DISTANCE_MODES:
            self.distance_label.hide()
            return
        self.distance_label.show()
        if (
            depth is None
            or image is None
            or image.isNull()
            or depth.shape[:2] != (image.height(), image.width())
        ):
            self.distance_label.setText(
                "Distance: not available in this view"
            )
            return
        pixel = None
        where = "Center"
        if self._cursor_pos is not None:
            pixel = image_pixel_at(
                self._cursor_pos.x(),
                self._cursor_pos.y(),
                image.width(),
                image.height(),
                self.view.width(),
                self.view.height(),
            )
            if pixel is not None:
                where = "Cursor"
        if pixel is None:
            pixel = (image.width() // 2, image.height() // 2)
        column, row = pixel
        meters = depth_sample_meters(depth, column, row)
        self.distance_label.setText(
            "%s: %s" % (where, format_distance(meters))
        )

    def _fit_frame(self) -> None:
        if self._last_image is None or self._last_image.isNull():
            return
        self.view.setPixmap(
            QPixmap.fromImage(self._last_image).scaled(
                self.view.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
        )

    def _show_error(self, message: str) -> None:
        if self._closing:
            return
        saw_frame = self._saw_frame
        self._stop_grabber(1500)
        if self._devices and not self._saw_frame:
            self._try_next_device()
            return
        self.status.setText(message)
        self.reconnect_button.setEnabled(True)
        if not saw_frame:
            self.view.setText(message)

    def _stop_grabber(self, wait_ms: int) -> None:
        grabber = self._grabber
        self._grabber = None
        if grabber is None:
            return
        grabber.blockSignals(True)
        grabber.stop()
        grabber.wait(wait_ms)

    def resizeEvent(self, event):
        """Keep the current frame fitted after the window changes size."""
        super().resizeEvent(event)
        self._fit_frame()

    def closeEvent(self, event):
        """Stop the grabber before Qt deletes this window."""
        self._closing = True
        self._stop_grabber(3000)
        super().closeEvent(event)

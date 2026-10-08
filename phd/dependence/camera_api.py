from __future__ import annotations

from importlib import import_module
from typing import Any, Optional, Tuple

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


def median_depth_in_box(
    depth_data: Any,
    box: Any,
    depth_scale: float,
    sample_fraction: float = 0.12,
    minimum_radius: int = 2,
    maximum_radius: int = 10,
) -> float:
    """Estimate object depth from a small valid-pixel patch at box centre.

    A single RealSense depth pixel is frequently zero or noisy. The local
    median is stable enough for robot approach control while still following
    small objects.
    """
    np = import_module("numpy")
    depth = np.asarray(depth_data)
    if depth.ndim != 2 or depth.size == 0:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
        scale = float(depth_scale)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite([x1, y1, x2, y2, scale]).all() or scale <= 0:
        return 0.0

    center_x = int(round((x1 + x2) * 0.5))
    center_y = int(round((y1 + y2) * 0.5))
    center_x = max(0, min(depth.shape[1] - 1, center_x))
    center_y = max(0, min(depth.shape[0] - 1, center_y))
    short_side = max(0.0, min(abs(x2 - x1), abs(y2 - y1)))
    radius = int(round(short_side * max(0.0, float(sample_fraction))))
    radius = max(int(minimum_radius), min(int(maximum_radius), radius))
    patch = depth[
        max(0, center_y - radius) : min(depth.shape[0], center_y + radius + 1),
        max(0, center_x - radius) : min(depth.shape[1], center_x + radius + 1),
    ]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid)) * scale


class YoloWorker(QThread):
    """
    Handles Intel RealSense connection and YOLO object detection.
    Runs in a separate thread to prevent freezing the UI.

    This version keeps the same public API but is safer to import and easier
    to stop when optional camera / ML dependencies are unavailable.
    """

    image_signal = pyqtSignal(QImage)
    data_signal = pyqtSignal(list, tuple)  # Sends [objects, (width, height)]
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, dev_source: Any = None, model_path: str = "yolov8n.pt", frame_timeout_ms: int = 5000):
        super().__init__()
        # Kept for backward compatibility. RealSense still discovers the device
        # automatically, but we store the value in case you want to expand this later.
        self.dev_source = dev_source
        self.model_path = model_path
        self.frame_timeout_ms = int(frame_timeout_ms)

        self.running = True
        self._pipeline: Optional[Any] = None
        self._align: Optional[Any] = None
        self._model: Optional[Any] = None
        self._cv2: Optional[Any] = None
        self._np: Optional[Any] = None
        self._rs: Optional[Any] = None
        self._depth_scale = 0.001

    def _report_error(self, message: str) -> None:
        text = str(message)
        print(f"[CameraAPI] {text}")
        self.error_signal.emit(text)

    # ------------------------------------------------------------------
    # Dependency / setup helpers
    # ------------------------------------------------------------------
    def _load_runtime_dependencies(self) -> bool:
        """Import optional heavy dependencies only when the worker starts."""
        try:
            self._cv2 = import_module("cv2")
            self._np = import_module("numpy")
            self._rs = import_module("pyrealsense2")
            ultralytics = import_module("ultralytics")
            yolo_cls = getattr(ultralytics, "YOLO")
        except Exception as exc:
            self._report_error(f"Dependency import failed: {exc}")
            return False

        try:
            self._model = yolo_cls(self.model_path)
            print(f"[CameraAPI] YOLO model loaded: {self.model_path}")
        except Exception as exc:
            self._report_error(
                f"Error loading YOLO model '{self.model_path}': {exc}"
            )
            self._model = None
            return False

        return True

    def _start_realsense(self) -> bool:
        """Create and start the RealSense pipeline."""
        if self._rs is None:
            return False

        profiles = (
            (640, 480, 30),
            (640, 360, 30),
            (1280, 720, 30),
        )
        last_error = "No compatible color/depth stream profile was found."
        for width, height, fps in profiles:
            if not self.running:
                return False
            pipeline = self._rs.pipeline()
            config = self._rs.config()
            try:
                config.enable_stream(
                    self._rs.stream.color,
                    width,
                    height,
                    self._rs.format.bgr8,
                    fps,
                )
                config.enable_stream(
                    self._rs.stream.depth,
                    width,
                    height,
                    self._rs.format.z16,
                    fps,
                )
                profile = pipeline.start(config)
            except Exception as exc:
                last_error = str(exc)
                try:
                    pipeline.stop()
                except Exception:
                    pass
                continue

            self._pipeline = pipeline
            self._align = self._rs.align(self._rs.stream.color)
            try:
                self._depth_scale = float(
                    profile.get_device().first_depth_sensor().get_depth_scale()
                )
            except Exception:
                self._depth_scale = 0.001
            print(
                "[CameraAPI] RealSense Camera Started "
                f"({width}x{height} at {fps} FPS)."
            )
            return True

        self._report_error(f"Could not start RealSense: {last_error}")
        self._pipeline = None
        self._align = None
        return False

    # ------------------------------------------------------------------
    # Frame processing helpers
    # ------------------------------------------------------------------
    def _get_aligned_frames(self) -> Optional[Tuple[Any, Any]]:
        pipeline = self._pipeline
        align = self._align
        if pipeline is None or align is None:
            return None

        frames = pipeline.wait_for_frames(timeout_ms=self.frame_timeout_ms)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        return color_frame, depth_frame

    def _emit_detection_result(self, annotated_frame: Any, detected_objects: list, frame_size: tuple) -> None:
        if self._cv2 is None:
            return

        rgb_image = self._cv2.cvtColor(annotated_frame, self._cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.image_signal.emit(qt_image.copy())
        self.data_signal.emit(detected_objects, frame_size)

    def _process_current_frame(self) -> bool:
        if self._np is None or self._model is None:
            return False

        aligned = self._get_aligned_frames()
        if aligned is None:
            return True  # no frame yet is not fatal

        color_frame, depth_frame = aligned
        frame = self._np.asanyarray(color_frame.get_data())
        depth_data = self._np.asanyarray(depth_frame.get_data())
        height, width, _ = frame.shape
        frame_size = (width, height)

        results = self._model(frame, stream=True, verbose=False)
        detected_objects = []

        for result in results:
            annotated_frame = result.plot()

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                name = self._model.names[cls_id]
                conf = float(box.conf[0])

                dist_meters = median_depth_in_box(
                    depth_data,
                    (x1, y1, x2, y2),
                    self._depth_scale,
                )

                detected_objects.append(
                    {
                        "name": name,
                        "box": [x1, y1, x2, y2],
                        "depth": dist_meters,
                        "conf": conf,
                    }
                )

            self._emit_detection_result(annotated_frame, detected_objects, frame_size)

        return True

    # ------------------------------------------------------------------
    # Thread entry / shutdown
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("[CameraAPI] Initializing Intel RealSense Pipeline...")

        if not self._load_runtime_dependencies():
            self.finished_signal.emit()
            return

        if not self._start_realsense():
            self.finished_signal.emit()
            return

        consecutive_errors = 0

        try:
            while self.running:
                try:
                    ok = self._process_current_frame()
                    if ok:
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                except Exception as exc:
                    if not self.running:
                        break
                    consecutive_errors += 1
                    self._report_error(
                        f"Stream error ({consecutive_errors}/10): {exc}"
                    )

                # Prevent endless noisy loops if the camera is repeatedly failing.
                if consecutive_errors >= 10:
                    self._report_error(
                        "Too many consecutive camera errors; detection stopped."
                    )
                    break
        finally:
            self.running = False
            self._stop_pipeline()
            print("[CameraAPI] RealSense Pipeline Stopped")
            self.finished_signal.emit()

    def _stop_pipeline(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._align = None

        if pipeline is None:
            return

        try:
            pipeline.stop()
        except Exception:
            # It may already be stopped or unavailable; that is safe to ignore.
            pass

    def stop(self) -> None:
        self.running = False
        self._stop_pipeline()
        self.wait(2000)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def shutdown(self) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

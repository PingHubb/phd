"""Modeless live view for the Intel RealSense color camera."""

from __future__ import annotations

import os
import threading

from PyQt5.QtCore import QCoreApplication, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from phd.ui import components, theme


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


class _CameraGrabber(QThread):
    """Read frames off the GUI thread so PingLab stays responsive."""

    frame_ready = pyqtSignal(QImage)
    failed = pyqtSignal(str)

    def __init__(self, device: str, parent=None):
        super().__init__(parent)
        self._device = device
        self._stop = False
        self._lock = threading.Lock()
        self._frame_pending = False

    def stop(self) -> None:
        """Ask the grabber to release the camera and exit."""
        self._stop = True

    def mark_frame_shown(self) -> None:
        """Allow the next frame through after the window has painted."""
        with self._lock:
            self._frame_pending = False

    def _emit_latest(self, image: QImage) -> None:
        with self._lock:
            if self._frame_pending:
                return
            self._frame_pending = True
        self.frame_ready.emit(image)

    def run(self) -> None:
        """Open one V4L2 node and emit color frames until stopped."""
        try:
            import cv2
        except ImportError:
            self.failed.emit(
                "OpenCV is not installed, so the camera cannot be shown."
            )
            return

        capture = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            self.failed.emit("Could not open %s." % self._device)
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
            self.failed.emit("%s is not a color stream." % self._device)
            return

        while not self._stop:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _channels = rgb.shape
            data = rgb.tobytes()
            image = QImage(
                data,
                width,
                height,
                width * 3,
                QImage.Format_RGB888,
            ).copy()
            self._emit_latest(image)
            if self._stop:
                break
            ok, frame = capture.read()
            if (
                not ok
                or frame is None
                or getattr(frame, "ndim", 0) != 3
                or frame.shape[2] != 3
            ):
                self.failed.emit("The camera stopped sending frames.")
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


class CameraPreviewWindow(QWidget):
    """Live RealSense color view. Closing the window releases the camera."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("RealSense Camera")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        components.size_to_screen(self, 980, 640)
        self.setStyleSheet(
            "QWidget { background: %s; color: %s; }"
            % (theme.WINDOW_BG, theme.TEXT_PRIMARY)
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.view = QLabel("Looking for the RealSense color camera...")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setStyleSheet(
            "background: %s; color: %s;"
            % (theme.VIEWPORT_BG, theme.TEXT_ON_ACCENT)
        )
        layout.addWidget(self.view, 1)
        self._grabber = None
        self._last_image = None
        self._closing = False
        self._devices = realsense_color_candidates()
        self._try_next_device()

    def _try_next_device(self) -> None:
        if not self._devices:
            self.view.setText(
                "No Intel RealSense color camera is connected."
            )
            return
        device = self._devices.pop(0)
        grabber = _CameraGrabber(device, self)
        grabber.frame_ready.connect(self._show_frame)
        grabber.failed.connect(self._show_error)
        self._grabber = grabber
        grabber.start()

    def _show_frame(self, image: QImage) -> None:
        if self._closing:
            return
        self._last_image = image
        self._fit_frame()
        grabber = self._grabber
        if grabber is not None:
            grabber.mark_frame_shown()

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
        grabber = self._grabber
        self._grabber = None
        if grabber is not None:
            grabber.stop()
            grabber.wait(500)
        if self._devices and self._last_image is None:
            self._try_next_device()
            return
        if self._last_image is None:
            self.view.setText(message)

    def resizeEvent(self, event):
        """Keep the current frame fitted after the window changes size."""
        super().resizeEvent(event)
        self._fit_frame()

    def closeEvent(self, event):
        """Stop the grabber before Qt deletes this window."""
        self._closing = True
        grabber = self._grabber
        self._grabber = None
        if grabber is not None:
            grabber.blockSignals(True)
            grabber.stop()
            QCoreApplication.removePostedEvents(self)
            grabber.wait(2000)
            QCoreApplication.removePostedEvents(self)
        super().closeEvent(event)

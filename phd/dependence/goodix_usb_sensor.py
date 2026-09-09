"""Goodix/WingCool USB tactile-sensor transport.

The controller exposes normal desktop touch events on USB interface 0 and a
vendor matrix-data protocol on interface 1.  The primary reader keeps that
second interface open through libusb; WingCool's UsbTouchCore utility remains
available as a compatibility fallback.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal


GOODIX_USB_VENDOR_ID = "27c6"
GOODIX_USB_PRODUCT_ID = "0528"
GOODIX_USB_SOURCE_ID = "goodix-usb:27c6:0528"
GOODIX_USB_SOURCE_LABEL = "Goodix USB Touch Sensor (10x8)"
GOODIX_USB_ROWS = 10
GOODIX_USB_COLUMNS = 8
GOODIX_USB_VALUE_COUNT = GOODIX_USB_ROWS * GOODIX_USB_COLUMNS
GOODIX_USB_TOOL_ENV = "PINGLAB_USB_TOUCH_CORE"
GOODIX_USB_BACKEND_ENV = "PINGLAB_GOODIX_BACKEND"
GOODIX_XINPUT_NAME = "WingCool Inc. TouchScreen"

GOODIX_USB_INTERFACE = 1
GOODIX_USB_ENDPOINT_OUT = 0x02
GOODIX_USB_ENDPOINT_IN = 0x82
GOODIX_USB_PACKET_SIZE = 64
GOODIX_USB_PACKET_PAYLOAD_SIZE = GOODIX_USB_PACKET_SIZE - 1

_GOODIX_CMD_RAW_MODE = bytes.fromhex(
    "03 86 01 01 74 00 06 00 00 04 02 06"
)
_GOODIX_CMD_COORDINATE_MODE = bytes.fromhex(
    "03 87 01 01 74 00 06 00 00 04 00 04"
)
_GOODIX_CMD_SYNC = bytes.fromhex("03 0f 01 01 74 00 01 00")
_GOODIX_CMD_READ_FRAME_READY = bytes.fromhex("03 0f 01 02 74 00 01")
_GOODIX_CMD_CLEAR_FRAME_READY = bytes.fromhex("03 8f 01 02 74 00 01")
_GOODIX_CMD_READ_RAW_BASE = bytes.fromhex("03 0f 01 a6 7c 00 00")

_GOODIX_MODE_SETTLE_SEC = 0.010
_GOODIX_FRAME_POLL_SEC = 0.001
_GOODIX_TRANSFER_TIMEOUT_MS = 250
_GOODIX_PROTOCOL_ATTEMPTS = 10
_GOODIX_FRAME_READY_ATTEMPTS = 25


class GoodixUsbError(RuntimeError):
    """Raised when the Goodix USB matrix stream cannot be read."""


def is_goodix_usb_source(value):
    """Return whether a UI/source value identifies the Goodix USB sensor."""
    text = str(value or "").strip().lower()
    return text in {
        GOODIX_USB_SOURCE_ID.lower(),
        GOODIX_USB_SOURCE_LABEL.lower(),
    }


def find_goodix_usb_devices(sysfs_root="/sys/bus/usb/devices"):
    """Return sysfs directories for connected matching USB devices."""
    matches = []
    root = Path(sysfs_root)
    if not root.is_dir():
        return matches

    for device_dir in root.iterdir():
        try:
            vendor = (device_dir / "idVendor").read_text().strip().lower()
            product = (device_dir / "idProduct").read_text().strip().lower()
        except (OSError, UnicodeError):
            continue
        if vendor == GOODIX_USB_VENDOR_ID and product == GOODIX_USB_PRODUCT_ID:
            matches.append(device_dir)
    return sorted(matches, key=lambda path: path.name)


def goodix_usb_connected():
    """Return whether at least one supported Goodix sensor is connected."""
    return bool(find_goodix_usb_devices())


def _tool_candidates():
    configured = os.environ.get(GOODIX_USB_TOOL_ENV, "").strip()
    if configured:
        yield Path(configured).expanduser()

    on_path = shutil.which("UsbTouchCore")
    if on_path:
        yield Path(on_path)

    download_dir = (
        Path.home() / "Downloads" / "Essential" / "Singa_New_Sensor"
    )
    yield download_dir / "UsbTouchCore_1.0.8_x86"
    if download_dir.is_dir():
        yield from sorted(download_dir.glob("UsbTouchCore*_x86"), reverse=True)


def find_usb_touch_core_tool():
    """Locate WingCool's external UsbTouchCore executable."""
    seen = set()
    for candidate in _tool_candidates():
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


_INTEGER_LINE_RE = re.compile(r"^[\s,+\-0-9]+$")
_INTEGER_RE = re.compile(r"[+\-]?\d+")


def parse_goodix_raw_matrix(
    output,
    rows=GOODIX_USB_ROWS,
    columns=GOODIX_USB_COLUMNS,
):
    """Parse a tool matrix and return historic column-major values."""
    rows = int(rows)
    columns = int(columns)
    numeric_blocks = []
    current_block = []

    def finish_block():
        if current_block:
            numeric_blocks.append(list(current_block))
            current_block.clear()

    for line in str(output or "").splitlines():
        stripped = line.strip()
        if not stripped or not _INTEGER_LINE_RE.fullmatch(stripped):
            finish_block()
            continue
        values = [int(token) for token in _INTEGER_RE.findall(stripped)]
        if len(values) not in {rows, columns}:
            finish_block()
            continue
        current_block.append(values)
    finish_block()

    matrices = []
    for block in numeric_blocks:
        if len(block) >= rows and all(len(line) == columns for line in block):
            matrices.append(np.asarray(block[-rows:], dtype=np.int64))
        if len(block) >= columns and all(len(line) == rows for line in block):
            matrices.append(np.asarray(block[-columns:], dtype=np.int64).T)

    if not matrices:
        raise GoodixUsbError(
            f"UsbTouchCore did not return a {rows}x{columns} raw matrix."
        )

    matrix = matrices[-1]
    if matrix.shape != (rows, columns):
        raise GoodixUsbError(
            f"UsbTouchCore returned matrix shape {matrix.shape}, expected "
            f"({rows}, {columns})."
        )
    return matrix.T.reshape(-1).astype(int).tolist()


def build_goodix_raw_read_command(value_count=GOODIX_USB_VALUE_COUNT):
    """Build the vendor memory-read command for unsigned 16-bit taxel data."""
    byte_count = int(value_count) * 2
    if byte_count <= 0 or byte_count > 0xFFFF:
        raise ValueError("value_count must fit in a 16-bit byte count")
    command = bytearray(_GOODIX_CMD_READ_RAW_BASE)
    command[5:7] = byte_count.to_bytes(2, byteorder="big", signed=False)
    return bytes(command)


def decode_goodix_raw_packets(packets, value_count=GOODIX_USB_VALUE_COUNT):
    """Decode 63-byte packet payloads into the controller's taxel order."""
    payload = bytearray()
    for packet in packets:
        packet = bytes(packet)
        if len(packet) < 2:
            raise GoodixUsbError(
                "Goodix USB returned an incomplete data packet."
            )
        payload.extend(packet[1:])

    required_bytes = int(value_count) * 2
    if len(payload) < required_bytes:
        raise GoodixUsbError(
            f"Goodix USB returned {len(payload)} raw bytes; expected "
            f"{required_bytes}."
        )

    payload = payload[:required_bytes]
    return [
        int.from_bytes(payload[offset:offset + 2], "little", signed=False)
        for offset in range(0, required_bytes, 2)
    ]


class _LibusbApi:
    """Small ctypes binding containing only the libusb calls used here."""

    def __init__(self, library_path=None):
        library_path = library_path or ctypes.util.find_library("usb-1.0")
        if not library_path:
            raise GoodixUsbError("libusb-1.0 is not installed.")
        try:
            self.lib = ctypes.CDLL(library_path)
        except OSError as exc:
            raise GoodixUsbError(f"Could not load libusb-1.0: {exc}") from exc
        self._configure_signatures()

    def _configure_signatures(self):
        void_pointer = ctypes.c_void_p
        byte_pointer = ctypes.POINTER(ctypes.c_ubyte)

        self.lib.libusb_init.argtypes = [ctypes.POINTER(void_pointer)]
        self.lib.libusb_init.restype = ctypes.c_int
        self.lib.libusb_exit.argtypes = [void_pointer]
        self.lib.libusb_exit.restype = None
        self.lib.libusb_open_device_with_vid_pid.argtypes = [
            void_pointer,
            ctypes.c_uint16,
            ctypes.c_uint16,
        ]
        self.lib.libusb_open_device_with_vid_pid.restype = void_pointer
        self.lib.libusb_close.argtypes = [void_pointer]
        self.lib.libusb_close.restype = None
        self.lib.libusb_set_auto_detach_kernel_driver.argtypes = [
            void_pointer,
            ctypes.c_int,
        ]
        self.lib.libusb_set_auto_detach_kernel_driver.restype = ctypes.c_int
        self.lib.libusb_kernel_driver_active.argtypes = [
            void_pointer,
            ctypes.c_int,
        ]
        self.lib.libusb_kernel_driver_active.restype = ctypes.c_int
        self.lib.libusb_detach_kernel_driver.argtypes = [
            void_pointer,
            ctypes.c_int,
        ]
        self.lib.libusb_detach_kernel_driver.restype = ctypes.c_int
        self.lib.libusb_attach_kernel_driver.argtypes = [
            void_pointer,
            ctypes.c_int,
        ]
        self.lib.libusb_attach_kernel_driver.restype = ctypes.c_int
        self.lib.libusb_get_configuration.argtypes = [
            void_pointer,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.libusb_get_configuration.restype = ctypes.c_int
        self.lib.libusb_set_configuration.argtypes = [
            void_pointer,
            ctypes.c_int,
        ]
        self.lib.libusb_set_configuration.restype = ctypes.c_int
        self.lib.libusb_claim_interface.argtypes = [void_pointer, ctypes.c_int]
        self.lib.libusb_claim_interface.restype = ctypes.c_int
        self.lib.libusb_release_interface.argtypes = [
            void_pointer,
            ctypes.c_int,
        ]
        self.lib.libusb_release_interface.restype = ctypes.c_int
        self.lib.libusb_clear_halt.argtypes = [void_pointer, ctypes.c_ubyte]
        self.lib.libusb_clear_halt.restype = ctypes.c_int
        self.lib.libusb_interrupt_transfer.argtypes = [
            void_pointer,
            ctypes.c_ubyte,
            byte_pointer,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
        ]
        self.lib.libusb_interrupt_transfer.restype = ctypes.c_int
        self.lib.libusb_error_name.argtypes = [ctypes.c_int]
        self.lib.libusb_error_name.restype = ctypes.c_char_p

    def error_text(self, result):
        try:
            value = self.lib.libusb_error_name(int(result))
            if value:
                return value.decode("ascii", errors="replace")
        except Exception:
            pass
        return f"libusb error {int(result)}"


class GoodixLibusbSession:
    """Persistent libusb session for the Goodix matrix-data interface."""

    def __init__(
        self,
        rows=GOODIX_USB_ROWS,
        columns=GOODIX_USB_COLUMNS,
        api=None,
    ):
        self.rows = int(rows)
        self.columns = int(columns)
        self.value_count = self.rows * self.columns
        self._api = api
        self._context = ctypes.c_void_p()
        self._handle = None
        self._claimed = False
        self._manually_detached = False
        self._raw_mode_active = False
        self._raw_mode_ready = False
        self._lock = threading.RLock()
        self._cancel_event = threading.Event()

    @staticmethod
    def library_available():
        return bool(ctypes.util.find_library("usb-1.0"))

    def _error(self, operation, result):
        detail = self._api.error_text(result) if self._api else str(result)
        return GoodixUsbError(f"{operation} failed: {detail} ({int(result)}).")

    def _check(self, result, operation):
        if int(result) < 0:
            raise self._error(operation, result)

    def open(self):
        with self._lock:
            if self._cancel_event.is_set():
                raise GoodixUsbError("Goodix USB reading was cancelled.")
            if self._handle:
                return
            if not goodix_usb_connected():
                raise GoodixUsbError(
                    "The Goodix USB touch sensor is not connected."
                )
            if self._api is None:
                self._api = _LibusbApi()

            try:
                result = self._api.lib.libusb_init(ctypes.byref(self._context))
                self._check(result, "libusb initialization")
                self._handle = self._api.lib.libusb_open_device_with_vid_pid(
                    self._context,
                    int(GOODIX_USB_VENDOR_ID, 16),
                    int(GOODIX_USB_PRODUCT_ID, 16),
                )
                if not self._handle:
                    raise GoodixUsbError(
                        "Could not open the Goodix USB device. Check its udev "
                        "permissions and reconnect it."
                    )

                auto_detach = (
                    self._api.lib.libusb_set_auto_detach_kernel_driver(
                        self._handle,
                        1,
                    )
                )
                if int(auto_detach) < 0:
                    active = self._api.lib.libusb_kernel_driver_active(
                        self._handle,
                        GOODIX_USB_INTERFACE,
                    )
                    if int(active) == 1:
                        result = self._api.lib.libusb_detach_kernel_driver(
                            self._handle,
                            GOODIX_USB_INTERFACE,
                        )
                        self._check(result, "Goodix kernel-driver detach")
                        self._manually_detached = True

                configuration = ctypes.c_int()
                result = self._api.lib.libusb_get_configuration(
                    self._handle,
                    ctypes.byref(configuration),
                )
                self._check(result, "Goodix USB configuration read")
                if configuration.value != 1:
                    result = self._api.lib.libusb_set_configuration(
                        self._handle,
                        1,
                    )
                    self._check(result, "Goodix USB configuration")

                result = self._api.lib.libusb_claim_interface(
                    self._handle,
                    GOODIX_USB_INTERFACE,
                )
                self._check(result, "Goodix USB interface claim")
                self._claimed = True
            except Exception:
                self._close_unlocked()
                raise

    def _close_unlocked(self):
        if self._handle and self._api:
            if self._claimed and self._raw_mode_active:
                # The raw-data mode is global to the controller. Restore normal
                # coordinate reporting before releasing the USB interface so
                # desktop touch can work again after PingLab exits.
                was_cancelled = self._cancel_event.is_set()
                self._cancel_event.clear()
                try:
                    self._leave_raw_mode()
                except Exception:
                    # Closing must still release the interface if the device was
                    # unplugged or no longer responds.
                    pass
                finally:
                    if was_cancelled:
                        self._cancel_event.set()
            if self._claimed:
                self._api.lib.libusb_release_interface(
                    self._handle,
                    GOODIX_USB_INTERFACE,
                )
            if self._manually_detached:
                self._api.lib.libusb_attach_kernel_driver(
                    self._handle,
                    GOODIX_USB_INTERFACE,
                )
            self._api.lib.libusb_close(self._handle)
        if self._context and self._api:
            self._api.lib.libusb_exit(self._context)
        self._context = ctypes.c_void_p()
        self._handle = None
        self._claimed = False
        self._manually_detached = False
        self._raw_mode_active = False
        self._raw_mode_ready = False

    def close(self):
        with self._lock:
            self._close_unlocked()

    @property
    def cancelled(self):
        return self._cancel_event.is_set()

    def request_cancel(self):
        """Request a bounded-time exit from the current protocol operation."""
        self._cancel_event.set()

    def _wait(self, duration_sec):
        if self._cancel_event.wait(max(0.0, float(duration_sec))):
            raise GoodixUsbError("Goodix USB reading was cancelled.")

    def _transfer(
        self,
        endpoint,
        output=None,
        timeout_ms=_GOODIX_TRANSFER_TIMEOUT_MS,
    ):
        if self._cancel_event.is_set():
            raise GoodixUsbError("Goodix USB reading was cancelled.")
        if not self._handle:
            raise GoodixUsbError("The Goodix libusb session is not open.")

        buffer = (ctypes.c_ubyte * GOODIX_USB_PACKET_SIZE)()
        if output is not None:
            output = bytes(output)
            if len(output) > GOODIX_USB_PACKET_SIZE:
                raise ValueError("Goodix USB command exceeds one packet")
            for index, value in enumerate(output):
                buffer[index] = value

        transferred = ctypes.c_int()
        result = self._api.lib.libusb_interrupt_transfer(
            self._handle,
            endpoint,
            buffer,
            GOODIX_USB_PACKET_SIZE,
            ctypes.byref(transferred),
            max(1, int(timeout_ms)),
        )
        if int(result) < 0:
            self._api.lib.libusb_clear_halt(self._handle, endpoint)
            raise self._error("Goodix USB interrupt transfer", result)
        if output is not None and transferred.value != GOODIX_USB_PACKET_SIZE:
            raise GoodixUsbError(
                f"Goodix USB wrote {transferred.value} bytes; expected "
                f"{GOODIX_USB_PACKET_SIZE}."
            )
        if output is None and transferred.value <= 0:
            raise GoodixUsbError("Goodix USB returned an empty packet.")
        return bytes(buffer[:transferred.value])

    def _write(self, command):
        self._transfer(GOODIX_USB_ENDPOINT_OUT, output=command)

    def _read(self):
        return self._transfer(GOODIX_USB_ENDPOINT_IN)

    @staticmethod
    def _is_acknowledgement(packet):
        return len(packet) > 1 and packet[1] == 0x80

    def _enter_raw_mode(self):
        self._write(_GOODIX_CMD_RAW_MODE)
        self._raw_mode_active = True
        for _ in range(_GOODIX_PROTOCOL_ATTEMPTS):
            if self._cancel_event.is_set():
                raise GoodixUsbError("Goodix USB reading was cancelled.")
            self._write(_GOODIX_CMD_SYNC)
            self._wait(_GOODIX_MODE_SETTLE_SEC)
            response = self._read()
            if self._is_acknowledgement(response):
                self._raw_mode_ready = True
                return
        raise GoodixUsbError("Goodix USB did not acknowledge raw-data mode.")

    def _leave_raw_mode(self):
        """Best-effort restoration of the controller's coordinate mode."""
        self._write(_GOODIX_CMD_COORDINATE_MODE)
        for _ in range(_GOODIX_PROTOCOL_ATTEMPTS):
            self._write(_GOODIX_CMD_SYNC)
            self._wait(_GOODIX_MODE_SETTLE_SEC)
            response = self._read()
            if self._is_acknowledgement(response):
                break
        self._raw_mode_active = False
        self._raw_mode_ready = False

    def _read_frame(self):
        for attempt in range(_GOODIX_FRAME_READY_ATTEMPTS):
            if self._cancel_event.is_set():
                raise GoodixUsbError("Goodix USB reading was cancelled.")
            self._write(_GOODIX_CMD_READ_FRAME_READY)
            response = self._read()
            if self._is_acknowledgement(response):
                break
            if attempt + 1 < _GOODIX_FRAME_READY_ATTEMPTS:
                self._wait(_GOODIX_FRAME_POLL_SEC)
        else:
            raise GoodixUsbError("Goodix USB did not produce a fresh raw frame.")

        self._write(build_goodix_raw_read_command(self.value_count))
        packet_count = (
            (self.value_count * 2) + GOODIX_USB_PACKET_PAYLOAD_SIZE - 1
        ) // GOODIX_USB_PACKET_PAYLOAD_SIZE
        try:
            packets = [self._read() for _ in range(packet_count)]
            # The BLB raw-memory buffer is already serialized column by
            # column, which is the historic flat payload consumed by PingLab.
            # UsbTouchCore transposes this buffer only for its printed table;
            # applying that display transpose here would scramble taxel
            # locations once PingLab reshapes the payload.
            return decode_goodix_raw_packets(packets, self.value_count)
        finally:
            # Acknowledge consumption so firmware can publish the next frame.
            self._write(_GOODIX_CMD_CLEAR_FRAME_READY)

    def read_raw(self):
        """Read a frame while retaining the interface and raw mode."""
        with self._lock:
            self.open()
            try:
                if not self._raw_mode_ready:
                    self._enter_raw_mode()
                return self._read_frame()
            except Exception:
                self._raw_mode_ready = False
                raise


class GoodixUsbSensorClient:
    """Read Goodix frames using persistent libusb with a utility fallback."""

    def __init__(
        self,
        tool_path=None,
        rows=GOODIX_USB_ROWS,
        columns=GOODIX_USB_COLUMNS,
        backend=None,
        direct_session=None,
    ):
        self.tool_path = tool_path or find_usb_touch_core_tool()
        self.rows = int(rows)
        self.columns = int(columns)
        self.backend = str(
            backend or os.environ.get(GOODIX_USB_BACKEND_ENV, "auto")
        ).strip().lower()
        if self.backend not in {"auto", "direct", "tool"}:
            raise ValueError(
                f"{GOODIX_USB_BACKEND_ENV} must be auto, direct, or tool"
            )
        self._direct_session = direct_session or GoodixLibusbSession(
            rows=self.rows,
            columns=self.columns,
        )
        self._active_backend = "not started"
        self._direct_disabled_reason = ""
        self._process = None
        self._process_lock = threading.Lock()

    @property
    def is_available(self):
        if not goodix_usb_connected():
            return False
        direct_available = GoodixLibusbSession.library_available()
        if self.backend == "direct":
            return direct_available
        if self.backend == "tool":
            return bool(self.tool_path)
        return bool(direct_available or self.tool_path)

    @property
    def backend_status(self):
        if self._active_backend == "direct":
            return "direct libusb (persistent BLB raw mode)"
        if self._active_backend == "tool":
            detail = ""
            if self._direct_disabled_reason:
                detail = f"; direct error: {self._direct_disabled_reason}"
            return f"vendor utility fallback (per-frame process{detail})"
        if self.backend == "tool":
            return "vendor utility requested"
        if self.backend == "direct":
            return "direct libusb requested"
        return "auto (direct libusb preferred)"

    def _permission_message(self):
        installer = (
            Path(__file__).resolve().parents[1]
            / "script"
            / "setup_goodix_usb_permissions.py"
        )
        return (
            "USB access was denied. Run "
            f"`sudo python3 {installer}`, reconnect "
            "the sensor, then restart phd_ui."
        )

    def _run_tool(self, option, timeout=3.0):
        if not self.tool_path:
            raise GoodixUsbError(
                "UsbTouchCore was not found. Set "
                "PINGLAB_USB_TOUCH_CORE to its "
                "executable path."
            )
        if not goodix_usb_connected():
            raise GoodixUsbError(
                "The Goodix USB touch sensor is not connected."
            )

        command = [self.tool_path, str(option)]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
        except OSError as exc:
            raise GoodixUsbError(
                f"Could not start UsbTouchCore: {exc}"
            ) from exc

        with self._process_lock:
            self._process = process
        try:
            output, _ = process.communicate(timeout=max(0.1, float(timeout)))
        except subprocess.TimeoutExpired as exc:
            process.kill()
            output, _ = process.communicate()
            raise GoodixUsbError(
                "UsbTouchCore timed out while reading a frame."
            ) from exc
        finally:
            with self._process_lock:
                if self._process is process:
                    self._process = None

        output = str(output or "")
        if "LIBUSB_ERROR_ACCESS" in output or "Can't open" in output:
            raise GoodixUsbError(self._permission_message())
        if process.returncode not in (0, None):
            detail = (
                output.strip().splitlines()[-1]
                if output.strip()
                else "unknown error"
            )
            raise GoodixUsbError(
                f"UsbTouchCore exited with code {process.returncode}: {detail}"
            )
        self._active_backend = "tool"
        return output

    def read_raw(self):
        """Read one 10x8 raw-capacitance frame."""
        if self.backend != "tool" and not self._direct_disabled_reason:
            direct_error = None
            for _ in range(2):
                try:
                    values = self._direct_session.read_raw()
                    self._active_backend = "direct"
                    return list(values)
                except GoodixUsbError as exc:
                    direct_error = exc
                    if self._direct_session.cancelled:
                        raise
                    self._direct_session.close()
            if self.backend == "direct":
                raise direct_error
            self._direct_disabled_reason = str(direct_error)

        if not self.tool_path:
            if self._direct_disabled_reason:
                raise GoodixUsbError(
                    "Direct Goodix USB reading failed and UsbTouchCore is not "
                    f"available as a fallback: {self._direct_disabled_reason}"
                )
            raise GoodixUsbError("UsbTouchCore is not available.")
        output = self._run_tool("-raw")
        return parse_goodix_raw_matrix(output, self.rows, self.columns)

    def measure_read_raw_hz(self, duration_sec=1.0):
        """Measure complete vendor-tool frame reads over a fixed interval."""
        duration_sec = float(duration_sec)
        if duration_sec <= 0.0:
            raise ValueError("duration_sec must be positive")
        started = time.perf_counter()
        successes = 0
        attempts = 0
        while (time.perf_counter() - started) < duration_sec:
            attempts += 1
            try:
                self.read_raw()
            except GoodixUsbError:
                continue
            successes += 1
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            "hz": successes / elapsed,
            "success_count": successes,
            "total_attempts": attempts,
            "elapsed_sec": elapsed,
            "backend": self.backend_status,
        }

    def cancel_current_read(self):
        """Terminate an in-flight utility process during shutdown."""
        self._direct_session.request_cancel()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    def close(self):
        self.cancel_current_read()
        self._direct_session.close()


class GoodixUsbReadWorker(QObject):
    """Continuously acquire Goodix frames outside the Qt event loop."""

    raw_payload_ready = pyqtSignal(int, str, list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, client, generation=0, retry_delay_sec=0.25):
        super().__init__()
        self.client = client
        self.generation = int(generation)
        self.retry_delay_sec = max(0.01, float(retry_delay_sec))
        self._running = False

    def stop(self):
        self._running = False
        self.client.cancel_current_read()

    def run(self):
        self._running = True
        try:
            while self._running:
                try:
                    values = self.client.read_raw()
                except GoodixUsbError as exc:
                    if self._running:
                        self.error.emit(f"Goodix USB read failed: {exc}")
                        time.sleep(self.retry_delay_sec)
                    continue
                if self._running:
                    self.raw_payload_ready.emit(
                        self.generation,
                        GOODIX_USB_SOURCE_ID,
                        list(values),
                    )
        finally:
            self.finished.emit()


def set_goodix_desktop_touch_enabled(enabled):
    """Enable or disable only the X11 desktop-touch device."""
    xinput = shutil.which("xinput")
    if not xinput:
        return False, "xinput is not installed or this is not an X11 session."
    action = "enable" if bool(enabled) else "disable"
    try:
        result = subprocess.run(
            [xinput, action, GOODIX_XINPUT_NAME],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Could not run xinput: {exc}"
    if result.returncode != 0:
        return False, str(result.stdout or "xinput failed").strip()
    state = "enabled" if enabled else "disabled"
    return True, f"Goodix desktop touch {state}."

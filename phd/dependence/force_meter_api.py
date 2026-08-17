"""Serial readers and protocol parsers for HANDPI HP-series force meters."""

from __future__ import annotations

import math
import re
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import serial
from PyQt5.QtCore import QObject, pyqtSignal


@dataclass(frozen=True)
class ForceReading:
    """One force-meter sample, normalized to newtons."""

    force_newtons: float
    native_value: float
    native_unit: str
    raw_text: str


PROTOCOL_MODBUS_RTU = "modbus_rtu"
PROTOCOL_TEXT_STREAM = "text_stream"

HP200_SLAVE_ADDRESS = 1
HP200_REGISTER_COUNT = 13
HP200_POLL_INTERVAL_SECONDS = 0.2

_HP200_UNIT_CODES = {
    0: "N",
    1: "kN",
    2: "gf",
    3: "kgf",
    4: "tf",
    5: "lbf",
    6: "klbf",
}

_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_UNIT_RE = r"kN|mN|cN|N|kgf?|gf|tf|klbf?|lbf?|ozf"
_MEASUREMENT_RE = re.compile(
    rf"(?<![\w.])(?P<value>{_NUMBER_RE})\s*(?P<unit>{_UNIT_RE})?(?![A-Za-z])",
    re.IGNORECASE,
)
_UNIT_ONLY_RE = re.compile(rf"(?<![A-Za-z])({_UNIT_RE})(?![A-Za-z])", re.IGNORECASE)


def _canonical_unit(unit: Optional[str]) -> str:
    value = str(unit or "N").strip().lower()
    return {
        "n": "N",
        "kn": "kN",
        "mn": "mN",
        "cn": "cN",
        "kg": "kgf",
        "kgf": "kgf",
        "g": "gf",
        "gf": "gf",
        "tf": "tf",
        "klb": "klbf",
        "klbf": "klbf",
        "lb": "lbf",
        "lbf": "lbf",
        "ozf": "ozf",
    }.get(value, "N")


def force_to_newtons(value: float, unit: str) -> float:
    """Convert a force value in an HP-series display unit to newtons."""

    factors = {
        "N": 1.0,
        "kN": 1e3,
        "mN": 1e-3,
        "cN": 1e-2,
        "kgf": 9.80665,
        "gf": 9.80665e-3,
        "tf": 9806.65,
        "klbf": 4448.2216152605,
        "lbf": 4.4482216152605,
        "ozf": 0.278013850953781,
    }
    return float(value) * factors[_canonical_unit(unit)]


def parse_force_text(text: str) -> Optional[ForceReading]:
    """Parse one ASCII force-meter record.

    HP-series software and cable revisions can include a unit after the value,
    before the value, or omit it. Bare values are interpreted as newtons.
    """

    cleaned = "".join(
        char if char.isprintable() or char in "\t " else " "
        for char in str(text or "")
    ).strip()
    if not cleaned:
        return None

    match = _MEASUREMENT_RE.search(cleaned)
    if match is None:
        return None

    try:
        native_value = float(match.group("value"))
    except (TypeError, ValueError):
        return None

    unit = match.group("unit")
    if not unit:
        unit_match = _UNIT_ONLY_RE.search(cleaned)
        unit = unit_match.group(1) if unit_match else "N"
    native_unit = _canonical_unit(unit)
    return ForceReading(
        force_newtons=force_to_newtons(native_value, native_unit),
        native_value=native_value,
        native_unit=native_unit,
        raw_text=cleaned,
    )


def modbus_crc16(payload: bytes) -> int:
    """Return the standard Modbus RTU CRC-16 value."""

    crc = 0xFFFF
    for byte in bytes(payload or b""):
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc & 0xFFFF


def build_hp200_read_request(
    slave_address: int = HP200_SLAVE_ADDRESS,
    start_register: int = 0,
    register_count: int = HP200_REGISTER_COUNT,
) -> bytes:
    """Build the read request used by the official HP-200 application."""

    if not 1 <= int(slave_address) <= 247:
        raise ValueError("Modbus slave address must be between 1 and 247")
    if not 0 <= int(start_register) <= 0xFFFF:
        raise ValueError("Modbus start register is outside the 16-bit range")
    if not 1 <= int(register_count) <= 125:
        raise ValueError("Modbus register count must be between 1 and 125")

    payload = struct.pack(
        ">BBHH",
        int(slave_address),
        0x03,
        int(start_register),
        int(register_count),
    )
    crc = modbus_crc16(payload)
    return payload + struct.pack("<H", crc)


def decode_hp200_registers(registers) -> ForceReading:
    """Decode the 13 holding registers returned by the HANDPI application."""

    values = [int(value) for value in registers]
    if len(values) < HP200_REGISTER_COUNT:
        raise ValueError(
            f"HP-200 reply has {len(values)} registers; "
            f"expected {HP200_REGISTER_COUNT}"
        )
    if any(value < 0 or value > 0xFFFF for value in values):
        raise ValueError("HP-200 reply contains a non-16-bit register value")

    # The official application combines register 0 as the high word and
    # register 1 as the low word of an IEEE-754 single-precision value.
    native_value = struct.unpack(
        ">f", struct.pack(">HH", values[0], values[1])
    )[0]
    if not math.isfinite(native_value):
        raise ValueError("HP-200 force register is not a finite number")

    unit_code = values[11]
    native_unit = _HP200_UNIT_CODES.get(unit_code)
    if native_unit is None:
        raise ValueError(f"HP-200 returned unsupported unit code {unit_code}")

    raw_text = f"{native_value:+.6g} {native_unit} (Modbus)"
    return ForceReading(
        force_newtons=force_to_newtons(native_value, native_unit),
        native_value=float(native_value),
        native_unit=native_unit,
        raw_text=raw_text,
    )


class Hp200ModbusRtuParser:
    """Incrementally validate and decode HP-200 Modbus RTU responses."""

    def __init__(
        self,
        slave_address: int = HP200_SLAVE_ADDRESS,
        register_count: int = HP200_REGISTER_COUNT,
    ):
        self.slave_address = int(slave_address)
        self.register_count = int(register_count)
        self._buffer = bytearray()
        self._notices = deque(maxlen=8)

    def _add_notice(self, message: str):
        if not self._notices or self._notices[-1] != message:
            self._notices.append(str(message))

    def pop_notice(self) -> Optional[str]:
        return self._notices.popleft() if self._notices else None

    def feed(self, payload: bytes) -> List[ForceReading]:
        self._buffer.extend(bytes(payload or b""))
        readings = []

        while self._buffer:
            try:
                frame_start = self._buffer.index(self.slave_address)
            except ValueError:
                self._buffer.clear()
                break
            if frame_start:
                del self._buffer[:frame_start]
            if len(self._buffer) < 2:
                break

            function_code = self._buffer[1]
            if function_code == 0x83:
                frame_size = 5
            elif function_code == 0x03:
                if len(self._buffer) < 3:
                    break
                byte_count = self._buffer[2]
                if byte_count > 250 or byte_count % 2:
                    self._add_notice(
                        f"Invalid Modbus byte count {byte_count}; resynchronizing"
                    )
                    del self._buffer[0]
                    continue
                frame_size = 5 + byte_count
            else:
                del self._buffer[0]
                continue

            if len(self._buffer) < frame_size:
                break
            frame = bytes(self._buffer[:frame_size])
            received_crc = struct.unpack("<H", frame[-2:])[0]
            expected_crc = modbus_crc16(frame[:-2])
            if received_crc != expected_crc:
                self._add_notice(
                    "Modbus CRC mismatch; check baud rate, wiring, and A/B polarity"
                )
                del self._buffer[0]
                continue
            del self._buffer[:frame_size]

            if function_code == 0x83:
                self._add_notice(
                    f"HP-200 returned Modbus exception code {frame[2]}"
                )
                continue

            byte_count = frame[2]
            expected_byte_count = self.register_count * 2
            if byte_count != expected_byte_count:
                self._add_notice(
                    f"HP-200 returned {byte_count // 2} registers; "
                    f"expected {self.register_count}"
                )
                continue

            registers = struct.unpack(
                ">" + ("H" * self.register_count), frame[3:-2]
            )
            try:
                readings.append(decode_hp200_registers(registers))
            except ValueError as exc:
                self._add_notice(str(exc))

        if len(self._buffer) > 1024:
            del self._buffer[:-1024]
        return readings


class Hp200StreamParser:
    """Incrementally parse newline-delimited or unit-terminated samples."""

    _DELIMITER_RE = re.compile(r"[\r\n;]+")
    _UNIT_TERMINATED_RE = re.compile(
        rf"(?<![\w.])(?P<value>{_NUMBER_RE})\s*(?P<unit>{_UNIT_RE})(?![A-Za-z])",
        re.IGNORECASE,
    )

    def __init__(self):
        self._buffer = ""

    def feed(self, payload: bytes) -> List[ForceReading]:
        decoded = bytes(payload or b"").decode("ascii", errors="ignore")
        if not decoded:
            return []
        self._buffer += decoded
        readings = []

        while True:
            delimiter = self._DELIMITER_RE.search(self._buffer)
            if delimiter is None:
                break
            record = self._buffer[:delimiter.start()]
            self._buffer = self._buffer[delimiter.end():]
            reading = parse_force_text(record)
            if reading is not None:
                readings.append(reading)

        # Some HP-series cable/software combinations send compact records such
        # as "+012.3N" without CR/LF. A trailing unit is a safe record boundary.
        while True:
            match = self._UNIT_TERMINATED_RE.search(self._buffer)
            if match is None:
                break
            reading = parse_force_text(match.group(0))
            self._buffer = self._buffer[match.end():]
            if reading is not None:
                readings.append(reading)

        if len(self._buffer) > 512:
            self._buffer = self._buffer[-512:]
        return readings


class Hp200ForceMeterWorker(QObject):
    """Own the HP-200 serial port and read it outside the Qt event loop."""

    connected = pyqtSignal(str, int, str)
    sample_ready = pyqtSignal(float, float, str, str, float)
    unparsed_data = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        port: str,
        baud_rate: int = 9600,
        protocol: str = PROTOCOL_MODBUS_RTU,
        display_hz: float = 30.0,
        slave_address: int = HP200_SLAVE_ADDRESS,
        poll_interval: float = HP200_POLL_INTERVAL_SECONDS,
    ):
        super().__init__()
        self.port = str(port)
        self.baud_rate = int(baud_rate)
        protocol = str(protocol or PROTOCOL_MODBUS_RTU).strip().lower()
        if protocol not in (PROTOCOL_MODBUS_RTU, PROTOCOL_TEXT_STREAM):
            raise ValueError(f"Unsupported HP-200 protocol: {protocol}")
        self.protocol = protocol
        self.slave_address = int(slave_address)
        self.poll_interval = max(0.05, float(poll_interval))
        self.display_interval = 1.0 / max(1.0, float(display_hz))
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    @staticmethod
    def _sample_rate(sample_times: deque, now: float) -> float:
        while sample_times and (now - sample_times[0]) > 1.0:
            sample_times.popleft()
        if len(sample_times) < 2:
            return 0.0
        duration = sample_times[-1] - sample_times[0]
        return (len(sample_times) - 1) / duration if duration > 0.0 else 0.0

    def run(self):
        if self.protocol == PROTOCOL_MODBUS_RTU:
            parser = Hp200ModbusRtuParser(self.slave_address)
            request = build_hp200_read_request(self.slave_address)
        else:
            parser = Hp200StreamParser()
            request = None
        sample_times = deque()
        next_display_time = 0.0
        next_poll_time = 0.0
        last_unparsed_notice = time.monotonic()
        last_valid_sample = time.monotonic()
        unparsed_preview = bytearray()
        pending_reading = None

        try:
            with serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                exclusive=True,
            ) as meter:
                meter.reset_input_buffer()
                self.connected.emit(self.port, self.baud_rate, self.protocol)

                while not self._stop_event.is_set():
                    now = time.monotonic()
                    if request is not None and now >= next_poll_time:
                        meter.write(request)
                        meter.flush()
                        next_poll_time = now + self.poll_interval

                    payload = meter.read(meter.in_waiting or 1)
                    now = time.monotonic()
                    if payload:
                        readings = parser.feed(payload)
                        if readings:
                            last_valid_sample = now
                            unparsed_preview.clear()
                            for reading in readings:
                                sample_times.append(now)
                                pending_reading = reading
                        elif self.protocol == PROTOCOL_TEXT_STREAM:
                            unparsed_preview.extend(payload)
                            if len(unparsed_preview) > 256:
                                del unparsed_preview[:-256]

                    if now - last_unparsed_notice >= 1.0:
                        preview = None
                        if self.protocol == PROTOCOL_MODBUS_RTU:
                            preview = parser.pop_notice()
                            if preview is None and now - last_valid_sample >= 1.5:
                                preview = (
                                    "No Modbus reply. Check meter power, use the "
                                    "USB-to-RS-485 port, and verify A/B polarity."
                                )
                        elif unparsed_preview:
                            preview = bytes(unparsed_preview).decode(
                                "ascii", errors="replace"
                            ).strip()
                            if not preview:
                                preview = bytes(unparsed_preview[:24]).hex(" ")
                            unparsed_preview.clear()
                        if preview:
                            self.unparsed_data.emit(str(preview)[:160])
                            last_unparsed_notice = now

                    if pending_reading is not None and now >= next_display_time:
                        self.sample_ready.emit(
                            pending_reading.force_newtons,
                            pending_reading.native_value,
                            pending_reading.native_unit,
                            pending_reading.raw_text,
                            self._sample_rate(sample_times, now),
                        )
                        pending_reading = None
                        next_display_time = now + self.display_interval
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

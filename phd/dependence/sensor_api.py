import time

from phd.dependence.sensor_protocol import (
    DEFAULT_SENSOR_BAUD_RATE,
    DEFAULT_SENSOR_RESPONSE_TIMEOUT_SEC,
    DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC,
    SENSOR_FRAME_CONTINUATION_GRACE_SEC,
    SENSOR_RESPONSE_IDLE_TIMEOUT_SEC,
    SENSOR_RESPONSE_START_GRACE_SEC,
    SENSOR_UPDATE_CAL_START_GRACE_SEC,
    discard_pending_serial_input,
    extract_sensor_frame,
    has_sensor_frame_end,
    has_sensor_frame_start,
    parse_serial_ints,
    read_complete_serial_line,
    sensor_response_timeout_for_values,
    take_complete_sensor_payload,
    trim_sensor_payload_accumulator,
    wait_for_serial_input,
)

try:
    import serial
except ImportError:
    serial = None


class ArduinoCommander:
    """Safe serial wrapper for the tactile sensor controller."""

    def __init__(
        self,
        serial_port="/dev/ttyACM0",
        baud_rate=DEFAULT_SENSOR_BAUD_RATE,
        timeout=DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC,
        connect_immediately=True,
    ):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.response_timeout = 2.0
        self.expected_payload_values = None
        self.ser = None
        if bool(connect_immediately):
            self._connect()

    # ------------------------------------------------------------------
    # Connection / lifecycle
    # ------------------------------------------------------------------
    def _connect(self):
        """Try to open the serial connection. Safe to call multiple times."""
        if serial is None:
            print(
                "[SensorAPI] pyserial is not installed. "
                "Sensor API is unavailable."
            )
            self.ser = None
            return False

        if self.is_connected():
            return True

        try:
            self.ser = serial.Serial(
                self.serial_port,
                self.baud_rate,
                timeout=self.timeout,
            )
            discard_pending_serial_input(self.ser)
            return True
        except serial.SerialException as exc:
            print(
                f"[SensorAPI] Serial port {self.serial_port} "
                f"not available: {exc}"
            )
            self.ser = None
            return False

    def reconnect(self):
        self.close()
        return self._connect()

    def set_serial_port(self, serial_port, reconnect=True):
        """Select a serial device and optionally connect to it immediately."""
        serial_port = str(serial_port or "").strip()
        if not serial_port:
            return False

        port_changed = serial_port != self.serial_port
        if port_changed:
            self.close()
            self.serial_port = serial_port

        if not bool(reconnect):
            return True
        return self.is_connected() or self._connect()

    def is_connected(self):
        return self.ser is not None and getattr(self.ser, "is_open", False)

    @property
    def is_available(self):
        return self.is_connected()

    def _ensure_connection(self):
        """Reconnect on demand if needed."""
        return self.is_connected() or self._connect()

    def close(self):
        if self.ser is not None:
            try:
                if self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass
        self.ser = None

    def shutdown(self):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _write_line(self, text):
        if not self._ensure_connection():
            return False

        try:
            self.ser.write((text + "\n").encode("utf-8"))
            return True
        except Exception as exc:
            print(f"[SensorAPI] Failed to write to sensor serial port: {exc}")
            self.close()
            return False

    def _read_line(self, timeout=None, inter_byte_timeout=None):
        if not self.is_connected():
            return ""

        try:
            line_timeout = self.timeout if timeout is None else timeout
            return read_complete_serial_line(
                self.ser,
                timeout=max(0.0, float(line_timeout)),
                inter_byte_timeout=inter_byte_timeout,
            )
        except Exception as exc:
            print(f"[SensorAPI] Failed to read from sensor serial port: {exc}")
            self.close()
            return ""

    @staticmethod
    def _parse_ints(text):
        return parse_serial_ints(text)

    @staticmethod
    def _extract_sensor_frame(data_list):
        """Extract the latest complete 55555...44444 framed payload."""
        return extract_sensor_frame(data_list)

    # ------------------------------------------------------------------
    # Command API
    # ------------------------------------------------------------------
    def send_command(self, command):
        if not self._write_line(command):
            return None
        return self.read_response(command)

    def channel_check(self):
        return self.send_command("channelCheck")

    def update_cal(self):
        return self.send_command("updateCal")

    def read_cal(self):
        return self.send_command("readCal")

    def read_raw(self):
        return self.send_command("readRaw")

    def measure_read_raw_hz(self, duration_sec=1.0):
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        if not self._ensure_connection():
            return None

        start_time = time.perf_counter()
        success_count = 0
        total_attempts = 0

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= duration_sec:
                break

            total_attempts += 1
            payload = self.read_raw()
            if payload is not None:
                success_count += 1

        elapsed = max(time.perf_counter() - start_time, 1e-9)
        return {
            "hz": success_count / elapsed,
            "success_count": success_count,
            "total_attempts": total_attempts,
            "elapsed_sec": elapsed,
        }

    def stop(self):
        return self.send_command("stop")

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def read_response(self, command, timeout=None):
        if not self.is_connected():
            return None

        sensor_frame_command = command in {"readRaw", "readCal", "updateCal"}
        if timeout is None:
            timeout = self.response_timeout
            if sensor_frame_command:
                timeout = sensor_response_timeout_for_values(
                    self.expected_payload_values or 0,
                    baud_rate=getattr(self.ser, "baudrate", self.baud_rate),
                    minimum=max(
                        DEFAULT_SENSOR_RESPONSE_TIMEOUT_SEC,
                        float(timeout),
                    ),
                )
        timeout = max(0.05, float(timeout))
        deadline = time.perf_counter() + timeout
        pending = []

        while time.perf_counter() < deadline:
            remaining = max(0.0, deadline - time.perf_counter())
            response_start_grace = (
                SENSOR_UPDATE_CAL_START_GRACE_SEC
                if command == "updateCal"
                else SENSOR_RESPONSE_START_GRACE_SEC
            )
            if (
                sensor_frame_command
                and not wait_for_serial_input(
                    self.ser,
                    timeout=min(
                        remaining,
                        response_start_grace,
                    ),
                )
            ):
                return None
            remaining = max(0.0, deadline - time.perf_counter())
            response = self._read_line(
                timeout=remaining,
                inter_byte_timeout=(
                    SENSOR_RESPONSE_IDLE_TIMEOUT_SEC
                    if sensor_frame_command
                    else None
                ),
            )
            if not response:
                return None

            data_list = self._parse_ints(response)

            if sensor_frame_command:
                # USB CDC devices can print a boot/debug line immediately
                # after the serial port opens.  That text is not the response
                # to our command, so keep waiting for a framed numeric packet
                # instead of returning an empty payload and failing sensor
                # calibration. Short remnants are accumulated, not accepted.
                if not data_list:
                    continue
                fresh_line = not pending
                pending.extend(data_list)
                saw_frame_end = has_sensor_frame_end(pending)
                payload, pending = take_complete_sensor_payload(
                    pending,
                    self.expected_payload_values,
                    allow_unframed=fresh_line,
                )
                if payload is None:
                    if saw_frame_end:
                        return None
                    if (
                        has_sensor_frame_start(pending)
                        and not wait_for_serial_input(
                            self.ser,
                            timeout=SENSOR_FRAME_CONTINUATION_GRACE_SEC,
                        )
                    ):
                        return None
                    pending = trim_sensor_payload_accumulator(
                        pending,
                        self.expected_payload_values,
                    )
                    continue
                return payload
            if command in {"channelCheck", "stop"}:
                if not data_list:
                    continue
                return data_list

        return None

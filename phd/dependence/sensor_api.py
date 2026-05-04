import time

try:
    import serial
except ImportError:
    serial = None


class ArduinoCommander:
    """Safe serial wrapper for the tactile sensor controller."""

    def __init__(self, serial_port="/dev/ttyACM0", baud_rate=9600, timeout=0.1):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self._connect()

    # ------------------------------------------------------------------
    # Connection / lifecycle
    # ------------------------------------------------------------------
    def _connect(self):
        """Try to open the serial connection. Safe to call multiple times."""
        if serial is None:
            print("[SensorAPI] pyserial is not installed. Sensor API is unavailable.")
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
            return True
        except serial.SerialException as exc:
            print(f"[SensorAPI] Serial port {self.serial_port} not available: {exc}")
            self.ser = None
            return False

    def reconnect(self):
        self.close()
        return self._connect()

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

    def _read_line(self):
        if not self.is_connected():
            return ""

        try:
            return self.ser.readline().decode("utf-8", errors="ignore").rstrip()
        except Exception as exc:
            print(f"[SensorAPI] Failed to read from sensor serial port: {exc}")
            self.close()
            return ""

    @staticmethod
    def _parse_ints(text):
        values = []
        for token in text.split():
            try:
                values.append(int(token))
            except ValueError:
                continue
        return values

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
    def read_response(self, command, timeout=2):
        if not self.is_connected():
            return None

        start_time = time.time()

        while (time.time() - start_time) < timeout:
            try:
                if self.ser.in_waiting <= 0:
                    time.sleep(0.01)
                    continue
            except Exception:
                self.close()
                return None

            response = self._read_line()
            if not response:
                continue

            data_list = self._parse_ints(response)

            if command in {"readRaw", "readCal", "updateCal"}:
                return data_list[2:-2] if len(data_list) >= 4 else data_list
            if command in {"channelCheck", "stop"}:
                return data_list

        return None

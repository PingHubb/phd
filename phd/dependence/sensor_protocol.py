"""Protocol constants and parsing shared by tactile sensor transports."""

import time


DEFAULT_SENSOR_BAUD_RATE = 9600
DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC = 0.1
DEFAULT_SENSOR_RESPONSE_TIMEOUT_SEC = 0.5
DEFAULT_SENSOR_IDLE_SLEEP_SEC = 0.002
SENSOR_READ_RAW_COMMAND = b"readRaw\n"

_SERIAL_LINE_BUFFER_ATTR = "_pinglab_serial_line_buffer"
_SERIAL_BITS_PER_BYTE = 10.0
_MAX_ASCII_CHARS_PER_VALUE = 6
_SENSOR_FRAME_OVERHEAD_VALUES = 4


def sensor_response_timeout_for_values(
    value_count,
    baud_rate=DEFAULT_SENSOR_BAUD_RATE,
    minimum=DEFAULT_SENSOR_RESPONSE_TIMEOUT_SEC,
):
    """Return a conservative deadline for one ASCII sensor frame.

    Each integer can occupy five digits plus a separator, while an 8-N-1
    serial byte takes ten wire bits. The result is only a deadline: a frame
    that arrives sooner is returned immediately.
    """
    try:
        value_count = max(0, int(value_count))
        baud_rate = max(1.0, float(baud_rate))
        minimum = max(0.05, float(minimum))
    except (TypeError, ValueError):
        return max(0.05, float(DEFAULT_SENSOR_RESPONSE_TIMEOUT_SEC))

    estimated_bytes = (
        (value_count + _SENSOR_FRAME_OVERHEAD_VALUES)
        * _MAX_ASCII_CHARS_PER_VALUE
        + 1
    )
    wire_time = estimated_bytes * _SERIAL_BITS_PER_BYTE / baud_rate
    return max(minimum, wire_time * 1.5 + 0.25)


def read_complete_serial_line(
    serial_port,
    timeout,
    idle_sleep_sec=DEFAULT_SENSOR_IDLE_SLEEP_SEC,
    should_continue=None,
):
    """Read one newline-terminated line, accumulating partial serial reads."""
    deadline = time.perf_counter() + max(0.0, float(timeout))
    try:
        buffered = getattr(serial_port, _SERIAL_LINE_BUFFER_ATTR, b"")
    except Exception:
        buffered = b""
    if isinstance(buffered, str):
        buffered = buffered.encode("utf-8", errors="ignore")
    buffer = bytearray(buffered or b"")

    def store_remainder(payload):
        try:
            setattr(serial_port, _SERIAL_LINE_BUFFER_ATTR, bytes(payload))
        except Exception:
            pass

    while time.perf_counter() < deadline:
        newline_index = buffer.find(b"\n")
        if newline_index >= 0:
            line = bytes(buffer[:newline_index]).rstrip(b"\r")
            store_remainder(buffer[newline_index + 1:])
            return line.decode("utf-8", errors="ignore")

        if should_continue is not None and not bool(should_continue()):
            store_remainder(b"")
            return ""

        try:
            waiting = int(getattr(serial_port, "in_waiting", 0))
        except Exception:
            store_remainder(b"")
            raise

        if waiting <= 0:
            time.sleep(max(0.0, float(idle_sleep_sec)))
            continue

        reader = getattr(serial_port, "read", None)
        if callable(reader):
            chunk = reader(waiting)
        else:
            chunk = serial_port.readline()
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", errors="ignore")
        if chunk:
            buffer.extend(chunk)

    # A partial response must not be parsed or joined to the next command.
    store_remainder(b"")
    return ""


def parse_serial_ints(text):
    """Return all integer tokens from a whitespace-delimited device line."""
    values = []
    for token in str(text).split():
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values

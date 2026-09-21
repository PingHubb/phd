"""Protocol constants and parsing shared by tactile sensor transports."""

import time


DEFAULT_SENSOR_BAUD_RATE = 9600
DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC = 0.1
DEFAULT_SENSOR_RESPONSE_TIMEOUT_SEC = 0.5
DEFAULT_SENSOR_IDLE_SLEEP_SEC = 0.002
SENSOR_RESPONSE_START_GRACE_SEC = 0.1
SENSOR_RESPONSE_IDLE_TIMEOUT_SEC = 0.1
SENSOR_FRAME_CONTINUATION_GRACE_SEC = 0.05
SENSOR_UPDATE_CAL_START_GRACE_SEC = 0.25
SENSOR_READ_RAW_COMMAND = b"readRaw\n"
SENSOR_FRAME_START = (55555, 55555)
SENSOR_FRAME_END = (44444, 44444)
_SENSOR_ACCUMULATOR_FLOOR = 32

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
    inter_byte_timeout=None,
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
    last_data_time = time.perf_counter() if buffer else None
    if inter_byte_timeout is not None:
        inter_byte_timeout = max(0.001, float(inter_byte_timeout))

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
            if (
                buffer
                and inter_byte_timeout is not None
                and last_data_time is not None
                and (time.perf_counter() - last_data_time)
                >= inter_byte_timeout
            ):
                store_remainder(b"")
                return ""
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
            last_data_time = time.perf_counter()

    # A partial response must not be parsed or joined to the next command.
    store_remainder(b"")
    return ""


def discard_pending_serial_input(serial_port):
    """Drop stale USB/UART bytes so the next command is not mixed with leftovers."""
    reset = getattr(serial_port, "reset_input_buffer", None)
    if callable(reset):
        try:
            reset()
            return True
        except Exception:
            return False
    return False


def wait_for_serial_input(
    serial_port,
    timeout=SENSOR_FRAME_CONTINUATION_GRACE_SEC,
    idle_sleep_sec=DEFAULT_SENSOR_IDLE_SLEEP_SEC,
    should_continue=None,
):
    """Briefly wait for buffered bytes that could continue a split frame."""
    deadline = time.perf_counter() + max(0.0, float(timeout))
    while time.perf_counter() < deadline:
        try:
            buffered = getattr(serial_port, _SERIAL_LINE_BUFFER_ATTR, b"")
            waiting = int(getattr(serial_port, "in_waiting", 0))
        except Exception:
            return False
        if buffered or waiting > 0:
            return True
        if should_continue is not None and not bool(should_continue()):
            return False
        time.sleep(max(0.0, float(idle_sleep_sec)))
    return False


def parse_serial_ints(text):
    """Return all integer tokens from a whitespace-delimited device line."""
    values = []
    for token in str(text).split():
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values


def extract_sensor_frame(data_list):
    """Return the latest complete 55555…44444 payload, or ``None``."""
    payload, _remaining = take_complete_sensor_payload(data_list)
    return payload


def has_sensor_frame_end(data_list):
    """Return whether tokenized input contains the complete frame trailer."""
    values = list(data_list or [])
    return any(
        tuple(values[index:index + 2]) == SENSOR_FRAME_END
        for index in range(len(values) - 1)
    )


def has_sensor_frame_start(data_list):
    """Return whether tokenized input contains the complete frame header."""
    values = list(data_list or [])
    return any(
        tuple(values[index:index + 2]) == SENSOR_FRAME_START
        for index in range(len(values) - 1)
    )


def take_complete_sensor_payload(
    data_list,
    expected_payload_values=None,
    allow_unframed=False,
):
    """Pull one usable sensor payload from accumulated integer tokens.

    Prefers a ``55555 55555 … 44444 44444`` frame. When ``expected_payload_values``
    is set, a complete frame of the wrong length is discarded so a short USB
    remnant cannot be treated as a sample. A headerless line whose length is
    exactly ``expected + 4`` is accepted only when ``allow_unframed`` is true,
    so leftover integers from a previous USB remnant cannot be concatenated
    into a fake sample.

    Returns ``(payload, remaining_tokens)``. ``payload`` is ``None`` until a
    usable frame is present.
    """
    values = list(data_list or [])
    try:
        expected = int(expected_payload_values or 0)
    except (TypeError, ValueError):
        expected = 0

    frames = []
    active_start = None
    index = 0
    while index + 1 < len(values):
        marker = tuple(values[index:index + 2])
        if marker == SENSOR_FRAME_START:
            # A new header supersedes an older unterminated frame. Without
            # this resynchronization, the old fragment and the next valid
            # frame are combined and both are discarded for the wrong length.
            active_start = index
            index += 2
            continue
        if marker == SENSOR_FRAME_END and active_start is not None:
            payload = values[active_start + 2:index]
            if payload:
                frames.append((payload, index + 2))
            active_start = None
            index += 2
            continue
        index += 1

    if frames:
        if expected > 0:
            matching = [
                item for item in frames if len(item[0]) == expected
            ]
            if matching:
                payload, consumed = matching[-1]
                return payload, values[consumed:]
            return None, values[frames[-1][1]:]
        payload, consumed = frames[-1]
        return payload, values[consumed:]

    if (
        allow_unframed
        and expected > 0
        and len(values) == expected + 4
    ):
        return values[2:-2], []
    return None, values


def trim_sensor_payload_accumulator(data_list, expected_payload_values=None):
    """Keep only the tokens that can still complete a frame."""
    values = list(data_list or [])
    try:
        expected = int(expected_payload_values or 0)
    except (TypeError, ValueError):
        expected = 0
    keep = max(expected + 4, _SENSOR_ACCUMULATOR_FLOOR)
    for index in range(len(values) - 2, -1, -1):
        if tuple(values[index:index + 2]) == SENSOR_FRAME_START:
            candidate = values[index:]
            return candidate if len(candidate) <= keep else candidate[-keep:]
    return values[-keep:]

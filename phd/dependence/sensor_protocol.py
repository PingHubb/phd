"""Protocol constants and parsing shared by tactile sensor transports."""


DEFAULT_SENSOR_BAUD_RATE = 9600
DEFAULT_SENSOR_SERIAL_TIMEOUT_SEC = 0.1
DEFAULT_SENSOR_RESPONSE_TIMEOUT_SEC = 0.5
DEFAULT_SENSOR_IDLE_SLEEP_SEC = 0.002
SENSOR_READ_RAW_COMMAND = b"readRaw\n"


def parse_serial_ints(text):
    """Return all integer tokens from a whitespace-delimited device line."""
    values = []
    for token in str(text).split():
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values

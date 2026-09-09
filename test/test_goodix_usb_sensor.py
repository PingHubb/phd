import numpy as np
import pytest

from phd.dependence.goodix_usb_sensor import (
    GOODIX_USB_SOURCE_ID,
    GOODIX_USB_SOURCE_LABEL,
    GoodixLibusbSession,
    GoodixUsbError,
    build_goodix_raw_read_command,
    decode_goodix_raw_packets,
    is_goodix_usb_source,
    parse_goodix_raw_matrix,
)


def _matrix_output(matrix):
    lines = ["[CMD]: send SendRawDataCmd cmd", "Rawdata:"]
    lines.extend(" ".join(f"{value:5d}" for value in row) for row in matrix)
    lines.append("Finish")
    return "\n".join(lines)


def test_goodix_source_recognizes_ui_label_and_stable_id():
    assert is_goodix_usb_source(GOODIX_USB_SOURCE_ID)
    assert is_goodix_usb_source(GOODIX_USB_SOURCE_LABEL)
    assert not is_goodix_usb_source("ttyACM0")


def test_goodix_raw_parser_converts_display_rows_to_column_major_payload():
    matrix = np.arange(80).reshape(10, 8)

    values = parse_goodix_raw_matrix(_matrix_output(matrix))

    assert values == matrix.T.reshape(-1).tolist()


def test_goodix_raw_parser_accepts_transposed_vendor_output():
    matrix = np.arange(80).reshape(10, 8)

    values = parse_goodix_raw_matrix(_matrix_output(matrix.T))

    assert values == matrix.T.reshape(-1).tolist()


def test_goodix_raw_parser_rejects_incomplete_frames():
    with pytest.raises(GoodixUsbError, match="10x8"):
        parse_goodix_raw_matrix("1 2 3\n4 5 6")


def test_goodix_raw_read_command_encodes_unsigned_16_bit_byte_count():
    assert build_goodix_raw_read_command(80) == bytes.fromhex(
        "03 0f 01 a6 7c 00 a0"
    )


def test_goodix_packet_decoder_skips_headers_and_decodes_little_endian():
    expected = list(range(80))
    payload = b"".join(value.to_bytes(2, "little") for value in expected)
    packets = [
        bytes([packet_index]) + payload[offset:offset + 63]
        for packet_index, offset in enumerate(range(0, len(payload), 63))
    ]

    assert decode_goodix_raw_packets(packets, value_count=80) == expected


def test_goodix_packet_decoder_rejects_incomplete_payload():
    with pytest.raises(GoodixUsbError, match="expected 160"):
        decode_goodix_raw_packets([b"\x00" + (b"\x01" * 63)], value_count=80)


def test_goodix_libusb_session_enters_raw_mode_only_once():
    session = GoodixLibusbSession()
    mode_entries = []
    session.open = lambda: None

    def enter_raw_mode():
        mode_entries.append(True)
        session._raw_mode_ready = True

    session._enter_raw_mode = enter_raw_mode
    session._read_frame = lambda: list(range(80))

    assert session.read_raw() == list(range(80))
    assert session.read_raw() == list(range(80))
    assert len(mode_entries) == 1


def test_goodix_blb_raw_mode_waits_for_controller_acknowledgement():
    session = GoodixLibusbSession()
    writes = []
    waits = []
    responses = iter(
        [
            bytes.fromhex("02 00"),
            bytes.fromhex("02 80"),
        ]
    )
    session._write = writes.append
    session._read = lambda: next(responses)
    session._wait = waits.append

    session._enter_raw_mode()

    assert writes == [
        bytes.fromhex("03 86 01 01 74 00 06 00 00 04 02 06"),
        bytes.fromhex("03 0f 01 01 74 00 01 00"),
        bytes.fromhex("03 0f 01 01 74 00 01 00"),
    ]
    assert len(waits) == 2
    assert session._raw_mode_active is True
    assert session._raw_mode_ready is True


def test_goodix_blb_frame_read_preserves_column_major_payload_and_clears_flag():
    session = GoodixLibusbSession(rows=10, columns=8)
    writes = []
    column_major = np.arange(80, dtype=np.uint16)
    payload = b"".join(
        int(value).to_bytes(2, "little") for value in column_major
    )
    packets = [
        bytes([2]) + payload[offset:offset + 63]
        for offset in range(0, len(payload), 63)
    ]
    responses = iter([bytes.fromhex("02 80"), *packets])
    session._write = writes.append
    session._read = lambda: next(responses)

    values = session._read_frame()

    expected = column_major.astype(int).tolist()
    assert values == expected
    assert writes == [
        bytes.fromhex("03 0f 01 02 74 00 01"),
        bytes.fromhex("03 0f 01 a6 7c 00 a0"),
        bytes.fromhex("03 8f 01 02 74 00 01"),
    ]

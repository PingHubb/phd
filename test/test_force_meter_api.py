import math
import struct

from phd.dependence.force_meter_api import (
    HP200_REGISTER_COUNT,
    Hp200ModbusRtuParser,
    Hp200StreamParser,
    build_hp200_read_request,
    decode_hp200_registers,
    force_to_newtons,
    modbus_crc16,
    parse_force_text,
)


def test_parse_signed_newton_reading():
    reading = parse_force_text("+123.4 N\r\n")

    assert reading is not None
    assert reading.native_value == 123.4
    assert reading.native_unit == "N"
    assert reading.force_newtons == 123.4


def test_parse_unit_before_value_and_convert_kgf():
    reading = parse_force_text("Force kgf: -2.50")

    assert reading is not None
    assert reading.native_unit == "kgf"
    assert math.isclose(reading.force_newtons, -24.516625)


def test_bare_force_value_defaults_to_newtons():
    reading = parse_force_text("-0.7")

    assert reading is not None
    assert reading.native_unit == "N"
    assert reading.force_newtons == -0.7


def test_stream_parser_handles_fragmented_lines():
    parser = Hp200StreamParser()

    assert parser.feed(b"+12") == []
    readings = parser.feed(b".3N\r\n-4.5 N\r\n")

    assert [reading.force_newtons for reading in readings] == [12.3, -4.5]


def test_stream_parser_handles_compact_records_without_newlines():
    parser = Hp200StreamParser()

    readings = parser.feed(b"+001.2N-000.8N")

    assert [reading.force_newtons for reading in readings] == [1.2, -0.8]


def test_supported_force_unit_conversions():
    assert math.isclose(force_to_newtons(1.0, "lbf"), 4.4482216152605)
    assert math.isclose(force_to_newtons(1000.0, "mN"), 1.0)
    assert math.isclose(force_to_newtons(1000.0, "gf"), 9.80665)


def _hp200_registers(force_value, unit_code=0):
    high_word, low_word = struct.unpack(">HH", struct.pack(">f", force_value))
    registers = [0] * HP200_REGISTER_COUNT
    registers[0:2] = [high_word, low_word]
    registers[11] = unit_code
    return registers


def _modbus_response(registers, slave_address=1):
    payload = bytes((slave_address, 3, len(registers) * 2)) + struct.pack(
        ">" + ("H" * len(registers)), *registers
    )
    return payload + struct.pack("<H", modbus_crc16(payload))


def test_official_hp200_read_request():
    assert build_hp200_read_request().hex(" ") == "01 03 00 00 00 0d 84 0f"


def test_decode_hp200_registers_uses_high_word_first_ieee_float():
    reading = decode_hp200_registers(_hp200_registers(-12.5))

    assert reading.native_value == -12.5
    assert reading.native_unit == "N"
    assert reading.force_newtons == -12.5


def test_decode_hp200_registers_converts_selected_meter_unit():
    reading = decode_hp200_registers(_hp200_registers(1.25, unit_code=1))

    assert reading.native_unit == "kN"
    assert reading.force_newtons == 1250.0


def test_modbus_parser_handles_fragmented_reply():
    parser = Hp200ModbusRtuParser()
    response = _modbus_response(_hp200_registers(37.25))

    assert parser.feed(response[:7]) == []
    readings = parser.feed(response[7:])

    assert len(readings) == 1
    assert readings[0].force_newtons == 37.25


def test_modbus_parser_rejects_bad_crc():
    parser = Hp200ModbusRtuParser()
    response = bytearray(_modbus_response(_hp200_registers(5.0)))
    response[-1] ^= 0xFF

    assert parser.feed(response) == []
    assert "CRC mismatch" in parser.pop_notice()

import serial
import sys
import time


class ArduinoCommander:
    def __init__(self, serial_port='/dev/ttyACM0', baud_rate=9600):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        try:
            self.ser = serial.Serial(serial_port, baud_rate)
        except serial.SerialException:
            print(f"Serial port {serial_port} not found.")
            # sys.exit(1)

    def send_command(self, command):
        full_command = command + '\n'
        self.ser.write(full_command.encode())
        return self.read_response(command)  # Pass the command as an argument

    def channel_check(self):
        return self.send_command("channelCheck")

    def update_cal(self):
        return self.send_command("updateCal")

    def read_cal(self):
        self.send_command("readCal")

    def read_raw(self):
        return self.send_command("readRaw")

    def stop(self):
        self.send_command("stop")

    def close(self):
        self.ser.close()

    def read_response(self, command, timeout=2):
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.ser.in_waiting > 0:
                response = self.ser.readline().decode('utf-8').rstrip()
                # Optionally, process the response into a list
                data_list = [int(value) for value in response.split() if value.isdigit()]

                if command == "readRaw":
                    # print(f"Data length: {len(data_list)}")
                    return data_list[2:-2]
                elif command == "readCal":
                    # print(f"Response from {command}: {data_list[2:-2]}")
                    return data_list[2:-2]
                elif command == "updateCal":
                    # print(f"Response from {command}: {data_list[2:-2]}")
                    return data_list[2:-2]
                elif command == "channelCheck":
                    # print(f"Response from {command}: {data_list}")
                    return data_list
        return None

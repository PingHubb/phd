from pymycobot.mycobot280 import MyCobot280
import time

class MyCobotAPI:
    def __init__(self, serial_port="/dev/ttyACM1", baud_rate: int = 115200):
        self.mc = MyCobot280(serial_port, baud_rate)
        self.set_fresh_mode(1)
        print("Fresh mode:", self.get_fresh_mode())
    def stop(self):
        self.mc.stop()

    def pause(self):
        return self.mc.pause()

    def resume(self):
        self.mc.resume()

    def get_current_coords(self):
        return self.mc.get_coords()

    def move_to_coords(self, coords: list, speed: int, mode: int = 1):
        self.mc.send_coords(coords, speed, mode)

    def sync_send_coords(self, coords, speed, mode=0, timeout=0.1):
        self.mc.sync_send_coords(coords, speed, mode, timeout)

    def move_single_angle(self, index: int, value: float, speed: int):
        self.mc.send_angle(index, value, speed)

    def wait(self, seconds: float):
        time.sleep(seconds)

    def set_encoders(self, encoders, sp):
        self.mc.set_encoders(encoders, sp)

    def release_all_servos(self):
        self.mc.release_all_servos()


    def set_fresh_mode(self, mode):
        self.mc.set_fresh_mode(mode)

    def get_fresh_mode(self):
        return self.mc.get_fresh_mode()

    def send_angles(self, angles, speed):
        self.mc.send_angles(angles, speed)

    def get_angles(self):
        return self.mc.get_angles()


def main():
    # Initialize the API.
    # start_time = time.time()
    api = MyCobotAPI("/dev/ttyACM1", 115200)
    # print("Time elapsed for command: {:.3f} seconds".format(time.time() - start_time))

    angle_list = [0, 0, 0, 0, 0, 0]
    print(api.get_angles())
    api.send_angles(angle_list, 80)

    counter = 0

    while True:

        choice = input("Enter:......").strip()

        if choice == '1':
            counter += 168
            print(f"Counter increased by {counter}.")
            # Calculate new angle.
            new_angle_list = [angle_list[0] + counter, 0, 0, 0, 0, 0]
            print("Sending coordinates:", new_angle_list)

            start_time = time.time()
            api.send_angles(new_angle_list, speed=80)
            elapsed_time = time.time() - start_time
            print("Time elapsed for command: {:.3f} seconds".format(elapsed_time))
        elif choice == '2':
            counter -= 168
            print(f"Counter decreased by {counter}.")
            # Calculate new angle.
            new_angle_list = [angle_list[0] + counter, 0, 0, 0, 0, 0]
            print("Sending coordinates:", new_angle_list)

            start_time = time.time()
            api.send_angles(new_angle_list, speed=80)
            elapsed_time = time.time() - start_time
            print("Time elapsed for command: {:.3f} seconds".format(elapsed_time))
        elif choice.lower() == '3':
            api.pause()
            print("Robot stopped.")
        elif choice == '4':
            print(api.get_angles())
        else:
            print("Invalid input. Please try again.")
            continue


if __name__ == "__main__":
    main()



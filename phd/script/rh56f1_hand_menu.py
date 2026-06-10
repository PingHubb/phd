#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from service_interfaces.srv import Setangle, Setspeed, Setforce, Getangleact


class RH56F1Menu(Node):
    def __init__(self):
        super().__init__("rh56f1_hand_menu")

        self.hand_id = 1

        self.set_angle_client = self.create_client(Setangle, "/Setangle")
        self.set_speed_client = self.create_client(Setspeed, "/Setspeed")
        self.set_force_client = self.create_client(Setforce, "/Setforce")
        self.get_angle_client = self.create_client(Getangleact, "/Getangleact")

        self.wait_for_services()

        # RH56F1 mapping:
        # angle0 = little finger
        # angle1 = ring finger
        # angle2 = middle finger
        # angle3 = index finger
        # angle4 = thumb bending
        # angle5 = thumb rotation
        self.fingers = {
            "1": {"name": "Little finger", "index": 0, "open": 1720, "close": 900},
            "2": {"name": "Ring finger", "index": 1, "open": 1720, "close": 900},
            "3": {"name": "Middle finger", "index": 2, "open": 1720, "close": 900},
            "4": {"name": "Index finger", "index": 3, "open": 1720, "close": 900},
            "5": {"name": "Thumb bending", "index": 4, "open": 1350, "close": 1100},
        }

        # Last force values remembered by this script.
        # Force range normally: 0 to 12000.
        # Start with safe moderate force.
        self.force_values = [2000, 2000, 2000, 2000, 2000, 2000]

        # Thumb rotation values.
        # If direction looks opposite on your hand, swap these two values.
        self.thumb_rotate_left = 600
        self.thumb_rotate_right = 1800
        self.thumb_rotate_center = 1000

    def wait_for_services(self):
        print("Waiting for RH56F1 services...")

        self.set_angle_client.wait_for_service()
        print("Found /Setangle")

        self.set_speed_client.wait_for_service()
        print("Found /Setspeed")

        self.set_force_client.wait_for_service()
        print("Found /Setforce")

        self.get_angle_client.wait_for_service()
        print("Found /Getangleact")

    def call_set_angle(self, angles):
        request = Setangle.Request()
        request.status = "set_angle"
        request.hand_id = self.hand_id

        request.angle0 = int(angles[0])
        request.angle1 = int(angles[1])
        request.angle2 = int(angles[2])
        request.angle3 = int(angles[3])
        request.angle4 = int(angles[4])
        request.angle5 = int(angles[5])

        future = self.set_angle_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            print("Setangle response:")
            print(future.result())
        else:
            print("Setangle failed.")

    def call_set_speed(self, speed):
        request = Setspeed.Request()
        request.status = "set_speed"
        request.hand_id = self.hand_id

        request.speed0 = int(speed)
        request.speed1 = int(speed)
        request.speed2 = int(speed)
        request.speed3 = int(speed)
        request.speed4 = int(speed)
        request.speed5 = int(speed)

        future = self.set_speed_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            print("Setspeed response:")
            print(future.result())
        else:
            print("Setspeed failed.")

    def call_set_force(self, forces):
        request = Setforce.Request()
        request.status = "set_force"
        request.hand_id = self.hand_id

        request.force0 = int(forces[0])
        request.force1 = int(forces[1])
        request.force2 = int(forces[2])
        request.force3 = int(forces[3])
        request.force4 = int(forces[4])
        request.force5 = int(forces[5])

        future = self.set_force_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            print("Setforce response:")
            print(future.result())
        else:
            print("Setforce failed.")

    def call_get_angle(self):
        request = Getangleact.Request()
        request.status = "get_angleact"
        request.hand_id = self.hand_id

        future = self.get_angle_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            print("Actual angle response:")
            print(future.result())
        else:
            print("Getangleact failed.")

    def open_all(self):
        print("Opening all fingers...")
        angles = [
            1720,  # little
            1720,  # ring
            1720,  # middle
            1720,  # index
            1350,  # thumb bending
            -1,    # thumb rotation unchanged
        ]
        self.call_set_angle(angles)

    def close_all(self):
        print("Closing all fingers...")
        angles = [
            900,   # little
            900,   # ring
            900,   # middle
            900,   # index
            1100,  # thumb bending
            -1,    # thumb rotation unchanged
        ]
        self.call_set_angle(angles)

    def selected_fingers_action(self, action):
        print("")
        print("Select finger(s). Example:")
        print("  1       = little finger only")
        print("  1 2 3   = little, ring, middle")
        print("  4 5     = index and thumb bending")
        print("")
        self.print_finger_list()

        user_input = input("Enter finger number(s): ").strip()
        selected = user_input.split()

        angles = [-1, -1, -1, -1, -1, -1]

        for key in selected:
            if key not in self.fingers:
                print(f"Invalid finger number: {key}")
                continue

            finger = self.fingers[key]
            idx = finger["index"]
            angles[idx] = finger[action]

            print(f"{action.capitalize()} command: {finger['name']} -> {angles[idx]}")

        self.call_set_angle(angles)

    def rotate_thumb_left(self):
        print("Rotating thumb left...")
        angles = [-1, -1, -1, -1, -1, self.thumb_rotate_left]
        self.call_set_angle(angles)

    def rotate_thumb_right(self):
        print("Rotating thumb right...")
        angles = [-1, -1, -1, -1, -1, self.thumb_rotate_right]
        self.call_set_angle(angles)

    def rotate_thumb_center(self):
        print("Rotating thumb to centre...")
        angles = [-1, -1, -1, -1, -1, self.thumb_rotate_center]
        self.call_set_angle(angles)

    def custom_thumb_rotation(self):
        print("")
        print("Custom thumb rotation")
        print("Suggested test values:")
        print("  600  = one side")
        print("  1000 = middle")
        print("  1800 = other side")
        print("")

        value = input("Enter thumb rotation angle5 value: ").strip()

        try:
            angle5 = int(value)
        except ValueError:
            print("Invalid number.")
            return

        angles = [-1, -1, -1, -1, -1, angle5]
        self.call_set_angle(angles)

    def custom_angle(self):
        print("")
        print("Custom angle mode")
        print("Use -1 if you do not want to move that finger.")
        print("")
        print("Suggested values:")
        print("  Little/Ring/Middle/Index: close=900, open=1720")
        print("  Thumb bending: close=1100, open=1350")
        print("  Thumb rotation: 600 / 1000 / 1800")
        print("")

        names = [
            "Little finger angle0",
            "Ring finger angle1",
            "Middle finger angle2",
            "Index finger angle3",
            "Thumb bending angle4",
            "Thumb rotation angle5",
        ]

        angles = []

        for name in names:
            value = input(f"{name}, blank = -1: ").strip()
            if value == "":
                value = "-1"

            try:
                angles.append(int(value))
            except ValueError:
                print("Invalid number. Cancelled.")
                return

        self.call_set_angle(angles)

    def custom_angle_single(self):
        """Adjust ONE finger's angle on its own. Useful for fine-tuning a
        single joint without having to retype the other five entries.

        Type ``q`` (or just press Enter on an empty prompt) at the finger
        selection step to bail out without sending anything.
        """
        single_finger_choices = {
            "1": ("Little finger", 0, "close=900, open=1720"),
            "2": ("Ring finger", 1, "close=900, open=1720"),
            "3": ("Middle finger", 2, "close=900, open=1720"),
            "4": ("Index finger", 3, "close=900, open=1720"),
            "5": ("Thumb bending", 4, "close=1100, open=1350"),
            "6": ("Thumb rotation", 5, "600 / 1000 / 1800"),
        }

        print("")
        print("Custom angle (single finger)")
        print("Pick exactly ONE actuator to move; all others stay put.")
        print("")
        for key, (name, idx, hint) in single_finger_choices.items():
            print(f"  {key} = {name:<16} (angle{idx}, suggested {hint})")
        print("")

        choice = input("Select finger number (or q to cancel): ").strip().lower()
        if not choice or choice in {"q", "quit", "exit"}:
            print("Cancelled.")
            return
        if choice not in single_finger_choices:
            print(f"Invalid finger number: {choice}")
            return

        name, idx, hint = single_finger_choices[choice]
        value = input(f"{name} (angle{idx}) — enter target value [{hint}]: ").strip()
        if value == "":
            print("Cancelled (empty value).")
            return
        try:
            target_angle = int(value)
        except ValueError:
            print("Invalid number. Cancelled.")
            return

        angles = [-1, -1, -1, -1, -1, -1]
        angles[idx] = target_angle
        print(f"Sending {name} angle{idx} -> {target_angle} "
              f"(other fingers unchanged).")
        self.call_set_angle(angles)

    def set_force_all(self):
        print("")
        print("Set force for all fingers")
        print("Suggested safe test range: 1000 to 3000")
        print("Maximum range may be up to 12000, but do not use high force at first.")
        print("")

        value = input("Enter force value for all fingers: ").strip()

        try:
            force = int(value)
        except ValueError:
            print("Invalid number.")
            return

        self.force_values = [force, force, force, force, force, force]
        self.call_set_force(self.force_values)

    def set_force_selected(self):
        print("")
        print("Set force for selected finger(s)")
        print("Suggested safe test range: 1000 to 3000")
        print("")
        self.print_finger_list()
        print("  6 = Thumb rotation actuator")
        print("")

        selected_input = input("Enter finger number(s): ").strip()
        selected = selected_input.split()

        value = input("Enter force value: ").strip()

        try:
            force = int(value)
        except ValueError:
            print("Invalid force number.")
            return

        for key in selected:
            if key in self.fingers:
                idx = self.fingers[key]["index"]
                self.force_values[idx] = force
                print(f"Set force: {self.fingers[key]['name']} -> {force}")
            elif key == "6":
                self.force_values[5] = force
                print(f"Set force: Thumb rotation -> {force}")
            else:
                print(f"Invalid finger number: {key}")

        self.call_set_force(self.force_values)

    def set_speed_menu(self):
        print("")
        print("Set speed for all fingers")
        print("Suggested safe speed: 300 to 800")
        print("")

        value = input("Enter speed: ").strip()

        if value == "":
            value = "300"

        try:
            speed = int(value)
        except ValueError:
            print("Invalid number.")
            return

        self.call_set_speed(speed)

    def print_finger_list(self):
        print("Finger list:")
        print("  1 = Little finger")
        print("  2 = Ring finger")
        print("  3 = Middle finger")
        print("  4 = Index finger")
        print("  5 = Thumb bending")

    def print_mapping(self):
        print("")
        print("ROS2 angle mapping:")
        print("  angle0 = little finger")
        print("  angle1 = ring finger")
        print("  angle2 = middle finger")
        print("  angle3 = index finger")
        print("  angle4 = thumb bending")
        print("  angle5 = thumb rotation")
        print("")
        print("Open/close values:")
        print("  Little/Ring/Middle/Index: open=1720, close=900")
        print("  Thumb bending: open=1350, close=1100")
        print("  Thumb rotation: test 600 / 1000 / 1800")
        print("")
        print("Current remembered force values:")
        print(f"  {self.force_values}")

    def print_menu(self):
        print("")
        print("========== RH56F1 Dexterous Hand Menu ==========")
        print("1. Open all fingers")
        print("2. Close all fingers")
        print("3. Open selected finger(s)")
        print("4. Close selected finger(s)")
        print("5. Rotate thumb left")
        print("6. Rotate thumb right")
        print("7. Rotate thumb centre")
        print("8. Custom thumb rotation")
        print("9. Set custom angles (all 6 fingers in one go)")
        print("10. Set custom angle (single finger)")
        print("11. Read actual angles")
        print("12. Set speed")
        print("13. Set force for all fingers")
        print("14. Set force for selected finger(s)")
        print("15. Show finger mapping")
        print("q. Quit")
        print("===============================================")

    def run_menu(self):
        while rclpy.ok():
            self.print_menu()
            choice = input("Select option: ").strip().lower()

            if choice == "1":
                self.open_all()

            elif choice == "2":
                self.close_all()

            elif choice == "3":
                self.selected_fingers_action("open")

            elif choice == "4":
                self.selected_fingers_action("close")

            elif choice == "5":
                self.rotate_thumb_left()

            elif choice == "6":
                self.rotate_thumb_right()

            elif choice == "7":
                self.rotate_thumb_center()

            elif choice == "8":
                self.custom_thumb_rotation()

            elif choice == "9":
                self.custom_angle()

            elif choice == "10":
                self.custom_angle_single()

            elif choice == "11":
                self.call_get_angle()

            elif choice == "12":
                self.set_speed_menu()

            elif choice == "13":
                self.set_force_all()

            elif choice == "14":
                self.set_force_selected()

            elif choice == "15":
                self.print_mapping()

            elif choice == "q":
                print("Exiting.")
                break

            else:
                print("Invalid option.")


def main(args=None):
    rclpy.init(args=args)
    node = RH56F1Menu()

    try:
        node.run_menu()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
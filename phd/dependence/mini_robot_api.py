from pymycobot.mycobot280 import MyCobot280
from scipy.optimize import minimize
import numpy as np
import time

# =======================================================================
# MyCobot API Class
# =======================================================================
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


# =======================================================================
# Kinematics Functions
# =======================================================================
def dh_transformation(theta, d, a, alpha, offset=0):
    """
    Compute the DH transformation matrix for a given joint.

    Parameters:
      theta  : Joint angle (variable).
      d      : Offset along the previous z axis.
      a      : Link length.
      alpha  : Link twist.
      offset : Additional fixed angular offset (if any).

    Returns:
      A 4x4 numpy array representing the transformation.
    """
    theta_eff = theta + offset  # Effective joint angle (include offset)
    T = np.array([
                 [np.cos(theta_eff) , -np.sin(theta_eff) * np.cos(alpha)  , np.sin(theta_eff) * np.sin(alpha)   , a * np.cos(theta_eff)],
                 [np.sin(theta_eff) , np.cos(theta_eff) * np.cos(alpha)   , -np.cos(theta_eff) * np.sin(alpha)  , a * np.sin(theta_eff)],
                 [0                 , np.sin(alpha)                       , np.cos(alpha)                       , d                    ],
                 [0                 ,      0                              , 0                                   , 1                    ]
    ])
    return T


def forward_kinematics(joint_angles, dh_params):
    """
    Compute the overall transformation matrix from base to end-effector.

    Parameters:
      joint_angles : List or array of joint angles (one per joint).
      dh_params    : List of DH parameter dictionaries for each joint.

    Returns:
      A 4x4 numpy array representing the end-effector pose.
    """
    T_total = np.eye(4)
    for i, params in enumerate(dh_params):
        theta = joint_angles[i]
        d = params["d"]
        a = params["a"]
        alpha = params["alpha"]
        offset = params.get("offset", 0)
        T_i = dh_transformation(theta, d, a, alpha, offset)
        T_total = T_total @ T_i
    return T_total


def pose_error(joint_angles, target_pose, dh_params):
    """
    Calculate the error between the current end-effector position and the target position.

    Parameters:
      joint_angles : Current estimate of joint angles.
      target_pose  : Desired end-effector pose (4x4 matrix).
      dh_params    : DH parameters of the robot.

    Returns:
      Euclidean distance between the current and target positions.
    """
    T_current = forward_kinematics(joint_angles, dh_params)
    pos_current = T_current[0:3, 3]
    pos_target = target_pose[0:3, 3]
    error = np.linalg.norm(pos_target - pos_current)
    return error


def solve_ik(target_pose, initial_guess, dh_params):
    """
    Solve the IK problem by finding joint angles that minimize the position error.

    Parameters:
      target_pose  : Desired end-effector pose (4x4 homogeneous transformation matrix).
      initial_guess: Initial guess for the joint angles (list or numpy array).
      dh_params    : DH parameters of the robot.

    Returns:
      A numpy array of joint angles that best achieve the target pose.
    """
    result = minimize(pose_error, initial_guess, args=(target_pose, dh_params), method='BFGS')

    if result.success:
        return result.x
    else:
        raise Exception("IK solver did not converge")

# =======================================================================
# Main Function: Command Robot to Target Pose
# =======================================================================
def main():
    # Initialize the API.
    api = MyCobotAPI("/dev/ttyACM1", 115200)

    # # angle_1 range (+- 168), angle_2 range (+- 135), angle_3 range (+- 150), angle_4 range (+- 145), angle_5 range (+- 165), angle_6 range (+- 360)
    # coords_list = [-170.2, -66.3, 294.9, 0.62, 0.78, -88.49]

    # x, y, z = -170.2, -66.3, 200.9
    x, y, z = -170.2, -66.3, 200.9


    # Build the homogeneous transformation matrix using identity for rotation.
    T_target = np.eye(4)
    T_target[0:3, 3] = np.array([x, y, z])

    print("Target Transformation Matrix (ignoring orientation):")
    print(T_target)

    # Define DH parameters for your 6-DOF robot.
    dh_params = [
        {"theta": 0, "offset": 0,          "d": 131.22, "a": 0,      "alpha": np.pi / 2},
        {"theta": 0, "offset": -np.pi / 2, "d": 0,      "a": -110.4, "alpha": 0},
        {"theta": 0, "offset": 0,          "d": 0,      "a": -96,    "alpha": 0},
        {"theta": 0, "offset": -np.pi / 2, "d": 63.4,   "a": 0,      "alpha": np.pi / 2},
        {"theta": 0, "offset": np.pi / 2,  "d": 75.05,  "a": 0,      "alpha": -np.pi / 2},
        {"theta": 0, "offset": 0,          "d": 45.6,   "a": 0,      "alpha": 0}
    ]

    # Provide an initial guess for the 6 joint angles (in radians)
    initial_guess = [0, 0, 0, 0, 0, 0]

    for i in range(len(initial_guess)):
        initial_guess[i] = np.deg2rad(initial_guess[i])

    # Solve the inverse kinematics to compute the joint angles that reach T_target
    try:
        joint_angles_solution = solve_ik(T_target, initial_guess, dh_params)
        print("IK Solution (joint angles in radians):")
        print(joint_angles_solution)

        # Command the robot by sending the computed joint angles (choose a speed, e.g., 80)
        api.send_angles(joint_angles_solution, speed=80)
        print("Command sent to move robot to the target position.")
    except Exception as e:
        print("IK Solver failed:", e)

if __name__ == "__main__":
    main()



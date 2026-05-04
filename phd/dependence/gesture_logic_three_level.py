import os
import time

import numpy as np
import torch
from PyQt5.QtCore import QTimer
from phd.dependence.sensor_layout import flatten_column_major_view
from phd.dependence.transformer import GestureBackbone, ThreeLevelHierarchicalModel


class ThreeLevelTransformer:
    FRAME_CONTROL_SHAPES = {(9, 10), (10, 10)}
    JOINT_CONTROL_SHAPE = (13, 10)
    SUPPORTED_SHAPES = FRAME_CONTROL_SHAPES | {JOINT_CONTROL_SHAPE}
    SHAPE_9X10 = (9, 10)
    SHAPE_10X10 = (10, 10)
    MODEL_DIR_BY_SHAPE = {
        JOINT_CONTROL_SHAPE: "/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/eblow/eblow_0123_threelevel_v6(30frames)/",
        SHAPE_9X10: "/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/cylinder/cylinder_finger012_gesture123456_100good/",
        SHAPE_10X10: "/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/cylinder/cylinder_10x10_01234_threelevel(3frames)/",
    }

    def __init__(self, ros_splitter_instance, my_sensor_instance, n_row, n_col):
        self.ros_splitter = ros_splitter_instance
        self.my_sensor = my_sensor_instance
        self.n_row = n_row
        self.n_col = n_col
        self.is_recognizing_gesture = False
        self.recognition_timer = QTimer()
        self.recognition_timer.timeout.connect(self.run_recognition_step)
        self.current_gesture_data = []
        self.window_size = 20
        self.latch_mode = False
        self.last_detected_finger = None
        self._skip_velocity_once = False
        self._joint_velocity_vec = [0.0] * 6
        self._joint_vel_mode_enabled = False
        self._latest_raw_mean = 0.0
        self.raw_mean_touch_threshold = -0.1
        self.performing_flag = False
        self.waiting_flag = True
        self.counter = 0
        self.last_gesture_time = time.time()

        self._anchor_pos = None
        self._anchor_quat = None
        self._anchor_R = None
        self.anchor_enabled = True

        self.STATE_WAITING = 0
        self.STATE_PERFORMING = 1
        self.current_state = self.STATE_WAITING
        self.last_detected_gesture = None

        self.activation_counter = 0
        self.activation_threshold = 10
        self.potential_gesture_to_activate = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.global_mean, self.global_std = 0.0, 1.0
        self.movement_x, self.movement_y, self.movement_z = 0.0, 0.0, 0.0
        self.rotation_x, self.rotation_y, self.rotation_z = 0.0, 0.0, 0.0

        model_dir = self._resolve_model_dir()
        self.model_path = os.path.join(model_dir, "3level_model.pth")
        self.config_path = os.path.join(model_dir, "backbone_3level_config.txt")
        self.scaler_path = os.path.join(model_dir, "scaler_3level.npz")
        self.three_level_config_path = os.path.join(model_dir, "3level_config.txt")
        self.load_model_and_scaler()

    def load_model_and_scaler(self):
        print("--- Loading 3-Level Transformer Model for Real-Time Recognition ---")
        try:
            for path in [self.model_path, self.config_path, self.scaler_path, self.three_level_config_path]:
                if not os.path.exists(path):
                    print(f"Error: Required file not found at {path}. Cannot load model.")
                    self.model = None
                    return

            scaler_data = np.load(self.scaler_path)
            self.global_mean, self.global_std = scaler_data["mean"], scaler_data["std"]
            print(f"Row: {self.n_row}, Col: {self.n_col}")
            print(f"Loaded standardization scaler: Mean={self.global_mean:.4f}, Std={self.global_std:.4f}")

            config = self._parse_config(self.config_path)
            d_model = int(config["D_MODEL"])
            n_head = int(config["N_HEAD"])
            num_enc_layers = int(config["NUM_ENC_LAYERS"])
            dropout = config["DROPOUT"]

            config_3level = self._parse_config(self.three_level_config_path)
            num_f = int(config_3level["NUM_FINGER_CLASSES"])
            num_g = int(config_3level["NUM_GESTURE_CLASSES"])
            num_q = int(config_3level["NUM_QUALITY_CLASSES"])

            backbone = GestureBackbone(d_model, n_head, num_enc_layers, d_model * 4, dropout, self.n_row, self.n_col)
            self.model = ThreeLevelHierarchicalModel(backbone, d_model, num_f, num_g, num_q).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print("Successfully loaded 3-Level Transformer model.")
        except Exception as exc:
            print(f"An error occurred while loading the model: {exc}")
            self.model = None

    def _parse_config(self, path):
        config = {}
        with open(path, "r") as file:
            for line in file:
                if ": " in line:
                    key, value = line.strip().split(": ", 1)
                    try:
                        config[key] = int(value)
                    except ValueError:
                        config[key] = float(value)
        return config

    def reset_movement_variables(self):
        self.movement_x, self.movement_y, self.movement_z = 0.0, 0.0, 0.0
        self.rotation_x, self.rotation_y, self.rotation_z = 0.0, 0.0, 0.0
        self._joint_velocity_vec = [0.0] * 6

    def _sensor_shape(self):
        return self.n_row, self.n_col

    def _resolve_model_dir(self):
        shape = self._sensor_shape()
        if shape not in self.SUPPORTED_SHAPES:
            raise ValueError(f"Unsupported 3-Level sensor shape: {shape}")
        return self.MODEL_DIR_BY_SHAPE[shape]

    def _is_frame_control_shape(self):
        return self._sensor_shape() in self.FRAME_CONTROL_SHAPES

    def _is_joint_control_shape(self):
        return self._sensor_shape() == self.JOINT_CONTROL_SHAPE

    def _current_diff_matrix(self):
        return self.my_sensor._data.diffPerDataAve

    def _current_model_input_frame(self):
        """
        Return the historic column-major flat frame expected by the 3-level model.

        Internally the sensor state now stays in `(n_row, n_col)` order, so the
        layout conversion is kept explicit at the model boundary.
        """
        return np.asarray(flatten_column_major_view(self._current_diff_matrix()), dtype=np.float32)

    def _current_raw_mean(self):
        return float(np.mean(self._current_diff_matrix()))

    def _start_frame_control_recognition(self):
        self.recognition_timer.start()
        self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.suspend_end_effector_velocity_mode())
        self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.enable_end_effector_velocity_mode())
        print("3-Level Transformer Recognition STARTED (Frame Control).")

    def _stop_frame_control_recognition(self):
        self.recognition_timer.stop()
        self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.suspend_end_effector_velocity_mode())
        self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_end_effector_velocity_mode())
        print("3-Level Transformer Recognition STOPPED (Frame Control).")

    def _apply_logged_assignment(self, message, attr_name, value):
        print(message)
        setattr(self, attr_name, value)

    def _apply_9x10_movement(self, finger_pred, gesture_pred, speed):
        if finger_pred == 0:
            print("Gesture: Nothing")
            return

        finger_one_actions = {
            1: ("Gesture: PUSH", "movement_x", speed),
            2: ("Gesture: PULL", "movement_x", -speed),
            3: ("Gesture: SWIPE LEFT", "movement_y", -speed),
            4: ("Gesture: SWIPE RIGHT", "movement_y", speed),
            5: ("Gesture: SWIPE DOWN", "movement_z", speed),
            6: ("Gesture: SWIPE UP", "movement_z", -speed),
        }
        finger_two_actions = {
            3: ("Finger: 2, Gesture: SWIPE LEFT", "rotation_z", 0.0001),
            4: ("Finger: 2, Gesture: SWIPE RIGHT", "rotation_z", -0.0001),
            5: ("Finger: 2, Gesture: SWIPE DoWN", "rotation_x", 0.0001),
            6: ("Finger: 2, Gesture: SWIPE UP", "rotation_x", -0.0001),
            7: ("Finger: 2, Gesture: CLOCKWISE", "rotation_y", 0.0001),
            8: ("Finger: 2, Gesture: ANTI-CLOCKWISE", "rotation_y", -0.0001),
        }

        action_map = {
            1: finger_one_actions,
            2: finger_two_actions,
        }
        action = action_map.get(finger_pred, {}).get(gesture_pred)
        if action is not None:
            self._apply_logged_assignment(*action)

    def _apply_10x10_movement(self, finger_pred, gesture_pred):
        if finger_pred == 0:
            print("Gesture: Nothing")
            return

        if finger_pred != 1:
            return

        action = {
            1: "Gesture: 1",
            2: "Gesture: 2",
            3: "Gesture: 3",
            4: "Gesture: 4",
        }.get(gesture_pred)
        if action is not None:
            print(action)

    def _apply_joint_control_movement(self, finger_pred, gesture_pred, joint_speed):
        if finger_pred != 0:
            return

        joint_velocity = [0.0] * 6
        action = {
            0: ("Gesture: Push → J0 +", 0, joint_speed),
            1: ("Gesture: Pull → J0 -", 0, -joint_speed),
            2: ("Gesture: Down → J1 +", 1, joint_speed),
            3: ("Gesture: Up → J1 -", 1, -joint_speed),
        }.get(gesture_pred)
        if action is None:
            return

        message, joint_idx, value = action
        print(message)
        joint_velocity[joint_idx] = value
        self._joint_velocity_vec = joint_velocity

    def _has_pending_motion(self):
        return any(
            value != 0
            for value in (
                self.movement_x,
                self.movement_y,
                self.movement_z,
                self.rotation_x,
                self.rotation_y,
                self.rotation_z,
                *self._joint_velocity_vec,
            )
        )

    def toggle_gesture_recognition(self):
        if self.model is None:
            print("Cannot start recognition: Model is not loaded properly.")
            return

        self.is_recognizing_gesture = not self.is_recognizing_gesture

        if self.is_recognizing_gesture:
            print(f"Row: {self.n_row}, Col: {self.n_col}")
            if self._is_frame_control_shape():
                self._start_frame_control_recognition()
            if self._is_joint_control_shape():
                self.recognition_timer.start()
                self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.suspend_end_effector_velocity_mode())
                self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.enable_joint_velocity_mode())
                print("3-Level Transformer Recognition STARTED (Joint Control).")
        else:
            if self._is_frame_control_shape():
                self._stop_frame_control_recognition()
            if self._is_joint_control_shape():
                self.recognition_timer.stop()
                self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_joint_velocity_mode())
                print("3-Level Transformer Recognition STOPPED (Joint Control).")

        self.current_gesture_data = []
        self.current_state = self.STATE_WAITING

    def toggle_latch_mode(self):
        self.latch_mode = not self.latch_mode
        print(f"[3Level] Latch mode = {'ON' if self.latch_mode else 'OFF'}")

    def _clear_anchor(self):
        self._anchor_pos = None
        self._anchor_quat = None
        self._anchor_R = None

    def _get_requested_frame(self):
        le = getattr(self.ros_splitter, "ai_frame_input", None)
        if le is None:
            return None
        text = le.text().strip().lower()
        return text or None

    def _get_R_for_frame(self, frame: str):
        import math
        import transforms3d

        frame_name = (frame or "tool").strip().lower()
        if frame_name in ("base", "world", "joint1", "j1"):
            return np.eye(3, dtype=float)

        pos_quat = self.ros_splitter.robot_api.get_current_tool_position()
        if not pos_quat or pos_quat[1] is None:
            return np.eye(3, dtype=float)

        _, quat = pos_quat
        rotation = transforms3d.quaternions.quat2mat(quat)

        if frame_name in ("tool", "tcp", "joint6", "j6"):
            return rotation

        if not frame_name.startswith("joint"):
            return rotation
        try:
            n = int(frame_name[5:])
        except Exception:
            n = 6
        n = max(1, min(6, n))
        if n >= 6:
            return rotation

        joints = self.ros_splitter.robot_api.get_current_positions()
        if not joints or len(joints) < 6:
            return rotation

        axes_map = {6: "z", 5: "y", 4: "z", 3: "y", 2: "z", 1: "z"}

        def _rx(theta):
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)

        def _ry(theta):
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)

        def _rz(theta):
            c, s = math.cos(theta), math.sin(theta)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)

        for joint_idx in range(6, n, -1):
            axis = axes_map.get(joint_idx, "z")
            q = float(joints[joint_idx - 1])
            rotation = rotation.dot(_rx(-q) if axis == "x" else _ry(-q) if axis == "y" else _rz(-q))
        return rotation

    def _set_anchor_from_current_frame(self):
        pos_quat = self.ros_splitter.robot_api.get_current_tool_position()
        if not pos_quat:
            return
        (px, py, pz), quat = pos_quat
        self._anchor_pos = np.array([px, py, pz], float)
        self._anchor_quat = tuple(quat)
        frame = self._get_requested_frame() or "tool"
        self._anchor_R = self._get_R_for_frame(frame)

    def run_recognition_step(self):
        model_input_frame = self._current_model_input_frame()
        standardized_frame = (model_input_frame - self.global_mean) / self.global_std
        self.current_gesture_data.append(standardized_frame)

        if len(self.current_gesture_data) > self.window_size:
            self.current_gesture_data.pop(0)
        if len(self.current_gesture_data) < self.window_size:
            return

        self._latest_raw_mean = self._current_raw_mean()
        self.predict_gesture(list(self.current_gesture_data))

    def predict_gesture(self, gesture_data_list):
        if self.model is None:
            return

        gesture_array = np.array(gesture_data_list, dtype=np.float32)
        data_tensor = torch.tensor(gesture_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        padding_mask = torch.zeros(1, self.window_size, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            f_logits, g_logits, q_logits = self.model(data_tensor, padding_mask)
            f_conf, f_pred = torch.softmax(f_logits, dim=1).max(1)
            g_conf, g_pred = torch.softmax(g_logits, dim=1).max(1)
            q_conf, q_pred = torch.softmax(q_logits, dim=1).max(1)
            self.handle_state_logic(
                f_pred.item(),
                f_conf.item(),
                g_pred.item(),
                g_conf.item(),
                q_pred.item(),
                q_conf.item(),
            )

    def _maybe_send_once(self):
        latch = getattr(self, "latch_mode", False)
        prev_state = getattr(self, "_last_state_for_print", None)
        cur_state = getattr(self, "current_state", None)
        suppress_send = False

        if latch and prev_state is not None and prev_state != cur_state:
            if prev_state == self.STATE_PERFORMING and cur_state == self.STATE_WAITING:
                print("change from perform to waiting")
                try:
                    print("Entering Waiting state.")
                except Exception:
                    pass
                suppress_send = False
            elif prev_state == self.STATE_WAITING and cur_state == self.STATE_PERFORMING:
                print("change from waiting to perform")
                try:
                    print("Entering Performing state.")
                except Exception:
                    pass

        self._last_state_for_print = cur_state

        if latch:
            key = ("WAITING", None) if self.current_state == self.STATE_WAITING else (
                "PERFORMING",
                getattr(self, "last_detected_gesture", None),
            )
            if getattr(self, "_last_command_key", None) != key:
                self._last_command_key = key
                if not suppress_send:
                    self.send_robot_command()
        else:
            self.send_robot_command()

    def handle_state_logic(self, f_pred, f_conf, g_pred, g_conf, q_pred, q_conf):
        try:
            raw_mean = float(getattr(self, "_latest_raw_mean", self._current_raw_mean()))
        except Exception:
            raw_mean = 0.0

        if self.current_state == self.STATE_PERFORMING and raw_mean > self.raw_mean_touch_threshold:
            self.reset_movement_variables()
            self.current_gesture_data.clear()
            self.activation_counter = 0
            self.potential_gesture_to_activate = None
            self.last_detected_gesture = None

            if getattr(self, "latch_mode", False):
                try:
                    self.my_sensor.updateCal()
                except Exception:
                    pass

            self.current_state = self.STATE_WAITING
            if self.anchor_enabled:
                self._set_anchor_from_current_frame()
            else:
                self._clear_anchor()
            self._maybe_send_once()
            return

        min_q = getattr(self, "min_quality_conf", 0.40)
        is_good_quality = q_pred == 0 and q_conf >= min_q
        if not is_good_quality:
            self.reset_movement_variables()
            self.current_gesture_data.clear()
            self.activation_counter = 0
            try:
                self.my_sensor.updateCal()
            except Exception:
                pass
            self.potential_gesture_to_activate = None
            if self.current_state == self.STATE_PERFORMING:
                self.current_state = self.STATE_WAITING
            if self.anchor_enabled:
                self._set_anchor_from_current_frame()
            else:
                self._clear_anchor()
            self._maybe_send_once()
            return

        if self._sensor_shape() == self.SHAPE_9X10:
            is_do_nothing = f_pred == 0
        elif self._is_joint_control_shape():
            is_do_nothing = g_pred == 4
        else:
            is_do_nothing = g_conf < getattr(self, "min_gesture_conf", 0.55)

        conf_f = getattr(self, "min_finger_conf", 0.55)
        conf_g = getattr(self, "min_gesture_conf", 0.55)
        is_real_gesture = (not is_do_nothing and f_conf >= conf_f and g_conf >= conf_g)

        if self.current_state == self.STATE_WAITING:
            if is_real_gesture:
                if getattr(self, "potential_gesture_to_activate", None) is None:
                    self.potential_gesture_to_activate = g_pred
                    self.activation_counter = 1
                elif g_pred == self.potential_gesture_to_activate:
                    self.activation_counter += 1
                else:
                    self.potential_gesture_to_activate = g_pred
                    self.activation_counter = 1
            else:
                self.activation_counter = 0
                self.potential_gesture_to_activate = None

            if self.activation_counter >= getattr(self, "activation_threshold", 10):
                self.current_state = self.STATE_PERFORMING
                self.last_detected_gesture = self.potential_gesture_to_activate
                self.last_detected_finger = f_pred
                self._set_anchor_from_current_frame()
                self.set_robot_movement(f_pred, self.last_detected_gesture)
                self.activation_counter = 0
                self.potential_gesture_to_activate = None
        else:
            if getattr(self, "latch_mode", False):
                if is_do_nothing:
                    self.current_state = self.STATE_WAITING
                    self.last_detected_gesture = None
                    self.reset_movement_variables()
                    try:
                        self.my_sensor.updateCal()
                    except Exception:
                        pass
                    if self.anchor_enabled:
                        self._set_anchor_from_current_frame()
                    else:
                        self._clear_anchor()
                else:
                    finger_used = getattr(self, "last_detected_finger", f_pred)
                    self.set_robot_movement(finger_used, self.last_detected_gesture)
            else:
                if is_real_gesture:
                    self.set_robot_movement(f_pred, g_pred)
                    self.last_detected_gesture = g_pred
                else:
                    self.current_state = self.STATE_WAITING
                    self.last_detected_gesture = None
                    self.reset_movement_variables()
                    if self.anchor_enabled:
                        self._set_anchor_from_current_frame()
                    else:
                        self._clear_anchor()

        self._maybe_send_once()

    def set_robot_movement(self, finger_pred, gesture_pred):
        speed = 0.1
        j_speed = 0.01

        if self._sensor_shape() == self.SHAPE_9X10:
            self._apply_9x10_movement(finger_pred, gesture_pred, speed)

        if self._sensor_shape() == self.SHAPE_10X10:
            self._apply_10x10_movement(finger_pred, gesture_pred)

        if self._is_joint_control_shape():
            self._apply_joint_control_movement(finger_pred, gesture_pred, j_speed)
            return

        if self._has_pending_motion():
            self.last_gesture_time = time.time()

    def send_robot_command(self):
        v_lin = np.array([self.movement_x, self.movement_y, self.movement_z], float)
        v_rot = np.array([self.rotation_x, self.rotation_y, self.rotation_z], float)

        if self._sensor_shape() == self.SHAPE_9X10:
            if self.anchor_enabled and self.current_state == self.STATE_PERFORMING and self._anchor_R is not None:
                v_lin_w = self._anchor_R.dot(v_lin)
                v_rot_w = self._anchor_R.dot(v_rot)
                self.ros_splitter.robot_api.send_request(
                    self.ros_splitter.robot_api.set_end_effector_velocity_in_frame(
                        v_lin_w.tolist(),
                        v_rot_w.tolist(),
                        frame="base",
                    )
                )
            else:
                frame = self._get_requested_frame() or "tool"
                self.ros_splitter.robot_api.send_request(
                    self.ros_splitter.robot_api.set_end_effector_velocity_in_frame(
                        v_lin.tolist(),
                        v_rot.tolist(),
                        frame=frame,
                    )
                )

        if self._is_joint_control_shape():
            if not getattr(self, "_joint_vel_mode_enabled", False):
                self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.suspend_end_effector_velocity_mode())
                self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.enable_joint_velocity_mode())
                self._joint_vel_mode_enabled = True
            self.ros_splitter.robot_api.send_request(
                self.ros_splitter.robot_api.set_joint_velocity(self._joint_velocity_vec)
            )

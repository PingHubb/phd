import numpy as np
import time
import random
import os
import re

from PyQt5.QtCore import QTimer
import torch
from torch import nn
from phd.dependence.cnn_lstm import GestureCNNLSTM
from phd.dependence.transformer import GestureBackbone, HierarchicalGestureModel, ThreeLevelHierarchicalModel
from scipy.spatial.transform import Rotation as R

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from phd.dependence.func_sensor import MySensor

import os
import re
import random
import numpy as np
from PyQt5.QtCore import QTimer  # Assuming you're using PyQt5


class RecordGesture:
    def __init__(self, my_sensor_instance):
        self.my_sensor = my_sensor_instance

        # --- Timers ---
        self.timer_record_gesture = QTimer()
        self.timer_record_gesture.timeout.connect(self.record_gesture)

        # For automatic noise recording
        self.auto_timer = QTimer()
        self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self.auto_record_timeout)

        # --- State Variables ---
        self.is_recording = False
        self.start_record_pressed = False
        self.current_gesture_data_diff = []
        self.trial_number = None
        self.gesture_number = None
        self.auto_recording = False  # For the special noise_auto case

        # --- Configuration ---
        self.trigger_mode = None  # Can be 'trigger', 'no_trigger', 'no_trigger_auto', or 'no_trigger_no_updatecal_auto'
        self.max_frames_manual_trigger = 70
        self.auto_chunk_size = 30

    def set_trigger_mode(self, mode):
        """Sets the recording mode for the class."""
        valid_modes = ["trigger", "no_trigger", "no_trigger_auto", "no_trigger_no_updatecal_auto"]
        if mode in valid_modes:
            self.trigger_mode = mode
            print(f"Trigger mode set to '{self.trigger_mode}'.")
        else:
            print("Invalid trigger mode specified.")

    def start_record_gesture(self, gesture_number):
        """Handles the main Start/Stop button press from the GUI."""
        if self.trigger_mode is None:
            print("Warning: Trigger mode not set. Please set the trigger mode before recording.")
            return

        if gesture_number == "noise_auto":
            self._handle_noise_auto_toggle()
            return

        if not self.start_record_pressed:
            # --- STARTING A RECORDING SESSION ---
            self.start_record_pressed = True
            self.is_recording = True
            self.gesture_number = self.sanitize_gesture_number(gesture_number)
            self.trial_number = self.get_next_trial_number(self.gesture_number)
            self.current_gesture_data_diff = []

            self.timer_record_gesture.start(0)  # Use a fast interval, e.g., 0 for as fast as possible

            print(f"Recording ARMED for gesture '{self.gesture_number}'.")
            if self.trigger_mode == 'trigger':
                print("Waiting for trigger (touch sensor)...")
            elif self.trigger_mode in ['no_trigger_auto', 'no_trigger_no_updatecal_auto']:
                print(f"Automatic chunk recording started. Saving every {self.auto_chunk_size} frames.")
            else:  # no_trigger mode
                print("Manual continuous recording started.")
        else:
            # --- STOPPING A RECORDING SESSION ---
            self.start_record_pressed = False
            self.is_recording = False
            self.timer_record_gesture.stop()

            # If there's any leftover data in the buffer when we stop manually, save it.
            if self.current_gesture_data_diff:
                self._save_and_reset(reason="Manually stopped")

            print("Recording session stopped by user.")

    def _handle_noise_auto_toggle(self):
        """Handles the specific logic for the automatic noise recording mode."""
        if not self.auto_recording:
            self.auto_recording = True
            self.is_recording = True
            self.gesture_number = "noise"
            self.trial_number = self.get_next_trial_number(self.gesture_number)
            self.current_gesture_data_diff = []
            self.start_auto_recording()
            print("Automatic noise recording started.")
        else:
            self.auto_recording = False
            self.is_recording = False
            self.auto_timer.stop()
            self.timer_record_gesture.stop()
            if self.current_gesture_data_diff:
                self._save_and_reset(reason="Automatic noise recording manually stopped")
            print("Automatic noise recording stopped.")

    def start_auto_recording(self):
        """Starts a single timed recording segment for noise."""
        self.random_duration = random.randint(5, 10)
        print(f"Next noise recording will be for {self.random_duration} seconds.")
        self.current_gesture_data_diff = []
        self.timer_record_gesture.start(0)
        self.auto_timer.start(self.random_duration * 1000)

    def auto_record_timeout(self):
        """Called when the random duration for noise recording elapses."""
        if self.is_recording:
            self._save_and_reset(reason=f"Automatic duration ({self.random_duration}s) elapsed")
        if self.auto_recording:
            self.start_auto_recording()  # Start the next one

    def _save_and_reset(self, reason=""):
        """Saves data, optionally updates calibration, and prepares for the next trial."""
        self.save_gesture_data()  # This also clears the buffer in the process

        if self.trigger_mode != 'no_trigger_no_updatecal_auto':
            self.my_sensor.updateCal()
            log_message = f"Saved chunk for Trial {self.trial_number} and recalibrated."
        else:
            log_message = f"Saved chunk for Trial {self.trial_number}. (No recalibration)"

        if reason:
            log_message += f" Reason: {reason}."
        print(log_message)

        self.trial_number = self.get_next_trial_number(self.gesture_number)

    def _transform_data(self, data):
        conditions = [data > 2, data < -1, data < -0.2, data >= -0.2]
        choices = [2, 1, 0.2, 0]
        return np.select(conditions, choices)

    def sanitize_gesture_number(self, gesture_number):
        return re.sub(r'[^a-zA-Z0-9_]', '_', gesture_number)

    def record_gesture(self):
        """This function is called by the timer and handles the core recording logic."""
        if not self.is_recording:
            self.timer_record_gesture.stop();
            return

        raw_data_frame = self.my_sensor._data.diffPerDataAve.T.flatten()
        # print(len(raw_data_frame), flush=True)

        # --- Automatic Chunking Logic ---
        if self.trigger_mode in ["no_trigger_auto", "no_trigger_no_updatecal_auto"]:
            self.current_gesture_data_diff.append(raw_data_frame)
            if len(self.current_gesture_data_diff) >= self.auto_chunk_size:
                self._save_and_reset(reason=f"Auto-saved chunk of {self.auto_chunk_size} frames")

        # --- Manual Continuous Recording Logic ---
        elif self.trigger_mode == "no_trigger":
            self.current_gesture_data_diff.append(raw_data_frame)

        # --- Trigger-based Recording Logic ---
        elif self.trigger_mode == "trigger":
            transformed_data = self._transform_data(raw_data_frame)
            is_triggered = np.any(transformed_data == 1)

            if not hasattr(self, 'is_currently_triggered'):
                self.is_currently_triggered = False

            if is_triggered and not self.is_currently_triggered:
                self.is_currently_triggered = True
                self.current_gesture_data_diff = []
                print(f"Trigger detected! Started recording Trial {self.trial_number}.")

            elif not is_triggered and self.is_currently_triggered:
                self.is_currently_triggered = False
                if len(self.current_gesture_data_diff) > 1:
                    self._save_and_reset(reason="Trigger signal lost")

            if self.is_currently_triggered:
                self.current_gesture_data_diff.append(raw_data_frame)
                if len(self.current_gesture_data_diff) > self.max_frames_manual_trigger:
                    self._save_and_reset(reason=f"Frame limit ({self.max_frames_manual_trigger}) exceeded")
                    self.is_currently_triggered = False

    def get_next_trial_number(self, gesture_number):
        gesture_diff_dir = f"/home/ping2/ros2_ws/src/phd/phd/resource/ai/data/offset/gesture_{gesture_number}"
        if not os.path.exists(gesture_diff_dir): os.makedirs(gesture_diff_dir)
        files = os.listdir(gesture_diff_dir)
        return len(files) + 1

    def save_gesture_data(self):
        gesture_diff_dir = f"/home/ping2/ros2_ws/src/phd/phd/resource/ai/data/offset/gesture_{self.gesture_number}"
        filename = os.path.join(gesture_diff_dir, f"{self.trial_number}.txt")

        if not self.current_gesture_data_diff:
            print(f"Warning: No data recorded for trial {self.trial_number}. File not saved.")
            return

        with open(filename, 'w') as file:
            for data_entry in self.current_gesture_data_diff:
                file.write(' '.join(map(str, data_entry)) + "\n")

        # Clear the buffer after saving is complete
        self.current_gesture_data_diff = []


class RuleBased:
    def __init__(self, ros_splitter_instance, my_sensor_instance):
        self.ros_splitter = ros_splitter_instance
        self.my_sensor = my_sensor_instance
        # Initialize attributes that were previously in MySensor
        self.reset_gesture_state()
        self.activate_rule_button = False
        self.initial_cells = []
        self.initial_cell = None
        self.four_fingers_detected = True
        self.last_large_detection_time = None
        self.found_finger_timer = None
        self.cell_cooldowns = {}
        self.saved_row = None
        self.saved_col = None
        self.saved_value = None
        self.consecutive_moves = 0
        self.movement_direction = None
        self.last_joint_1_movement = 0.0
        self.last_joint_3_movement = 0.0
        self.initial_message = ""
        # Initialize the timer
        self.timer_2 = QTimer()
        self.timer_2.timeout.connect(self.handle_timer)
        # Initialize the gesture recognition timer
        self.gesture_timer = QTimer()
        self.gesture_timer.timeout.connect(self.gesture_recognition)

    def activate_rule_based(self):
        self.activate_rule_button = not self.activate_rule_button
        if self.activate_rule_button:
            print("Activate Rule-Based")
            # Start the gesture recognition timer with an interval (e.g., 50 ms)
            self.gesture_timer.start(0)  # Adjust the interval as needed
        else:
            print("Deactivate Rule-Based")
            # Stop the gesture recognition timer
            self.gesture_timer.stop()

    def gesture_recognition(self):
        if self.activate_rule_button is True:
            diffPerDataAve_Reverse = self.my_sensor._data.diffPerDataAve.T
            detected_fingers = 0

            for i in range(self.my_sensor.n_col):
                for j in range(self.my_sensor.n_row):
                    if self.check_cooldown(i, j):
                        continue
                    current_value = diffPerDataAve_Reverse[i][j]
                    if current_value < -1:
                        detected_fingers += 1
                        if not self.timer_2.isActive() and self.four_fingers_detected:
                            self.add_initial_cell(i, j)
                            self.set_focus(i, j, current_value)
                            # print(f"No. of fingers detected: {detected_fingers}")

            if detected_fingers > 20:
                self.four_fingers_detected = False
                # print("10+ fingers are detected")
                # self.ros_splitter.robot_api.send_request("StopContinueVmode()")
                # self.ros_splitter.robot_api.send_request("StopAndClearBuffer()")
                # self.ros_splitter.robot_api.send_and_process_request([1.0, -0.49, 1.57, 0.48, 1.57, 0.0])
                self.last_large_detection_time = time.time()  # Set the timestamp when more than 20 fingers are detected

            # Check the time since last large detection
            if detected_fingers == 4:
                current_time = time.time()
                if (self.last_large_detection_time is None or
                        (current_time - self.last_large_detection_time) > 1):
                    self.four_fingers_detected = True
                    # print("4 fingers are detected")
                else:
                    print("Detection of 4 fingers ignored due to recent large detection")

    def check_cooldown(self, row, col):
        current_time = time.time()
        if (row, col) in self.cell_cooldowns and current_time - self.cell_cooldowns[(row, col)] < 0.5:
            return True
        return False

    def add_initial_cell(self, row, col):
        # Add cell to initial_cells and manage bounds
        for (r, c) in self.initial_cells:
            if abs(r - row) <= 1 and abs(c - col) <= 1:
                return  # Already within the initial touch area
        self.initial_cells.append((row, col))

    def set_focus(self, row, col, value):
        if not self.initial_cell:
            self.initial_cell = (row, col)
        self.saved_row, self.saved_col = row, col
        self.saved_value = value
        self.found_finger_timer = time.time()
        self.initial_message = f"Initial detection at ({row}, {col})."
        self.timer_2.start(0)  # This is handle_timer function

    def handle_timer(self):
        if self.found_finger_timer is None:
            print("Error: Timer fired but start time was never set.", flush=True)
            self.timer_2.stop()
            return

        current_time = time.time()
        elapsed_time = current_time - self.found_finger_timer
        current_value = self.my_sensor._data.diffPerDataAve.T[self.saved_row][self.saved_col]

        if current_value > -1:
            self.timer_2.stop()
            # print(f"\r{self.initial_message} Finger is removed.", flush=True)
            self.reset_gesture_state()
            return

        self.check_adjacent_cells()
        # print(
        #     f"\r{self.initial_message} Elapsed time since initial detection: {elapsed_time:.2f} seconds, Current value: {current_value:.2f}",
        #     flush=True, end="")

    def reset_gesture_state(self):
        self.initial_cells = []
        self.movement_direction = None
        self.consecutive_moves = 0
        self.last_joint_1_movement = 0.0
        self.last_joint_3_movement = 0.0
        # self.ros_splitter.robot_api.send_request("SuspendContinueVmode()")

    def check_adjacent_cells(self):
        # Calculate bounds of initial touch area
        min_row = min(self.initial_cells, key=lambda x: x[0])[0]
        max_row = max(self.initial_cells, key=lambda x: x[0])[0]
        min_col = min(self.initial_cells, key=lambda x: x[1])[1]
        max_col = max(self.initial_cells, key=lambda x: x[1])[1]

        # Check horizontally adjacent cells at the bounds
        for col in range(min_col - 1, max_col + 2):
            self.check_cell(min_row - 1, col, "up")  # Check row above the top
            self.check_cell(max_row + 1, col, "down")  # Check row below the bottom

        # Check vertically adjacent cells at the bounds
        for row in range(min_row, max_row + 1):
            self.check_cell(row, min_col - 1, "left")  # Check column left of the leftmost
            self.check_cell(row, max_col + 1, "right")  # Check column right of the rightmost

    def check_cell(self, row, col, direction):
        if 0 <= row < 13 and 0 <= col < 10:
            if self.check_cooldown(row, col):
                return  # Skip if the cell is cooling down
            adjacent_value = self.my_sensor._data.diffPerDataAve.T[row][col]
            if adjacent_value < -1:
                move_direction = 'right' if direction == "right" else 'left' if direction in ["left", "right"] else None
                if move_direction and (move_direction == self.movement_direction or not self.movement_direction):
                    self.consecutive_moves += 1
                    # print(
                    #     f"\r{self.consecutive_moves} {direction.capitalize()} adjacent cell at [{row}][{col}] is touched.",
                    #     flush=True)
                else:
                    self.consecutive_moves = 1
                    # print(
                    #     f"\rFirst {direction.capitalize()} adjacent cell at [{row}][{col}] is touched. Direction reset.",
                    #     flush=True)

                # self.switch_focus(row, col, adjacent_value)
                # if hasattr(self.ros_splitter, 'robot_api'):
                #     movement_factor = 0.05 * self.consecutive_moves
                #     if direction in ["left", "right"]:
                #         x_movement = -movement_factor if direction == "left" else movement_factor
                #         self.last_joint_1_movement = x_movement
                #     elif direction in ["up", "down"]:
                #         z_movement = -movement_factor if direction == "up" else movement_factor
                #         self.last_joint_3_movement = z_movement
                #     Send the combined movement command
                    # self.ros_splitter.robot_api.combined_end_effector_velocity(
                    #     [self.last_joint_1_movement, 0.0, self.last_joint_3_movement, 0.0, 0.0, 0.0])

    def switch_focus(self, row, col, value):
        self.cell_cooldowns[(self.saved_row, self.saved_col)] = time.time()
        self.saved_row, self.saved_col = row, col
        self.saved_value = value
        self.found_finger_timer = time.time()
        self.timer_2.start(0)


class LSTM:
    def __init__(self, ros_splitter_instance, my_sensor_instance):
        self.ros_splitter = ros_splitter_instance
        self.my_sensor = my_sensor_instance
        # Rest of your initialization code
        self.last_prediction = None
        self.movement_x = 0.0
        self.movement_y = 0.0
        self.movement_z = 0.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.prediction_counter = 0
        self.window_size = 50
        # self.minimum_sequence_length = 5
        self.minimum_sequence_length = 10

        self.triggerd_value = -1
        self.proximity_triggerd_value = -0.3
        self.is_recognizing_gesture = False
        self.recognition_timer = QTimer()
        self.recognition_timer.timeout.connect(self.start_gesture_recognition)
        self.current_predict_data = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.touching = False  # Add this line
        self.delay_active = False  # Flag to indicate delay state
        self.pred_2_counter = 0  # Counter for consecutive pred == 2 detections

        self.coords_list = [-170.2, -66.3, 240, 0.62, 0.78, -88.49]
        self.coords_x_counter = 0
        self.coords_y_counter = 0
        self.coords_z_counter = 0

        # Define model paths and parameters
        self.model_paths = {
            'model1': {
                'model_txt_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_table/lstm/arm_0123/best_model_38.txt',
                'model_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_table/lstm/arm_0123/best_model_38.pth'
            },
            'model2': {
                'model_txt_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_table/lstm/arm_0123/best_model_36.txt',
                'model_path': '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_table/lstm/arm_0123/best_model_36.pth'
            }
        }

        # Define transformation functions
        self.transformation_functions = {
            'model1': self.transformation_function_model1,
            'model2': self.transformation_function_model2,
        }

        self.current_model_name = 'model1'
        self.load_model(self.current_model_name)

    def load_model(self, model_name):
        try:
            paths = self.model_paths[model_name]
            model_txt_path = paths['model_txt_path']
            model_path = paths['model_path']
            parameters = self.parse_parameters_from_file(model_txt_path)
            self.hidden_dim = parameters.get('hidden_dim', 64)
            self.output_dim = parameters.get('output_dim', 4)
            print(f"Loaded {model_name} with {self.output_dim} classes.")
            self.epochs = parameters.get('epochs', 50)
            self.batch_size = parameters.get('batch_size', 16)
            self.k_folds = parameters.get('k_folds', 5)
            self.num_layers = parameters.get('num_layers', 1)
            self.learning_rate = parameters.get('learning_rate', 0.001)
            self.dropout_rate = parameters.get('dropout_rate', 0.3)
            self.input_size = parameters.get('input_size', 56)   # 56

            # Initialize the model
            self.model = GestureCNNLSTM(
                self.input_size, self.hidden_dim, self.output_dim,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.current_transformation = self.transformation_functions[model_name]
            self.current_model_name = model_name
            print(f"Model {model_name} loaded.")
        except Exception as e:
            print(f'Error loading gesture recognition model {model_name}: {e}')
            self.model = None

    def reset_movement_variables(self):
        self.movement_x = 0.0
        self.movement_y = 0.0
        self.movement_z = 0.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0

    def transformation_function_model1(self, diffPerDataAve_Reverse):
        transformed_data = np.where(diffPerDataAve_Reverse < self.triggerd_value, 1, 0)
        return transformed_data

    def transformation_function_model2(self, diffPerDataAve_Reverse):
        transformed_data = np.where(diffPerDataAve_Reverse > 2, 2,
                                    np.where(diffPerDataAve_Reverse < -1, 1,
                                             np.where(diffPerDataAve_Reverse < self.proximity_triggerd_value, 0.2, 0)))
        return transformed_data

    def toggle_gesture_recognition(self):
        """Toggle the real-time gesture recognition process."""
        if not self.is_recognizing_gesture:
            self.is_recognizing_gesture = True
            self.recognition_timer.start(0)
            print("Gesture recognition started.")
            self.current_predict_data = []  # Reset the data at the start
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.enable_end_effector_velocity_mode())
        else:
            self.recognition_timer.stop()
            self.is_recognizing_gesture = False
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_end_effector_velocity_mode())
            self.current_predict_data = []  # Reset after saving and prediction
            print("Gesture recognition stopped.")

    def toggle_model(self):
        """Toggle the currently loaded model."""
        if self.current_model_name == 'model1':
            print("Switching from model1 to model2")
            # self.load_model('model2')
            self.delay_active = True  # Activate the delay
            self.start_delay_timer('model2')
            self.current_predict_data = []
        else:
            print("Switching from model2 to model1")
            # self.load_model('model1')
            self.pred_2_counter = 0  # Reset counter
            self.delay_active = True
            self.start_delay_timer('model1')
            self.current_predict_data = []
            self.ros_splitter.robot_api.send_request(
                self.ros_splitter.robot_api.enable_end_effector_velocity_mode())

    def start_gesture_recognition(self):
        if self.delay_active:
            # Skip processing during delay
            return

        diffPerDataAve_Reverse = self.my_sensor._data.diffPerDataAve.T.flatten()
        transformed_data = self.current_transformation(diffPerDataAve_Reverse)

        touching_now = np.any(transformed_data > 0)

        if touching_now and not self.touching:
            # Touch has just started
            self.touching = True
            self.current_predict_data = []  # Reset data collected
            self.touch_start_time = time.time()  # Record touch start time if needed
            # Reset last prediction and counter on touch start
            self.last_prediction = None
        elif not touching_now and self.touching:
            # Touch has just ended
            self.touching = False
            if self.current_predict_data:
                # print("Finger lifted")

                # threading.Thread(target=self.ros_splitter.mini_robot.sync_send_coords,
                #                  args=([-170.2, -66.3, 294.9, 0.62, 0.78, -88.49], 1, 1)).start()

                # if len(self.current_predict_data) >= self.minimum_sequence_length:
                #     # For both models, make a final prediction after touch ends
                #     self.predict_gesture(self.current_predict_data)
                # else:
                #     print("Not enough data to make a final prediction.")

                # Reset movement variables
                self.reset_movement_variables()
                self.ros_splitter.robot_api.send_request(
                    self.ros_splitter.robot_api.suspend_end_effector_velocity_mode())

            # Reset variables
            self.current_predict_data = []
            self.prediction_counter = 0
            # Do not reset last_prediction here to keep track between touches
        elif touching_now:
            # Touch is continuing
            self.current_predict_data.append(transformed_data.tolist())

            # Ensure the data buffer does not exceed the window size
            if len(self.current_predict_data) > self.window_size:
                self.current_predict_data.pop(0)

            if self.current_model_name == 'model1':
                # For model1, only start predicting after minimum_sequence_length data frames collected
                if len(self.current_predict_data) >= self.minimum_sequence_length:
                    self.predict_gesture(self.current_predict_data)
            else:
                # For model2, do not predict yet during touch
                pass
        else:
            # Not touching, and touch has not just ended
            pass

    def predict_gesture(self, gesture_data):
        # Convert to numpy array
        gesture_data = np.array(gesture_data, dtype=np.float32)
        seq_length, input_size = gesture_data.shape

        # Ensure input_size matches model's expected input size
        if input_size != self.input_size:
            print(f"Input size mismatch: expected {self.input_size}, got {input_size}")
            return

        # Reshape data to match model input shape
        gesture_data = gesture_data.reshape(1, seq_length, input_size)

        # Convert to tensor
        data_tensor = torch.tensor(gesture_data, dtype=torch.float32).to(self.device)
        lengths = torch.tensor([seq_length], dtype=torch.long).to(self.device)

        # Prepare the data for the model
        packed_input = nn.utils.rnn.pack_padded_sequence(
            data_tensor, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Run the model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(packed_input)
            probabilities = torch.softmax(outputs.data, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            pred = predicted.item()
            confidence = max_prob.item()

        # Call the appropriate handler based on the current model
        if self.current_model_name == 'model1':
            self.handle_prediction_model1(pred, confidence)
        elif self.current_model_name == 'model2':
            self.handle_prediction_model2(pred, confidence)
        else:
            print(f"Unknown model {self.current_model_name}")

    def handle_prediction_model1(self, pred, confidence):
        # Set a confidence threshold
        confidence_threshold = 0.8
        if confidence < confidence_threshold:
            # print(f"Low confidence ({confidence:.2f}) for {pred}, prediction ignored.")
            return

        if pred == 8:
            if self.touching:
                print("Touching is True, ignoring pred == 8 during touch")
                return
            else:
                print("Touch has ended, handling pred == 8")
                print("Model1 Predicted Gesture (8): Switch to model2 after 1 second delay")
                self.delay_active = True  # Activate the delay
                self.start_delay_timer('model2')
                self.current_predict_data = []
                return  # Wait for delay to finish

        # For other predictions, act as before
        if self.touching:
            # During touch, check if prediction is same as last
            if pred == self.last_prediction:
                return
            else:
                self.last_prediction = pred

        # For other predictions, act as before
        # Reset movement variables
        self.reset_movement_variables()

        # Define actions for model1's gestures
        if pred == 0:
            print("Model1 Predicted Gesture (0): Left")
            self.movement_y = -0.1
            # self.new_coords_list = [self.coords_list[0], 200,
            #                         self.coords_list[2], self.coords_list[3], self.coords_list[4], self.coords_list[5]]
            # threading.Thread(target=self.ros_splitter.mini_robot.sync_send_coords,
            #                      args=(self.new_coords_list, 60, 1)).start()

        elif pred == 1:
            print("Model1 Predicted Gesture (1): Right")
            self.movement_y = 0.1
            # self.new_coords_list = [self.coords_list[0], -200,
            #                         self.coords_list[2], self.coords_list[3], self.coords_list[4], self.coords_list[5]]
            # threading.Thread(target=self.ros_splitter.mini_robot.sync_send_coords,
            #                      args=(self.new_coords_list, 60, 1)).start()
        elif pred == 2:
            print("Model1 Predicted Gesture (2): Down")
            self.movement_z = -0.1
            # self.new_coords_list = [self.coords_list[0], self.coords_list[1],
            #                         240, self.coords_list[3], self.coords_list[4], self.coords_list[5]]
            # threading.Thread(target=self.ros_splitter.mini_robot.sync_send_coords,
            #                      args=(self.new_coords_list, 60, 1)).start()
        elif pred == 3:
            print("Model1 Predicted Gesture (3): Up")
            self.movement_z = 0.1
            # self.new_coords_list = [self.coords_list[0], self.coords_list[1],
            #                         360 , self.coords_list[3], self.coords_list[4], self.coords_list[5]]
            # threading.Thread(target=self.ros_splitter.mini_robot.sync_send_coords,
            #                      args=(self.new_coords_list, 60, 1)).start()
        elif pred == 4:
            print("Model1 Predicted Gesture (4): Double Left")
            self.movement_x = -0.1
        elif pred == 5:
            print("Model1 Predicted Gesture (5): Double Right")
            self.movement_x = 0.1
        elif pred == 6:
            print("Model1 Predicted Gesture (6): Double Down")
            self.movement_z = -0.1
        elif pred == 7:
            print("Model1 Predicted Gesture (7): Double Up")
            self.movement_z = 0.1

        # Send the movement command
        self.ros_splitter.robot_api.send_request(
            self.ros_splitter.robot_api.set_end_effector_velocity([
                self.movement_x,
                self.movement_y,
                self.movement_z,
                self.rotation_x,
                self.rotation_y,
                self.rotation_z
            ])
        )

    def handle_prediction_model2(self, pred, confidence):
        # Set a confidence threshold
        confidence_threshold = 0.7
        if confidence < confidence_threshold:
            print(f"Low confidence ({confidence:.2f}) for {pred}, prediction ignored.")
            return

        # Handle pred == 2 detections
        if pred == 2:
            if not self.touching:
                # Touch has ended after pred == 2
                self.pred_2_counter += 1
                print(f"pred == 2 detected after touch end. Counter: {self.pred_2_counter}")
            else:
                # Touch is ongoing, do not increment counter
                print("Touching is True, pred == 2 detected during touch")
                return

            if self.pred_2_counter >= 2:
                print("Two consecutive pred == 2 detected after touch end. Switching back to model1")
                self.pred_2_counter = 0  # Reset counter
                self.delay_active = True
                self.start_delay_timer('model1')
                self.current_predict_data = []
                self.ros_splitter.robot_api.send_request(
                    self.ros_splitter.robot_api.enable_end_effector_velocity_mode())
                return  # Wait for delay to finish
        else:
            # If any other prediction is detected, reset the counter
            self.pred_2_counter = 0

        # Handle other gestures
        if pred == 0:
            print("Model2 Predicted Gesture (0): Swipe to Left")
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_end_effector_velocity_mode())
            self.ros_splitter.robot_api.send_and_process_request([-0.3, -0.7, 1.8, 0.5, 1.6, -1.3])
        elif pred == 1:
            print("Model2 Predicted Gesture (1): Swipe to Right")
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_end_effector_velocity_mode())
            self.ros_splitter.robot_api.send_and_process_request([1.0, 0.0, 1.57, 0.0, 1.57, 0.0])
        # ... [other gestures]

    def start_delay_timer(self, model_name):
        self.delay_timer = QTimer()
        self.delay_timer.setSingleShot(True)  # Timer will fire only once
        self.delay_timer.timeout.connect(lambda: self.finish_delay(model_name))
        self.delay_timer.start(1000)  # Delay duration in milliseconds (1000ms = 1 second)
        print(f"Delay timer started for switching to {model_name}")

    def finish_delay(self, model_name):
        self.delay_active = False
        self.my_sensor.updateCal()
        self.load_model(model_name)
        print(f"Delay finished. Switched to {model_name}. Gesture recognition can start.")

    def parse_parameters_from_file(self, file_path):
        params = {}
        with open(file_path, 'r') as file:
            for line in file:
                if ': ' in line:
                    key, value = line.strip().split(': ')
                    params[key] = float(value) if '.' in value else int(value)
        return params


class HierarchicalTransformer:
    def __init__(self, ros_splitter_instance, my_sensor_instance, n_row, n_col):
        self.ros_splitter = ros_splitter_instance
        self.my_sensor = my_sensor_instance
        self.n_row = n_row
        self.n_col = n_col

        # --- Real-time detection state ---
        self.is_recognizing_gesture = False
        self.recognition_timer = QTimer()
        self.recognition_timer.timeout.connect(self.run_recognition_step)

        self.current_gesture_data = []
        self.touching = False
        self.minimum_sequence_length = 10
        self.window_size = 50
        self.last_finger_pred = None
        self.last_gesture_pred = None

        # --- Dual-mode prediction control ---
        self.prediction_mode = 'continuous'  # Default mode: 'continuous' or 'post_touch'
        self.max_frames_post_touch = 30  # Frame limit for making a prediction
        self.max_frames_recalibrate = 200  # Frame limit to trigger a timeout and recalibration
        self.prediction_locked = False  # State variable to lock prediction in post-touch mode

        # --- Model and Scaler Loading ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.global_mean, self.global_std = 0.0, 1.0

        # --- Robot Movement Variables ---
        self.movement_x, self.movement_y, self.movement_z = 0.0, 0.0, 0.0
        self.rotation_x, self.rotation_y, self.rotation_z = 0.0, 0.0, 0.0

        # --- PATHS for your hierarchical model ---
        if self.n_row == 13 and self.n_col == 10:
            # Use the elbow model paths
            models_path_classify = '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/elbow_0123_v2/'
            models_path_hierarchical = '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/elbow_0123_v2/'
            self.model_path = os.path.join(models_path_hierarchical, 'hierarchical_model.pth')
            self.config_path = os.path.join(models_path_classify, 'backbone_config.txt')
            self.scaler_path = os.path.join(models_path_classify, 'classification_scaler.npz')
        elif self.n_row == 8 and self.n_col == 7:
            models_path_classify = '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_table/transformer/arm_01234567/'
            models_path_hierarchical = '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_table/transformer/arm_01234567/'
            self.model_path = os.path.join(models_path_hierarchical, 'hierarchical_model.pth')
            self.config_path = os.path.join(models_path_classify, 'backbone_config.txt')
            self.scaler_path = os.path.join(models_path_classify, 'classification_scaler.npz')

        self.load_model_and_scaler()

    def load_model_and_scaler(self):
        """Loads the fine-tuned hierarchical model, its config, and its scaler."""
        print("--- Loading Hierarchical Transformer Model for Real-Time Recognition ---")
        try:
            for p in [self.model_path, self.config_path, self.scaler_path]:
                if not os.path.exists(p):
                    print(f"Error: Required file not found at {p}. Cannot load model.");
                    self.model = None;
                    return

            scaler_data = np.load(self.scaler_path)
            self.global_mean, self.global_std = scaler_data['mean'], scaler_data['std']
            print(f"Loaded standardization scaler: Mean={self.global_mean:.4f}, Std={self.global_std:.4f}")

            config = {};
            with open(self.config_path, 'r') as f:
                for line in f:
                    key, value = line.strip().split(': ')
                    try:
                        config[key] = int(value)
                    except ValueError:
                        config[key] = float(value)

            D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
                config['NUM_ENC_LAYERS']), config['DROPOUT']
            print(f"Loaded backbone configuration: {config}")

            state_dict = torch.load(self.model_path, map_location=self.device)
            num_finger_classes = state_dict['finger_classifier.bias'].shape[0]
            num_gesture_classes = state_dict['gesture_classifier.bias'].shape[0]
            print(f"Inferred classes: {num_finger_classes} finger types, {num_gesture_classes} gesture types.")

            backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT, sensor_rows=self.n_row, sensor_cols=self.n_col)
            self.model = HierarchicalGestureModel(backbone, d_model=D_MODEL, num_finger_classes=num_finger_classes,
                                                  num_gesture_classes=num_gesture_classes).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Successfully loaded hierarchical Transformer model.")

        except Exception as e:
            print(f"An error occurred while loading the model: {e}");
            self.model = None

    def toggle_prediction_mode(self):
        """Switches between continuous and post-touch prediction modes."""

        if self.prediction_mode == 'continuous':
            self.prediction_mode = 'post_touch'
        else:
            self.prediction_mode = 'continuous'
        print(f"Prediction mode switched to: '{self.prediction_mode}'")
        self.current_gesture_data = []
        self.touching = False
        self.prediction_locked = False

    def reset_movement_variables(self):
        """Resets all robot movement commands and prediction states."""
        self.movement_x, self.movement_y, self.movement_z = 0.0, 0.0, 0.0
        self.rotation_x, self.rotation_y, self.rotation_z = 0.0, 0.0, 0.0
        self.last_finger_pred, self.last_gesture_pred = None, None
        self.prediction_locked = False

    def toggle_gesture_recognition(self):
        """Toggles the real-time gesture recognition process."""
        if self.model is None:
            print(f"Row: {self.n_row}, Col: {self.n_col}")
            print("Cannot start recognition: Model is not loaded properly.")
            return

        self.is_recognizing_gesture = not self.is_recognizing_gesture
        if self.is_recognizing_gesture:
            self.recognition_timer.start(0)
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.enable_end_effector_velocity_mode())
            print(f"Hierarchical Transformer Recognition STARTED in '{self.prediction_mode}' mode.")
        else:
            self.recognition_timer.stop()
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_end_effector_velocity_mode())
            print("Hierarchical Transformer Recognition STOPPED.")

        self.current_gesture_data = [];
        self.touching = False

    def run_recognition_step(self):
        """
        This function is called repeatedly to process sensor data.
        It now correctly handles continuous data buffering in post-touch mode.
        """
        # print(len(self.current_gesture_data))

        raw_data_frame = self.my_sensor._data.diffPerDataAve.T.flatten()
        touch_threshold = -1.0
        is_currently_touching = np.any(raw_data_frame < touch_threshold)

        # print(f"is_currently_touching: {is_currently_touching}, self.touching: {self.touching}")

        # --- Handle start and end of a touch event ---
        if is_currently_touching and not self.touching:
            # print(f"1: is_currently_touching: {is_currently_touching}, self.touching: {self.touching}")
            self.touching = True
            self.current_gesture_data = []  # Always clear buffer on new touch
            self.prediction_locked = False  # Reset lock on every new touch

        elif not is_currently_touching and self.touching:
            # print(f"2: is_currently_touching: {is_currently_touching}, self.touching: {self.touching}")
            self.touching = False
            # Stop any potential robot movement and reset all states
            self.reset_movement_variables()  # This also resets the prediction_locked flag
            self.my_sensor.updateCal()  # Update sensor calibration
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.suspend_end_effector_velocity_mode())
            self.current_gesture_data = []

        # --- Handle data buffering and prediction logic while touching ---
        if self.touching:
            # --- MODIFICATION: Always buffer data when touching ---
            standardized_frame = (raw_data_frame - self.global_mean) / self.global_std
            self.current_gesture_data.append(standardized_frame)

            # --- LOGIC FOR CONTINUOUS MODE ---
            if self.prediction_mode == 'continuous':
                if len(self.current_gesture_data) > self.window_size:
                    self.current_gesture_data.pop(0)
                if len(self.current_gesture_data) >= self.minimum_sequence_length:
                    self.predict_gesture(list(self.current_gesture_data))

            # --- LOGIC FOR POST-TOUCH (PREDICT-AND-HOLD) MODE ---
            elif self.prediction_mode == 'post_touch':
                # --- MODIFICATION: The two checks are now independent ---

                # 1. Check if we should make a prediction (and haven't already)
                if len(self.current_gesture_data) >= self.max_frames_post_touch and not self.prediction_locked:
                    print(f"Prediction limit ({self.max_frames_post_touch}) reached. Predicting and locking action.")
                    self.predict_gesture(list(self.current_gesture_data))
                    self.prediction_locked = True  # Lock after the first prediction

                # 2. Independently, check if the gesture has gone on for too long
                if len(self.current_gesture_data) >= self.max_frames_recalibrate:
                    print(f"Recalibration limit ({self.max_frames_recalibrate}) reached. Gesture timed out.")
                    print("Updating sensor calibration...")
                    self.my_sensor.updateCal()
                    self.reset_movement_variables()  # Reset movement variables
                    self.ros_splitter.robot_api.suspend_end_effector_velocity_mode()
                    self.touching = False
                    # The reset logic in the `elif not is_currently_touching...` block will handle the rest.

    def predict_gesture(self, gesture_data_list):
        """Takes gesture frames, formats them, and runs hierarchical inference."""
        if self.model is None: return

        gesture_array = np.array(gesture_data_list, dtype=np.float32)
        seq_len = gesture_array.shape[0]
        data_tensor = torch.tensor(gesture_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        padding_mask = torch.zeros(1, seq_len, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            finger_logits, gesture_logits = self.model(data_tensor, padding_mask)

            finger_probs = torch.softmax(finger_logits, dim=1)
            finger_conf, finger_pred = torch.max(finger_probs, 1)

            gesture_probs = torch.softmax(gesture_logits, dim=1)
            gesture_conf, gesture_pred = torch.max(gesture_probs, 1)

            self.handle_hierarchical_prediction(
                finger_pred.item(), finger_conf.item(),
                gesture_pred.item(), gesture_conf.item()
            )

    def handle_hierarchical_prediction(self, finger_pred, finger_conf, gesture_pred, gesture_conf):
        """Interprets the hierarchical prediction and defines actions."""
        confidence_threshold = 0.8
        if finger_conf < confidence_threshold or gesture_conf < confidence_threshold: return

        # In continuous mode, prevent spamming the same command repeatedly
        if self.prediction_mode == 'continuous' and \
                (finger_pred == self.last_finger_pred and gesture_pred == self.last_gesture_pred):
            return

        self.last_finger_pred = finger_pred
        self.last_gesture_pred = gesture_pred
        self.reset_movement_variables()

        # --- HIERARCHICAL ACTION MAPPING ---
        if self.my_sensor.n_row == 10 and self.my_sensor.n_col == 13:  # Assuming 13x10 sensor
            finger_str = "\u2714" if finger_pred == 0 else "???"
            gesture_map = {0: "Push", 1: "Pull", 2: "Down", 3: "Up"}
            gesture_str = gesture_map.get(gesture_pred, "Unknown")
            print(f"Predicted: {finger_str} {gesture_str} (Conf F: {finger_conf:.2f}, G: {gesture_conf:.2f})")
            if finger_pred == 0:  # 1-Finger Gestures
                if gesture_pred == 0:
                    self.movement_x = -0.03  # Push
                    self.movement_y = -0.03
                elif gesture_pred == 1:
                    self.movement_x = 0.03  # Pull
                    self.movement_y = 0.03
                elif gesture_pred == 2:
                    self.movement_z = -0.03  # Down
                elif gesture_pred == 3:
                    self.movement_z = 0.03  # Up

        elif self.my_sensor.n_row == 7 and self.my_sensor.n_col == 8:  # Assuming 8x7 sensor
            finger_str = "1-Finger" if finger_pred == 0 else "2-Finger"
            gesture_map = {0: "Left", 1: "Right", 2: "Down", 3: "Up"}
            gesture_str = gesture_map.get(gesture_pred, "Unknown")
            print(f"Predicted: {finger_str} {gesture_str} (Conf F: {finger_conf:.2f}, G: {gesture_conf:.2f})")
            if finger_pred == 0:  # 1-Finger Gestures
                if gesture_pred == 0:
                    self.movement_y = -0.1  # Left
                elif gesture_pred == 1:
                    self.movement_y = 0.1  # Right
                elif gesture_pred == 2:
                    self.movement_z = -0.1  # Down
                elif gesture_pred == 3:
                    self.movement_z = 0.1  # Up

            elif finger_pred == 1:  # 2-Finger Gestures
                if gesture_pred == 0:
                    self.movement_y = -0.1  # Left
                elif gesture_pred == 1:
                    self.movement_y = 0.1  # Right
                elif gesture_pred == 2:
                    self.movement_z = -0.1  # Down
                elif gesture_pred == 3:
                    self.movement_z = 0.1  # Up

        else:
            print(f"Action: No action defined for this sensor size or combination.")
            return

        # Send the continuous velocity command to the robot
        self.ros_splitter.robot_api.send_request(
            self.ros_splitter.robot_api.set_end_effector_velocity([
                self.movement_x, self.movement_y, self.movement_z,
                self.rotation_x, self.rotation_y, self.rotation_z
            ])
        )


class ThreeLevelTransformer:
    def __init__(self, ros_splitter_instance, my_sensor_instance, n_row, n_col):
        # ... (all other __init__ variables are the same)
        self.ros_splitter = ros_splitter_instance
        self.my_sensor = my_sensor_instance
        self.n_row = n_row
        self.n_col = n_col
        self.is_recognizing_gesture = False
        self.recognition_timer = QTimer()
        self.recognition_timer.timeout.connect(self.run_recognition_step)
        self.current_gesture_data = []
        self.window_size = 30

        # --- State Machine Variables ---
        self.STATE_WAITING = 0
        self.STATE_PERFORMING = 1
        self.current_state = self.STATE_WAITING
        self.last_detected_gesture = None

        # --- NEW: Activation/Deactivation Logic ---
        self.activation_counter = 0
        self.activation_threshold = 10  # Require X consecutive frames of a gesture to activate
        self.potential_gesture_to_activate = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.global_mean, self.global_std = 0.0, 1.0
        self.movement_x, self.movement_y, self.movement_z = 0.0, 0.0, 0.0
        self.rotation_x, self.rotation_y, self.rotation_z = 0.0, 0.0, 0.0

        models_path_3level_backbone = '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/cylinder_012345_threelevel(30frames)/'
        models_path_3level_model = '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/cylinder_012345_threelevel(30frames)/'
        if self.n_row == 13 and self.n_col == 10:
            models_path_3level_backbone = '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/eblow_0123_threelevel_v6(30frames)/'
            models_path_3level_model = '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/eblow_0123_threelevel_v6(30frames)/'
        if self.n_row == 9 and self.n_col == 10:
            models_path_3level_backbone = '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/cylinder_012345_threelevel(30frames)/'
            models_path_3level_model = '/home/ping2/ros2_ws/src/phd/phd/resource/ai/models/NEW_CHIP/on_robot/transformer/cylinder_012345_threelevel(30frames)/'

        self.model_path = os.path.join(models_path_3level_model, '3level_model.pth')
        self.config_path = os.path.join(models_path_3level_backbone, 'backbone_3level_config.txt')
        self.scaler_path = os.path.join(models_path_3level_backbone, 'scaler_3level.npz')
        self.three_level_config_path = os.path.join(models_path_3level_model, '3level_config.txt')
        self.load_model_and_scaler()

    def load_model_and_scaler(self):
        """Loads the fine-tuned 3-level model, its specialized config, and its scaler."""
        print("--- Loading 3-Level Transformer Model for Real-Time Recognition ---")
        try:
            for p in [self.model_path, self.config_path, self.scaler_path, self.three_level_config_path]:
                if not os.path.exists(p):
                    print(f"Error: Required file not found at {p}. Cannot load model.")
                    self.model = None
                    return

            scaler_data = np.load(self.scaler_path)
            self.global_mean, self.global_std = scaler_data['mean'], scaler_data['std']
            print(f"Loaded standardization scaler: Mean={self.global_mean:.4f}, Std={self.global_std:.4f}")

            config = self._parse_config(self.config_path)
            D_MODEL, N_HEAD, NUM_ENC_LAYERS, DROPOUT = int(config['D_MODEL']), int(config['N_HEAD']), int(
                config['NUM_ENC_LAYERS']), config['DROPOUT']

            config_3level = self._parse_config(self.three_level_config_path)
            num_f = int(config_3level['NUM_FINGER_CLASSES'])
            num_g = int(config_3level['NUM_GESTURE_CLASSES'])
            num_q = int(config_3level['NUM_QUALITY_CLASSES'])

            backbone = GestureBackbone(D_MODEL, N_HEAD, NUM_ENC_LAYERS, D_MODEL * 4, DROPOUT, self.n_row, self.n_col)
            self.model = ThreeLevelHierarchicalModel(backbone, D_MODEL, num_f, num_g, num_q).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print("Successfully loaded 3-Level Transformer model.")

        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            self.model = None

    def _parse_config(self, path):
        """Helper to parse config files."""
        config = {}
        with open(path, 'r') as f:
            for line in f:
                if ': ' in line:
                    key, value = line.strip().split(': ', 1)
                    try:
                        config[key] = int(value)
                    except ValueError:
                        config[key] = float(value)
        return config

    def reset_movement_variables(self):
        self.movement_x, self.movement_y, self.movement_z = 0.0, 0.0, 0.0
        self.rotation_x, self.rotation_y, self.rotation_z = 0.0, 0.0, 0.0

    def toggle_gesture_recognition(self):
        if self.model is None:
            print("Cannot start recognition: Model is not loaded properly.")
            return

        self.is_recognizing_gesture = not self.is_recognizing_gesture
        if self.is_recognizing_gesture:
            self.recognition_timer.start()
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.enable_end_effector_velocity_mode())
            print("3-Level Transformer Recognition STARTED.")
        else:
            self.recognition_timer.stop()
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.stop_end_effector_velocity_mode())
            print("3-Level Transformer Recognition STOPPED.")

        self.current_gesture_data = []
        self.current_state = self.STATE_WAITING

    def run_recognition_step(self):
        """ The main recognition loop, simplified without cooldown. """
        # --- Step 1: Buffer the Data in a Sliding Window ---
        raw_data_frame = self.my_sensor._data.diffPerDataAve.T.flatten()
        standardized_frame = (raw_data_frame - self.global_mean) / self.global_std
        self.current_gesture_data.append(standardized_frame)

        # print(f"Current Gesture Data Length: {len(self.current_gesture_data)}")

        if len(self.current_gesture_data) > self.window_size:
            self.current_gesture_data.pop(0)

        if len(self.current_gesture_data) < self.window_size:
            return

        # --- Step 2: Predict on the current window ---
        self.predict_gesture(list(self.current_gesture_data))

    def predict_gesture(self, gesture_data_list):
        """Runs inference and passes result to the state machine handler."""
        if self.model is None: return

        gesture_array = np.array(gesture_data_list, dtype=np.float32)
        data_tensor = torch.tensor(gesture_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        padding_mask = torch.zeros(1, self.window_size, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            f_logits, g_logits, q_logits = self.model(data_tensor, padding_mask)
            f_conf, f_pred = torch.softmax(f_logits, dim=1).max(1)
            g_conf, g_pred = torch.softmax(g_logits, dim=1).max(1)
            q_conf, q_pred = torch.softmax(q_logits, dim=1).max(1)

            self.handle_state_logic(f_pred.item(), f_conf.item(), g_pred.item(), g_conf.item(), q_pred.item(),
                                    q_conf.item())

    def handle_state_logic(self, f_pred, f_conf, g_pred, g_conf, q_pred, q_conf):
        """
        Implements the State Machine with an activation counter to START,
        but an INSTANT deactivation to STOP.
        """
        # --- Step 1: Print raw prediction (no change) ---
        finger_str = f"{f_pred + 1}-Finger"
        gesture_map = {0: "Push", 1: "Pull", 2: "Down", 3: "Up", 4: "Do Nothing"}
        gesture_str = gesture_map.get(g_pred, f"Unknown({g_pred})")
        quality_str = "Good" if q_pred == 0 else "Damaged"
        # print(f"Activation Counter: {self.activation_counter}, State: {self.current_state}")

        # --- Step 2: High-Priority Quality Check (no change) ---
        confidence_threshold = 0.8
        is_good_quality = (q_pred == 0 and q_conf > confidence_threshold)
        if not is_good_quality:
            print("BROKEN!!!")
            self.reset_movement_variables()
            self.send_robot_command()
            self.my_sensor.updateCal()
            self.ros_splitter.robot_api.send_request(self.ros_splitter.robot_api.suspend_end_effector_velocity_mode())
            self.current_gesture_data = []  # Clearing buffer might be useful here
            # Reset all state
            self.activation_counter = 0
            self.potential_gesture_to_activate = None
            if self.current_state == self.STATE_PERFORMING:
                self.current_state = self.STATE_WAITING
                self.last_detected_gesture = None
                print(">>> State: PERFORMING -> WAITING (due to poor signal quality)")
            return

        # --- Step 3: State Machine Logic for GOOD Quality Signals ---

        is_do_nothing = (g_pred == 6)

        if self.n_row == 9 and self.n_col == 10:
            is_do_nothing = (g_pred == 6)   # PLEASE EDIT THIS!!!
        if self.n_row == 13 and self.n_col == 10:
            is_do_nothing = (g_pred == 4)   # PLEASE EDIT THIS!!!

        is_real_gesture = (not is_do_nothing and g_conf > confidence_threshold and f_conf > confidence_threshold)

        if self.current_state == self.STATE_WAITING:
            self.reset_movement_variables()
            # Activation logic remains the same
            if is_real_gesture:
                if self.potential_gesture_to_activate is None:
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

            if self.activation_counter >= self.activation_threshold:
                self.current_state = self.STATE_PERFORMING
                self.last_detected_gesture = self.potential_gesture_to_activate
                self.set_robot_movement(f_pred, self.last_detected_gesture)
                self.activation_counter = 0
                self.potential_gesture_to_activate = None
                print(f">>> State: WAITING -> PERFORMING ({gesture_map.get(self.last_detected_gesture, 'Unknown')})")

        elif self.current_state == self.STATE_PERFORMING:
            # --- NEW, SIMPLIFIED DEACTIVATION LOGIC ---
            if is_real_gesture and g_pred == self.last_detected_gesture:
                # The gesture is still active. Continue the movement.
                self.set_robot_movement(f_pred, g_pred)
            else:
                # The prediction is NO LONGER the active gesture. STOP IMMEDIATELY.
                self.current_state = self.STATE_WAITING
                self.last_detected_gesture = None
                self.reset_movement_variables()
                print(f">>> State: PERFORMING -> WAITING (Gesture Ended)")

        # Send the final command to the robot
        self.send_robot_command()

    def set_robot_movement(self, finger_pred, gesture_pred):
        """A helper to set movement variables based on a valid gesture prediction."""
        # Assuming 1-Finger for now
        print("Row:", self.n_row, "Col:", self.n_col)

        if self.n_row == 9 and self.n_col == 10:
            if finger_pred == 0:
                if gesture_pred == 0:
                    print("Gesture: Push")
                    # self.movement_x = -0.06
                    # self.movement_y = -0.06
                    self.movement_x = -0.06
                    self.movement_y = -0.06
                elif gesture_pred == 1:
                    print("Gesture: Pull")
                    # self.movement_x = 0.06
                    # self.movement_y = 0.06
                    self.movement_x = 0.06
                    self.movement_y = 0.06
                elif gesture_pred == 2:
                    print("Gesture: Left")
                    # self.movement_z = -0.06
                    self.movement_x = 0.06
                elif gesture_pred == 3:
                    print("Gesture: Right")
                    # self.movement_z = 0.06
                    self.movement_y = 0.06
                elif gesture_pred == 4:
                    print("Gesture: Down")
                    self.movement_z = -0.06
                elif gesture_pred == 5:
                    print("Gesture: Up")
                    self.movement_z = 0.06

        if self.n_row == 13 and self.n_col == 10:
            if finger_pred == 0:
                if gesture_pred == 0:
                    print("Gesture: Push")
                    self.movement_x = -0.06
                    self.movement_y = -0.06
                elif gesture_pred == 1:
                    print("Gesture: Left")
                    self.movement_x = 0.06
                    self.movement_y = 0.06
                elif gesture_pred == 2:
                    print("Gesture: Down")
                    self.movement_z = -0.06
                elif gesture_pred == 3:
                    print("Gesture: Up")
                    self.movement_z = 0.06

    def set_robot_movement_JJJJJJJJJJJJJJJJJJJJJJJ(self, finger_pred, gesture_pred):
        """
        Sets the desired robot velocity, transforming it from the TOOL FRAME
        to the WORLD FRAME using the robot's current orientation.
        """
        self.reset_movement_variables()

        # --- Step 1: Define Desired Velocity in the TOOL's coordinate system ---
        # Let's assume:
        # - Tool X-axis is forward(+)/backward(-) -> Pull/Push
        # - Tool Z-axis is up(+)/down(-)
        # NOTE: You may need to swap axes or negate values depending on your tool's definition.
        tool_velocity = np.array([0.0, 0.0, 0.0])
        speed = 0.03  # m/s

        if finger_pred == 0:  # 1-Finger Gestures
            if gesture_pred == 0:  # Push
                print("Gesture: Push (Tool Frame)")
                tool_velocity[0] = -speed
            elif gesture_pred == 1:  # Pull
                print("Gesture: Pull (Tool Frame)")
                tool_velocity[0] = speed
            elif gesture_pred == 2:  # Down
                print("Gesture: Down (Tool Frame)")
                tool_velocity[2] = speed
            elif gesture_pred == 3:  # Up
                print("Gesture: Up (Tool Frame)")
                tool_velocity[2] = -speed

        # --- Step 2: Get the Robot's Current Orientation (Quaternion) ---
        # This is the critical link to your robot_api.py
        _, orientation_quat = self.ros_splitter.robot_api.get_current_tool_position()

        if orientation_quat is None or R is None:
            if R is None:
                print("Scipy not available, cannot perform rotation.")
            else:
                print("Warning: Could not get robot orientation. Using global frame as fallback.")
            # Fallback to old global coordinate behavior if orientation is not ready
            self.movement_x, self.movement_y, self.movement_z = tool_velocity[0], tool_velocity[1], tool_velocity[2]
            return

        # --- Step 3: Transform Tool Velocity to World Velocity ---
        try:
            # Your API provides (w, x, y, z). Scipy expects (x, y, z, w).
            # We need to reorder the quaternion components.
            w, x, y, z = orientation_quat
            scipy_quat = [x, y, z, w]

            # Create a Rotation object from the quaternion
            rotation = R.from_quat(scipy_quat)

            # Use the 'apply' method to rotate the tool_velocity vector
            # This transforms it from the tool's coordinate frame to the world's frame.
            world_velocity = rotation.apply(tool_velocity)

            # --- Step 4: Assign the new World-Frame Velocities ---
            self.movement_x = world_velocity[0]
            self.movement_y = world_velocity[1]
            self.movement_z = world_velocity[2]

        except Exception as e:
            print(f"Error during velocity transformation: {e}. Using global frame as fallback.")
            self.movement_x, self.movement_y, self.movement_z = tool_velocity[0], tool_velocity[1], tool_velocity[2]

    def send_robot_command(self):
        self.ros_splitter.robot_api.send_request(
            self.ros_splitter.robot_api.set_end_effector_velocity([
                self.movement_x, self.movement_y, self.movement_z,
                self.rotation_x, self.rotation_y, self.rotation_z
            ])
        )
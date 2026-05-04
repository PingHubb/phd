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


from PyQt5 import QtCore
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import (QSplitter, QWidget, QGridLayout, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
                             QTabWidget, QLineEdit, QTextEdit, QGroupBox, QListWidget, QMainWindow, QAction, QMenuBar)
from pyvistaqt import QtInteractor
from phd.dependence.sensor_api import ArduinoCommander
from phd.dependence.robot_api import RobotController
from phd.dependence.mini_robot_api import MyCobotAPI
from phd.dependence.func_meshLab import MyMeshLab
from phd.dependence.func_sensor import MySensor
import os


class PlotterWidget(QWidget):
    filesDropped = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        # Style is now handled by QSS, but we can set object name for specific rules if needed
        # self.setObjectName("plotterWidget")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        file_paths = [str(url.toLocalFile()) for url in event.mimeData().urls()]
        self.filesDropped.emit(file_paths)


class RobotPositionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.position_edits = []
        self.labels = []
        # --- MODIFICATION: Removed inline stylesheet ---

        self.presets = {
            1: [1.0, 0.0, 1.57, 0.0, 1.57, 0.0],
            2: [0.0, 0.5, 1.0, 0.5, 1.0, 0.0]
        }

        # --- MODIFICATION: Use QGroupBox for title and layout ---
        self.angle_group_box = QGroupBox("Angle")
        # You can use QSS to style this group box, e.g., self.angle_group_box.setObjectName("angleGroup")
        grid_layout = QGridLayout(self.angle_group_box)
        grid_layout.setContentsMargins(10, 10, 10, 10)

        for i in range(6):
            label = QLabel(f"Joint {i + 1}:")
            line_edit = QLineEdit()
            line_edit.setText(f"{self.presets[1][i]}")

            # Arrange in 3 rows, 2 columns
            row = i % 3
            col = (i // 3) * 2  # Column 0 for first 3, Column 2 for next 3

            grid_layout.addWidget(label, row, col)
            grid_layout.addWidget(line_edit, row, col + 1)

            self.labels.append(label)
            self.position_edits.append(line_edit)

        # Initially hide the group box
        self.angle_group_box.setVisible(False)
        self.layout().addWidget(self.angle_group_box)

        # Group all buttons in a single horizontal layout to save vertical space
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 10, 0, 0)

        self.preset_buttons = []
        for i in range(1, 3):
            btn = QPushButton(f"Preset {i}")
            btn.clicked.connect(lambda checked, p=i: self.apply_preset(p))
            action_layout.addWidget(btn)
            self.preset_buttons.append(btn)

        action_layout.addStretch()

        self.send_button = QPushButton("Send Positions")
        self.send_button.clicked.connect(self.send_positions)
        action_layout.addWidget(self.send_button)

        self.layout().addWidget(action_widget)
        self.action_widget = action_widget
        self.action_widget.setVisible(False)

    def apply_preset(self, preset_number):
        preset_values = self.presets.get(preset_number)
        if preset_values:
            for i, value in enumerate(preset_values):
                self.position_edits[i].setText(f"{value:.2f}")

    def send_positions(self):
        positions = [float(edit.text()) for edit in self.position_edits]
        print("Sending positions:", positions)
        RobotController().send_positions_joint_angle(positions)

    def toggle_visibility(self):
        isVisible = not self.isVisible()
        self.angle_group_box.setVisible(isVisible)
        self.action_widget.setVisible(isVisible)
        self.setVisible(isVisible)


class RobotToolPositionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        # --- MODIFICATION: Removed inline stylesheet ---

        self.presets = {
            1: ([-0.05, -0.2, 0.5], [0.0, 1.0, 0.0, 0.0]),
            2: ([0.15, 0.3, 0.4], [0.0, 1.0, 0.0, 0.0])
        }
        self.labels = {}
        self.line_edits = {}

        # Group inputs into Position and Orientation
        self.input_container = QWidget()
        top_layout = QHBoxLayout(self.input_container)
        top_layout.setContentsMargins(0, 0, 0, 0)

        position_group = QGroupBox("Position (XYZ)")
        position_layout = QVBoxLayout(position_group)

        orientation_group = QGroupBox("Orientation (Quaternion)")
        orientation_layout = QVBoxLayout(orientation_group)

        self.setupControls('X', self.presets[1][0][0], position_layout)
        self.setupControls('Y', self.presets[1][0][1], position_layout)
        self.setupControls('Z', self.presets[1][0][2], position_layout)

        self.setupControls('w', self.presets[1][1][0], orientation_layout)
        self.setupControls('i', self.presets[1][1][1], orientation_layout)
        self.setupControls('j', self.presets[1][1][2], orientation_layout)
        self.setupControls('k', self.presets[1][1][3], orientation_layout)

        top_layout.addWidget(position_group)
        top_layout.addWidget(orientation_group)

        self.layout().addWidget(self.input_container)
        self.input_container.setVisible(False)

        # Group all buttons in a single horizontal layout to save vertical space
        action_widget = QWidget()
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 10, 0, 0)

        self.preset_buttons = []
        for i in range(1, 3):
            btn = QPushButton(f"Preset {i}")
            btn.clicked.connect(lambda checked, p=i: self.apply_preset(p))
            action_layout.addWidget(btn)
            self.preset_buttons.append(btn)

        action_layout.addStretch()

        self.send_button = QPushButton("Send Tool Position")
        self.send_button.clicked.connect(self.send_positions)
        action_layout.addWidget(self.send_button)

        self.layout().addWidget(action_widget)
        self.action_widget = action_widget
        self.action_widget.setVisible(False)

    def apply_preset(self, preset_number):
        preset_values = self.presets.get(preset_number)
        if preset_values:
            positions, quaternion = preset_values
            for i, coord in enumerate(['X', 'Y', 'Z']):
                self.line_edits[coord].setText(f"{positions[i]:.2f}")
            for i, part in enumerate(['w', 'i', 'j', 'k']):
                self.line_edits[part].setText(f"{quaternion[i]:.2f}")

    def setupControls(self, identifier, preset_value, layout):
        control_layout = QHBoxLayout()
        label = QLabel(f"{identifier}:")
        line_edit = QLineEdit()
        line_edit.setText(f"{preset_value:.2f}")

        control_layout.addWidget(label)
        control_layout.addWidget(line_edit)
        layout.addLayout(control_layout)

        self.labels[identifier] = label
        self.line_edits[identifier] = line_edit

    def send_positions(self):
        try:
            positions = [float(self.line_edits[coord].text()) for coord in ['X', 'Y', 'Z']]
            quaternion = tuple(float(self.line_edits[part].text()) for part in ['w', 'i', 'j', 'k'])
            print("Sending tool position:", positions, "with quaternion:", quaternion)
            RobotController().send_positions_tool_position(positions, quaternion)
        except ValueError:
            print("Invalid input! Please enter valid numbers.")

    def toggle_visibility(self):
        isVisible = not self.isVisible()
        self.input_container.setVisible(isVisible)
        self.action_widget.setVisible(isVisible)
        self.setVisible(isVisible)


class MiniRobotToolPositionController:
    # (No changes needed in this class)
    def __init__(self, mini_robot, coord_inputs, speed_input, angle_inputs, angle_speed_input, log_display):
        self.mini_robot = mini_robot
        self.coord_inputs = coord_inputs
        self.speed_input = speed_input
        self.angle_inputs = angle_inputs
        self.angle_speed_input = angle_speed_input
        self.log_display = log_display

    def send_coords(self):
        if not self.mini_robot:
            self.log_display.append("Robot not connected.")
            return
        try:
            coords = [float(edit.text()) for edit in self.coord_inputs]
            speed = int(self.speed_input.text())
            self.mini_robot.move_to_coords(coords, speed)
            self.log_display.append(f"Sent coords: {coords} at speed {speed}")
        except ValueError:
            self.log_display.append("Invalid coordinate or speed input!")

    def send_angles(self):
        if not self.mini_robot:
            self.log_display.append("Robot not connected.")
            return
        try:
            angles = [float(edit.text()) for edit in self.angle_inputs]
            speed = int(self.angle_speed_input.text())
            self.mini_robot.send_angles(angles, speed)
            self.log_display.append(f"Sent angles: {angles} at speed {speed}")
        except ValueError:
            self.log_display.append("Invalid angle or speed input!")

    def get_coords(self):
        if not self.mini_robot:
            self.log_display.append("Robot not connected.")
            return
        coords = self.mini_robot.get_current_coords()
        self.log_display.append("Current Coords: " + str(coords))

    def stop(self):
        if not self.mini_robot:
            self.log_display.append("Robot not connected.")
            return
        self.mini_robot.stop()
        self.log_display.append("Stop command sent.")

    def pause(self):
        if not self.mini_robot:
            self.log_display.append("Robot not connected.")
            return
        self.mini_robot.pause()
        self.log_display.append("Pause command sent.")

    def resume(self):
        if not self.mini_robot:
            self.log_display.append("Robot not connected.")
            return
        self.mini_robot.resume()
        self.log_display.append("Resume command sent.")


class UI(QSplitter):
    def __init__(self, orientation: QtCore.Qt.Orientation):
        super().__init__(orientation)
        # --- MODIFICATION: Removed inline stylesheet ---

        # Core APIs
        self.sensor_api = ArduinoCommander()
        self.robot_api = RobotController()
        self.mini_robot = None
        self.mini_robot_connected = False

        self.setHandleWidth(3)
        self.setup_layout()

        self.mesh_functions = MyMeshLab(self)
        self.sensor_functions = MySensor(self)
        self.read_timer = QTimer()
        self.connect_function()
        self.adjust_splitter_sizes()

        if not self.robot_api.use_ros:
            self.tab_widget.setTabEnabled(1, False)
            self.robots_sub_tabs.setTabEnabled(0, False)
            self.robots_sub_tabs.setTabEnabled(1, False)
            self.disable_robot_controls(True)

    def disable_robot_controls(self, disable: bool):
        for btn in [self.send_coords_button, self.send_angles_button, self.get_coords_button,
                    self.stop_button, self.pause_button, self.resume_button]:
            btn.setDisabled(disable)
        self.robots_sub_tabs.setTabEnabled(0, not disable)
        self.robots_sub_tabs.setTabEnabled(1, not disable)

    def setup_layout(self):
        self.widget_plotter = PlotterWidget()
        layout_plotter = QGridLayout(self.widget_plotter)
        layout_plotter.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.widget_plotter)
        self.plotter.background_color = '#202020'
        layout_plotter.addWidget(self.plotter.interactor)
        self.widget_plotter.setVisible(False)

        self.widget_plotter_2 = PlotterWidget()
        layout_plotter_2 = QGridLayout(self.widget_plotter_2)
        layout_plotter_2.setContentsMargins(0, 0, 0, 0)
        self.plotter_2 = QtInteractor(self.widget_plotter_2)
        self.plotter_2.background_color = '#202020'
        layout_plotter_2.addWidget(self.plotter_2.interactor)

        self.log_display = QTextEdit()
        self.log_display.setObjectName("logDisplay")  # Set object name for QSS
        self.log_display.setReadOnly(True)
        self.log_display.setVisible(False)
        self.log_display.textChanged.connect(self.show_log_if_hidden)

        self.splitter_1 = QSplitter(Qt.Horizontal, self)
        self.splitter_1.addWidget(self.widget_plotter)
        self.splitter_1.addWidget(self.widget_plotter_2)
        self.splitter_1.addWidget(self.log_display)
        self.splitter_1.setHandleWidth(3)

        self.splitter_2 = QSplitter(Qt.Vertical, self)
        # --- MODIFICATION: Removed inline stylesheet ---
        self.splitter_2.setHandleWidth(3)

        self.position_entry_widget = RobotPositionWidget()
        self.position_quaternion_widget = RobotToolPositionWidget()
        self.position_entry_widget.setVisible(False)
        self.position_quaternion_widget.setVisible(False)

        self.widget_func = QWidget()
        self.layout_func = QVBoxLayout(self.widget_func)
        self.layout_func.addWidget(self.position_entry_widget)
        self.layout_func.addWidget(self.position_quaternion_widget)

        self.tab_widget = QTabWidget()
        self.tab_widget.setUsesScrollButtons(False)
        self.setup_tabs()

        self.layout_func.addWidget(self.tab_widget)
        self.splitter_2.addWidget(self.widget_func)
        self.addWidget(self.splitter_1)
        self.addWidget(self.splitter_2)

    def setup_tabs(self):
        # Tab 1: Sensor
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.setup_tab1(tab1_layout)
        self.tab_widget.addTab(tab1, "Sensor")

        # Tab 2: ROBOTS (contains two subtabs)
        robots_tab = QWidget()
        robots_layout = QVBoxLayout(robots_tab)
        self.robots_sub_tabs = QTabWidget()
        self.robots_sub_tabs.setUsesScrollButtons(False)

        # ─── Subtab “Robot” ───
        robot_page = QWidget()
        robot_layout = QVBoxLayout(robot_page)
        self.setup_tab2(robot_layout)
        self.robots_sub_tabs.addTab(robot_page, "Robot")

        # ─── Subtab “Mini Robot” ───
        mini_page = QWidget()
        mini_layout = QVBoxLayout(mini_page)
        self.setup_tab5(mini_layout)
        self.robots_sub_tabs.addTab(mini_page, "Mini Robot")

        robots_layout.addWidget(self.robots_sub_tabs)
        self.tab_widget.addTab(robots_tab, "Robots")

        # Tab 3: AI
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        self.setup_tab3(tab3_layout)
        self.tab_widget.addTab(tab3, "AI")

        # Tab 4: Extra
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        self.setup_tab4(tab4_layout)
        self.tab_widget.addTab(tab4, "Extra")

    def setup_tab1(self, layout):
        read_group = QGroupBox("Read Operations")
        send_group = QGroupBox("Send Operations")
        read_layout = QVBoxLayout()
        send_layout = QVBoxLayout()

        self.read_sensor_api_button = QPushButton("Sensor API Raw Data (ADJUST IN API!!)")
        self.read_sensor_channel_button = QPushButton("Sensor API Channel (ADJUST IN API!!)")
        self.read_sensor_raw_button = QPushButton("Sensor Raw Data")
        self.read_sensor_raw_ave_button = QPushButton("Sensor Raw Ave Data")
        self.read_sensor_diff_button = QPushButton("Sensor Diff Data")

        read_layout.addWidget(self.read_sensor_api_button)
        read_layout.addWidget(self.read_sensor_channel_button)
        read_layout.addWidget(self.read_sensor_raw_button)
        read_layout.addWidget(self.read_sensor_raw_ave_button)
        read_layout.addWidget(self.read_sensor_diff_button)

        read_group.setLayout(read_layout)
        send_group.setLayout(send_layout)

        layout.addWidget(read_group)
        layout.addWidget(send_group)

        self.sensor_choice = QListWidget(self.widget_func)
        self.sensor_choice.setSelectionMode(QListWidget.SingleSelection)
        self.sensor_choice.addItems([
            "Elbow", "Kuka", "Double Curve", "2D", "Half Cylinder Surface",
            "Mini-Robot Large Skin", "Mini-Robot Small Skin", "Geneva Demo"
        ])
        self.sensor_choice.setCurrentRow(0)
        send_layout.addWidget(self.sensor_choice)

        self.serial_channel = QListWidget(self.widget_func)
        self.serial_channel.setSelectionMode(QListWidget.MultiSelection)
        send_layout.addWidget(self.serial_channel)

        self.buildScene = QPushButton("Build Scene", self.widget_func)
        send_layout.addWidget(self.buildScene)

        self.sensor_update = QPushButton("Update Sensor", self.widget_func)
        send_layout.addWidget(self.sensor_update)

    def setup_tab2(self, layout):
        self.read_group_robot = QGroupBox("Read Operations")
        send_group = QGroupBox("Send Operations")
        read_layout = QVBoxLayout()
        send_layout = QVBoxLayout()

        self.read_joint_angle_button = QPushButton("Read Joint Angle")
        self.read_tool_position_button = QPushButton("Read Tool Position")
        self.send_position_PTP_J_button = QPushButton("Send Joint Angle")
        self.send_position_PTP_T_button = QPushButton("Send Tool Position")
        self.send_script_button = QPushButton("Send Script")
        self.show_robot_button = QPushButton("Import 3D Robot Model")
        self.continuous_read_button = QPushButton("Real-time Live 3D Robot Model", self.widget_func)

        read_layout.addWidget(self.read_joint_angle_button)
        read_layout.addWidget(self.read_tool_position_button)
        read_layout.addWidget(self.show_robot_button)
        read_layout.addWidget(self.continuous_read_button)
        send_layout.addWidget(self.send_position_PTP_J_button)
        send_layout.addWidget(self.send_position_PTP_T_button)
        send_layout.addWidget(self.send_script_button)

        self.read_group_robot.setLayout(read_layout)
        send_group.setLayout(send_layout)
        layout.addWidget(self.read_group_robot)
        layout.addWidget(send_group)

    def setup_tab3(self, layout):
        training_group = QGroupBox("Data Training")
        testing_group = QGroupBox("AI Model")
        gesture_layout = QVBoxLayout()
        testing_layout = QVBoxLayout()

        self.set_no_trigger_button = QPushButton("Set No Trigger Mode")
        gesture_layout.addWidget(self.set_no_trigger_button)
        self.set_no_trigger_auto_button = QPushButton("Set No Trigger Auto Mode")
        gesture_layout.addWidget(self.set_no_trigger_auto_button)
        self.set_no_trigger_no_updatecal_auto_button = QPushButton("Set No Trigger No UpdateCal Auto Mode")
        gesture_layout.addWidget(self.set_no_trigger_no_updatecal_auto_button)
        self.set_trigger_button = QPushButton("Set Trigger Mode")
        gesture_layout.addWidget(self.set_trigger_button)

        first_row_layout = QHBoxLayout()
        gesture_label = QLabel("Enter Gesture Number:")
        self.gesture_number_input = QLineEdit()
        self.gesture_number_input.setFixedSize(50, 40)
        first_row_layout.addWidget(gesture_label)
        first_row_layout.addWidget(self.gesture_number_input)
        self.record_gesture_button = QPushButton("Record")
        gesture_layout.addLayout(first_row_layout)
        gesture_layout.addWidget(self.record_gesture_button)

        self.predict_lstm_gesture_button = QPushButton("Predict (LSTM)")
        self.predict_hierarchical_transformer_gesture_button = QPushButton("Predict (HierarchicalTransformer)")
        self.toggle_prediction_mode_button = QPushButton("Toggle Prediction Mode (Continues/Single)")
        self.predict_threelevel_hierarchical_transformer_gesture_button = QPushButton("Predict (ThreeLevel)")
        self.update_sensor_button = QPushButton("Update Sensor")
        self.activate_switch_model_button = QPushButton("Activate Switch Model")
        self.activate_rule_based_button = QPushButton("Activate Rule Based")

        testing_layout.addWidget(self.predict_lstm_gesture_button)
        testing_layout.addWidget(self.predict_hierarchical_transformer_gesture_button)
        testing_layout.addWidget(self.toggle_prediction_mode_button)
        testing_layout.addWidget(self.predict_threelevel_hierarchical_transformer_gesture_button)
        testing_layout.addWidget(self.update_sensor_button)
        testing_layout.addWidget(self.activate_switch_model_button)
        testing_layout.addWidget(self.activate_rule_based_button)

        training_group.setLayout(gesture_layout)
        testing_group.setLayout(testing_layout)
        layout.addWidget(training_group)
        layout.addWidget(testing_group)

    def setup_tab4(self, layout):
        test_group = QGroupBox("Testing Operations")
        test_layout = QVBoxLayout()
        self.read_raw_all_prots_button = QPushButton("Read All Port")
        self.enable_joint_velocity_mode_button = QPushButton("Enable Joint Velocity Mode")
        self.suspend_end_effector_velocity_mode_button = QPushButton("Suspend End Effector Velocity")
        self.stop_joint_velocity_mode_button = QPushButton("Stop Joint Velocity")
        self.enable_end_effector_velocity_mode_button = QPushButton("Enable End Effector Velocity")
        self.stop_end_effector_velocity_mode_button = QPushButton("Stop End Effector Velocity")
        self.stop_and_clear_buffer_button = QPushButton("Stop and Clear Buffer")
        self.sensor_start_geneva = QPushButton("startSensorGeneva", self.widget_func)
        self.sensor_update_geneva = QPushButton("updateSensorGeneva", self.widget_func)

        test_layout.addWidget(self.read_raw_all_prots_button)
        test_layout.addWidget(self.enable_joint_velocity_mode_button)
        test_layout.addWidget(self.stop_joint_velocity_mode_button)
        test_layout.addWidget(self.enable_end_effector_velocity_mode_button)
        test_layout.addWidget(self.suspend_end_effector_velocity_mode_button)
        test_layout.addWidget(self.stop_end_effector_velocity_mode_button)
        test_layout.addWidget(self.stop_and_clear_buffer_button)
        test_layout.addWidget(self.sensor_start_geneva)
        test_layout.addWidget(self.sensor_update_geneva)
        test_group.setLayout(test_layout)
        layout.addWidget(test_group)

    def setup_tab5(self, layout):
        basic_group = QGroupBox("Basic Controls")
        basic_layout = QVBoxLayout()
        self.stop_button = QPushButton("Stop")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        for btn in (self.stop_button, self.pause_button, self.resume_button):
            basic_layout.addWidget(btn)
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

        conn_group = QGroupBox("Connect Mini Robot")
        conn_layout = QHBoxLayout(conn_group)
        conn_layout.addWidget(QLabel("/dev/ttyACM"))
        self.port_input = QLineEdit()
        self.port_input.setPlaceholderText("e.g. 3")
        self.connect_robot_button = QPushButton("Connect")
        conn_layout.addWidget(self.port_input)
        conn_layout.addWidget(self.connect_robot_button)
        layout.addWidget(conn_group)
        self.connect_robot_button.clicked.connect(self.connect_mini_robot)

        coord_group = QGroupBox("Coordinate Control")
        coord_layout = QVBoxLayout()
        self.coord_inputs = []
        for label_text in ["X", "Y", "Z", "Rx", "Ry", "Rz"]:
            h = QHBoxLayout()
            h.addWidget(QLabel(f"{label_text}:"))
            le = QLineEdit()
            h.addWidget(le)
            coord_layout.addLayout(h)
            self.coord_inputs.append(le)
        speed_h = QHBoxLayout()
        speed_h.addWidget(QLabel("Speed:"))
        self.speed_input = QLineEdit("50")
        speed_h.addWidget(self.speed_input)
        coord_layout.addLayout(speed_h)
        self.send_coords_button = QPushButton("Move to Coordinates")
        coord_layout.addWidget(self.send_coords_button)
        coord_group.setLayout(coord_layout)
        layout.addWidget(coord_group)

        angle_group = QGroupBox("Angle Control")
        angle_layout = QVBoxLayout()
        self.angle_inputs = []
        for jt in ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]:
            h = QHBoxLayout()
            h.addWidget(QLabel(f"{jt}:"))
            le = QLineEdit()
            h.addWidget(le)
            angle_layout.addLayout(h)
            self.angle_inputs.append(le)
        ah = QHBoxLayout()
        ah.addWidget(QLabel("Speed:"))
        self.angle_speed_input = QLineEdit("50")
        ah.addWidget(self.angle_speed_input)
        angle_layout.addLayout(ah)
        self.send_angles_button = QPushButton("Send Angles")
        angle_layout.addWidget(self.send_angles_button)
        angle_group.setLayout(angle_layout)
        layout.addWidget(angle_group)

        self.get_coords_button = QPushButton("Get Current Coordinates")
        layout.addWidget(self.get_coords_button)

        self.tool_controller = MiniRobotToolPositionController(
            mini_robot=self.mini_robot, coord_inputs=self.coord_inputs,
            speed_input=self.speed_input, angle_inputs=self.angle_inputs,
            angle_speed_input=self.angle_speed_input, log_display=self.log_display
        )
        self.send_coords_button.clicked.connect(self.tool_controller.send_coords)
        self.send_angles_button.clicked.connect(self.tool_controller.send_angles)
        self.get_coords_button.clicked.connect(self.tool_controller.get_coords)
        self.stop_button.clicked.connect(self.tool_controller.stop)
        self.pause_button.clicked.connect(self.tool_controller.pause)
        self.resume_button.clicked.connect(self.tool_controller.resume)

        self.disable_robot_controls(True)

    def connect_mini_robot(self):
        suffix = self.port_input.text().strip()
        if not suffix.isdigit():
            self.log_display.append("❌ Port suffix must be a number.")
            return

        port = f"/dev/ttyACM{suffix}"
        if os.path.exists(port):
            try:
                self.mini_robot = MyCobotAPI(serial_port=port, baud_rate=115200)
                self.mini_robot_connected = True
                self.log_display.append(f"✅ Connected to mini robot on {port}")
                self.tool_controller.mini_robot = self.mini_robot
                self.disable_robot_controls(False)
            except Exception as e:
                self.mini_robot = None
                self.log_display.append(f"❌ Failed to init mini robot: {e}")
        else:
            self.log_display.append(f"❌ Port {port} not found. Check cable & try again.")
            self.mini_robot = None

    def disable_robot_controls(self, disable: bool):
        for btn in [
            self.stop_button, self.pause_button, self.resume_button,
            self.send_coords_button, self.send_angles_button,
            self.get_coords_button
        ]:
            btn.setDisabled(disable)
        self.tab_widget.setTabEnabled(4, not disable)

    def connect_function(self):
        self.read_sensor_api_button.pressed.connect(
            lambda: self.log_display.append(f"API raw data: {self.sensor_api.read_raw()}")
        )
        self.read_sensor_channel_button.pressed.connect(
            lambda: self.log_display.append(f"Sensor channel data: {self.sensor_api.channel_check()}")
        )
        self.read_sensor_raw_button.pressed.connect(
            lambda: self.log_display.append(f"Raw data: {self.sensor_functions.read_sensor_raw_data()}")
        )
        self.read_sensor_raw_ave_button.pressed.connect(
            lambda: self.log_display.append(f"Raw ave data: {self.sensor_functions.read_sensor_raw_ave_data()}")
        )
        self.read_sensor_diff_button.pressed.connect(
            lambda: self.log_display.append(f"Diff data: {self.sensor_functions.read_sensor_diff_data()}")
        )

        self.read_joint_angle_button.pressed.connect(
            lambda: self.log_display.append(f"Joint angles: {self.robot_api.get_current_positions()}")
        )
        self.read_tool_position_button.pressed.connect(
            lambda: self.log_display.append(f"Tool position: {self.robot_api.get_current_tool_position()}")
        )
        self.send_position_PTP_J_button.pressed.connect(
            self.toggle_joint_angle_input
        )
        self.send_position_PTP_T_button.pressed.connect(
            self.toggle_tool_position_input
        )
        self.show_robot_button.pressed.connect(lambda: self.mesh_functions.addRobot())
        self.read_raw_all_prots_button.pressed.connect(lambda: self.sensor_functions.read_raw_all_ports())
        self.continuous_read_button.pressed.connect(self.toggle_continuous_read)
        self.read_timer.timeout.connect(self.update_robot_status)
        self.buildScene.pressed.connect(lambda: self.sensor_functions.buildScene())
        self.sensor_update.pressed.connect(lambda: self.sensor_functions.updateCal())
        self.sensor_start_geneva.pressed.connect(lambda: self.sensor_functions.startSensor_geneva())
        self.sensor_update_geneva.pressed.connect(lambda: self.sensor_functions.updateCal_geneva())
        self.record_gesture_button.pressed.connect(self.start_record_gesture)
        self.set_no_trigger_button.pressed.connect(
            lambda: self.sensor_functions.record_gesture_class.set_trigger_mode("no_trigger")
        )
        self.set_no_trigger_auto_button.pressed.connect(
            lambda: self.sensor_functions.record_gesture_class.set_trigger_mode("no_trigger_auto")
        )
        self.set_no_trigger_no_updatecal_auto_button.pressed.connect(
            lambda: self.sensor_functions.record_gesture_class.set_trigger_mode("no_trigger_no_updatecal_auto")
        )
        self.set_trigger_button.pressed.connect(
            lambda: self.sensor_functions.record_gesture_class.set_trigger_mode("trigger")
        )
        self.predict_lstm_gesture_button.pressed.connect(
            lambda: self.sensor_functions.lstm_class.toggle_gesture_recognition()
        )
        self.predict_hierarchical_transformer_gesture_button.pressed.connect(
            lambda: self.sensor_functions.hierarchical_transformer_class.toggle_gesture_recognition()
        )
        self.toggle_prediction_mode_button.pressed.connect(
            lambda: self.sensor_functions.hierarchical_transformer_class.toggle_prediction_mode()
        )
        self.predict_threelevel_hierarchical_transformer_gesture_button.pressed.connect(
            lambda: self.sensor_functions.threelevel_hierarchical_transformer_class.toggle_gesture_recognition()
        )
        self.activate_switch_model_button.pressed.connect(
            lambda: self.sensor_functions.lstm_class.toggle_model()
        )
        self.activate_rule_based_button.pressed.connect(
            lambda: self.sensor_functions.rule_based_class.activate_rule_based()
        )
        self.update_sensor_button.pressed.connect(
            lambda: self.sensor_functions.updateCal()
        )

    def start_record_gesture(self):
        gesture_number = self.gesture_number_input.text().strip()
        if not gesture_number:
            self.log_display.append("Please enter a gesture number or name.")
            return
        self.sensor_functions.record_gesture_class.start_record_gesture(gesture_number)

    def toggle_continuous_read(self):
        if self.read_timer.isActive():
            self.read_timer.stop()
            self.continuous_read_button.setText("Start Continuous Read")
        else:
            self.read_timer.start(0)
            self.continuous_read_button.setText("Stop Continuous Read")

    def update_robot_status(self):
        print("DELETED")

    def toggle_plotter_visibility(self):
        self.log_display.setVisible(not self.log_display.isVisible())
        self.adjust_splitter_sizes()

    def toggle_joint_angle_input(self):
        if self.position_quaternion_widget.isVisible():
            self.position_quaternion_widget.toggle_visibility()
        self.position_entry_widget.toggle_visibility()
        self.read_group_robot.setVisible(not self.position_entry_widget.isVisible())

    def toggle_tool_position_input(self):
        if self.position_entry_widget.isVisible():
            self.position_entry_widget.toggle_visibility()
        self.position_quaternion_widget.toggle_visibility()
        self.read_group_robot.setVisible(not self.position_quaternion_widget.isVisible())

    def show_log_if_hidden(self):
        if not self.log_display.isVisible():
            self.log_display.setVisible(True)
            self.adjust_splitter_sizes()

    def adjust_splitter_sizes(self):
        total_width = self.splitter_1.width()
        if self.log_display.isVisible():
            self.splitter_1.setSizes([int(total_width * 0.4), int(total_width * 0.4), int(total_width * 0.2)])
        else:
            self.splitter_1.setSizes([int(total_width * 0.5), int(total_width * 0.5), 0])

    def reLayout(self):
        self.setSizes([round(self.width() * 4), round(self.width())])
        self.splitter_1.setSizes([self.width(), self.width(), 0])  # Log hidden initially
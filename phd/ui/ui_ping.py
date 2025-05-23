from PyQt5 import QtCore
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon, QColor, QCursor
from PyQt5.QtCore import pyqtSignal, Qt, QRect, QEvent, QTimer
from PyQt5.QtWidgets import (QSplitter, QWidget, QGridLayout, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
                             QTreeWidget, QTreeWidgetItem, QTabWidget, QDialog, QLineEdit, QTextEdit, QSlider,
                             QGroupBox, QComboBox, QScrollArea, QListWidget, QListWidgetItem)
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
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        file_paths = []
        for url in event.mimeData().urls():
            file_paths.append(str(url.toLocalFile()))
        self.filesDropped.emit(file_paths)  # 发射带有文件路径列表参数的信号


class RobotPositionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.position_edits = []
        self.labels = []
        self.sliders = []
        self.setStyleSheet("QLabel, QLineEdit, QPushButton { color: white; }")

        # Preset values for joint angles
        preset_angles = [1.0, 0.0, 1.57, 0.0, 1.57, 0.0]

        for i in range(1, 7):
            label = QLabel(f"Joint {i} Position:")
            line_edit = QLineEdit()
            line_edit.setText(f"{preset_angles[i-1]}")

            # Creating a slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-314)  # Slider range -3.14 to 3.14
            slider.setMaximum(314)
            slider.setValue(int(preset_angles[i-1] * 100))  # Convert angle to slider position
            slider.valueChanged.connect(lambda value, le=line_edit: le.setText(f"{value / 100:.2f}"))

            # Initially hide slider and its related widgets
            label.setVisible(False)
            line_edit.setVisible(False)
            slider.setVisible(False)

            self.layout().addWidget(label)
            self.layout().addWidget(line_edit)
            self.layout().addWidget(slider)

            self.labels.append(label)
            self.position_edits.append(line_edit)
            self.sliders.append(slider)

        self.send_button = QPushButton("Send Positions")
        self.layout().addWidget(self.send_button)
        self.send_button.clicked.connect(self.send_positions)
        self.send_button.setVisible(False)  # Button is hidden initially

    def send_positions(self):
        positions = [float(edit.text()) for edit in self.position_edits]
        print("Sending positions:", positions)
        RobotController().send_positions_joint_angle(positions)

    def toggle_visibility(self):
        isVisible = not self.isVisible()  # Check the current visibility and toggle it
        for label, edit, slider in zip(self.labels, self.position_edits, self.sliders):
            label.setVisible(isVisible)
            edit.setVisible(isVisible)
            slider.setVisible(isVisible)
        self.send_button.setVisible(isVisible)
        self.setVisible(isVisible)


class RobotToolPositionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.setStyleSheet("QLabel, QLineEdit, QPushButton { color: white; }")

        # Preset values
        preset_positions = [-0.455, -0.575, 0.29]
        preset_quaternion = [0.0, 1.0, 0.0, 0.0]

        # Initialize dictionaries for storing widgets
        self.labels = {}
        self.line_edits = {}
        self.sliders = {}

        # Set up coordinates and quaternion inputs with sliders
        self.setupControls('X', preset_positions[0], -1, 1)
        self.setupControls('Y', preset_positions[1], -1, 1)
        self.setupControls('Z', preset_positions[2], -1, 1)
        self.setupControls('w', preset_quaternion[0], -1, 1)
        self.setupControls('i', preset_quaternion[1], -1, 1)
        self.setupControls('j', preset_quaternion[2], -1, 1)
        self.setupControls('k', preset_quaternion[3], -1, 1)

        # Send button setup
        self.send_button = QPushButton("Send Tool Position")
        self.layout().addWidget(self.send_button)
        self.send_button.clicked.connect(self.send_positions)
        self.send_button.setVisible(False)  # Initially hidden

    def setupControls(self, identifier, preset_value, min_value, max_value):
        # Create and set up labels, line edits, and sliders
        label = QLabel(f"{identifier} Coordinate:")
        line_edit = QLineEdit()
        line_edit.setText(f"{preset_value:.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value * 100)  # Assuming slider works with integers
        slider.setMaximum(max_value * 100)
        slider.setValue(int(preset_value * 100))
        slider.valueChanged.connect(lambda value, id=identifier: self.updateLineEdit(value, id))

        # Add to layout
        self.layout().addWidget(label)
        self.layout().addWidget(line_edit)
        self.layout().addWidget(slider)

        # Store widgets
        self.labels[identifier] = label
        self.line_edits[identifier] = line_edit
        self.sliders[identifier] = slider

        # Initially hidden
        label.setVisible(False)
        line_edit.setVisible(False)
        slider.setVisible(False)

    def updateLineEdit(self, value, identifier):
        # Update the corresponding line edit from slider value
        self.line_edits[identifier].setText(f"{value / 100:.2f}")

    def send_positions(self):
        try:
            positions = [float(self.line_edits[coord].text()) for coord in ['X', 'Y', 'Z']]
            quaternion = tuple(float(self.line_edits[part].text()) for part in ['w', 'i', 'j', 'k'])
            print("Sending tool position:", positions, "with quaternion:", quaternion)
            RobotController().send_positions_tool_position(positions, quaternion)
        except ValueError:
            print("Invalid input! Please enter valid numbers.")

    def toggle_visibility(self):
        isVisible = not self.isVisible()  # Check the current visibility and toggle it
        for widget in list(self.labels.values()) + list(self.line_edits.values()) + list(self.sliders.values()):
            widget.setVisible(isVisible)
        self.send_button.setVisible(isVisible)
        self.setVisible(isVisible)


class MiniRobotToolPositionController:
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
        self.setStyleSheet("QPushButton { color: white; }")

        # Core APIs
        self.sensor_api = ArduinoCommander()
        self.robot_api  = RobotController()

        # Defer mini‐robot until user connects
        self.mini_robot            = None
        self.mini_robot_connected  = False

        self.setHandleWidth(3)
        self.setup_layout()

        # Helpers
        self.mesh_functions    = MyMeshLab(self)
        self.sensor_functions  = MySensor(self)

        # Timer for continuous robot reads
        self.read_timer = QTimer()
        self.connect_function()
        self.adjust_splitter_sizes()

        # If not using ROS, disable all robot-related tabs & controls
        if not self.robot_api.use_ros:
            # main “Robots” tab is index 1 now
            self.tab_widget.setTabEnabled(1, False)
            # also disable the two subtabs
            self.robots_sub_tabs.setTabEnabled(0, False)
            self.robots_sub_tabs.setTabEnabled(1, False)
            self.disable_robot_controls(True)

    def disable_robot_controls(self, disable: bool):
        """Toggle enable/disable of all robot controls and subtabs."""
        # first disable/enable all robot buttons
        for btn in [
            self.send_coords_button,
            self.send_angles_button,
            self.get_coords_button,
            self.stop_button,
            self.pause_button,
            self.resume_button
        ]:
            btn.setDisabled(disable)

        # then toggle the subtabs under “Robots”
        self.robots_sub_tabs.setTabEnabled(0, not disable)  # Robot
        self.robots_sub_tabs.setTabEnabled(1, not disable)  # Mini Robot
    def setup_layout(self):
        # SET UP WIDGET 1
        self.widget_plotter = PlotterWidget()
        self.widget_plotter.setObjectName("widgetPlotter")
        self.layout_plotter = QGridLayout(self.widget_plotter)
        self.layout_plotter.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.widget_plotter)
        self.plotter.background_color = '#202020'
        self.widget_plotter.setAcceptDrops(True)
        self.layout_plotter.addWidget(self.plotter.interactor)
        self.widget_plotter.setLayout(self.layout_plotter)
        self.widget_plotter.setVisible(False)

        # SET UP WIDGET 2
        self.widget_plotter_2 = PlotterWidget()
        self.widget_plotter_2.setObjectName("widgetPlotter2")
        self.layout_plotter_2 = QGridLayout(self.widget_plotter_2)
        self.layout_plotter_2.setContentsMargins(0, 0, 0, 0)
        self.plotter_2 = QtInteractor(self.widget_plotter_2)
        self.plotter_2.background_color = '#202020'
        self.widget_plotter_2.setAcceptDrops(True)
        self.layout_plotter_2.addWidget(self.plotter_2.interactor)
        self.widget_plotter_2.setLayout(self.layout_plotter_2)

        # SET UP WIDGET 3
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("QTextEdit { background-color: #202020; color: white; border: 1px solid #303030; }")
        self.log_display.setObjectName("logDisplay")
        self.log_display.setVisible(False)

        # SET UP SPLITTER 1
        self.splitter_1 = QSplitter(Qt.Horizontal, self)
        self.splitter_1.addWidget(self.widget_plotter)
        self.splitter_1.addWidget(self.widget_plotter_2)
        self.splitter_1.addWidget(self.log_display)  # Add log_display to the splitter
        self.splitter_1.setHandleWidth(3)

        # SET UP SPLITTER 2
        self.splitter_2 = QSplitter(Qt.Vertical, self)
        self.splitter_2.setStyleSheet("background-color: #303030;")
        self.splitter_2.setHandleWidth(3)

        # SET UP WIDGET 4 & 5
        self.position_entry_widget = RobotPositionWidget()
        self.position_quaternion_widget = RobotToolPositionWidget()
        self.position_entry_widget.setVisible(False)
        self.position_quaternion_widget.setVisible(False)

        self.widget_func = QWidget()
        self.layout_func = QVBoxLayout(self.widget_func)
        self.layout_func.addWidget(self.position_entry_widget)
        self.layout_func.addWidget(self.position_quaternion_widget)

        # SET UP TAB WIDGET
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
        robots_tab       = QWidget()
        robots_layout    = QVBoxLayout(robots_tab)
        self.robots_sub_tabs = QTabWidget()
        self.robots_sub_tabs.setUsesScrollButtons(False)

        # ─── Subtab “Robot” ───
        robot_page    = QWidget()
        robot_layout  = QVBoxLayout(robot_page)
        self.setup_tab2(robot_layout)
        self.robots_sub_tabs.addTab(robot_page, "Robot")

        # ─── Subtab “Mini Robot” ───
        mini_page    = QWidget()
        mini_layout  = QVBoxLayout(mini_page)
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

        self.read_sensor_raw_button = QPushButton("Read Sensor Raw Data")
        self.read_sensor_diff_button = QPushButton("Read Sensor Diff Data")
        self.read_sensor_raw_2_button = QPushButton("Read Sensor Raw Data Ave")

        read_layout.addWidget(self.read_sensor_raw_button)
        read_layout.addWidget(self.read_sensor_diff_button)
        read_layout.addWidget(self.read_sensor_raw_2_button)

        read_group.setLayout(read_layout)
        send_group.setLayout(send_layout)

        layout.addWidget(read_group)
        layout.addWidget(send_group)

        # NEW SENSOR WIDGET
        self.sensor_choice = QComboBox(self.widget_func)
        self.sensor_choice.addItem("Elbow")
        self.sensor_choice.addItem("Robot")
        self.sensor_choice.addItem("Double Curve")
        self.sensor_choice.addItem("2D")
        self.sensor_choice.addItem("Half Cylinder Surface")
        self.sensor_choice.addItem("Geneva Large Skin")
        self.sensor_choice.addItem("Geneva Small Skin")

        send_layout.addWidget(self.sensor_choice)

        # Replace the QComboBox with a QListWidget for multi-selection:
        self.serial_channel = QListWidget(self.widget_func)
        self.serial_channel.setSelectionMode(QListWidget.MultiSelection)
        send_layout.addWidget(self.serial_channel)

        self.buildScene = QPushButton("buildScene", self.widget_func)
        send_layout.addWidget(self.buildScene)

        self.sensor_start = QPushButton("startSensor", self.widget_func)
        send_layout.addWidget(self.sensor_start)
        # self.sensor_start.setDisabled(True)

        self.sensor_update = QPushButton("updateSensor", self.widget_func)
        send_layout.addWidget(self.sensor_update)
        # self.sensor_update.setDisabled(True)

        self.sensor_start_geneva = QPushButton("startSensorGeneva", self.widget_func)
        send_layout.addWidget(self.sensor_start_geneva)
        # self.sensor_start_geneva.setDisabled(True)

        self.sensor_update_geneva = QPushButton("updateSensorGeneva", self.widget_func)
        send_layout.addWidget(self.sensor_update_geneva)
        # self.sensor_update_geneva.setDisabled(True)

        self.erase_mesh = QPushButton("eraseMesh", self.widget_func)
        send_layout.addWidget(self.erase_mesh)

    def setup_tab2(self, layout):
        # Read & Send operations for the main Robot
        read_group = QGroupBox("Read Operations")
        send_group = QGroupBox("Send Operations")
        read_layout = QVBoxLayout()
        send_layout = QVBoxLayout()

        self.read_joint_angle_button  = QPushButton("Read Joint Angle")
        self.read_tool_position_button = QPushButton("Read Tool Position")
        self.send_position_PTP_J_button = QPushButton("Send Joint Angle")
        self.send_position_PTP_T_button = QPushButton("Send Tool Position")
        self.send_script_button        = QPushButton("Send Script")
        self.show_robot_button         = QPushButton("Import 3D Robot Model")
        self.continuous_read_button    = QPushButton("Real-time Live 3D Robot Model", self.widget_func)

        read_layout.addWidget(self.read_joint_angle_button)
        read_layout.addWidget(self.read_tool_position_button)
        read_layout.addWidget(self.show_robot_button)
        read_layout.addWidget(self.continuous_read_button)
        send_layout.addWidget(self.send_position_PTP_J_button)
        send_layout.addWidget(self.send_position_PTP_T_button)
        send_layout.addWidget(self.send_script_button)

        read_group.setLayout(read_layout)
        send_group.setLayout(send_layout)
        layout.addWidget(read_group)
        layout.addWidget(send_group)
    def setup_tab3(self, layout):
        # Create group boxes
        training_group = QGroupBox("Gesture")
        testing_group = QGroupBox("Algorithm")

        # Create a vertical layout for the gesture group
        gesture_layout = QVBoxLayout()
        testing_layout = QVBoxLayout()

        self.set_no_trigger_button = QPushButton("Set No Trigger Mode")
        self.set_no_trigger_button.setFixedSize(200, 40)
        gesture_layout.addWidget(self.set_no_trigger_button)
        self.set_trigger_button = QPushButton("Set Trigger Mode")
        self.set_trigger_button.setFixedSize(200, 40)
        gesture_layout.addWidget(self.set_trigger_button)

        # Create a horizontal layout for the first row
        first_row_layout = QHBoxLayout()

        # Create a label and input field
        gesture_label = QLabel("Enter Gesture Number:")
        self.gesture_number_input = QLineEdit()
        self.gesture_number_input.setFixedSize(50, 40)

        # Add label and input field to the first row layout
        first_row_layout.addWidget(gesture_label)
        first_row_layout.addWidget(self.gesture_number_input)

        # Create the record gesture button
        self.record_gesture_button = QPushButton("Record")
        self.record_gesture_button.setFixedSize(100, 40)

        # Add the first row layout and the button to the main gesture layout
        gesture_layout.addLayout(first_row_layout)
        gesture_layout.addWidget(self.record_gesture_button)

        # Continue with the rest of your code...
        # Create the test gesture button
        self.record_noise_auto_button = QPushButton("Record Noise Auto")
        self.record_noise_auto_button.setFixedSize(200, 40)
        gesture_layout.addWidget(self.record_noise_auto_button)


        # Create additional Algorithm buttons
        self.predict_gesture_button = QPushButton("Predict Gesture")
        self.update_sensor_button = QPushButton("Update Sensor")
        self.activate_switch_model_button = QPushButton("Activate Switch Model")
        self.activate_rule_based_button = QPushButton("Activate Rule Based")

        # Set fixed sizes for consistency
        for btn in [
            self.predict_gesture_button,
            self.update_sensor_button,
            self.activate_switch_model_button,
            self.activate_rule_based_button,
        ]:
            btn.setFixedSize(200, 40)

        # Add testing operation buttons to the testing layout
        testing_layout.addWidget(self.predict_gesture_button)
        testing_layout.addWidget(self.update_sensor_button)
        testing_layout.addWidget(self.activate_switch_model_button)
        testing_layout.addWidget(self.activate_rule_based_button)

        # Assign layouts to their respective group boxes
        training_group.setLayout(gesture_layout)
        testing_group.setLayout(testing_layout)

        # Add group boxes to the main layout
        layout.addWidget(training_group)
        layout.addWidget(testing_group)

    def setup_tab4(self, layout):
        # Create a group box for testing buttons
        test_group = QGroupBox("Testing Operations")
        test_layout = QVBoxLayout()

        # Button for testing
        self.plotter_visibility_button = QPushButton("Show/Hide Log Display")
        self.read_raw_all_prots_button = QPushButton("Read All Port")
        self.render_button = QPushButton("Render")
        self.enable_joint_velocity_mode_button = QPushButton("Enable Joint Velocity Mode")
        self.suspend_end_effector_velocity_mode_button = QPushButton("Suspend End Effector Velocity")
        self.stop_joint_velocity_mode_button = QPushButton("Stop Joint Velocity")
        self.enable_end_effector_velocity_mode_button = QPushButton("Enable End Effector Velocity")
        self.stop_end_effector_velocity_mode_button = QPushButton("Stop End Effector Velocity")
        self.stop_and_clear_buffer_button = QPushButton("Stop and Clear Buffer")


        # Add buttons to layout
        test_layout.addWidget(self.plotter_visibility_button)
        test_layout.addWidget(self.read_raw_all_prots_button)
        test_layout.addWidget(self.render_button)
        test_layout.addWidget(self.enable_joint_velocity_mode_button)
        test_layout.addWidget(self.stop_joint_velocity_mode_button)
        test_layout.addWidget(self.enable_end_effector_velocity_mode_button)
        test_layout.addWidget(self.suspend_end_effector_velocity_mode_button)
        test_layout.addWidget(self.stop_end_effector_velocity_mode_button)
        test_layout.addWidget(self.stop_and_clear_buffer_button)

        # Set layout to group
        test_group.setLayout(test_layout)

        # Add group to tab layout
        layout.addWidget(test_group)

    def setup_tab5(self, layout):
        # Basic Controls for Mini Robot
        basic_group = QGroupBox("Basic Controls")
        basic_layout = QVBoxLayout()
        self.stop_button   = QPushButton("Stop")
        self.pause_button  = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        for btn in (self.stop_button, self.pause_button, self.resume_button):
            basic_layout.addWidget(btn)
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

        # Connection UI
        conn_group  = QGroupBox("Connect Mini Robot")
        conn_layout = QHBoxLayout(conn_group)
        conn_layout.addWidget(QLabel("/dev/ttyACM"))
        self.port_input           = QLineEdit()
        self.port_input.setPlaceholderText("e.g. 3")
        self.connect_robot_button = QPushButton("Connect")
        conn_layout.addWidget(self.port_input)
        conn_layout.addWidget(self.connect_robot_button)
        layout.addWidget(conn_group)
        self.connect_robot_button.clicked.connect(self.connect_mini_robot)

        # ---- Coordinate Control Group ----
        coord_group  = QGroupBox("Coordinate Control")
        coord_layout = QVBoxLayout()
        self.coord_inputs = []
        for label_text in ["X","Y","Z","Rx","Ry","Rz"]:
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

        # ---- Angle Control Group ----
        angle_group  = QGroupBox("Angle Control")
        angle_layout = QVBoxLayout()
        self.angle_inputs = []
        for jt in ["Joint1","Joint2","Joint3","Joint4","Joint5","Joint6"]:
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

        # ---- Get Coordinates Button ----
        self.get_coords_button = QPushButton("Get Current Coordinates")
        layout.addWidget(self.get_coords_button)

        # Instantiate your mini‐robot controller
        self.tool_controller = MiniRobotToolPositionController(
            mini_robot=self.mini_robot,
            coord_inputs=self.coord_inputs,
            speed_input=self.speed_input,
            angle_inputs=self.angle_inputs,
            angle_speed_input=self.angle_speed_input,
            log_display=self.log_display
        )

        # Wire up control signals
        self.send_coords_button.clicked.connect(self.tool_controller.send_coords)
        self.send_angles_button.clicked.connect(self.tool_controller.send_angles)
        self.get_coords_button.clicked.connect(self.tool_controller.get_coords)
        self.stop_button.clicked.connect(self.tool_controller.stop)
        self.pause_button.clicked.connect(self.tool_controller.pause)
        self.resume_button.clicked.connect(self.tool_controller.resume)

        # Start with all mini‐robot controls disabled until connected
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
        """Toggle enable/disable of all mini‐robot buttons/tabs."""
        for btn in [
            self.stop_button, self.pause_button, self.resume_button,
            self.send_coords_button, self.send_angles_button,
            self.get_coords_button
        ]:
            btn.setDisabled(disable)
        # Also toggle the Mini‐Robot tab itself
        self.tab_widget.setTabEnabled(4, not disable)
    def connect_function(self):
        # Existing button connections
        self.read_sensor_raw_button.pressed.connect(
            lambda: self.log_display.append(f"Sensor raw data: {self.sensor_api.read_raw()}")
        )
        self.read_sensor_diff_button.pressed.connect(
            lambda: self.log_display.append(f"Sensor offset data: {self.sensor_functions.read_sensor_diff_data()}")
        )
        self.read_sensor_raw_2_button.pressed.connect(
            lambda: self.log_display.append(f"Sensor raw data 2: {self.sensor_functions.read_sensor_raw_data()}")
        )
        self.send_position_PTP_J_button.pressed.connect(
            self.position_entry_widget.toggle_visibility
        )
        self.send_position_PTP_T_button.pressed.connect(
            self.position_quaternion_widget.toggle_visibility
        )
        self.show_robot_button.pressed.connect(lambda: self.mesh_functions.addRobot())
        self.plotter_visibility_button.pressed.connect(self.toggle_plotter_visibility)
        self.read_raw_all_prots_button.pressed.connect(lambda: self.sensor_functions.read_raw_all_ports())
        self.render_button.pressed.connect(lambda: self.plotter.render())
        self.continuous_read_button.pressed.connect(self.toggle_continuous_read)
        self.read_timer.timeout.connect(self.update_robot_status)
        self.buildScene.pressed.connect(lambda: self.sensor_functions.buildScene())
        self.sensor_start.pressed.connect(lambda: self.sensor_functions.startSensor())
        self.sensor_update.pressed.connect(lambda: self.sensor_functions.updateCal())
        self.sensor_start_geneva.pressed.connect(lambda: self.sensor_functions.startSensor_geneva())
        self.sensor_update_geneva.pressed.connect(lambda: self.sensor_functions.updateCal_geneva())

        self.erase_mesh.pressed.connect(lambda: self.plotter.clear())

        # Connect the record gesture button
        self.record_gesture_button.pressed.connect(self.start_record_gesture)

        # Connect the test gesture button separately
        self.record_noise_auto_button.pressed.connect(
            lambda: self.sensor_functions.record_gesture_class.start_record_gesture("noise_auto")
        )
        self.set_no_trigger_button.pressed.connect(
            lambda: self.sensor_functions.record_gesture_class.set_trigger_mode("no_trigger")
        )
        self.set_trigger_button.pressed.connect(
            lambda: self.sensor_functions.record_gesture_class.set_trigger_mode("trigger")
        )
        self.predict_gesture_button.pressed.connect(
            lambda: self.sensor_functions.lstm_class.toggle_gesture_recognition()
        )

        self.update_sensor_button.pressed.connect(
            lambda: self.sensor_functions.updateCal()
        )
        self.activate_switch_model_button.pressed.connect(
            lambda: self.sensor_functions.lstm_class.toggle_model()
        )
        self.activate_rule_based_button.pressed.connect(
            lambda: self.sensor_functions.rule_based_class.activate_rule_based()
        )

    def start_record_gesture(self):
        gesture_number = self.gesture_number_input.text().strip()
        if not gesture_number:
            self.log_display.append("Please enter a gesture number or name.")
            return
        # Pass the gesture_number directly, whether it's a number or a string like "test"
        self.sensor_functions.record_gesture_class.start_record_gesture(gesture_number)

    def toggle_continuous_read(self):
        if self.read_timer.isActive():
            self.read_timer.stop()
            self.continuous_read_button.setText("Start Continuous Read")
        else:
            self.read_timer.start(0)  # Update every 0 milliseconds
            self.continuous_read_button.setText("Stop Continuous Read")

    def update_robot_status(self):
        # current_positions = self.robot_api.get_current_positions()
        # self.mesh_functions.update_robot_joints(current_positions)
        # self.plotter.render()
        print("DELETED")

    def toggle_plotter_visibility(self):
        self.log_display.setVisible(not self.log_display.isVisible())
        self.adjust_splitter_sizes()

    def adjust_splitter_sizes(self):
        total_width = self.splitter_1.width()
        if self.log_display.isVisible():
            # Allocate space for log_display
            self.splitter_1.setSizes([
                int(total_width * 0.4),  # widget_plotter
                int(total_width * 0.4),  # widget_plotter_2
                int(total_width * 0.2)   # log_display
            ])
        else:
            # Hide log_display by setting its size to zero
            self.splitter_1.setSizes([
                int(total_width * 0.5),  # widget_plotter
                int(total_width * 0.5),  # widget_plotter_2
                0                        # log_display
            ])

    def reLayout(self):
        self.setSizes([round((self.width() - 3) * 6 / 7), round((self.width() - 3) / 7)])
        self.splitter_1.setSizes([self.width() // 2, self.width() // 2])



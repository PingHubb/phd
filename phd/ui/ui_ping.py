from PyQt5 import QtCore
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QIcon, QColor, QCursor
from PyQt5.QtCore import pyqtSignal, Qt, QRect, QEvent, QTimer
from PyQt5.QtWidgets import QSplitter, QWidget, QGridLayout, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem, QTabWidget, QDialog, QLineEdit, QTextEdit, QSlider, QGroupBox, QComboBox, QScrollArea
from pyvistaqt import QtInteractor
from phd.dependence.sensor_api import ArduinoCommander
from phd.dependence.robot_api import RobotController
from phd.dependence.func_meshLab import MyMeshLab
from phd.dependence.func_sensor import MySensor, LSTM


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


class TreeWidgetItem(QTreeWidgetItem):
    def __init__(self, parent, name: str, level: int, type: int):
        super().__init__(parent)
        self.setTextAlignment(0, Qt.AlignLeft | Qt.AlignVCenter)  # 第一列左对齐
        self.iconVisible = QIcon("/home/ping2/ros2_ws/src/phd/phd/resource/icon/visible.png")
        self.iconUnvisible = QIcon("/home/ping2/ros2_ws/src/phd/phd/resource/icon/unvisible.png")
        self.iconVisibleSelected = QIcon("/home/ping2/ros2_ws/src/phd/phd/resource/icon/visible_selected.png")
        self.iconUnvisibleSelected = QIcon("/home/ping2/ros2_ws/src/phd/phd/resource/icon/unvisible_selected.png")
        self.level = level
        self._type = type
        self.visible = True
        self.setIcon(0, self.iconVisible)
        self.setText(0, name)
        if not self.level:
            if self._type == 0:
                childVertex = TreeWidgetItem(self, "Vertex", 1, 0)
                childEdges = TreeWidgetItem(self, "Edges", 1, 1)
                childFaces = TreeWidgetItem(self, "Faces", 1, 2)
                childNVertex = TreeWidgetItem(self, "N_Vertex", 1, 3)
                childNEdges = TreeWidgetItem(self, "N_Edges", 1, 4)
                childNFaces = TreeWidgetItem(self, "N_Faces", 1, 5)


class TreeWidget(QTreeWidget):
    icon_clicked = pyqtSignal(TreeWidgetItem)

    def __init__(self) -> None:
        super().__init__()
        self.setAlternatingRowColors(True)
        self.setHeaderHidden(True)
        self.itemClicked.connect(self.handle_item_clicked)

    def handle_item_clicked(self, item: TreeWidgetItem):
        if item.level:
            item.parent().setSelected(True)
            item.parent().setForeground(0, QColor(255, 0, 0))
        item_rect = self.visualItemRect(item)
        icon_rect = QRect(0, 0, 20, 20)

        if icon_rect.contains(self.viewport().mapFromGlobal(QCursor.pos()) - item_rect.topLeft()):
            if item.visible:
                item.setIcon(0, item.iconUnvisible)
                item.visible = False
                if not item.level:
                    for i in range(item.childCount()):
                        item.child(i).visible = False
                        item.child(i).setIcon(0, item.iconUnvisible)
                else:
                    any_show = False
                    for i in range(item.parent().childCount()):
                        if item.parent().child(i).visible:
                            any_show = True
                    if not any_show:
                        item.parent().visible = False
                        item.parent().setIcon(0, item.iconUnvisible)
            else:
                item.setIcon(0, item.iconVisible)
                item.visible = True
                if not item.level:
                    for i in range(item.childCount()):
                        item.child(i).visible = True
                        item.child(i).setIcon(0, item.iconVisible)
                else:
                    any_show = False
                    for i in range(item.parent().childCount()):
                        if item.parent().child(i).visible:
                            any_show = True
                        print(any_show)
                    if any_show:
                        item.parent().visible = True
                        item.parent().setIcon(0, item.iconVisible)

            self.icon_clicked.emit(item)

    def viewportEvent(self, event):
        if event.type() == QEvent.HoverMove:
            pos = event.pos()
            item = self.itemAt(pos)
            if item:
                item_rect = self.visualItemRect(item)
                icon_rect = QRect(2, 2, 16, 16)
                if icon_rect.contains(self.viewport().mapFromGlobal(QCursor.pos()) - item_rect.topLeft()):
                    if item.visible:
                        item.setIcon(0, item.iconVisibleSelected)
                    else:
                        item.setIcon(0, item.iconUnvisibleSelected)
                else:
                    if item.visible:
                        item.setIcon(0, item.iconVisible)
                    else:
                        item.setIcon(0, item.iconUnvisible)
        return super().viewportEvent(event)


class PositionEntryWidget(QWidget):
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


class PositionQuaternionWidget(QWidget):
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


class RosSplitter(QSplitter):
    def __init__(self, orientation: QtCore.Qt.Orientation):
        super().__init__(orientation)
        self.setStyleSheet("QPushButton { color: white; }")
        self.sensor_api = ArduinoCommander()
        self.robot_api = RobotController()
        self.setHandleWidth(3)
        self.setup_layout()
        self.mesh_functions = MyMeshLab(self)
        self.sensor_functions = MySensor(self)
        self.read_timer = QTimer(self)  # Timer for continuous reading
        self.connect_function()

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
        # self.splitter_1.addWidget(self.log_display)
        self.splitter_1.addWidget(self.widget_plotter_2)
        self.splitter_1.setHandleWidth(3)

        # SET UP SPLITTER 2
        self.splitter_2 = QSplitter(Qt.Vertical, self)
        self.splitter_2.setStyleSheet("background-color: #303030;")
        self.splitter_2.setHandleWidth(3)

        # SET UP WIDGET 4 & 5
        self.position_entry_widget = PositionEntryWidget()
        self.position_quaternion_widget = PositionQuaternionWidget()
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
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.setup_tab1(tab1_layout)
        self.tab_widget.addTab(tab1, "Sensor")

        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        self.setup_tab2(tab2_layout)
        self.tab_widget.addTab(tab2, "Robot")

        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        self.setup_tab3(tab3_layout)
        self.tab_widget.addTab(tab3, "Testing")

        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)
        self.setup_tab4(tab4_layout)
        self.tab_widget.addTab(tab4, "AI")

    def setup_tab1(self, layout):
        read_group = QGroupBox("Read Operations")
        send_group = QGroupBox("Send Operations")
        read_layout = QVBoxLayout()
        send_layout = QVBoxLayout()

        self.read_sensor_raw_button = QPushButton("Read Sensor Raw Data")
        self.read_sensor_diff_button = QPushButton("Read Sensor Diff Data")
        self.read_sensor_raw_2_button = QPushButton("Read Sensor Raw Data 2")

        read_layout.addWidget(self.read_sensor_raw_button)
        read_layout.addWidget(self.read_sensor_diff_button)
        read_layout.addWidget(self.read_sensor_raw_2_button)

        read_group.setLayout(read_layout)
        send_group.setLayout(send_layout)

        layout.addWidget(read_group)
        layout.addWidget(send_group)

        # NEW SENSOR WIDGET
        self.sensor_choice = QComboBox(self.widget_func)
        self.sensor_choice.addItem("elbow")
        self.sensor_choice.addItem("DualC")
        self.sensor_choice.addItem("2D")
        self.sensor_choice.addItem("Cylinder (FAKE)")

        send_layout.addWidget(self.sensor_choice)

        self.serial_channel = QComboBox(self.widget_func)
        send_layout.addWidget(self.serial_channel)

        self.buildScene = QPushButton("buildScene", self.widget_func)
        send_layout.addWidget(self.buildScene)

        self.sensor_start = QPushButton("startSensor", self.widget_func)
        send_layout.addWidget(self.sensor_start)
        self.sensor_start.setDisabled(True)

        self.sensor_update = QPushButton("updateSensor", self.widget_func)
        send_layout.addWidget(self.sensor_update)
        self.sensor_update.setDisabled(True)

    def setup_tab2(self, layout):
        read_group = QGroupBox("Read Operations")
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


        read_group.setLayout(read_layout)
        send_group.setLayout(send_layout)

        layout.addWidget(read_group)
        layout.addWidget(send_group)

    def setup_tab3(self, layout):
        # Create a group box for testing buttons
        test_group = QGroupBox("Testing Operations")
        test_layout = QVBoxLayout()

        # Button for testing
        self.plotter_visibility_button = QPushButton("Show/Hide Log Display")
        self.render_button = QPushButton("Render")
        self.enable_joint_velocity_mode_button = QPushButton("Enable Joint Velocity Mode")
        self.suspend_end_effector_velocity_mode_button = QPushButton("Suspend End Effector Velocity")
        self.stop_joint_velocity_mode_button = QPushButton("Stop Joint Velocity")
        self.enable_end_effector_velocity_mode_button = QPushButton("Enable End Effector Velocity")
        self.stop_end_effector_velocity_mode_button = QPushButton("Stop End Effector Velocity")
        self.stop_and_clear_buffer_button = QPushButton("Stop and Clear Buffer")

        # Add buttons to layout
        test_layout.addWidget(self.plotter_visibility_button)
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

    def setup_tab4(self, layout):
        # Create group boxes
        training_group = QGroupBox("Gesture")
        testing_group = QGroupBox("Testing Operations")

        # Create layouts for the groups
        gesture_layout = QHBoxLayout()
        testing_layout = QVBoxLayout()

        # Create a label, input field, and record button
        gesture_label = QLabel("Enter Gesture Number:")
        self.gesture_number_input = QLineEdit()
        self.gesture_number_input.setFixedSize(50, 30)
        self.record_gesture_button = QPushButton("Record")
        self.record_gesture_button.setFixedSize(100, 40)

        # Add widgets to the gesture layout
        gesture_layout.addWidget(gesture_label)
        gesture_layout.addWidget(self.gesture_number_input)
        gesture_layout.addWidget(self.record_gesture_button)

        # Create the test gesture button
        self.record_gesture_test_button = QPushButton("Record Test Gesture")
        self.record_gesture_test_button.setFixedSize(150, 40)
        gesture_layout.addWidget(self.record_gesture_test_button)

        # Create additional testing operation buttons
        self.predict_gesture_button = QPushButton("Predict Gesture")
        self.update_sensor_button = QPushButton("Update Sensor")
        self.activate_rule_based_button = QPushButton("Activate Rule Based")

        # Set fixed sizes for consistency
        for btn in [
            self.predict_gesture_button,
            self.update_sensor_button,
            self.activate_rule_based_button,
        ]:
            btn.setFixedSize(150, 40)

        # Add testing operation buttons to the testing layout
        testing_layout.addWidget(self.predict_gesture_button)
        testing_layout.addWidget(self.update_sensor_button)
        testing_layout.addWidget(self.activate_rule_based_button)

        # Assign layouts to their respective group boxes
        training_group.setLayout(gesture_layout)
        testing_group.setLayout(testing_layout)

        # Add group boxes to the main layout
        layout.addWidget(training_group)
        layout.addWidget(testing_group)

    def connect_function(self):
        # Existing button connections
        self.read_sensor_raw_button.pressed.connect(
            lambda: self.log_display.append(f"Sensor raw data: {self.sensor_api.read_raw()}")
        )
        self.read_sensor_diff_button.pressed.connect(
            lambda: self.log_display.append(f"Sensor diff data: {self.sensor_functions.read_sensor_diff_data()}")
        )
        self.read_sensor_raw_2_button.pressed.connect(
            lambda: self.log_display.append(f"Sensor raw data 2: {self.sensor_functions.read_sensor_raw_data()}")
        )
        self.enable_joint_velocity_mode_button.pressed.connect(
            lambda: self.robot_api.send_request(self.robot_api.enable_joint_velocity_mode())
        )
        self.stop_joint_velocity_mode_button.pressed.connect(
            lambda: self.robot_api.send_request(self.robot_api.stop_joint_velocity_mode())
        )
        self.enable_end_effector_velocity_mode_button.pressed.connect(
            lambda: self.robot_api.send_request(self.robot_api.enable_end_effector_velocity_mode())
        )
        self.suspend_end_effector_velocity_mode_button.pressed.connect(
            lambda: self.robot_api.send_request(self.robot_api.suspend_end_effector_velocity_mode())
        )
        self.stop_end_effector_velocity_mode_button.pressed.connect(
            lambda: self.robot_api.send_request(self.robot_api.stop_end_effector_velocity_mode())
        )
        self.stop_and_clear_buffer_button.pressed.connect(
            lambda: self.robot_api.send_request(self.robot_api.stop_and_clear_buffer())
        )
        self.send_script_button.pressed.connect(
            lambda: self.robot_api.send_and_process_request([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        self.read_joint_angle_button.pressed.connect(
            lambda: self.log_display.append(f"Joint Angles: {self.robot_api.get_current_positions()}")
        )
        self.read_tool_position_button.pressed.connect(
            lambda: self.log_display.append(
                f"Tool Position: {self.robot_api.get_current_tool_position()[0]}, "
                f"Orientation: {self.robot_api.get_current_tool_position()[1]}"
            )
        )
        self.send_position_PTP_J_button.pressed.connect(
            self.position_entry_widget.toggle_visibility
        )
        self.send_position_PTP_T_button.pressed.connect(
            self.position_quaternion_widget.toggle_visibility
        )
        self.show_robot_button.pressed.connect(lambda: self.mesh_functions.addRobot())
        self.plotter_visibility_button.pressed.connect(self.toggle_plotter_visibility)
        self.render_button.pressed.connect(lambda: self.plotter.render())
        self.continuous_read_button.pressed.connect(self.toggle_continuous_read)
        self.read_timer.timeout.connect(self.update_robot_status)
        self.buildScene.pressed.connect(lambda: self.sensor_functions.buildScene())
        self.sensor_start.pressed.connect(lambda: self.sensor_functions.startSensor())
        self.sensor_update.pressed.connect(lambda: self.sensor_functions.updateCal())

        # Connect the record gesture button
        self.record_gesture_button.pressed.connect(self.start_record_gesture)

        # Connect the test gesture button separately
        self.record_gesture_test_button.pressed.connect(
            lambda: self.sensor_functions.start_record_gesture("test")
        )

        self.predict_gesture_button.pressed.connect(
            lambda: self.sensor_functions.lstm_class.toggle_gesture_recognition()
        )
        self.update_sensor_button.pressed.connect(
            lambda: self.sensor_functions.updateCal()
        )
        self.activate_rule_based_button.pressed.connect(
            lambda: self.sensor_functions.rule_based_class.activate_rule_based()
        )

    def start_record_gesture(self):
        gesture_number_str = self.gesture_number_input.text()
        try:
            gesture_number = int(gesture_number_str)
            self.sensor_functions.record_gesture_class.start_record_gesture(gesture_number)
        except ValueError:
            # Handle invalid input
            self.log_display.append("Invalid gesture number. Please enter an integer.")

    def toggle_continuous_read(self):
        if self.read_timer.isActive():
            self.read_timer.stop()
            self.continuous_read_button.setText("Start Continuous Read")
        else:
            self.read_timer.start(0)  # Update every 0 milliseconds
            self.continuous_read_button.setText("Stop Continuous Read")

    def update_robot_status(self):
        current_positions = self.robot_api.get_current_positions()
        self.mesh_functions.update_robot_joints(current_positions)
        self.plotter.render()

    def toggle_plotter_visibility(self):
        self.log_display.setVisible(not self.log_display.isVisible())
        self.adjust_splitter_sizes()

    def adjust_splitter_sizes(self):
        if self.log_display.isVisible():
            self.splitter_1.setSizes([self.width() // 2, self.width() // 2])
        else:
            self.splitter_1.setSizes([self.width(), 0])

    def reLayout(self):
        self.setSizes([round((self.width() - 3) * 6 / 7), round((self.width() - 3) / 7)])
        self.splitter_1.setSizes([self.width() // 2, self.width() // 2])



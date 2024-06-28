class RosSplitter(QSplitter):
    def __init__(self, orientation: QtCore.Qt.Orientation):
        super().__init__(orientation)
        self.sensor_api = ArduinoCommander()
        self.robot_api = RobotController()
        self.setHandleWidth(3)
        self.setup_layout()
        self.connect_function()

    def setup_layout(self):
        # 创建Plotter部分
        self.widget_plotter = PlotterWidget()
        self.widget_plotter.setObjectName("widgetPlotter")
        self.widget_plotter_2 = PlotterWidget()
        self.widget_plotter_2.setObjectName("widgetPlotter 2")

        # Grid Layout Setup
        self.layout_plotter = QGridLayout(self.widget_plotter)
        self.layout_plotter.setContentsMargins(0, 0, 0, 0)
        self.layout_plotter_2 = QGridLayout(self.widget_plotter_2)
        self.layout_plotter_2.setContentsMargins(0, 0, 0, 0)

        # Plotter Interaction:
        self.plotter = QtInteractor(self)
        self.plotter.background_color = '#303030'
        self.plotter_2 = QtInteractor(self)
        self.plotter_2.background_color = '#303030'

        self.widget_plotter.setAcceptDrops(True)
        self.layout_plotter.addWidget(self.plotter.interactor)
        self.widget_plotter.setLayout(self.layout_plotter)
        self.widget_plotter_2.setAcceptDrops(True)
        self.layout_plotter_2.addWidget(self.plotter_2.interactor)
        self.widget_plotter_2.setLayout(self.layout_plotter_2)

        # Splitter for Left and Right
        self.splitter_1 = QSplitter(Qt.Horizontal, self)
        self.splitter_1.setObjectName("splitter1")
        self.splitter_1.setHandleWidth(5)

        # 创建上下分割器2(上方树，下方功能)
        self.splitter_2 = QSplitter(Qt.Vertical, self)
        self.splitter_2.setObjectName("splitter2")
        self.splitter_2.setHandleWidth(5)

        # 创建Tree部分
        self.widget_tree = TreeWidget()
        self.widget_tree.setObjectName("widgetTree")

        # 创建Function部分
        self.widget_func = QWidget()
        self.widget_func.setObjectName("widgetFunc")
        self.layout_func = QVBoxLayout()
        self.widget_func.setLayout(self.layout_func)

        # 添加组件到分割器1
        self.splitter_1.addWidget(self.widget_plotter)
        self.splitter_1.addWidget(self.widget_plotter_2)

        # 添加组件到分割器2
        self.splitter_2.addWidget(self.widget_tree)
        self.splitter_2.addWidget(self.widget_func)

        # 添加组件到分割器1
        self.addWidget(self.splitter_1)
        self.addWidget(self.splitter_2)

        self.read_sensor_button = QPushButton("Read Sensor", self.widget_func)
        self.layout_func.addWidget(self.read_sensor_button)

        self.robot_position_button = QPushButton("Read Robot Position", self.widget_func)
        self.layout_func.addWidget(self.robot_position_button)

        self.testing_button = QPushButton("Testing", self.widget_func)
        self.layout_func.addWidget(self.testing_button)

    def reLayout(self):
        self.setSizes([round((self.width() - 3) * 6 / 7), round((self.width() - 3) / 7)])
        self.splitter_2.setSizes([round((self.height() - 3) / 2), self.height() - 3 - round((self.height() - 3) / 2)])

    def connect_function(self):
        self.read_sensor_button.pressed.connect(self.sensor_api.read_raw)
        self.robot_position_button.pressed.connect(self.robot_api.get_current_positions)
        self.testing_button.pressed.connect(lambda: print("Testing"))
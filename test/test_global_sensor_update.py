from phd.ui.ui_initial import MyMainWindow


class _Status:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = str(text)


class _HumanoidViewer:
    def __init__(self):
        self.update_calls = 0

    def request_sensor_update(self):
        self.update_calls += 1
        return True


class _Ui:
    def __init__(self):
        self.humanoid_viewer = _HumanoidViewer()
        self.plotter_update_calls = 0

    def _on_sensor_update(self):
        self.plotter_update_calls += 1


class _Harness:
    def __init__(self):
        self.ui_ros = _Ui()
        self.info_process = _Status()

    @staticmethod
    def _require_ui_ros(_message):
        return True


def test_global_update_calibrates_plotter_and_humanoid_sensors():
    harness = _Harness()

    MyMainWindow._trigger_global_sensor_update(harness)

    assert harness.ui_ros.humanoid_viewer.update_calls == 1
    assert harness.ui_ros.plotter_update_calls == 1
    assert "Sensor Plotter and humanoid" in harness.info_process.text

import numpy as np

from phd.ui.plotter_video_recorder import PlotterVideoRecorder


class _Writer:
    def __init__(self):
        self.frames = []
        self.released = False

    @staticmethod
    def isOpened():
        return True

    def write(self, frame):
        self.frames.append(np.array(frame, copy=True))

    def release(self):
        self.released = True


class _Cv2:
    COLOR_RGB2BGR = 1
    INTER_AREA = 2

    def __init__(self):
        self.writer = _Writer()
        self.requested_size = None

    @staticmethod
    def VideoWriter_fourcc(*_args):
        return 123

    def VideoWriter(self, _path, _fourcc, _fps, size):
        self.requested_size = tuple(size)
        return self.writer

    @staticmethod
    def cvtColor(frame, _conversion):
        return frame[..., ::-1]

    @staticmethod
    def resize(frame, size, interpolation=None):
        del interpolation
        width, height = size
        return np.resize(frame, (height, width, 3))


class _Plotter:
    def __init__(self):
        self.frame = np.zeros((5, 7, 3), dtype=np.uint8)
        self.frame[..., 0] = 10
        self.frame[..., 2] = 30

    def screenshot(self, return_img=False):
        assert return_img
        return np.array(self.frame, copy=True)


def test_plotter_video_recorder_captures_and_finalizes_frames(tmp_path):
    cv2 = _Cv2()
    recorder = PlotterVideoRecorder(fps=20.0, cv2_module=cv2)

    assert recorder.start(_Plotter(), tmp_path / "recording.mp4")
    assert cv2.requested_size == (6, 4)
    assert recorder.frame_count == 1
    assert recorder.capture_frame()

    output_path = recorder.stop(capture_final=True)

    assert output_path == tmp_path / "recording.mp4"
    assert recorder.frame_count == 3
    assert cv2.writer.released
    assert cv2.writer.frames[0].shape == (4, 6, 3)
    np.testing.assert_array_equal(cv2.writer.frames[0][0, 0], [30, 0, 10])


def test_plotter_video_recorder_fills_missed_frames_from_elapsed_time(tmp_path):
    cv2 = _Cv2()
    recorder = PlotterVideoRecorder(fps=60.0, cv2_module=cv2)

    assert recorder.start(_Plotter(), tmp_path / "timed.mp4")
    assert recorder.frames_due_for_elapsed(0.01) == 0

    frames_due = recorder.frames_due_for_elapsed(2.0)
    assert frames_due == 119
    assert recorder.capture_frame(repeat_count=frames_due)
    assert recorder.frame_count == 120

    recorder.stop(capture_final=False)
    assert len(cv2.writer.frames) == 120
    assert recorder.frame_count / recorder.fps == 2.0

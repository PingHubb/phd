import os
import random
import re

import numpy as np
from PyQt5.QtCore import QTimer

from phd.dependence.sensor_layout import flatten_column_major_view


class RecordGesture:
    RESOURCE_ROOT = "/home/ping2/ros2_ws/src/phd/phd/resource"
    OFFSET_DATA_DIR = os.path.join(RESOURCE_ROOT, "ai", "data", "offset")
    TRIGGER_MODES = ("trigger", "no_trigger", "no_trigger_auto", "no_trigger_no_updatecal_auto")
    AUTO_TRIGGER_MODES = ("no_trigger_auto", "no_trigger_no_updatecal_auto")

    def __init__(self, my_sensor_instance):
        self.my_sensor = my_sensor_instance

        self.timer_record_gesture = QTimer()
        self.timer_record_gesture.timeout.connect(self.record_gesture)

        self.auto_timer = QTimer()
        self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self.auto_record_timeout)

        self.is_recording = False
        self.start_record_pressed = False
        self.current_gesture_data_diff = []
        self.trial_number = None
        self.gesture_number = None
        self.auto_recording = False

        self.trigger_mode = None
        self.max_frames_manual_trigger = 70
        self.auto_chunk_size = 30

    def _reset_recorded_frames(self):
        self.current_gesture_data_diff = []

    def _gesture_diff_dir(self, gesture_number):
        return os.path.join(self.OFFSET_DATA_DIR, f"gesture_{gesture_number}")

    def set_trigger_mode(self, mode):
        if mode in self.TRIGGER_MODES:
            self.trigger_mode = mode
            print(f"Trigger mode set to '{self.trigger_mode}'.")
        else:
            print("Invalid trigger mode specified.")

    def start_record_gesture(self, gesture_number):
        if self.trigger_mode is None:
            print("Warning: Trigger mode not set. Please set the trigger mode before recording.")
            return

        if gesture_number == "noise_auto":
            self._handle_noise_auto_toggle()
            return

        if not self.start_record_pressed:
            self.start_record_pressed = True
            self.is_recording = True
            self.gesture_number = self.sanitize_gesture_number(gesture_number)
            self.trial_number = self.get_next_trial_number(self.gesture_number)
            self._reset_recorded_frames()

            self.timer_record_gesture.start(0)

            print(f"Recording ARMED for gesture '{self.gesture_number}'.")
            if self.trigger_mode == "trigger":
                print("Waiting for trigger (touch sensor)...")
            elif self.trigger_mode in self.AUTO_TRIGGER_MODES:
                print(f"Automatic chunk recording started. Saving every {self.auto_chunk_size} frames.")
            else:
                print("Manual continuous recording started.")
        else:
            self.start_record_pressed = False
            self.is_recording = False
            self.timer_record_gesture.stop()

            if self.current_gesture_data_diff:
                self._save_and_reset(reason="Manually stopped")

            print("Recording session stopped by user.")

    def _handle_noise_auto_toggle(self):
        if not self.auto_recording:
            self.auto_recording = True
            self.is_recording = True
            self.gesture_number = "noise"
            self.trial_number = self.get_next_trial_number(self.gesture_number)
            self._reset_recorded_frames()
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
        self.random_duration = random.randint(5, 10)
        print(f"Next noise recording will be for {self.random_duration} seconds.")
        self._reset_recorded_frames()
        self.timer_record_gesture.start(0)
        self.auto_timer.start(self.random_duration * 1000)

    def auto_record_timeout(self):
        if self.is_recording:
            self._save_and_reset(reason=f"Automatic duration ({self.random_duration}s) elapsed")
        if self.auto_recording:
            self.start_auto_recording()

    def _save_and_reset(self, reason=""):
        self.save_gesture_data()

        if self.trigger_mode != "no_trigger_no_updatecal_auto":
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
        return re.sub(r"[^a-zA-Z0-9_]", "_", gesture_number)

    def record_gesture(self):
        if not self.is_recording:
            self.timer_record_gesture.stop()
            return

        column_major_flat_frame = flatten_column_major_view(self.my_sensor._data.diffPerDataAve)

        if self.trigger_mode in self.AUTO_TRIGGER_MODES:
            self.current_gesture_data_diff.append(column_major_flat_frame)
            if len(self.current_gesture_data_diff) >= self.auto_chunk_size:
                self._save_and_reset(reason=f"Auto-saved chunk of {self.auto_chunk_size} frames")

        elif self.trigger_mode == "no_trigger":
            self.current_gesture_data_diff.append(column_major_flat_frame)

        elif self.trigger_mode == "trigger":
            transformed_data = self._transform_data(column_major_flat_frame)
            is_triggered = np.any(transformed_data == 1)

            if not hasattr(self, "is_currently_triggered"):
                self.is_currently_triggered = False

            if is_triggered and not self.is_currently_triggered:
                self.is_currently_triggered = True
                self._reset_recorded_frames()
                print(f"Trigger detected! Started recording Trial {self.trial_number}.")

            elif not is_triggered and self.is_currently_triggered:
                self.is_currently_triggered = False
                if len(self.current_gesture_data_diff) > 1:
                    self._save_and_reset(reason="Trigger signal lost")

            if self.is_currently_triggered:
                self.current_gesture_data_diff.append(column_major_flat_frame)
                if len(self.current_gesture_data_diff) > self.max_frames_manual_trigger:
                    self._save_and_reset(reason=f"Frame limit ({self.max_frames_manual_trigger}) exceeded")
                    self.is_currently_triggered = False

    def get_next_trial_number(self, gesture_number):
        gesture_diff_dir = self._gesture_diff_dir(gesture_number)
        if not os.path.exists(gesture_diff_dir):
            os.makedirs(gesture_diff_dir)
        files = os.listdir(gesture_diff_dir)
        return len(files) + 1

    def save_gesture_data(self):
        gesture_diff_dir = self._gesture_diff_dir(self.gesture_number)
        filename = os.path.join(gesture_diff_dir, f"{self.trial_number}.txt")

        if not self.current_gesture_data_diff:
            print(f"Warning: No data recorded for trial {self.trial_number}. File not saved.")
            return

        with open(filename, "w") as file:
            for data_entry in self.current_gesture_data_diff:
                file.write(" ".join(map(str, data_entry)) + "\n")

        self._reset_recorded_frames()

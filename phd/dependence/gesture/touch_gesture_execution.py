"""Live CNN-GRU/TCN adapter for the existing Touch Policy velocity executor.

Acquisition only queues complete sensor snapshots. Inference uses the existing
worker; validation and zero-velocity decisions remain on the control timer.
"""
import queue
import time

import numpy as np


class TouchGestureExecutionMixin:
    def _load_touch_gesture_checkpoint(self, checkpoint):
        from phd.dependence.touch_gesture_policy import TouchGesturePolicy

        self.model = TouchGesturePolicy(checkpoint).to(self.device)
        # Historical dispatch key shared by all folder-labelled touch policies;
        # UI architecture/labels come from the validated bundle, not this key.
        self.model_kind = "cnn_tcn_gesture"
        self._touch_gesture_bundle = checkpoint
        self.touch_gesture_feedback = None
        self.seq_len = checkpoint["preprocess"]["sequence_length"]
        self.input_channels = list(checkpoint["preprocess"]["channels"])
        self.aux_feature_names = []
        self.use_aux_features = False
        self.lock_rotation_axes = True
        self.max_angular_speed = 0.0
        # The speed control is a scale of 50 mm/s. Do not clamp it again here;
        # _clip_velocity6 still rejects non-finite values through this bound.
        self.max_linear_speed = 1.0e6
        self.model_conf_threshold = float(checkpoint["control"]["confidence_threshold"])
        self.min_motion_command_speed = 0.0001
        self.move_confirmation_frames = max(2, int(checkpoint["control"]["confirmation_predictions"]))

    @staticmethod
    def _touch_sensor_binding(sensor, data):
        return (id(sensor), id(data), getattr(sensor, "_primary_sensor_port", None),
                getattr(sensor, "current_effective_reorder_logic", None),
                getattr(sensor, "current_reorder_mode", None), getattr(data, "windowSize", None))

    def _touch_live_data(self):
        sensor = self.my_sensor
        if not getattr(sensor, "is_connected", False):
            raise ValueError("Start the 10x10 sensor stream before Touch Policy")
        if getattr(sensor, "_sensor_calibration_in_progress", False):
            raise ValueError("Sensor calibration is in progress; restart Touch Policy afterwards")
        data = getattr(sensor, "_data", None)
        stamp = getattr(data, "frame_monotonic_sec", None)
        if stamp is None or not 0 <= time.monotonic() - stamp <= 0.2:
            raise ValueError("Sensor frames are stale; Touch Policy stopped")
        bundle = self._touch_gesture_bundle
        if tuple(np.shape(data.diffPerData)) != tuple(bundle["sensor_shape"]):
            raise ValueError("This Touch Policy requires the trained 10x10 sensor")
        if getattr(sensor, "current_reorder_mode", None) != bundle["sensor_reorder_mode"]:
            raise ValueError("Touch Policy requires the training sensor layout: row_to_col")
        if getattr(data, "windowSize", None) != bundle["sensor_average_window_size"]:
            raise ValueError("Touch Policy requires sensor averaging of 3 frames")
        return data

    def _start_touch_gesture_stream(self):
        from phd.dependence.touch_gesture_policy import TouchGestureFeedback, TouchPrefixBuffer

        self._detach_touch_gesture_stream()
        data = self._touch_live_data()
        if not callable(getattr(data, "add_frame_listener", None)):
            raise ValueError("Sensor source does not support complete-frame capture")
        self._touch_stream_data = data
        self._touch_binding = self._touch_sensor_binding(self.my_sensor, data)
        self._touch_control_frame = self._get_requested_frame()
        self._touch_calibration = np.array(data.calData, copy=True)
        self._touch_queue = queue.Queue(maxsize=64)
        self._touch_stream_error = None
        self._touch_prefix = TouchPrefixBuffer(self._touch_gesture_bundle["sensor_shape"], self._touch_gesture_bundle["preprocess"])
        self.touch_gesture_feedback = TouchGestureFeedback()
        self._touch_last_submitted_sequence = None
        self._touch_pending_started = None
        self._touch_latched_gesture = None
        self._touch_latched_confidence = 0.0
        data.add_frame_listener(self._on_touch_gesture_frame)
        self._touch_stop_prediction("Lift finger briefly to arm")
        self._arm_touch_if_hand_is_away()

    def _arm_touch_if_hand_is_away(self):
        """A hand already away at Execute is enough evidence to accept the next touch.

        The buffer otherwise waits for another 0.12–0.20 s of non-contact, so the
        first swipe after every Execute is discarded and only the next one runs.
        A finger that is down at Execute still has to lift before motion can start.
        """
        frame = getattr(self._touch_stream_data, "completed_frame_snapshot", None)
        prefix = self._touch_prefix
        if frame is None or prefix is None:
            return False
        averaged = np.asarray(getattr(frame, "diffPerDataAve", None), dtype=np.float32)
        threshold = float(prefix.config["touch_threshold_percent"])
        if averaged.shape != prefix.shape or not np.isfinite(averaged).all() or np.any(averaged < threshold):
            return False
        prefix.state = "ready"
        prefix.touching = False
        prefix.frames = []
        prefix.last_sequence = int(frame.frame_sequence)
        prefix.last_time = float(frame.frame_monotonic_sec)
        if self.last_prediction is not None:
            self.last_prediction["reason"] = "Hand is away. Swipe to start."
        return True

    def _warmup_touch_gesture_inference(self):
        """Run one forward pass before the first live gesture.

        The first CUDA forward is close to the 0.2 s prediction deadline. Crossing
        it used to stop the whole policy, so the first Execute never moved and
        the second one did, after the kernels were already warm.
        """
        torch = self._torch
        shape = tuple(int(v) for v in self._touch_gesture_bundle["sensor_shape"])
        dummy = torch.zeros((1, int(self.seq_len), len(self.input_channels), *shape), device=self.device)

        def _once():
            with torch.inference_mode():
                self.model(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        self._ensure_inference_executor().submit(_once).result(timeout=5)

    def _detach_touch_gesture_stream(self):
        feedback = getattr(self, "touch_gesture_feedback", None)
        if feedback is not None:
            feedback.abort()
        data = getattr(self, "_touch_stream_data", None)
        self._touch_stream_data = None
        if data is not None:
            data.remove_frame_listener(self._on_touch_gesture_frame)

    def _on_touch_gesture_frame(self, data):
        if data is not getattr(self, "_touch_stream_data", None):
            return
        try:
            frame = getattr(data, "completed_frame_snapshot", None)
            if frame is None:
                raise ValueError("No complete sensor snapshot")
            self._touch_queue.put_nowait((int(frame.frame_sequence), float(frame.frame_monotonic_sec),
                                         frame.diffPerData, frame.diffPerDataAve))
        except Exception as exc:
            self._touch_stream_error = "Sensor frame queue overflow" if isinstance(exc, queue.Full) else str(exc)

    def _touch_stop_prediction(self, reason):
        self._touch_latched_gesture = None
        self._touch_latched_confidence = 0.0
        self._apply_prediction({"mode": "stop", "mode_conf": 1.0, "finger_idx": 0,
                                "velocity_sent": np.zeros(6, dtype=np.float32),
                                "gesture": "stop", "reason": str(reason)})

    def _reuse_latched_touch(self, reason):
        """Keep the last confirmed direction while this contact continues."""
        gesture = getattr(self, "_touch_latched_gesture", None)
        prefix = getattr(self, "_touch_prefix", None)
        if not gesture or prefix is None or not prefix.continuous or not prefix.touching:
            return False
        from phd.dependence.touch_gesture_policy import TOUCH_POLICY_UNIT_SPEED_MPS, gesture_velocity, gesture_control_mode

        self._apply_prediction({
            "mode": gesture_control_mode(gesture), "mode_conf": float(self._touch_latched_confidence), "finger_idx": 0,
            "velocity_sent": gesture_velocity(gesture, TOUCH_POLICY_UNIT_SPEED_MPS, self.velocity_scale),
            "gesture": gesture, "reason": str(reason),
        })
        return self.last_prediction.get("mode") != "stop"

    def _touch_stream_fault(self, reason):
        self._discard_pending_predictions()
        self._detach_touch_gesture_stream()
        self._touch_stop_prediction(reason)
        if not self.dry_run_predictions_only:
            try:
                self._stop_robot_motion_execution(stop_mode=True)
            except Exception as exc:
                print(f"[Touch Policy] Robot stop could not be submitted: {exc}")
        self.is_running = False
        self.control_timer.stop()
        print(f"[Touch Policy] Stopped: {reason}")

    def _touch_robot_command_failed(self):
        # Do not claim "running" or silently restart motion if the service
        # later reappears. The operator must explicitly restart the policy.
        self.is_running = False
        self.control_timer.stop()
        self._discard_pending_predictions()
        self._detach_touch_gesture_stream()
        try:
            self._stop_robot_motion_execution(stop_mode=True)
        except Exception as exc:
            print(f"[Touch Policy] Robot stop could not be submitted: {exc}")
        self.last_prediction = {
            "mode": "stop", "gesture": "stop", "mode_conf": 1.0, "finger_idx": 0,
            "velocity_sent": [0.0] * 6, "dry_run": False,
            "reason": "Robot command could not be submitted. Check robot connection and restart Touch Policy.",
        }

    def _run_touch_gesture_step(self):
        try:
            data = self._touch_live_data()
            if (self._touch_sensor_binding(self.my_sensor, data) != self._touch_binding
                    or not np.array_equal(data.calData, self._touch_calibration)
                    or self._get_requested_frame() != self._touch_control_frame):
                raise ValueError("Sensor, layout, calibration or control frame changed; restart Touch Policy")
            if self._touch_stream_error:
                raise ValueError(self._touch_stream_error)
            # Consume release/reset evidence BEFORE accepting background results.
            while True:
                try:
                    frame = self._touch_queue.get_nowait()
                except queue.Empty:
                    break
                old_episode = self._touch_prefix.episode
                old_state = self._touch_prefix.state
                active = self._touch_prefix.feed(*frame)
                # Observe segmentation without changing command timing or
                # doing another inference after release.
                feedback = self.touch_gesture_feedback
                if self._touch_prefix.episode != old_episode:
                    feedback.begin(self._touch_prefix.episode, self._touch_prefix.continuous)
                elif old_state == "recording" and self._touch_prefix.state == "ready":
                    feedback.finish()
                elif old_state == "recording" and self._touch_prefix.state == "release":
                    feedback.abort()
                if not active or self._touch_prefix.episode != old_episode:
                    self._discard_pending_predictions()
                    self._touch_pending_started = None
                    self._touch_stop_prediction("Lift finger to arm" if self._touch_prefix.state == "release" else "Released / waiting for touch")
            prefix = self._touch_prefix
            if not prefix.touching or prefix.state != "recording":
                return
            if prefix.continuous and not prefix.motion_present():
                self._discard_pending_predictions()
                self._touch_pending_started = None
                if not self._reuse_latched_touch("Holding last gesture"):
                    self._touch_stop_prediction("Waiting for signal movement / finger paused")
                return
            if prefix.last_time is None or time.monotonic() - prefix.last_time > 0.2:
                raise ValueError("No fresh captured sensor frames")
            if (self._pending_inference is not None and self._touch_pending_started is not None
                    and time.monotonic() - self._touch_pending_started > 0.2):
                raise ValueError("Gesture inference stalled; motion stopped")
            if self.inference_in_background:
                self._collect_pending_prediction()
                if not self.is_running:
                    return
            if self._pending_inference is not None or prefix.last_sequence == self._touch_last_submitted_sequence:
                return
            prepared = prefix.snapshot()
            if prepared is None:
                if not self._reuse_latched_touch("Holding last gesture"):
                    self._touch_stop_prediction("Collecting touch frames")
                return
            self._touch_last_submitted_sequence = prefix.last_sequence
            if self.inference_in_background:
                future = self._ensure_inference_executor().submit(self._run_touch_gesture_inference, prepared, prefix.last_time)
                self._pending_inference = (self._inference_epoch, future)
                self._touch_pending_started = time.monotonic()
            else:
                self._apply_prediction(self._run_touch_gesture_inference(prepared, prefix.last_time))
        except Exception as exc:
            self._touch_stream_fault(str(exc))

    def _run_touch_gesture_inference(self, prefix, source_time):
        from phd.dependence.touch_gesture_policy import (
            CLASS_NAMES, TOUCH_POLICY_UNIT_SPEED_MPS, gesture_velocity, gesture_control_mode,
        )

        torch = self._torch
        class_names = self._touch_gesture_bundle.get("class_names", CLASS_NAMES)
        with torch.inference_mode():
            probabilities = self.model(torch.from_numpy(prefix[None]).to(self.device))[0].cpu().numpy()
        if probabilities.shape != (len(class_names),) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid gesture probabilities")
        order = np.argsort(probabilities)
        gesture = class_names[int(order[-1])]
        confidence = float(probabilities[order[-1]])
        margin = float(confidence - probabilities[order[-2]])
        control = self._touch_gesture_bundle["control"]
        accepted = confidence >= self.model_conf_threshold and margin >= control["margin_threshold"]
        velocity = gesture_velocity(gesture, TOUCH_POLICY_UNIT_SPEED_MPS, self.velocity_scale) if accepted else np.zeros(6, dtype=np.float32)
        return {"mode": gesture_control_mode(gesture) if accepted else "stop",
                "mode_conf": confidence, "finger_idx": 0, "velocity_sent": velocity,
                "gesture": gesture if accepted else "uncertain", "top_gesture": gesture,
                "gesture_margin": margin, "gesture_probabilities": dict(zip(class_names, probabilities.tolist())),
                "source_monotonic_sec": source_time, "reason": "" if accepted else "Uncertain gesture"}

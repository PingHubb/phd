from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QLineEdit, QShortcut, QTextEdit


class AiControlsMixin:
    def _set_ai_frame(self, frame: str):
        self.ai_selected_frame = frame
        if hasattr(self, "ai_frame_input") and self.ai_frame_input is not None:
            self.ai_frame_input.setText(frame)
        self._update_ai_frame_buttons()

    def _update_ai_frame_buttons(self):
        selected = getattr(self, "ai_selected_frame", "tool")
        for frame_key, btn in getattr(self, "ai_frame_buttons", {}).items():
            is_selected = frame_key == selected
            btn.setChecked(is_selected)
            self._set_button_active(btn, is_selected)

    def _init_ai_toggle_states(self):
        self._lstm_active = False
        self._hier_active = False
        self._three_active = False
        self._direct_finger_active = False
        self._direct_finger_v2_active = False
        self._console_control_active = False
        self._console_control_sensor_active = False
        self._ai_direct_finger_active = False
        self._ai_direct_finger_execution_active = False
        self._hier_mode_is_continues = True

        self._set_button_active(self.predict_threelevel_hierarchical_transformer_gesture_button, False)
        self._set_button_active(self.direct_finger_motion_button, False)
        self._set_button_active(self.direct_finger_motion_v2_button, False)
        self._set_button_active(self.console_control_button, False)
        if hasattr(self, "console_control_sensor_button"):
            self._set_button_active(self.console_control_sensor_button, False)
        self._set_button_active(self.ai_direct_finger_motion_button, False)
        self._set_button_active(self.ai_direct_finger_motion_execution_button, False)

        if all(
            hasattr(self, name)
            for name in (
                "send_position_PTP_J_button",
                "send_position_PTP_T_button",
                "send_position_PTP_T_toolframe_button",
                "send_script_button",
            )
        ):
            self._set_button_active(self.send_position_PTP_J_button, False)
            self._set_button_active(self.send_position_PTP_T_button, False)
            self._set_button_active(self.send_position_PTP_T_toolframe_button, False)
            self._set_button_active(self.send_script_button, False)

        three = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
        latch_on = bool(getattr(three, "latch_mode", False)) if three else False

        self._set_button_active(self.btn_toggle_3lvl_latch, latch_on)
        self.btn_toggle_3lvl_latch.setText(f"3-Level: Latch {'ON' if latch_on else 'OFF'}")
        self._update_anchor_button_label()

    def _install_keyboard_shortcuts(self):
        self.shortcut_toggle_ai_direct_finger_motion = QShortcut(QKeySequence("Space"), self)
        self.shortcut_toggle_ai_direct_finger_motion.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_toggle_ai_direct_finger_motion.activated.connect(
            self._shortcut_toggle_ai_direct_finger_motion
        )

        self.shortcut_stop_ai_direct_finger_motion = QShortcut(QKeySequence("Esc"), self)
        self.shortcut_stop_ai_direct_finger_motion.setContext(Qt.WidgetWithChildrenShortcut)
        self.shortcut_stop_ai_direct_finger_motion.activated.connect(
            self._shortcut_stop_ai_direct_finger_motion
        )

        if hasattr(self, "ai_direct_finger_motion_button"):
            self.ai_direct_finger_motion_button.setToolTip(
                "Start/stop AI Direct Finger Motion recording. Shortcut: Space. Emergency stop: Esc."
            )
        if hasattr(self, "ai_direct_finger_motion_execution_button"):
            self.ai_direct_finger_motion_execution_button.setToolTip(
                "Start/stop AI Direct Finger Motion execution using a trained checkpoint."
            )

    def _focus_on_text_input(self):
        focused = self.focusWidget()
        return isinstance(focused, (QLineEdit, QTextEdit))

    def _is_data_training_subtab_active(self):
        return hasattr(self, "ai_sub_tabs") and self.ai_sub_tabs.currentIndex() == 1

    def _shortcut_toggle_ai_direct_finger_motion(self):
        if self._focus_on_text_input():
            return
        if not self._is_data_training_subtab_active():
            return
        if not self.ai_direct_finger_motion_button.isEnabled():
            return
        self._on_toggle_ai_direct_finger_motion()

    def _shortcut_stop_ai_direct_finger_motion(self):
        if self._focus_on_text_input():
            return
        if not getattr(self, "_ai_direct_finger_active", False):
            return
        if not self.ai_direct_finger_motion_button.isEnabled():
            return
        self._on_toggle_ai_direct_finger_motion()

    def on_toggle_threelevel_latch(self):
        try:
            three = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
            if not three:
                print("[UI] 3-Level instance not available")
                return
            three.toggle_latch_mode()
            latch = bool(getattr(three, "latch_mode", False))
            self.btn_toggle_3lvl_latch.setText(f"3-Level: Latch {'ON' if latch else 'OFF'}")
            self._set_button_active(self.btn_toggle_3lvl_latch, latch)
        except Exception as exc:
            print(f"[UI] Could not toggle 3-Level latch mode: {exc}")

    def _on_sensitivity_changed(self, value: int):
        sensitivity_float = value / 1000.0
        self.sensitivity_value_label.setText(f"{sensitivity_float:.3f}")
        self.sensor_functions.set_touch_sensitivity(sensitivity_float)

    def _on_toggle_threelevel_predict(self):
        try:
            three = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
            if three is None:
                raise AttributeError("threelevel_hierarchical_transformer_class is not available")
            previous_state = bool(getattr(three, "is_recognizing_gesture", False))
            three.toggle_gesture_recognition()
            current_state = bool(getattr(three, "is_recognizing_gesture", False))
            if current_state == previous_state:
                print("[UI] ThreeLevel state unchanged after toggle request.")
            self._three_active = current_state
            self._set_button_active(
                self.predict_threelevel_hierarchical_transformer_gesture_button,
                self._three_active,
            )
        except Exception as exc:
            print(f"[UI] ThreeLevel toggle failed: {exc}")
            self._three_active = bool(getattr(self, "_three_active", False))
            self._set_button_active(
                self.predict_threelevel_hierarchical_transformer_gesture_button,
                self._three_active,
            )
        self._update_anchor_button_label()

    def _update_anchor_button_label(self):
        if not hasattr(self, "btn_toggle_anchor_axes"):
            return

        anchor_available = (
            bool(getattr(self, "_three_active", False))
            and not bool(getattr(self, "_direct_finger_active", False))
        )

        self.btn_toggle_anchor_axes.setEnabled(anchor_available)

        if not anchor_available:
            self.btn_toggle_anchor_axes.setText("Axes: Anchored OFF")
            self.btn_toggle_anchor_axes.setStyleSheet(
                "QPushButton { background-color: #cfcfcf; color: #7a7a7a; }"
            )
            return

        three = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
        anchored = bool(getattr(three, "anchor_enabled", True)) if three else True
        self.btn_toggle_anchor_axes.setText(f"Axes: Anchored {'ON' if anchored else 'OFF'}")
        self._set_button_active(self.btn_toggle_anchor_axes, anchored)

    def _on_toggle_anchor_axes(self):
        if hasattr(self, "btn_toggle_anchor_axes") and not self.btn_toggle_anchor_axes.isEnabled():
            return

        three = self._get_sensor_helper("threelevel_hierarchical_transformer_class")
        if not three:
            print("[UI] 3-Level instance not available yet.")
            return

        new_val = not bool(getattr(three, "anchor_enabled", True))
        setattr(three, "anchor_enabled", new_val)

        if new_val and hasattr(three, "_set_anchor_from_current_frame"):
            try:
                three._set_anchor_from_current_frame()
            except Exception as exc:
                print(f"[UI] Could not set anchor: {exc}")

        self._update_anchor_button_label()

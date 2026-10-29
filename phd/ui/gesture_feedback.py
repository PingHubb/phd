"""Shared gesture-result styling and read-only Touch Policy feedback."""
import math

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget

from phd.ui import components, theme


def gesture_label(name):
    labels = {
        "anticlockwise": "Anticlockwise", "clockwise": "Clockwise",
        "diagonal_bottomleft_to_topright": "Diagonal: bottom-left → top-right",
        "diagonal_bottomright_to_topleft": "Diagonal: bottom-right → top-left",
        "diagonal_topleft_to_bottomright": "Diagonal: top-left → bottom-right",
        "diagonal_topright_to_bottomleft": "Diagonal: top-right → bottom-left",
    }
    return labels.get(name, name.replace("_", " ").capitalize())


class GestureResultCard(QWidget):
    """Same title, wrapped gesture name, confidence caption and 6px bar."""

    def __init__(self, title, hint, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        box, rows = components.section(title)
        self.result_label = self._label("Waiting for a gesture", "title")
        self.result_label.setAccessibleName(title)
        self.confidence_label = self._label(hint, "caption")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 1000)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setFixedHeight(6)
        self.confidence_bar.setAccessibleName(title + " confidence")
        for widget in (self.result_label, self.confidence_label, self.confidence_bar):
            rows.add(widget)
        layout.addWidget(box)
        self._display = None

    @staticmethod
    def _label(text, role):
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.PlainText)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        label.setStyleSheet(theme.type_style(role))
        return label

    def show_state(self, title, detail, confidence=None, warning=False):
        score = float(confidence) if confidence is not None else 0.0
        score = min(1.0, max(0.0, score)) if math.isfinite(score) else 0.0
        display = (title, detail, round(score * 1000), bool(warning))
        if display == self._display:
            return
        self._display = display
        self.result_label.setText(title)
        self.result_label.setStyleSheet(theme.type_style("title", color=theme.WARNING if warning else theme.TEXT_PRIMARY))
        self.confidence_label.setText(detail)
        self.confidence_bar.setValue(display[2])
        self.result_label.setAccessibleDescription(detail)

    def show_result(self, result, context):
        confidence = result.get("confidence")
        if confidence is None:
            self.show_state("Not recognised", "Touch ended before a usable prediction was available.", warning=True)
            return
        accepted = bool(result.get("accepted"))
        detail = f"{context} · {confidence * 100:.2f}% confidence · {result.get('margin', 0) * 100:.2f}-point lead"
        if not accepted and result.get("top_prediction"):
            detail += "\nClosest match: " + gesture_label(result["top_prediction"]) + ". Not confirmed."
        self.show_state(gesture_label(result["gesture"]), detail, confidence, warning=not accepted)


class TouchPolicyGestureFeedback(QWidget):
    """Render existing executor results; no model, listener or robot ownership."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.SPACE_MD)
        self.current = GestureResultCard("Current gesture", "Updates while you touch the sensor.")
        self.completed = GestureResultCard("Last completed gesture", "The last result stays here after you lift your finger.")
        layout.addWidget(self.current)
        layout.addWidget(self.completed)

    def clear(self):
        self.current.show_state("Waiting for a gesture", "Press Execute to see live Touch Policy predictions.")
        self.completed.show_state("Waiting for a gesture", "The last result stays here after you lift your finger.")

    def update_from_helper(self, helper):
        if helper is None or not getattr(helper, "model_loaded", False):
            self.clear()
            return
        if getattr(helper, "model_kind", None) != "cnn_tcn_gesture":
            self.current.show_state("Velocity policy", "This checkpoint predicts velocity, not named gestures.")
            self.completed.show_state("Gesture labels unavailable", "Select a gesture-policy checkpoint for named results.")
            return
        feedback = getattr(helper, "touch_gesture_feedback", None)
        completed = getattr(feedback, "completed", None)
        if completed:
            context = ("Last direction before release" if completed.get("continuous") else "Last recognised direction before release")
            self.completed.show_result(completed, context)
        else:
            self.completed.show_state("Waiting for a gesture", "The result appears after confirmed finger release.")
        prediction = getattr(helper, "last_prediction", None) or {}
        if not getattr(helper, "is_running", False):
            self.current.show_state("Stopped", prediction.get("reason") or "Press Execute to start. The completed result is kept below.")
            return
        gesture = prediction.get("gesture")
        from phd.dependence.touch_gesture_policy import SUPPORTED_GESTURE_NAMES
        if gesture in SUPPORTED_GESTURE_NAMES | {"uncertain"}:
            confidence = float(prediction.get("mode_conf", 0))
            if gesture != "uncertain" and prediction.get("mode") == "stop":
                self.current.show_state("Confirming " + gesture_label(gesture),
                                        "Waiting for consecutive matching predictions.", confidence)
            else:
                self.current.show_result({"gesture": gesture, "confidence": confidence,
                    "accepted": gesture != "uncertain", "margin": prediction.get("gesture_margin", 0),
                    "top_prediction": prediction.get("top_gesture")},
                    "Live · dry run" if prediction.get("dry_run", True) else "Live prediction")
            return
        reason = prediction.get("reason") or "Waiting for touch frames."
        prefix = getattr(helper, "_touch_prefix", None)
        if getattr(prefix, "touching", False):
            title = "Paused" if "paused" in reason else "Collecting gesture"
        else:
            title = "Waiting for touch"
        self.current.show_state(title, reason)

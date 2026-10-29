"""Read-only checkpoint details, without constructing a policy or controller."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QLabel, QSizePolicy, QVBoxLayout, QWidget

from phd.ui import theme


def describe_touch_checkpoint(bundle):
    """Use checkpoint metadata, never filename guesses or a fixed gesture list."""
    if not isinstance(bundle, dict):
        raise ValueError("Checkpoint has no model metadata")
    if bundle.get("format") == "pinglab_touch_gesture_policy_v1":
        labels = bundle.get("class_names", [])
        members = bundle.get("members", [])
        if not labels or not members or not all(isinstance(name, str) for name in labels):
            raise ValueError("Missing gesture labels or architecture")
        architectures = set()
        for member in members:
            config = member["config"]
            if config["classes"] != len(labels):
                raise ValueError("Output size does not match gesture labels")
            architectures.add(config["architecture"])
        names = {"cnn_gru": "CNN–GRU", "cnn_tcn": "CNN–TCN"}
        architecture = " + ".join(names.get(a, a) for a in sorted(architectures))
        if len(members) > 1:
            architecture += f" ensemble ({len(members)})"
        shape = bundle.get("sensor_shape", [])
        shape_text = f" · {shape[0]} × {shape[1]} sensor" if len(shape) == 2 else ""
        lines = [f"{architecture} · {len(labels)} gestures{shape_text}",
                 "Outputs: " + " · ".join(name.replace("_", " ").capitalize() for name in labels)]
        preprocess = bundle.get("preprocess", {})
        if preprocess.get("prediction_unit") == "recent_touch_window":
            lines.append(f"Input: latest {float(preprocess['window_duration_sec']) * 1000:g} ms of touch")
        else:
            lines.append("Input: current touch history")
            if preprocess.get("prediction_unit") == "available_touch_prefix":
                lines.append("Single-gesture mode: lift your finger before the next gesture.")
        lines.append("Gesture is mapped to velocity; this model does not estimate finger speed.")
        return "\n".join(lines)
    config = bundle.get("config", {})
    if (bundle.get("model_class") == "TactileCNNGRUPolicy"
            or config.get("model_family") == "tactile_cnn_gru_policy"):
        return "CNN–GRU · velocity policy\nOutputs: velocity and control mode, not named swipe gestures."
    raise ValueError("No supported Touch Policy metadata. This may be a legacy or different-task checkpoint.")


def read_touch_checkpoint_details(path):
    # Restricted CPU deserialization only. Never fall back to arbitrary pickle
    # execution just to display a selected file's metadata.
    import torch
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    return describe_touch_checkpoint(bundle)


class TouchPolicyModelDetails(QWidget):
    """Asynchronous preview; obsolete selections cannot overwrite current text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("Select a model to see its architecture and outputs.")
        self.label.setTextFormat(Qt.PlainText)
        self.label.setWordWrap(True)
        self.label.setMinimumWidth(0)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.label.setAccessibleName("Selected Touch Policy model details")
        self.label.setStyleSheet(theme.type_style("caption"))
        layout.addWidget(self.label)
        self._executor = None
        self._future = None
        self._key = None
        self._closed = False
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._poll)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def set_source(self, path, *, loaded_bundle=None, loaded_kind=None):
        if self._closed:
            return
        # Running details always come from the actual loaded model, even if
        # someone replaces its file on disk while execution is active.
        if loaded_kind is not None:
            key = ("loaded", str(path), loaded_kind,
                   id(loaded_bundle) if loaded_kind == "cnn_tcn_gesture" else None)
            if key == self._key:
                return
            self._invalidate(key)
            try:
                if loaded_kind == "cnn_tcn_gesture":
                    detail = describe_touch_checkpoint(loaded_bundle)
                else:
                    architecture = {"cnn_gru": "CNN–GRU", "legacy_transformer": "CNN–Transformer"}.get(loaded_kind, loaded_kind)
                    detail = f"{architecture} · velocity policy\nOutputs: velocity and control mode, not named swipe gestures."
                self.label.setText(detail)
            except Exception as exc:
                self.label.setText("Model details unavailable: " + str(exc))
            return
        path = Path(path).expanduser() if path else None
        try:
            stat = path.stat() if path is not None else None
            key = (str(path), stat.st_mtime_ns, stat.st_size) if stat else ("empty",)
        except OSError:
            key = (str(path), "missing")
        if key == self._key:
            return
        self._invalidate(key)
        if path is None or key[-1] == "missing":
            self.label.setText("Select an existing model to see its architecture and outputs.")
            return
        self.label.setText("Reading model architecture and outputs…")
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="touch-model-details")
        self._future = self._executor.submit(read_touch_checkpoint_details, str(path))
        self.timer.start()

    def _invalidate(self, key):
        self._key = key
        self.label.setToolTip("")
        self.timer.stop()
        if self._future is not None:
            self._future.cancel()
        self._future = None

    def _poll(self):
        future = self._future
        if future is None or not future.done():
            return
        self.timer.stop()
        self._future = None
        try:
            self.label.setText(future.result())
        except Exception as exc:
            # Do not call an unrestricted loader for legacy pickle files.
            self.label.setText("Could not safely preview this checkpoint's metadata.\n"
                               "Choose a supported Touch Policy model; no model has been started.")
            self.label.setToolTip(str(exc))

    def shutdown(self):
        self._closed = True
        self._invalidate(None)
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

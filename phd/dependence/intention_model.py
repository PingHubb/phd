"""Causal, context-conditioned goal prediction. No robot control API.

This is a reproducible baseline, not a claim of a novel cognitive architecture.
Only tactile windows and ALL candidate positions enter forward(); no chosen
target, declaration time, participant identity or trial clock is an input.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from phd.dependence.intention_data import CLASSES, SCHEMA, validate_goals
from phd.dependence.touch_tracking_data import CausalPreprocessor, LAYOUT, safe_name
from phd.dependence.touch_tracking_model import normalize_frames


class IntentionNet(nn.Module):
    def __init__(self, rows, cols, mode="fusion", hidden=48):
        super().__init__()
        if mode not in ("fusion", "tactile_only", "context_only"):
            raise ValueError("Unknown intention model mode.")
        self.mode = mode
        if mode != "context_only":
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 12, 3, padding=1), nn.GroupNorm(3, 12), nn.GELU(),
                nn.Conv2d(12, 12, 3, padding=1), nn.GELU(), nn.Flatten(),
                nn.Linear(12*rows*cols, hidden), nn.GELU(), nn.Dropout(.1),
            )
            self.temporal = nn.GRU(hidden, hidden, batch_first=True)
            self.norm = nn.LayerNorm(hidden)
        else:
            hidden = 1
        if mode == "tactile_only":
            self.head = nn.Linear(hidden, 4)
        else:
            # Shared scorer makes swapping A/B/C positions swap their logits.
            self.score = nn.Sequential(nn.Linear(hidden+2, 32), nn.GELU(), nn.Linear(32, 1))
            self.no_goal = nn.Linear(hidden, 1)

    def forward(self, frames, goals):
        n, t, c, rows, cols = frames.shape
        if self.mode == "context_only":
            h = frames.new_zeros((n, 1))
        else:
            x = self.encoder(frames.reshape(n*t, c, rows, cols)).reshape(n, t, -1)
            x, _ = self.temporal(x)
            h = self.norm(x[:, -1])
        if self.mode == "tactile_only":
            return self.head(h)
        target = self.score(torch.cat([h[:, None].expand(-1, 3, -1), goals], dim=-1)).squeeze(-1)
        return torch.cat([self.no_goal(h), target], dim=-1)


def load_intention_model(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("layout") != LAYOUT:
        raise ValueError("Choose an Intention Learning checkpoint, not a tracking or robot-policy model.")
    config = checkpoint["config"]
    if config.get("classes") != CLASSES or config["model"]["mode"] != "fusion":
        raise ValueError("Live preview requires the four-class, context-conditioned main model.")
    if config["seq_len"] < 2 or config["temperature"] <= 0 or not np.isfinite(config["temperature"]):
        raise ValueError("Invalid checkpoint sequence length or temperature.")
    for key in ("mean", "std"):
        value = np.asarray(config["normalization"][key])
        if value.shape != (4,) or not np.isfinite(value).all() or (key == "std" and np.any(value <= 0)):
            raise ValueError("Invalid checkpoint normalization.")
    if not np.isfinite(config["confidence_threshold"]) or not 0 <= config["confidence_threshold"] <= 1.01:
        raise ValueError("Invalid checkpoint confidence threshold.")
    model = IntentionNet(**config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


class IntentionPreview:
    """One worker owns this state. Every invalid/gap/reset clears history.

    No labels are accepted by this interface. Low confidence and 'No current
    goal' are different states, and neither constitutes a safety stop command.
    """
    def __init__(self, path):
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        self.model, self.config = load_intention_model(path)
        self.shape = (self.config["model"]["rows"], self.config["model"]["cols"])
        self.reset()

    def reset(self):
        self.pp = CausalPreprocessor(self.config["hz"], self.config["max_gap"])
        self.history = deque(maxlen=self.config["seq_len"])
        self.last_t = self.last_seq = self.baseline = self.mask = self.goals = None

    def invalid(self, reason):
        self.reset()
        return {"valid": False, "reason": reason, "robot_commands_enabled": False}

    def update_many(self, frames, *, sensor_id, sensor_model, reorder_mode, goals):
        goals = validate_goals(goals)
        if (safe_name(sensor_id) not in self.config["sensor_ids"]
                or sensor_model != self.config["sensor_model"] or reorder_mode != self.config["reorder_mode"]):
            return self.invalid("Sensor identity/model/order differs from the training data.")
        if self.goals is not None and not np.array_equal(self.goals, goals):
            self.reset()
        self.goals = goals
        if not frames:
            return {"valid": False, "reason": "Waiting for fresh frames.", "robot_commands_enabled": False}
        for timestamp, sequence, raw, baseline, mask in frames:
            raw, baseline, mask = np.asarray(raw), np.asarray(baseline), np.asarray(mask, bool)
            if any(x.shape != self.shape for x in (raw, baseline, mask)) or not np.isfinite(timestamp):
                return self.invalid("Sensor geometry or timestamp changed.")
            if self.last_seq is not None:
                if sequence == self.last_seq:
                    continue
                if (sequence < self.last_seq or timestamp <= self.last_t
                        or not np.array_equal(baseline, self.baseline, equal_nan=True)
                        or not np.array_equal(mask, self.mask)):
                    return self.invalid("Stream restarted, or calibration/mask changed. Rewarming required.")
            for _, sample, valid in self.pp.push(float(timestamp), raw, baseline, mask):
                if not valid:
                    self.history.clear()
                else:
                    self.history.append(sample)
            self.last_t, self.last_seq = timestamp, sequence
            self.baseline, self.mask = baseline.copy(), mask.copy()
        if len(self.history) < self.config["seq_len"]:
            return {"valid": False, "reason": f"Warming up: {len(self.history)}/{self.config['seq_len']} frames.",
                    "robot_commands_enabled": False}
        x = normalize_frames(np.stack(self.history), self.config["normalization"])
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(x[None]), torch.from_numpy(goals[None]))
            p = torch.softmax(logits/self.config["temperature"], dim=-1)[0].numpy()
        index = int(p.argmax())
        accepted = float(p[index]) >= self.config["confidence_threshold"]
        return {"valid": True, "accepted": accepted, "goal": index if accepted else None,
                "label": CLASSES[index] if accepted else "Uncertain — prediction withheld",
                "probabilities": p.tolist(), "timestamp": float(self.last_t),
                "robot_commands_enabled": False}

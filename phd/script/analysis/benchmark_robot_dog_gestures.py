# Grouped implementation; the former flat script path remains supported.
"""Baselines, ablations, data efficiency, earliness and a cross-sensor pilot.

Run: python3 -m phd.script.benchmark_robot_dog_gestures --output /new/folder [--only E1,E2]

Uses the same chronological per-class split as train_robot_dog_gestures
(60/20/20). The test split is only reported, never used for selection or
early stopping. Source recordings are read, never modified.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from phd.dependence.paths import ai_resource_path
from phd.dependence.robot_dog_gesture_model import (
    CLASS_NAMES, PREPROCESS, GestureClassifier, load_trial, prepare_trial,
)
from phd.script.train_robot_dog_gestures import augment_batch, classification_metrics, split_trials

DOG_ROOT = Path(ai_resource_path("data", "tactile_recordings"))
OFFSET_ROOT = Path(ai_resource_path("data", "offset"))
TRANSFER_CLASSES = ("swipe_up", "swipe_down", "swipe_left", "swipe_right", "clockwise", "anticlockwise")
SEEDS = (17, 43, 79)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------- data

def load_dog():
    records, trials = [], []
    for folder in sorted(DOG_ROOT.glob("session_robot_dog_*")):
        label = folder.name.removeprefix("session_robot_dog_")
        label = {"anitclockwise": "anticlockwise"}.get(label, label)
        for path in sorted(folder.glob("*.npz")):
            raw, averaged, times, meta = load_trial(path)
            records.append({"path": str(path), "label": label, "class_index": CLASS_NAMES.index(label),
                            "trial_number": int(meta["trial_number"]), "started_at": float(meta["started_at"])})
            trials.append((raw, averaged, times))
    return records, trials, np.array([r["class_index"] for r in records])


def prepare_all(trials, config, perturb=None, rng=None):
    out = []
    for raw, averaged, times in trials:
        if perturb is not None:
            raw, averaged, times = perturb(raw.copy(), averaged.copy(), times.copy(), rng)
        try:
            out.append(prepare_trial(raw, averaged, times, config))
        except ValueError:
            out.append(None)
    return out


def stack(prepared, fallback_shape):
    valid = np.array([item is not None for item in prepared])
    filled = [item if item is not None else np.zeros(fallback_shape, np.float32) for item in prepared]
    return np.stack(filled), valid


# ------------------------------------------------------------ perturbations

def _offset(raw, averaged, times, rng):
    return raw + 2.0, averaged + 2.0, times


def _gain(raw, averaged, times, rng):
    return raw * 0.6, averaged * 0.6, times


def _noise(raw, averaged, times, rng):
    return raw + rng.normal(0, 0.5, raw.shape), averaged + rng.normal(0, 0.25, averaged.shape), times


def _dead(raw, averaged, times, rng):
    mask = rng.random(raw.shape[1:]) < 0.10
    raw[:, mask] = 0.0
    averaged[:, mask] = 0.0
    return raw, averaged, times


def _shift(raw, averaged, times, rng):
    def roll(a):
        out = np.zeros_like(a)
        out[:, :, 1:] = a[:, :, :-1]
        return out
    return roll(raw), roll(averaged), times


def _warp(gamma):
    def apply(raw, averaged, times, rng):
        span = times[-1] - times[0]
        u = (times - times[0]) / max(span, 1e-9)
        return raw, averaged, times[0] + span * u ** gamma
    return apply


PERTURBATIONS = {
    "clean": None, "offset_+2%": _offset, "gain_x0.6": _gain, "noise": _noise,
    "dead_10%": _dead, "shift_1_taxel": _shift, "warp_fast_start": _warp(0.6), "warp_slow_start": _warp(1.6),
}


# --------------------------------------------------------------------- models

class FlatBiLSTM(nn.Module):
    """TaCo-style BiLSTM baseline on flattened frames."""

    def __init__(self, cells, classes=10, width=96):
        super().__init__()
        self.lstm = nn.LSTM(3 * cells, width // 2, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(width, 1)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(0.2), nn.Linear(width, classes))

    def forward(self, x):
        sequence, _ = self.lstm(x.flatten(2))
        weights = self.attention(sequence).softmax(dim=1)
        return self.head((sequence * weights).sum(dim=1))


class Conv3D(nn.Module):
    def __init__(self, classes=10, width=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1), nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool3d((4, 2, 2)), nn.Flatten(),
            nn.Linear(32 * 16, width), nn.GELU(), nn.Dropout(0.2), nn.Linear(width, classes),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


class Masked(nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.model = model
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32).view(1, 1, -1, 1, 1))

    def forward(self, x):
        return self.model(x * self.mask)


def make_model(name, classes, cells=64):
    if name in ("cnn_gru", "cnn_tcn", "cnn_transformer"):
        return GestureClassifier(name, classes=classes)
    if name == "bilstm_flat":
        return FlatBiLSTM(cells, classes)
    if name == "cnn3d":
        return Conv3D(classes)
    raise ValueError(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def predict(model, x, batch_size=256):
    model.eval()
    return torch.cat([model(batch).softmax(-1) for batch in x.split(batch_size)]).cpu().numpy()


def fit(model, train_x, train_y, val_x, val_y, *, augment=True, epochs=60, patience=12, lr=1e-3, batch_size=64):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, stale = None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for index in torch.randperm(len(train_x), device=DEVICE).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch = augment_batch(train_x[index]) if augment else train_x[index]
            loss = loss_function(model(batch), train_y[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        metrics = classification_metrics(predict(model, val_x), val_y)
        rank = (metrics["macro_f1"], -metrics["nll"])
        if best is None or rank > best[0]:
            best = (rank, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best[2])
    return model, best[1]


def summarise(probabilities, labels, valid=None):
    """Metrics over every trial; an unprocessable trial counts as wrong."""
    valid = np.ones(len(labels), bool) if valid is None else np.asarray(valid, bool)
    if not valid.any():
        return {"accuracy": 0.0, "macro_f1": 0.0, "nll": None, "correct": 0, "samples": int(len(labels)), "coverage": 0.0}
    metrics = classification_metrics(probabilities[valid], labels[valid])
    return {"accuracy": metrics["correct"] / len(labels), "macro_f1": metrics["macro_f1"], "nll": metrics["nll"],
            "correct": metrics["correct"], "samples": int(len(labels)), "coverage": float(valid.mean())}


# ---------------------------------------------------------- centroid baseline

def centroid_features(x):
    w = np.clip(-x[:, :, 1], 0, None)
    rows, cols = np.mgrid[0:x.shape[3], 0:x.shape[4]]
    total = w.sum(axis=(2, 3)) + 1e-6
    cy = (w * rows).sum(axis=(2, 3)) / total
    cx = (w * cols).sum(axis=(2, 3)) / total
    path = np.stack([cx - cx[:, :1], cy - cy[:, :1]], -1)
    step = np.diff(path, axis=1)
    turn = (step[:, :-1, 0] * step[:, 1:, 1] - step[:, :-1, 1] * step[:, 1:, 0]).sum(1, keepdims=True)
    return np.concatenate([path.reshape(len(path), -1), turn], 1)


def knn_probabilities(train_f, train_y, test_f, classes, k=5):
    mu, sd = train_f.mean(0), train_f.std(0) + 1e-6
    a, b = (train_f - mu) / sd, (test_f - mu) / sd
    distance = ((b[:, None] - a[None]) ** 2).sum(-1)
    votes = train_y[np.argsort(distance, axis=1)[:, :k]]
    counts = np.stack([np.bincount(row, minlength=classes) for row in votes]).astype(float)
    return (counts + 0.01) / (counts + 0.01).sum(1, keepdims=True)


# ----------------------------------------------------------------- helpers

class Recorder:
    def __init__(self, output):
        self.output = output
        self.path = output / "results.json"
        self.results = json.loads(self.path.read_text()) if self.path.exists() else {}

    def put(self, *keys, value):
        node = self.results
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
        self.path.write_text(json.dumps(self.results, indent=1) + "\n")

    def get(self, *keys):
        node = self.results
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node


def normalise(x, train_index):
    mean = x[train_index].mean(axis=(0, 1, 3, 4))
    std = np.maximum(x[train_index].std(axis=(0, 1, 3, 4)), 0.05)
    return lambda a: ((a - mean[None, None, :, None, None]) / std[None, None, :, None, None]).astype(np.float32)


def tensor(a):
    return torch.tensor(a, device=DEVICE)


# -------------------------------------------------------- E1 + E2: models

def run_models(rec, dog, splits, args):
    records, trials, y = dog
    variants = {
        # name: (model, config, channel mask, augment)
        "cnn_tcn": ("cnn_tcn", PREPROCESS, (1, 1, 1), True),
        "cnn_gru": ("cnn_gru", PREPROCESS, (1, 1, 1), True),
        "cnn_transformer": ("cnn_transformer", PREPROCESS, (1, 1, 1), True),
        "bilstm_flat": ("bilstm_flat", PREPROCESS, (1, 1, 1), True),
        "cnn3d": ("cnn3d", PREPROCESS, (1, 1, 1), True),
        "ablate_no_augmentation": ("cnn_tcn", PREPROCESS, (1, 1, 1), False),
        "ablate_no_spatial_median": ("cnn_tcn", dict(PREPROCESS, subtract_spatial_median=False), (1, 1, 1), True),
        "ablate_instant_only": ("cnn_tcn", PREPROCESS, (1, 0, 0), True),
        "ablate_averaged_only": ("cnn_tcn", PREPROCESS, (0, 1, 0), True),
        "ablate_averaged_plus_delta": ("cnn_tcn", PREPROCESS, (0, 1, 1), True),
        "ablate_no_delta": ("cnn_tcn", PREPROCESS, (1, 1, 0), True),
    }
    cache = {}
    for name, (model_name, config, mask, augment) in variants.items():
        key = json.dumps(config, sort_keys=True)
        if key not in cache:
            prepared = {p: stack(prepare_all(trials, config, fn, np.random.default_rng(0)), (32, 3, 8, 8))
                        for p, fn in PERTURBATIONS.items()}
            norm = normalise(prepared["clean"][0], splits["train"])
            cache[key] = {p: (norm(x), valid) for p, (x, valid) in prepared.items()}
        data = cache[key]
        x = data["clean"][0]
        for seed in SEEDS:
            if rec.get("models", name, str(seed)) is not None:
                continue
            seed_everything(seed)
            started = time.monotonic()
            model = Masked(make_model(model_name, 10), mask)
            model, epoch = fit(model, tensor(x[splits["train"]]), tensor(y[splits["train"]]).long(),
                               tensor(x[splits["validation"]]), y[splits["validation"]], augment=augment)
            entry = {"best_epoch": epoch, "seconds": time.monotonic() - started,
                     "parameters": sum(p.numel() for p in model.parameters())}
            for perturbation, (px, valid) in data.items():
                probabilities = predict(model, tensor(px[splits["test"]]))
                entry[perturbation] = summarise(probabilities, y[splits["test"]], valid[splits["test"]])
            rec.put("models", name, str(seed), value=entry)
            print(f"E1/E2 {name} seed={seed} clean={entry['clean']['accuracy']:.3f} "
                  f"epoch={epoch} {entry['seconds']:.0f}s", flush=True)
    data = cache[json.dumps(PREPROCESS, sort_keys=True)]
    train_f = centroid_features(data["clean"][0][splits["train"]])
    entry = {}
    for perturbation, (px, valid) in data.items():
        probabilities = knn_probabilities(train_f, y[splits["train"]], centroid_features(px[splits["test"]]), 10)
        entry[perturbation] = summarise(probabilities, y[splits["test"]], valid[splits["test"]])
    rec.put("models", "centroid_5nn", "0", value=entry)
    return data


# ------------------------------------------------------- E3: data efficiency

def run_data_efficiency(rec, dog, splits, data, args):
    _records, _trials, y = dog
    x = data["clean"][0]
    train_by_class = {c: splits["train"][y[splits["train"]] == c] for c in range(10)}
    for k in (1, 2, 5, 10, 20, 40, 60):
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            chosen = np.concatenate([rng.choice(train_by_class[c], size=min(k, len(train_by_class[c])), replace=False)
                                     for c in range(10)])
            if rec.get("data_efficiency", "centroid_5nn", str(k), str(seed)) is None:
                probabilities = knn_probabilities(centroid_features(x[chosen]), y[chosen],
                                                  centroid_features(x[splits["test"]]), 10, k=min(5, k))
                rec.put("data_efficiency", "centroid_5nn", str(k), str(seed),
                        value=summarise(probabilities, y[splits["test"]]))
            for model_name in ("cnn_tcn", "cnn_gru"):
                if rec.get("data_efficiency", model_name, str(k), str(seed)) is not None:
                    continue
                seed_everything(seed)
                model, epoch = fit(make_model(model_name, 10), tensor(x[chosen]), tensor(y[chosen]).long(),
                                   tensor(x[splits["validation"]]), y[splits["validation"]],
                                   epochs=80 if k < 10 else 60, patience=20 if k < 10 else 12,
                                   batch_size=min(64, len(chosen)))
                result = summarise(predict(model, tensor(x[splits["test"]])), y[splits["test"]])
                rec.put("data_efficiency", model_name, str(k), str(seed), value=result)
                print(f"E3 {model_name} k={k} seed={seed} acc={result['accuracy']:.3f}", flush=True)


# -------------------------------------------------------------- E4: earliness

PREFIXES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TRAIN_PREFIXES = (0.3, 0.45, 0.6, 0.75, 0.9, 1.0)


def prefix_trial(raw, averaged, times, fraction):
    touching = np.flatnonzero(np.any(averaged < PREPROCESS["touch_threshold_percent"], axis=(1, 2)))
    start, end = int(touching[0]), min(len(raw), int(touching[-1]) + 2)
    cut = max(start + 2, int(round(start + fraction * (end - start))))
    return raw[:cut], averaged[:cut], times[:cut]


def run_earliness(rec, dog, splits, args):
    records, trials, y = dog
    by_prefix = {}
    for fraction in sorted(set(PREFIXES) | set(TRAIN_PREFIXES)):
        prepared = []
        for raw, averaged, times in trials:
            try:
                prepared.append(prepare_trial(*prefix_trial(raw, averaged, times, fraction)))
            except (ValueError, IndexError):
                prepared.append(None)
        by_prefix[fraction] = stack(prepared, (32, 3, 8, 8))
    norm = normalise(by_prefix[1.0][0], splits["train"])
    by_prefix = {f: (norm(x), valid) for f, (x, valid) in by_prefix.items()}
    durations = [times[-1] - times[0] for _raw, _averaged, times in trials]
    rec.put("earliness", "median_touch_duration_s", value=float(np.median(durations)))
    for model_name, train_on in (("cnn_tcn", (1.0,)), ("cnn_tcn_prefix_trained", TRAIN_PREFIXES)):
        architecture = "cnn_tcn"
        for seed in SEEDS:
            if rec.get("earliness", model_name, str(seed)) is not None:
                continue
            seed_everything(seed)
            tx = np.concatenate([by_prefix[f][0][splits["train"]] for f in train_on])
            ty = np.concatenate([y[splits["train"]] for _ in train_on])
            vx = np.concatenate([by_prefix[f][0][splits["validation"]] for f in train_on])
            vy = np.concatenate([y[splits["validation"]] for _ in train_on])
            model, _epoch = fit(make_model(architecture, 10), tensor(tx), tensor(ty).long(), tensor(vx), vy)
            entry = {}
            for fraction in PREFIXES:
                px, valid = by_prefix[fraction]
                entry[str(fraction)] = summarise(predict(model, tensor(px[splits["test"]])),
                                                 y[splits["test"]], valid[splits["test"]])
            rec.put("earliness", model_name, str(seed), value=entry)
            print(f"E4 {model_name} seed={seed} " + " ".join(
                f"{f}:{entry[str(f)]['accuracy']:.2f}" for f in PREFIXES), flush=True)
    entry = {}
    train_f = centroid_features(by_prefix[1.0][0][splits["train"]])
    for fraction in PREFIXES:
        px, valid = by_prefix[fraction]
        probabilities = knn_probabilities(train_f, y[splits["train"]], centroid_features(px[splits["test"]]), 10)
        entry[str(fraction)] = summarise(probabilities, y[splits["test"]], valid[splits["test"]])
    rec.put("earliness", "centroid_5nn", "0", value=entry)


# --------------------------------------------------- E5: cross-sensor pilot

DIHEDRAL = [(k, flip) for flip in (False, True) for k in range(4)]
CANONICAL = {"swipe_up": (-1, 0), "swipe_down": (1, 0), "swipe_left": (0, -1), "swipe_right": (0, 1)}


def transform(frames, k, flip):
    out = np.flip(frames, axis=2) if flip else frames
    return np.rot90(out, k, axes=(1, 2))


def transfer_frames(frames, threshold=0.25):
    """Peak-normalised, median-removed touch episode of averaged signal."""
    frames = frames - np.median(frames, axis=(1, 2), keepdims=True)
    peak = max(float(-frames.min()), 1e-6)
    frames = frames / peak
    active = np.flatnonzero(frames.reshape(len(frames), -1).min(1) <= -threshold)
    if len(active) < 2:
        raise ValueError("no touch")
    return frames[active[0]:min(len(frames), active[-1] + 2)]


def to_input(frames, canvas, steps=32):
    t = torch.tensor(frames.copy(), dtype=torch.float32)[:, None]
    if canvas == "canvas16":
        t = F.interpolate(t, size=(16, 16), mode="bilinear", align_corners=True)
    else:
        pad = (0, 10 - t.shape[3], 0, 10 - t.shape[2])
        t = F.pad(t, pad)
    a = t[:, 0].numpy()
    source = np.linspace(0, 1, len(a))
    target = np.linspace(0, 1, steps)
    flat = a.reshape(len(a), -1)
    resampled = np.stack([np.interp(target, source, flat[:, i]) for i in range(flat.shape[1])], 1)
    resampled = np.arcsinh(3.0 * resampled.reshape(steps, *a.shape[1:]))
    delta = np.zeros_like(resampled)
    delta[1:] = np.diff(resampled, axis=0)
    return np.stack([np.zeros_like(resampled), resampled, delta], 1).astype(np.float32)


def displacement(frames):
    w = np.clip(-frames, 0, None)
    rows, cols = np.mgrid[0:frames.shape[1], 0:frames.shape[2]]
    total = w.sum((1, 2)) + 1e-9
    r = (w * rows).sum((1, 2)) / total / max(frames.shape[1] - 1, 1)
    c = (w * cols).sum((1, 2)) / total / max(frames.shape[2] - 1, 1)
    return np.array([r[-1] - r[0], c[-1] - c[0]])


def register(frames_by_class):
    """Dihedral orientation that best matches the canonical swipe directions."""
    episodes = {}
    for label in CANONICAL:
        for frames in frames_by_class.get(label, []):
            try:
                episodes.setdefault(label, []).append(transfer_frames(frames))
            except ValueError:
                pass
    best = None
    for k, flip in DIHEDRAL:
        score = 0.0
        for label, vector in CANONICAL.items():
            for frames in episodes.get(label, []):
                score += float(np.dot(displacement(transform(frames, k, flip)), vector))
        if best is None or score > best[0]:
            best = (score, k, flip)
    return best[1], best[2]


def load_txt(path, rows, cols):
    a = np.loadtxt(path, ndmin=2)
    return a.reshape(len(a), cols, rows).transpose(0, 2, 1)


def run_transfer(rec, dog, splits, args):
    records, trials, y = dog
    sources = {"cylinder_10x9": {}, "flat_10x10": {}}
    for label in TRANSFER_CLASSES:
        for path in sorted(OFFSET_ROOT.glob(f"gesture_cylinder_{label}/*.txt")):
            sources["cylinder_10x9"].setdefault(label, []).append(load_txt(path, 10, 9))
        for path in sorted(OFFSET_ROOT.glob(f"gesture_2d_10x10_1_finger_{label}/*.txt")):
            sources["flat_10x10"].setdefault(label, []).append(load_txt(path, 10, 10))
    labels = {name: i for i, name in enumerate(TRANSFER_CLASSES)}
    dog_index = {split: np.array([i for i in splits[split] if records[i]["label"] in labels]) for split in splits}
    dog_frames = {}
    for split, indices in dog_index.items():
        for i in indices:
            dog_frames.setdefault(split, []).append((records[i]["label"], trials[i][1]))
    # Orientation registration stands in for the taxel map; it uses each
    # sensor's own training recordings, never the dog test split.
    orientation = {name: register(by_class) for name, by_class in sources.items()}
    dog_train_by_class = {}
    for label, frames in dog_frames["train"]:
        dog_train_by_class.setdefault(label, []).append(frames)
    orientation["dog_8x8"] = register(dog_train_by_class)
    rec.put("transfer", "orientation", value={k: {"rot90": int(v[0]), "flip": bool(v[1])} for k, v in orientation.items()})

    def build(items, name, canvas, registered):
        xs, ys = [], []
        k, flip = orientation[name] if registered else (0, False)
        for label, frames in items:
            try:
                episode = transfer_frames(frames)
            except ValueError:
                continue
            xs.append(to_input(transform(episode, k, flip), canvas))
            ys.append(labels[label])
        return np.stack(xs), np.array(ys)

    for canvas in ("index_pad10", "canvas16"):
        for registered in (False, True):
            tag = f"{canvas}_{'registered' if registered else 'unregistered'}"
            if rec.get("transfer", tag, "done"):
                continue
            sx, sy = [], []
            for name in sources:
                items = [(label, f) for label, fs in sources[name].items() for f in fs]
                bx, by = build(items, name, canvas, registered)
                sx.append(bx)
                sy.append(by)
            sx, sy = np.concatenate(sx), np.concatenate(sy)
            dx = {split: build(dog_frames[split], "dog_8x8", canvas, registered) for split in dog_frames}
            mean = sx.mean(axis=(0, 1, 3, 4))
            std = np.maximum(sx.std(axis=(0, 1, 3, 4)), 0.05)
            norm = lambda a: ((a - mean[None, None, :, None, None]) / std[None, None, :, None, None]).astype(np.float32)
            rng = np.random.default_rng(0)
            order = rng.permutation(len(sx))
            hold = order[: len(sx) // 5]
            fit_index = order[len(sx) // 5:]
            test_x, test_y = tensor(norm(dx["test"][0])), dx["test"][1]
            results = {"source_trials": int(len(sx)), "dog_test_trials": int(len(test_y))}
            for seed in SEEDS:
                seed_everything(seed)
                source_model, _ = fit(GestureClassifier("cnn_tcn", classes=len(labels)),
                                      tensor(norm(sx[fit_index])), tensor(sy[fit_index]).long(),
                                      tensor(norm(sx[hold])), sy[hold])
                state = {key: value.detach().clone() for key, value in source_model.state_dict().items()}
                results.setdefault("zero_shot", {})[str(seed)] = summarise(predict(source_model, test_x), test_y)
                dog_train_x, dog_train_y = dx["train"]
                for k_shot in (1, 5, 10, 60):
                    rng = np.random.default_rng(seed)
                    chosen = np.concatenate([rng.choice(np.flatnonzero(dog_train_y == c), k_shot, replace=False)
                                             for c in range(len(labels))])
                    for init in ("pretrained", "scratch"):
                        seed_everything(seed)
                        model = GestureClassifier("cnn_tcn", classes=len(labels))
                        if init == "pretrained":
                            model.load_state_dict(state)
                        model, _ = fit(model, tensor(norm(dog_train_x[chosen])), tensor(dog_train_y[chosen]).long(),
                                       tensor(norm(dx["validation"][0])), dx["validation"][1],
                                       epochs=60, patience=15, lr=5e-4 if init == "pretrained" else 1e-3,
                                       batch_size=min(64, len(chosen)))
                        results.setdefault(f"{init}_{k_shot}shot", {})[str(seed)] = summarise(predict(model, test_x), test_y)
                print(f"E5 {tag} seed={seed} zero-shot={results['zero_shot'][str(seed)]['accuracy']:.3f} "
                      f"1-shot pre/scratch={results['pretrained_1shot'][str(seed)]['accuracy']:.2f}/"
                      f"{results['scratch_1shot'][str(seed)]['accuracy']:.2f}", flush=True)
            results["done"] = True
            rec.put("transfer", tag, value=results)


# ------------------------------------------------------------------- report

def _mean_sd(values):
    values = [v * 100 for v in values]
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def make_report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = json.loads((output / "results.json").read_text())
    figure, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    ax = axes[0, 0]
    names = ["centroid_5nn", "bilstm_flat", "cnn3d", "cnn_gru", "cnn_transformer", "cnn_tcn"]
    perts = ["clean", "noise", "dead_10%", "shift_1_taxel", "gain_x0.6", "offset_+2%", "warp_fast_start"]
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        stats = [_mean_sd([s[p]["accuracy"] for s in r["models"][name].values()]) for p in perts]
        ax.bar(np.arange(len(perts)) + i * width, [s[0] for s in stats], width,
               yerr=[s[1] for s in stats], label=name, capsize=2)
    ax.set_xticks(np.arange(len(perts)) + 0.4 - width / 2, perts, rotation=20)
    ax.set_ylim(50, 101)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Baselines under test-time perturbations")
    ax.legend(fontsize=8, ncol=3)

    ax = axes[0, 1]
    names = ["cnn_tcn", "ablate_no_augmentation", "ablate_no_spatial_median", "ablate_instant_only",
             "ablate_averaged_only", "ablate_no_delta", "ablate_averaged_plus_delta"]
    perts = ["clean", "noise", "shift_1_taxel", "gain_x0.6", "offset_+2%"]
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        stats = [_mean_sd([s[p]["accuracy"] for s in r["models"][name].values()]) for p in perts]
        ax.bar(np.arange(len(perts)) + i * width, [s[0] for s in stats], width,
               yerr=[s[1] for s in stats], label=name.replace("ablate_", "− " if "no" in name else "only "), capsize=2)
    ax.set_xticks(np.arange(len(perts)) + 0.4 - width / 2, perts)
    ax.set_ylim(0, 101)
    ax.set_title("CNN-TCN ablations")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    for name, by_k in r["data_efficiency"].items():
        ks = sorted(by_k, key=float)
        stats = [_mean_sd([v["accuracy"] for v in by_k[k].values()]) for k in ks]
        ax.errorbar([float(k) for k in ks], [s[0] for s in stats], yerr=[s[1] for s in stats],
                    marker="o", capsize=3, label=f"dog only · {name}")
    tag = r["transfer"]["index_pad10_registered"]
    shots = [1, 5, 10, 60]
    for init, label in (("pretrained", "6 classes · cylinder+flat pretrained, registered"),
                        ("scratch", "6 classes · dog only, from scratch")):
        stats = [_mean_sd([v["accuracy"] for v in tag[f"{init}_{k}shot"].values()]) for k in shots]
        ax.errorbar(shots, [s[0] for s in stats], yerr=[s[1] for s in stats], marker="s", capsize=3,
                    linestyle="--", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Training trials per class on the dog skin")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Data efficiency")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    duration = r["earliness"]["median_touch_duration_s"] * 1000
    for name in ("centroid_5nn", "cnn_tcn", "cnn_tcn_prefix_trained"):
        fractions = sorted(next(iter(r["earliness"][name].values())), key=float)
        stats = [_mean_sd([s[f]["accuracy"] for s in r["earliness"][name].values()]) for f in fractions]
        ax.errorbar([float(f) * duration for f in fractions], [s[0] for s in stats],
                    yerr=[s[1] for s in stats], marker="o", capsize=3, label=name)
    ax.set_xlabel(f"Time after touch onset (ms; median gesture {duration:.0f} ms)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Earliness: decision before the gesture ends")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "benchmark_summary.png", dpi=140)
    plt.close(figure)


# --------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--only", default="E1,E3,E4,E5", help="Comma-separated: E1 (models+ablations), E3, E4, E5")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    rec = Recorder(args.output)
    dog = load_dog()
    splits = split_trials(dog[0])
    rec.put("split", value={name: int(len(index)) for name, index in splits.items()})
    only = set(args.only.split(","))
    data = None
    if "E1" in only or "E3" in only:
        data = run_models(rec, dog, splits, args) if "E1" in only else None
    if "E3" in only:
        if data is None:
            prepared = stack(prepare_all(dog[1], PREPROCESS), (32, 3, 8, 8))
            data = {"clean": (normalise(prepared[0], splits["train"])(prepared[0]), prepared[1])}
        run_data_efficiency(rec, dog, splits, data, args)
    if "E4" in only:
        run_earliness(rec, dog, splits, args)
    if "E5" in only:
        run_transfer(rec, dog, splits, args)
    make_report(args.output)
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check, train and evaluate continuous touch tracking with explicit provenance.

No ROS or hardware connection is created. Old policy recordings are deliberately
not accepted. Every run is immutable and has a fresh directory, fixed split,
training-only normalization, validation-based selection, and one final test.
Camera-free mode uses operator states and sensor-estimated position targets;
its scores are agreement metrics, not independent physical accuracy.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import uuid

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phd.dependence.touch_tracking_data import (
    CHANNELS, LAYOUT, SCHEMA, dataset_report, import_reference_csv, save_json,
    split_trials, window_indices,
)


def training_normalization(trials):
    values = np.concatenate([t["frames"][t["frame_valid"]] for t in trials])
    mean, std = np.zeros(4, np.float32), np.ones(4, np.float32)
    mask = values[:, 2] > .5
    for channel in (0, 1):
        samples = values[:, channel][mask]
        if not len(samples):
            raise ValueError("No valid training taxels.")
        mean[channel] = samples.mean()
        std[channel] = max(float(samples.std()), 0.01)
    # Age is a physical-time input, not an arbitrary frame-count feature.
    std[3] = 0.02
    return {"mean": mean.tolist(), "std": std.tolist()}


def _window_specs(trials, seq_len):
    return [(i, int(k)) for i, t in enumerate(trials) for k in window_indices(t, seq_len)]


class TrackingDataset:
    def __init__(self, trials, seq_len, normalization, velocity_scale, *, augment=False):
        self.trials, self.seq_len, self.normalization = trials, seq_len, normalization
        self.velocity_scale, self.augment = velocity_scale, augment
        self.indices = _window_specs(trials, seq_len)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        from phd.dependence.touch_tracking_model import normalize_frames
        i, k = self.indices[index]
        trial = self.trials[i]
        x = trial["frames"][k-self.seq_len+1:k+1].copy()
        label = trial["labels"]
        xy, velocity = label["xy"][k].copy(), label["velocity"][k].copy()
        if self.augment:
            # Identical transform across the entire window AND regression labels.
            if np.random.random() < .5:
                x = x[..., ::-1].copy()
                xy[0] = 1 - xy[0]
                velocity[0] *= -1
            if np.random.random() < .5:
                x = x[..., ::-1, :].copy()
                xy[1] = 1 - xy[1]
                velocity[1] *= -1
            gain = np.random.uniform(.95, 1.05)
            x[:, :2] *= gain
            x[:, 0] += np.random.uniform(-.05, .05) * x[:, 2]
        return {"x": normalize_frames(x, self.normalization), "xy": xy,
                "velocity": velocity / self.velocity_scale,
                "contact": np.float32(label["contact"][k]),
                "position_valid": np.float32(label["position_valid"][k]),
                "velocity_valid": np.float32(label["velocity_valid"][k]),
                "trial": i, "frame": k}


def _loss(out, batch):
    import torch
    import torch.nn.functional as f
    contact = f.binary_cross_entropy_with_logits(out["contact_logits"], batch["contact"])
    def masked(pred, truth, mask):
        per_sample = f.smooth_l1_loss(pred, truth, reduction="none").mean(dim=-1)
        return (per_sample * mask).sum() / torch.clamp(mask.sum(), min=1)
    position = masked(out["xy"], batch["xy"], batch["position_valid"])
    velocity = masked(out["velocity"], batch["velocity"], batch["velocity_valid"])
    return position + .25 * velocity + .1 * contact


def _loader(dataset, batch_size, training=False, seed=42):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    sampler = None
    if training:
        classes = []
        for i, k in dataset.indices:
            labels = dataset.trials[i]["labels"]
            moving = labels["velocity_valid"][k] and np.linalg.norm(labels["velocity"][k]) > .02
            classes.append(0 if labels["contact"][k] == 0 else (2 if moving else 1))
        classes = np.array(classes)
        counts = np.bincount(classes, minlength=3)
        weights = 1.0 / np.maximum(counts[classes], 1)
        sampler = WeightedRandomSampler(torch.as_tensor(weights), len(weights), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def predictions(model, dataset, batch_size, device):
    import torch
    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in _loader(dataset, batch_size):
            out = model(batch["x"].to(device))
            xy = out["xy"].cpu().numpy()
            velocity = out["velocity"].cpu().numpy() * dataset.velocity_scale
            probabilities = torch.sigmoid(out["contact_logits"]).cpu().numpy()
            for j, (i, k) in enumerate(zip(batch["trial"].tolist(), batch["frame"].tolist())):
                labels = dataset.trials[i]["labels"]
                rows.append({"trial": i, "frame": k, "t": float(dataset.trials[i]["t"][k]),
                             "truth_contact": int(labels["contact"][k]), "probability": float(probabilities[j]),
                             "truth_xy": labels["xy"][k].tolist(), "xy": xy[j].tolist(),
                             "truth_velocity": labels["velocity"][k].tolist(), "velocity": velocity[j].tolist(),
                             "position_valid": bool(labels["position_valid"][k]),
                             "velocity_valid": bool(labels["velocity_valid"][k])})
    return rows


def select_contact_threshold(rows):
    # Threshold uses validation only. It is not a calibrated confidence score.
    truth = np.array([r["truth_contact"] for r in rows], bool)
    p = np.array([r["probability"] for r in rows])
    if not truth.any() or truth.all():
        raise ValueError("Validation needs independently labelled touch AND no-touch examples.")
    candidates = np.linspace(.2, .9, 29)
    scores = [np.mean(p[~truth] >= t) + np.mean(p[truth] < t) for t in candidates]
    return float(candidates[int(np.argmin(scores))])


def metrics(rows, threshold=.5, surface_size_mm=None):
    if not rows:
        raise ValueError("No evaluation samples.")
    truth_contact = np.array([r["truth_contact"] for r in rows], bool)
    prediction_contact = np.array([r["probability"] >= threshold for r in rows])
    pos_valid = np.array([r["position_valid"] for r in rows], bool)
    vel_valid = np.array([r["velocity_valid"] for r in rows], bool)
    xy = np.array([r["xy"] for r in rows])
    xy_truth = np.array([r["truth_xy"] for r in rows])
    velocity = np.array([r["velocity"] for r in rows])
    velocity_truth = np.array([r["truth_velocity"] for r in rows])
    gated = np.where(prediction_contact[:, None], velocity, 0.)
    idle = ~truth_contact
    stationary = truth_contact & vel_valid & (np.linalg.norm(velocity_truth, axis=1) <= .02)
    def mean_or_none(array):
        return float(np.mean(array)) if len(array) else None
    position_error = np.linalg.norm(xy[pos_valid] - xy_truth[pos_valid], axis=1)
    velocity_error = gated[vel_valid] - velocity_truth[vel_valid]
    result = {
        "samples": len(rows), "position_samples": int(pos_valid.sum()), "velocity_samples": int(vel_valid.sum()),
        "contact_accuracy": float(np.mean(prediction_contact == truth_contact)),
        "false_contact_rate": mean_or_none(prediction_contact[idle]),
        "missed_contact_rate": mean_or_none(~prediction_contact[truth_contact]),
        "position_rmse_surface_units": float(np.sqrt(np.mean(position_error**2))) if len(position_error) else None,
        "position_p95_surface_units": float(np.quantile(position_error, .95)) if len(position_error) else None,
        "velocity_rmse_surface_units_per_sec": float(np.sqrt(np.mean(velocity_error**2))) if len(velocity_error) else None,
        "no_touch_false_motion_rate": mean_or_none(np.linalg.norm(gated[idle], axis=1) > .02),
        "stationary_false_motion_rate": mean_or_none(np.linalg.norm(gated[stationary], axis=1) > .02),
        "stationary_speed_rms": float(np.sqrt(np.mean(gated[stationary]**2))) if stationary.any() else None,
        "motion_threshold_surface_units_per_sec": .02,
        "note": "Position error includes all labelled contact samples, even model misses. Velocity uses contact-gated output. Rates are frame-level, not certified safety rates.",
    }
    moving = vel_valid & truth_contact & (np.linalg.norm(velocity_truth, axis=1) > .02) & (np.linalg.norm(gated, axis=1) > .02)
    if moving.any():
        a, b = velocity_truth[moving], gated[moving]
        cos = np.sum(a*b, axis=1) / (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
        result["direction_error_deg_when_both_moving"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())
    else:
        result["direction_error_deg_when_both_moving"] = None
    if surface_size_mm and min(surface_size_mm) > 0:
        scale = np.asarray(surface_size_mm)
        err = (xy[pos_valid] - xy_truth[pos_valid]) * scale
        result["position_rmse_mm"] = float(np.sqrt(np.mean(np.sum(err**2, axis=1)))) if len(err) else None
        err = velocity_error * scale
        result["velocity_rmse_mm_per_sec"] = float(np.sqrt(np.mean(err**2))) if len(err) else None
    return result


def estimate_velocity_lag_ms(rows, threshold=.5):
    """Diagnostic only; predictions/errors are NEVER shifted to improve scores.

    Positive lag means prediction trails the reference. Needs varying motion;
    static/short/weakly correlated trajectories return None, not a false zero.
    """
    if len(rows) < 30:
        return None
    times = np.array([r["t"] for r in rows])
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return None
    truth = np.array([r["truth_velocity"] for r in rows])
    pred = np.array([r["velocity"] if r["probability"] >= threshold else [0, 0] for r in rows])
    valid = np.array([r["velocity_valid"] and r["truth_contact"] == 1 for r in rows])
    candidates = []
    for offset in range(-int(.3/dt), int(.3/dt)+1):
        wanted = times - offset*dt
        right = np.clip(np.searchsorted(times, wanted), 0, len(times)-1)
        left = np.maximum(right-1, 0)
        indices = np.where(np.abs(times[left]-wanted) < np.abs(times[right]-wanted), left, right)
        keep = valid & valid[indices] & (np.abs(times[indices]-wanted) < dt*.2)
        if keep.sum() < 25:
            continue
        a, b = pred[keep], truth[indices[keep]]
        a, b = a-a.mean(axis=0), b-b.mean(axis=0)
        energy = np.linalg.norm(a)*np.linalg.norm(b)
        if energy < 1e-6:
            continue
        correlation = float((a*b).sum()/energy)
        candidates.append((correlation, -abs(offset), offset*dt*1000))
    if not candidates:
        return None
    best = max(candidates)
    return float(best[2]) if best[0] >= .5 else None


def centroid_predictions(trials, seq_len, contact_threshold, *, both_polarities=False):
    """Simple comparison baseline, separate from both reference/weak targets."""
    rows = []
    for i, trial in enumerate(trials):
        r, c = trial["frames"].shape[-2:]
        yy, xx = np.meshgrid((np.arange(r)+.5)/r if both_polarities else np.linspace(0, 1, r),
                             (np.arange(c)+.5)/c if both_polarities else np.linspace(0, 1, c), indexing="ij")
        prev_xy, prev_t, prev_contact = None, None, False
        ends = set(window_indices(trial, seq_len).tolist())
        for k, (t, frame) in enumerate(zip(trial["t"], trial["frames"])):
            w = (np.abs(frame[0]) if both_polarities else np.maximum(-frame[0], 0)) * frame[2]
            total = w.sum()
            strength = float(w.max())
            touch = strength > contact_threshold and trial["frame_valid"][k]
            xy = np.array([float((w*xx).sum()), float((w*yy).sum())]) / max(float(total), 1e-6)
            velocity = np.zeros(2)
            if touch and prev_contact and prev_t is not None and t > prev_t:
                xy = .5*xy + .5*prev_xy
                velocity = (xy - prev_xy) / (t-prev_t)
            if k in ends:
                l = trial["labels"]
                rows.append({"trial": i, "frame": k, "t": float(t), "truth_contact": int(l["contact"][k]),
                             "probability": float(touch), "truth_xy": l["xy"][k].tolist(), "xy": xy.tolist(),
                             "truth_velocity": l["velocity"][k].tolist(), "velocity": velocity.tolist(),
                             "position_valid": bool(l["position_valid"][k]), "velocity_valid": bool(l["velocity_valid"][k])})
            prev_xy, prev_t, prev_contact = xy, t, touch
    return rows


def save_evaluation(folder, rows, trials, threshold, size):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    report = metrics(rows, threshold, size)
    report["per_trial"] = {Path(trial["path"]).name: metrics([r for r in rows if r["trial"] == i], threshold, size)
                           for i, trial in enumerate(trials) if any(r["trial"] == i for r in rows)}
    for i, trial in enumerate(trials):
        per_trial = report["per_trial"].get(Path(trial["path"]).name)
        if per_trial is not None:
            per_trial["velocity_lag_ms_diagnostic"] = estimate_velocity_lag_ms([r for r in rows if r["trial"] == i], threshold)
    report["lag_note"] = "Diagnostic correlation over +/-300 ms; positive means delayed prediction. None means insufficient evidence. Never used to align/reduce reported errors."
    camera_free = trials[0].get("supervision") == "camera_free_sensor_estimates_v1"
    target_name = "Sensor-estimated target (not ground truth)" if camera_free else "Independent reference"
    if camera_free:
        def relabel(values):
            converted = {("sensor_target_agreement_" + k if k.startswith(("position_", "velocity_", "direction_", "stationary_")) else
                     "operator_label_" + k if k in {"contact_accuracy", "false_contact_rate", "missed_contact_rate", "no_touch_false_motion_rate"} else k):
                    ({name: relabel(data) for name, data in v.items()} if k == "per_trial" else v)
                    for k, v in values.items()}
            converted["note"] = "Agreement with sensor-estimated positions/velocities and operator-declared contact; not independent physical accuracy."
            converted["independent_physical_accuracy_measured"] = False
            return converted
        report = relabel(report)
        report["independent_physical_accuracy_measured"] = False
        report["note"] = "Weak supervision: position/velocity scores measure agreement with sensor-estimated targets, NOT real finger accuracy. Contact intervals were declared by the operator. Ambiguous positions are excluded."
    save_json(folder / "metrics.json", report)
    with (folder / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "t", "declared_contact" if camera_free else "true_contact", "contact_probability",
                         *(["estimated_target_x", "estimated_target_y"] if camera_free else ["true_x", "true_y"]),
                         "pred_x", "pred_y",
                         *(["estimated_target_vx", "estimated_target_vy"] if camera_free else ["true_vx", "true_vy"]),
                         "pred_vx", "pred_vy", "position_valid", "velocity_valid"])
        for row in rows:
            writer.writerow([Path(trials[row["trial"]]["path"]).name, row["t"], row["truth_contact"], row["probability"],
                             *row["truth_xy"], *row["xy"], *row["truth_velocity"], *row["velocity"], row["position_valid"], row["velocity_valid"]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for i, trial in enumerate(trials):
            selected = [r for r in rows if r["trial"] == i]
            if not selected:
                continue
            t = np.array([r["t"] for r in selected])
            xy = np.array([r["xy"] for r in selected])
            truth = np.array([r["truth_xy"] for r in selected])
            pv = np.array([r["position_valid"] for r in selected])
            xy[~pv], truth[~pv] = np.nan, np.nan
            fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
            for axis, name in enumerate(("x", "y")):
                axes[axis].plot(t, truth[:, axis], label=target_name)
                axes[axis].plot(t, xy[:, axis], label="Tactile model", alpha=.8)
                axes[axis].set_ylabel(name + " / surface size")
                axes[axis].legend()
            vtruth = np.array([r["truth_velocity"] for r in selected])
            vpred = np.array([r["velocity"] if r["probability"] >= threshold else [0, 0] for r in selected])
            vv = np.array([r["velocity_valid"] for r in selected])
            vtruth[~vv] = np.nan
            for axis in range(2):
                axes[axis+2].plot(t, vtruth[:, axis], label=target_name)
                axes[axis+2].plot(t, vpred[:, axis], label="Contact-gated prediction", alpha=.8)
                axes[axis+2].set_ylabel(("vx" if axis == 0 else "vy") + " / surface units/s")
                axes[axis+2].legend()
            axes[4].plot(t, [r["truth_contact"] for r in selected], label="Operator-declared contact" if camera_free else "Reference contact")
            axes[4].plot(t, [r["probability"] for r in selected], label="Predicted contact probability")
            axes[4].axhline(threshold, color="gray", linestyle="--")
            axes[4].set_xlabel("Time / s")
            axes[4].legend()
            fig.suptitle(Path(trial["path"]).name)
            fig.tight_layout()
            fig.savefig(folder / f"trial_{i:03d}.png", dpi=130)
            plt.close(fig)
    except ImportError:
        print("Matplotlib unavailable; metrics and prediction CSV saved without figures.", flush=True)
    return report


def train(args):
    import torch
    from phd.dependence.touch_tracking_model import TouchTracker, load_tracker
    if args.epochs < 1 or args.batch_size < 1 or args.seq_len < 2 or args.patience < 1 or args.lr <= 0:
        raise ValueError("Epochs/batch size must be positive; sequence length must be >= 2.")
    supervision = getattr(args, "supervision", "independent")
    report, trials = dataset_report(args.root, args.seq_len, args.hz, args.max_gap, args.label_gap, supervision=supervision)
    if report["invalid_trials"] or not trials:
        print(json.dumps(report, indent=2), flush=True)
        raise ValueError("Dataset check failed. Put only reviewed, complete trials in the training root; originals are not changed.")
    geometry = {(tuple(t["metadata"]["sensor_shape"]), tuple(t["metadata"].get("surface_size_mm", [0, 0]))) for t in trials}
    if len(geometry) != 1:
        raise ValueError("Train each sensor geometry/physical size separately; no silent resizing.")
    ordering = {t["metadata"].get("reorder_mode", "") for t in trials}
    if len(ordering) != 1:
        raise ValueError("Taxel ordering differs between trials; use one consistent sensor coordinate layout.")
    splits = split_trials(trials, group_by=args.group_by, seed=args.seed)
    for name, group in splits.items():
        spec = _window_specs(group, args.seq_len)
        states = [group[i]["labels"]["contact"][k] for i, k in spec]
        moving = sum(group[i]["labels"]["velocity_valid"][k] and np.linalg.norm(group[i]["labels"]["velocity"][k]) > .02 for i, k in spec)
        if not spec or not {0, 1}.issubset(set(states)) or not moving:
            raise ValueError(f"{name} needs valid windows covering touch movement AND no touch. Collect both in each independent session.")
    normalization = training_normalization(splits["train"])
    velocities = np.concatenate([t["labels"]["velocity"][t["labels"]["velocity_valid"]] for t in splits["train"]])
    velocity_scale = max(.05, float(np.quantile(np.abs(velocities), .95)))
    shape, size = next(iter(geometry))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(max(1, args.threads))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:6]
    output = Path(args.output) / (("camera_free_" if supervision == "camera_free" else "tracking_") + stamp)
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIR={output.resolve()}", flush=True)
    save_json(output / "dataset_check.json", report)
    split_manifest = {key: [{"path": t["path"], "participant_id": t["metadata"]["participant_id"],
                             "session_id": t["metadata"]["session_id"], "data_hash": t["data_hash"],
                             "reference_hash": t["reference_hash"]} for t in group] for key, group in splits.items()}
    save_json(output / "split.json", {"group_by": args.group_by, "seed": args.seed, "splits": split_manifest})
    config_common = {"seq_len": args.seq_len, "hz": args.hz, "max_gap": args.max_gap, "label_gap": args.label_gap,
                     "normalization": normalization, "velocity_scale": velocity_scale,
                     "surface_size_mm": list(size), "channels": CHANNELS,
                     "reorder_mode": next(iter(ordering)),
                     "sensor_ids": sorted({t["metadata"]["sensor_id"] for t in splits["train"]}),
                     "training_reference": "camera_free_sensor_estimates_v1" if supervision == "camera_free" else "independent_reviewed_position_and_contact",
                     "robot_commands_enabled": False}
    config_common["training_arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    datasets = {key: TrackingDataset(group, args.seq_len, normalization, velocity_scale, augment=key == "train" and not args.no_augment)
                for key, group in splits.items()}
    comparisons = []
    for architecture in args.architectures.split(","):
        architecture = architecture.strip()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
        config = {**config_common, "model": {"rows": shape[0], "cols": shape[1], "architecture": architecture, "hidden": 64, "dropout": .1, "context_frames": args.seq_len},
                  "contact_threshold": .5}
        model = TouchTracker(**config["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best_loss, stale, history = float("inf"), 0, []
        checkpoint_path = output / f"{architecture}.pt"
        train_loader = _loader(datasets["train"], args.batch_size, True, args.seed)
        val_loader = _loader(datasets["validation"], args.batch_size)
        for epoch in range(args.epochs):
            epoch_metrics = {"epoch": epoch + 1}
            for stage, loader in (("train", train_loader), ("validation", val_loader)):
                model.train(stage == "train")
                total, count = 0., 0
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.set_grad_enabled(stage == "train"):
                        loss = _loss(model(batch["x"]), batch)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite loss; stop and inspect labels/signals.")
                        if stage == "train":
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                            optimizer.step()
                    n = len(batch["contact"])
                    total += float(loss.detach()) * n
                    count += n
                epoch_metrics[stage + "_loss"] = total / count
            history.append(epoch_metrics)
            print(f"{architecture}: epoch {epoch+1} train={epoch_metrics['train_loss']:.5f} val={epoch_metrics['validation_loss']:.5f}", flush=True)
            if epoch_metrics["validation_loss"] < best_loss:
                best_loss, stale = epoch_metrics["validation_loss"], 0
                torch.save({"schema": SCHEMA, "layout": LAYOUT, "config": config,
                            "model_state": model.state_dict(), "best_epoch": epoch+1}, checkpoint_path)
            else:
                stale += 1
            save_json(output / f"{architecture}_history.json", history)
            if stale >= args.patience:
                break
        model, config = load_tracker(checkpoint_path, device)
        val_rows = predictions(model, datasets["validation"], args.batch_size, device)
        threshold = select_contact_threshold(val_rows)
        config["contact_threshold"] = threshold
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["config"] = config
        torch.save(checkpoint, checkpoint_path)
        val_report = save_evaluation(output / (architecture + "_validation"), val_rows, splits["validation"], threshold, size)
        comparisons.append({"architecture": architecture, "best_validation_loss": best_loss,
                            "checkpoint": checkpoint_path.name, "validation": val_report})
    selected = min(comparisons, key=lambda v: v["best_validation_loss"])
    model, config = load_tracker(output / selected["checkpoint"], device)
    test_rows = predictions(model, datasets["test"], args.batch_size, device)
    test_report = save_evaluation(output / "test", test_rows, splits["test"], config["contact_threshold"], size)
    strength = lambda x: np.abs(x) if supervision == "camera_free" else np.maximum(-x, 0)
    idle_strength = np.concatenate([np.max(strength(t["frames"][:, 0]) * t["frames"][:, 2], axis=(1, 2))[t["labels"]["contact"] == 0] for t in splits["train"]])
    baseline_threshold = float(np.quantile(idle_strength, .99)) + 1e-6
    baseline_rows = centroid_predictions(splits["test"], args.seq_len, baseline_threshold, both_polarities=supervision == "camera_free")
    baseline_report = save_evaluation(output / "centroid_test", baseline_rows, splits["test"], .5, size)
    # Separate explicit checkpoint, never update an old latest-policy alias.
    chosen = torch.load(output / selected["checkpoint"], map_location="cpu", weights_only=True)
    torch.save(chosen, output / "selected_model.pt")
    summary = {"schema": SCHEMA, "selection": "lowest validation loss; test never used for selection",
               "supervision": supervision, "independent_physical_accuracy_measured": supervision == "independent",
               "selected_architecture": selected["architecture"], "comparison": comparisons, "test": test_report,
               "centroid_test": baseline_report, "robot_commands_enabled": False,
               "limitations": ["Single-contact surface tracking, not 3D finger pose or force.",
                               "Host timestamps require independently checked synchronization.",
                               "Contact probability is not a safety-certified confidence.",
                               "Check test plots and new live sessions before any robot integration."]}
    if supervision == "camera_free":
        summary["limitations"].extend([
            "Position/velocity targets come from the sensor estimator, not independent physical measurement.",
            "Touch/no-touch intervals are operator-declared; transitions and ambiguous positions are excluded.",
            "The neural network is not guaranteed to outperform the direct tracker.",
            "Live camera-free neural preview adds a signal-quality gate; offline metrics describe model outputs before that gate."])
    save_json(output / "summary.json", summary)
    print(json.dumps({"selected_architecture": selected["architecture"], "test": test_report}, indent=2), flush=True)
    print(f"COMPLETED={output.resolve()}", flush=True)
    return output


def main(argv=None):
    from phd.dependence.paths import ai_resource_path
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Read-only label/timing/geometry audit")
    check.add_argument("root", type=Path)
    check.add_argument("--seq-len", type=int, default=16)
    check.add_argument("--supervision", choices=["independent", "camera_free"], default="independent")
    fit = sub.add_parser("train", help="Offline model comparison and held-out evaluation")
    fit.add_argument("root", type=Path)
    fit.add_argument("--output", type=Path, help="New model run parent folder; defaults to the selected supervision namespace")
    fit.add_argument("--group-by", choices=["session", "participant"], default="session")
    fit.add_argument("--supervision", choices=["independent", "camera_free"], default="independent")
    fit.add_argument("--architectures", choices=["gru", "tcn", "gru,tcn"], default="gru,tcn")
    fit.add_argument("--seq-len", type=int, default=16)
    fit.add_argument("--hz", type=float, default=50.)
    fit.add_argument("--max-gap", type=float, default=.1)
    fit.add_argument("--label-gap", type=float, default=.12)
    fit.add_argument("--epochs", type=int, default=60)
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--lr", type=float, default=.001)
    fit.add_argument("--patience", type=int, default=8)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    fit.add_argument("--threads", type=int, default=2)
    fit.add_argument("--no-augment", action="store_true")
    ref = sub.add_parser("import-reference", help="Import independently measured normalized coordinates")
    ref.add_argument("trial", type=Path)
    ref.add_argument("csv", type=Path)
    ref.add_argument("--source", choices=["external_tracking", "robot_probe", "manual_camera"], default="external_tracking")
    ref.add_argument("--time-offset-sec", type=float, default=0.)
    ref.add_argument("--verified", action="store_true", help="Confirm coordinate/contact/timing review")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            report, _ = dataset_report(args.root, args.seq_len, supervision=args.supervision)
            print(json.dumps(report, indent=2))
            return 1 if report["invalid_trials"] or not report["ready_trials"] else 0
        if args.command == "import-reference":
            if not args.verified:
                raise ValueError("Review the external measurements first, then supply --verified.")
            import_reference_csv(args.trial, args.csv, source=args.source, time_offset_sec=args.time_offset_sec)
            print("Verified reference imported. Original sensor frames unchanged.")
        else:
            if args.output is None:
                args.output = Path(ai_resource_path("models", "touch_tracking_camera_free" if args.supervision == "camera_free" else "touch_tracking"))
            train(args)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

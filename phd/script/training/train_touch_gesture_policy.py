"""Compare streaming CNN-GRU/TCN policies using session_10x10_* folder labels.

Whole trials are split chronologically per class before prefix extraction.
Teacher velocities, robot state and proximity recordings are never labels or
features. Run with --output NEW_DIRECTORY; existing outputs are not replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from phd.dependence.paths import ai_resource_path
from phd.dependence.robot_dog_gesture_model import GestureClassifier, load_trial
from phd.dependence.touch_gesture_policy import (
    CLASS_NAMES, FORMAT, PREPROCESS_CONFIG, CONTINUOUS_PREPROCESS_CONFIG,
    SWIPE_CLASS_NAMES, SIX_GESTURE_CLASS_NAMES, TouchGesturePolicy, prepare_prefix, policy_directions,
)
from phd.script.training.train_robot_dog_gestures import augment_batch


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metrics(probabilities, labels, class_names=CLASS_NAMES):
    predicted = np.argmax(probabilities, axis=1)
    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    np.add.at(matrix, (labels, predicted), 1)
    precision = matrix.diagonal() / np.maximum(1, matrix.sum(0))
    recall = matrix.diagonal() / np.maximum(1, matrix.sum(1))
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    top = np.sort(probabilities, axis=1)
    accepted = (top[:, -1] >= 0.8) & (top[:, -1] - top[:, -2] >= 0.15)
    return {
        "samples": len(labels), "correct": int(np.sum(predicted == labels)),
        "accuracy": float(np.mean(predicted == labels)), "macro_f1": float(f1.mean()),
        "nll": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-8)).mean()),
        "accepted": int(accepted.sum()), "coverage": float(accepted.mean()),
        "accepted_accuracy": float(np.mean(predicted[accepted] == labels[accepted])) if accepted.any() else None,
        "confusion_matrix": matrix.tolist(),
        "per_class": [{"gesture": name, "support": int(matrix[i].sum()),
                       "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                      for i, name in enumerate(class_names)],
    }


def discover(root, class_names=CLASS_NAMES, config=None):
    records, trials, seen = [], [], {}
    for label in class_names:
        folder = Path(root) / ("session_10x10_" + label)
        paths = sorted(folder.glob("*.npz"))
        if len(paths) < 20:
            raise ValueError(f"Need at least 20 whole trials in {folder}; found {len(paths)}")
        for path in paths:
            raw, avg, times, meta = load_trial(path)
            if (raw.ndim != 3 or raw.shape[1:] != (10, 10) or avg.shape != raw.shape
                    or times.shape != (len(raw),) or len(raw) < 5
                    or not all(np.isfinite(a).all() for a in (raw, avg, times))
                    or np.any(np.diff(times) <= 0)):
                raise ValueError(f"Invalid 10x10 recording: {path}")
            if meta.get("completion_reason") != "touch_released":
                raise ValueError(f"Incomplete touch trial: {path}")
            if meta.get("sensor_reorder_mode") != "row_to_col" or meta.get("sensor_average_window_size") != 3:
                raise ValueError(f"Incompatible recording settings: {path}")
            signal_hash = hashlib.sha256(raw.tobytes() + avg.tobytes()).hexdigest()
            if signal_hash in seen:
                raise ValueError(f"Duplicate signal in {path} and {seen[signal_hash]}")
            seen[signal_hash] = str(path)
            contact = np.flatnonzero(np.any(avg < -3, axis=(1, 2)))
            if len(contact) < 5:
                raise ValueError(f"Insufficient contact: {path}")
            # End at the final contact frame: release cannot leak the label.
            first, last = int(contact[0]), int(contact[-1])
            trial = (raw[first:last+1], avg[first:last+1], times[first:last+1] - times[first])
            prepare_prefix(*trial, config=config)
            records.append({"path": str(path.resolve()), "label": label, "class_index": class_names.index(label),
                            "started_at": float(meta["started_at"]), "trial_number": int(meta["trial_number"]),
                            "metadata_session_tag": meta.get("session_tag"), "frames": len(raw),
                            "contact_frames": len(trial[0]), "contact_duration_sec": float(trial[2][-1]),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "signal_sha256": signal_hash})
            trials.append(trial)
    splits = {name: [] for name in ("train", "validation", "test")}
    for label in class_names:
        ids = sorted([i for i, r in enumerate(records) if r["label"] == label],
                     key=lambda i: (records[i]["started_at"], records[i]["trial_number"], records[i]["path"]))
        a, b = int(len(ids) * 0.6), int(len(ids) * 0.8)
        for name, indices in zip(splits, (ids[:a], ids[a:b], ids[b:])):
            splits[name].extend(indices)
            for i in indices:
                records[i]["split"] = name
    return records, trials, splits


def prefix_dataset(trials, records, indices, config=None):
    config = PREPROCESS_CONFIG if config is None else config
    frames, labels, owners, fractions = [], [], [], []
    for i in indices:
        raw, avg, times = trials[i]
        for end in range(5, len(raw) + 1):
            if not np.any(avg[end-1] < -3) or times[end-1] < config["minimum_duration_sec"]:
                continue
            try:
                prepared = prepare_prefix(raw[:end], avg[:end], times[:end], config)
            except ValueError as exc:
                if str(exc) == "Collecting more touch frames":
                    continue
                raise
            frames.append(prepared)
            labels.append(records[i]["class_index"])
            owners.append(i)
            fractions.append((end - 1) / max(1, len(raw) - 1))
    return np.stack(frames), np.asarray(labels), np.asarray(owners), np.asarray(fractions)


@torch.inference_mode()
def predict(model, x, batch_size=128):
    model.eval()
    return torch.cat([model(batch).softmax(-1) for batch in x.split(batch_size)]).cpu().numpy()


def calibrate(probabilities, temperature):
    p = np.log(np.maximum(probabilities, 1e-8)) / temperature
    p = np.exp(p - p.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def train_candidate(seed, train, validation, args, output, architecture="cnn_tcn", class_names=CLASS_NAMES):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    x, y, owners = train
    val_x, val_y = validation
    config = {"architecture": architecture, "classes": len(class_names), "width": 96, "dropout": 0.2, "sequence_length": 32}
    model = GestureClassifier(**config).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    # Equal class and trial contribution; long trials/push's extra files cannot dominate.
    owner_counts = {i: int(np.sum(owners == i)) for i in np.unique(owners)}
    class_trials = {c: len(np.unique(owners[y.cpu().numpy() == c])) for c in range(len(class_names))}
    weights = torch.tensor([1 / (owner_counts[i] * class_trials[int(c)]) for i, c in zip(owners, y.cpu().numpy())], device=x.device)
    history, best, stale = [], None, 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0
        selected = torch.multinomial(weights, len(owner_counts) * 6, replacement=True)
        for ids in selected.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(augment_batch(x[ids])), y[ids])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
        scheduler.step()
        prob = predict(model, val_x)
        score = metrics(prob, val_y, class_names)
        history.append({"epoch": epoch, "loss": total_loss / len(selected), "validation": score})
        rank = (score["macro_f1"], -score["nll"])
        if best is None or rank > best["rank"]:
            best = {"rank": rank, "epoch": epoch, "validation": score, "probabilities": prob,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{architecture} seed={seed} epoch={epoch} validation-prefix accuracy={score['accuracy']:.3f} macro-F1={score['macro_f1']:.3f}", flush=True)
        if stale >= args.patience:
            break
    name = f"{architecture}_seed{seed}"
    member = {"name": name, "config": config, "state_dict": best["state"]}
    torch.save(member, output / f"{name}.pt")
    save_json(output / f"{name}_history.json", history)
    summary = {"name": name, "architecture": architecture, "epoch": best["epoch"], "validation": best["validation"],
               "parameters": sum(p.numel() for p in model.parameters()), "seconds": time.monotonic() - started}
    print("CANDIDATE", json.dumps(summary), flush=True)
    return member, best["probabilities"], summary


def train(args):
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    class_names = {"six-directions": CLASS_NAMES, "four-swipes": SWIPE_CLASS_NAMES,
                   "six-gestures": SIX_GESTURE_CLASS_NAMES}[args.gesture_set]
    score_metrics = lambda p, y: metrics(p, y, class_names)
    preprocess = dict(CONTINUOUS_PREPROCESS_CONFIG if args.continuous else PREPROCESS_CONFIG)
    records, trials, splits = discover(args.dataset_root, class_names, preprocess)
    save_json(output / "dataset_manifest.json", {"records": records, "split": splits, "classes": list(class_names),
              "label_source": "folder name", "split_policy": "chronological whole trials per class: 60/20/20"})
    train_x, train_y, owners, _ = prefix_dataset(trials, records, splits["train"], preprocess)
    val_x, val_y, _, _ = prefix_dataset(trials, records, splits["validation"], preprocess)
    mean = train_x.mean(axis=(0, 1, 3, 4)); std = np.maximum(train_x.std(axis=(0, 1, 3, 4)), 0.05)
    normalize = lambda x: torch.tensor((x - mean[None, None, :, None, None]) / std[None, None, :, None, None], device=device)
    tx, vx = normalize(train_x), normalize(val_x)
    ty = torch.tensor(train_y, device=device)
    print("DATA", json.dumps({"trials": len(records), "splits": {k: len(v) for k, v in splits.items()},
                              "training_prefixes": len(tx), "validation_prefixes": len(vx), "device": str(device)}), flush=True)
    members, probabilities, candidates = [], [], []
    for architecture in args.architectures:
        for seed in args.seeds:
            member, prob, summary = train_candidate(seed, (tx, ty, owners), (vx, val_y), args, output, architecture, class_names)
            members.append(member); probabilities.append(prob); candidates.append(summary)
            save_json(output / "candidates.json", candidates)
    choices = [(m["name"], [i], score_metrics(p, val_y), p) for i, (m, p) in enumerate(zip(members, probabilities))]
    for architecture in args.architectures:
        ids = [i for i, m in enumerate(members) if m["config"]["architecture"] == architecture]
        if len(ids) > 1:
            prob = np.mean([probabilities[i] for i in ids], axis=0)
            choices.append((architecture + "_ensemble", ids, score_metrics(prob, val_y), prob))
    selected = max(choices, key=lambda c: (c[2]["macro_f1"], -c[2]["nll"], -len(c[1])))
    temperature = float(min(np.linspace(0.5, 3.0, 101), key=lambda t: score_metrics(calibrate(selected[3], t), val_y)["nll"]))
    bundle = {"format": FORMAT, "class_names": list(class_names), "sensor_shape": [10, 10],
              "sensor_reorder_mode": "row_to_col", "sensor_average_window_size": 3,
              "members": [members[i] for i in selected[1]], "selected_model": selected[0],
              "preprocess": preprocess, "temperature": temperature,
              "normalization": {"mean": mean.tolist(), "std": std.tolist()},
              "control": {"directions": policy_directions(class_names), "base_speed_mps": 0.05,
                          "confidence_threshold": 0.8, "margin_threshold": 0.15, "confirmation_predictions": 2,
                          "frame": "UI selected control frame", "release_stops_immediately": True},
              "training": {"dataset_manifest": str(output / "dataset_manifest.json"), "trials": len(records),
                           "split_counts": {k: len(v) for k, v in splits.items()}, "folder_labels_only": True,
                           "test_used_for_selection": False, "unknown_class_trained": False}}
    if args.continuous:
        bundle["training"].update(continuous_transitions_recorded=False, stationary_class_trained=False)
    torch.save(bundle, output / "best_touch_gesture_policy.pt")
    # Freeze an independently validation-selected checkpoint for each family
    # before opening the test split. This permits an honest side-by-side replay.
    family_bundles = {}
    for architecture in args.architectures:
        family = max((c for c in choices if c[0].startswith(architecture)),
                     key=lambda c: (c[2]["macro_f1"], -c[2]["nll"], -len(c[1])))
        family_temperature = float(min(np.linspace(0.5, 3.0, 101),
            key=lambda t: score_metrics(calibrate(family[3], t), val_y)["nll"]))
        family_bundles[architecture] = {**bundle, "members": [members[i] for i in family[1]],
                                      "selected_model": family[0], "temperature": family_temperature}
        torch.save(family_bundles[architecture], output / f"best_{architecture}_touch_policy.pt")
    save_json(output / "model_selection.json", {"selected": selected[0], "temperature": temperature,
              "options": [{"name": c[0], "members": c[1], "validation": c[2]} for c in choices],
              "test_used_for_selection": False, "arguments": vars(args)})
    # Test only after selecting and freezing the complete checkpoint.
    model = TouchGesturePolicy(bundle).to(device).eval()
    test_x, test_y, test_owners, fractions = prefix_dataset(trials, records, splits["test"], preprocess)
    with torch.inference_mode():
        test_prob = torch.cat([model(torch.tensor(b, device=device)) for b in np.array_split(test_x, max(1, int(np.ceil(len(test_x)/128))))]).cpu().numpy()
    final_indices = np.array([np.flatnonzero(test_owners == i)[-1] for i in splits["test"]])
    report = {"selected_model": selected[0], "class_names": list(class_names), "trials": len(records),
              "split_counts": {k: len(v) for k, v in splits.items()},
              "test_final_contact": score_metrics(test_prob[final_indices], test_y[final_indices]),
              "test_all_prefixes": score_metrics(test_prob, test_y),
              "limitations": ["Same sensor and collection session; no new-person or new-day validation.",
                              "Folder labels assumed correct; unknown gestures and idle are not trained classes.",
                              "Fixed gesture-to-velocity mapping; human speed/force is not regressed.",
                              "Prefix observations from the same trial are correlated; trial count is the independent unit."]}
    if args.continuous:
        report["limitations"].extend([
            "Recent windows from isolated gestures, not recorded uninterrupted direction changes.",
            "Stationary contact is not a learned class; runtime may hold the last confirmed direction until release.",
            "Continuous transition latency/accuracy and real robot path tracking are not established."])
    for fraction in (0.25, 0.5, 0.75):
        ids = [np.flatnonzero(test_owners == owner)[np.argmin(np.abs(fractions[test_owners == owner] - fraction))] for owner in splits["test"]]
        report[f"test_prefix_{int(fraction*100)}pct"] = score_metrics(test_prob[ids], test_y[ids])
    comparison = {}
    for architecture, family_bundle in family_bundles.items():
        family_model = TouchGesturePolicy(family_bundle).to(device).eval()
        with torch.inference_mode():
            family_prob = torch.cat([family_model(torch.tensor(b, device=device))
                for b in np.array_split(test_x, max(1, int(np.ceil(len(test_x)/128))))]).cpu().numpy()
        comparison[architecture] = {"selected_model": family_bundle["selected_model"],
            "test_final_contact": score_metrics(family_prob[final_indices], test_y[final_indices]),
            "test_all_prefixes": score_metrics(family_prob, test_y)}
    save_json(output / "architecture_comparison.json", {"selection_source": "validation only",
        "recommended": selected[0], "architectures": comparison})
    save_json(output / "evaluation.json", report)
    save_json(output / "test_predictions.json", [
        {"path": records[int(owner)]["path"], "true_label": records[int(owner)]["label"],
         "contact_fraction": float(fraction), "prediction": class_names[int(prob.argmax())], "probabilities": prob.tolist()}
        for owner, fraction, prob in zip(test_owners, fractions, test_prob)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix = np.asarray(report["test_final_contact"]["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues", vmin=0)
    for r in range(len(class_names)):
        for c in range(len(class_names)):
            ax.text(c, r, str(matrix[r,c]), ha="center", va="center", color="white" if matrix[r,c] > matrix.max()/2 else "black")
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)), xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted gesture", ylabel="Folder label", title="Held-out trials: final contact frame (before release)")
    fig.tight_layout(); fig.savefig(output / "test_confusion_matrix.png", dpi=180); plt.close(fig)
    print("FINAL", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=ai_resource_path("data", "tactile_recordings"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 43, 79])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--continuous", action="store_true", help="Train on recent 200 ms windows (continuous-touch candidate)")
    parser.add_argument("--gesture-set", choices=("six-directions", "four-swipes", "six-gestures"), default="six-directions")
    parser.add_argument("--architectures", nargs="+", choices=("cnn_gru", "cnn_tcn"), default=["cnn_tcn"])
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience and batch-size must be positive")
    train(args)


if __name__ == "__main__":
    main()

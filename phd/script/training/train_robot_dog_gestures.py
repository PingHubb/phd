# Grouped implementation; the former flat script path remains supported.
"""Compare whole-trial gesture classifiers using chronological trial holdouts.

Run: python3 -m phd.script.train_robot_dog_gestures --output /new/output/folder
The 10 folder names supply labels. Rule-based teaching labels are never used.
No hardware imports, connections, robot commands or changes to source recordings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from phd.dependence.paths import ai_resource_path
from phd.dependence.robot_dog_gesture_model import (
    ARCHITECTURES, CLASS_NAMES, PREPROCESS, GestureClassifier, load_trial, prepare_trial,
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def discover_dataset(root):
    records, frames, seen = [], [], {}
    for folder in sorted(Path(root).glob("session_robot_dog_*")):
        label = folder.name.removeprefix("session_robot_dog_")
        label = {"anitclockwise": "anticlockwise"}.get(label, label)
        if label not in CLASS_NAMES:
            raise ValueError(f"Unknown gesture folder: {folder.name}")
        for path in folder.glob("*.npz"):
            raw, averaged, timestamps, metadata = load_trial(path)
            prepared = prepare_trial(raw, averaged, timestamps)
            fingerprint = hashlib.sha256(raw.tobytes() + averaged.tobytes()).hexdigest()
            if fingerprint in seen:
                raise ValueError(f"Duplicate sensor sequence: {path} and {seen[fingerprint]}")
            seen[fingerprint] = str(path)
            records.append({
                "path": str(path.resolve()), "label": label, "class_index": CLASS_NAMES.index(label),
                "trial_number": int(metadata["trial_number"]), "started_at": float(metadata["started_at"]),
                "frames": len(raw), "duration_sec": float(timestamps[-1] - timestamps[0]),
                "sensor_shape": list(raw.shape[1:]), "completion_reason": metadata.get("completion_reason"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "signal_sha256": fingerprint,
            })
            frames.append(prepared)
    if not records:
        raise ValueError("No robot-dog trials found")
    if len({tuple(record["sensor_shape"]) for record in records}) != 1:
        raise ValueError("Cannot mix sensor dimensions in one model")
    return records, np.stack(frames), np.array([r["class_index"] for r in records])


def split_trials(records):
    """Each class: earliest 60% train, next 20% validation, latest 20% test."""
    result = {name: [] for name in ("train", "validation", "test")}
    for label in CLASS_NAMES:
        ordered = sorted(
            [i for i, record in enumerate(records) if record["label"] == label],
            key=lambda i: (records[i]["started_at"], records[i]["trial_number"], records[i]["path"]),
        )
        if len(ordered) < 20:
            raise ValueError(f"Need at least 20 whole trials for {label}, found {len(ordered)}")
        train_end, valid_end = int(len(ordered) * 0.6), int(len(ordered) * 0.8)
        for name, indices in zip(result, (ordered[:train_end], ordered[train_end:valid_end], ordered[valid_end:])):
            result[name].extend(indices)
    return {name: np.asarray(indices, dtype=int) for name, indices in result.items()}


def classification_metrics(probabilities, labels):
    predicted = np.argmax(probabilities, axis=1)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    np.add.at(confusion, (labels, predicted), 1)
    tp = confusion.diagonal()
    precision = tp / np.maximum(confusion.sum(axis=0), 1)
    recall = tp / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(np.mean(predicted == labels)), "macro_f1": float(f1.mean()),
        "nll": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-8)).mean()),
        "samples": len(labels), "correct": int(np.sum(predicted == labels)),
        "confusion_matrix": confusion.tolist(),
        "per_class": [{"gesture": name, "precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i]), "support": int(confusion[i].sum())} for i, name in enumerate(CLASS_NAMES)],
    }


def augment_batch(x):
    """Label-preserving amplitude/noise/translation/time-warp only; never flip."""
    batch, steps, channels, rows, cols = x.shape
    x = x * torch.empty(batch, 1, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
    x = x + torch.randn_like(x) * 0.025
    # A single sub-taxel displacement shared by every frame in an episode.
    theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(batch, 1, 1)
    theta[:, 0, 2].uniform_(-0.8 / cols, 0.8 / cols)
    theta[:, 1, 2].uniform_(-0.8 / rows, 0.8 / rows)
    theta = theta.repeat_interleave(steps, dim=0)
    flat = x.reshape(batch * steps, channels, rows, cols)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    x = F.grid_sample(flat, grid, padding_mode="border", align_corners=False).reshape(x.shape)
    increments = torch.empty(batch, steps - 1, device=x.device).uniform_(0.65, 1.35)
    phase = F.pad(increments.cumsum(dim=1), (1, 0))
    phase = phase / phase[:, -1:] * (steps - 1)
    low = phase.long().clamp_max(steps - 2)
    weight = (phase - low).view(batch, steps, 1, 1, 1)
    batch_index = torch.arange(batch, device=x.device)[:, None]
    return x[batch_index, low] * (1 - weight) + x[batch_index, low + 1] * weight


@torch.inference_mode()
def evaluate(model, x, y, batch_size=128):
    model.eval()
    probabilities = torch.cat([model(batch).softmax(dim=-1) for batch in x.split(batch_size)]).cpu().numpy()
    return probabilities, classification_metrics(probabilities, y)


def train_candidate(architecture, seed, train_x, train_y, val_x, val_y, args, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = {"architecture": architecture, "classes": len(CLASS_NAMES), "width": 96,
              "dropout": 0.2, "sequence_length": PREPROCESS["sequence_length"]}
    model = GestureClassifier(**config).to(train_x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, stale, history = None, 0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        permutation = torch.randperm(len(train_x), device=train_x.device)
        for index in permutation.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(augment_batch(train_x[index]))
            loss = loss_function(logits, train_y[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(index)
        scheduler.step()
        probabilities, metrics = evaluate(model, val_x, val_y)
        history.append({"epoch": epoch, "training_loss": total_loss / len(train_x),
                        "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"],
                        "validation_nll": metrics["nll"]})
        rank = (metrics["macro_f1"], -metrics["nll"])
        if best is None or rank > best["rank"]:
            best = {"rank": rank, "epoch": epoch, "metrics": metrics, "probabilities": probabilities,
                    "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{architecture} seed={seed} epoch={epoch} val_acc={metrics['accuracy']:.3f} val_f1={metrics['macro_f1']:.3f}", flush=True)
        if stale >= args.patience:
            break
    candidate = {"name": f"{architecture}_seed{seed}", "config": config, "seed": seed,
                 "best_epoch": best["epoch"], "validation": best["metrics"],
                 "parameter_count": sum(p.numel() for p in model.parameters()),
                 "training_seconds": time.monotonic() - started}
    torch.save({**candidate, "state_dict": best["state_dict"]}, output / f"{candidate['name']}.pt")
    save_json(output / f"{candidate['name']}_history.json", history)
    print("CANDIDATE COMPLETE", candidate["name"], candidate["validation"]["accuracy"], flush=True)
    return candidate, best["state_dict"], best["probabilities"]


def calibrated_probabilities(probabilities, temperature):
    logits = np.log(np.maximum(probabilities, 1e-8)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def plot_results(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [name.replace("diagonal_", "diag ").replace("_to_", " → ").replace("_", " ") for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(12, 10))
    matrix = np.array(report["test"]["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "#20303f")
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted gesture"); ax.set_ylabel("Recorded folder label")
    ax.set_title(f"Held-out later trials: {report['test']['correct']}/{report['test']['samples']} correct\n"
                 f"{report['selected_model']} · rows/columns are counts of whole trials")
    fig.tight_layout(); fig.savefig(output / "test_confusion_matrix.png", dpi=150); plt.close(fig)


def train(args):
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, x, y = discover_dataset(args.dataset_root)
    splits = split_trials(records)
    for name, indices in splits.items():
        for i in indices:
            records[i]["split"] = name
    save_json(output / "dataset_manifest.json", {"records": records, "preprocess": PREPROCESS,
        "label_source": "session folder name; no teacher/motion/robot targets used",
        "split_protocol": "within each class: earliest 60% train, middle 20% validation, latest 20% test"})
    train_values = x[splits["train"]]
    mean = train_values.mean(axis=(0, 1, 3, 4))
    std = np.maximum(train_values.std(axis=(0, 1, 3, 4)), 0.05)
    normalized = (x - mean[None, None, :, None, None]) / std[None, None, :, None, None]
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_x = torch.tensor(normalized[splits["train"]], device=device)
    train_y = torch.tensor(y[splits["train"]], dtype=torch.long, device=device)
    val_x = torch.tensor(normalized[splits["validation"]], device=device)
    val_y = y[splits["validation"]]
    save_json(output / "training_config.json", {**vars(args), "resolved_device": str(device),
        "torch_version": str(torch.__version__), "class_names": list(CLASS_NAMES),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "split_counts": {name: len(indices) for name, indices in splits.items()}})
    print(f"DATA: {len(records)} trials; train={len(train_y)}, validation={len(val_y)}, test={len(splits['test'])}; device={device}", flush=True)
    candidates, states, validation_probabilities = [], {}, {}
    for architecture in args.architectures:
        for seed in args.seeds:
            candidate, state, probabilities = train_candidate(architecture, seed, train_x, train_y, val_x, val_y, args, output)
            candidates.append(candidate)
            states[candidate["name"]] = state
            validation_probabilities[candidate["name"]] = probabilities
            save_json(output / "candidate_comparison.json", candidates)
    # Ensemble candidates are fixed before inspecting test data: one per
    # architecture and all members. Selection still uses validation only.
    eligible = [{"name": c["name"], "members": [c["name"]], "validation": c["validation"]} for c in candidates]
    for architecture in (*args.architectures, "all"):
        members = [c["name"] for c in candidates if architecture == "all" or c["config"]["architecture"] == architecture]
        if len(members) < 2:
            continue
        probability = np.mean([validation_probabilities[n] for n in members], axis=0)
        eligible.append({"name": f"{architecture}_ensemble", "members": members,
                         "validation": classification_metrics(probability, val_y)})
    # Prefer fewer networks if macro-F1 and validation NLL are exactly tied.
    selected = max(eligible, key=lambda c: (c["validation"]["macro_f1"], -c["validation"]["nll"], -len(c["members"])))
    validation_probability = np.mean([validation_probabilities[n] for n in selected["members"]], axis=0)
    temperature = min(np.linspace(0.5, 3.0, 101), key=lambda t:
        classification_metrics(calibrated_probabilities(validation_probability, t), val_y)["nll"])
    members = []
    for name in selected["members"]:
        candidate = next(c for c in candidates if c["name"] == name)
        members.append({"name": name, "config": candidate["config"], "state_dict": states[name]})
    bundle = {"format": "pinglab_robot_dog_gesture_v1", "class_names": list(CLASS_NAMES),
        "members": members, "preprocess": dict(PREPROCESS), "sensor_shape": records[0]["sensor_shape"],
        "normalization": {"mean": mean.tolist(), "std": std.tolist()}, "temperature": float(temperature),
        "selected_model": selected["name"], "training_only_normalization": True,
        "prediction_unit": "completed_touch_trial", "robot_control_enabled": False,
        "unknown_class_trained": False}
    torch.save(bundle, output / "best_gesture_model.pt")
    save_json(output / "model_selection.json", {"selected": selected, "eligible": eligible,
        "temperature": float(temperature), "test_used_for_selection": False})
    # Only now evaluate the selected frozen model on the held-out test set.
    test_x = torch.tensor(normalized[splits["test"]], device=device)
    test_y = y[splits["test"]]
    test_probability = []
    for member in members:
        model = GestureClassifier(**member["config"]).to(device)
        model.load_state_dict(member["state_dict"])
        test_probability.append(evaluate(model, test_x, test_y)[0])
    test_probability = calibrated_probabilities(np.mean(test_probability, axis=0), temperature)
    test_metrics = classification_metrics(test_probability, test_y)
    z = 1.96; n = len(test_y); p = test_metrics["accuracy"]
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    report = {"selected_model": selected["name"], "ensemble_members": selected["members"],
        "trials": len(records), "class_names": list(CLASS_NAMES), "validation": selected["validation"],
        "test": test_metrics, "test_accuracy_wilson_95_ci": [center-half, center+half],
        "limits": ["Same collection session and operator; not a new-person/day/sensor test.",
                   "Folder names are assumed correct; human label errors are not independently verified.",
                   "Only 10 known gestures trained; confidence rejection is not validated unknown-gesture detection.",
                   "Whole-trial post-release classifier, not early or continuous gesture recognition.",
                   "Existing robot-velocity policy UI cannot load this separate classification checkpoint."]}
    save_json(output / "evaluation.json", report)
    save_json(output / "test_predictions.json", [
        {"path": records[i]["path"], "true_label": records[i]["label"],
         "prediction": CLASS_NAMES[int(prob.argmax())], "probabilities": prob.tolist()}
        for i, prob in zip(splits["test"], test_probability)])
    plot_results(report, output)
    print("FINAL", json.dumps({"model": selected["name"], "test_accuracy": test_metrics["accuracy"],
          "test_macro_f1": test_metrics["macro_f1"], "correct": test_metrics["correct"],
          "test_samples": len(test_y), "checkpoint": str(output / 'best_gesture_model.pt')}), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=ai_resource_path("data", "tactile_recordings"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 43, 79])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience and batch-size must be positive")
    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Grouped implementation; the former flat script path remains supported.
"""Camera-free goal-intention baseline; grouped evaluation, no hardware access."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import uuid

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from phd.dependence.intention_data import CLASSES, SCHEMA, dataset_report, intention_indices
from phd.dependence.touch_tracking_data import CHANNELS, LAYOUT, save_json, split_trials


def probabilities(logits, temperature=1.):
    x = np.asarray(logits, float)/temperature
    x -= x.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)


def metrics(y, p, threshold=0.):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if not len(y):
        return {"windows": 0, "accuracy": None, "macro_f1": None, "coverage": 0., "accepted_accuracy": None}
    pred = p.argmax(1)
    matrix = np.zeros((4, 4), int)
    np.add.at(matrix, (y, pred), 1)
    f1 = 2*np.diag(matrix)/np.maximum(matrix.sum(0)+matrix.sum(1), 1)
    accepted = p.max(1) >= threshold
    return {"windows": len(y), "accuracy": float((pred == y).mean()), "macro_f1": float(f1.mean()),
            "confusion_true_rows_predicted_columns": matrix.tolist(), "class_support": matrix.sum(1).tolist(),
            "nll": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
            "brier": float(((p-np.eye(4)[y])**2).sum(1).mean()), "coverage": float(accepted.mean()),
            "accepted_accuracy": float((pred[accepted] == y[accepted]).mean()) if accepted.any() else None}


def calibrate(logits, y, *, desired_accuracy=.85, min_accepted=30):
    # All choices use validation only. Overlapping windows are not independent
    # statistical evidence: this is a development heuristic, not a guarantee.
    temperatures = np.geomspace(.25, 8., 49)
    temp = float(min(temperatures, key=lambda t: metrics(y, probabilities(logits, t))["nll"]))
    p = probabilities(logits, temp)
    for threshold in np.linspace(.25, 1., 76):
        result = metrics(y, p, float(threshold))
        if ((p.max(1) >= threshold).sum() >= min_accepted and result["accepted_accuracy"] is not None
                and result["accepted_accuracy"] >= desired_accuracy):
            return temp, float(threshold)
    return temp, 1.01  # No threshold meets the empirical criterion: withhold all.


class IntentionDataset:
    def __init__(self, trials, seq_len, normalization, stride=1):
        from phd.dependence.touch_tracking_model import normalize_frames
        self.trials, self.seq_len = trials, seq_len
        self.frames = [normalize_frames(t["frames"], normalization) for t in trials]
        self.indices = [(i, int(k)) for i, t in enumerate(trials) for k in intention_indices(t, seq_len)[::stride]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, k = self.indices[index]
        return (self.frames[i][k-self.seq_len+1:k+1], self.trials[i]["goals"],
                int(self.trials[i]["labels"][k]))


def infer(model, dataset, batch_size):
    import torch
    from torch.utils.data import DataLoader
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for x, context, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits.append(model(x, context).numpy())
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def temporal_metrics(dataset, p, threshold, *, max_gap=.15):
    """Operator-event-relative early predictions and target-switch response.

    These are NOT times measured from a person's private decision or physical
    touch onset. Missed/insufficient-data events stay in the denominator.
    """
    by_trial = {}
    for j, (i, k) in enumerate(dataset.indices):
        by_trial.setdefault(i, []).append((float(dataset.trials[i]["t"][k]), p[j]))
    prefix = {str(s): {"events": 0, "usable": 0, "correct": 0, "accepted_correct": 0} for s in (.5, 1., 2.)}
    switches = []
    for i, trial in enumerate(dataset.trials):
        predictions = by_trial.get(i, [])
        events = trial["protocol"]["events"]
        for e, event in enumerate(events):
            goal, start = event["goal"], event["t"]
            end = events[e+1]["t"] if e+1 < len(events) else float(trial["t"][-1])
            if goal < 1:
                continue
            rows = [(t, q) for t, q in predictions if start <= t < end]
            for s in (.5, 1., 2.):
                target = start+s
                if target >= end:
                    continue
                result = prefix[str(s)]
                result["events"] += 1
                sample = next(((t, q) for t, q in rows if t >= target), None)
                if sample is not None and sample[0]-target <= max_gap:
                    q = sample[1]
                    correct = int(q.argmax()) == goal
                    result["usable"] += 1
                    result["correct"] += int(correct)
                    result["accepted_correct"] += int(correct and q.max() >= threshold)
            if e == 0 or events[e-1]["goal"] < 1 or events[e-1]["goal"] == goal:
                continue
            stable_start = previous = latency = None
            for t, q in rows:
                if t-start > 5.:
                    break
                good = int(q.argmax()) == goal and float(q.max()) >= threshold
                if not good:
                    stable_start = None
                elif stable_start is None or (previous is not None and t-previous > max_gap):
                    stable_start = t
                if stable_start is not None and t-stable_start >= .2-1e-6:
                    latency = t-start
                    break
                previous = t
            switches.append({"trial": trial["path"], "declared_at": start, "new_goal": goal,
                             "latency_sec": latency, "observable_seconds": min(5., end-start)})
    detected = [s["latency_sec"] for s in switches if s["latency_sec"] is not None]
    return {"prefix_seconds_after_operator_declaration": prefix,
            "switch_events": switches, "switch_count": len(switches), "switch_detected": len(detected),
            "median_detected_switch_latency_sec": float(np.median(detected)) if detected else None,
            "note": "Switch success: accepted correct goal for >=0.2 s, within 5 s or next declaration. Operator timing is approximate."}


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from phd.dependence.intention_model import IntentionNet
    from phd.script.train_touch_tracking import training_normalization

    if args.epochs < 1 or args.batch_size < 1 or args.seq_len < 2 or args.stride < 1:
        raise ValueError("Epochs/batch size/stride must be positive; sequence length >=2.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report, trials = dataset_report(args.root, seq_len=args.seq_len, hz=args.hz,
                                     max_gap=args.max_gap, boundary_margin=args.boundary_margin)
    if not trials or report["invalid_trials"]:
        raise ValueError("Dataset contains missing/invalid trials. Run 'check' and inspect errors first. " + json.dumps(report))
    # A different order is not another harmless numeric sensor configuration.
    signatures = {(tuple(t["metadata"]["sensor_shape"]), t["metadata"].get("sensor_model", ""),
                   t["metadata"].get("reorder_mode", "")) for t in trials}
    if len(signatures) != 1:
        raise ValueError("Use one sensor shape/model/reorder mode in a training run.")
    ids = {t["metadata"]["sensor_id"] for t in trials}
    if len(ids) != 1:
        raise ValueError("First version trains one identified sensor at a time; cross-sensor generalization is not established.")
    splits = split_trials(trials, group_by=args.group_by, seed=args.seed)
    normalization = training_normalization(splits["train"])
    datasets = {name: IntentionDataset(items, args.seq_len, normalization, args.stride if name == "train" else 1)
                for name, items in splits.items()}
    split_details = {}
    for name, ds in datasets.items():
        counts = np.bincount([ds.trials[i]["labels"][k] for i, k in ds.indices], minlength=4)
        if not len(ds) or (counts == 0).any():
            raise ValueError(f"{name} split lacks classes: {dict(zip(CLASSES, counts.tolist()))}. Collect all goals and no-goal in each independent group.")
        split_details[name] = {"class_windows": counts.tolist(), "trials": [
            {"path": t["path"], "participant": t["metadata"]["participant_id"], "session": t["metadata"]["session_id"],
             "raw_hash": t["data_hash"], "protocol_hash": t["protocol_hash"], "goals": t["goals"].tolist()}
            for t in ds.trials]}
    shape, sensor_model, reorder_mode = next(iter(signatures))
    output = Path(args.output) / (datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    print(f"OUTPUT_DIR={output.resolve()}", flush=True)
    save_json(output/"dataset_report.json", report)
    save_json(output/"split.json", {"group_by": args.group_by, "seed": args.seed, "splits": split_details})
    config = {"classes": CLASSES, "channels": CHANNELS, "normalization": normalization,
              "seq_len": args.seq_len, "hz": args.hz, "max_gap": args.max_gap,
              "boundary_margin": args.boundary_margin, "sensor_ids": sorted(ids),
              "sensor_model": sensor_model, "reorder_mode": reorder_mode,
              "robot_commands_enabled": False, "group_by": args.group_by,
              "task": "surface_target_selection", "temperature": 1., "confidence_threshold": 1.01}
    save_json(output/"run.json", {"status": "running", "arguments": vars(args), "config": config})
    results, main_checkpoint = {}, None
    try:
        for mode in ("fusion", "tactile_only", "context_only"):
            torch.manual_seed(args.seed)
            model_config = {"rows": shape[0], "cols": shape[1], "mode": mode}
            model = IntentionNet(**model_config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            counts = np.array(split_details["train"]["class_windows"], float)
            weights = torch.tensor(counts.sum()/(4*counts), dtype=torch.float32)
            loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(args.seed), num_workers=0)
            best, best_epoch, best_state, patience, history = float("inf"), 0, None, 0, []
            for epoch in range(args.epochs):
                model.train()
                total = 0.
                for x, context, y in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(x, context), y, weight=weights)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss; check sensor data.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    total += float(loss.detach())*len(y)
                vl, vy = infer(model, datasets["validation"], args.batch_size)
                score = metrics(vy, probabilities(vl))["nll"]
                history.append({"epoch": epoch+1, "training_loss": total/len(datasets["train"]), "validation_nll": score})
                print(f"{mode}: epoch {epoch+1}/{args.epochs}, validation NLL {score:.4f}", flush=True)
                if score < best-1e-5:
                    best, best_epoch = score, epoch+1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 5:
                    break
            model.load_state_dict(best_state)
            vl, vy = infer(model, datasets["validation"], args.batch_size)
            temperature, threshold = calibrate(vl, vy)
            # Test is evaluated only after freezing model + all thresholds.
            logits, y = infer(model, datasets["test"], args.batch_size)
            p = probabilities(logits, temperature)
            per_trial, per_participant = {}, {}
            for j, (i, _) in enumerate(datasets["test"].indices):
                t = datasets["test"].trials[i]
                per_trial.setdefault(t["path"], []).append(j)
                per_participant.setdefault(t["metadata"]["participant_id"], []).append(j)
            result = {"best_epoch": best_epoch, "temperature": temperature, "confidence_threshold": threshold,
                      "validation": metrics(vy, probabilities(vl, temperature), threshold),
                      "test": metrics(y, p, threshold), "history": history,
                      "test_per_trial": {k: metrics(y[idx], p[idx], threshold) for k, idx in per_trial.items()},
                      "test_per_participant": {k: metrics(y[idx], p[idx], threshold) for k, idx in per_participant.items()},
                      "temporal": temporal_metrics(datasets["test"], p, threshold, max_gap=args.max_gap)}
            results[mode] = result
            mode_config = {**config, "model": model_config, "temperature": temperature, "confidence_threshold": threshold}
            checkpoint = {"schema": SCHEMA, "layout": LAYOUT, "config": mode_config, "model_state": best_state}
            torch.save(checkpoint, output/f"{mode}.pt")
            if mode == "fusion":
                main_checkpoint = checkpoint
            save_json(output/f"{mode}_metrics.json", result)
        layouts = {tuple(np.asarray(t["goals"]).flatten()) for t in trials}
        summary = {"schema": SCHEMA, "status": "complete", "main_model": "fusion (predeclared, not selected by test scores)",
                   "classes": CLASSES, "group_by": args.group_by, "layout_count": len(layouts),
                   "models": {k: {"test": v["test"], "confidence_threshold": v["confidence_threshold"]} for k, v in results.items()},
                   "limitations": ["Predicts human-declared surface target, not mind reading, contact state, physical position or robot commands.",
                                   "Temperature and abstention threshold tuned on validation; empirical confidence, not a safety guarantee.",
                                   "Window scores are correlated. Inspect participant/trial results, not sample count as independent evidence.",
                                   "Three groups is a software minimum, not a sufficient research sample size.",
                                   "Rotate A/B/C assignments; balance goals and vary timing to prevent shortcuts.",
                                   "Synthetic tests validate software only. External target/robot-task validation is still required."]}
        save_json(output/"summary.json", summary)
        # Stable name is confined to THIS immutable run; never overwrites another run.
        torch.save(main_checkpoint, output/"intention_preview.pt")
        save_json(output/"run.json", {"status": "complete", "arguments": vars(args), "config": config})
    except Exception as exc:
        save_json(output/"run.json", {"status": "failed", "error": str(exc), "arguments": vars(args)})
        raise
    print("Main model test: " + json.dumps(results["fusion"]["test"]), flush=True)
    print(f"COMPLETED={output.resolve()}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "train"):
        p = sub.add_parser(name)
        p.add_argument("root")
        p.add_argument("--seq-len", type=int, default=20)
        p.add_argument("--hz", type=float, default=20.)
        p.add_argument("--max-gap", type=float, default=.15)
        p.add_argument("--boundary-margin", type=float, default=.25)
        if name == "train":
            p.add_argument("--output", required=True)
            p.add_argument("--group-by", choices=("participant", "session"), default="participant")
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--batch-size", type=int, default=64)
            p.add_argument("--stride", type=int, default=2)
            p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "check":
        report, _ = dataset_report(args.root, seq_len=args.seq_len, hz=args.hz,
                                   max_gap=args.max_gap, boundary_margin=args.boundary_margin)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["ready_trials"] and not report["invalid_trials"] else 2
    train(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)

#!/usr/bin/env python3
"""Phase 2 robustness benchmark: rule-based pipeline vs trained AI policy.

Replays recorded tactile episodes through BOTH the rule-based DirectFingerMotion
pipeline and the deployed AI_DirectFingerMotion_execution pipeline while
injecting sensor perturbations (Gaussian noise, baseline drift, dead taxels,
gain error). Both are scored against the clean-input teacher command
(intended_velocity_target), producing error-vs-perturbation curves.

The replay drives the real runtime classes with a fake sensor, so the rules
(including push/pull hold counters, grace frames, smoothing) and the AI
(including buffers, normalization, confidence gating) behave exactly as
deployed. Never connects to the robot.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt5.QtCore import QCoreApplication  # noqa: E402

_QT_APP = QCoreApplication.instance() or QCoreApplication([])

from phd.dependence.gesture.gesture_logic_direct_finger_motion import (  # noqa: E402
    AI_DirectFingerMotion_execution,
    DirectFingerMotion,
)

LINEAR_AXES = (0, 1, 2)
MODE_CANONICAL = {
    "stop": "stop",
    "single_finger_swipe": "move",
    "move": "move",
    "normal_swipe": "move",
    "two_finger_swipe": "move",
    "push": "push",
    "two_finger_pull": "pull",
    "pull": "pull",
}


def canonical_mode(name: str, velocity: np.ndarray) -> str:
    mapped = MODE_CANONICAL.get(str(name))
    if mapped is not None:
        return mapped
    return "move" if float(np.linalg.norm(velocity[:3])) > 1e-8 else "stop"


class _FakeSensorData(SimpleNamespace):
    pass


class FakeSensor:
    """Minimal stand-in for MySensor: internal (n_row, n_col) layout arrays."""

    def __init__(self, n_row: int, n_col: int):
        self.n_row = int(n_row)
        self.n_col = int(n_col)
        self._data = _FakeSensorData(
            diffPerData=np.zeros((self.n_row, self.n_col), dtype=np.float32),
            diffPerDataAve=np.zeros((self.n_row, self.n_col), dtype=np.float32),
            frame_sequence=0,
        )

    def set_frame(self, diff: np.ndarray, ave: np.ndarray, frame_sequence: int) -> None:
        self._data.diffPerData = diff
        self._data.diffPerDataAve = ave
        self._data.frame_sequence = int(frame_sequence)


def make_stub_splitter() -> SimpleNamespace:
    return SimpleNamespace(
        robot_api=None,
        ai_selected_frame="tool",
        ai_frame_input=None,
        log_display=None,
    )


# ---------------------------------------------------------------------------
# Perturbation models (all operate on internal-layout (n_row, n_col) arrays).
#
# diffPerDataAve is flipud(rolling mean of diffPerData) inside the sensor
# stack, so every perturbation must appear consistently in both arrays:
# additive fields are averaged over the window and flipped for the ave view.
# ---------------------------------------------------------------------------


class Perturbation:
    name = "identity"

    def reset(self, rng: np.random.Generator, shape: tuple[int, int]) -> None:
        pass

    def apply(self, diff: np.ndarray, ave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return diff, ave


class GaussianNoise(Perturbation):
    name = "gaussian_noise"

    def __init__(self, std: float, window: int):
        self.std = float(std)
        self.window = max(1, int(window))
        self._buffer: list[np.ndarray] = []
        self._rng: np.random.Generator | None = None
        self._shape: tuple[int, int] = (0, 0)

    def reset(self, rng, shape):
        self._rng = rng
        self._shape = shape
        self._buffer = []

    def apply(self, diff, ave):
        if self.std <= 0.0:
            return diff, ave
        noise = self._rng.normal(0.0, self.std, size=self._shape).astype(np.float32)
        self._buffer.append(noise)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        ave_noise = np.flipud(np.mean(self._buffer, axis=0))
        return diff + noise, ave + ave_noise


class BaselineDrift(Perturbation):
    name = "baseline_drift"

    def __init__(self, offset: float):
        # Negative offsets push readings toward the (negative) touch threshold,
        # the direction that makes rule thresholds misfire.
        self.offset = float(offset)

    def apply(self, diff, ave):
        if self.offset == 0.0:
            return diff, ave
        return diff + self.offset, ave + self.offset


class DeadTaxels(Perturbation):
    name = "dead_taxels"

    def __init__(self, fraction: float):
        self.fraction = float(fraction)
        self._mask: np.ndarray | None = None
        self._mask_flipped: np.ndarray | None = None

    def reset(self, rng, shape):
        count = int(round(self.fraction * shape[0] * shape[1]))
        mask = np.zeros(shape, dtype=bool)
        if count > 0:
            flat = rng.choice(shape[0] * shape[1], size=count, replace=False)
            mask.reshape(-1)[flat] = True
        self._mask = mask
        self._mask_flipped = np.flipud(mask)

    def apply(self, diff, ave):
        if self._mask is None or not self._mask.any():
            return diff, ave
        diff = diff.copy()
        ave = ave.copy()
        diff[self._mask] = 0.0
        ave[self._mask_flipped] = 0.0
        return diff, ave


class GainError(Perturbation):
    name = "gain_error"

    def __init__(self, factor: float):
        self.factor = float(factor)

    def apply(self, diff, ave):
        if self.factor == 1.0:
            return diff, ave
        return diff * self.factor, ave * self.factor


# ---------------------------------------------------------------------------
# Episode loading and replay
# ---------------------------------------------------------------------------


def load_episode(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}
        # npz frames are column_major_matrix_view (transposed); restore internal layout
        diff = np.asarray(data["diffPerData"], dtype=np.float32).transpose(0, 2, 1)
        ave = np.asarray(data["diffPerDataAve"], dtype=np.float32).transpose(0, 2, 1)
        intended = np.asarray(data["intended_velocity_target"], dtype=np.float32)
        teacher_sent = np.asarray(data["teacher_velocity_sent"], dtype=np.float32)
        intended_mode = [str(v) for v in np.asarray(data["intended_mode"]).reshape(-1)]
    return {
        "path": path,
        "metadata": metadata,
        "diff": diff,
        "ave": ave,
        "intended": intended,
        "teacher_sent": teacher_sent,
        "intended_mode": intended_mode,
        "frame_count": int(diff.shape[0]),
        "n_row": int(diff.shape[1]),
        "n_col": int(diff.shape[2]),
    }


def _episode_settings(episode: dict) -> dict:
    return dict(episode["metadata"].get("dfm_settings", {}))


def replay_rules(episode: dict, perturbation: Perturbation, seed: int) -> dict:
    sensor = FakeSensor(episode["n_row"], episode["n_col"])
    with contextlib.redirect_stdout(io.StringIO()):
        rules = DirectFingerMotion(make_stub_splitter(), sensor)
    settings = _episode_settings(episode)
    if settings:
        rules.apply_settings(settings, save_to_file=False)
    rules.robot_command_output_enabled = False
    rules.debug_output = False
    rules.motion_ratio_log_enabled = False
    rules.is_running = True
    rules._reset_state()

    rng = np.random.default_rng(seed)
    perturbation.reset(rng, (episode["n_row"], episode["n_col"]))

    frames = episode["frame_count"]
    velocity = np.zeros((frames, 6), dtype=np.float32)
    modes: list[str] = []
    for t in range(frames):
        diff, ave = perturbation.apply(episode["diff"][t], episode["ave"][t])
        sensor.set_frame(diff, ave, t + 1)
        rules.run_step()
        cmd = rules.last_robot_velocity_cmd or [0.0] * 6
        velocity[t] = np.asarray(cmd, dtype=np.float32)
        modes.append(str(rules.current_motion_mode))
    return {"velocity": velocity, "modes": modes}


def make_ai_helper(model_path: Path) -> AI_DirectFingerMotion_execution:
    sensor = FakeSensor(2, 2)
    with contextlib.redirect_stdout(io.StringIO()):
        helper = AI_DirectFingerMotion_execution(make_stub_splitter(), sensor)
    helper.dry_run_predictions_only = True
    # The benchmark drives run_step() synchronously and reads last_prediction
    # right after each step, so inference must run inline (not in background).
    helper.inference_in_background = False
    helper.load_model(str(model_path))
    if not helper.model_loaded:
        raise RuntimeError(f"Could not load AI checkpoint: {model_path}")
    return helper


def replay_ai(
    episode: dict,
    helper: AI_DirectFingerMotion_execution,
    perturbation: Perturbation,
    seed: int,
) -> dict:
    sensor = FakeSensor(episode["n_row"], episode["n_col"])
    helper.my_sensor = sensor
    settings = _episode_settings(episode)
    if settings:
        helper.apply_settings(settings, save_to_file=False)
    helper.robot_command_output_enabled = False
    helper.debug_output = False
    helper.motion_ratio_log_enabled = False
    # Lift the deployment safety clip so it does not mask model error: allow
    # up to the teacher's own maximum command speed.
    max_speed = max(
        abs(float(settings.get("robot_speed", 0.1))),
        abs(float(settings.get("push_speed", 0.1))),
        abs(float(settings.get("pull_speed", 0.1))),
    ) * max(1.0, float(settings.get("max_speed_ratio", 1.0)))
    helper.max_linear_speed = max_speed
    helper.is_running = True
    helper._reset_state()
    helper._reset_execution_buffers()
    helper._reset_execution_runtime_state()

    rng = np.random.default_rng(seed)
    perturbation.reset(rng, (episode["n_row"], episode["n_col"]))

    frames = episode["frame_count"]
    velocity = np.zeros((frames, 6), dtype=np.float32)
    modes: list[str] = []
    for t in range(frames):
        diff, ave = perturbation.apply(episode["diff"][t], episode["ave"][t])
        sensor.set_frame(diff, ave, t + 1)
        helper.run_step()
        prediction = helper.last_prediction
        if prediction is not None:
            velocity[t] = np.asarray(prediction["velocity_sent"], dtype=np.float32)
            modes.append(str(prediction["mode"]))
        else:
            modes.append("stop")
    helper.is_running = False
    return {"velocity": velocity, "modes": modes}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def score(pred: dict, episode: dict) -> dict:
    target = episode["intended"]
    velocity = pred["velocity"]
    error = velocity[:, :3] - target[:, :3]
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = np.mean(np.abs(error), axis=0)

    target_modes = [
        canonical_mode(name, target[idx]) for idx, name in enumerate(episode["intended_mode"])
    ]
    pred_modes = [
        canonical_mode(name, velocity[idx]) for idx, name in enumerate(pred["modes"])
    ]
    agreement = float(np.mean([p == t for p, t in zip(pred_modes, target_modes)]))

    moving_true = np.linalg.norm(target[:, :3], axis=1) > 1e-8
    moving_pred = np.linalg.norm(velocity[:, :3], axis=1) > 0.005
    stop_frames = int(np.count_nonzero(~moving_true))
    move_frames = int(np.count_nonzero(moving_true))
    stop_false_move_count = int(np.count_nonzero(moving_pred[~moving_true]))
    move_false_stop_count = int(np.count_nonzero(~moving_pred[moving_true]))
    stop_false_move = stop_false_move_count / stop_frames if stop_frames else 0.0
    move_false_stop = move_false_stop_count / move_frames if move_frames else 0.0
    mode_correct_count = int(sum(p == t for p, t in zip(pred_modes, target_modes)))

    return {
        "rmse_linear": rmse,
        "mae_vx": float(mae[0]),
        "mae_vy": float(mae[1]),
        "mae_vz": float(mae[2]),
        "mode_agreement": agreement,
        "stop_false_move_rate": stop_false_move,
        "move_false_stop_rate": move_false_stop,
        "frames": int(target.shape[0]),
        "linear_squared_error_sum": float(np.sum(error**2)),
        "linear_value_count": int(error.size),
        "absolute_error_sum": np.sum(np.abs(error), axis=0).astype(float).tolist(),
        "mode_correct_count": mode_correct_count,
        "stop_frames": stop_frames,
        "move_frames": move_frames,
        "stop_false_move_count": stop_false_move_count,
        "move_false_stop_count": move_false_stop_count,
    }


def combine_scores(per_episode: list[dict]) -> dict:
    total_frames = sum(item["frames"] for item in per_episode)
    squared_error_sum = sum(item["linear_squared_error_sum"] for item in per_episode)
    linear_value_count = sum(item["linear_value_count"] for item in per_episode)
    absolute_error_sum = np.sum(
        [item["absolute_error_sum"] for item in per_episode], axis=0
    )
    stop_frames = sum(item["stop_frames"] for item in per_episode)
    move_frames = sum(item["move_frames"] for item in per_episode)
    return {
        "frames": total_frames,
        "rmse_linear": float(
            np.sqrt(squared_error_sum / max(1, linear_value_count))
        ),
        "mae_vx": float(absolute_error_sum[0] / max(1, total_frames)),
        "mae_vy": float(absolute_error_sum[1] / max(1, total_frames)),
        "mae_vz": float(absolute_error_sum[2] / max(1, total_frames)),
        "mode_agreement": float(
            sum(item["mode_correct_count"] for item in per_episode)
            / max(1, total_frames)
        ),
        "stop_false_move_rate": float(
            sum(item["stop_false_move_count"] for item in per_episode)
            / max(1, stop_frames)
        ),
        "move_false_stop_rate": float(
            sum(item["move_false_stop_count"] for item in per_episode)
            / max(1, move_frames)
        ),
    }


def stable_perturbation_seed(
    sweep_name: str,
    level: float,
    episode_index: int,
    base_seed: int,
) -> int:
    payload = (
        f"{int(base_seed)}|{sweep_name}|{float(level):.9g}|{int(episode_index)}"
    ).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


def default_trials(
    dataset_root: Path,
    model_path: Path,
    *,
    allow_validation_fallback: bool = False,
) -> list[Path]:
    """Return the checkpoint's held-out test trials.

    Older checkpoints have no test split. Reusing validation data is allowed
    only through an explicit exploratory flag so reported robustness numbers
    cannot silently use data that influenced early stopping.
    """
    test_basenames: set[str] = set()
    val_basenames: set[str] = set()
    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        for entry in config.get("test_files", []):
            test_basenames.add(Path(str(entry)).name)
        for entry in config.get("val_files", []):
            val_basenames.add(Path(str(entry)).name)
    except Exception:
        pass

    sessions = sorted(
        path
        for path in dataset_root.glob("session_*")
        if path.is_dir() and not path.name.startswith("_")
    )
    test_matched: list[Path] = []
    validation_matched: list[Path] = []
    for session in sessions:
        session_trials = sorted(session.glob("trial_*.npz"))
        if not session_trials:
            continue
        test_matched.extend(
            trial for trial in session_trials if trial.name in test_basenames
        )
        validation_matched.extend(
            trial for trial in session_trials if trial.name in val_basenames
        )
    if test_matched:
        return test_matched
    if allow_validation_fallback:
        return validation_matched
    return []


def build_sweeps(args: argparse.Namespace, window: int) -> dict[str, list[tuple[float, Perturbation]]]:
    return {
        "gaussian_noise": [
            (std, GaussianNoise(std, window)) for std in args.noise_levels
        ],
        "baseline_drift": [
            (offset, BaselineDrift(-abs(offset))) for offset in args.drift_levels
        ],
        "dead_taxels": [
            (fraction, DeadTaxels(fraction)) for fraction in args.dead_levels
        ],
        "gain_error": [
            (factor, GainError(factor)) for factor in args.gain_levels
        ],
    }


def sanity_check_rules_replay(episodes: list[dict]) -> dict:
    """Zero-perturbation rules replay must reproduce the recorded teacher."""
    results = []
    for episode in episodes:
        replayed = replay_rules(episode, Perturbation(), seed=0)
        recorded = episode["teacher_sent"]
        error = np.abs(replayed["velocity"] - recorded)
        exact = float(np.mean(np.all(error < 1e-5, axis=1)))
        results.append(
            {
                "trial": episode["path"].name,
                "exact_match_fraction": exact,
                "max_abs_error": float(error.max()),
                "mean_abs_error": float(error.mean()),
            }
        )
        print(
            f"  sanity {episode['path'].name}: exact-match {exact * 100.0:.1f}% "
            f"| max abs err {error.max():.5f}"
        )
    return {"per_trial": results}


def run_benchmark(args: argparse.Namespace) -> Path:
    model_path = args.model.expanduser()
    dataset_root = args.dataset_root.expanduser()
    if args.trials:
        trial_paths = [Path(trial).expanduser() for trial in args.trials]
    else:
        trial_paths = default_trials(
            dataset_root,
            model_path,
            allow_validation_fallback=args.allow_validation_fallback,
        )
    if not trial_paths:
        raise FileNotFoundError(
            "No independent checkpoint test trials were found. Supply "
            "--trials explicitly, train a checkpoint with --test-trials, or "
            "use --allow-validation-fallback for exploratory analysis only."
        )

    print("Phase 2 robustness benchmark: rules vs AI")
    print(f"Model: {model_path}")
    for trial in trial_paths:
        print(f"Trial: {trial}")

    episodes = [load_episode(path) for path in trial_paths]
    if args.max_frames > 0:
        for episode in episodes:
            limit = min(episode["frame_count"], int(args.max_frames))
            for key in ("diff", "ave", "intended", "teacher_sent"):
                episode[key] = episode[key][:limit]
            episode["intended_mode"] = episode["intended_mode"][:limit]
            episode["frame_count"] = limit

    window = int(episodes[0]["metadata"].get("sensor_average_window_size", 3) or 3)
    helper = make_ai_helper(model_path)

    print("\nSanity check (zero perturbation, rules replay vs recorded teacher):")
    sanity = sanity_check_rules_replay(episodes)

    sweeps = build_sweeps(args, window)
    rows: list[dict] = []
    detail: dict[str, dict] = {}
    started = time.time()
    for sweep_name, levels in sweeps.items():
        detail[sweep_name] = {"levels": [], "rules": [], "ai": []}
        for level, perturbation in levels:
            for pipeline_name, replay_fn in (("rules", replay_rules), ("ai", replay_ai)):
                per_episode = []
                for episode_idx, episode in enumerate(episodes):
                    seed = stable_perturbation_seed(
                        sweep_name,
                        float(level),
                        episode_idx,
                        args.seed,
                    )
                    if pipeline_name == "rules":
                        pred = replay_fn(episode, perturbation, seed)
                    else:
                        pred = replay_fn(episode, helper, perturbation, seed)
                    per_episode.append(score(pred, episode))
                combined = combine_scores(per_episode)
                combined.update(
                    {
                        "perturbation": sweep_name,
                        "level": float(level),
                        "pipeline": pipeline_name,
                    }
                )
                rows.append(combined)
                detail[sweep_name][pipeline_name].append(combined)
                print(
                    f"{sweep_name:>15} level={level:<6g} {pipeline_name:>5}: "
                    f"RMSE={combined['rmse_linear']:.5f} | mode-agree={combined['mode_agreement']:.3f}"
                )
            detail[sweep_name]["levels"].append(float(level))

    elapsed = time.time() - started
    print(f"\nBenchmark finished in {elapsed:.1f}s")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.expanduser() / f"robustness_benchmark_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "perturbation",
        "level",
        "pipeline",
        "frames",
        "rmse_linear",
        "mae_vx",
        "mae_vy",
        "mae_vz",
        "mode_agreement",
        "stop_false_move_rate",
        "move_false_stop_rate",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)

    summary = {
        "model": str(model_path),
        "trials": [str(path) for path in trial_paths],
        "sanity_check": sanity,
        "results": rows,
        "elapsed_sec": elapsed,
        "seed": int(args.seed),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_plots:
        _save_plots(output_dir, detail)

    print(f"Results: {output_dir}")
    return output_dir


def _save_plots(output_dir: Path, detail: dict[str, dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")
        return

    axis_labels = {
        "gaussian_noise": "Noise std (diff units)",
        "baseline_drift": "Baseline drift magnitude (toward threshold)",
        "dead_taxels": "Dead taxel fraction",
        "gain_error": "Gain factor",
    }

    for metric, ylabel, filename in (
        ("rmse_linear", "Velocity RMSE (linear axes)", "rmse_vs_perturbation.png"),
        ("mode_agreement", "Mode agreement", "mode_agreement_vs_perturbation.png"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (sweep_name, data) in zip(axes.reshape(-1), detail.items()):
            levels = data["levels"]
            for pipeline_name, style in (("rules", "o-"), ("ai", "s-")):
                values = [entry[metric] for entry in data[pipeline_name]]
                ax.plot(levels, values, style, label=pipeline_name, linewidth=1.6)
            ax.set_title(sweep_name.replace("_", " "))
            ax.set_xlabel(axis_labels.get(sweep_name, "level"))
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"{ylabel} under sensor perturbations (lower RMSE / higher agreement is better)")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> int:
    default_model = (
        PACKAGE_ROOT
        / "resource"
        / "ai"
        / "models"
        / "ai_direct_finger_motion"
        / "latest_cnn_gru_model_10x10.pt"
    )
    default_dataset = PACKAGE_ROOT / "resource" / "ai" / "data" / "ai_direct_finger_motion"
    default_output = PACKAGE_ROOT / "resource" / "ai" / "models" / "ai_direct_finger_motion"

    parser = argparse.ArgumentParser(description="Robustness benchmark: rule-based pipeline vs AI policy.")
    parser.add_argument("--model", type=Path, default=default_model, help="AI checkpoint (.pt).")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument(
        "--trials",
        nargs="*",
        default=[],
        help="Explicit independent .npz trial paths. Default: checkpoint test trials.",
    )
    parser.add_argument(
        "--allow-validation-fallback",
        action="store_true",
        help=(
            "Allow old checkpoints to reuse validation trials. Exploratory only; "
            "do not report these results as an independent test."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames per trial (0 = all).")
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="*",
        default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument(
        "--drift-levels",
        type=float,
        nargs="*",
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    )
    parser.add_argument(
        "--dead-levels",
        type=float,
        nargs="*",
        default=[0.0, 0.02, 0.05, 0.1, 0.2],
    )
    parser.add_argument(
        "--gain-levels",
        type=float,
        nargs="*",
        default=[0.5, 0.7, 0.85, 1.0, 1.15, 1.3],
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260623,
        help="Deterministic base seed for all perturbation trials.",
    )
    args = parser.parse_args()

    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

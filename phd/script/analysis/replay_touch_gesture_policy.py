"""Replay held-out sensor trials through Touch Policy in dry-run mode only.

Uses recorded timing and the actual UI execution adapter, with a synthetic
sensor and no robot API. Source recordings and the model are never changed.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PyQt5.QtCore import QCoreApplication

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from phd.dependence.gesture.gesture_logic_direct_finger_motion import AI_DirectFingerMotion_execution
from phd.dependence.robot_dog_gesture_model import load_trial
from phd.dependence.sensor_data import SensorDataBuffer


def replay(checkpoint, manifest, output):
    app = QCoreApplication.instance() or QCoreApplication([])
    torch.set_num_threads(2)
    sensor = SimpleNamespace(n_row=10, n_col=10, is_connected=True, current_reorder_mode="row_to_col",
                             _sensor_calibration_in_progress=False, _data=SensorDataBuffer(10, 10, 3))
    helper = AI_DirectFingerMotion_execution(SimpleNamespace(robot_api=None, ai_selected_frame="tool"), sensor)
    helper.device = torch.device("cpu")
    helper.dry_run_predictions_only = True
    helper.inference_in_background = False
    if not helper.load_model(checkpoint):
        raise ValueError("Could not load the touch-gesture policy")
    # Fail loudly if a future edit accidentally enters the hardware-send path.
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline replay must never send a robot command")
    helper._send_robot_velocity_execution = forbidden
    rows, costs = [], []
    records = json.loads(Path(manifest).read_text())["records"]
    for record in (r for r in records if r["split"] == "test"):
        raw, avg, timestamps, _ = load_trial(record["path"])
        helper.is_running = False
        helper._reset_execution_buffers(); helper._reset_state(); helper._reset_execution_runtime_state()
        sensor._data = data = SensorDataBuffer(10, 10, 3)
        clock = [100.0]
        data.frame_monotonic_sec = clock[0]
        commands, first_motion, wrong = [], None, []
        contact = np.flatnonzero(np.any(avg < -3, axis=(1, 2)))
        contact_start = timestamps[contact[0]]
        with patch("time.monotonic", lambda: clock[0]):
            helper.is_running = True
            helper._start_touch_gesture_stream()

            def feed(raw_frame, avg_frame, stamp):
                clock[0] = float(stamp)
                data.frame_sequence += 1
                data.frame_monotonic_sec = clock[0]
                data.diffPerData, data.diffPerDataAve = raw_frame, avg_frame
                data.completed_frame_snapshot = SimpleNamespace(frame_sequence=data.frame_sequence,
                    frame_monotonic_sec=clock[0], diffPerData=raw_frame.copy(), diffPerDataAve=avg_frame.copy())
                for callback in tuple(data._frame_listeners):
                    callback(data)
                started = time.perf_counter()
                helper.run_step()
                costs.append(time.perf_counter() - started)
                if not helper.is_running:
                    raise AssertionError(f"Replay stopped: {helper.last_prediction}")

            zero = np.zeros((10, 10), np.float32)
            for i in range(10):
                feed(zero, zero, 100.02 + i * 0.02)
            start = 101.0
            # Continuous hand-away samples bridge to the trial's first frame.
            for stamp in np.arange(100.22, start, 0.02):
                feed(zero, zero, stamp)
            for i, (r, a, t) in enumerate(zip(raw, avg, timestamps)):
                feed(r, a, start + t - timestamps[0])
                pred = helper.last_prediction
                if not np.any(a < -3) and any(pred["velocity_sent"]):
                    raise AssertionError("Nonzero velocity after finger release")
                if pred["mode"] != "stop":
                    commands.append(pred["gesture"])
                    if first_motion is None:
                        first_motion = float(t - contact_start)
                    if pred["gesture"] != record["label"]:
                        wrong.append({"frame": i, "time_sec": float(t), "gesture": pred["gesture"]})
            feed(zero, zero, clock[0] + 0.02)
            assert not any(helper.last_prediction["velocity_sent"])
        rows.append({"path": record["path"], "label": record["label"], "motion_predictions": len(commands),
                     "wrong_motion_predictions": wrong, "first_motion_sec_after_contact": first_motion,
                     "recognized_before_release": record["label"] in commands, "release_stopped": True})
    helper.is_running = False
    helper.shutdown()
    latencies = [r["first_motion_sec_after_contact"] for r in rows if r["first_motion_sec_after_contact"] is not None]
    report = {"mode": "offline dry-run; no robot API; synchronous CPU inference",
              "test_trials": len(rows), "recognized_before_release": sum(r["recognized_before_release"] for r in rows),
              "trials_with_wrong_motion": sum(bool(r["wrong_motion_predictions"]) for r in rows),
              "wrong_motion_predictions": sum(len(r["wrong_motion_predictions"]) for r in rows),
              "total_motion_predictions": sum(r["motion_predictions"] for r in rows),
              "release_stop_passed": all(r["release_stopped"] for r in rows),
              "first_motion_latency_sec_median": float(np.median(latencies)) if latencies else None,
              "first_motion_latency_sec_p95": float(np.percentile(latencies, 95)) if latencies else None,
              "processing_ms_p95": float(np.percentile(costs, 95) * 1000),
              "trials": rows}
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output must be a new file")
    replay(args.checkpoint, args.manifest, args.output)


if __name__ == "__main__":
    main()

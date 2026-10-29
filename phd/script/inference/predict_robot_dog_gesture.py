# Grouped implementation; the former flat script path remains supported.
"""Recognise one saved touch trial; never connect to or command a robot."""

import argparse
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from phd.dependence.robot_dog_gesture_model import GesturePredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to best_gesture_model.pt")
    parser.add_argument("--trial", required=True, help="A completed touch-trial .npz file")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--min-margin", type=float, default=0.15)
    args = parser.parse_args()
    if not (0 <= args.min_confidence <= 1 and 0 <= args.min_margin <= 1):
        parser.error("Confidence and margin must lie between 0 and 1")
    torch.set_num_threads(4)
    predictor = GesturePredictor(args.model, device=args.device)
    try:
        result = predictor.predict_file(
            args.trial, min_confidence=args.min_confidence, min_margin=args.min_margin,
        )
    except (ValueError, KeyError) as exc:
        result = {"gesture": "invalid_trial", "accepted": False, "reason": str(exc),
                  "robot_command_sent": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

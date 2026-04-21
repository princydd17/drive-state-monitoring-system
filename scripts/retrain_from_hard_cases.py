#!/usr/bin/env python3
"""
Retrain pipeline using filtered hard cases.

Flow:
1) Load hard cases from JSONL
2) Filter low-value noise
3) Convert to feature rows with weak labels
4) Merge with base training set
5) Train v2 model and save improvement summary
"""

import argparse
import json
import os
import subprocess
from datetime import datetime

import pandas as pd


FEATURES = [
    "left_ear",
    "right_ear",
    "avg_ear",
    "ear_variance",
    "blink_frequency",
    "eye_closure_duration",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mar",
    "yawn_frequency",
]


def weak_label_from_state(state: str, detection: dict) -> str:
    if state == "distraction_transient":
        return "distraction"
    if state in {"fatigue_risk_high", "fatigue_risk_low"}:
        if detection.get("yawn", {}).get("detected"):
            return "yawn"
        return "drowsy"
    if state == "no_face":
        return "no_face"
    return "alert"


def to_feature_row(case: dict):
    bundle = case.get("signal_bundle", {})
    detection = case.get("detection", {})
    head_pose = bundle.get("head_pose", [0.0, 0.0, 0.0])
    ear = float(bundle.get("ear", 0.0))
    row = {
        "left_ear": ear,
        "right_ear": ear,
        "avg_ear": ear,
        "ear_variance": 0.0,
        "blink_frequency": float(bundle.get("blink_frequency", 0.0)),
        "eye_closure_duration": float(bundle.get("eye_closure_duration", 0.0)),
        "head_pitch": float(head_pose[0]) if len(head_pose) > 0 else 0.0,
        "head_yaw": float(head_pose[1]) if len(head_pose) > 1 else 0.0,
        "head_roll": float(head_pose[2]) if len(head_pose) > 2 else 0.0,
        "mar": float(bundle.get("mar", 0.0)),
        "yawn_frequency": float(bundle.get("yawn_frequency", 0.0)),
        "label": weak_label_from_state(case.get("state", "alert"), detection),
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Retrain model from hard-case feedback.")
    parser.add_argument("--base-dataset", required=True, help="Base feature CSV (with label column).")
    parser.add_argument("--hard-cases", default="data/hard_cases/hard_cases.jsonl", help="Hard-case JSONL path.")
    parser.add_argument("--output-dataset", default="data/v2/features.csv", help="Merged output dataset path.")
    parser.add_argument("--output-model-dir", default="models/v2", help="Output model directory.")
    parser.add_argument("--summary", default="reports/retrain_summary.json", help="Summary report path.")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Max confidence for low-confidence cases.")
    args = parser.parse_args()

    if not os.path.exists(args.base_dataset):
        raise ValueError(f"Base dataset not found: {args.base_dataset}")
    if not os.path.exists(args.hard_cases):
        raise ValueError(f"Hard cases file not found: {args.hard_cases}")

    base_df = pd.read_csv(args.base_dataset)
    hard_rows = []
    with open(args.hard_cases, "r") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            low_conf = float(case.get("ml_confidence", 1.0)) < args.min_confidence
            disagreement = bool(case.get("model_disagreement", False))
            # Keep only disagreement or low-confidence fallback cases.
            if not (disagreement or low_conf):
                continue
            hard_rows.append(to_feature_row(case))

    if not hard_rows:
        raise ValueError("No qualifying hard cases found after filtering.")

    hard_df = pd.DataFrame(hard_rows)
    merged_df = pd.concat([base_df, hard_df], ignore_index=True)
    os.makedirs(os.path.dirname(args.output_dataset), exist_ok=True)
    merged_df.to_csv(args.output_dataset, index=False)

    # Retrain v2 model using existing training script.
    train_cmd = [
        "python",
        "scripts/train_model.py",
        "--input",
        args.output_dataset,
        "--output-dir",
        args.output_model_dir,
    ]
    subprocess.run(train_cmd, check=True)

    metadata_path = os.path.join(args.output_model_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        new_meta = json.load(f)
    prev_f1 = None
    new_f1 = None
    try:
        new_f1 = max(v["f1_macro"] for v in new_meta["results"].values())
    except Exception:
        new_f1 = None

    summary = {
        "created_at": datetime.utcnow().isoformat(),
        "base_dataset": args.base_dataset,
        "hard_cases_file": args.hard_cases,
        "data_added": int(len(hard_df)),
        "merged_rows": int(len(merged_df)),
        "prev_f1": prev_f1,
        "new_f1": new_f1,
        "output_model_dir": args.output_model_dir,
        "output_dataset": args.output_dataset,
    }
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

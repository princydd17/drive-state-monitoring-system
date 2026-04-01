#!/usr/bin/env python3
"""
Evaluate model predictions from CSV.

Expected CSV columns:
  - ground_truth (required)
  - prediction (required)
Optional:
  - timestamp (unix seconds)
  - fps (if timestamp not present, used for false alerts/hour)
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def false_alerts_per_hour(df, gt_col, pred_col, positive_label, timestamp_col=None, fps_col=None):
    false_alerts = int(((df[pred_col] == positive_label) & (df[gt_col] != positive_label)).sum())

    if timestamp_col and timestamp_col in df.columns and len(df) > 1:
        total_seconds = float(df[timestamp_col].max() - df[timestamp_col].min())
    elif fps_col and fps_col in df.columns and len(df) > 0:
        mean_fps = float(df[fps_col].mean())
        total_seconds = len(df) / max(mean_fps, 1e-6)
    else:
        # Conservative fallback when no timing info exists.
        total_seconds = float(len(df) / 30.0)

    total_hours = max(total_seconds / 3600.0, 1e-9)
    return false_alerts / total_hours


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification predictions.")
    parser.add_argument("--input", required=True, help="CSV with ground_truth and prediction columns.")
    parser.add_argument("--gt-col", default="ground_truth", help="Ground-truth column name.")
    parser.add_argument("--pred-col", default="prediction", help="Prediction column name.")
    parser.add_argument(
        "--positive-label",
        default="drowsy",
        help="Label used for false-alerts/hour (default: drowsy).",
    )
    parser.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--fps-col", default="fps", help="FPS column name.")
    parser.add_argument("--output", default="reports/eval_report.json", help="Output JSON report path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.gt_col not in df.columns or args.pred_col not in df.columns:
        raise ValueError(f"CSV must contain '{args.gt_col}' and '{args.pred_col}' columns.")

    y_true = df[args.gt_col].astype(str)
    y_pred = df[args.pred_col].astype(str)
    labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    fah = false_alerts_per_hour(
        df,
        args.gt_col,
        args.pred_col,
        args.positive_label,
        args.timestamp_col if args.timestamp_col in df.columns else None,
        args.fps_col if args.fps_col in df.columns else None,
    )

    result = {
        "input": args.input,
        "labels": labels,
        "accuracy": acc,
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "false_alerts_per_hour": float(fah),
        "confusion_matrix": {"labels": labels, "values": matrix},
        "per_class": {k: v for k, v in report.items() if k in labels},
    }

    print(json.dumps(result, indent=2))

    # Save report
    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()

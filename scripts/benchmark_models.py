#!/usr/bin/env python3
"""
Benchmark rule-only vs ML-only vs fused predictions from one CSV.

Expected CSV columns:
  - ground_truth
  - rule_pred
  - ml_pred
  - fused_pred (hybrid/fused)
Optional:
  - timestamp
  - fps
"""

import argparse
import json
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def false_alerts_per_hour(df, gt_col, pred_col, positive_label):
    false_alerts = int(((df[pred_col] == positive_label) & (df[gt_col] != positive_label)).sum())
    if "timestamp" in df.columns and len(df) > 1:
        seconds = float(df["timestamp"].max() - df["timestamp"].min())
    elif "fps" in df.columns and len(df) > 0:
        seconds = float(len(df) / max(df["fps"].mean(), 1e-6))
    else:
        seconds = float(len(df) / 30.0)
    hours = max(seconds / 3600.0, 1e-9)
    return float(false_alerts / hours)


def evaluate_variant(df, gt_col, pred_col, positive_label):
    y_true = df[gt_col].astype(str)
    y_pred = df[pred_col].astype(str)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "false_alerts_per_hour": false_alerts_per_hour(df, gt_col, pred_col, positive_label),
    }

def state_flips_per_minute(df, pred_col):
    if len(df) < 2:
        return 0.0
    flips = (df[pred_col].astype(str) != df[pred_col].astype(str).shift(1)).sum() - 1
    flips = max(0, int(flips))
    if "timestamp" in df.columns and len(df) > 1:
        seconds = float(df["timestamp"].max() - df["timestamp"].min())
    elif "fps" in df.columns and len(df) > 0:
        seconds = float(len(df) / max(df["fps"].mean(), 1e-6))
    else:
        seconds = float(len(df) / 30.0)
    minutes = max(seconds / 60.0, 1e-9)
    return float(flips / minutes)


def main():
    parser = argparse.ArgumentParser(description="Benchmark rule-only vs ML-only vs fused.")
    parser.add_argument("--input", required=True, help="CSV file with predictions for all variants.")
    parser.add_argument("--gt-col", default="ground_truth", help="Ground-truth column.")
    parser.add_argument("--rule-col", default="rule_pred", help="Rule-only prediction column.")
    parser.add_argument("--ml-col", default="ml_pred", help="ML-only prediction column.")
    parser.add_argument("--fused-col", default="fused_pred", help="Fused prediction column.")
    parser.add_argument("--positive-label", default="drowsy", help="Positive label for FA/hour.")
    parser.add_argument("--output", default="reports/benchmark_report.json", help="Output report path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = [args.gt_col, args.rule_col, args.ml_col, args.fused_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = {
        "input": args.input,
        "rule_only": evaluate_variant(df, args.gt_col, args.rule_col, args.positive_label),
        "ml_only": evaluate_variant(df, args.gt_col, args.ml_col, args.positive_label),
        "fused": evaluate_variant(df, args.gt_col, args.fused_col, args.positive_label),
    }

    # Rank by F1 macro then FA/hour
    ranked = sorted(
        [
            ("rule_only", result["rule_only"]),
            ("ml_only", result["ml_only"]),
            ("fused", result["fused"]),
        ],
        key=lambda x: (-x[1]["f1_macro"], x[1]["false_alerts_per_hour"]),
    )
    result["ranking"] = [name for name, _ in ranked]
    result["stability"] = {
        "rule_only_flips_per_min": state_flips_per_minute(df, args.rule_col),
        "ml_only_flips_per_min": state_flips_per_minute(df, args.ml_col),
        "fused_flips_per_min": state_flips_per_minute(df, args.fused_col),
    }

    print(json.dumps(result, indent=2))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()

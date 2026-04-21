#!/usr/bin/env python3
"""
Cross-dataset experiment runner.

Runs train-on-A/test-on-B and optional bidirectional evaluation.
Produces structured JSON reports with generalization metrics.
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from benchmark_models import false_alerts_per_hour, state_flips_per_minute
from train_model import FEATURES


def ensure_columns(df, label_col):
    required = FEATURES + [label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def run_single_experiment(train_csv, test_csv, train_name, test_name, output_model_dir, label_col="label"):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    ensure_columns(train_df, label_col)
    ensure_columns(test_df, label_col)

    # Train same candidates as train_model.py and choose best by train-val split macro F1.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    X = train_df[FEATURES].astype(float)
    y = train_df[label_col].astype(str)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "logreg": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
    }
    best_name = None
    best_model = None
    best_f1 = -1.0
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred_val = model.predict(X_val)
        f1 = f1_score(y_val, pred_val, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    os.makedirs(output_model_dir, exist_ok=True)
    model_path = os.path.join(output_model_dir, "model.joblib")
    joblib.dump({"model": best_model, "feature_names": FEATURES, "label_col": label_col}, model_path)

    # Evaluate on unseen dataset
    X_test = test_df[FEATURES].astype(float)
    y_true = test_df[label_col].astype(str)
    y_pred = best_model.predict(X_test).astype(str)

    eval_df = pd.DataFrame(
        {
            "ground_truth": y_true,
            "prediction": y_pred,
            "timestamp": list(range(len(y_true))),  # pseudo time axis for comparable rates
            "fps": [30] * len(y_true),
        }
    )

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "false_alerts_per_hour": float(false_alerts_per_hour(eval_df, "ground_truth", "prediction", "drowsy")),
        "state_flips_per_min": float(state_flips_per_minute(eval_df, "prediction")),
        "per_class_recall": {k: float(v["recall"]) for k, v in report.items() if isinstance(v, dict) and "recall" in v},
    }

    result = {
        "experiment": f"{train_name.lower()}_to_{test_name.lower()}",
        "train_dataset": train_name,
        "test_dataset": test_name,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "model_dir": output_model_dir,
        "best_model": best_name,
        "metrics": metrics,
    }
    return result, eval_df


def main():
    parser = argparse.ArgumentParser(description="Run cross-dataset generalization experiments.")
    parser.add_argument("--train", required=True, help="Training feature CSV path.")
    parser.add_argument("--test", required=True, help="Testing feature CSV path.")
    parser.add_argument("--train-name", default="TRAIN", help="Display name for train dataset.")
    parser.add_argument("--test-name", default="TEST", help="Display name for test dataset.")
    parser.add_argument("--label-col", default="label", help="Label column name in both CSVs.")
    parser.add_argument("--bidirectional", action="store_true", help="Also run reverse direction test->train.")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports.")
    parser.add_argument("--notes", default="", help="Optional notes tag (e.g., synthetic dry-run validation).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary = {
        "created_at": datetime.utcnow().isoformat(),
        "notes": args.notes,
        "experiments": [],
    }

    exp1_model_dir = os.path.join("models", f"exp_{args.train_name.lower()}_to_{args.test_name.lower()}")
    exp1, eval_df1 = run_single_experiment(
        train_csv=args.train,
        test_csv=args.test,
        train_name=args.train_name,
        test_name=args.test_name,
        output_model_dir=exp1_model_dir,
        label_col=args.label_col,
    )
    summary["experiments"].append(exp1)
    eval_path_1 = os.path.join(args.output_dir, f"pred_{exp1['experiment']}_{stamp}.csv")
    eval_df1.to_csv(eval_path_1, index=False)
    exp1["prediction_csv"] = eval_path_1

    if args.bidirectional:
        exp2_model_dir = os.path.join("models", f"exp_{args.test_name.lower()}_to_{args.train_name.lower()}")
        exp2, eval_df2 = run_single_experiment(
            train_csv=args.test,
            test_csv=args.train,
            train_name=args.test_name,
            test_name=args.train_name,
            output_model_dir=exp2_model_dir,
            label_col=args.label_col,
        )
        summary["experiments"].append(exp2)
        eval_path_2 = os.path.join(args.output_dir, f"pred_{exp2['experiment']}_{stamp}.csv")
        eval_df2.to_csv(eval_path_2, index=False)
        exp2["prediction_csv"] = eval_path_2

    out_json = os.path.join(args.output_dir, f"cross_dataset_experiments_{stamp}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved experiment summary to {out_json}")


if __name__ == "__main__":
    main()

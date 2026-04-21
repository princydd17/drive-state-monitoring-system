#!/usr/bin/env python3
"""
Train a real model on extracted feature CSV.

Required columns:
  left_ear,right_ear,avg_ear,ear_variance,blink_frequency,eye_closure_duration,
  head_pitch,head_yaw,head_roll,mar,yawn_frequency,label
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def main():
    parser = argparse.ArgumentParser(description="Train v1/v2 driver-state ML model.")
    parser.add_argument("--input", required=True, help="Feature CSV path.")
    parser.add_argument("--output-dir", default="models/v1", help="Model artifact directory.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = FEATURES + [args.label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES].astype(float)
    y = df[args.label_col].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    models = {
        "logreg": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=args.seed))]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=args.seed, n_jobs=-1
        ),
    }

    results = {}
    best_name = None
    best_f1 = -1.0
    best_model = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        macro_f1 = f1_score(y_test, pred, average="macro")
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(macro_f1),
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_model = model

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.joblib")
    joblib.dump({"model": best_model, "feature_names": FEATURES, "label_col": args.label_col}, model_path)

    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "version": os.path.basename(args.output_dir.rstrip("/")) or "v1",
        "input": args.input,
        "features": FEATURES,
        "label_col": args.label_col,
        "best_model": best_name,
        "results": results,
        "artifact": model_path,
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps({"best_model": best_name, "f1_macro": best_f1, "artifact": model_path}, indent=2))
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()

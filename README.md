# Driver State Monitoring Platform

A real-time driver monitoring system focused on perception, decision-making, and measurable safety performance.
Built as a modular platform that combines vision signals, temporal modeling, and hybrid ML scoring.

## Overview

This project is designed as a full pipeline rather than a standalone detection script. It integrates signal extraction, model inference, decision logic, and backend logging into a single system.

* **Perception:** facial landmarks, eye aspect ratio (EAR), mouth aspect ratio (MAR), head pose
* **Decision logic:** session calibration, smoothing, hysteresis, and alert policies
* **Scoring:** rule-based score, ML score, hybrid score, and fatigue accumulation
* **Platform layer:** API logging, dataset tooling, evaluation, and benchmarking

## Current Capabilities

* Real-time monitoring using webcam input
* Detection of drowsiness, yawning, and distraction
* Session-based calibration for personalized thresholds
* Hybrid scoring combining rule-based logic and ML predictions
* Fatigue accumulation over time using decay-based modeling
* Structured backend events using a standardized `driver_state` schema
* Reproducible evaluation and benchmarking workflow

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Start backend (optional, enables logging):

```bash
python start_backend.py
```

Run the monitoring system:

```bash
python start_monitor.py
```

## Runtime Options

```bash
python src/start_monitor.py --sensitivity high
python src/start_monitor.py --no-ai
python src/start_monitor.py --no-api
```

## Dataset Collection

Collect labeled video clips for training and evaluation:

```bash
python scripts/collect_dataset.py --output data/v1
```

Controls:

* `1` alert
* `2` drowsy
* `3` yawn
* `4` distraction
* `5` no_face
* `R` stop and save clip
* `Q` quit

Outputs:

* Clips: `data/v1/clips/<label>/`
* Metadata: `data/v1/labels.csv`

## External Dataset Feature Extraction

Extract training features from dataset videos (for NTHU/UTA-style folders):

```bash
python scripts/extract_features_from_dataset.py --dataset-root <dataset_path> --output data/v1/features_dataset.csv
```

Optional label mapping with regex rules:

```bash
python scripts/extract_features_from_dataset.py --dataset-root <dataset_path> --label-map data/label_map.json --output data/v1/features_dataset.csv
```

Example `data/label_map.json`:

```json
{
  "drowsy|sleepy|fatigue": "drowsy",
  "yawn": "yawn",
  "distract|phone|text": "distraction",
  "alert|normal": "alert"
}
```

## Model Training

Train a model from extracted feature data:

```bash
python scripts/train_model.py --input <feature_csv> --output-dir models/v1
```

The training pipeline:

* compares Logistic Regression and Random Forest
* selects the best model based on macro F1 score
* exports:

  * `model.joblib`
  * `metadata.json` (features and metrics)

### Expected Features

* `left_ear`, `right_ear`, `avg_ear`, `ear_variance`
* `blink_frequency`, `eye_closure_duration`
* `head_pitch`, `head_yaw`, `head_roll`
* `mar`, `yawn_frequency`
* `label`

## Retraining from Hard Cases

Build a v2 dataset/model using logged hard cases:

```bash
python scripts/retrain_from_hard_cases.py --base-dataset <feature_csv> --hard-cases data/hard_cases/hard_cases.jsonl --output-dataset data/v2/features.csv --output-model-dir models/v2
```

The retraining script:
* filters for disagreement and low-confidence cases
* maps cases into the training feature schema
* merges with the base dataset
* retrains using `scripts/train_model.py`
* writes summary to `reports/retrain_summary.json`

## Results

### Pipeline Validation (Dry Run)

We validated the end-to-end retraining pipeline using a controlled dataset to confirm that ingestion, filtering, retraining, and versioned model updates work correctly.

* Hard cases added: **105**
* Dataset size after merge: **605 samples**
* Models trained: Logistic Regression (v1), Random Forest (v2)
* Artifacts generated:
  * `models/v1/`, `models/v2/`
  * `reports/retrain_summary.json`

Observed behavior:

* v1 achieved **macro F1 = 1.0**, indicating overfitting on a clean synthetic dataset
* v2 achieved **macro F1 = 0.906**, reflecting a more regularized model

This run validates the data flywheel:
hard-case capture -> dataset augmentation -> retraining -> versioned model deployment

This behavior highlights the importance of evaluating on realistic, noisy data rather than relying on clean synthetic datasets.

### Real-World Evaluation (In Progress)

The same pipeline is being applied to live monitoring sessions for realistic performance measurement.

Metrics tracked:

* Macro F1 score
* False alerts per hour
* State transitions per minute (stability)
* Time-to-detection for fatigue events

Current evaluation targets:

* Reduce false alerts through hard-case retraining
* Improve stability (fewer rapid state flips)
* Maintain performance under varied lighting, pose, and occlusion

## Evaluation

Evaluate prediction performance:

```bash
python scripts/evaluate.py --input data/v1/sample_predictions.csv --output reports/sample_eval_report.json
```

Required:

* `ground_truth`
* `prediction`

Optional:

* `timestamp`, `fps`
* `scenario` (enables per-scenario breakdown)

## Benchmarking

Compare rule-based, ML-based, and hybrid approaches:

```bash
python scripts/benchmark_models.py --input data/v1/sample_benchmark.csv --output reports/sample_benchmark_report.json
```

Includes:

* Accuracy, precision, recall, F1
* False alerts per hour
* State transitions per minute (stability)
* Overall ranking

## Cross-Dataset Generalization

Run a train-on-A, test-on-B experiment:

```bash
python scripts/experiment_split.py --train data/nthu_features.csv --test data/uta_features.csv --train-name NTHU --test-name UTA
```

Run bidirectional evaluation (A->B and B->A):

```bash
python scripts/experiment_split.py --train data/nthu_features.csv --test data/uta_features.csv --train-name NTHU --test-name UTA --bidirectional
```

Outputs:
* experiment JSON summary with macro F1, false alerts/hour, and state flips/min
* per-experiment prediction CSVs in `reports/`

Cross-dataset evaluation is used to assess robustness under distribution shift and approximate real-world deployment conditions.

### Example Output (Dry Run)

Below is a sample output structure from a synthetic dry run, included to illustrate the experiment format and reporting schema:

```json
{
  "created_at": "2026-04-21T03:11:06",
  "notes": "synthetic dry-run validation",
  "experiments": [
    {
      "experiment": "features_v1_to_features_v2",
      "train_dataset": "features_v1",
      "test_dataset": "features_v2",
      "model_selected": "random_forest",
      "metrics": {
        "accuracy": 0.85,
        "f1_macro": 0.82,
        "false_alerts_per_hour": 4.7,
        "state_flips_per_min": 6.2
      }
    }
  ]
}
```

> This example demonstrates output structure only. Metrics from synthetic data are not used for performance claims.

### Real-Data Interpretation Checklist

When running experiments on real datasets (for example, NTHU -> UTA), focus on:

* **Generalization gap**  
  Expect performance to drop when evaluating on unseen datasets. This reflects real-world distribution shift.

* **False alerts per hour**  
  Lower values indicate more usable systems in practice.

* **State stability (flips/min)**  
  High-frequency state changes often indicate noisy or unstable predictions.

* **Per-class recall**  
  Check whether critical states (for example, drowsy) are consistently detected.

* **Confidence + fallback behavior**  
  Verify that low-confidence predictions trigger rule-based fallback as expected.

## Session Timeline Plot

Generate a timeline plot for one session:

```bash
python scripts/plot_session_timeline.py --input <monitor_stats_json> --output reports/session_plot.png
```

Includes:
* hybrid risk score
* fatigue score
* ML confidence overlay
* fallback trigger markers

## Core Components

* `src/core/driver_monitor.py` – main runtime pipeline and decision logic
* `src/vision/vision_engine.py` – landmark detection and feature extraction
* `src/ai/hybrid_scorer.py` – rule + ML score fusion
* `src/ai/ai_detector.py` – model inference
* `backend/app.py` – API and event handling
* `scripts/collect_dataset.py` – dataset collection
* `scripts/train_model.py` – model training
* `scripts/evaluate.py` – evaluation metrics
* `scripts/benchmark_models.py` – benchmarking


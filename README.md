# Driver State Monitoring Platform

Real-time Driver State Monitoring Platform with computer vision, ML scoring, and a decision layer for risk-aware alerts and backend telemetry.

## Project Positioning

This is a **Driver State Monitoring Platform (Real-Time ML + Decision Layer)**.

It is positioned as a system platform, not a single-script detector:
- Perception layer: face, landmarks, EAR, MAR, head pose
- Intelligence layer: temporal features, calibrated thresholds, ML/fused inference
- Decision layer: system state, risk score, and policy actions
- Platform layer: backend event schema, analytics-ready logging, evaluation and benchmarking tools

## What this platform does

- Detects drowsiness from eye closure (EAR)
- Detects yawning from mouth opening (MAR)
- Detects distraction from head pose direction
- Shows live alerts on screen and optional audio alerts
- Logs events to a Flask backend (optional)

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start backend (optional):

```bash
python start_backend.py
```

3. Start monitor:

```bash
python start_monitor.py
```

## Useful commands

- Run monitor with high sensitivity:

```bash
python src/start_monitor.py --sensitivity high
```

- Run monitor without AI:

```bash
python src/start_monitor.py --no-ai
```

- Run monitor without backend logging:

```bash
python src/start_monitor.py --no-api
```

## Dataset collection (v1)

Collect labeled clips from webcam:

```bash
python scripts/collect_dataset.py --output data/v1
```

Controls:

- `1` alert
- `2` drowsy
- `3` yawn
- `4` distraction
- `5` no_face
- `R` stop/save clip
- `Q` quit

Output:

- Clips: `data/v1/clips/<label>/`
- Labels: `data/v1/labels.csv`

## Evaluation

Evaluate one prediction file:

```bash
python scripts/evaluate.py --input data/v1/sample_predictions.csv --output reports/sample_eval_report.json
```

Required CSV columns:

- `ground_truth`
- `prediction`

Optional:

- `timestamp`
- `fps`

## Benchmark (rule vs ML vs fused)

```bash
python scripts/benchmark_models.py --input data/v1/sample_benchmark.csv --output reports/sample_benchmark_report.json
```

Required CSV columns:

- `ground_truth`
- `rule_pred`
- `ml_pred`
- `fused_pred`

## Main files

- `src/core/driver_monitor.py` main detection pipeline
- `src/vision/vision_engine.py` face/landmark/pose processing
- `src/ai/ai_detector.py` ML detector and ensemble
- `backend/app.py` API and database logging
- `scripts/collect_dataset.py` data collection
- `scripts/evaluate.py` evaluation metrics
- `scripts/benchmark_models.py` model benchmarking


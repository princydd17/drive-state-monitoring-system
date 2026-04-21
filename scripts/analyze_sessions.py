#!/usr/bin/env python3
"""
Analyze monitoring session outputs and compare baseline vs stabilized behavior.
"""

import argparse
import json
from typing import Dict, List


def load_session(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def state_flips_per_min(timeline: List[Dict]) -> float:
    if len(timeline) < 2:
        return 0.0
    flips = 0
    prev_state = timeline[0].get("state", "unknown")
    for item in timeline[1:]:
        state = item.get("state", "unknown")
        if state != prev_state:
            flips += 1
        prev_state = state

    t0 = float(timeline[0].get("timestamp", 0.0))
    t1 = float(timeline[-1].get("timestamp", t0))
    minutes = max((t1 - t0) / 60.0, 1e-9)
    return flips / minutes


def false_alerts_per_hour(stats: Dict, timeline: List[Dict]) -> float:
    alerts = float(stats.get("alerts_triggered", 0))
    if timeline:
        t0 = float(timeline[0].get("timestamp", 0.0))
        t1 = float(timeline[-1].get("timestamp", t0))
        hours = max((t1 - t0) / 3600.0, 1e-9)
    else:
        duration = float(stats.get("duration", 0.0))
        hours = max(duration / 3600.0, 1e-9)
    return alerts / hours


def compute_metrics(session_data: Dict) -> Dict:
    stats = session_data.get("detection_stats", {})
    timeline = session_data.get("timeline", [])
    return {
        "false_alerts_per_hour": float(false_alerts_per_hour(stats, timeline)),
        "state_flips_per_min": float(state_flips_per_min(timeline)),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline vs stabilized monitoring sessions.")
    parser.add_argument("--baseline", required=True, help="Path to baseline monitor_stats JSON.")
    parser.add_argument("--stabilized", required=True, help="Path to stabilized monitor_stats JSON.")
    parser.add_argument("--output", default="reports/session_comparison.json", help="Output JSON path.")
    args = parser.parse_args()

    base = load_session(args.baseline)
    stab = load_session(args.stabilized)

    base_m = compute_metrics(base)
    stab_m = compute_metrics(stab)

    # Higher flips/alerts are worse; reduction is improvement.
    false_alert_reduction_pct = 0.0
    if base_m["false_alerts_per_hour"] > 0:
        false_alert_reduction_pct = (
            (base_m["false_alerts_per_hour"] - stab_m["false_alerts_per_hour"])
            / base_m["false_alerts_per_hour"]
        ) * 100.0

    stability_improvement_pct = 0.0
    if base_m["state_flips_per_min"] > 0:
        stability_improvement_pct = (
            (base_m["state_flips_per_min"] - stab_m["state_flips_per_min"])
            / base_m["state_flips_per_min"]
        ) * 100.0

    result = {
        "baseline": base_m,
        "stabilized": stab_m,
        "improvement": {
            "false_alert_reduction_pct": float(false_alert_reduction_pct),
            "stability_improvement_pct": float(stability_improvement_pct),
        },
    }

    print(json.dumps(result, indent=2))
    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved comparison report to {args.output}")


if __name__ == "__main__":
    main()

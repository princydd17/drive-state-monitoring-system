#!/usr/bin/env python3
"""
Generate session timeline plot from monitor stats JSON.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot risk/fatigue timeline from monitor stats file.")
    parser.add_argument("--input", required=True, help="Path to monitor_stats_*.json")
    parser.add_argument("--output", default="reports/session_plot.png", help="Output plot file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)
    timeline = data.get("timeline", [])
    if not timeline:
        raise ValueError("No timeline found in input stats file.")

    df = pd.DataFrame(timeline).dropna(subset=["timestamp"])
    t0 = float(df["timestamp"].iloc[0])
    df["t_sec"] = df["timestamp"].astype(float) - t0

    plt.figure(figsize=(12, 5))
    plt.plot(df["t_sec"], df["risk_score"], label="hybrid risk score", linewidth=2)
    plt.plot(df["t_sec"], df["fatigue_score"], label="fatigue score", linewidth=2)
    if "ml_confidence" in df.columns:
        plt.plot(df["t_sec"], df["ml_confidence"], label="ml confidence", linewidth=1.5, linestyle="--", alpha=0.9)
    if "fallback_triggered" in df.columns:
        fallback_points = df[df["fallback_triggered"] == True]
        if not fallback_points.empty:
            plt.scatter(
                fallback_points["t_sec"],
                fallback_points["risk_score"],
                marker="o",
                s=20,
                alpha=0.7,
                label="fallback triggered",
            )

    # Mark non-alert state transitions
    transitions = df[df["state"] != "alert"]
    if not transitions.empty:
        plt.scatter(transitions["t_sec"], transitions["risk_score"], marker="x", label="state transition", alpha=0.8)

    plt.title("Session Timeline: Risk, Fatigue, and Model Confidence")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved timeline plot to {args.output}")


if __name__ == "__main__":
    main()

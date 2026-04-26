#!/usr/bin/env python3
"""
Session-level explanation utilities for risk interpretation and action policy.
"""

from typing import Dict, List, Tuple


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_ratio(value: float, baseline: float, max_ratio: float = 2.0) -> float:
    if baseline <= 1e-6:
        return 0.0
    ratio = float(value) / float(baseline)
    return _clamp01((ratio - 1.0) / max(max_ratio - 1.0, 1e-6))


def _normalize_inverse_gap(value: float, threshold: float, max_gap_fraction: float = 0.45) -> float:
    if threshold <= 1e-6:
        return 0.0
    gap = max(0.0, float(threshold) - float(value))
    normalized = gap / max(float(threshold) * max_gap_fraction, 1e-6)
    return _clamp01(normalized)


def compute_factor_scores(metrics: Dict, thresholds: Dict) -> Dict[str, float]:
    """
    Compute normalized per-signal contribution scores in [0, 1].
    """
    ear = float(metrics.get("ear", 0.0))
    mar = float(metrics.get("mar", 0.0))
    blink_frequency = float(metrics.get("blink_frequency", 0.0))
    eye_closure_duration = float(metrics.get("eye_closure_duration", 0.0))
    yawn_frequency = float(metrics.get("yawn_frequency", 0.0))
    head_pose = metrics.get("head_pose", (0.0, 0.0, 0.0))
    pitch, yaw, _ = head_pose

    ear_thr = float(thresholds.get("ear", 0.25))
    mar_thr = float(thresholds.get("mar", 0.5))

    head_tilt_magnitude = max(abs(float(pitch)), abs(float(yaw)))

    return {
        "prolonged_eye_closure": _clamp01(eye_closure_duration / 2.0),
        "high_blink_frequency": _clamp01(blink_frequency / 35.0),
        "low_eye_aspect_ratio": _normalize_inverse_gap(ear, ear_thr),
        "mouth_opening_or_yawn_pattern": max(
            _normalize_ratio(mar, mar_thr, max_ratio=2.1),
            _clamp01(yawn_frequency / 5.0),
        ),
        "head_tilt_deviation": _clamp01(head_tilt_magnitude / 25.0),
    }


def top_factor_labels(factor_scores: Dict[str, float], limit: int = 3, min_score: float = 0.2) -> List[str]:
    """
    Return top non-trivial human-readable factor labels.
    """
    ranked: List[Tuple[str, float]] = sorted(
        factor_scores.items(), key=lambda item: item[1], reverse=True
    )
    selected = [name for name, score in ranked if score >= min_score][:limit]
    if not selected:
        return ["no strong fatigue factors"]
    return [name.replace("_", " ") for name in selected]


def recommended_action(risk_score: float, fatigue_score: float, state: str) -> str:
    """
    Policy mapping from risk context to user-facing recommended action.
    """
    if state == "no_face":
        return "check_camera"
    if risk_score >= 0.8 or fatigue_score >= 0.8:
        return "take_break"
    if risk_score >= 0.6 or fatigue_score >= 0.6:
        return "stay_alert"
    return "none"

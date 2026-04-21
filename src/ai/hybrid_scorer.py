#!/usr/bin/env python3
"""
Hybrid scorer for combining rule-based and ML-based risk signals.
"""

from dataclasses import dataclass


@dataclass
class HybridWeights:
    ml_weight: float = 0.6
    rule_weight: float = 0.4


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def combine_scores(rule_score: float, ml_score: float, weights: HybridWeights) -> float:
    """Compute normalized hybrid score."""
    raw = (weights.ml_weight * ml_score) + (weights.rule_weight * rule_score)
    return clamp01(raw)

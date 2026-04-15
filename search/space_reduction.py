"""Placeholder search-space reduction utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def select_key_features(
    sensitivity_scores: dict[int, float],
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[int]:
    """Select key features by a simple placeholder ranking."""
    ranked = sorted(sensitivity_scores.items(), key=lambda item: item[1], reverse=True)
    features = [idx for idx, score in ranked if threshold is None or score >= threshold]
    if top_k is not None:
        return features[:top_k]
    return features


def filter_candidate_methods(method_scores: dict[str, float], top_r: int = 3) -> list[str]:
    """Keep the top scoring methods as a placeholder pruning strategy."""
    ranked = sorted(method_scores.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:top_r]]


def group_features(
    X: np.ndarray,
    feature_scores: dict[int, float],
    **kwargs: Any,
) -> dict[str, list[int]]:
    """Return a coarse placeholder grouping for later structured search."""
    _ = (X, kwargs)
    return {
        "high_priority": [idx for idx, score in feature_scores.items() if score > 0.5],
        "low_priority": [idx for idx, score in feature_scores.items() if score <= 0.5],
    }

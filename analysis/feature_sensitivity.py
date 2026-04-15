"""Placeholder interfaces for structural sensitivity analysis."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_feature_sensitivity(
    X_missing: np.ndarray,
    X_imputed_dict: dict[str, np.ndarray],
    clusterer: str,
    **kwargs: Any,
) -> dict[int, float]:
    """Return feature-level sensitivity scores.

    Future implementations are expected to quantify how much each feature's
    imputation affects cluster assignments, separation, and stability.
    """
    _ = (X_missing, X_imputed_dict, clusterer, kwargs)
    return {}


def compute_method_compatibility(
    X_missing: np.ndarray,
    X_imputed_dict: dict[str, np.ndarray],
    clusterer: str,
    **kwargs: Any,
) -> dict[str, float]:
    """Return placeholder method compatibility scores for future ranking logic."""
    _ = (X_missing, X_imputed_dict, clusterer, kwargs)
    return {name: 0.0 for name in X_imputed_dict}

"""Statistics helpers for current and future evaluation modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.validation import ensure_numeric_matrix


def compute_missing_rate(X: np.ndarray | pd.DataFrame) -> float:
    """Return overall missing ratio."""
    array = ensure_numeric_matrix(X)
    return float(np.isnan(array).mean())


def compute_column_missing_rates(X: np.ndarray | pd.DataFrame) -> dict[int, float]:
    """Return per-column missing ratios."""
    array = ensure_numeric_matrix(X)
    return {idx: float(rate) for idx, rate in enumerate(np.isnan(array).mean(axis=0))}


def compare_basic_statistics(
    X_before: np.ndarray | pd.DataFrame,
    X_after: np.ndarray | pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Summarize how basic numeric moments changed after imputation."""
    before = ensure_numeric_matrix(X_before)
    after = ensure_numeric_matrix(X_after)
    return {
        "mean_abs_shift": _metric_delta(np.nanmean(before, axis=0), np.mean(after, axis=0)),
        "std_abs_shift": _metric_delta(np.nanstd(before, axis=0), np.std(after, axis=0)),
        "min_abs_shift": _metric_delta(np.nanmin(before, axis=0), np.min(after, axis=0)),
        "max_abs_shift": _metric_delta(np.nanmax(before, axis=0), np.max(after, axis=0)),
    }


def validate_imputed_result(X: np.ndarray | pd.DataFrame) -> dict[str, int | bool]:
    """Check whether an imputed result is fully numeric and finite."""
    array = ensure_numeric_matrix(X)
    return {
        "remaining_nan": int(np.isnan(array).sum()),
        "has_inf": bool(np.isinf(array).any()),
        "is_valid": bool((not np.isnan(array).any()) and (not np.isinf(array).any())),
    }


def _metric_delta(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    diff = np.abs(np.asarray(before, dtype=float) - np.asarray(after, dtype=float))
    return {
        "mean": float(np.mean(diff)),
        "max": float(np.max(diff)),
    }

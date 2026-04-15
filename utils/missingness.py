"""Reusable helpers for injecting synthetic missingness into numeric matrices."""

from __future__ import annotations

import numpy as np


def inject_mcar_missing(
    X: np.ndarray,
    missing_rate: float,
    random_state: int,
    *,
    protect_full_row_col: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject MCAR missingness while optionally preventing all-missing rows/columns."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D numeric array.")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1).")

    rng = np.random.default_rng(random_state)
    missing_mask = rng.random(X.shape) < missing_rate

    if protect_full_row_col and X.size > 0:
        for col in range(X.shape[1]):
            if missing_mask[:, col].all():
                missing_mask[rng.integers(0, X.shape[0]), col] = False
        for row in range(X.shape[0]):
            if missing_mask[row, :].all():
                missing_mask[row, rng.integers(0, X.shape[1])] = False

    X_missing = np.array(X, copy=True)
    X_missing[missing_mask] = np.nan
    return X_missing, missing_mask


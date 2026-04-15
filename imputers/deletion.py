"""Deletion-based handling for missing data."""

from __future__ import annotations

from typing import Any

import numpy as np

from imputers.base import BaseImputer


class DeletionImputer(BaseImputer):
    """Remove rows that contain missing values.

    This baseline is intentionally included because listwise deletion is still a
    common benchmark in missing-data studies. Unlike true imputers, it changes
    the number of rows, so downstream experiments should account for the reduced
    sample size explicitly.
    """

    name = "deletion"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rows_retained_: int | None = None
        self.rows_removed_: int | None = None

    def _fit(self, X: np.ndarray) -> None:
        keep_mask = ~np.isnan(X).any(axis=1)
        self.rows_retained_ = int(np.sum(keep_mask))
        self.rows_removed_ = int(X.shape[0] - self.rows_retained_)
        if self.rows_retained_ == 0:
            raise ValueError("deletion removed all rows because every row contains missing values.")

    def _transform(self, X: np.ndarray) -> np.ndarray:
        keep_mask = ~np.isnan(X).any(axis=1)
        X_kept = X[keep_mask]
        if X_kept.shape[0] == 0:
            raise ValueError("deletion removed all rows because every row contains missing values.")
        return X_kept

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "rows_retained": self.rows_retained_,
                "rows_removed": self.rows_removed_,
            }
        )
        return params

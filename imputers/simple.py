"""Simple numeric imputers built from deterministic statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from imputers.base import BaseImputer


class _SklearnSimpleImputer(BaseImputer):
    strategy: str = "mean"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._imputer: SimpleImputer | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._imputer = SimpleImputer(strategy=self.strategy)
        self._imputer.fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._imputer is None:
            raise RuntimeError("Simple imputer backend is not fitted.")
        return self._imputer.transform(X)


class MeanImputer(_SklearnSimpleImputer):
    name = "mean"
    strategy = "mean"


class MedianImputer(_SklearnSimpleImputer):
    name = "median"
    strategy = "median"


class MostFrequentImputer(_SklearnSimpleImputer):
    name = "most_frequent"
    strategy = "most_frequent"


class ConstantValueImputer(BaseImputer):
    name = "constant_value"

    def __init__(self, *, fill_value: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fill_value = float(fill_value)

    def _fit(self, X: np.ndarray) -> None:
        return None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_work = np.array(X, dtype=float, copy=True)
        X_work[np.isnan(X_work)] = self.fill_value
        return X_work

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({"fill_value": self.fill_value})
        return params


class ConstantZeroImputer(ConstantValueImputer):
    name = "constant_zero"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(fill_value=0.0, **kwargs)


class _ColumnStatisticImputer(BaseImputer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.statistics_: np.ndarray | None = None

    def _compute_statistics(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _fit(self, X: np.ndarray) -> None:
        self.statistics_ = self._compute_statistics(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.statistics_ is None:
            raise RuntimeError("Column statistics are not fitted.")
        X_work = np.array(X, dtype=float, copy=True)
        missing_mask = np.isnan(X_work)
        for col_idx in range(X_work.shape[1]):
            X_work[missing_mask[:, col_idx], col_idx] = self.statistics_[col_idx]
        return X_work


class MinImputer(_ColumnStatisticImputer):
    name = "min"

    def _compute_statistics(self, X: np.ndarray) -> np.ndarray:
        return np.nanmin(X, axis=0)


class MaxImputer(_ColumnStatisticImputer):
    name = "max"

    def _compute_statistics(self, X: np.ndarray) -> np.ndarray:
        return np.nanmax(X, axis=0)


class RandomSampleImputer(BaseImputer):
    name = "random_sample"

    def __init__(self, *, random_state: int | None = 0, **kwargs: Any) -> None:
        super().__init__(random_state=random_state, **kwargs)
        self._rng = np.random.default_rng(random_state)
        self._observed_values: list[np.ndarray] = []

    def _fit(self, X: np.ndarray) -> None:
        observed_values: list[np.ndarray] = []
        for col_idx in range(X.shape[1]):
            observed = X[~np.isnan(X[:, col_idx]), col_idx]
            if observed.size == 0:
                raise ValueError(
                    f"Column {col_idx} has no observed values for random sampling."
                )
            observed_values.append(observed)
        self._observed_values = observed_values

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not self._observed_values:
            raise RuntimeError("Observed value cache is not fitted.")
        X_work = np.array(X, dtype=float, copy=True)
        missing_mask = np.isnan(X_work)
        for col_idx in range(X_work.shape[1]):
            n_missing = int(missing_mask[:, col_idx].sum())
            if n_missing:
                sampled = self._rng.choice(self._observed_values[col_idx], size=n_missing, replace=True)
                X_work[missing_mask[:, col_idx], col_idx] = sampled
        return X_work


class ColumnInterpolationImputer(BaseImputer):
    """Approximate interpolation baseline for generic numeric matrices."""

    name = "interpolate"

    def __init__(self, *, method: str = "linear", limit_direction: str = "both", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.method = method
        self.limit_direction = limit_direction
        self._fallback_means: np.ndarray | None = None

    def _fit(self, X: np.ndarray) -> None:
        self._fallback_means = np.nanmean(X, axis=0)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._fallback_means is None:
            raise RuntimeError("Interpolation fallback statistics are not fitted.")
        frame = pd.DataFrame(X)
        interpolated = frame.interpolate(
            method=self.method,
            axis=0,
            limit_direction=self.limit_direction,
        )
        X_work = interpolated.to_numpy(dtype=float)
        missing_mask = np.isnan(X_work)
        for col_idx in range(X_work.shape[1]):
            X_work[missing_mask[:, col_idx], col_idx] = self._fallback_means[col_idx]
        return X_work

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({"method": self.method, "limit_direction": self.limit_direction})
        return params

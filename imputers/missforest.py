"""MissForest-style imputer wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from imputers.base import BaseImputer


class MissForestImputer(BaseImputer):
    """MissForest approximation using IterativeImputer + RandomForestRegressor.

    The original MissForest algorithm is iterative and tree-based. To keep the
    project lightweight and directly runnable, this implementation uses sklearn's
    IterativeImputer with a RandomForestRegressor backend as a practical
    approximation.
    """

    name = "missforest"

    initial_strategy_vals = ["mean", "median", "most_frequent", "constant"]
    imputation_order_vals = ["ascending", "descending", "roman", "arabic", "random"]

    def __init__(
        self,
        *,
        n_estimators: int = 50,
        max_iter: int = 20,
        initial_strategy: str = "mean",
        imputation_order: str = "ascending",
        fill_value: float = 0.0,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if initial_strategy not in self.initial_strategy_vals:
            raise ValueError(
                f"initial_strategy must be one of {self.initial_strategy_vals}, got {initial_strategy!r}."
            )
        if imputation_order not in self.imputation_order_vals:
            raise ValueError(
                f"imputation_order must be one of {self.imputation_order_vals}, got {imputation_order!r}."
            )
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.fill_value = fill_value
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._model: IterativeImputer | None = None

    def _fit(self, X: np.ndarray) -> None:
        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=None,
        )
        self._model = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            initial_strategy=self.initial_strategy,
            imputation_order=self.imputation_order,
            fill_value=self.fill_value,
            random_state=self.random_state,
        )
        self._model.fit(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MissForest model is not fitted.")
        return self._model.transform(X)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "n_estimators": self.n_estimators,
                "max_iter": self.max_iter,
                "initial_strategy": self.initial_strategy,
                "imputation_order": self.imputation_order,
                "fill_value": self.fill_value,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
            }
        )
        return params

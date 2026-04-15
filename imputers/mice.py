"""MICE-style multiple imputation wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from imputers.base import BaseImputer


class MICEImputer(BaseImputer):
    """Multiple-imputation variant built on top of sklearn IterativeImputer.

    We fit several posterior-sampling imputers with different random seeds and
    average their completed matrices. This is a practical research-friendly
    approximation of MICE that stays within the project's lightweight stack.
    """

    name = "mice"

    initial_strategy_vals = ["mean", "median", "most_frequent", "constant"]
    imputation_order_vals = ["ascending", "descending", "roman", "arabic", "random"]

    def __init__(
        self,
        *,
        n_imputations: int = 3,
        max_iter: int = 20,
        tol: float = 1e-3,
        initial_strategy: str = "mean",
        imputation_order: str = "ascending",
        fill_value: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if n_imputations < 1:
            raise ValueError("n_imputations must be at least 1.")
        if initial_strategy not in self.initial_strategy_vals:
            raise ValueError(
                f"initial_strategy must be one of {self.initial_strategy_vals}, "
                f"got {initial_strategy!r}."
            )
        if imputation_order not in self.imputation_order_vals:
            raise ValueError(
                f"imputation_order must be one of {self.imputation_order_vals}, "
                f"got {imputation_order!r}."
            )
        self.n_imputations = n_imputations
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.fill_value = fill_value
        self._models: list[IterativeImputer] = []

    def _build_model(self, offset: int) -> IterativeImputer:
        random_state = None if self.random_state is None else self.random_state + offset
        return IterativeImputer(
            max_iter=self.max_iter,
            tol=self.tol,
            initial_strategy=self.initial_strategy,
            imputation_order=self.imputation_order,
            fill_value=self.fill_value,
            sample_posterior=True,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray) -> None:
        self._models = []
        for idx in range(self.n_imputations):
            model = self._build_model(idx)
            model.fit(X)
            self._models.append(model)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("MICE models are not fitted.")
        imputations = [model.transform(X) for model in self._models]
        return np.mean(imputations, axis=0)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "n_imputations": self.n_imputations,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "initial_strategy": self.initial_strategy,
                "imputation_order": self.imputation_order,
                "fill_value": self.fill_value,
            }
        )
        return params

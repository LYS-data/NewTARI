"""Hyperparameter optimization utilities for imputers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from imputers.registry import DEFAULT_REGISTRY, ImputerRegistry
from utils.logging_utils import get_logger, log_kv
from utils.validation import ensure_numeric_matrix


@dataclass
class OptimizationResult:
    best_params: dict[str, Any]
    best_score: float
    best_trial_index: int
    evaluated_trials: list[dict[str, Any]]
    stopped_early: bool


class ImputerOptimizer:
    """Self-supervised hyperparameter tuning for imputers.

    A small subset of observed entries is hidden temporarily, and candidate
    imputers are scored by RMSE on those held-out values.
    """

    def __init__(self, registry: ImputerRegistry | None = None, *, logger_name: str = "imputation_reco.optimizer") -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.logger = get_logger(logger_name)

    def optimize(
        self,
        name: str,
        X: np.ndarray,
        *,
        base_params: dict[str, Any] | None = None,
        max_trials: int = 10,
        holdout_fraction: float = 0.1,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 1e-4,
        random_state: int | None = 0,
    ) -> OptimizationResult:
        X_array = ensure_numeric_matrix(X)
        base_params = dict(base_params or {})
        search_space = self.registry.get_search_space(name)
        candidates = self._build_candidates(search_space, max_trials=max_trials, random_state=random_state)
        if not candidates:
            score = self.evaluate(name, X_array, base_params=base_params, holdout_fraction=holdout_fraction, random_state=random_state)
            return OptimizationResult(base_params, score, 0, [{"trial": 0, "params": base_params, "score": score}], False)

        best_score = float("inf")
        best_params = dict(base_params)
        best_trial_index = -1
        no_improve_rounds = 0
        stopped_early = False
        evaluated_trials: list[dict[str, Any]] = []

        for trial_index, candidate in enumerate(candidates, start=1):
            params = {**base_params, **candidate}
            score = self.evaluate(
                name,
                X_array,
                base_params=params,
                holdout_fraction=holdout_fraction,
                random_state=None if random_state is None else random_state + trial_index,
            )
            evaluated_trials.append({"trial": trial_index, "params": params, "score": score})
            log_kv(self.logger, f"tuning trial for {name}", trial=trial_index, score=round(score, 6), params=params)

            if score + early_stopping_min_delta < best_score:
                best_score = score
                best_params = params
                best_trial_index = trial_index
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if no_improve_rounds >= early_stopping_patience:
                stopped_early = True
                break

        return OptimizationResult(best_params, best_score, best_trial_index, evaluated_trials, stopped_early)

    def evaluate(
        self,
        name: str,
        X: np.ndarray,
        *,
        base_params: dict[str, Any],
        holdout_fraction: float,
        random_state: int | None,
    ) -> float:
        X_array = ensure_numeric_matrix(X)
        masked, holdout_mask = self._mask_observed_entries(X_array, holdout_fraction=holdout_fraction, random_state=random_state)
        if int(holdout_mask.sum()) == 0:
            return 0.0
        imputer = self.registry.build(name, **base_params)
        imputed = imputer.fit_transform(masked)
        truth = X_array[holdout_mask]
        pred = imputed[holdout_mask]
        rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
        return rmse

    def _build_candidates(
        self,
        search_space: dict[str, list[Any]],
        *,
        max_trials: int,
        random_state: int | None,
    ) -> list[dict[str, Any]]:
        if not search_space:
            return []
        keys = sorted(search_space)
        all_combinations = [dict(zip(keys, values)) for values in itertools.product(*(search_space[key] for key in keys))]
        rng = np.random.default_rng(random_state)
        rng.shuffle(all_combinations)
        return all_combinations[:max_trials]

    def _mask_observed_entries(
        self,
        X: np.ndarray,
        *,
        holdout_fraction: float,
        random_state: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        observed_positions = np.argwhere(~np.isnan(X))
        if observed_positions.size == 0:
            raise ValueError("Cannot tune imputer because the matrix has no observed values.")
        n_holdout = max(1, int(len(observed_positions) * holdout_fraction))
        chosen = observed_positions[rng.choice(len(observed_positions), size=n_holdout, replace=False)]
        masked = np.array(X, dtype=float, copy=True)
        holdout_mask = np.zeros_like(masked, dtype=bool)
        for row_idx, col_idx in chosen:
            masked[row_idx, col_idx] = np.nan
            holdout_mask[row_idx, col_idx] = True
        return masked, holdout_mask

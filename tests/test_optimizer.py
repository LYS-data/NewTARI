"""Unit tests for imputer hyperparameter optimization."""

from __future__ import annotations

import numpy as np

from runners.imputation_optimizer import ImputerOptimizer


def make_matrix(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(20, 4))
    X[rng.random(X.shape) < 0.2] = np.nan
    return X


def test_optimizer_returns_best_params_and_trials() -> None:
    optimizer = ImputerOptimizer()
    result = optimizer.optimize(
        "knni",
        make_matrix(),
        max_trials=3,
        holdout_fraction=0.1,
        early_stopping_patience=2,
        random_state=0,
    )
    assert isinstance(result.best_params, dict)
    assert isinstance(result.best_score, float)
    assert len(result.evaluated_trials) >= 1


def test_optimizer_handles_empty_search_space() -> None:
    optimizer = ImputerOptimizer()
    result = optimizer.optimize(
        "mean",
        make_matrix(),
        max_trials=3,
        holdout_fraction=0.1,
        random_state=0,
    )
    assert result.best_params == {}
    assert result.best_trial_index == 0

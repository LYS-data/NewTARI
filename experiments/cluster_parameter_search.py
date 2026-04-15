"""Cluster-aware hyperparameter search for imputers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.cluster_experiment_utils import evaluate_imputed_clustering
from imputers.registry import DEFAULT_REGISTRY, ImputerRegistry
from utils.logging_utils import get_logger, log_kv
from utils.validation import ensure_numeric_matrix


@dataclass(slots=True)
class ClusterSearchResult:
    best_params: dict[str, Any]
    best_score: float
    best_trial_index: int
    evaluated_trials: list[dict[str, Any]]
    stopped_early: bool


class ClusterAwareParameterSearch:
    """Tune imputers against clustering-preservation objectives with fixed K."""

    def __init__(
        self,
        registry: ImputerRegistry | None = None,
        *,
        logger_name: str = "imputation_reco.cluster_search",
    ) -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.logger = get_logger(logger_name)

    def optimize(
        self,
        name: str,
        X_complete: np.ndarray,
        X_missing: np.ndarray,
        missing_mask: np.ndarray,
        *,
        n_clusters: int,
        y_true: np.ndarray | None = None,
        base_params: dict[str, Any] | None = None,
        max_trials: int = 8,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 1e-4,
        random_state: int | None = 0,
    ) -> ClusterSearchResult:
        X_complete = ensure_numeric_matrix(X_complete)
        X_missing = ensure_numeric_matrix(X_missing)
        candidates = self._build_candidates(
            self.registry.get_search_space(name),
            max_trials=max_trials,
            random_state=random_state,
        )
        base_params = dict(base_params or {})

        if not candidates:
            score, metrics = self.evaluate(
                name,
                X_complete,
                X_missing,
                missing_mask,
                n_clusters=n_clusters,
                y_true=y_true,
                params=base_params,
                random_state=random_state,
            )
            return ClusterSearchResult(
                best_params=base_params,
                best_score=score,
                best_trial_index=0,
                evaluated_trials=[{"trial": 0, "params": base_params, "score": score, "metrics": metrics}],
                stopped_early=False,
            )

        best_score = float("-inf")
        best_params = dict(base_params)
        best_trial_index = -1
        no_improve_rounds = 0
        stopped_early = False
        evaluated_trials: list[dict[str, Any]] = []

        for trial_index, candidate in enumerate(candidates, start=1):
            params = {**base_params, **candidate}
            score, metrics = self.evaluate(
                name,
                X_complete,
                X_missing,
                missing_mask,
                n_clusters=n_clusters,
                y_true=y_true,
                params=params,
                random_state=None if random_state is None else random_state + trial_index,
            )
            evaluated_trials.append(
                {"trial": trial_index, "params": params, "score": score, "metrics": metrics}
            )
            log_kv(
                self.logger,
                f"cluster-aware tuning trial for {name}",
                trial=trial_index,
                score=round(score, 6),
                params=params,
            )

            if score > best_score + early_stopping_min_delta:
                best_score = score
                best_params = params
                best_trial_index = trial_index
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            if no_improve_rounds >= early_stopping_patience:
                stopped_early = True
                break

        return ClusterSearchResult(
            best_params=best_params,
            best_score=best_score,
            best_trial_index=best_trial_index,
            evaluated_trials=evaluated_trials,
            stopped_early=stopped_early,
        )

    def evaluate(
        self,
        name: str,
        X_complete: np.ndarray,
        X_missing: np.ndarray,
        missing_mask: np.ndarray,
        *,
        n_clusters: int,
        y_true: np.ndarray | None,
        params: dict[str, Any],
        random_state: int | None,
    ) -> tuple[float, dict[str, float]]:
        X_complete = ensure_numeric_matrix(X_complete)
        X_missing = ensure_numeric_matrix(X_missing)
        imputer = self.registry.build(name, **params)
        X_imputed = imputer.fit_transform(X_missing)
        metrics = evaluate_imputed_clustering(
            X_complete=X_complete,
            X_missing=X_missing,
            X_imputed=X_imputed,
            missing_mask=missing_mask,
            method_name=name,
            n_clusters=n_clusters,
            random_state=0 if random_state is None else random_state,
            y_true=y_true,
        )
        score = self._compose_score(metrics)
        return score, metrics

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
        all_combinations = [
            dict(zip(keys, values))
            for values in itertools.product(*(search_space[key] for key in keys))
        ]
        rng = np.random.default_rng(random_state)
        rng.shuffle(all_combinations)
        return all_combinations[:max_trials]

    def _compose_score(self, metrics: dict[str, float]) -> float:
        """Higher is better."""
        truth_term = metrics["ari_to_truth"]
        if np.isnan(truth_term):
            truth_term = metrics["cluster_consistency_ari"]
        score = (
            0.50 * metrics["cluster_consistency_ari"]
            + 0.30 * truth_term
            + 0.20 * metrics["silhouette"]
            - 0.10 * metrics["masked_nrmse"]
        )
        return float(score)


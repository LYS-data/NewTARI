"""Feature-wise recommendation logic for clustering-oriented imputation experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from imputers.registry import DEFAULT_REGISTRY, ImputerRegistry
from utils.clustering import cluster_and_score


@dataclass(slots=True)
class FeaturewiseRecommendationConfig:
    candidate_methods: list[str]
    baseline_method: str = "best_single"
    budget: int = 10
    top_k_features: int = 5
    top_r_methods: int = 3
    beam_width: int = 4
    random_state: int = 42


def bounded_internal_objective(
    metrics: dict[str, float],
    *,
    reference_metrics: dict[str, float],
) -> float:
    """Stable internal clustering objective for comparing imputation strategies."""

    silhouette_score = float(np.clip((metrics["silhouette"] + 1.0) / 2.0, 0.0, 1.0))
    dbi_score = float(np.clip(1.0 / (1.0 + max(metrics["davies_bouldin"], 1e-8)), 0.0, 1.0))
    reference_ch = max(reference_metrics["calinski_harabasz"], 1e-8)
    ch_ratio = float(np.clip(metrics["calinski_harabasz"] / reference_ch, 0.0, 1.5))
    ch_score = ch_ratio / 1.5
    return float(0.4 * silhouette_score + 0.3 * dbi_score + 0.3 * ch_score)


def _evaluate_internal_metrics(
    X: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
) -> dict[str, float]:
    _, metrics = cluster_and_score(X, n_clusters=n_clusters, random_state=random_state)
    return metrics


def _compose_matrix(
    strategy: dict[int, str],
    baseline_matrix: np.ndarray,
    imputed_dict: dict[str, np.ndarray],
) -> np.ndarray:
    X = np.array(baseline_matrix, copy=True)
    for feature_idx, method_name in strategy.items():
        X[:, feature_idx] = imputed_dict[method_name][:, feature_idx]
    return X


def _feature_missing_indices(X_missing: np.ndarray) -> list[int]:
    return [idx for idx in range(X_missing.shape[1]) if np.isnan(X_missing[:, idx]).any()]


def _strategy_key(strategy: dict[int, str]) -> tuple[tuple[int, str], ...]:
    return tuple(sorted(strategy.items(), key=lambda item: item[0]))


def recommend_featurewise_strategy(
    X_missing: np.ndarray,
    *,
    n_clusters: int,
    config: FeaturewiseRecommendationConfig,
    candidate_params: dict[str, dict[str, Any]] | None = None,
    reference_complete_data: np.ndarray | None = None,
    registry: ImputerRegistry | None = None,
) -> dict[str, Any]:
    """Recommend a feature-wise imputation strategy under a small search budget."""

    registry = registry or DEFAULT_REGISTRY
    candidate_params = candidate_params or {}
    X_missing = np.asarray(X_missing, dtype=float)
    if X_missing.ndim != 2:
        raise ValueError("X_missing must be a 2D numeric matrix.")

    imputed_dict: dict[str, np.ndarray] = {}
    whole_method_records: list[dict[str, Any]] = []

    if reference_complete_data is not None:
        reference_metrics = _evaluate_internal_metrics(
            np.asarray(reference_complete_data, dtype=float),
            n_clusters=n_clusters,
            random_state=config.random_state,
        )
    else:
        reference_metrics = None

    for method_name in config.candidate_methods:
        params = dict(candidate_params.get(method_name, {}))
        run_start = perf_counter()
        imputer = registry.build(method_name, **params)
        X_imputed = imputer.fit_transform(X_missing)
        runtime_sec = perf_counter() - run_start
        imputed_dict[method_name] = X_imputed
        metrics = _evaluate_internal_metrics(
            X_imputed,
            n_clusters=n_clusters,
            random_state=config.random_state,
        )
        whole_method_records.append(
            {
                "method": method_name,
                "runtime_sec": runtime_sec,
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                **metrics,
            }
        )

    if reference_metrics is None:
        best_ch_record = max(whole_method_records, key=lambda item: item["calinski_harabasz"])
        reference_metrics = {
            "silhouette": best_ch_record["silhouette"],
            "davies_bouldin": best_ch_record["davies_bouldin"],
            "calinski_harabasz": best_ch_record["calinski_harabasz"],
        }

    for record in whole_method_records:
        record["internal_objective_q"] = bounded_internal_objective(
            record,
            reference_metrics=reference_metrics,
        )

    best_single_record = max(whole_method_records, key=lambda item: item["internal_objective_q"])
    baseline_method = best_single_record["method"] if config.baseline_method == "best_single" else config.baseline_method
    baseline_matrix = imputed_dict[baseline_method]
    baseline_metrics = _evaluate_internal_metrics(
        baseline_matrix,
        n_clusters=n_clusters,
        random_state=config.random_state,
    )
    baseline_score = bounded_internal_objective(
        baseline_metrics,
        reference_metrics=reference_metrics,
    )

    feature_indices = _feature_missing_indices(X_missing)
    sensitivity_records: list[dict[str, Any]] = []
    sensitivity_by_feature: dict[int, float] = {}
    candidates_by_feature: dict[int, list[str]] = {}

    for feature_idx in feature_indices:
        feature_trials: list[tuple[str, float, dict[str, float]]] = []
        for method_name in config.candidate_methods:
            trial_matrix = np.array(baseline_matrix, copy=True)
            trial_matrix[:, feature_idx] = imputed_dict[method_name][:, feature_idx]
            trial_metrics = _evaluate_internal_metrics(
                trial_matrix,
                n_clusters=n_clusters,
                random_state=config.random_state,
            )
            trial_score = bounded_internal_objective(
                trial_metrics,
                reference_metrics=reference_metrics,
            )
            feature_trials.append((method_name, trial_score, trial_metrics))
        feature_trials.sort(key=lambda item: item[1], reverse=True)
        best_method, best_score, best_metrics = feature_trials[0]
        sensitivity_gain = best_score - baseline_score
        sensitivity_by_feature[feature_idx] = sensitivity_gain
        candidates_by_feature[feature_idx] = [
            name for name, _, _ in feature_trials[: config.top_r_methods]
        ]
        sensitivity_records.append(
            {
                "feature_idx": feature_idx,
                "missing_rate_feature": float(np.isnan(X_missing[:, feature_idx]).mean()),
                "baseline_method": baseline_method,
                "best_alternative_method": best_method,
                "sensitivity_gain": sensitivity_gain,
                "best_feature_score": best_score,
                "baseline_score": baseline_score,
                "candidate_methods_json": json.dumps(candidates_by_feature[feature_idx], ensure_ascii=False),
                "best_silhouette": best_metrics["silhouette"],
                "best_davies_bouldin": best_metrics["davies_bouldin"],
                "best_calinski_harabasz": best_metrics["calinski_harabasz"],
            }
        )

    ranked_features = sorted(feature_indices, key=lambda idx: sensitivity_by_feature[idx], reverse=True)
    ranked_features = ranked_features[: min(config.top_k_features, len(ranked_features))]

    baseline_strategy: dict[int, str] = {}
    baseline_key = _strategy_key(baseline_strategy)
    strategy_cache: dict[tuple[tuple[int, str], ...], dict[str, Any]] = {
        baseline_key: {
            "strategy": baseline_strategy,
            "matrix": baseline_matrix,
            "metrics": baseline_metrics,
            "score": baseline_score,
        }
    }
    beam: list[dict[str, Any]] = [strategy_cache[baseline_key]]
    best_result = strategy_cache[baseline_key]
    strategy_trial_records: list[dict[str, Any]] = []
    eval_budget_used = 1

    for feature_idx in ranked_features:
        next_pool: list[dict[str, Any]] = list(beam)
        for state in beam:
            current_strategy = dict(state["strategy"])
            for method_name in candidates_by_feature[feature_idx]:
                if eval_budget_used >= config.budget:
                    break
                if method_name == baseline_method:
                    continue
                trial_strategy = dict(current_strategy)
                trial_strategy[feature_idx] = method_name
                trial_key = _strategy_key(trial_strategy)
                if trial_key in strategy_cache:
                    continue
                trial_matrix = _compose_matrix(trial_strategy, baseline_matrix, imputed_dict)
                trial_metrics = _evaluate_internal_metrics(
                    trial_matrix,
                    n_clusters=n_clusters,
                    random_state=config.random_state,
                )
                trial_score = bounded_internal_objective(
                    trial_metrics,
                    reference_metrics=reference_metrics,
                )
                trial_state = {
                    "strategy": trial_strategy,
                    "matrix": trial_matrix,
                    "metrics": trial_metrics,
                    "score": trial_score,
                }
                strategy_cache[trial_key] = trial_state
                next_pool.append(trial_state)
                strategy_trial_records.append(
                    {
                        "eval_index": eval_budget_used,
                        "feature_idx": feature_idx,
                        "candidate_method": method_name,
                        "trial_score": trial_score,
                        "previous_score": state["score"],
                        "improved": trial_score > state["score"],
                        "strategy_json": json.dumps(trial_strategy, ensure_ascii=False, sort_keys=True),
                        "silhouette": trial_metrics["silhouette"],
                        "davies_bouldin": trial_metrics["davies_bouldin"],
                        "calinski_harabasz": trial_metrics["calinski_harabasz"],
                    }
                )
                eval_budget_used += 1
                if trial_score > best_result["score"]:
                    best_result = trial_state
            if eval_budget_used >= config.budget:
                break
        unique_pool = {_strategy_key(item["strategy"]): item for item in next_pool}
        beam = sorted(
            unique_pool.values(),
            key=lambda item: (item["score"], -len(item["strategy"])),
            reverse=True,
        )[: config.beam_width]
        if eval_budget_used >= config.budget:
            break

    selected_strategy = {
        feature_idx: best_result["strategy"].get(feature_idx, baseline_method)
        for feature_idx in ranked_features
    }
    feature_assignment_records = [
        {
            "feature_idx": feature_idx,
            "selected_method": selected_strategy[feature_idx],
            "sensitivity_gain": sensitivity_by_feature[feature_idx],
            "candidate_methods_json": json.dumps(candidates_by_feature[feature_idx], ensure_ascii=False),
        }
        for feature_idx in ranked_features
    ]

    return {
        "config": {
            "candidate_methods": list(config.candidate_methods),
            "baseline_method": baseline_method,
            "budget": config.budget,
            "top_k_features": config.top_k_features,
            "top_r_methods": config.top_r_methods,
            "beam_width": config.beam_width,
            "random_state": config.random_state,
        },
        "reference_metrics": reference_metrics,
        "whole_method_scores": whole_method_records,
        "feature_sensitivity": sensitivity_records,
        "strategy_trials": strategy_trial_records,
        "feature_assignments": feature_assignment_records,
        "final_result": {
            "baseline_method": baseline_method,
            "best_single_method": best_single_record["method"],
            "best_single_q": best_single_record["internal_objective_q"],
            "recommended_q": best_result["score"],
            "recommended_improvement_over_baseline": best_result["score"] - baseline_score,
            "recommended_improvement_over_best_single": best_result["score"] - best_single_record["internal_objective_q"],
            "baseline_silhouette": baseline_metrics["silhouette"],
            "baseline_davies_bouldin": baseline_metrics["davies_bouldin"],
            "baseline_calinski_harabasz": baseline_metrics["calinski_harabasz"],
            "recommended_silhouette": best_result["metrics"]["silhouette"],
            "recommended_davies_bouldin": best_result["metrics"]["davies_bouldin"],
            "recommended_calinski_harabasz": best_result["metrics"]["calinski_harabasz"],
            "recommended_strategy": selected_strategy,
            "ranked_features": ranked_features,
        },
    }

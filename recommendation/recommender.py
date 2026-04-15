"""Recommendation entrypoints for clustering-oriented imputation strategy selection."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from recommendation.featurewise import (
    FeaturewiseRecommendationConfig,
    recommend_featurewise_strategy,
)


def recommend(X_missing: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """Recommend a feature-wise imputation strategy.

    This exposes the concrete feature-wise search logic used by the clustering
    experiments instead of returning a placeholder object.
    """

    n_clusters = kwargs.pop("n_clusters", None)
    if n_clusters is None:
        raise ValueError("recommend(...) requires `n_clusters`.")

    candidate_methods = kwargs.pop("candidate_methods", None)
    if not candidate_methods:
        raise ValueError("recommend(...) requires `candidate_methods`.")

    config = FeaturewiseRecommendationConfig(
        candidate_methods=list(candidate_methods),
        baseline_method=kwargs.pop("baseline_method", "best_single"),
        budget=int(kwargs.pop("budget", 10)),
        top_k_features=int(kwargs.pop("top_k_features", 5)),
        top_r_methods=int(kwargs.pop("top_r_methods", 3)),
        beam_width=int(kwargs.pop("beam_width", 4)),
        random_state=int(kwargs.pop("random_state", 42)),
    )

    result = recommend_featurewise_strategy(
        X_missing,
        n_clusters=int(n_clusters),
        config=config,
        candidate_params=kwargs.pop("candidate_params", None),
        reference_complete_data=kwargs.pop("reference_complete_data", None),
        registry=kwargs.pop("registry", None),
    )
    if kwargs:
        result["unused_kwargs"] = kwargs
    result["status"] = "success"
    return result


def format_recommendation(result: dict[str, Any]) -> str:
    """Serialize a recommendation result into a compact text summary."""

    final_result = result.get("final_result", {})
    strategy = final_result.get("recommended_strategy")
    strategy_text = json.dumps(strategy, ensure_ascii=False, sort_keys=True) if strategy else "{}"
    return (
        f"status={result.get('status', 'unknown')}, "
        f"baseline_method={final_result.get('baseline_method')}, "
        f"best_single_method={final_result.get('best_single_method')}, "
        f"recommended_q={final_result.get('recommended_q')}, "
        f"recommended_strategy={strategy_text}"
    )

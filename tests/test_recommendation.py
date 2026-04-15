"""Tests for reusable recommendation and missingness helpers."""

from __future__ import annotations

import numpy as np

from recommendation import (
    FeaturewiseRecommendationConfig,
    format_recommendation,
    recommend,
    recommend_featurewise_strategy,
)
from utils.missingness import inject_mcar_missing


def make_clustered_matrix(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-2.0, scale=0.5, size=(20, 3))
    b = rng.normal(loc=2.0, scale=0.5, size=(20, 3))
    return np.vstack([a, b])


def test_inject_mcar_missing_protects_full_rows_and_columns() -> None:
    X = make_clustered_matrix()
    X_missing, mask = inject_mcar_missing(X, missing_rate=0.5, random_state=0)
    assert X_missing.shape == X.shape
    assert mask.shape == X.shape
    assert not mask.all(axis=0).any()
    assert not mask.all(axis=1).any()


def test_featurewise_recommendation_returns_structured_result() -> None:
    X_complete = make_clustered_matrix()
    X_missing, _ = inject_mcar_missing(X_complete, missing_rate=0.2, random_state=1)
    config = FeaturewiseRecommendationConfig(
        candidate_methods=["mean", "median", "knni"],
        budget=4,
        top_k_features=2,
        top_r_methods=2,
        beam_width=2,
        random_state=1,
    )
    result = recommend_featurewise_strategy(
        X_missing,
        n_clusters=2,
        config=config,
        reference_complete_data=X_complete,
    )
    assert "whole_method_scores" in result
    assert "feature_sensitivity" in result
    assert "final_result" in result
    assert result["final_result"]["best_single_method"] in {"mean", "median", "knni"}


def test_recommend_wrapper_and_formatter_work() -> None:
    X_complete = make_clustered_matrix()
    X_missing, _ = inject_mcar_missing(X_complete, missing_rate=0.2, random_state=2)
    result = recommend(
        X_missing,
        n_clusters=2,
        candidate_methods=["mean", "median", "knni"],
        budget=4,
        top_k_features=2,
        top_r_methods=2,
        beam_width=2,
        random_state=2,
        reference_complete_data=X_complete,
    )
    text = format_recommendation(result)
    assert result["status"] == "success"
    assert "recommended_strategy=" in text

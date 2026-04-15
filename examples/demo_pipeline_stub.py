"""Demo: show how the future pipeline can be wired together today."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from analysis.feature_sensitivity import compute_feature_sensitivity, compute_method_compatibility
from cluster_selection.selector import select_default_clusterer
from datasets.loader import DatasetLoader
from recommendation.recommender import format_recommendation, recommend
from runners.imputation_runner import ImputationRunner
from search.space_reduction import filter_candidate_methods, select_key_features


def build_demo_frame(seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 4))
    X[rng.random(X.shape) < 0.15] = np.nan
    frame = pd.DataFrame(X, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    frame["species"] = rng.choice(["setosa", "versicolor", "virginica"], size=len(frame))
    return frame


def main() -> None:
    loader = DatasetLoader()
    bundle = loader.load_dataframe(
        build_demo_frame(),
        dataset_name="demo_like_iris",
        label_column="species",
    )

    runner = ImputationRunner()
    imputed_results = runner.run(
        bundle.X,
        ["deletion", "mean", "knni", "mice"],
        output_json_path=Path("results"),
        dataset_name=bundle.dataset_name,
        scenario_name="demo_pipeline_stub",
        run_tag="pipeline_stub",
    )
    successful = {
        name: payload["imputed_array"]
        for name, payload in imputed_results.items()
        if payload["success"]
    }

    chosen_matrix = successful["mean"]
    clusterer = select_default_clusterer(chosen_matrix, candidate_clusterers=["kmeans", "gmm"])
    sensitivity_scores = compute_feature_sensitivity(bundle.X, successful, clusterer)
    method_scores = compute_method_compatibility(bundle.X, successful, clusterer)
    key_features = select_key_features(sensitivity_scores, top_k=3)
    shortlisted_methods = filter_candidate_methods(method_scores, top_r=2)
    recommendation = recommend(
        bundle.X,
        successful_imputations=successful,
        clusterer=clusterer,
        key_features=key_features,
        shortlisted_methods=shortlisted_methods,
        label_reference_available=bundle.y is not None,
    )

    print("Dataset:", bundle.dataset_name)
    print("Feature columns:", bundle.feature_names)
    print("Label column excluded from imputation:", bundle.label_column)
    print("Clusterer placeholder:", clusterer)
    print("Key features placeholder:", key_features)
    print("Shortlisted methods placeholder:", shortlisted_methods)
    print("Recommendation placeholder:", format_recommendation(recommendation))


if __name__ == "__main__":
    main()

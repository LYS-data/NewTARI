"""Run a budget-aware feature-wise imputation recommendation experiment on three datasets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_ID = "budget_aware_featurewise_recommendation_v2__3datasets__mcar20__seed42"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / EXPERIMENT_ID
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR = RESULTS_DIR / "mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from datasets import DatasetLoader, load_dataset_manifest
from experiments.cluster_experiment_utils import (
    DATASET_CLUSTER_COUNTS,
    RAW_DIR,
    build_dataset_specs,
    inject_mcar_missing,
)
from recommendation import FeaturewiseRecommendationConfig, recommend_featurewise_strategy


DATASETS = {"iris", "wine", "seeds"}
BASELINE_METHOD = "median"
CANDIDATE_METHODS = [
    "mean",
    "median",
    "knni",
    "em",
    "mice",
    "missforest",
    "iterative_xgboost",
    "nomi",
]
MISSING_RATE = 0.20
RANDOM_SEED = 42
BUDGET = 10
TOP_K_FEATURES = 5
TOP_R_METHODS = 3
BEAM_WIDTH = 4

PREVIOUS_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "tuned_cluster_experiment__fixedk__mcar20__seed42"
    / "tables"
    / "02_formal_experiment_results.csv"
)


def _style() -> None:
    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "savefig.transparent": False,
            "axes.edgecolor": "#d9d9d9",
            "grid.color": "#e9e9e9",
            "axes.labelcolor": "#2f2a24",
            "text.color": "#2f2a24",
            "xtick.color": "#4a4036",
            "ytick.color": "#4a4036",
            "font.family": "DejaVu Serif",
            "axes.titleweight": "bold",
        }
    )


def load_tuned_params() -> dict[tuple[str, str], dict[str, Any]]:
    df = pd.read_csv(PREVIOUS_RESULTS)
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in df.iterrows():
        params = json.loads(row["best_params_json"]) if isinstance(row["best_params_json"], str) else {}
        mapping[(row["dataset_name"], row["method"])] = params
    return mapping

def save_plot(fig: plt.Figure, file_name: str) -> Path:
    path = PLOTS_DIR / file_name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> None:
    _style()
    loader = DatasetLoader(data_root=RAW_DIR)
    specs = [spec for spec in build_dataset_specs(load_dataset_manifest(RAW_DIR)) if spec.dataset_name in DATASETS]
    tuned_params = load_tuned_params()

    whole_method_records: list[dict[str, Any]] = []
    sensitivity_records: list[dict[str, Any]] = []
    strategy_trial_records: list[dict[str, Any]] = []
    final_records: list[dict[str, Any]] = []
    assignment_records: list[dict[str, Any]] = []
    config = FeaturewiseRecommendationConfig(
        candidate_methods=CANDIDATE_METHODS,
        baseline_method="best_single",
        budget=BUDGET,
        top_k_features=TOP_K_FEATURES,
        top_r_methods=TOP_R_METHODS,
        beam_width=BEAM_WIDTH,
        random_state=RANDOM_SEED,
    )

    for spec in specs:
        bundle = loader.load_csv(
            spec.file_name,
            dataset_name=spec.dataset_name,
            label_column=spec.label_column,
            drop_columns=spec.drop_columns,
        )
        X_complete = bundle.X
        y_true: np.ndarray | None = None
        if bundle.y is not None:
            y_true = LabelEncoder().fit_transform(bundle.y.astype(str))
        X_missing, missing_mask = inject_mcar_missing(X_complete, MISSING_RATE, RANDOM_SEED)
        n_clusters = DATASET_CLUSTER_COUNTS[spec.dataset_name]
        method_params = {
            method_name: dict(tuned_params.get((spec.dataset_name, method_name), {}))
            for method_name in CANDIDATE_METHODS
        }
        recommendation = recommend_featurewise_strategy(
            X_missing,
            n_clusters=n_clusters,
            config=config,
            candidate_params=method_params,
            reference_complete_data=X_complete,
        )

        whole_method_records.extend(
            [{"dataset_name": spec.dataset_name, **row} for row in recommendation["whole_method_scores"]]
        )
        sensitivity_records.extend(
            [{"dataset_name": spec.dataset_name, **row} for row in recommendation["feature_sensitivity"]]
        )
        strategy_trial_records.extend(
            [{"dataset_name": spec.dataset_name, **row} for row in recommendation["strategy_trials"]]
        )

        final_result = recommendation["final_result"]
        final_records.append(
            {
                "dataset_name": spec.dataset_name,
                "missing_rate": MISSING_RATE,
                "budget": BUDGET,
                "baseline_method": final_result["baseline_method"],
                "best_single_method": final_result["best_single_method"],
                "best_single_q": final_result["best_single_q"],
                "recommended_q": final_result["recommended_q"],
                "recommended_improvement_over_baseline": final_result["recommended_improvement_over_baseline"],
                "recommended_improvement_over_best_single": final_result["recommended_improvement_over_best_single"],
                "baseline_silhouette": final_result["baseline_silhouette"],
                "baseline_davies_bouldin": final_result["baseline_davies_bouldin"],
                "baseline_calinski_harabasz": final_result["baseline_calinski_harabasz"],
                "recommended_silhouette": final_result["recommended_silhouette"],
                "recommended_davies_bouldin": final_result["recommended_davies_bouldin"],
                "recommended_calinski_harabasz": final_result["recommended_calinski_harabasz"],
                "recommended_strategy_json": json.dumps(final_result["recommended_strategy"], ensure_ascii=False, sort_keys=True),
                "ranked_features_json": json.dumps(final_result["ranked_features"], ensure_ascii=False),
            }
        )

        assignment_records.extend(
            [{"dataset_name": spec.dataset_name, **row} for row in recommendation["feature_assignments"]]
        )

    whole_df = pd.DataFrame(whole_method_records)
    whole_df.to_csv(TABLES_DIR / "01_whole_method_internal_scores.csv", index=False)

    sensitivity_df = pd.DataFrame(sensitivity_records)
    sensitivity_df.to_csv(TABLES_DIR / "02_feature_sensitivity.csv", index=False)

    trials_df = pd.DataFrame(strategy_trial_records)
    trials_df.to_csv(TABLES_DIR / "03_strategy_trials.csv", index=False)

    final_df = pd.DataFrame(final_records)
    final_df.to_csv(TABLES_DIR / "04_final_recommendation_results.csv", index=False)

    assignment_df = pd.DataFrame(assignment_records)
    assignment_df.to_csv(TABLES_DIR / "05_feature_assignments.csv", index=False)

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "datasets": sorted(DATASETS),
        "missing_rate": MISSING_RATE,
        "random_seed": RANDOM_SEED,
        "baseline_method": "dataset-wise best single method",
        "candidate_methods": CANDIDATE_METHODS,
        "budget": BUDGET,
        "top_k_features": TOP_K_FEATURES,
        "top_r_methods": TOP_R_METHODS,
        "beam_width": BEAM_WIDTH,
        "objective": "Bounded internal objective: 0.4*scaled silhouette + 0.3*inverse Davies-Bouldin + 0.3*Calinski-Harabasz ratio to complete-data reference.",
    }
    (TABLES_DIR / "00_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    metric_cols = [
        ("recommended_silhouette", "Silhouette"),
        ("recommended_davies_bouldin", "Davies-Bouldin"),
        ("recommended_calinski_harabasz", "Calinski-Harabasz"),
    ]
    for ax, (col, title) in zip(axes, metric_cols):
        plot_df = final_df.melt(
            id_vars=["dataset_name"],
            value_vars=[col, col.replace("recommended_", "baseline_")],
            var_name="type",
            value_name="value",
        )
        plot_df["type"] = plot_df["type"].map(
            {
                col: "Recommended",
                col.replace("recommended_", "baseline_"): "Best single baseline",
            }
        )
        sns.barplot(data=plot_df, x="dataset_name", y="value", hue="type", ax=ax, palette=["#4c78a8", "#f58518"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Value")
        ax.legend(frameon=False, title="")
    save_plot(fig, "01_baseline_vs_recommended_internal_metrics.png")

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    compare_df = final_df.melt(
        id_vars=["dataset_name"],
        value_vars=["best_single_q", "recommended_q"],
        var_name="strategy",
        value_name="q_value",
    )
    compare_df["strategy"] = compare_df["strategy"].map(
        {"best_single_q": "Best single method", "recommended_q": "Feature-wise recommendation"}
    )
    sns.barplot(data=compare_df, x="dataset_name", y="q_value", hue="strategy", ax=ax, palette=["#7f7f7f", "#2a9d8f"])
    ax.set_title("Best Single vs Feature-wise Recommendation", loc="left", fontsize=18)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Objective Q")
    ax.legend(frameon=False, title="")
    save_plot(fig, "02_best_single_vs_featurewise_q.png")

    print(f"Wrote metadata to: {TABLES_DIR / '00_metadata.json'}")
    print(f"Wrote whole-method scores to: {TABLES_DIR / '01_whole_method_internal_scores.csv'}")
    print(f"Wrote feature sensitivity to: {TABLES_DIR / '02_feature_sensitivity.csv'}")
    print(f"Wrote strategy trials to: {TABLES_DIR / '03_strategy_trials.csv'}")
    print(f"Wrote final results to: {TABLES_DIR / '04_final_recommendation_results.csv'}")
    print(f"Wrote assignments to: {TABLES_DIR / '05_feature_assignments.csv'}")


if __name__ == "__main__":
    main()

"""Tune each imputer first, then run the formal fixed-K clustering experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from datasets import DatasetLoader, load_dataset_manifest
from experiments.cluster_experiment_utils import (
    DATASET_CLUSTER_COUNTS,
    RAW_DIR,
    build_dataset_specs,
    evaluate_imputed_clustering,
    inject_mcar_missing,
    maybe_subsample,
)
from experiments.cluster_parameter_search import ClusterAwareParameterSearch
from imputers.registry import DEFAULT_REGISTRY


EXPERIMENT_ID = "tuned_cluster_experiment__fixedk__mcar20__seed42"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / EXPERIMENT_ID
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_METHODS = [
    "deletion",
    "mean",
    "median",
    "knni",
    "em",
    "mice",
    "missforest",
    "iterative_xgboost",
    "gain",
    "miwae",
    "hivae",
    "nomi",
    "diffputer",
    "grape",
]

BASE_PARAMS: dict[str, dict[str, Any]] = {
    "deletion": {},
    "mean": {},
    "median": {},
    "knni": {"random_state": 42},
    "em": {"random_state": 42},
    "mice": {"random_state": 42},
    "missforest": {"random_state": 42},
    "iterative_xgboost": {"random_state": 42},
    "gain": {"random_state": 42},
    "miwae": {"random_state": 42},
    "hivae": {"random_state": 42},
    "nomi": {"random_state": 42},
    "diffputer": {"random_state": 42},
    "grape": {"random_state": 42},
}

SEARCH_CONFIG: dict[str, dict[str, Any]] = {
    "deletion": {"max_trials": 1},
    "mean": {"max_trials": 1},
    "median": {"max_trials": 1},
    "knni": {"max_trials": 3},
    "em": {"max_trials": 4},
    "mice": {"max_trials": 4},
    "missforest": {"max_trials": 4},
    "iterative_xgboost": {"max_trials": 4},
    "gain": {"max_trials": 4},
    "miwae": {"max_trials": 4},
    "hivae": {"max_trials": 4},
    "nomi": {"max_trials": 4},
    "diffputer": {"max_trials": 4},
    "grape": {"max_trials": 4},
}

MISSING_RATE = 0.20
MISSING_SCHEME = "MCAR"
RANDOM_SEED = 42


def main() -> None:
    loader = DatasetLoader(data_root=RAW_DIR)
    tuner = ClusterAwareParameterSearch(DEFAULT_REGISTRY)
    specs = build_dataset_specs(load_dataset_manifest(RAW_DIR))

    tuning_records: list[dict[str, Any]] = []
    experiment_records: list[dict[str, Any]] = []
    best_params_by_dataset_method: dict[tuple[str, str], dict[str, Any]] = {}

    for spec in specs:
        bundle = loader.load_csv(
            spec.file_name,
            dataset_name=spec.dataset_name,
            label_column=spec.label_column,
            drop_columns=spec.drop_columns,
        )
        X = bundle.X
        y: np.ndarray | None = None
        if bundle.y is not None:
            encoder = LabelEncoder()
            y = encoder.fit_transform(bundle.y.astype(str))
        X, y = maybe_subsample(X, y, spec.dataset_name, RANDOM_SEED)
        X_missing, missing_mask = inject_mcar_missing(X, MISSING_RATE, RANDOM_SEED)
        n_clusters = DATASET_CLUSTER_COUNTS[spec.dataset_name]

        for method_name in ALL_METHODS:
            base_params = dict(BASE_PARAMS.get(method_name, {}))
            search_options = SEARCH_CONFIG.get(method_name, {"max_trials": 4})
            tuning_start = perf_counter()
            search_result = tuner.optimize(
                method_name,
                X,
                X_missing,
                missing_mask,
                n_clusters=n_clusters,
                y_true=y,
                base_params=base_params,
                max_trials=search_options.get("max_trials", 4),
                early_stopping_patience=search_options.get("early_stopping_patience", 2),
                early_stopping_min_delta=search_options.get("early_stopping_min_delta", 1e-4),
                random_state=RANDOM_SEED,
            )
            tuning_runtime_sec = perf_counter() - tuning_start
            best_params = dict(search_result.best_params)
            best_params_by_dataset_method[(spec.dataset_name, method_name)] = best_params

            for trial in search_result.evaluated_trials:
                trial_metrics = dict(trial["metrics"])
                tuning_records.append(
                    {
                        "dataset_name": spec.dataset_name,
                        "method": method_name,
                        "n_samples": int(X.shape[0]),
                        "n_features": int(X.shape[1]),
                        "n_clusters": n_clusters,
                        "trial": int(trial["trial"]),
                        "trial_score": float(trial["score"]),
                        "is_best_trial": bool(trial["trial"] == search_result.best_trial_index),
                        "best_trial_index": int(search_result.best_trial_index),
                        "stopped_early": bool(search_result.stopped_early),
                        "params_json": json.dumps(trial["params"], ensure_ascii=False, sort_keys=True),
                        "cluster_consistency_ari": trial_metrics["cluster_consistency_ari"],
                        "ari_to_truth": trial_metrics["ari_to_truth"],
                        "nmi_to_truth": trial_metrics["nmi_to_truth"],
                        "silhouette": trial_metrics["silhouette"],
                        "davies_bouldin": trial_metrics["davies_bouldin"],
                        "calinski_harabasz": trial_metrics["calinski_harabasz"],
                        "masked_rmse": trial_metrics["masked_rmse"],
                        "masked_nrmse": trial_metrics["masked_nrmse"],
                        "row_retention": trial_metrics["row_retention"],
                        "tuning_runtime_sec": float(tuning_runtime_sec),
                    }
                )

            run_start = perf_counter()
            imputer = DEFAULT_REGISTRY.build(method_name, **best_params)
            X_imputed = imputer.fit_transform(X_missing)
            formal_runtime_sec = perf_counter() - run_start
            final_metrics = evaluate_imputed_clustering(
                X_complete=X,
                X_missing=X_missing,
                X_imputed=X_imputed,
                missing_mask=missing_mask,
                method_name=method_name,
                n_clusters=n_clusters,
                random_state=RANDOM_SEED,
                y_true=y,
            )
            experiment_records.append(
                {
                    "dataset_name": spec.dataset_name,
                    "method": method_name,
                    "missing_scheme": MISSING_SCHEME,
                    "missing_rate": MISSING_RATE,
                    "seed": RANDOM_SEED,
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "n_clusters": n_clusters,
                    "best_params_json": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
                    "best_tuning_score": float(search_result.best_score),
                    "best_trial_index": int(search_result.best_trial_index),
                    "tuning_trials": int(len(search_result.evaluated_trials)),
                    "tuning_stopped_early": bool(search_result.stopped_early),
                    "formal_runtime_sec": float(formal_runtime_sec),
                    **final_metrics,
                }
            )

    tuning_df = pd.DataFrame(tuning_records)
    tuning_path = TABLES_DIR / "01_tuning_trials.csv"
    tuning_df.to_csv(tuning_path, index=False)

    final_df = pd.DataFrame(experiment_records)
    final_path = TABLES_DIR / "02_formal_experiment_results.csv"
    final_df.to_csv(final_path, index=False)

    summary = (
        final_df.groupby("method", dropna=False)
        .agg(
            mean_cluster_consistency_ari=("cluster_consistency_ari", "mean"),
            mean_ari_to_truth=("ari_to_truth", "mean"),
            mean_nmi_to_truth=("nmi_to_truth", "mean"),
            mean_silhouette=("silhouette", "mean"),
            mean_masked_nrmse=("masked_nrmse", "mean"),
            mean_formal_runtime_sec=("formal_runtime_sec", "mean"),
            mean_row_retention=("row_retention", "mean"),
            mean_best_tuning_score=("best_tuning_score", "mean"),
        )
        .reset_index()
    )
    summary["score_cluster"] = summary["mean_cluster_consistency_ari"].rank(pct=True)
    summary["score_truth"] = summary["mean_ari_to_truth"].fillna(summary["mean_ari_to_truth"].mean()).rank(pct=True)
    summary["score_nrmse"] = (-summary["mean_masked_nrmse"]).rank(pct=True)
    summary["score_runtime"] = (-summary["mean_formal_runtime_sec"]).rank(pct=True)
    summary["score_retention"] = summary["mean_row_retention"].rank(pct=True)
    summary["overall_score"] = summary[
        ["score_cluster", "score_truth", "score_nrmse", "score_runtime", "score_retention"]
    ].mean(axis=1)
    summary = summary.sort_values("overall_score", ascending=False)
    summary_path = TABLES_DIR / "03_method_summary.csv"
    summary.to_csv(summary_path, index=False)

    winners = (
        final_df.sort_values(["dataset_name", "cluster_consistency_ari"], ascending=[True, False])
        .groupby("dataset_name")
        .head(3)
    )
    winners_path = TABLES_DIR / "04_dataset_top3_methods.csv"
    winners.to_csv(winners_path, index=False)

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "objective": "Tune each imputer with fixed-K cluster-aware search, then run formal fixed-K clustering experiment.",
        "datasets": [spec.dataset_name for spec in specs],
        "methods": ALL_METHODS,
        "missing_scheme": MISSING_SCHEME,
        "missing_rate": MISSING_RATE,
        "random_seed": RANDOM_SEED,
        "fixed_k_policy": "Use known dataset class count as K for KMeans.",
        "base_params": BASE_PARAMS,
        "search_config": SEARCH_CONFIG,
    }
    metadata_path = TABLES_DIR / "00_experiment_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote metadata to: {metadata_path}")
    print(f"Wrote tuning trials to: {tuning_path}")
    print(f"Wrote formal results to: {final_path}")
    print(f"Wrote method summary to: {summary_path}")
    print(f"Wrote dataset winners to: {winners_path}")


if __name__ == "__main__":
    main()


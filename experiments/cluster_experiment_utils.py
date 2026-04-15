"""Shared helpers for clustering-oriented imputation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

from utils.clustering import cluster_and_score
from utils.missingness import inject_mcar_missing


RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASET_CLUSTER_COUNTS = {
    "iris": 3,
    "wine": 3,
    "seeds": 3,
    "pendigits": 10,
    "twomoons": 2,
    "flame": 2,
}

DATASET_MAX_SAMPLES = {
    "pendigits": 800,
}


@dataclass(slots=True)
class DatasetSpec:
    dataset_name: str
    file_name: str
    label_column: str | None
    drop_columns: list[str]


def build_dataset_specs(manifest: list[dict[str, object]]) -> list[DatasetSpec]:
    wanted = ["iris", "wine", "seeds", "pendigits", "twomoons", "flame"]
    specs: list[DatasetSpec] = []
    for item in manifest:
        name = str(item["dataset_name"])
        if name not in wanted:
            continue
        specs.append(
            DatasetSpec(
                dataset_name=name,
                file_name=str(item["file_name"]),
                label_column=item.get("label_column"),
                drop_columns=list(item.get("extra_drop_columns", [])),
            )
        )
    specs.sort(key=lambda spec: wanted.index(spec.dataset_name))
    return specs


def maybe_subsample(
    X: np.ndarray,
    y: np.ndarray | None,
    dataset_name: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    max_samples = DATASET_MAX_SAMPLES.get(dataset_name)
    if max_samples is None or X.shape[0] <= max_samples:
        return X, y
    indices = np.arange(X.shape[0])
    stratify = y if y is not None else None
    kept, _ = train_test_split(
        indices,
        train_size=max_samples,
        random_state=random_state,
        stratify=stratify,
    )
    kept = np.sort(kept)
    return X[kept], (y[kept] if y is not None else None)


def masked_error_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    missing_mask: np.ndarray,
) -> tuple[float, float]:
    masked_true = X_true[missing_mask]
    masked_pred = X_pred[missing_mask]
    if masked_true.size == 0:
        return 0.0, 0.0
    rmse = float(np.sqrt(np.mean((masked_true - masked_pred) ** 2)))
    scale = float(np.nanstd(masked_true))
    nrmse = rmse / max(scale, 1e-8)
    return rmse, nrmse


def evaluate_imputed_clustering(
    X_complete: np.ndarray,
    X_missing: np.ndarray,
    X_imputed: np.ndarray,
    missing_mask: np.ndarray,
    method_name: str,
    n_clusters: int,
    random_state: int,
    y_true: np.ndarray | None = None,
) -> dict[str, float]:
    keep_mask = (
        ~np.isnan(X_missing).any(axis=1)
        if method_name == "deletion"
        else np.ones(X_complete.shape[0], dtype=bool)
    )
    X_eval_true = X_complete[keep_mask]
    X_eval_missing = missing_mask[keep_mask]
    X_eval_imputed = X_imputed
    y_eval = y_true[keep_mask] if y_true is not None else None

    full_labels, _ = cluster_and_score(X_eval_true, n_clusters=n_clusters, random_state=random_state)
    imputed_labels, cluster_metrics = cluster_and_score(
        X_eval_imputed,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    masked_rmse, masked_nrmse = masked_error_metrics(X_eval_true, X_eval_imputed, X_eval_missing)

    result = {
        "row_retention": float(keep_mask.mean()),
        "remaining_rows": int(X_eval_imputed.shape[0]),
        "cluster_consistency_ari": float(adjusted_rand_score(full_labels, imputed_labels)),
        "silhouette": cluster_metrics["silhouette"],
        "davies_bouldin": cluster_metrics["davies_bouldin"],
        "calinski_harabasz": cluster_metrics["calinski_harabasz"],
        "masked_rmse": masked_rmse,
        "masked_nrmse": masked_nrmse,
        "ari_to_truth": np.nan,
        "nmi_to_truth": np.nan,
    }
    if y_eval is not None:
        result["ari_to_truth"] = float(adjusted_rand_score(y_eval, imputed_labels))
        result["nmi_to_truth"] = float(normalized_mutual_info_score(y_eval, imputed_labels))
    return result

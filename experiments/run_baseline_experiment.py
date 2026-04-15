"""Run a compact clustering-oriented baseline experiment over built-in datasets."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from datasets import DatasetLoader, load_dataset_manifest
from imputers.registry import DEFAULT_REGISTRY

RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'baseline_experiment'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_METHODS = [
    'deletion',
    'mean',
    'median',
    'knni',
    'em',
    'mice',
    'missforest',
    'iterative_xgboost',
    'gain',
    'diffputer',
    'grape',
]

METHOD_PARAMS: dict[str, dict[str, Any]] = {
    'deletion': {},
    'mean': {},
    'median': {},
    'knni': {'n_neighbors': 5, 'random_state': 42},
    'em': {'maxit': 100, 'convergence_threshold': 1e-5, 'random_state': 42},
    'mice': {'n_imputations': 2, 'max_iter': 10, 'tol': 1e-3, 'random_state': 42},
    'missforest': {'n_estimators': 40, 'max_iter': 10, 'random_state': 42},
    'iterative_xgboost': {'max_iter': 5, 'n_estimators': 40, 'random_state': 42},
    'gain': {'batch_size': 64, 'n_epochs': 40, 'hint_rate': 0.8, 'loss_alpha': 10.0, 'random_state': 42},
    'diffputer': {'hid_dim': 96, 'num_steps': 8, 'num_trials': 1, 'max_epochs': 25, 'batch_size': 128, 'random_state': 42},
    'grape': {'hidden_dim': 48, 'max_epochs': 40, 'learning_rate': 1e-3, 'random_state': 42},
}

DATASET_CLUSTER_COUNTS = {
    'iris': 3,
    'wine': 3,
    'seeds': 3,
    'pendigits': 10,
    'twomoons': 2,
    'flame': 2,
}

DATASET_MAX_SAMPLES = {
    'pendigits': 2500,
}

MISSING_RATE = 0.20
MISSING_SCHEME = 'MCAR'
RANDOM_SEED = 42


@dataclass(slots=True)
class DatasetSpec:
    dataset_name: str
    file_name: str
    label_column: str | None
    drop_columns: list[str]


def build_dataset_specs() -> list[DatasetSpec]:
    manifest = load_dataset_manifest(RAW_DIR)
    wanted = {'iris', 'wine', 'seeds', 'pendigits', 'twomoons', 'flame'}
    specs: list[DatasetSpec] = []
    for item in manifest:
        name = str(item['dataset_name'])
        if name not in wanted:
            continue
        specs.append(
            DatasetSpec(
                dataset_name=name,
                file_name=str(item['file_name']),
                label_column=item.get('label_column'),
                drop_columns=list(item.get('extra_drop_columns', [])),
            )
        )
    specs.sort(key=lambda spec: ['iris', 'wine', 'seeds', 'pendigits', 'twomoons', 'flame'].index(spec.dataset_name))
    return specs


def inject_mcar_missing(X: np.ndarray, missing_rate: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    missing_mask = rng.random(X.shape) < missing_rate
    if X.shape[1] > 0:
        for col in range(X.shape[1]):
            if missing_mask[:, col].all():
                missing_mask[rng.integers(0, X.shape[0]), col] = False
    if X.shape[0] > 0:
        for row in range(X.shape[0]):
            if missing_mask[row, :].all():
                missing_mask[row, rng.integers(0, X.shape[1])] = False
    X_missing = X.copy()
    X_missing[missing_mask] = np.nan
    return X_missing, missing_mask


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
    X_sub = X[kept]
    y_sub = y[kept] if y is not None else None
    return X_sub, y_sub


def cluster_and_score(X: np.ndarray, n_clusters: int, random_state: int) -> tuple[np.ndarray, dict[str, float]]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    metrics = {
        'silhouette': float(silhouette_score(X_scaled, labels)),
        'davies_bouldin': float(davies_bouldin_score(X_scaled, labels)),
        'calinski_harabasz': float(calinski_harabasz_score(X_scaled, labels)),
    }
    return labels, metrics


def masked_error_metrics(X_true: np.ndarray, X_pred: np.ndarray, missing_mask: np.ndarray) -> tuple[float, float]:
    masked_true = X_true[missing_mask]
    masked_pred = X_pred[missing_mask]
    if masked_true.size == 0:
        return 0.0, 0.0
    rmse = float(np.sqrt(np.mean((masked_true - masked_pred) ** 2)))
    scale = float(np.nanstd(masked_true))
    nrmse = rmse / max(scale, 1e-8)
    return rmse, nrmse


def evaluate_method(
    method_name: str,
    X_complete: np.ndarray,
    X_missing: np.ndarray,
    missing_mask: np.ndarray,
    y_true: np.ndarray | None,
    n_clusters: int,
    random_state: int,
) -> dict[str, Any]:
    params = dict(METHOD_PARAMS.get(method_name, {}))
    start = perf_counter()
    imputer = DEFAULT_REGISTRY.build(method_name, **params)
    X_imputed = imputer.fit_transform(X_missing)
    runtime_sec = perf_counter() - start

    if method_name == 'deletion':
        keep_mask = ~np.isnan(X_missing).any(axis=1)
    else:
        keep_mask = np.ones(X_complete.shape[0], dtype=bool)

    X_eval_true = X_complete[keep_mask]
    X_eval_missing = missing_mask[keep_mask]
    X_eval_imputed = X_imputed
    y_eval = y_true[keep_mask] if y_true is not None else None

    full_labels, _ = cluster_and_score(X_eval_true, n_clusters=n_clusters, random_state=random_state)
    imputed_labels, cluster_metrics = cluster_and_score(X_eval_imputed, n_clusters=n_clusters, random_state=random_state)

    masked_rmse, masked_nrmse = masked_error_metrics(X_eval_true, X_eval_imputed, X_eval_missing)
    result = {
        'success': True,
        'runtime_sec': float(runtime_sec),
        'row_retention': float(keep_mask.mean()),
        'remaining_rows': int(X_eval_imputed.shape[0]),
        'cluster_consistency_ari': float(adjusted_rand_score(full_labels, imputed_labels)),
        'silhouette': cluster_metrics['silhouette'],
        'davies_bouldin': cluster_metrics['davies_bouldin'],
        'calinski_harabasz': cluster_metrics['calinski_harabasz'],
        'masked_rmse': masked_rmse,
        'masked_nrmse': masked_nrmse,
        'ari_to_truth': np.nan,
        'nmi_to_truth': np.nan,
    }
    if y_eval is not None:
        result['ari_to_truth'] = float(adjusted_rand_score(y_eval, imputed_labels))
        result['nmi_to_truth'] = float(normalized_mutual_info_score(y_eval, imputed_labels))
    return result


def main() -> None:
    loader = DatasetLoader(data_root=RAW_DIR)
    records: list[dict[str, Any]] = []

    for spec in build_dataset_specs():
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

        for method_name in BASELINE_METHODS:
            try:
                metrics = evaluate_method(
                    method_name=method_name,
                    X_complete=X,
                    X_missing=X_missing,
                    missing_mask=missing_mask,
                    y_true=y,
                    n_clusters=n_clusters,
                    random_state=RANDOM_SEED,
                )
                metrics['error'] = None
                metrics['success'] = True
            except Exception as exc:  # noqa: BLE001
                metrics = {
                    'success': False,
                    'runtime_sec': np.nan,
                    'row_retention': np.nan,
                    'remaining_rows': np.nan,
                    'cluster_consistency_ari': np.nan,
                    'silhouette': np.nan,
                    'davies_bouldin': np.nan,
                    'calinski_harabasz': np.nan,
                    'masked_rmse': np.nan,
                    'masked_nrmse': np.nan,
                    'ari_to_truth': np.nan,
                    'nmi_to_truth': np.nan,
                    'error': str(exc),
                }
            records.append(
                {
                    'dataset_name': spec.dataset_name,
                    'method': method_name,
                    'missing_scheme': MISSING_SCHEME,
                    'missing_rate': MISSING_RATE,
                    'seed': RANDOM_SEED,
                    'n_samples': int(X.shape[0]),
                    'n_features': int(X.shape[1]),
                    'n_clusters': n_clusters,
                    **metrics,
                }
            )

    detailed = pd.DataFrame(records)
    detailed_path = RESULTS_DIR / 'baseline_detailed_results.csv'
    detailed.to_csv(detailed_path, index=False)

    summary = (
        detailed.groupby('method', dropna=False)
        .agg(
            success_rate=('success', 'mean'),
            mean_cluster_consistency_ari=('cluster_consistency_ari', 'mean'),
            mean_ari_to_truth=('ari_to_truth', 'mean'),
            mean_nmi_to_truth=('nmi_to_truth', 'mean'),
            mean_silhouette=('silhouette', 'mean'),
            mean_masked_rmse=('masked_rmse', 'mean'),
            mean_masked_nrmse=('masked_nrmse', 'mean'),
            mean_runtime_sec=('runtime_sec', 'mean'),
            mean_row_retention=('row_retention', 'mean'),
        )
        .reset_index()
    )
    summary['score_cluster'] = summary['mean_cluster_consistency_ari'].rank(pct=True)
    summary['score_truth'] = summary['mean_ari_to_truth'].fillna(summary['mean_ari_to_truth'].mean()).rank(pct=True)
    summary['score_nrmse'] = (-summary['mean_masked_nrmse']).rank(pct=True)
    summary['score_runtime'] = (-summary['mean_runtime_sec']).rank(pct=True)
    summary['score_retention'] = summary['mean_row_retention'].rank(pct=True)
    summary['overall_score'] = summary[['score_cluster', 'score_truth', 'score_nrmse', 'score_runtime', 'score_retention']].mean(axis=1)
    summary = summary.sort_values('overall_score', ascending=False)
    summary_path = RESULTS_DIR / 'baseline_summary.csv'
    summary.to_csv(summary_path, index=False)

    metadata = {
        'experiment_name': 'baseline_clustering_imputation_benchmark',
        'datasets': [spec.dataset_name for spec in build_dataset_specs()],
        'methods': BASELINE_METHODS,
        'missing_scheme': MISSING_SCHEME,
        'missing_rate': MISSING_RATE,
        'random_seed': RANDOM_SEED,
        'dataset_max_samples': DATASET_MAX_SAMPLES,
        'method_params': METHOD_PARAMS,
    }
    metadata_path = RESULTS_DIR / 'baseline_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'Wrote detailed results to: {detailed_path}')
    print(f'Wrote summary results to: {summary_path}')
    print(f'Wrote metadata to: {metadata_path}')


if __name__ == '__main__':
    main()

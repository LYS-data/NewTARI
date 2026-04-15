"""Reusable clustering evaluation helpers for clustering-oriented imputation studies."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def cluster_and_score(
    X: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Cluster the input matrix with KMeans and compute internal indices."""

    X = np.asarray(X, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    metrics = {
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
    }
    return labels, metrics

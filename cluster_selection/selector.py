"""Placeholder clusterer selector."""

from __future__ import annotations

from typing import Any

import numpy as np

from cluster_selection.base import BaseClusterSelector


class DefaultClusterSelector(BaseClusterSelector):
    """Placeholder implementation for future clusterer selection."""

    def select_default_clusterer(
        self,
        X_imputed: np.ndarray,
        candidate_clusterers: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        _ = (X_imputed, kwargs)
        if candidate_clusterers:
            return candidate_clusterers[0]
        return "kmeans"


def select_default_clusterer(
    X_imputed: np.ndarray,
    candidate_clusterers: list[str] | None = None,
    **kwargs: Any,
) -> str:
    selector = DefaultClusterSelector()
    return selector.select_default_clusterer(X_imputed, candidate_clusterers, **kwargs)

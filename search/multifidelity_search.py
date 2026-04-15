"""Placeholder interfaces for future multi-fidelity strategy search."""

from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_strategy(
    strategy: dict[str, Any],
    X_missing: np.ndarray,
    clusterer: str,
    fidelity_config: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Placeholder strategy evaluation result."""
    _ = (strategy, X_missing, clusterer, fidelity_config, kwargs)
    return {
        "strategy": strategy,
        "clusterer": clusterer,
        "score": None,
        "status": "placeholder",
    }


def search_best_strategy(
    search_space: dict[str, Any],
    X_missing: np.ndarray,
    clusterer: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Future multi-fidelity search entrypoint."""
    _ = (search_space, X_missing, clusterer, kwargs)
    raise NotImplementedError(
        "Multi-fidelity strategy search is reserved for a later research phase."
    )

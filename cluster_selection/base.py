"""Base interfaces for future clusterer selection logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseClusterSelector(ABC):
    """Abstract interface for choosing a clustering backend after imputation."""

    @abstractmethod
    def select_default_clusterer(
        self,
        X_imputed: np.ndarray,
        candidate_clusterers: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Return the preferred clusterer name for downstream evaluation."""

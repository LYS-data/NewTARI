"""Project-level default configuration values."""

from __future__ import annotations

DEFAULT_IMPUTERS = [
    "deletion",
    "mean",
    "median",
    "knni",
    "em",
    "mice",
    "missforest",
]

OPTIONAL_IMPUTERS = [
    "diffputer",
    "grape",
    "gain",
    "miwae",
    "hivae",
    "iterative_xgboost",
    "nomi",
]

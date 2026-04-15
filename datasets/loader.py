"""Dataset loading helpers for clustering-oriented numeric imputation research."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    """Container for loaded dataset artifacts.

    The current project focuses on numeric feature imputation for clustering.
    Therefore ``X`` is always the cleaned numeric feature matrix. Label-like
    columns can optionally be returned in ``y`` for bookkeeping, but they are
    intentionally excluded from the imputation pipeline by default.
    """

    dataset_name: str
    X: np.ndarray
    feature_names: list[str]
    y: pd.Series | None = None
    label_column: str | None = None
    original_frame: pd.DataFrame | None = None


class DatasetLoader:
    """Load CSV datasets and keep only clustering-relevant numeric features."""

    def __init__(self, data_root: str | Path = "data/raw") -> None:
        self.data_root = Path(data_root)

    def load_csv(
        self,
        file_path: str | Path,
        *,
        dataset_name: str | None = None,
        label_column: str | None = None,
        drop_columns: list[str] | None = None,
        numeric_only: bool = True,
        keep_original_frame: bool = False,
    ) -> DatasetBundle:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.data_root / path
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        frame = pd.read_csv(path)
        return self.load_dataframe(
            frame,
            dataset_name=dataset_name or path.stem,
            label_column=label_column,
            drop_columns=drop_columns,
            numeric_only=numeric_only,
            keep_original_frame=keep_original_frame,
        )

    def load_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        dataset_name: str = "dataset",
        label_column: str | None = None,
        drop_columns: list[str] | None = None,
        numeric_only: bool = True,
        keep_original_frame: bool = False,
    ) -> DatasetBundle:
        working = frame.copy()
        y: pd.Series | None = None

        if label_column is not None:
            if label_column not in working.columns:
                raise ValueError(f"label_column '{label_column}' not found in dataset.")
            y = working[label_column].copy()
            working = working.drop(columns=[label_column])

        if drop_columns:
            missing = [column for column in drop_columns if column not in working.columns]
            if missing:
                raise ValueError(f"drop_columns contains unknown columns: {missing}")
            working = working.drop(columns=drop_columns)

        if numeric_only:
            numeric_frame = working.select_dtypes(include=[np.number])
            if numeric_frame.shape[1] == 0:
                raise ValueError("No numeric feature columns remain after filtering.")
            working = numeric_frame
        else:
            non_numeric = [column for column in working.columns if not pd.api.types.is_numeric_dtype(working[column])]
            if non_numeric:
                raise TypeError(
                    "Non-numeric columns remain in feature matrix: "
                    f"{non_numeric}. Use numeric_only=True or drop those columns explicitly."
                )

        X = working.to_numpy(dtype=float)
        return DatasetBundle(
            dataset_name=dataset_name,
            X=X,
            feature_names=working.columns.tolist(),
            y=y,
            label_column=label_column,
            original_frame=frame.copy() if keep_original_frame else None,
        )


def load_dataset(
    file_path: str | Path,
    *,
    dataset_name: str | None = None,
    label_column: str | None = None,
    drop_columns: list[str] | None = None,
    numeric_only: bool = True,
    keep_original_frame: bool = False,
    data_root: str | Path = "data/raw",
) -> DatasetBundle:
    """Convenience wrapper for loading a CSV dataset."""
    loader = DatasetLoader(data_root=data_root)
    return loader.load_csv(
        file_path,
        dataset_name=dataset_name,
        label_column=label_column,
        drop_columns=drop_columns,
        numeric_only=numeric_only,
        keep_original_frame=keep_original_frame,
    )


def load_dataset_manifest(data_root: str | Path = "data/raw") -> list[dict[str, Any]]:
    """Load the local dataset manifest if it exists."""
    root = Path(data_root)
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("dataset_manifest.json must contain a list of dataset descriptors.")
    return data


def list_available_datasets(data_root: str | Path = "data/raw") -> list[str]:
    """List dataset names registered in the local manifest."""
    manifest = load_dataset_manifest(data_root=data_root)
    return [str(item["dataset_name"]) for item in manifest if "dataset_name" in item]

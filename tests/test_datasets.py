"""Unit tests for dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from datasets.loader import DatasetLoader, list_available_datasets, load_dataset_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [1.0, 2.0, None],
            "f2": [0.5, None, 1.5],
            "species": ["setosa", "versicolor", "virginica"],
            "id": [101, 102, 103],
        }
    )


def test_loader_drops_label_column_from_feature_matrix() -> None:
    loader = DatasetLoader()
    bundle = loader.load_dataframe(make_frame(), dataset_name="iris_like", label_column="species")

    assert bundle.dataset_name == "iris_like"
    assert bundle.label_column == "species"
    assert bundle.y is not None
    assert bundle.feature_names == ["f1", "f2", "id"]
    assert bundle.X.shape == (3, 3)


def test_loader_can_drop_extra_columns() -> None:
    loader = DatasetLoader()
    bundle = loader.load_dataframe(
        make_frame(),
        dataset_name="iris_like",
        label_column="species",
        drop_columns=["id"],
    )

    assert bundle.feature_names == ["f1", "f2"]
    assert bundle.X.shape == (3, 2)


def test_loader_raises_when_non_numeric_columns_remain() -> None:
    loader = DatasetLoader()
    with pytest.raises(TypeError):
        loader.load_dataframe(make_frame(), dataset_name="iris_like", numeric_only=False)


def test_manifest_lists_recommended_datasets() -> None:
    manifest = load_dataset_manifest(RAW_DIR)
    names = {item["dataset_name"] for item in manifest}

    assert {"iris", "wine", "seeds", "pendigits", "twomoons", "flame"}.issubset(names)


def test_list_available_datasets_reads_manifest() -> None:
    names = set(list_available_datasets(RAW_DIR))

    assert {"iris", "wine", "seeds", "pendigits", "twomoons", "flame"}.issubset(names)


@pytest.mark.parametrize(
    ("file_name", "label_column", "drop_columns", "expected_shape"),
    [
        ("iris.csv", "species", ["target"], (150, 4)),
        ("wine.csv", "target_name", ["target"], (178, 13)),
        ("seeds.csv", None, None, (210, 7)),
        ("pendigits.csv", "target", None, (10992, 16)),
        ("twomoons.csv", "target", None, (300, 2)),
        ("flame.csv", "target", None, (240, 2)),
    ],
)
def test_recommended_dataset_files_can_be_loaded(
    file_name: str,
    label_column: str | None,
    drop_columns: list[str] | None,
    expected_shape: tuple[int, int],
) -> None:
    loader = DatasetLoader(data_root=RAW_DIR)
    bundle = loader.load_csv(
        file_name,
        dataset_name=Path(file_name).stem,
        label_column=label_column,
        drop_columns=drop_columns,
    )

    assert bundle.X.shape == expected_shape
    assert len(bundle.feature_names) == expected_shape[1]

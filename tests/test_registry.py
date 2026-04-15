"""Unit tests for the imputer registry."""

from __future__ import annotations

from imputers.registry import DEFAULT_REGISTRY, ImputerRegistry, build_imputer


def test_registry_lists_retained_imputers() -> None:
    registry = ImputerRegistry()
    names = registry.list_imputers()
    assert names == ["deletion", "diffputer", "em", "gain", "grape", "hivae", "iterative_xgboost", "knni", "mean", "median", "mice", "missforest", "miwae", "nomi"]


def test_build_imputer_uses_default_params() -> None:
    imputer = build_imputer("knni")
    assert imputer.get_name() == "knni"
    assert imputer.get_params()["n_neighbors"] == 5


def test_registry_describe_optional_method() -> None:
    description = DEFAULT_REGISTRY.describe("gain")
    assert description["name"] == "gain"
    assert "available" in description
    assert "default_params" in description


def test_registry_describe_xgboost_method() -> None:
    description = DEFAULT_REGISTRY.describe("iterative_xgboost")
    assert description["name"] == "iterative_xgboost"
    assert description["requires"] == ["xgboost"]


def test_registry_describe_nomi_method() -> None:
    description = DEFAULT_REGISTRY.describe("nomi")
    assert description["name"] == "nomi"
    assert description["requires"] == ["tensorflow", "hnswlib", "neural_tangents"]


def test_registry_describe_diffputer_method() -> None:
    description = DEFAULT_REGISTRY.describe("diffputer")
    assert description["name"] == "diffputer"
    assert description["requires"] == ["torch"]


def test_registry_describe_grape_method() -> None:
    description = DEFAULT_REGISTRY.describe("grape")
    assert description["name"] == "grape"
    assert description["requires"] == ["torch", "torch_geometric", "torch_scatter", "torch_sparse"]

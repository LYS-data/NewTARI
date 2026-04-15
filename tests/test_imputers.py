"""Unit tests for the imputation module."""

from __future__ import annotations

import numpy as np
import pytest

from imputers.diffputer import HAS_TORCH as HAS_TORCH_DIFFPUTER
from imputers.grape import HAS_GRAPE_DEPS
from imputers.gain import HAS_TENSORFLOW as HAS_TENSORFLOW_GAIN
from imputers.hivae import HAS_TENSORFLOW
from imputers.miwae import HAS_TORCH as HAS_TORCH_MIWAE
from imputers.registry import DEFAULT_REGISTRY, build_imputer


def make_matrix(seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(24, 5))
    X[rng.random(X.shape) < 0.25] = np.nan
    return X


@pytest.mark.parametrize("name", DEFAULT_REGISTRY.list_imputers(available_only=True))
def test_all_available_imputers_run_without_nan(name: str) -> None:
    X = make_matrix()
    imputer = build_imputer(name)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape[1] == X.shape[1]
    if name != "deletion":
        assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()
    assert np.isfinite(X_imputed).all()


def test_invalid_non_numeric_input_raises() -> None:
    with pytest.raises(TypeError):
        build_imputer("mean").fit_transform([["a", "b"], ["c", "d"]])


def test_invalid_one_dimensional_input_raises() -> None:
    with pytest.raises(ValueError):
        build_imputer("mean").fit_transform(np.array([1.0, np.nan, 2.0]))


def test_all_missing_column_raises_by_default() -> None:
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    with pytest.raises(ValueError):
        build_imputer("mean").fit_transform(X)


def test_all_missing_column_constant_policy_fallback() -> None:
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    imputer = build_imputer("mean", all_missing_policy="constant", all_missing_fill_value=-1.0)
    X_imputed = imputer.fit_transform(X)
    assert np.allclose(X_imputed[:, 1], -1.0)


def test_deletion_imputer_drops_rows_with_missing_values() -> None:
    X = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0], [5.0, np.nan]])
    X_out = build_imputer("deletion").fit_transform(X)
    assert X_out.shape == (2, 2)
    assert not np.isnan(X_out).any()


def test_em_imputer_runs() -> None:
    X = make_matrix(seed=3)
    imputer = build_imputer("em", maxit=20, convergence_threshold=1e-5, random_state=0)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()


def test_em_imputer_can_transform_new_data_without_falling_back_to_nan() -> None:
    X_train = make_matrix(seed=31)
    X_test = make_matrix(seed=32)[:8]
    imputer = build_imputer("em", maxit=30, convergence_threshold=1e-5, random_state=0)
    imputer.fit(X_train)
    X_imputed = imputer.transform(X_test)
    assert X_imputed.shape == X_test.shape
    assert not np.isnan(X_imputed).any()


def test_mice_imputer_runs() -> None:
    X = make_matrix(seed=5)
    imputer = build_imputer("mice", n_imputations=2, max_iter=5, random_state=0)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()


def test_missforest_imputer_runs() -> None:
    X = make_matrix(seed=9)
    imputer = build_imputer("missforest", n_estimators=20, max_iter=5, random_state=0)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()


@pytest.mark.skipif(not HAS_TORCH_DIFFPUTER, reason="torch is not installed in this interpreter")
def test_diffputer_imputer_runs_with_torch() -> None:
    X = make_matrix(seed=15)
    imputer = build_imputer(
        "diffputer",
        hid_dim=64,
        num_steps=8,
        num_trials=1,
        max_epochs=5,
        batch_size=16,
        early_stopping_patience=3,
        random_state=0,
    )
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()



def test_iterative_xgboost_imputer_runs_when_available() -> None:
    description = DEFAULT_REGISTRY.describe("iterative_xgboost")
    if not description["available"]:
        pytest.skip("xgboost is not installed in this interpreter")
    X = make_matrix(seed=17)
    imputer = build_imputer("iterative_xgboost", max_iter=3, n_estimators=20, random_state=0)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()



@pytest.mark.skipif(not HAS_GRAPE_DEPS, reason="official GRAPE dependencies are not installed in this interpreter")
def test_grape_imputer_runs_with_torch() -> None:
    X = make_matrix(seed=19)
    imputer = build_imputer(
        "grape",
        hidden_dim=32,
        max_epochs=10,
        early_stopping_patience=4,
        random_state=0,
    )
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()



@pytest.mark.skipif(not HAS_TENSORFLOW_GAIN, reason="tensorflow is not installed in this interpreter")
def test_gain_imputer_runs_with_torch() -> None:
    X = make_matrix(seed=7)
    imputer = build_imputer("gain", n_epochs=5, batch_size=8, random_state=0)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()


@pytest.mark.skipif(not HAS_TORCH_MIWAE, reason="torch is not installed in this interpreter")
def test_miwae_imputer_runs_with_torch() -> None:
    X = make_matrix(seed=13)
    imputer = build_imputer("miwae", n_epochs=10, batch_size=8, latent_size=2, n_hidden=8, K=5, random_state=0)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="tensorflow is not installed in this interpreter")
def test_hivae_imputer_runs_with_tensorflow() -> None:
    X = make_matrix(seed=21)
    imputer = build_imputer(
        "hivae",
        epochs=2,
        batch_size=8,
        dim_latent_z=2,
        dim_latent_y=2,
        dim_latent_s=2,
        display_epoch=1,
        random_state=0,
    )
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()



def test_hivae_registry_entry_exists() -> None:
    description = DEFAULT_REGISTRY.describe("hivae")
    assert description["name"] == "hivae"
    assert description["requires"] == ["tensorflow"]
    if not HAS_TENSORFLOW:
        assert description["available"] is False


def test_nomi_registry_entry_exists() -> None:
    description = DEFAULT_REGISTRY.describe("nomi")
    assert description["name"] == "nomi"
    assert description["requires"] == ["tensorflow", "hnswlib", "neural_tangents"]


def test_nomi_imputer_runs_when_available() -> None:
    description = DEFAULT_REGISTRY.describe("nomi")
    if not description["available"]:
        pytest.skip("nomi optional dependencies are not installed in this interpreter")
    X = make_matrix(seed=41)
    imputer = build_imputer("nomi", k_neighbors=5, max_iterations=2, random_state=0)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == X.shape
    assert not np.isnan(X_imputed).any()

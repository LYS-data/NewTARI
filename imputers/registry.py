"""Registry and factory for imputation methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from imputers.base import BaseImputer
from imputers.deletion import DeletionImputer
from imputers.diffputer import DiffPuterImputer, HAS_TORCH as HAS_TORCH_DIFFPUTER
from imputers.grape import GRAPEImputer, HAS_GRAPE_DEPS
from imputers.em import EMImputer
from imputers.iterative import HAS_XGBOOST, IterativeXGBoostImputer
from imputers.gain import GAINImputer, HAS_TENSORFLOW as HAS_TENSORFLOW_GAIN
from imputers.hivae import HIVAEImputer, HAS_TENSORFLOW
from imputers.knn import KNNIImputer
from imputers.nomi import HAS_NOMI_DEPS, NOMIImputer
from imputers.mice import MICEImputer
from imputers.missforest import MissForestImputer
from imputers.miwae import MIWAEImputer, HAS_TORCH as HAS_TORCH_MIWAE
from imputers.simple import MeanImputer, MedianImputer


@dataclass(slots=True)
class ImputerSpec:
    name: str
    builder: Callable[..., BaseImputer]
    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    available: bool = True
    requires: tuple[str, ...] = ()
    notes: str | None = None


class ImputerRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, ImputerSpec] = self._build_default_registry()

    @staticmethod
    def _build_default_registry() -> dict[str, ImputerSpec]:
        return {
            "deletion": ImputerSpec("deletion", DeletionImputer, notes="Listwise deletion baseline that removes rows with missing values."),
            "mean": ImputerSpec("mean", MeanImputer),
            "median": ImputerSpec("median", MedianImputer),
            "knni": ImputerSpec(
                "knni",
                KNNIImputer,
                default_params={"n_neighbors": 5},
                search_space={"n_neighbors": [3, 5, 7]},
                notes="Single retained K-nearest-neighbor imputation method.",
            ),
            "diffputer": ImputerSpec(
                "diffputer",
                DiffPuterImputer,
                default_params={"hid_dim": 128, "max_iter": 1, "num_steps": 10, "num_trials": 2, "max_epochs": 50, "batch_size": 128},
                search_space={"hid_dim": [64, 128], "max_iter": [1, 2], "num_steps": [10, 20], "num_trials": [1, 2], "max_epochs": [30, 50]},
                available=HAS_TORCH_DIFFPUTER,
                requires=("torch",),
                notes="Official DiffPuter model and diffusion sampler vendored from the original repository, wrapped for direct matrix input.",
            ),
            "em": ImputerSpec(
                "em",
                EMImputer,
                default_params={"maxit": 200, "convergence_threshold": 1e-6},
                search_space={"maxit": [100, 200], "convergence_threshold": [1e-5, 1e-6]},
                notes="Gaussian EM imputer with mean fallback for numerical instability.",
            ),
            "mice": ImputerSpec(
                "mice",
                MICEImputer,
                default_params={
                    "n_imputations": 3,
                    "max_iter": 20,
                    "tol": 1e-3,
                    "initial_strategy": "mean",
                    "imputation_order": "ascending",
                },
                search_space={
                    "n_imputations": [1, 3],
                    "max_iter": [10, 20],
                    "tol": [1e-2, 1e-3],
                    "initial_strategy": ["mean", "median"],
                    "imputation_order": ["ascending", "random"],
                },
                notes="Multiple-imputation chained equations approximation using sklearn IterativeImputer.",
            ),
            "missforest": ImputerSpec(
                "missforest",
                MissForestImputer,
                default_params={
                    "n_estimators": 50,
                    "max_iter": 20,
                    "initial_strategy": "mean",
                    "imputation_order": "ascending",
                },
                search_space={
                    "n_estimators": [30, 50],
                    "max_iter": [10, 20],
                    "initial_strategy": ["mean", "median"],
                    "imputation_order": ["ascending", "random"],
                },
                notes="Approximate MissForest implementation using IterativeImputer with RandomForestRegressor.",
            ),
            "grape": ImputerSpec(
                "grape",
                GRAPEImputer,
                default_params={"hidden_dim": 64, "max_epochs": 100, "learning_rate": 1e-3},
                search_space={"hidden_dim": [32, 64], "max_epochs": [50, 100], "dropout": [0.0, 0.1]},
                available=HAS_GRAPE_DEPS,
                requires=("torch", "torch_geometric", "torch_scatter", "torch_sparse"),
                notes="Official GRAPE source dependency stack wired in; full matrix-input adaptation is still pending.",
            ),
            "gain": ImputerSpec(
                "gain",
                GAINImputer,
                default_params={"batch_size": 64, "n_epochs": 100, "hint_rate": 0.9, "loss_alpha": 100.0},
                search_space={"batch_size": [32, 64], "n_epochs": [50, 100], "hint_rate": [0.8, 0.9], "loss_alpha": [10.0, 100.0]},
                available=HAS_TENSORFLOW_GAIN,
                requires=("tensorflow",),
                notes="Official GAIN core from jsyoon0823/GAIN wrapped for direct matrix input.",
            ),
            "miwae": ImputerSpec(
                "miwae",
                MIWAEImputer,
                default_params={"n_epochs": 200, "batch_size": 128, "latent_size": 3, "n_hidden": 32, "K": 20},
                search_space={"latent_size": [2, 3], "n_hidden": [16, 32], "K": [10, 20]},
                available=HAS_TORCH_MIWAE,
                requires=("torch",),
                notes="MIWAE deep generative imputer implemented with PyTorch.",
            ),
            "hivae": ImputerSpec(
                "hivae",
                HIVAEImputer,
                default_params={"dim_latent_z": 2, "dim_latent_y": 3, "dim_latent_s": 4, "batch_size": 128, "epochs": 50},
                search_space={"dim_latent_z": [2, 3], "dim_latent_y": [2, 3], "dim_latent_s": [3, 4]},
                available=HAS_TENSORFLOW,
                requires=("tensorflow",),
                notes="Numeric-only HI-VAE wrapper over the vendored TensorFlow 1.x implementation.",
            ),
            "nomi": ImputerSpec(
                "nomi",
                NOMIImputer,
                default_params={"k_neighbors": 10, "max_iterations": 3, "tau": 1.0, "beta": 1.0},
                search_space={"k_neighbors": [5, 10], "max_iterations": [2, 3], "tau": [0.8, 1.0], "beta": [0.5, 1.0]},
                available=HAS_NOMI_DEPS,
                requires=("tensorflow", "hnswlib", "neural_tangents"),
                notes="Official NOMI core from guaiyoui/NOMI wrapped for direct matrix input.",
            ),
            "iterative_xgboost": ImputerSpec(
                "iterative_xgboost",
                IterativeXGBoostImputer,
                default_params={"max_iter": 10, "n_estimators": 50, "initial_strategy": "mean", "imputation_order": "ascending"},
                search_space={"max_iter": [5, 10], "n_estimators": [30, 50], "initial_strategy": ["mean", "median"], "imputation_order": ["ascending", "random"]},
                available=HAS_XGBOOST,
                requires=("xgboost",),
                notes="XGBoost-backed iterative chained-equations imputer for nonlinear tabular relationships.",
            ),
        }

    def list_imputers(self, *, available_only: bool = False) -> list[str]:
        names = sorted(self._registry)
        if not available_only:
            return names
        return [name for name in names if self._registry[name].available]

    def is_available(self, name: str) -> bool:
        return self.get_spec(name).available

    def get_spec(self, name: str) -> ImputerSpec:
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Unknown imputer: {name}")
        return self._registry[key]

    def get_default_params(self, name: str) -> dict[str, Any]:
        return dict(self.get_spec(name).default_params)

    def get_search_space(self, name: str) -> dict[str, list[Any]]:
        return dict(self.get_spec(name).search_space)

    def describe(self, name: str) -> dict[str, Any]:
        spec = self.get_spec(name)
        return {
            "name": spec.name,
            "available": spec.available,
            "requires": list(spec.requires),
            "default_params": dict(spec.default_params),
            "search_space": dict(spec.search_space),
            "notes": spec.notes,
        }

    def build(self, name: str, **kwargs: Any) -> BaseImputer:
        spec = self.get_spec(name)
        if not spec.available:
            requirements = ", ".join(spec.requires) or "optional dependencies"
            raise ImportError(f"Imputer '{name}' is unavailable because {requirements} is missing.")
        params = {**spec.default_params, **kwargs}
        return spec.builder(**params)


DEFAULT_REGISTRY = ImputerRegistry()


def build_imputer(name: str, **kwargs: Any) -> BaseImputer:
    return DEFAULT_REGISTRY.build(name, **kwargs)

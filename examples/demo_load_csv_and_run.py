"""Demo: load iris.csv, inject missingness, run retained imputers, and save JSON output."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from datasets.loader import DatasetLoader
from runners.imputation_runner import ImputationRunner
from utils.stats import compute_column_missing_rates, compute_missing_rate


def inject_missing_values(X: np.ndarray, *, missing_rate: float = 0.15, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_missing = np.array(X, dtype=float, copy=True)
    mask = rng.random(X_missing.shape) < missing_rate
    X_missing[mask] = np.nan
    return X_missing


def main() -> None:
    loader = DatasetLoader(data_root="data/raw")
    bundle = loader.load_csv(
        "iris.csv",
        dataset_name="iris",
        label_column="species",
        drop_columns=["target"],
    )

    X_missing = inject_missing_values(bundle.X, missing_rate=0.15, seed=42)
    runner = ImputationRunner()
    methods = ["deletion", "mean", "median", "knni", "mice", "missforest"]
    output_dir = Path("results")

    results = runner.run(
        X_missing,
        methods,
        output_json_path=output_dir,
        dataset_name=bundle.dataset_name,
        scenario_name="mcar15",
        run_tag="iris_csv_demo",
    )

    latest_file = sorted(output_dir.glob("imputation_runner_output__iris__mcar15__*.json"))[-1]
    print("Dataset:", bundle.dataset_name)
    print("Feature columns:", bundle.feature_names)
    print("Label column excluded:", bundle.label_column)
    print("Shape:", X_missing.shape)
    print("Overall missing rate:", round(compute_missing_rate(X_missing), 4))
    print("Per-column missing rates:", compute_column_missing_rates(X_missing))
    print("JSON saved to:", latest_file.resolve())
    print("-" * 80)

    for name, result in results.items():
        print(f"Method: {name}")
        print(f"  success: {result['success']}")
        print(f"  runtime_sec: {result['runtime_sec']}")
        print(f"  remaining_nan: {result['remaining_nan']}")
        print(f"  error: {result['error']}")
        print()


if __name__ == "__main__":
    main()

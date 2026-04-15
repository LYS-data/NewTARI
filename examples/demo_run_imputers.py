"""Demo: batch run several retained imputers on a synthetic matrix with missing values."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from runners.imputation_runner import ImputationRunner
from utils.stats import compute_column_missing_rates, compute_missing_rate


def make_demo_matrix(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(60, 6))
    missing_mask = rng.random(X.shape) < 0.2
    X[missing_mask] = np.nan
    return X


def main() -> None:
    X_missing = make_demo_matrix()
    runner = ImputationRunner()
    methods = [
        "deletion",
        "mean",
        "median",
        "knni",
        "mice",
        "missforest",
        "em",
    ]
    output_dir = Path("results")
    results = runner.run(
        X_missing,
        methods,
        output_json_path=output_dir,
        dataset_name="synthetic_numeric_demo",
        scenario_name="random_missing_20pct",
        run_tag="demo",
    )

    latest_file = sorted(output_dir.glob("imputation_runner_output__*.json"))[-1]
    print("Overall missing rate:", round(compute_missing_rate(X_missing), 4))
    print("Per-column missing rates:", compute_column_missing_rates(X_missing))
    print("JSON saved to:", latest_file.resolve())
    print("-" * 80)
    for name, result in results.items():
        print(f"Method: {name}")
        print(f"  success: {result['success']}")
        print(f"  runtime_sec: {result['runtime_sec']}")
        print(f"  remaining_nan: {result['remaining_nan']}")
        print(f"  params: {result['params']}")
        print(f"  error: {result['error']}")
        print()


if __name__ == "__main__":
    main()

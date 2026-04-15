"""Unit tests for the imputation runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from runners.imputation_runner import ImputationRunner
from utils.io import build_standard_json_filename


def make_matrix(seed: int = 321) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(30, 4))
    X[rng.random(X.shape) < 0.2] = np.nan
    return X


def test_runner_can_execute_multiple_methods_without_crashing() -> None:
    runner = ImputationRunner(logger_name="test.runner.basic")
    try:
        X = make_matrix()
        results = runner.run(X, ["mean", "knni", "does_not_exist"])
        assert results["mean"]["success"] is True
        assert results["knni"]["success"] is True
        assert results["mean"]["remaining_nan"] == 0
        assert results["knni"]["remaining_nan"] == 0
        assert results["does_not_exist"]["success"] is False
        assert results["does_not_exist"]["error"] is not None
    finally:
        runner.close()


def test_runner_result_schema_is_standardized() -> None:
    runner = ImputationRunner(logger_name="test.runner.schema")
    try:
        result = runner.run(make_matrix(), "median")["median"]
        assert sorted(result.keys()) == sorted(
            [
                "success",
                "imputed_array",
                "runtime_sec",
                "params",
                "remaining_nan",
                "has_inf",
                "stats_delta",
                "optimization",
                "error",
            ]
        )
    finally:
        runner.close()


def test_standard_json_filename_builder() -> None:
    filename = build_standard_json_filename(
        result_type="imputation_runner_output",
        dataset_name="Demo Dataset",
        scenario_name="MCAR 20%",
        methods=["mean", "knni"],
        run_tag="baseline v1",
        timestamp="20260310_120000",
    )
    assert filename == (
        "imputation_runner_output__demo_dataset__mcar_20__mean_knni__baseline_v1__20260310_120000.json"
    )


def test_runner_can_save_results_to_local_json() -> None:
    runner = ImputationRunner(logger_name="test.runner.save")
    output_dir = Path("tests") / ".artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_name = build_standard_json_filename(
        result_type="imputation_runner_output",
        dataset_name="demo_dataset",
        scenario_name="mcar20",
        methods=["mean", "median"],
        run_tag="unit_test",
        timestamp="20260310_120000",
    )
    output_path = output_dir / expected_name

    try:
        runner.save_results(
            runner.run(make_matrix(), ["mean", "median"]),
            output_path,
            dataset_name="demo_dataset",
            scenario_name="mcar20",
            run_tag="unit_test",
            methods=["mean", "median"],
        )
        assert output_path.exists()
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["storage"] == "local_json"
        assert payload["result_type"] == "imputation_runner_output"
        assert payload["methods"]["mean"]["success"] is True
        assert isinstance(payload["methods"]["mean"]["imputed_array"], list)
    finally:
        runner.close()
        if output_path.exists():
            output_path.unlink()


def test_runner_can_log_and_optimize() -> None:
    output_dir = Path("tests") / ".artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "runner.log"
    runner = ImputationRunner(logger_name="test.runner.log_opt", log_file=log_path)
    try:
        result = runner.run(
            make_matrix(),
            "knni",
            optimize_hyperparams=True,
            tuning_config={"knni": {"max_trials": 2, "early_stopping_patience": 1, "random_state": 0}},
        )["knni"]
        assert result["success"] is True
        assert result["optimization"]["enabled"] is True
        assert result["optimization"]["best_params"]
        assert log_path.exists()
        log_text = log_path.read_text(encoding="utf-8")
        assert "runner started" in log_text
    finally:
        runner.close()
        if log_path.exists():
            log_path.unlink()

"""High-level execution entrypoint for trying many imputers safely."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imputers.registry import DEFAULT_REGISTRY, ImputerRegistry
from runners.imputation_optimizer import ImputerOptimizer
from utils.io import resolve_json_output_path, save_json
from utils.logging_utils import close_logger, configure_logger, log_kv
from utils.stats import compare_basic_statistics, validate_imputed_result
from utils.timing import measure_runtime
from utils.validation import ensure_numeric_matrix


class ImputationRunner:
    """Run one or many imputers and collect standardized outputs."""

    def __init__(
        self,
        registry: ImputerRegistry | None = None,
        *,
        logger_name: str = "imputation_reco.runner",
        log_file: str | Path | None = None,
        log_level: str = "INFO",
    ) -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.logger = configure_logger(logger_name, level=log_level, log_file=log_file)
        self.optimizer = ImputerOptimizer(self.registry, logger_name=f"{logger_name}.optimizer")

    def close(self) -> None:
        close_logger(self.logger)
        close_logger(self.optimizer.logger)

    def run(
        self,
        X: np.ndarray | pd.DataFrame,
        imputers: str | list[str],
        *,
        imputer_params: dict[str, dict[str, Any]] | None = None,
        feature_strategy_plan: dict[int, str] | None = None,
        output_json_path: str | Path | None = None,
        dataset_name: str | None = None,
        scenario_name: str | None = None,
        run_tag: str | None = None,
        optimize_hyperparams: bool = False,
        tuning_config: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        X_array = ensure_numeric_matrix(X)
        names = [imputers] if isinstance(imputers, str) else list(imputers)
        if feature_strategy_plan is not None:
            raise NotImplementedError(
                "feature_strategy_plan is reserved for future feature-wise imputation research."
            )

        log_kv(self.logger, "runner started", imputers=names, optimize_hyperparams=optimize_hyperparams)
        results: dict[str, dict[str, Any]] = {}
        for name in names:
            params = (imputer_params or {}).get(name, {})
            tune_params = (tuning_config or {}).get(name, {})
            result = self._run_single(
                name=name,
                X=X_array,
                params=params,
                optimize_hyperparams=optimize_hyperparams,
                tuning_options=tune_params,
            )
            results[name] = result

        if output_json_path is not None:
            self.save_results(
                results,
                output_json_path,
                dataset_name=dataset_name,
                scenario_name=scenario_name,
                run_tag=run_tag,
                methods=names,
            )
        return results

    def save_results(
        self,
        results: dict[str, dict[str, Any]],
        output_json_path: str | Path,
        *,
        dataset_name: str | None = None,
        scenario_name: str | None = None,
        run_tag: str | None = None,
        methods: list[str] | None = None,
    ) -> Path:
        payload = {
            "storage": "local_json",
            "result_type": "imputation_runner_output",
            "dataset_name": dataset_name,
            "scenario_name": scenario_name,
            "run_tag": run_tag,
            "methods": results,
        }
        final_path = resolve_json_output_path(
            output_json_path,
            result_type="imputation_runner_output",
            dataset_name=dataset_name,
            scenario_name=scenario_name,
            methods=methods,
            run_tag=run_tag,
        )
        saved = save_json(payload, final_path)
        log_kv(self.logger, "results saved", path=saved)
        return saved

    def _run_single(
        self,
        *,
        name: str,
        X: np.ndarray,
        params: dict[str, Any],
        optimize_hyperparams: bool,
        tuning_options: dict[str, Any],
    ) -> dict[str, Any]:
        optimization_payload: dict[str, Any] | None = None
        final_params = dict(params)
        try:
            if optimize_hyperparams:
                optimization = self.optimizer.optimize(
                    name,
                    X,
                    base_params=final_params,
                    max_trials=tuning_options.get("max_trials", 10),
                    holdout_fraction=tuning_options.get("holdout_fraction", 0.1),
                    early_stopping_patience=tuning_options.get("early_stopping_patience", 3),
                    early_stopping_min_delta=tuning_options.get("early_stopping_min_delta", 1e-4),
                    random_state=tuning_options.get("random_state", 0),
                )
                final_params = optimization.best_params
                optimization_payload = {
                    "enabled": True,
                    "best_score": optimization.best_score,
                    "best_trial_index": optimization.best_trial_index,
                    "best_params": optimization.best_params,
                    "stopped_early": optimization.stopped_early,
                    "trials": optimization.evaluated_trials,
                }
                log_kv(self.logger, f"optimization finished for {name}", best_score=round(optimization.best_score, 6), best_params=optimization.best_params)
            else:
                optimization_payload = {"enabled": False}

            imputer = self.registry.build(name, **final_params)
            with measure_runtime() as timer:
                imputed = imputer.fit_transform(X)
            validity = validate_imputed_result(imputed)
            stats_delta = compare_basic_statistics(X, imputed)
            log_kv(self.logger, f"imputer finished: {name}", runtime_sec=round(timer.elapsed_sec or 0.0, 6), remaining_nan=validity["remaining_nan"])
            return {
                "success": True,
                "imputed_array": imputed,
                "runtime_sec": timer.elapsed_sec,
                "params": imputer.get_params(),
                "remaining_nan": validity["remaining_nan"],
                "has_inf": validity["has_inf"],
                "stats_delta": stats_delta,
                "optimization": optimization_payload,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            log_kv(self.logger, f"imputer failed: {name}", error=str(exc))
            return {
                "success": False,
                "imputed_array": None,
                "runtime_sec": None,
                "params": dict(final_params),
                "remaining_nan": None,
                "has_inf": None,
                "stats_delta": None,
                "optimization": optimization_payload,
                "error": str(exc),
            }

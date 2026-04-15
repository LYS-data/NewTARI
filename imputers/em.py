"""Expectation-Maximization imputer."""

from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer

from imputers.base import BaseImputer
from utils.logging_utils import get_logger, log_kv


class EMImputer(BaseImputer):
    """Gaussian EM imputer for numeric tabular data."""

    name = "em"

    def __init__(
        self,
        *,
        maxit: int = 200,
        convergence_threshold: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.maxit = maxit
        self.convergence_threshold = convergence_threshold
        self.logger = get_logger("imputation_reco.em")
        self.iterations_trained_: int = 0
        self.converged_: bool = False
        self._reconstructed_: np.ndarray | None = None
        self._mu_: np.ndarray | None = None
        self._sigma_: np.ndarray | None = None

    def _converged(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        mu_new: np.ndarray,
        sigma_new: np.ndarray,
    ) -> bool:
        return (
            np.linalg.norm(mu - mu_new) < self.convergence_threshold
            and np.linalg.norm(sigma - sigma_new, ord=2) < self.convergence_threshold
        )

    def _em_step(
        self,
        X_reconstructed: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        observed: list[np.ndarray],
        missing: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows, columns = X_reconstructed.shape
        sigma_tilde: dict[int, np.ndarray] = {}

        for row_idx in range(rows):
            sigma_tilde[row_idx] = np.zeros((columns, columns))
            observed_i = observed[row_idx]
            missing_i = missing[row_idx]
            if len(missing_i) == 0:
                continue
            if len(observed_i) == 0:
                X_reconstructed[row_idx, missing_i] = mu[missing_i]
                sigma_tilde[row_idx][np.ix_(missing_i, missing_i)] = sigma[np.ix_(missing_i, missing_i)]
                continue

            s_mm = sigma[np.ix_(missing_i, missing_i)]
            s_mo = sigma[np.ix_(missing_i, observed_i)]
            s_om = s_mo.T
            s_oo = sigma[np.ix_(observed_i, observed_i)]
            s_oo_inv = np.linalg.pinv(s_oo)

            mu_tilde = mu[missing_i] + s_mo @ s_oo_inv @ (X_reconstructed[row_idx, observed_i] - mu[observed_i])
            X_reconstructed[row_idx, missing_i] = mu_tilde
            s_mm_o = s_mm - s_mo @ s_oo_inv @ s_om
            sigma_tilde[row_idx][np.ix_(missing_i, missing_i)] = s_mm_o

        mu_new = np.mean(X_reconstructed, axis=0)
        sigma_new = np.cov(X_reconstructed.T, bias=True) + reduce(np.add, sigma_tilde.values()) / rows
        sigma_new = np.atleast_2d(sigma_new)
        if sigma_new.shape != (columns, columns):
            sigma_new = np.diag(np.diag(sigma_new))
        return mu_new, sigma_new, X_reconstructed

    def _fit(self, X: np.ndarray) -> None:
        rows, columns = X.shape
        mask = ~np.isnan(X)
        observed = [np.where(mask[row_idx])[0] for row_idx in range(rows)]
        missing = [np.where(~mask[row_idx])[0] for row_idx in range(rows)]

        mu = np.nanmean(X, axis=0)
        complete_rows = np.where(~np.isnan(X).any(axis=1))[0]
        if len(complete_rows) >= 2:
            sigma = np.cov(X[complete_rows].T, bias=True)
        else:
            sigma = np.diag(np.nanvar(X, axis=0))
        sigma = np.atleast_2d(sigma)
        if np.isnan(sigma).any() or sigma.shape != (columns, columns):
            sigma = np.diag(np.nanvar(X, axis=0))
        sigma = np.nan_to_num(sigma, nan=1.0)
        sigma += np.eye(columns) * 1e-6

        X_reconstructed = np.array(X, dtype=float, copy=True)
        for col_idx in range(columns):
            missing_mask = np.isnan(X_reconstructed[:, col_idx])
            X_reconstructed[missing_mask, col_idx] = mu[col_idx]

        self.converged_ = False
        for iteration in range(1, self.maxit + 1):
            try:
                mu_new, sigma_new, X_reconstructed = self._em_step(
                    X_reconstructed,
                    mu,
                    sigma,
                    observed,
                    missing,
                )
                if self._converged(mu, sigma, mu_new, sigma_new):
                    self.converged_ = True
                    self.iterations_trained_ = iteration
                    log_kv(self.logger, "em converged", iteration=iteration)
                    mu, sigma = mu_new, sigma_new
                    break
                mu, sigma = mu_new, sigma_new
            except Exception as exc:  # noqa: BLE001
                log_kv(self.logger, "em iteration failed", iteration=iteration, error=str(exc))
                self.iterations_trained_ = iteration
                break
        else:
            self.iterations_trained_ = self.maxit

        if np.isnan(X_reconstructed).any():
            X_reconstructed = SimpleImputer(strategy="mean").fit_transform(X_reconstructed)
        self._reconstructed_ = X_reconstructed
        self._mu_ = mu
        self._sigma_ = sigma

    def _impute_with_fitted_distribution(self, X: np.ndarray) -> np.ndarray:
        if self._mu_ is None or self._sigma_ is None:
            raise RuntimeError("EM distribution parameters are not fitted.")

        completed = np.array(X, dtype=float, copy=True)
        rows = completed.shape[0]
        observed_mask = ~np.isnan(completed)

        for row_idx in range(rows):
            observed_idx = np.where(observed_mask[row_idx])[0]
            missing_idx = np.where(~observed_mask[row_idx])[0]
            if len(missing_idx) == 0:
                continue
            if len(observed_idx) == 0:
                completed[row_idx, missing_idx] = self._mu_[missing_idx]
                continue

            s_mo = self._sigma_[np.ix_(missing_idx, observed_idx)]
            s_oo = self._sigma_[np.ix_(observed_idx, observed_idx)]
            s_oo_inv = np.linalg.pinv(s_oo)
            conditional_mean = self._mu_[missing_idx] + s_mo @ s_oo_inv @ (
                completed[row_idx, observed_idx] - self._mu_[observed_idx]
            )
            completed[row_idx, missing_idx] = conditional_mean

        if np.isnan(completed).any():
            completed = SimpleImputer(strategy="mean").fit_transform(completed)
        return completed

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self._reconstructed_ is None:
            raise RuntimeError("EM model is not fitted.")
        completed = np.array(X, dtype=float, copy=True)
        if completed.shape == self._reconstructed_.shape:
            missing_mask = np.isnan(completed)
            completed[missing_mask] = self._reconstructed_[missing_mask]
            if not np.isnan(completed).any():
                return completed
        return self._impute_with_fitted_distribution(completed)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "maxit": self.maxit,
                "convergence_threshold": self.convergence_threshold,
                "iterations_trained": self.iterations_trained_,
                "converged": self.converged_,
                "has_distribution_parameters": self._mu_ is not None and self._sigma_ is not None,
            }
        )
        return params

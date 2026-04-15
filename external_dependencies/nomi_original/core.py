"""NOMI core based on the official guaiyoui/NOMI implementation.

The nearest-neighbor retrieval, NTK regressor, normalization, and iterative
update rules are preserved from the original code. The main adaptation is
splitting the one-shot ``fit_transform`` flow into ``fit`` and ``transform``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import hnswlib
import neural_tangents as nt
import numpy as np
from neural_tangents import stax


def sample_batch_index(total, batch_size):
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


def dist2sim(neigh_dist):
    neigh_dist = np.asarray(neigh_dist, dtype=float)
    with np.errstate(divide="ignore"):
        dist = 1.0 / neigh_dist

    inf_mask = np.isinf(dist)
    inf_row = np.any(inf_mask, axis=1)
    dist[inf_row] = inf_mask[inf_row]
    denom = np.sum(dist, axis=1)
    denom = denom.reshape((-1, 1))
    denom[denom == 0] = 1.0

    return dist / denom


def prediction(pred_fn, X_test, kernel_type="nngp", compute_cov=True):
    pred_mean, pred_cov = pred_fn(x_test=X_test, get=kernel_type, compute_cov=compute_cov)
    return pred_mean, pred_cov


def normalization_std(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    denom = max_vals - min_vals
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    normalized_data = (data - min_vals) / denom
    return normalized_data + 1


def normalization(data, parameters=None):
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)
        norm_parameters = {"min_val": min_val, "max_val": max_val}
    else:
        min_val = parameters["min_val"]
        max_val = parameters["max_val"]
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)
        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    min_val = norm_parameters["min_val"]
    max_val = norm_parameters["max_val"]
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data


@dataclass
class NOMIState:
    index_dct: dict[int, Any]
    predict_fn_dct: dict[int, Any]
    y_train_dct: dict[int, np.ndarray]
    norm_parameters: dict[str, np.ndarray]
    missing_columns: list[int]
    kernel_fn: Any
    fit_imputed: np.ndarray


class NOMICore:
    def __init__(
        self,
        *,
        k_neighbors: int = 10,
        similarity_metric: str = "l2",
        max_iterations: int = 3,
        tau: float = 1.0,
        beta: float = 1.0,
        batch_cap: int = 300,
        diag_reg: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.max_iterations = max_iterations
        self.tau = tau
        self.beta = beta
        self.batch_cap = batch_cap
        self.diag_reg = diag_reg
        self.random_state = random_state
        self.state_: NOMIState | None = None

    def _seed(self) -> None:
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _build_kernel(self, dims: int):
        _, _, kernel_fn = stax.serial(
            stax.Dense(2 * dims),
            stax.Relu(),
            stax.Dense(dims),
            stax.Relu(),
            stax.Dense(1),
            stax.Sigmoid_like(),
        )
        return kernel_fn

    def fit(self, X: np.ndarray) -> "NOMICore":
        self._seed()
        data_x = np.asarray(X, dtype=float)
        data_m = 1 - np.isnan(data_x)
        norm_data, norm_parameters = normalization(data_x)
        norm_data_x = np.nan_to_num(norm_data, 0)
        imputed_X = norm_data_x.copy()
        data_m_imputed = data_m.copy()
        col_indices_with_nulls = np.where(np.isnan(data_x).any(axis=0))[0].tolist()

        _, dims = norm_data_x.shape
        kernel_fn = self._build_kernel(dims)
        index_dct: dict[int, Any] = {}
        predict_fn_dct: dict[int, Any] = {}
        y_train_dct: dict[int, np.ndarray] = {}

        for iteration in range(self.max_iterations):
            for dim in col_indices_with_nulls:
                X_wo_dim = np.delete(imputed_X, dim, 1)
                i_not_nan_index = data_m_imputed[:, dim].astype(bool)

                if not np.any(i_not_nan_index):
                    continue

                X_train = X_wo_dim[i_not_nan_index]
                Y_train = imputed_X[i_not_nan_index, dim]
                X_test = X_wo_dim[~i_not_nan_index]
                true_indices = np.where(~i_not_nan_index)[0]

                if X_test.shape[0] == 0:
                    continue

                no, d = X_train.shape
                k_query = min(self.k_neighbors, no)
                if k_query < 2:
                    continue

                index = hnswlib.Index(space=self.similarity_metric, dim=d)
                index.init_index(max_elements=no, ef_construction=200, M=16)
                index.add_items(X_train)
                index.set_ef(max(int(k_query * 1.2), k_query))

                if X_train.shape[0] > self.batch_cap:
                    batch_idx = sample_batch_index(X_train.shape[0], self.batch_cap)
                else:
                    batch_idx = sample_batch_index(X_train.shape[0], X_train.shape[0])

                X_batch = X_train[batch_idx, :]
                Y_batch = Y_train[batch_idx]

                neigh_ind, neigh_dist = index.knn_query(X_batch, k=k_query, filter=None)
                neigh_dist = np.sqrt(neigh_dist)
                weights = dist2sim(neigh_dist[:, 1:])
                y_neighbors = Y_train[neigh_ind[:, 1:]]
                train_input = weights * y_neighbors

                neigh_ind_test, neigh_dist_test = index.knn_query(X_test, k=k_query, filter=None)
                neigh_dist_test = np.sqrt(neigh_dist_test)
                weights_test = dist2sim(neigh_dist_test[:, 1:])
                y_neighbors_test = Y_train[neigh_ind_test[:, 1:]]
                test_input = weights_test * y_neighbors_test

                predict_fn = nt.predict.gradient_descent_mse_ensemble(
                    kernel_fn=kernel_fn,
                    x_train=train_input,
                    y_train=Y_batch.reshape(-1, 1),
                    diag_reg=self.diag_reg,
                )

                y_pred, pred_cov = prediction(predict_fn, test_input, kernel_type="nngp")

                if iteration == 0:
                    imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)
                elif iteration <= 3:
                    pred_std = np.sqrt(np.diag(pred_cov))
                    pred_std = np.ravel(np.array(pred_std))
                    pred_std = normalization_std(pred_std)
                    pred_std = np.nan_to_num(pred_std, nan=1.0)

                    greater_than_threshold_indices = np.where(pred_std <= self.tau)[0]
                    for idx in range(greater_than_threshold_indices.shape[0]):
                        data_m_imputed[true_indices[greater_than_threshold_indices[idx]], dim] = 1

                    update = self.beta / pred_std
                    imputed_X[~i_not_nan_index, dim] = (1 - update) * imputed_X[~i_not_nan_index, dim] + update * y_pred.reshape(-1)
                else:
                    imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)

                index_dct[dim] = index
                predict_fn_dct[dim] = predict_fn
                y_train_dct[dim] = Y_train

        fit_imputed = renormalization(imputed_X, norm_parameters)
        self.state_ = NOMIState(
            index_dct=index_dct,
            predict_fn_dct=predict_fn_dct,
            y_train_dct=y_train_dct,
            norm_parameters=norm_parameters,
            missing_columns=col_indices_with_nulls,
            kernel_fn=kernel_fn,
            fit_imputed=fit_imputed,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.state_ is None:
            raise RuntimeError("NOMICore must be fitted before transform.")

        data_x = np.asarray(X, dtype=float)
        data_m = 1 - np.isnan(data_x)
        norm_data, norm_parameters = normalization(data_x, self.state_.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)

        imputed_X = norm_data_x.copy()
        data_m_imputed = data_m.copy()

        for dim in self.state_.missing_columns:
            X_wo_dim = np.delete(imputed_X, dim, 1)
            i_not_nan_index = data_m_imputed[:, dim].astype(bool)
            if not np.any(i_not_nan_index):
                continue

            X_test = X_wo_dim[~i_not_nan_index]
            if X_test.shape[0] == 0:
                continue
            if dim not in self.state_.index_dct:
                continue

            index = self.state_.index_dct[dim]
            predict_fn = self.state_.predict_fn_dct[dim]
            Y_train = self.state_.y_train_dct[dim]
            k_query = min(self.k_neighbors, Y_train.shape[0])
            if k_query < 2:
                continue

            neigh_ind_test, neigh_dist_test = index.knn_query(X_test, k=k_query, filter=None)
            neigh_dist_test = np.sqrt(neigh_dist_test)
            weights_test = dist2sim(neigh_dist_test[:, 1:])
            y_neighbors_test = Y_train[neigh_ind_test[:, 1:]]
            test_input = weights_test * y_neighbors_test

            y_pred, _ = prediction(predict_fn, test_input, kernel_type="nngp")
            imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)

        imputed_data = renormalization(imputed_X, norm_parameters)
        return np.asarray(imputed_data, dtype=float)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).state_.fit_imputed

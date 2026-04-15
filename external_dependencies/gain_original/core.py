"""GAIN core based on the official jsyoon0823/GAIN implementation.

The training objective, generator/discriminator definitions, normalization, and
sampling logic follow the original repository. The main adaptation is splitting
the original one-shot ``gain()`` function into ``fit`` and ``transform`` so it
can be used inside the local imputer interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def normalization(data: np.ndarray, parameters: dict[str, np.ndarray] | None = None):
    norm_data = data.copy().astype(float)
    no, dim = norm_data.shape
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


def renormalization(norm_data: np.ndarray, norm_parameters: dict[str, np.ndarray]) -> np.ndarray:
    renorm_data = norm_data.copy()
    min_val = norm_parameters["min_val"]
    max_val = norm_parameters["max_val"]
    _, dim = norm_data.shape
    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]
    return renorm_data


def rounding(imputed_data: np.ndarray, data_x: np.ndarray) -> np.ndarray:
    rounded_data = imputed_data.copy()
    _, dim = data_x.shape
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
    return rounded_data


def binary_sampler(p: float, rows: int, cols: int) -> np.ndarray:
    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])
    return 1 * (unif_random_matrix < p)


def uniform_sampler(low: float, high: float, rows: int, cols: int) -> np.ndarray:
    return np.random.uniform(low, high, size=[rows, cols])


def sample_batch_index(total: int, batch_size: int) -> np.ndarray:
    total_idx = np.random.permutation(total)
    return total_idx[:batch_size]


def xavier_init(size: tuple[int, int]) -> tf.Tensor:
    in_dim = size[0]
    xavier_stddev = 1.0 / np.sqrt(in_dim / 2.0)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


@dataclass
class GainTensors:
    x: Any
    m: Any
    h: Any
    new_x: Any
    G_sample: Any
    D_prob: Any
    D_loss: Any
    G_loss: Any
    MSE_loss: Any
    D_solver: Any
    G_solver: Any


class GAINCore:
    def __init__(
        self,
        *,
        batch_size: int = 128,
        hint_rate: float = 0.9,
        alpha: float = 100.0,
        iterations: int = 10000,
        random_state: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state

        self.norm_parameters_: dict[str, np.ndarray] | None = None
        self._graph: tf.Graph | None = None
        self._sess: tf.Session | None = None
        self._tensors: GainTensors | None = None
        self._dim: int | None = None

    def _seed(self) -> None:
        if self.random_state is None:
            return
        np.random.seed(self.random_state)
        tf.set_random_seed(self.random_state)

    def _build_graph(self, dim: int) -> None:
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._seed()

            x = tf.placeholder(tf.float32, shape=[None, dim])
            m = tf.placeholder(tf.float32, shape=[None, dim])
            h = tf.placeholder(tf.float32, shape=[None, dim])

            D_W1 = tf.Variable(xavier_init((dim * 2, dim)))
            D_b1 = tf.Variable(tf.zeros(shape=[dim]))
            D_W2 = tf.Variable(xavier_init((dim, dim)))
            D_b2 = tf.Variable(tf.zeros(shape=[dim]))
            D_W3 = tf.Variable(xavier_init((dim, dim)))
            D_b3 = tf.Variable(tf.zeros(shape=[dim]))

            theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

            G_W1 = tf.Variable(xavier_init((dim * 2, dim)))
            G_b1 = tf.Variable(tf.zeros(shape=[dim]))
            G_W2 = tf.Variable(xavier_init((dim, dim)))
            G_b2 = tf.Variable(tf.zeros(shape=[dim]))
            G_W3 = tf.Variable(xavier_init((dim, dim)))
            G_b3 = tf.Variable(tf.zeros(shape=[dim]))

            theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

            def generator(x_in, m_in):
                inputs = tf.concat(values=[x_in, m_in], axis=1)
                G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
                G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
                return tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)

            def discriminator(x_in, h_in):
                inputs = tf.concat(values=[x_in, h_in], axis=1)
                D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
                D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
                return tf.nn.sigmoid(tf.matmul(D_h2, D_W3) + D_b3)

            G_sample = generator(x, m)
            hat_x = x * m + G_sample * (1 - m)
            D_prob = discriminator(hat_x, h)

            D_loss_temp = -tf.reduce_mean(
                m * tf.log(D_prob + 1e-8) + (1 - m) * tf.log(1.0 - D_prob + 1e-8)
            )

            G_loss_temp = -tf.reduce_mean((1 - m) * tf.log(D_prob + 1e-8))
            MSE_loss = tf.reduce_mean((m * x - m * G_sample) ** 2) / tf.reduce_mean(m)
            G_loss = G_loss_temp + self.alpha * MSE_loss

            D_solver = tf.train.AdamOptimizer().minimize(D_loss_temp, var_list=theta_D)
            G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

            init = tf.global_variables_initializer()
            sess = tf.Session(graph=self._graph)
            sess.run(init)

            self._sess = sess
            self._tensors = GainTensors(
                x=x,
                m=m,
                h=h,
                new_x=hat_x,
                G_sample=G_sample,
                D_prob=D_prob,
                D_loss=D_loss_temp,
                G_loss=G_loss,
                MSE_loss=MSE_loss,
                D_solver=D_solver,
                G_solver=G_solver,
            )

    def fit(self, data_x: np.ndarray) -> "GAINCore":
        x_array = np.asarray(data_x, dtype=float)
        no, dim = x_array.shape
        data_m = 1 - np.isnan(x_array)

        norm_data, norm_parameters = normalization(x_array)
        norm_data_x = np.nan_to_num(norm_data, 0)
        self.norm_parameters_ = norm_parameters
        self._dim = dim
        self._build_graph(dim)
        assert self._sess is not None and self._tensors is not None

        batch_size = min(self.batch_size, no)
        for _it in range(self.iterations):
            batch_idx = sample_batch_index(no, batch_size)
            x_mb = norm_data_x[batch_idx, :]
            m_mb = data_m[batch_idx, :]
            z_mb = uniform_sampler(0, 0.01, batch_size, dim)
            h_mb_temp = binary_sampler(self.hint_rate, batch_size, dim)
            h_mb = m_mb * h_mb_temp
            x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

            feed_dict = {
                self._tensors.x: x_mb,
                self._tensors.m: m_mb,
                self._tensors.h: h_mb,
            }
            self._sess.run([self._tensors.D_solver], feed_dict=feed_dict)
            self._sess.run([self._tensors.G_solver], feed_dict=feed_dict)
        return self

    def transform(self, data_x: np.ndarray) -> np.ndarray:
        if self._sess is None or self._tensors is None or self.norm_parameters_ is None or self._dim is None:
            raise RuntimeError("GAINCore must be fitted before transform.")

        x_array = np.asarray(data_x, dtype=float)
        no, dim = x_array.shape
        if dim != self._dim:
            raise ValueError(f"GAINCore expected {self._dim} features, got {dim}.")

        data_m = 1 - np.isnan(x_array)
        norm_data, _ = normalization(x_array, self.norm_parameters_)
        norm_data_x = np.nan_to_num(norm_data, 0)
        z_mb = uniform_sampler(0, 0.01, no, dim)
        m_mb = data_m
        x_mb = m_mb * norm_data_x + (1 - m_mb) * z_mb

        imputed_data = self._sess.run(
            [self._tensors.G_sample],
            feed_dict={
                self._tensors.x: x_mb,
                self._tensors.m: m_mb,
                self._tensors.h: m_mb,
            },
        )[0]

        imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
        imputed_data = renormalization(imputed_data, self.norm_parameters_)
        imputed_data = rounding(imputed_data, x_array)
        return np.asarray(imputed_data, dtype=float)

    def fit_transform(self, data_x: np.ndarray) -> np.ndarray:
        return self.fit(data_x).transform(data_x)

    def close(self) -> None:
        if self._sess is not None:
            self._sess.close()
            self._sess = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

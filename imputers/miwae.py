"""MIWAE imputer adapted for the local research framework."""

from __future__ import annotations

from typing import Any

import numpy as np

from imputers.base import BaseImputer
from utils.logging_utils import get_logger, log_kv

try:
    import torch
    from torch import nn, optim
    import torch.distributions as td

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    optim = None
    td = None
    HAS_TORCH = False


if HAS_TORCH:

    def _weights_init(layer: Any) -> None:
        if isinstance(layer, nn.Linear):
            torch.nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

else:

    def _weights_init(layer: Any) -> None:  # pragma: no cover
        return None


class MIWAEImputer(BaseImputer):
    """MIWAE imputer for numeric tabular data.

    This implementation keeps the core idea of MIWAE while simplifying the
    engineering surface so it fits the project's unified imputer interface.
    """

    name = "miwae"

    def __init__(
        self,
        *,
        n_epochs: int = 200,
        batch_size: int = 128,
        latent_size: int = 3,
        n_hidden: int = 32,
        K: int = 20,
        learning_rate: float = 1e-3,
        validation_fraction: float = 0.1,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        imputation_samples: int = 10,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("miwae requires the optional dependency 'torch'.")
        super().__init__(**kwargs)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.n_hidden = n_hidden
        self.K = K
        self.learning_rate = learning_rate
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.imputation_samples = imputation_samples
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_name)
        self.p_z: td.Independent | None = None
        self.decoder: nn.Module | None = None
        self.encoder: nn.Module | None = None
        self.best_validation_loss_: float | None = None
        self.epochs_trained_: int = 0
        self.logger = get_logger("imputation_reco.miwae")

    def _set_seed(self) -> None:
        if self.random_state is None:
            return
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_validation_mask(self, observed_mask: np.ndarray) -> np.ndarray:
        observed_positions = np.argwhere(observed_mask > 0)
        if len(observed_positions) <= 1:
            return np.zeros_like(observed_mask, dtype=np.float32)
        n_val = max(1, int(len(observed_positions) * self.validation_fraction))
        rng = np.random.default_rng(self.random_state)
        chosen = observed_positions[rng.choice(len(observed_positions), size=n_val, replace=False)]
        val_mask = np.zeros_like(observed_mask, dtype=np.float32)
        for row_idx, col_idx in chosen:
            val_mask[row_idx, col_idx] = 1.0
        return val_mask

    def _miwae_loss(self, x_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.encoder is None or self.decoder is None or self.p_z is None:
            raise RuntimeError("MIWAE modules are not initialized.")
        batch_size = x_input.shape[0]
        p = x_input.shape[1]

        out_encoder = self.encoder(x_input)
        q_zgivenxobs = td.Independent(
            td.Normal(
                loc=out_encoder[..., : self.latent_size],
                scale=torch.nn.Softplus()(out_encoder[..., self.latent_size : (2 * self.latent_size)]) + 1e-3,
            ),
            1,
        )

        z_given_x = q_zgivenxobs.rsample([self.K])
        z_flat = z_given_x.reshape([self.K * batch_size, self.latent_size])

        out_decoder = self.decoder(z_flat)
        means = out_decoder[..., :p]
        scales = torch.nn.Softplus()(out_decoder[..., p : (2 * p)]) + 1e-3
        dfs = torch.nn.Softplus()(out_decoder[..., (2 * p) : (3 * p)]) + 3.0

        tiled_x = torch.repeat_interleave(x_input, repeats=self.K, dim=0)
        tiled_mask = torch.repeat_interleave(mask, repeats=self.K, dim=0)

        log_px_flat = td.StudentT(loc=means, scale=scales, df=dfs).log_prob(tiled_x)
        log_px_obs = torch.sum(log_px_flat * tiled_mask, dim=1).reshape([self.K, batch_size])
        log_pz = self.p_z.log_prob(z_given_x)
        log_qz = q_zgivenxobs.log_prob(z_given_x)

        return -torch.mean(torch.logsumexp(log_px_obs + log_pz - log_qz, dim=0))

    def _miwae_impute(self, x_input: torch.Tensor, mask: torch.Tensor, L: int) -> torch.Tensor:
        if self.encoder is None or self.decoder is None or self.p_z is None:
            raise RuntimeError("MIWAE modules are not initialized.")
        batch_size = x_input.shape[0]
        p = x_input.shape[1]

        out_encoder = self.encoder(x_input)
        q_zgivenxobs = td.Independent(
            td.Normal(
                loc=out_encoder[..., : self.latent_size],
                scale=torch.nn.Softplus()(out_encoder[..., self.latent_size : (2 * self.latent_size)]) + 1e-3,
            ),
            1,
        )

        z_given_x = q_zgivenxobs.rsample([L])
        z_flat = z_given_x.reshape([L * batch_size, self.latent_size])
        out_decoder = self.decoder(z_flat)
        means = out_decoder[..., :p]
        scales = torch.nn.Softplus()(out_decoder[..., p : (2 * p)]) + 1e-3
        dfs = torch.nn.Softplus()(out_decoder[..., (2 * p) : (3 * p)]) + 3.0

        tiled_x = torch.repeat_interleave(x_input, repeats=L, dim=0)
        tiled_mask = torch.repeat_interleave(mask, repeats=L, dim=0)
        log_px_flat = td.StudentT(loc=means, scale=scales, df=dfs).log_prob(tiled_x)
        log_px_obs = torch.sum(log_px_flat * tiled_mask, dim=1).reshape([L, batch_size])
        log_pz = self.p_z.log_prob(z_given_x)
        log_qz = q_zgivenxobs.log_prob(z_given_x)

        x_given_z = td.Independent(td.StudentT(loc=means, scale=scales, df=dfs), 1)
        importance_weights = torch.nn.functional.softmax(log_px_obs + log_pz - log_qz, dim=0)
        samples = x_given_z.sample().reshape([L, batch_size, p])
        return torch.einsum("lb,lbd->bd", importance_weights, samples)

    def _fit(self, X: np.ndarray) -> None:
        self._set_seed()
        X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().to(self.device)
        observed_mask = np.isfinite(np.asarray(X, dtype=float)).astype(np.float32)
        validation_mask_np = self._build_validation_mask(observed_mask)
        train_mask_np = np.clip(observed_mask - validation_mask_np, 0.0, 1.0)
        train_mask = torch.from_numpy(train_mask_np).float().to(self.device)
        validation_mask = torch.from_numpy(validation_mask_np).float().to(self.device)

        x_filled = torch.clone(X_tensor)
        x_filled[torch.isnan(x_filled)] = 0.0

        n, p = X_tensor.shape
        self.p_z = td.Independent(
            td.Normal(loc=torch.zeros(self.latent_size).to(self.device), scale=torch.ones(self.latent_size).to(self.device)),
            1,
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 3 * p),
        ).to(self.device)
        self.encoder = nn.Sequential(
            nn.Linear(p, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 2 * self.latent_size),
        ).to(self.device)
        self.encoder.apply(_weights_init)
        self.decoder.apply(_weights_init)

        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate)
        batch_size = min(self.batch_size, n)
        best_val_loss = float("inf")
        best_encoder_state = None
        best_decoder_state = None
        patience_counter = 0

        for epoch in range(1, self.n_epochs + 1):
            permutation = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_x = x_filled[batch_indices]
                batch_mask = train_mask[batch_indices]
                optimizer.zero_grad()
                loss = self._miwae_loss(batch_x, batch_mask)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                full_loss = self._miwae_loss(x_filled, train_mask)
                if float(validation_mask.sum().item()) > 0:
                    imputed = self._miwae_impute(x_filled, train_mask, L=self.imputation_samples)
                    val_loss = torch.sum(((imputed - x_filled) * validation_mask) ** 2) / (torch.sum(validation_mask) + 1e-8)
                    val_loss_value = float(val_loss.item())
                else:
                    val_loss_value = float(full_loss.item())

            if val_loss_value + self.early_stopping_min_delta < best_val_loss:
                best_val_loss = val_loss_value
                best_encoder_state = {key: value.detach().cpu().clone() for key, value in self.encoder.state_dict().items()}
                best_decoder_state = {key: value.detach().cpu().clone() for key, value in self.decoder.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch == 1 or epoch % 20 == 0:
                log_kv(self.logger, "miwae training", epoch=epoch, train_loss=round(float(full_loss.item()), 6), validation_loss=round(val_loss_value, 6))

            if patience_counter >= self.early_stopping_patience:
                log_kv(self.logger, "miwae early stopping", epoch=epoch, best_validation_loss=round(best_val_loss, 6))
                self.epochs_trained_ = epoch
                break
        else:
            self.epochs_trained_ = self.n_epochs

        if best_encoder_state is not None and best_decoder_state is not None:
            self.encoder.load_state_dict(best_encoder_state)
            self.decoder.load_state_dict(best_decoder_state)
        self.best_validation_loss_ = best_val_loss

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("MIWAE model is not fitted.")
        X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32)).float().to(self.device)
        observed_mask = torch.from_numpy(np.isfinite(np.asarray(X, dtype=float)).astype(np.float32)).float().to(self.device)
        x_filled = torch.clone(X_tensor)
        x_filled[torch.isnan(x_filled)] = 0.0
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            imputed = self._miwae_impute(x_filled, observed_mask, L=self.imputation_samples)
        completed = observed_mask * torch.nan_to_num(X_tensor, nan=0.0) + (1 - observed_mask) * imputed
        return completed.detach().cpu().numpy().astype(float)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "latent_size": self.latent_size,
                "n_hidden": self.n_hidden,
                "K": self.K,
                "learning_rate": self.learning_rate,
                "validation_fraction": self.validation_fraction,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_min_delta": self.early_stopping_min_delta,
                "imputation_samples": self.imputation_samples,
                "device": self.device_name,
                "best_validation_loss": getattr(self, 'best_validation_loss_', None),
                "epochs_trained": getattr(self, 'epochs_trained_', 0),
            }
        )
        return params

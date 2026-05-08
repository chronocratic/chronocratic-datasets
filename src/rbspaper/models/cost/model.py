from __future__ import annotations

__all__ = ['CoST']

import itertools
from typing import cast, override, TYPE_CHECKING

import lightning.pytorch as pl
import numpy as np
import torch
from torch import fft, nn
import torch.nn.functional as F  # noqa: N812 # torch.nn.functional convention
from torch.optim import SGD

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.rbspaper.models.augmentation.strategies import CosTRandomFunctionAugmentation

from src.rbspaper.models.abstract import EncodingFunctionalityMixin
from src.rbspaper.models.augmentation.enums import CoSTAugmentationMode
from src.rbspaper.models.augmentation.factories import CoSTAugmentationMethodFactory
from src.rbspaper.models.cost.utils import compute_amplitude_and_phase
from src.rbspaper.models.encoders import CoSTTimeSeriesEncoder
from src.rbspaper.models.encoders.masking import MaskMode
from src.rbspaper.models.losses import instance_contrastive_loss
from src.rbspaper.models.utils import extract_features_from_batch, process_sample_length


class CoST(pl.LightningModule, EncodingFunctionalityMixin):
    queue: torch.Tensor
    queue_insert_index: torch.Tensor

    def __init__(
        self,
        input_dims: int,
        sequence_length: int,
        kernel_sizes: list[int],
        augmentation_mode: CoSTAugmentationMode,
        augmentation_method_params: dict,
        augmentation_mode_params: dict | None = None,
        max_train_length: int = 201,
        hidden_dims: int = 64,
        output_dims: int = 320,
        depth: int = 10,
        dropout_rate: float = 0.1,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        learning_rate: float = 1e-3,
        seasonal_loss_weight: float = 0.1,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
        *,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self._learning_rate = learning_rate
        self._max_train_length = max_train_length
        self._sync_dist = sync_dist

        self._augmentation_mode = augmentation_mode
        self._augmentation_method: CosTRandomFunctionAugmentation

        length = min(max_train_length, sequence_length)

        self.query_encoder = CoSTTimeSeriesEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=depth,
            dropout_rate=dropout_rate,
            kernel_sizes=kernel_sizes,
            length=length,
            mask_mode=mask_mode,
        )

        component_dims = self.query_encoder.component_dims

        self.query_projection_head = nn.Sequential(
            nn.Linear(component_dims, component_dims),
            nn.ReLU(),
            nn.Linear(component_dims, component_dims),
        )

        self.key_encoder = CoSTTimeSeriesEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=depth,
            dropout_rate=dropout_rate,
            kernel_sizes=kernel_sizes,
            length=length,
            mask_mode=mask_mode,
        )

        self.key_projection_head = nn.Sequential(
            nn.Linear(component_dims, component_dims),
            nn.ReLU(),
            nn.Linear(component_dims, component_dims),
        )

        for param_query_encoder, param_key_encoder in zip(
            self.query_encoder.parameters(), self.key_encoder.parameters(), strict=False
        ):
            param_key_encoder.data.copy_(param_query_encoder.data)  # initialize
            param_key_encoder.requires_grad = False  # not update by gradient
        for param_query_projection_head, param_key_projection_head in zip(
            self.query_projection_head.parameters(),
            self.key_projection_head.parameters(),
            strict=False,
        ):
            param_key_projection_head.data.copy_(param_query_projection_head.data)  # initialize
            param_key_projection_head.requires_grad = False  # not update by gradient

        self.register_buffer('queue', F.normalize(torch.randn(component_dims, queue_size), dim=0))
        self.register_buffer('queue_insert_index', torch.zeros(1, dtype=torch.long))

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.seasonal_loss_weight = seasonal_loss_weight

        self._init_augmentation_method(augmentation_method_params)

        self._init_augmentation_mode_params(augmentation_mode_params)

    def configure_optimizers(self) -> SGD:
        """Configure the optimizer for training.

        Returns:
            An SGD optimizer for the query encoder and projection head parameters.
        """
        model_params = itertools.chain(
            self.query_encoder.parameters(), self.query_projection_head.parameters()
        )

        model_params = filter(lambda p: p.requires_grad, model_params)

        optimizer = SGD(model_params, lr=self._learning_rate, momentum=0.9, weight_decay=1e-4)
        return optimizer

    def _init_augmentation_method(self, augmentation_method_params: dict) -> None:
        self._augmentation_method = cast(
            'CosTRandomFunctionAugmentation',
            CoSTAugmentationMethodFactory.get_augmentation_method(
                mode=self._augmentation_mode, params=augmentation_method_params
            ),
        )

    def _init_augmentation_mode_params(self, _augmentation_mode_param: dict | None) -> None:
        self.automatic_optimization = False

    def _compute_contrastive_loss(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        negative_key_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss between query and key embeddings.

        Args:
            query_embeddings: Query embeddings of shape (batch_size, dim).
            key_embeddings: Key embeddings of shape (batch_size, dim).
            negative_key_embeddings: Negative key embeddings of shape (dim, queue_size).

        Returns:
            Contrastive loss as a scalar tensor.
        """
        # positive logits: Nx1
        positive_logits = torch.einsum('nc,nc->n', [query_embeddings, key_embeddings]).unsqueeze(-1)
        # negative logits: NxK
        negative_logits = torch.einsum('nc,ck->nk', [query_embeddings, negative_key_embeddings])

        logits = torch.cat([positive_logits, negative_logits], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cpu()
        loss = F.cross_entropy(logits, labels)

        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """Momentum update for key encoder."""
        for param_query_encoder, param_key_encoder in zip(
            self.query_encoder.parameters(), self.key_encoder.parameters(), strict=False
        ):
            param_key_encoder.data = (
                param_key_encoder.data * self.momentum
                + param_query_encoder.data * (1 - self.momentum)
            )
        for param_query_projection_head, param_key_projection_head in zip(
            self.query_projection_head.parameters(),
            self.key_projection_head.parameters(),
            strict=False,
        ):
            param_key_projection_head.data = (
                param_key_projection_head.data * self.momentum
                + param_query_projection_head.data * (1 - self.momentum)
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Dequeue and enqueue key embeddings into the memory queue.

        Args:
            keys: Key embeddings of shape (batch_size, dim).
        """
        batch_size = keys.shape[0]

        ptr = int(self.queue_insert_index)
        if self.queue_size % batch_size != 0:
            msg = f'Queue size {self.queue_size} must be divisible by batch size {batch_size}'
            raise ValueError(msg)

        # replace keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_insert_index[0] = ptr

    def _compute_total_loss(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute total loss including contrastive and seasonal components.

        Args:
            query: Augmented query time series of shape (batch_size, seq_len, features).
            key: Augmented key time series of shape (batch_size, seq_len, features).

        Returns:
            Total loss as a scalar tensor.
        """
        # compute query features
        random_index = np.random.default_rng().integers(0, query.shape[1])

        query_trend, query_seasonality = self.query_encoder(query)
        if query_trend is not None:
            query_trend = F.normalize(
                self.query_projection_head(query_trend[:, random_index]), dim=-1
            )

        # compute key features
        with torch.no_grad():  # no gradient for keys
            self._momentum_update_key_encoder()  # update key encoder
            key_trend, key_seasonality = self.key_encoder(key)
            if key_trend is not None:
                key_trend = F.normalize(
                    self.key_projection_head(key_trend[:, random_index]), dim=-1
                )

        loss = self._compute_contrastive_loss(query_trend, key_trend, self.queue.clone().detach())
        self._dequeue_and_enqueue(key_trend)

        query_seasonality = F.normalize(query_seasonality, dim=-1)
        _, key_seasonality = self.query_encoder(key)
        key_seasonality = F.normalize(key_seasonality, dim=-1)

        query_seasonality_freq = fft.rfft(query_seasonality, dim=1)
        key_seasonality_freq = fft.rfft(key_seasonality, dim=1)
        query_seasonality_amplitude, query_seasonality_phase = compute_amplitude_and_phase(
            query_seasonality_freq
        )
        key_seasonality_amplitude, key_seasonality_phase = compute_amplitude_and_phase(
            key_seasonality_freq
        )

        seasonal_loss = instance_contrastive_loss(
            query_seasonality_amplitude, key_seasonality_amplitude
        ) + instance_contrastive_loss(query_seasonality_phase, key_seasonality_phase)

        loss += self.seasonal_loss_weight * (seasonal_loss / 2)

        return loss

    def training_step(self, batch: object, _batch_idx: int) -> torch.Tensor:
        """Execute a single training step.

        Args:
            batch: Input batch containing time series data.
            _batch_idx: Index of the current batch (unused).

        Returns:
            Training loss as a scalar tensor.
        """
        x = extract_features_from_batch(batch)

        optimizer = cast('SGD', self.optimizers())

        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)

        query = self._augmentation_method.augment(data=x)
        key = self._augmentation_method.augment(data=x)

        train_loss = self._compute_total_loss(query, key)

        self.log(
            'train_loss',
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        optimizer.zero_grad()
        self.manual_backward(train_loss)
        optimizer.step()

        return train_loss

    def validation_step(self, batch: object, _batch_idx: int) -> torch.Tensor:
        """Execute a single validation step.

        Args:
            batch: Input batch containing time series data.
            _batch_idx: Index of the current batch (unused).

        Returns:
            Validation loss as a scalar tensor.
        """
        x = extract_features_from_batch(batch)

        query = self._augmentation_method.augment(data=x)
        key = self._augmentation_method.augment(data=x)

        val_loss = self._compute_total_loss(query, key)

        self.log(
            'val_loss',
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        return val_loss

    @override
    def _get_encoder(self) -> nn.Module:
        return self.query_encoder

    @override
    def _get_eval_method(self) -> Callable[..., torch.Tensor]:
        return self._evaluate_with_feature_concatenation

    @override
    def _get_slice(self, _sliding_padding: int, _sliding_length: int) -> None:
        return None

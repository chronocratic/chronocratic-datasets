__all__ = ['TS2Vec']

from typing import cast, TYPE_CHECKING

import lightning.pytorch as pl
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

if TYPE_CHECKING:
    from src.rbspaper.models.augmentation.strategies import CropShiftAugmentation

from src.rbspaper.models.abstract import EncodingFunctionalityMixin
from src.rbspaper.models.augmentation.enums import TS2VecAugmentationMode
from src.rbspaper.models.augmentation.factories import TS2VecAugmentationMethodFactory
from src.rbspaper.models.encoders import TS2VecTimeSeriesEncoder
from src.rbspaper.models.encoders.masking import MaskMode
from src.rbspaper.models.losses import hierarchical_contrastive_loss
from src.rbspaper.models.utils import extract_features_from_batch, process_sample_length


class TS2Vec(pl.LightningModule, EncodingFunctionalityMixin):
    def __init__(
        self,
        input_dims: int,
        augmentation_mode: TS2VecAugmentationMode,
        augmentation_method_params: dict | None,
        augmentation_mode_params: dict | None = None,
        hidden_dims: int = 64,
        output_dims: int = 320,
        depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        learning_rate: float = 1e-3,
        max_train_length: int | None = None,
        temporal_unit: int = 0,
        *,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self._learning_rate = learning_rate
        self._max_train_length = max_train_length
        self._temporal_unit = temporal_unit
        self._sync_dist = sync_dist

        self._augmentation_mode = augmentation_mode
        self._augmentation_method: CropShiftAugmentation

        self._encoder = TS2VecTimeSeriesEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

        self._averaged_encoder = AveragedModel(self._encoder)
        self._averaged_encoder.update_parameters(self._encoder)

        self._init_augmentation_method(augmentation_method_params)

        self._init_augmentation_mode_params(augmentation_mode_params)

    def configure_optimizers(self) -> AdamW:
        """Configure the optimizer for training the encoder."""
        optimizer = AdamW(self._encoder.parameters(), lr=self._learning_rate)
        return optimizer

    def _init_augmentation_method(self, _augmentation_method_params: dict | None) -> None:
        self._augmentation_method = cast(
            'CropShiftAugmentation',
            TS2VecAugmentationMethodFactory.get_augmentation_method(mode=self._augmentation_mode),
        )

    def _init_augmentation_mode_params(self, _augmentation_mode_param: dict | None) -> None:
        self.automatic_optimization = False

    def _calculate_encoder_loss(
        self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        return hierarchical_contrastive_loss(
            embeddings_1, embeddings_2, temporal_unit=self._temporal_unit
        )

    def _crop_shift_augmentation_strategy(
        self, x: torch.Tensor, encoder: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform TS2Vec crop-shift augmentation strategy on the input tensor.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).
            encoder: Encoder module to produce embeddings from augmented subsequences.

        Returns:
            Two augmented embeddings of the input tensor.
        """
        augmented_subsequences_1, augmented_subsequences_2, crop_length = (
            self._augmentation_method.augment(data=x, temporal_unit=self._temporal_unit)
        )

        augmented_subsequences_1_embeddings = encoder(augmented_subsequences_1)
        augmented_subsequences_2_embeddings = encoder(augmented_subsequences_2)

        augmented_subsequences_1_embeddings = augmented_subsequences_1_embeddings[:, -crop_length:]

        augmented_subsequences_2_embeddings = augmented_subsequences_2_embeddings[:, :crop_length]

        return augmented_subsequences_1_embeddings, augmented_subsequences_2_embeddings

    def _execute_augmentation_strategy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self._encoder if self.training else self._averaged_encoder

        if self._augmentation_mode == TS2VecAugmentationMode.CROP_SHIFT:
            return self._crop_shift_augmentation_strategy(x=x, encoder=encoder)
        msg = f'Unsupported augmentation mode: {self._augmentation_mode.value}'
        raise NotImplementedError(msg)

    def training_step(self, batch: object, _batch_idx: int) -> torch.Tensor:
        """Execute one training step with contrastive loss on augmented views.

        Args:
            batch: Batch of time series data from the dataloader.
            _batch_idx: Index of the current batch (unused).

        Returns:
            The computed training loss tensor.
        """
        x = extract_features_from_batch(batch)

        optimizer = cast('AdamW', self.optimizers())

        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)

        embeddings_1, embeddings_2 = self._execute_augmentation_strategy(x)

        train_loss = self._calculate_encoder_loss(embeddings_1, embeddings_2)

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

        self._averaged_encoder.update_parameters(self._encoder)

        return train_loss

    def validation_step(self, batch: object, _batch_idx: int) -> torch.Tensor:
        """Execute one validation step with contrastive loss on augmented views.

        Args:
            batch: Batch of time series data from the dataloader.
            _batch_idx: Index of the current batch (unused).

        Returns:
            The computed validation loss tensor.
        """
        x = extract_features_from_batch(batch)

        embeddings_1, embeddings_2 = self._execute_augmentation_strategy(x)

        val_loss = self._calculate_encoder_loss(embeddings_1, embeddings_2)

        self.log(
            'val_loss',
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        return val_loss

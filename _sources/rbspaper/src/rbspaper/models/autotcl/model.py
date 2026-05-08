__all__ = ['AutoTCL']

from typing import cast

import lightning.pytorch as pl
import torch
import torch.nn.functional as F  # noqa: N812 # torch.nn.functional convention
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

from src.rbspaper.models.abstract import EncodingFunctionalityMixin
from src.rbspaper.models.augmentation.enums import (
    AutoTCLAugmentationMode,
    AutoTCLNeuralNetworkAugmentationTrainingMode,
)
from src.rbspaper.models.augmentation.factories import AutoTCLAugmentationMethodFactory
from src.rbspaper.models.augmentation.strategies import AutoTCLNeuralNetworkAugmentation
from src.rbspaper.models.autotcl.utils import (
    calculate_mutual_information,
    calculate_regular_consistency,
)
from src.rbspaper.models.encoders import (
    AutoTCLAugmentationTimeSeriesEncoder,
    AutoTCLTimeSeriesEncoder,
)
from src.rbspaper.models.encoders.masking import MaskMode
from src.rbspaper.models.losses import (
    info_nce_loss,
    local_info_nce_loss,
    maximum_mean_discrepancy_with_gaussian_kernel_loss,
)
from src.rbspaper.models.utils import extract_features_from_batch, process_sample_length


class AutoTCL(pl.LightningModule, EncodingFunctionalityMixin):
    """AutoTCL model for self-supervised time series representation learning.

    Combines a time series encoder with configurable augmentation strategies
    and InfoNCE-based contrastive learning.
    """

    def __init__(
        self,
        input_dims: int,
        kernel_sizes: list[int],
        augmentation_mode: AutoTCLAugmentationMode,
        augmentation_method_params: dict,
        augmentation_mode_params: dict | None = None,
        hidden_dims: int = 64,
        output_dims: int = 320,
        depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        learning_rate: float = 1e-3,
        max_train_length: int | None = None,
        *,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self._learning_rate = learning_rate
        self._max_train_length = max_train_length
        self._sync_dist = sync_dist

        self._augmentation_mode = augmentation_mode
        self._augmentation_method: AutoTCLAugmentationTimeSeriesEncoder | None = None

        self._encoder = AutoTCLTimeSeriesEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            kernel_sizes=kernel_sizes,
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

    # -------------- Setup functions --------------

    def _init_augmentation_method(self, augmentation_method_params: dict) -> None:
        aug_method = AutoTCLAugmentationMethodFactory.get_augmentation_method(
            mode=self._augmentation_mode, params=augmentation_method_params
        )
        if not isinstance(aug_method, AutoTCLNeuralNetworkAugmentation):
            msg = f'Unexpected augmentation method type: {type(aug_method).__name__}'
            raise TypeError(msg)
        self._augmentation_method = aug_method.get_model()

    def _init_augmentation_mode_params(self, augmentation_mode_param: dict | None) -> None:
        if augmentation_mode_param is None:
            augmentation_mode_param = {}

        self.automatic_optimization = False

        self._local_loss_weight = augmentation_mode_param.get('local_loss_weight', 0.1)

        if self._augmentation_mode == AutoTCLAugmentationMode.NEURAL_NETWORK:
            self._regular_consistency_weight = augmentation_mode_param.get(
                'regular_consistency_weight', 0.001
            )
            self._regularization_weight = augmentation_mode_param.get(
                'regularization_weight', 0.001
            )
            self._regularization_threshold = augmentation_mode_param.get(
                'regularization_threshold', 0.4
            )
            self._augmentation_network_training_ratio_step = augmentation_mode_param.get(
                'augmentation_network_training_ratio_step', 1
            )
            self._meta_learning_rate = augmentation_mode_param.get('meta_learning_rate', 1e-2)
            self._augmentation_network_training_mode = augmentation_mode_param.get(
                'augmentation_network_training_mode',
                AutoTCLNeuralNetworkAugmentationTrainingMode.RELEVANT_INFORMATION_PRINCIPLE,
            )

    def _configure_optimizers_default(self) -> AdamW:
        return AdamW(self._encoder.parameters(), lr=self._learning_rate)

    def _configure_optimizers_neural_network_augmentation(self) -> list[AdamW]:
        aug_model = self._augmentation_method
        if aug_model is None:
            msg = 'Augmentation model not initialized'
            raise RuntimeError(msg)
        augmentation_network_params = aug_model.parameters()

        main_optimizer = AdamW(self._encoder.parameters(), lr=self._learning_rate)
        meta_optimizer = AdamW(augmentation_network_params, lr=self._meta_learning_rate)

        return [main_optimizer, meta_optimizer]

    def configure_optimizers(self) -> AdamW | list[AdamW]:
        """Configure optimizers based on the augmentation mode.

        Returns:
            A single optimizer for default augmentation, or a list of two
            optimizers (main encoder + augmentation network) for neural
            network augmentation mode.
        """
        if self._augmentation_mode == AutoTCLAugmentationMode.NEURAL_NETWORK:
            return self._configure_optimizers_neural_network_augmentation()
        return self._configure_optimizers_default()

    # -------------- End of setup functions --------------

    # -------------- Loss calculation functions --------------

    def _eval_mutual_information(self, batch: torch.Tensor) -> float:
        aug_model = self._augmentation_method
        if aug_model is None:
            msg = 'Augmentation model not initialized'
            raise RuntimeError(msg)
        augmentation_network_mode = aug_model.training
        aug_model.eval()

        if self._max_train_length is None:
            msg = 'max_train_length is not set'
            raise RuntimeError(msg)
        mutual_information = calculate_mutual_information(
            batch=batch, augmentation_method=aug_model, max_train_length=self._max_train_length
        )

        aug_model.train(augmentation_network_mode)

        return mutual_information

    def _calculate_encoder_loss(
        self, x_embeddings: torch.Tensor, augmented_x_embeddings: torch.Tensor
    ) -> torch.Tensor:
        local_loss = local_info_nce_loss(x_embeddings, augmented_x_embeddings)
        loss = info_nce_loss(x_embeddings, augmented_x_embeddings, temperature=1.0)
        total_loss = loss + self._local_loss_weight * local_loss

        return total_loss

    def _augmentation_loss_network_augmentation_relevant_information_principle(
        self,
        x_embeddings: torch.Tensor,
        augmented_x_embeddings: torch.Tensor,
        augmentation_factor: torch.Tensor,
    ) -> torch.Tensor:

        vx_distance = maximum_mean_discrepancy_with_gaussian_kernel_loss(
            x_embeddings, augmented_x_embeddings
        )
        regular_consistency = calculate_regular_consistency(weights=augmentation_factor)
        regularization_loss = F.relu(
            torch.sum(augmentation_factor, dim=-1).mean() - self._regularization_threshold
        )

        aug_total_loss = (
            vx_distance
            + self._regularization_weight * regularization_loss
            + self._regular_consistency_weight * regular_consistency
        )

        return aug_total_loss

    def _augmentation_loss_neural_network_augmentation_adversarial(
        self, x_embeddings: torch.Tensor, augmented_x_embeddings: torch.Tensor
    ) -> torch.Tensor:
        aug_total_loss = -1 * info_nce_loss(x_embeddings, augmented_x_embeddings, temperature=1.0)
        return aug_total_loss

    def _calculate_augmentation_loss_neural_network_augmentation(
        self,
        x_embeddings: torch.Tensor,
        augmented_x_embeddings: torch.Tensor,
        augmentation_factor: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self._augmentation_network_training_mode
            == AutoTCLNeuralNetworkAugmentationTrainingMode.RELEVANT_INFORMATION_PRINCIPLE
        ):
            return self._augmentation_loss_network_augmentation_relevant_information_principle(
                x_embeddings, augmented_x_embeddings, augmentation_factor
            )
        if (
            self._augmentation_network_training_mode
            == AutoTCLNeuralNetworkAugmentationTrainingMode.ADVERSARIAL
        ):
            return self._augmentation_loss_neural_network_augmentation_adversarial(
                x_embeddings, augmented_x_embeddings
            )
        msg = f'Logic for training mode {self._augmentation_network_training_mode} not implemented.'
        raise ValueError(msg)

    # -------------- End of loss calculation functions --------------

    # -------------- Training step functions --------------

    def _augmentation_model(self) -> AutoTCLAugmentationTimeSeriesEncoder:
        """Return the augmentation model, raising if not initialized."""
        if self._augmentation_method is None:
            msg = 'Augmentation model not initialized'
            raise RuntimeError(msg)
        return self._augmentation_method

    def _training_step_function_default(self, batch: object, _batch_idx: int) -> torch.Tensor:
        x = extract_features_from_batch(batch)

        optimizer = cast('AdamW', self.optimizers())

        x = process_sample_length(sample=x, max_sample_length=self._max_train_length)

        augmented_x = self._augmentation_model().augment(x)

        augmented_x = augmented_x.to(x.device)

        x_embeddings = self._encoder(x)
        augmented_x_embeddings = self._encoder(augmented_x)

        encoder_total_loss = self._calculate_encoder_loss(x_embeddings, augmented_x_embeddings)

        self.log(
            'train_loss',
            encoder_total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )

        optimizer.zero_grad()
        self.manual_backward(encoder_total_loss)
        optimizer.step()

        self._averaged_encoder.update_parameters(self._encoder)

        return encoder_total_loss

    def _training_step_function_neural_network_augmentation(
        self, batch: object, _batch_idx: int
    ) -> None:
        x = extract_features_from_batch(batch)
        aug_model = self._augmentation_model()

        optimizers = cast('list[AdamW]', self.optimizers())
        main_optimizer, meta_optimizer = optimizers

        augmentation_total_loss = None

        if self.current_epoch % self._augmentation_network_training_ratio_step == 0:
            self._encoder.eval()
            aug_model.train()

            features = aug_model.get_features(x)
            augmentation_factor = features['augmentation_factor']
            augmented_x = features['augmented_data']
            x_embeddings = self._encoder(x)
            augmented_x_embeddings = self._encoder(augmented_x)

            augmentation_total_loss = self._calculate_augmentation_loss_neural_network_augmentation(
                x_embeddings, augmented_x_embeddings, augmentation_factor
            )

            meta_optimizer.zero_grad()
            self.manual_backward(augmentation_total_loss)
            meta_optimizer.step()

        self._encoder.train()
        aug_model.eval()

        augmented_x = aug_model.augment(x)

        x_embeddings = self._encoder(x)
        augmented_x_embeddings = self._encoder(augmented_x)

        encoder_total_loss = self._calculate_encoder_loss(x_embeddings, augmented_x_embeddings)

        main_optimizer.zero_grad()
        self.manual_backward(encoder_total_loss)
        main_optimizer.step()

        self._averaged_encoder.update_parameters(self._encoder)

        if augmentation_total_loss is not None:
            self.log_dict(
                {'enc_train_loss': encoder_total_loss, 'aug_train_loss': augmentation_total_loss},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self._sync_dist,
            )
        else:
            self.log_dict(
                {'enc_train_loss': encoder_total_loss},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self._sync_dist,
            )

    def _exec_training_step_function(self, batch: object, _batch_idx: int) -> torch.Tensor | None:
        if self._augmentation_mode == AutoTCLAugmentationMode.NEURAL_NETWORK:
            return self._training_step_function_neural_network_augmentation(batch, _batch_idx)
        return self._training_step_function_default(batch, _batch_idx)

    # -------------- End of training step functions ---------------

    def training_step(self, batch: object, batch_idx: int) -> torch.Tensor | None:
        """Execute a single training step.

        Args:
            batch: Input batch containing time series data.
            batch_idx: Index of the current batch.

        Returns:
            The total training loss, or None for neural network augmentation mode.
        """
        total_loss = self._exec_training_step_function(batch, batch_idx)
        return total_loss

    def validation_step(self, batch: object, _batch_idx: int) -> torch.Tensor:
        """Execute a single validation step.

        Args:
            batch: Input batch containing time series data.
            _batch_idx: Index of the current batch (unused).

        Returns:
            The validation loss computed on the averaged encoder.
        """
        x = extract_features_from_batch(batch)

        augmented_x = self._augmentation_model().augment(x)

        x_embeddings = self._averaged_encoder(x)
        augmented_x_embeddings = self._averaged_encoder(augmented_x)

        encoder_total_loss = self._calculate_encoder_loss(x_embeddings, augmented_x_embeddings)

        self.log_dict(
            {'val_loss': encoder_total_loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._sync_dist,
        )
        return encoder_total_loss

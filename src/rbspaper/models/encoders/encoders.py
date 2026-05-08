"""Time series encoder architectures for AutoTCL, CoST, and TS2Vec models."""

__all__ = [
    'AutoTCLAugmentationTimeSeriesEncoder',
    'AutoTCLTimeSeriesEncoder',
    'CoSTTimeSeriesEncoder',
    'TS2VecTimeSeriesEncoder',
]

from abc import ABC, abstractmethod
import itertools

from einops import rearrange, reduce
import torch
from torch import nn

from src.rbspaper.models.encoders.masking import generate_mask, generate_not_nan_mask, MaskMode
from src.rbspaper.models.layers import BandedFourierLayer
from src.rbspaper.models.layers.convolutions import Conv1dDilatedEncoder


class BaseTimeSeriesEncoder(nn.Module, ABC):
    """Base class with common mask handling and input processing for encoders.

    Args:
        input_dims: Number of input dimensions (features).
        output_dims: Number of output dimensions (embedding size).
        hidden_dims: Number of hidden dimensions in the encoder.
        feature_extractor_depth: Depth of the feature extractor (number of conv layers).
        dropout_rate: Dropout rate applied between layers.
        conv_kernel_size: Kernel size for convolutional layers.
        mask_mode: Default masking mode when not overridden at forward time.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 64,
        feature_extractor_depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
    ) -> None:
        super().__init__()

        self.mask_mode = mask_mode

        self.input_fc_layer = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = Conv1dDilatedEncoder(
            in_channels=hidden_dims,
            channels=[hidden_dims] * feature_extractor_depth + [output_dims],
            kernel_size=conv_kernel_size,
        )
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    @abstractmethod
    def _process_not_nan_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Handle NaN values in the input and return a cleaned tensor with a mask.

        Args:
            x: Raw input tensor.

        Returns:
            Tuple of (cleaned_tensor, boolean_not_nan_mask).
        """

    @abstractmethod
    def _apply_mask(
        self, x: torch.Tensor, not_nan_mask: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply a random mask and NaN mask to the input tensor.

        Args:
            x: Input tensor (after FC layer).
            not_nan_mask: Boolean mask of non-NaN positions.
            mask: Random mask to apply.

        Returns:
            Masked tensor.
        """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, *, mask_mode: MaskMode
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Forward pass through the encoder."""

    def _process_mask_mode(self, mask_mode: MaskMode | None) -> MaskMode:
        """Resolve effective mask mode based on training state.

        Args:
            mask_mode: Explicitly provided mask mode, or None to use default.

        Returns:
            Effective mask mode (ALL_TRUE when evaluating).
        """
        if mask_mode is None:
            mask_mode = self.mask_mode if self.training else MaskMode.ALL_TRUE
        return mask_mode

    def _common_forward(self, x: torch.Tensor, mask_mode: MaskMode | None = None) -> torch.Tensor:
        """Shared forward logic: process NaNs, apply FC, mask, and extract features.

        Args:
            x: Input tensor of shape (batch_size, time_steps, input_dims).
            mask_mode: Optional mask mode override.

        Returns:
            Encoded tensor of shape (batch_size, output_dims, time_steps).
        """
        # x: BatchSize x TimeSteps x InputDim
        x, not_nan_mask = self._process_not_nan_mask(x=x)

        x = self.input_fc_layer(x)  # BatchSize x TimeSteps x Channels

        mask_mode = self._process_mask_mode(mask_mode=mask_mode)

        mask = generate_mask(x=x, mask_mode=mask_mode)
        x = self._apply_mask(x=x, not_nan_mask=not_nan_mask, mask=mask)

        x = x.transpose(1, 2)  # BatchSize x Channels x TimeSteps
        x = self.feature_extractor(x)  # BatchSize x OutputChannels x TimeSteps

        return x


# -------- AutoTCL Time Series Encoders --------


class AutoTCLTimeSeriesEncoder(BaseTimeSeriesEncoder):
    """AutoTCL encoder with dilated convolutions and temporal decoding.

    Based on the AutoTCL paper: https://github.com/AslanDing/AutoTCL

    Args:
        input_dims: Number of input dimensions (features).
        output_dims: Number of output dimensions (embedding size).
        kernel_sizes: List of kernel sizes for the temporal decoders.
        hidden_dims: Number of hidden dimensions in the encoder.
        feature_extractor_depth: Depth of the feature extractor (number of conv layers).
        dropout_rate: Dropout rate applied between layers.
        conv_kernel_size: Kernel size for convolutional layers.
        mask_mode: Default masking mode when not overridden at forward time.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernel_sizes: list[int],
        hidden_dims: int = 64,
        feature_extractor_depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=feature_extractor_depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

        self.kernel_sizes = kernel_sizes
        self.temporal_feature_decoders = nn.ModuleList(
            [nn.Conv1d(output_dims, output_dims, k, padding=k - 1) for k in kernel_sizes]
        )

    def _process_not_nan_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero out NaN positions and return mask.

        Args:
            x: Raw input tensor.

        Returns:
            Tuple of (zeroed_tensor, boolean_not_nan_mask).
        """
        not_nan_mask = generate_not_nan_mask(x=x)
        nan_mask_as_float = not_nan_mask.float()
        modified_x = x.clone()
        modified_x = modified_x * nan_mask_as_float.unsqueeze(2)

        return modified_x, not_nan_mask

    def _apply_mask(
        self, x: torch.Tensor, not_nan_mask: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply combined random and NaN masks.

        Args:
            x: Input tensor after FC layer.
            not_nan_mask: Boolean mask of non-NaN positions.
            mask: Random mask to apply.

        Returns:
            Masked tensor.
        """
        mask = mask.float()
        nan_mask_as_float = not_nan_mask.float()
        mask = mask * nan_mask_as_float
        modified_x = x.clone()
        modified_x = modified_x * mask.unsqueeze(2)
        return modified_x

    def _decode_trend(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Decode temporal features using multi-kernel convolutions.

        Args:
            x: Encoded tensor of shape (batch_size, output_dims, time_steps).

        Returns:
            Tuple containing the averaged trend tensor.
        """
        trend = []
        for idx, mod in enumerate(self.temporal_feature_decoders):
            out = mod(x)  # BatchSize x Channels x TimeSteps
            if self.kernel_sizes[idx] != 1:
                out = out[..., : -(self.kernel_sizes[idx] - 1)]  # Adjust for kernel size if not 1
            trend.append(out.transpose(1, 2))  # BatchSize x TimeSteps x Channels

        trend = reduce(
            rearrange(
                trend, 'list BatchSize TimeSteps Channels -> list BatchSize TimeSteps Channels'
            ),
            'list BatchSize TimeSteps Channels -> BatchSize TimeSteps Channels',
            'mean',
        )
        return tuple(trend)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_tcn_output: bool = False,
        mask_mode: MaskMode = MaskMode.ALL_TRUE,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Forward pass for the AutoTCLTimeSeriesEncoder.

        Args:
            x: Input tensor of shape (batch_size, time_steps, input_dims).
            return_tcn_output: If True, return raw TCN output instead of decoded trend.
            mask_mode: Mask mode override.

        Returns:
            Trend tuple if return_tcn_output is False; otherwise TCN output tensor.
        """
        x = self._common_forward(x=x, mask_mode=mask_mode)

        if return_tcn_output:
            return x.transpose(1, 2)  # BatchSize x TimeSteps x OutputChannels

        trend = self._decode_trend(x)

        return trend


class AutoTCLAugmentationTimeSeriesEncoder(nn.Module):
    """AutoTCL encoder that learns augmentation via learned masking (reparameterization trick).

    Args:
        input_dims: Number of input dimensions (features).
        output_dims: Number of output dimensions (embedding size).
        kernel_sizes: List of kernel sizes for the temporal decoders.
        hidden_dims: Number of hidden dimensions in the encoder.
        feature_extractor_depth: Depth of the feature extractor (number of conv layers).
        dropout_rate: Dropout rate applied between layers.
        conv_kernel_size: Kernel size for convolutional layers.
        mask_mode: Default masking mode when not overridden at forward time.
        num_augmentation_channels: Number of channels in the augmentation factor.
        gumbel_bias: Bias for Gumbel-Softmax sampling.
        zeta: Upper bound for stretched Gumbel values.
        gamma_zeta: Lower bound offset for stretched Gumbel values.
        hard_mask: If True, apply straight-through estimator for hard masking.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernel_sizes: list[int],
        *,
        hidden_dims: int = 64,
        feature_extractor_depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        num_augmentation_channels: int = 1,
        gumbel_bias: float = 0.001,
        zeta: float = 1.0,
        gamma_zeta: float = 0.05,
        hard_mask: bool = True,
    ) -> None:
        super().__init__()

        self.gumbel_bias = gumbel_bias
        self.zeta = zeta
        self.gamma_zeta = -gamma_zeta
        self.hard_mask = hard_mask

        self.augmentation_network = AutoTCLTimeSeriesEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            kernel_sizes=kernel_sizes,
            hidden_dims=hidden_dims,
            feature_extractor_depth=feature_extractor_depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

        self.factor_augmentation_network = nn.Sequential(
            nn.Linear(output_dims, num_augmentation_channels), nn.Sigmoid()
        )
        self.augmentation_projector = nn.Sequential(
            nn.Linear(output_dims, output_dims),
            nn.ReLU(),
            nn.Linear(output_dims, num_augmentation_channels),
            nn.Sigmoid(),
        )

    def _sample_graph(
        self,
        sampling_weights: torch.Tensor,
        temperature: float = 1.0,
        bias: float = 0.0,
        *,
        is_training: bool = True,
    ) -> torch.Tensor:
        """Sample a graph using the Gumbel-Softmax reparameterization trick.

        Args:
            sampling_weights: Weights provided by the MLP.
            temperature: Annealing temperature for sampling (default 1.0).
            bias: Bias on the weights to control determinism (default 0.0).
            is_training: If False, sampling is fully deterministic (default True).

        Returns:
            Sampled graph tensor.
        """
        if is_training:
            bias = bias + self.gumbel_bias
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()).to(
                sampling_weights.device
            ) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)

        stretched_values = graph * (self.zeta - self.gamma_zeta) + self.gamma_zeta
        clipped = torch.clip(stretched_values, max=1.0, min=0.0)

        return clipped

    def get_parameters(self) -> itertools.chain:
        """Get the parameters of all augmentation-related sub-networks.

        Returns:
            Iterator over parameters of augmentation, factor, and projector networks.
        """
        return itertools.chain(
            self.augmentation_network.parameters(),
            self.factor_augmentation_network.parameters(),
            self.augmentation_projector.parameters(),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_tcn_output: bool = False,
        mask_mode: MaskMode = MaskMode.ALL_TRUE,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for the AutoTCLAugmentationTimeSeriesEncoder.

        Args:
            x: Input tensor of shape (batch_size, time_steps, input_dims).
            return_tcn_output: If True, return raw embeddings only.
            mask_mode: Mask mode override.

        Returns:
            Dict with 'embeddings', 'augmented_data', 'augmentation_factor',
            'projection_factor'. If return_tcn_output, only 'embeddings'.
        """
        trend = self.augmentation_network.forward(
            x=x, return_tcn_output=return_tcn_output, mask_mode=mask_mode
        )

        if not isinstance(trend, torch.Tensor):
            msg = f'Expected tensor from augmentation network, got {type(trend).__name__}'
            raise TypeError(msg)

        if return_tcn_output:
            return {'embeddings': trend}

        augmentation_factor = self.factor_augmentation_network(trend)

        trend_ = trend.clone()
        projection_factor = self.augmentation_projector(trend_.detach())

        augmentation_mask = self._sample_graph(
            sampling_weights=augmentation_factor, is_training=self.training
        )

        if self.hard_mask:
            hard_mask_h = (torch.sign(augmentation_mask - 0.5) + 1) / 2
            augmentation_mask = (hard_mask_h - augmentation_mask).detach() + augmentation_mask

        augmented_x = projection_factor * augmentation_mask * x

        return {
            'embeddings': trend,
            'augmented_data': augmented_x,
            'augmentation_factor': augmentation_factor,
            'projection_factor': projection_factor,
        }

    def augment(self, data: torch.Tensor) -> torch.Tensor:
        """Obtain augmented data from the augmentation network.

        Args:
            data: Input tensor to augment.

        Returns:
            Augmented tensor.
        """
        model_output: dict[str, torch.Tensor] = self.forward(data)

        augmented_x = model_output['augmented_data']

        return augmented_x

    def get_features(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get all feature outputs from a forward pass.

        Args:
            data: Input tensor.

        Returns:
            Dict of feature outputs from the forward pass.
        """
        model_output = self.forward(data)

        return model_output


# -------- End AutoTCL Time Series Encoders --------

# -------- CoST Time Series Encoders --------


class CoSTTimeSeriesEncoder(BaseTimeSeriesEncoder):
    """CoST encoder with trend-seasonal decomposition via Fourier layers.

    Based on the CoST paper: https://github.com/salesforce/CoST

    Args:
        input_dims: Number of input dimensions (features).
        output_dims: Number of output dimensions (embedding size).
        kernel_sizes: List of kernel sizes for the temporal decoders.
        length: Length of the input sequence.
        hidden_dims: Number of hidden dimensions (default 64).
        feature_extractor_depth: Depth of the feature extractor (default 10).
        dropout_rate: Dropout rate (default 0.1).
        conv_kernel_size: Kernel size for convolutions (default 3).
        mask_mode: Masking mode (default BINOMIAL).
        num_bands: Number of frequency bands for the Banded Fourier Layer (default 1).
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernel_sizes: list[int],
        length: int,
        hidden_dims: int = 64,
        feature_extractor_depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        num_bands: int = 1,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=feature_extractor_depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

        self.kernel_sizes = kernel_sizes

        self.component_dims = output_dims // 2

        self.temporal_feature_decoders = nn.ModuleList(
            [nn.Conv1d(output_dims, output_dims // 2, k, padding=k - 1) for k in kernel_sizes]
        )

        self.seasonal_feature_decoders = nn.ModuleList(
            [
                BandedFourierLayer(
                    in_channels=output_dims,
                    out_channels=self.component_dims,
                    band=b,
                    num_bands=num_bands,
                    length=length,
                )
                for b in range(num_bands)
            ]
        )

    def _process_not_nan_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero out NaN positions and return mask.

        Args:
            x: Raw input tensor.

        Returns:
            Tuple of (zeroed_tensor, boolean_not_nan_mask).
        """
        not_nan_mask = generate_not_nan_mask(x=x)
        modified_x = x.clone()
        modified_x[~not_nan_mask] = 0
        return modified_x, not_nan_mask

    def _apply_mask(
        self, x: torch.Tensor, not_nan_mask: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply combined random and NaN masks by zeroing.

        Args:
            x: Input tensor after FC layer.
            not_nan_mask: Boolean mask of non-NaN positions.
            mask: Random mask to apply.

        Returns:
            Masked tensor.
        """
        mask &= not_nan_mask
        modified_x = x.clone()
        modified_x[~mask] = 0
        return modified_x

    def _decode_trend(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Decode temporal (trend) features using multi-kernel convolutions.

        Args:
            x: Encoded tensor of shape (batch_size, output_dims, time_steps).

        Returns:
            Tuple containing the averaged trend tensor.
        """
        trend = []
        for idx, mod in enumerate(self.temporal_feature_decoders):
            out = mod(x)  # BatchSize x Channels x TimeSteps
            if self.kernel_sizes[idx] != 1:
                out = out[..., : -(self.kernel_sizes[idx] - 1)]  # Adjust for kernel size if not 1
            trend.append(out.transpose(1, 2))  # BatchSize x TimeSteps x Channels

        trend = reduce(
            rearrange(
                trend, 'list BatchSize TimeSteps Channels -> list BatchSize TimeSteps Channels'
            ),
            'list BatchSize TimeSteps Channels -> BatchSize TimeSteps Channels',
            'mean',
        )
        return tuple(trend)

    def _decode_season(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Decode seasonal features using Banded Fourier Layers.

        Args:
            x: Encoded tensor of shape (batch_size, time_steps, output_dims).

        Returns:
            Tuple of seasonal tensors from each Fourier band.
        """
        season = []
        for mod in self.seasonal_feature_decoders:
            out = mod(x)  # BatchSize x TimeSteps x Channels
            season.append(out)

        return tuple(season)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_tcn_output: bool = False,
        mask_mode: MaskMode = MaskMode.ALL_TRUE,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor] | torch.Tensor:
        """Forward pass for the CoSTTimeSeriesEncoder.

        Args:
            x: Input tensor of shape (batch_size, time_steps, input_dims).
            return_tcn_output: If True, return raw TCN output instead of decomposition.
            mask_mode: Mask mode override.

        Returns:
            Tuple of (trend_tuple, seasonal_tensor) if decomposing;
            otherwise raw TCN output tensor.
        """
        x = self._common_forward(x=x, mask_mode=mask_mode)

        if return_tcn_output:
            return x.transpose(1, 2)

        trend = self._decode_trend(x)

        x = x.transpose(1, 2)  # BatchSize x TimeSteps x OutputChannels

        season = self._decode_season(x)

        season = season[0]

        return trend, self.dropout_layer(season)


# -------- End CoST Time Series Encoders --------

# -------- TS2Vec Time Series Encoders --------


class TS2VecTimeSeriesEncoder(BaseTimeSeriesEncoder):
    """TS2Vec encoder based on dilated convolutions for hierarchical representation learning.

    Based on the TS2Vec paper: https://github.com/zhihanyue/ts2vec

    Args:
        input_dims: Number of input dimensions (features).
        output_dims: Number of output dimensions (embedding size).
        hidden_dims: Number of hidden dimensions in the encoder.
        feature_extractor_depth: Depth of the feature extractor (number of conv layers).
        dropout_rate: Dropout rate applied between layers.
        conv_kernel_size: Kernel size for convolutional layers.
        mask_mode: Default masking mode when not overridden at forward time.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 64,
        feature_extractor_depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=feature_extractor_depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

    def _process_not_nan_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero out NaN positions and return mask.

        Args:
            x: Raw input tensor.

        Returns:
            Tuple of (zeroed_tensor, boolean_not_nan_mask).
        """
        not_nan_mask = generate_not_nan_mask(x=x)
        modified_x = x.clone()
        modified_x[~not_nan_mask] = 0
        return modified_x, not_nan_mask

    def _apply_mask(
        self, x: torch.Tensor, not_nan_mask: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply combined random and NaN masks by zeroing.

        Args:
            x: Input tensor after FC layer.
            not_nan_mask: Boolean mask of non-NaN positions.
            mask: Random mask to apply.

        Returns:
            Masked tensor.
        """
        mask &= not_nan_mask
        modified_x = x.clone()
        modified_x[~mask] = 0
        return modified_x

    def forward(self, x: torch.Tensor, *, mask_mode: MaskMode = MaskMode.ALL_TRUE) -> torch.Tensor:
        """Forward pass for the TS2VecTimeSeriesEncoder.

        Args:
            x: Input tensor of shape (batch_size, time_steps, input_dims).
            mask_mode: Mask mode override.

        Returns:
            Encoded tensor of shape (batch_size, time_steps, output_dims).
        """
        x = self._common_forward(x=x, mask_mode=mask_mode)

        x = self.dropout_layer(x)
        x = x.transpose(1, 2)  # BatchSize x TimeSteps x OutputChannels

        return x

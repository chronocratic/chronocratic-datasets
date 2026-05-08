"""Utility functions for tensor manipulation and pooling in time series models."""

__all__ = [
    'apply_slicing',
    'concat_last_step_features',
    'extract_features_from_batch',
    'full_series_pooling',
    'integer_pooling',
    'multiscale_pooling',
    'process_sample_length',
    'process_sliding_window',
]

from typing import cast

from einops import rearrange
import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812 # torch.nn.functional convention

from src.rbspaper.utils import pad_tensor_with_nan


def extract_features_from_batch(batch: object) -> torch.Tensor:
    """Extract the features (inputs) from a batch.

    Args:
        batch: The input batch which may contain only features or a tuple with
            features and other elements (e.g., labels).

    Returns:
        The extracted features.

    Raises:
        ValueError: If the batch format is unsupported.
    """
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)):
        return cast('torch.Tensor', batch[0])
    msg = f'Unsupported batch format; {type(batch)}'
    raise ValueError(msg)


def process_sample_length(
    sample: torch.Tensor, max_sample_length: int | None = None
) -> torch.Tensor:
    """Process a sample by optionally truncating to a maximum length.

    Args:
        sample: Input tensor of shape (batch_size, sequence_length, ...).
        max_sample_length: Maximum allowed sequence length. If the sample
            exceeds this, a random window is extracted.

    Returns:
        The (potentially truncated) sample tensor.
    """
    if max_sample_length is not None and sample.size(1) > max_sample_length:
        device = sample.device
        rng = np.random.default_rng()
        window_offset = rng.integers(sample.size(1) - max_sample_length + 1)
        sample = sample[:, window_offset : window_offset + max_sample_length]

        sample = sample.to(device)

    return sample


def apply_slicing(tensor: torch.Tensor, slicing: slice | None = None) -> torch.Tensor:
    """Apply optional slicing along the time axis.

    Args:
        tensor: Input tensor of shape (batch_size, sequence_length, ...).
        slicing: The slice object to apply along the time dimension.

    Returns:
        The sliced tensor (or the original if slicing is None).
    """
    if slicing is not None:
        tensor = tensor[:, slicing]
    return tensor


def full_series_pooling(tensor: torch.Tensor, slicing: slice | None = None) -> torch.Tensor:
    """Apply max pooling over the entire sequence to obtain a single representation.

    Args:
        tensor: Input tensor of shape (batch_size, sequence_length, features).
        slicing: The slice object to apply after pooling.

    Returns:
        The pooled tensor of shape (batch_size, 1, features).
    """
    tensor = apply_slicing(tensor=tensor, slicing=slicing)
    pooled_tensor = F.max_pool1d(tensor.transpose(1, 2), kernel_size=tensor.size(1)).transpose(1, 2)
    return pooled_tensor


def multiscale_pooling(tensor: torch.Tensor, slicing: slice | None = None) -> torch.Tensor:
    """Apply max pooling at multiple scales and concatenate results.

    Args:
        tensor: Input tensor of shape (batch_size, sequence_length, features).
        slicing: The slice object to apply after each scale's pooling.

    Returns:
        The concatenated multiscale pooled tensor.
    """
    all_representations = []
    scale_factor = 0
    while (1 << scale_factor) + 1 < tensor.size(1):
        pooled_output = F.max_pool1d(
            tensor.transpose(1, 2),
            kernel_size=(1 << (scale_factor + 1)) + 1,
            stride=1,
            padding=1 << scale_factor,
        ).transpose(1, 2)
        pooled_output = apply_slicing(tensor=pooled_output, slicing=slicing)
        all_representations.append(pooled_output)
        scale_factor += 1
    multiscale_pooled_tensor = torch.cat(all_representations, dim=-1)
    return multiscale_pooled_tensor


def integer_pooling(
    tensor: torch.Tensor, encoding_window: int, slicing: slice | None = None
) -> torch.Tensor:
    """Apply max pooling with a fixed kernel size along the time axis.

    Args:
        tensor: Input tensor of shape (batch_size, sequence_length, features).
        encoding_window: The kernel size for max pooling.
        slicing: The slice object to apply after pooling.

    Returns:
        The pooled tensor with integer-based pooling applied.
    """
    pooled_tensor = F.max_pool1d(
        tensor.transpose(1, 2), kernel_size=encoding_window, stride=1, padding=encoding_window // 2
    ).transpose(1, 2)
    if encoding_window % 2 == 0:
        pooled_tensor = pooled_tensor[:, :-1]
    pooled_tensor = apply_slicing(tensor=pooled_tensor, slicing=slicing)
    return pooled_tensor


def process_sliding_window(
    input_tensor: torch.Tensor, left_index: int, right_index: int, time_series_length: int
) -> torch.Tensor:
    """Extract a sliding window from the input tensor with NaN padding at boundaries.

    Args:
        input_tensor: Input tensor of shape (batch_size, sequence_length, ...).
        left_index: The left (start) index for the window slice.
        right_index: The right (end) index for the window slice.
        time_series_length: The total length of the original time series.

    Returns:
        The padded sliding window tensor.
    """
    return pad_tensor_with_nan(
        tensor=input_tensor[:, max(left_index, 0) : min(right_index, time_series_length)],
        left_pad=-left_index if left_index < 0 else 0,
        right_pad=right_index - time_series_length if right_index > time_series_length else 0,
        axis=1,
    )


def concat_last_step_features(
    trend_embeddings: torch.Tensor, seasonality_embeddings: torch.Tensor
) -> torch.Tensor:
    """Extract last time-step features, concatenate, and add a singleton dimension.

    Args:
        trend_embeddings: Trend tensor of shape (batch_size, sequence_length, feature_dim).
        seasonality_embeddings: Seasonality tensor of shape
            (batch_size, sequence_length, feature_dim).

    Returns:
        Concatenated tensor of shape (batch_size, 1, 2 * feature_dim).
    """
    # Extract features from the last time step
    last_step_out_trend = trend_embeddings[:, -1, :]  # Shape: (batch_size, feature_dim)
    last_step_out_seasonality = seasonality_embeddings[:, -1, :]  # Shape: (batch_size, feature_dim)

    # Concatenate along the feature dimension
    concatenated_features = torch.cat(
        [last_step_out_trend, last_step_out_seasonality], dim=-1
    )  # Shape: (batch_size, total_feature_dim)

    # Rearrange to add an extra dimension
    concatenated_features = rearrange(concatenated_features, 'b d -> b () d')

    return concatenated_features

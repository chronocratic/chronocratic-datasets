"""Mask generation utilities for time series encoders."""

__all__ = ['MaskMode', 'generate_mask', 'generate_not_nan_mask']

from collections.abc import Callable
from enum import Enum

import numpy as np
import torch


class MaskMode(Enum):
    """Modes for generating random masks over time series data."""

    BINOMIAL = 'binomial'
    CONTINUOUS = 'continuous'
    ALL_TRUE = 'all_true'
    ALL_FALSE = 'all_false'
    MASK_LAST = 'mask_last'


def generate_continuous_mask(
    batch_size: int, seq_length: int, n_segments: int = 5, segment_length: float = 0.1
) -> torch.Tensor:
    """Generate a mask with continuous masked segments.

    Args:
        batch_size: The batch size.
        seq_length: The sequence length.
        n_segments: Number of segments to mask. If float, interpreted as fraction
            of seq_length.
        segment_length: Length of each segment. If float, interpreted as fraction
            of seq_length.

    Returns:
        Boolean mask tensor of shape (batch_size, seq_length).
    """
    rng = np.random.default_rng()

    mask = torch.full(size=(batch_size, seq_length), fill_value=True, dtype=torch.bool)

    if isinstance(n_segments, float):
        n_segments = int(n_segments * seq_length)
    n_segments = max(min(n_segments, seq_length // 2), 1)

    if isinstance(segment_length, float):
        segment_length = int(segment_length * seq_length)
    segment_length = max(segment_length, 1)

    for i in range(batch_size):
        for _ in range(n_segments):
            start = rng.integers(seq_length - segment_length + 1)
            mask[i, start : start + segment_length] = False

    return mask


def generate_binomial_mask(
    batch_size: int, seq_length: int, probability: float = 0.5
) -> torch.Tensor:
    """Generate a mask with independently masked elements.

    Args:
        batch_size: The batch size.
        seq_length: The sequence length.
        probability: Probability of masking each element.

    Returns:
        Boolean mask tensor of shape (batch_size, seq_length).
    """
    rng = np.random.default_rng()

    return torch.from_numpy(rng.binomial(1, probability, size=(batch_size, seq_length))).to(
        torch.bool
    )


def generate_all_true_mask(batch_size: int, seq_length: int) -> torch.Tensor:
    """Generate a mask where all elements are True (no masking).

    Args:
        batch_size: The batch size.
        seq_length: The sequence length.

    Returns:
        Boolean mask tensor of shape (batch_size, seq_length), all True.
    """
    return torch.full(size=(batch_size, seq_length), fill_value=True, dtype=torch.bool)


def generate_all_false_mask(batch_size: int, seq_length: int) -> torch.Tensor:
    """Generate a mask where all elements are False (full masking).

    Args:
        batch_size: The batch size.
        seq_length: The sequence length.

    Returns:
        Boolean mask tensor of shape (batch_size, seq_length), all False.
    """
    return torch.full(size=(batch_size, seq_length), fill_value=False, dtype=torch.bool)


def generate_mask_last_mask(batch_size: int, seq_length: int) -> torch.Tensor:
    """Generate a mask where all elements are True except the last.

    Args:
        batch_size: The batch size.
        seq_length: The sequence length.

    Returns:
        Boolean mask tensor of shape (batch_size, seq_length).
    """
    mask = torch.full(size=(batch_size, seq_length), fill_value=True, dtype=torch.bool)
    mask[:, -1] = False
    return mask


def get_mask_function(mask_mode: MaskMode | str) -> Callable[..., torch.Tensor]:
    """Return the mask generation function for the given mode.

    Args:
        mask_mode: Mask mode enum or string identifier.

    Returns:
        Callable that generates a mask given batch_size and seq_length.

    Raises:
        ValueError: If mask_mode is unsupported.
    """
    if isinstance(mask_mode, str):
        try:
            mask_mode = MaskMode(mask_mode)
        except ValueError as exc:
            message = f'Unsupported mask mode: {mask_mode!r}'
            raise ValueError(message) from exc

    mask_functions = {
        MaskMode.BINOMIAL: generate_binomial_mask,
        MaskMode.CONTINUOUS: generate_continuous_mask,
        MaskMode.ALL_TRUE: generate_all_true_mask,
        MaskMode.ALL_FALSE: generate_all_false_mask,
        MaskMode.MASK_LAST: generate_mask_last_mask,
    }

    if mask_mode not in mask_functions:
        message = f'Unsupported mask mode: {mask_mode!r}'
        raise ValueError(message)

    return mask_functions[mask_mode]


def generate_mask(x: torch.Tensor, mask_mode: MaskMode) -> torch.Tensor:
    """Generate a mask for the input tensor based on the specified mask mode.

    Args:
        x: Input tensor of shape (batch_size, sequence_length, ...).
        mask_mode: The mask mode to use.

    Returns:
        Boolean mask tensor of shape (batch_size, sequence_length) on same device.
    """
    mask_function = get_mask_function(mask_mode)
    return mask_function(x.size(0), x.size(1)).to(x.device)


def generate_not_nan_mask(x: torch.Tensor) -> torch.Tensor:
    """Generate a mask that is True for non-NaN values and False for NaN values.

    Args:
        x: Input tensor of shape (batch_size, sequence_length, features).

    Returns:
        Boolean mask tensor of shape (batch_size, sequence_length).
    """
    return ~x.isnan().any(dim=-1)

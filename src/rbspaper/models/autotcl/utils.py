"""Utility functions for AutoTCL model evaluation and metrics."""

__all__ = ['calculate_mutual_information', 'calculate_regular_consistency']

import numpy as np
import torch

from src.rbspaper.models.encoders import AutoTCLAugmentationTimeSeriesEncoder
from src.rbspaper.models.losses import l1_out_loss


def calculate_regular_consistency(weights: torch.Tensor) -> torch.Tensor:
    """Calculate regular consistency from weights by comparing nearby time steps.

    Args:
        weights: Weight tensor of shape (batch_size, time_steps, channels).

    Returns:
        Mean consistency measure across the batch.
    """
    batch_size, time_steps, _ = weights.shape

    # Select random time steps for comparison
    selected_steps = torch.randint(1, time_steps - 2, [batch_size])
    left_steps = selected_steps - 1
    right_steps = selected_steps + 1
    other_selected_steps = torch.randint(1, time_steps - 2, [batch_size])

    # Create mask to differentiate between near and far time steps
    mask = torch.where(
        (other_selected_steps - selected_steps) > 1,
        torch.ones_like(other_selected_steps),
        torch.zeros_like(selected_steps),
    ).to(weights.device)

    # Calculate differences for consistency
    differences = (
        mask.view(1, batch_size, 1)
        * torch.abs(weights[:, selected_steps, :] - weights[:, other_selected_steps, :])
        + torch.abs(weights[:, selected_steps, :] - weights[:, left_steps, :])
        + torch.abs(weights[:, selected_steps, :] - weights[:, right_steps, :])
        + (1 - mask).view(1, batch_size, 1)
        * (1 - torch.abs(weights[:, selected_steps, :] - weights[:, other_selected_steps, :]))
    )

    return differences.mean()


def calculate_mutual_information(
    batch: torch.Tensor,
    *,
    augmentation_method: AutoTCLAugmentationTimeSeriesEncoder,
    max_train_length: int,
) -> float:
    """Calculate mutual information between original and augmented data via L1out loss.

    Args:
        batch: Input batch of data (batch_size, sequence_length, features).
        augmentation_method: Augmentation encoder to apply.
        max_train_length: Maximum sequence length (truncates if longer).

    Returns:
        Average MI between original and augmented data (as float).
    """
    with torch.inference_mode():
        x = batch
        device = x.device
        rng = np.random.default_rng()

        if max_train_length is not None and x.size(1) > max_train_length:
            window_offset = rng.integers(x.size(1) - max_train_length + 1)
            x = x[:, window_offset : window_offset + max_train_length]
        x = x.to(device)

        original_x = x
        augmented_x = augmentation_method.augment(x)

        batch_info_nce_loss = l1_out_loss(original_x, augmented_x)

    return batch_info_nce_loss.item()

"""Dilated convolutional encoder layers for time series feature extraction."""

__all__ = ['Conv1dDilatedEncoder']

from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812 # torch.nn.functional convention

from src.rbspaper.models.layers.convolutions.same_pad import Conv1dSamePadMultiBlock


class Conv1dDilatedEncoder(nn.Module):
    """Sequence of Conv1dSamePadMultiBlock layers with exponentially increasing dilation.

    Args:
        in_channels: Number of input channels.
        channels: List of output channels for each layer stage.
        kernel_size: Kernel size for all convolutional layers.
        stride: Stride for the first block in each stage.
        num_blocks: Number of convolutional layers per stage.
        activation_fn: Activation function between conv layers.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        kernel_size: int,
        stride: int = 1,
        num_blocks: int = 2,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *[
                Conv1dSamePadMultiBlock(
                    in_channels=channels[i - 1] if i > 0 else in_channels,
                    out_channels=channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    stride=stride,
                    num_blocks=num_blocks,
                    activation_fn=activation_fn,
                    is_final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dilated encoder.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            Encoded tensor of shape (batch_size, channels[-1], sequence_length).
        """
        return self.net(x)

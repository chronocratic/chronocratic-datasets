"""Same-padding 1D convolutional layers that preserve output length."""

__all__ = ['Conv1dSamePad', 'Conv1dSamePadMultiBlock']

from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812 # torch.nn.functional convention


class Conv1dSamePad(nn.Module):
    """1D convolution that preserves the input sequence length via same-padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with same-padding logic.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        output = self.conv(x)
        if self.remove > 0:
            output = output[:, :, : -self.remove]
        return output


class Conv1dSamePadMultiBlock(nn.Module):
    """Residual block of multiple Conv1dSamePad layers with activations and skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        stride: int = 1,
        num_blocks: int = 2,
        *,
        is_final: bool = False,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
    ) -> None:
        super().__init__()

        self.activation_fn = activation_fn

        self.__initiate_blocks(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            num_blocks=num_blocks,
        )

        self.__initiate_projector(
            in_channels=in_channels, out_channels=out_channels, stride=stride, is_final=is_final
        )

    def __initiate_blocks(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        stride: int,
        num_blocks: int,
    ) -> None:
        """Initialize the sequence of Conv1dSamePad convolutional blocks.

        Args:
            in_channels: Input channels for the first block.
            out_channels: Output channels for all blocks.
            kernel_size: Kernel size for all convolutions.
            dilation: Dilation factor for all convolutions.
            stride: Stride for the first block (subsequent blocks use stride 1).
            num_blocks: Number of conv layers in this block sequence.
        """
        self.blocks = nn.ModuleList()

        self.blocks.append(
            Conv1dSamePad(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
            )
        )

        for _i in range(1, num_blocks):
            self.blocks.append(
                Conv1dSamePad(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=1,
                )
            )

    def __initiate_projector(
        self, in_channels: int, out_channels: int, stride: int, *, is_final: bool = False
    ) -> None:
        """Initialize the 1x1 convolutional projector for the skip connection.

        Args:
            in_channels: Input channels for the projector.
            out_channels: Output channels for the projector.
            stride: Stride to match the main path.
            is_final: Whether this block is the final one in the encoder.
        """
        if stride == 1:
            if in_channels != out_channels or is_final:
                self.projector = nn.Conv1d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1
                )
            else:
                self.projector = None
        else:
            self.projector = nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            Output tensor with residual addition applied.
        """
        residual = x if self.projector is None else self.projector(x)
        for block in self.blocks:
            x = self.activation_fn(x)
            x = block(x)
        return x + residual

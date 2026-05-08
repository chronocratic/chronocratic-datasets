"""Shared utility helpers for RBSPaper."""

import torch


def pad_tensor_with_nan(
    *, tensor: torch.Tensor, left_pad: int = 0, right_pad: int = 0, axis: int = 0
) -> torch.Tensor:
    """Pad tensor with NaNs on a selected axis.

    Args:
        tensor: Input tensor.
        left_pad: Number of NaN entries to prepend on ``axis``.
        right_pad: Number of NaN entries to append on ``axis``.
        axis: Axis to pad.

    Returns:
        Tensor padded with NaNs.
    """
    if left_pad == 0 and right_pad == 0:
        return tensor

    pad_shape = list(tensor.shape)
    segments: list[torch.Tensor] = []

    if left_pad > 0:
        pad_shape[axis] = left_pad
        left = torch.full(
            size=tuple(pad_shape), fill_value=float('nan'), device=tensor.device, dtype=tensor.dtype
        )
        segments.append(left)

    segments.append(tensor)

    if right_pad > 0:
        pad_shape[axis] = right_pad
        right = torch.full(
            size=tuple(pad_shape), fill_value=float('nan'), device=tensor.device, dtype=tensor.dtype
        )
        segments.append(right)

    return torch.cat(tensors=segments, dim=axis)

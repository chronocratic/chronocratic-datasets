"""Data transformation helpers for PyTorch datasets."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ['convert_data_to_np_array', 'convert_numpy_to_tensor', 'expand_data_dimensionality']


def convert_numpy_to_tensor(data: np.ndarray, dtype: str = 'float') -> torch.Tensor:
    """Convert a numpy array to a PyTorch tensor.

    Args:
        data: Input numpy array.
        dtype: Target dtype name ('float', 'long', 'int', 'double').

    Returns:
        PyTorch tensor on CPU.

    Raises:
        TypeError: If data is not a numpy array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f'Expected np.ndarray, got {type(data).__name__}. '
            'Use convert_data_to_np_array() for list/tuple inputs.'
        )
    dtype_map = {
        'float': torch.float,
        'long': torch.long,
        'int': torch.int,
        'double': torch.double,
    }
    return torch.from_numpy(data).to(dtype=dtype_map[dtype])


def convert_data_to_np_array(data: list | tuple, dtype: str = 'float') -> np.ndarray:
    """Convert a list or tuple to a numpy array.

    Args:
        data: Input iterable.
        dtype: Target dtype name ('float' or 'int').

    Returns:
        Numpy array.
    """
    dtype_map = {'float': np.float32, 'int': np.int32}
    return np.array(data).astype(dtype_map[dtype])


def expand_data_dimensionality(
    data: np.ndarray | torch.Tensor | list | tuple, expand_dims_axis: int
) -> np.ndarray:
    """Expand data dimensionality by one along a specified axis.

    Args:
        data: Input array-like.
        expand_dims_axis: Axis along which to expand.

    Returns:
        NumPy array with an additional dimension.

    Raises:
        ValueError: If expand_dims_axis is out of range for the input array.
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    ndim = data.ndim
    if expand_dims_axis < 0 or expand_dims_axis > ndim:
        raise ValueError(
            f'expand_dims_axis={expand_dims_axis} is out of range '
            f'for input with {ndim} dimensions. Must be in [0, {ndim}].'
        )
    return np.expand_dims(data, axis=expand_dims_axis)

"""Data transformation helpers for PyTorch datasets."""

import numpy as np
import torch

__all__ = ['convert_numpy_to_tensor', 'expand_data_dimensionality']


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
        msg = f'Expected np.ndarray, got {type(data).__name__}.'
        raise TypeError(msg)
    dtype_map = {'float': torch.float, 'long': torch.long, 'int': torch.int, 'double': torch.double}
    if dtype not in dtype_map:
        msg = f'Unsupported dtype "{dtype}". Choose from {list(dtype_map.keys())}.'
        raise ValueError(msg)
    return torch.from_numpy(data.copy()).to(dtype=dtype_map[dtype])


def expand_data_dimensionality(
    data: np.ndarray | torch.Tensor | list | tuple, expand_dims_axis: int
) -> np.ndarray | torch.Tensor:
    """Expand data dimensionality by one along a specified axis.

    Args:
        data: Input array-like.
        expand_dims_axis: Axis along which to expand.

    Returns:
        Array with an additional dimension, preserving the input type
        (torch.Tensor or np.ndarray).

    Raises:
        ValueError: If expand_dims_axis is out of range for the input array.
    """
    was_tensor = isinstance(data, torch.Tensor)
    if was_tensor:
        data = data.numpy()
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    ndim = data.ndim
    if expand_dims_axis < 0 or expand_dims_axis > ndim:
        msg = (
            f'expand_dims_axis={expand_dims_axis} is out of range '
            f'for input with {ndim} dimensions. Must be in [0, {ndim}].'
        )
        raise ValueError(msg)
    result = np.expand_dims(data, axis=expand_dims_axis)
    return torch.from_numpy(result) if was_tensor else result

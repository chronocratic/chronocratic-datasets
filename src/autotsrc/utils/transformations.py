__all__ = ['convert_data_to_np_array', 'convert_numpy_to_tensor', 'expand_data_dimensionality']

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch


def convert_numpy_to_tensor(
    data: np.ndarray,
    dtype: Literal['float', 'long', 'int', 'double'] = 'float',
) -> torch.Tensor:
    """Convert a NumPy array into a torch Tensor with the requested dtype."""
    dtype_map = {
        'float': torch.float,
        'long': torch.long,
        'int': torch.int,
        'double': torch.double,
    }
    tensor = torch.from_numpy(data)

    return tensor.to(dtype=dtype_map[dtype])


def convert_data_to_np_array(
    data: Sequence[float] | Sequence[int], dtype: Literal['float', 'int'] = 'float'
) -> np.ndarray:
    """Convert a numeric sequence into a NumPy array with a target dtype."""
    dtype_map = {'float': np.float32, 'int': np.int32}
    return np.array(data).astype(dtype_map[dtype])


def expand_data_dimensionality(
    data: np.ndarray | torch.Tensor | list | tuple,
    expand_dims_axis: int,
) -> np.ndarray | torch.Tensor:
    """Expand dimensionality while preserving tensor inputs as tensors."""
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(dim=expand_dims_axis)
    return np.expand_dims(np.asarray(data), axis=expand_dims_axis)

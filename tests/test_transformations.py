"""Tests for data transformation utilities (DST-01, DST-02).

Verifies that convert_numpy_to_tensor, expand_data_dimensionality,
and convert_data_to_np_array produce correct output types and shapes.
"""

import numpy as np
import pytest
import torch

from tscollection.datasets.transformations import (
    convert_data_to_np_array,
    convert_numpy_to_tensor,
    expand_data_dimensionality,
)


def test_convert_numpy_to_tensor_float():
    """DST-01: convert_numpy_to_tensor returns torch.Tensor with dtype torch.float."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = convert_numpy_to_tensor(data, dtype='float')
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float


def test_convert_numpy_to_tensor_long():
    """convert_numpy_to_tensor returns torch.Tensor with dtype torch.long."""
    data = np.array([0, 1, 2], dtype=np.int64)
    result = convert_numpy_to_tensor(data, dtype='long')
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.long


def test_expand_data_dimensionality():
    """DST-01: expand_data_dimensionality adds dimension at specified axis."""
    data = np.array([1.0, 2.0, 3.0])  # shape (3,)
    result = expand_data_dimensionality(data, expand_dims_axis=1)
    assert result.shape == (3, 1)
    assert isinstance(result, np.ndarray)


def test_convert_data_to_np_array():
    """DST-02: convert_data_to_np_array converts list/tuple to numpy array."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    result = convert_data_to_np_array(data, dtype='float')
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (2, 2)


def test_convert_numpy_to_tensor_type_error_list():
    """TST-03: convert_numpy_to_tensor raises TypeError for list input (lines 25-29)."""
    with pytest.raises(TypeError, match='Expected np.ndarray'):
        convert_numpy_to_tensor(data=[1, 2, 3], dtype='float')


def test_convert_numpy_to_tensor_type_error_dict():
    """TST-03: convert_numpy_to_tensor raises TypeError for dict input (lines 25-29)."""
    with pytest.raises(TypeError, match='Expected np.ndarray'):
        convert_numpy_to_tensor(data={'a': 1}, dtype='float')


def test_convert_numpy_to_tensor_unsupported_dtype():
    """TST-03: convert_numpy_to_tensor raises ValueError for unsupported dtype (lines 39-40)."""
    data = np.array([1.0])
    with pytest.raises(ValueError, match='Unsupported dtype'):
        convert_numpy_to_tensor(data=data, dtype='bool')


def test_expand_data_dimensionality_axis_out_of_range():
    """TST-03: expand_data_dimensionality raises ValueError for axis out of range (lines 82-86)."""
    data = np.array([1.0, 2.0])  # 1-D array
    with pytest.raises(ValueError, match='out of range'):
        expand_data_dimensionality(data, expand_dims_axis=5)


def test_expand_data_dimensionality_list_input():
    """TST-03: expand_data_dimensionality handles list input via np.asarray conversion (line 78)."""
    result = expand_data_dimensionality(data=[1.0, 2.0], expand_dims_axis=1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1)


def test_expand_data_dimensionality_tensor_input_preserves_type():
    """TST-03: expand_data_dimensionality preserves torch.Tensor type (lines 74-76, 90)."""
    data = torch.tensor([1.0, 2.0])
    result = expand_data_dimensionality(data, expand_dims_axis=1)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 1)

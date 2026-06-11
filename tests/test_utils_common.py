"""Tests for common utility functions.

Verifies that DataForm enum has correct StrEnum members and
flatten_list_of_np_arrays produces correct 1-D flattened output.
"""

import numpy as np

from chronocratic.datasets.enums.data import DataForm
from chronocratic.datasets.utils.common import flatten_list_of_np_arrays

# --------------------------------------------------------------------------- #
# DataForm enum tests                                                          #
# --------------------------------------------------------------------------- #


def test_dataform_regular_value() -> None:
    """DataForm.REGULAR equals 'regular' string."""
    assert DataForm.REGULAR == "regular"


def test_dataform_nested_value() -> None:
    """DataForm.NESTED equals 'nested' string."""
    assert DataForm.NESTED == "nested"


def test_dataform_multi_files_value() -> None:
    """DataForm.MULTI_FILES equals 'multi_files' string."""
    assert DataForm.MULTI_FILES == "multi_files"


def test_dataform_is_strenum() -> None:
    """DataForm is a StrEnum subclass."""
    from enum import StrEnum

    assert issubclass(DataForm, StrEnum)


# --------------------------------------------------------------------------- #
# flatten_list_of_np_arrays tests                                              #
# --------------------------------------------------------------------------- #


def test_flatten_two_arrays() -> None:
    """flatten_list_of_np_arrays concatenates two arrays into 1-D."""
    result = flatten_list_of_np_arrays(list_of_np_arrays=[np.array([1, 2]), np.array([3, 4])])
    assert list(result) == [1, 2, 3, 4]


def test_flatten_float_arrays() -> None:
    """flatten_list_of_np_arrays preserves float dtype."""
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    result = flatten_list_of_np_arrays(list_of_np_arrays=[a, b])
    assert result.dtype == np.float64
    assert result.shape == (4,)


def test_flatten_single_array() -> None:
    """flatten_list_of_np_arrays with single element list."""
    result = flatten_list_of_np_arrays(list_of_np_arrays=[np.array([5, 6, 7])])
    assert list(result) == [5, 6, 7]

"""Tests for common utility functions.

Verifies that DataForm enum has correct StrEnum members and
flatten_list_of_np_arrays produces correct 1-D flattened output.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from chronocratic.datasets.enums.data import DataForm
from chronocratic.datasets.utils.common import encode_labels_jointly, flatten_list_of_np_arrays

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


# --------------------------------------------------------------------------- #
# encode_labels_jointly tests                                                 #
# --------------------------------------------------------------------------- #


def test_encode_labels_jointly_negative_binary() -> None:
    """{-1, 1} labels encode to {0, 1} with classes_ == [-1, 1]."""
    train_codes, test_codes, classes = encode_labels_jointly(
        train_labels=np.array([-1, 1, -1]), test_labels=np.array([1, -1])
    )
    assert set(train_codes) <= {0, 1}
    assert set(test_codes) <= {0, 1}
    assert list(classes) == [-1, 1]


def test_encode_labels_jointly_class_only_in_test() -> None:
    """A class present only in test still gets a code, and K counts it."""
    _train_codes, test_codes, classes = encode_labels_jointly(
        train_labels=np.array([1, 2]), test_labels=np.array([1, 2, 3])
    )
    assert len(classes) == 3
    label_to_code = dict(zip(classes, range(len(classes)), strict=True))
    assert test_codes[list(np.array([1, 2, 3])).index(3)] == label_to_code[3]


def test_encode_labels_jointly_same_value_same_code() -> None:
    """The same original value gets the same code in train and test."""
    train_codes, test_codes, classes = encode_labels_jointly(
        train_labels=np.array([1, 2, 1]), test_labels=np.array([2, 1])
    )
    label_to_code = dict(zip(classes, range(len(classes)), strict=True))
    assert train_codes[0] == label_to_code[1]
    assert test_codes[1] == label_to_code[1]
    assert train_codes[1] == label_to_code[2]
    assert test_codes[0] == label_to_code[2]


def test_encode_labels_jointly_order_preserved() -> None:
    """[1, 2, 3] maps to [0, 1, 2] in natural numeric order."""
    train_codes, test_codes, classes = encode_labels_jointly(
        train_labels=np.array([1, 2, 3]), test_labels=np.array([1, 2, 3])
    )
    assert list(train_codes) == [0, 1, 2]
    assert list(test_codes) == [0, 1, 2]
    assert list(classes) == [1, 2, 3]


def test_encode_labels_jointly_matches_sklearn_labelencoder() -> None:
    """String labels match plain LabelEncoder().fit_transform on the same values."""
    values = np.array(["b", "a", "c"])
    expected = LabelEncoder().fit_transform(values)
    train_codes, test_codes, _classes = encode_labels_jointly(
        train_labels=values, test_labels=values
    )
    assert list(train_codes) == list(expected)
    assert list(test_codes) == list(expected)


def test_encode_labels_jointly_output_dtype() -> None:
    """Encoded arrays are int64."""
    train_codes, test_codes, _classes = encode_labels_jointly(
        train_labels=np.array([1, 2]), test_labels=np.array([1, 2])
    )
    assert train_codes.dtype == np.int64
    assert test_codes.dtype == np.int64

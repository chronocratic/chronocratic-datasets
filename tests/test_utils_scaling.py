"""Tests for scaling utility functions (UTI-02).

Verifies that create_data_scaler correctly wires ScalingMethod and DataForm
enums, applies MinMax/Standard scaling across regular, nested, and
multi-file data forms, and returns unchanged data when scale=False.
"""

import numpy as np
import pandas as pd
import pytest

from tscollection.datasets.enums.data import DataForm, ScalingMethod


# --------------------------------------------------------------------------- #
# create_data_scaler import tests                                              #
# --------------------------------------------------------------------------- #


def test_create_data_scaler_import() -> None:
    """UTI-02: create_data_scaler is importable from scaling module."""
    from tscollection.datasets.utils.scaling import create_data_scaler

    assert callable(create_data_scaler)


def test_no_dataformenum_defined() -> None:
    """UTI-02: scaling.py does not define a local DataFormEnum class."""
    import tscollection.datasets.utils.scaling as scaling_mod

    assert not hasattr(scaling_mod, 'DataFormEnum')


# --------------------------------------------------------------------------- #
# create_data_scaler — REGULAR data form                                       #
# --------------------------------------------------------------------------- #


def test_create_data_scaler_regular_minmax() -> None:
    """UTI-02: create_data_scaler with REGULAR data and MinMax scaling.

    Train data is fit, all splits are transformed.
    Values should be in the specified range.
    """
    from tscollection.datasets.utils.scaling import create_data_scaler

    scaler_fn = create_data_scaler(
        scale=True,
        scaling_range=(0.0, 1.0),
        scaling_method=ScalingMethod.MINMAX,
        data_form=DataForm.REGULAR,
    )
    train = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    valid = np.array([[2.5, 15.0], [7.5, 25.0]])
    test = np.array([[0.0, 0.0], [10.0, 40.0]])

    scaled_train, scaled_valid, scaled_test = scaler_fn(
        train_data=train, valid_data=valid, test_data=test
    )

    assert scaled_train.min() >= 0.0
    assert scaled_train.max() <= 1.0
    assert scaled_train.shape == train.shape
    assert np.allclose(scaled_train[0], [0.0, 0.0])
    assert np.allclose(scaled_train[2], [1.0, 1.0])


def test_create_data_scaler_regular_standard() -> None:
    """UTI-02: create_data_scaler with REGULAR data and Standard scaling.

    Values should be zero-mean, unit-variance per feature.
    """
    from tscollection.datasets.utils.scaling import create_data_scaler

    scaler_fn = create_data_scaler(
        scale=True,
        scaling_range=(0.0, 1.0),
        scaling_method=ScalingMethod.STANDARD,
        data_form=DataForm.REGULAR,
    )
    train = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    valid = np.array([[2.5, 15.0]])
    test = np.array([[7.5, 25.0]])

    scaled_train, scaled_valid, scaled_test = scaler_fn(
        train_data=train, valid_data=valid, test_data=test
    )

    assert scaled_train.shape == train.shape
    mean = scaled_train.mean(axis=0)
    std = scaled_train.std(axis=0, ddof=1)
    np.testing.assert_allclose(mean, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(std, [1.0, 1.0], atol=1e-6)


def test_create_data_scaler_regular_dataframe() -> None:
    """UTI-02: create_data_scaler preserves DataFrame type for regular data."""
    from tscollection.datasets.utils.scaling import create_data_scaler

    scaler_fn = create_data_scaler(
        scale=True,
        scaling_range=(0.0, 1.0),
        scaling_method=ScalingMethod.MINMAX,
        data_form=DataForm.REGULAR,
    )
    train = pd.DataFrame({'a': [0.0, 10.0], 'b': [10.0, 30.0]})
    test = pd.DataFrame({'a': [5.0], 'b': [20.0]})

    scaled_train, scaled_valid, scaled_test = scaler_fn(
        train_data=train, valid_data=None, test_data=test
    )

    assert isinstance(scaled_train, pd.DataFrame)
    assert list(scaled_train.columns) == ['a', 'b']


# --------------------------------------------------------------------------- #
# create_data_scaler — NESTED data form                                        #
# --------------------------------------------------------------------------- #


def test_create_data_scaler_nested_preserves_shape() -> None:
    """UTI-02: create_data_scaler with NESTED data preserves 3-D shape."""
    from tscollection.datasets.utils.scaling import create_data_scaler

    scaler_fn = create_data_scaler(
        scale=True,
        scaling_range=(0.0, 1.0),
        scaling_method=ScalingMethod.MINMAX,
        data_form=DataForm.NESTED,
    )
    # Shape: (samples=2, timesteps=3, features=4)
    train = np.random.rand(2, 3, 4)
    valid = np.random.rand(2, 3, 4)
    test = np.random.rand(2, 3, 4)

    scaled_train, scaled_valid, scaled_test = scaler_fn(
        train_data=train, valid_data=valid, test_data=test
    )

    assert scaled_train.shape == train.shape
    assert scaled_valid.shape == valid.shape
    assert scaled_test.shape == test.shape


# --------------------------------------------------------------------------- #
# create_data_scaler — MULTI_FILES data form                                   #
# --------------------------------------------------------------------------- #


def test_create_data_scaler_multi_files() -> None:
    """UTI-02: create_data_scaler with MULTI_FILES scales list of 1-D arrays."""
    from tscollection.datasets.utils.scaling import create_data_scaler

    scaler_fn = create_data_scaler(
        scale=True,
        scaling_range=(0.0, 1.0),
        scaling_method=ScalingMethod.MINMAX,
        data_form=DataForm.MULTI_FILES,
    )
    train = [np.array([0.0, 5.0, 10.0]), np.array([2.0, 8.0])]
    test = [np.array([1.0, 9.0])]

    scaled_train, scaled_valid, scaled_test = scaler_fn(
        train_data=train, valid_data=None, test_data=test
    )

    assert len(scaled_train) == len(train)
    assert len(scaled_test) == len(test)


# --------------------------------------------------------------------------- #
# create_data_scaler — scale=False                                             #
# --------------------------------------------------------------------------- #


def test_create_data_scaler_no_scale() -> None:
    """UTI-02: create_data_scaler with scale=False returns data unchanged."""
    from tscollection.datasets.utils.scaling import create_data_scaler

    scaler_fn = create_data_scaler(
        scale=False,
        scaling_range=(0.0, 1.0),
        scaling_method=ScalingMethod.MINMAX,
        data_form=DataForm.REGULAR,
    )
    train = np.array([1.0, 2.0, 3.0])
    valid = np.array([4.0, 5.0])
    test = np.array([6.0])

    result_train, result_valid, result_test = scaler_fn(
        train_data=train, valid_data=valid, test_data=test
    )

    np.testing.assert_array_equal(result_train, train)
    np.testing.assert_array_equal(result_valid, valid)
    np.testing.assert_array_equal(result_test, test)


# --------------------------------------------------------------------------- #
# _get_scaler enum comparison tests                                            #
# --------------------------------------------------------------------------- #


def test_get_scaler_minmax_enum() -> None:
    """UTI-02: _get_scaler accepts ScalingMethod.MINMAX enum member."""
    from tscollection.datasets.utils.scaling import _get_scaler
    from sklearn.preprocessing import MinMaxScaler

    scaler = _get_scaler(
        scaling_method=ScalingMethod.MINMAX, scaling_range=(0.0, 1.0)
    )
    assert isinstance(scaler, MinMaxScaler)


def test_get_scaler_standard_enum() -> None:
    """UTI-02: _get_scaler accepts ScalingMethod.STANDARD enum member."""
    from tscollection.datasets.utils.scaling import _get_scaler
    from sklearn.preprocessing import StandardScaler

    scaler = _get_scaler(
        scaling_method=ScalingMethod.STANDARD, scaling_range=(0.0, 1.0)
    )
    assert isinstance(scaler, StandardScaler)


def test_get_scaler_invalid_raises() -> None:
    """UTI-02: _get_scaler raises ValueError for unknown method."""
    from tscollection.datasets.utils.scaling import _get_scaler

    with pytest.raises(ValueError, match='Unsupported scaling method'):
        _get_scaler(scaling_method=ScalingMethod.NONE, scaling_range=(0.0, 1.0))


# --------------------------------------------------------------------------- #
# __all__ export test                                                           #
# --------------------------------------------------------------------------- #


def test_all_exports() -> None:
    """UTI-02: __all__ exports only create_data_scaler (not private helpers)."""
    import tscollection.datasets.utils.scaling as scaling_mod

    assert scaling_mod.__all__ == ['create_data_scaler']

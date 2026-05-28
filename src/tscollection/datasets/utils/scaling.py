"""Data scaling strategies for time series datasets."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tscollection.datasets.enums.data import DataForm, ScalingMethod
from tscollection.datasets.utils.common import flatten_list_of_np_arrays

__all__ = ['create_data_scaler']


def create_data_scaler(
    *,
    scale: bool,
    scaling_range: tuple[float, float],
    scaling_method: ScalingMethod = ScalingMethod.MINMAX,
    data_form: DataForm = DataForm.REGULAR,
) -> Callable:
    """Create a data scaling function.

    Returns a callable that, when invoked with (train, valid, test) data,
    fits a scaler on train and transforms all splits.

    Args:
        scale: Whether to apply scaling at all.
        scaling_range: Target (min, max) for MinMaxScaler.
        scaling_method: Scaling algorithm to use.
        data_form: Shape category of the data.

    Returns:
        A callable that accepts (train_data, valid_data, test_data) and
        returns scaled versions of the same.
    """

    def scale_data(train_data: Any, valid_data: Any, test_data: Any) -> tuple[Any, Any, Any]:
        if not scale:
            return train_data, valid_data, test_data

        if data_form == DataForm.REGULAR:
            return _scale_regular_data_and_return_same_type(
                train_data=train_data,
                valid_data=valid_data,
                test_data=test_data,
                scaling_method=scaling_method,
                scaling_range=scaling_range,
            )
        if data_form == DataForm.MULTI_FILES:
            return _scale_multi_file_data(
                train_data=train_data,
                valid_data=valid_data,
                test_data=test_data,
                scaling_method=scaling_method,
                scaling_range=scaling_range,
            )
        if data_form == DataForm.NESTED:
            return _scale_nested_data_all_dimensions(
                train_data=train_data,
                valid_data=valid_data,
                test_data=test_data,
                scaling_method=scaling_method,
                scaling_range=scaling_range,
            )

        msg = f'Unsupported data form: {data_form}'
        raise ValueError(msg)

    return scale_data


def _get_scaler(
    scaling_method: ScalingMethod, scaling_range: tuple[float, float]
) -> MinMaxScaler | StandardScaler:
    """Instantiate the appropriate scikit-learn scaler.

    Args:
        scaling_method: Scaling algorithm identifier.
        scaling_range: Target range for MinMaxScaler.

    Returns:
        A scaler instance ready for fitting.
    """
    if scaling_method == ScalingMethod.MINMAX:
        return MinMaxScaler(feature_range=scaling_range)
    if scaling_method == ScalingMethod.STANDARD:
        return StandardScaler()
    msg = f'Unsupported scaling method: {scaling_method}'
    raise ValueError(msg)


def _scale_regular_data(
    train_data: np.ndarray | pd.DataFrame,
    valid_data: np.ndarray | pd.DataFrame | None,
    test_data: np.ndarray | pd.DataFrame,
    scaling_method: ScalingMethod,
    scaling_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Fit scaler on train data and transform all splits.

    Args:
        train_data: Training data.
        valid_data: Validation data (may be None).
        test_data: Test data.
        scaling_method: Scaling algorithm to use.
        scaling_range: Target range for min-max scaling.

    Returns:
        Scaled (train, valid, test) tuple as numpy arrays.
    """
    scaler = _get_scaler(scaling_method=scaling_method, scaling_range=scaling_range)
    train_scaled = scaler.fit_transform(train_data)
    valid_scaled = scaler.transform(valid_data) if valid_data is not None else None
    test_scaled = scaler.transform(test_data)
    return train_scaled, valid_scaled, test_scaled


def _scale_regular_data_and_return_same_type(
    train_data: np.ndarray | pd.DataFrame,
    valid_data: np.ndarray | pd.DataFrame | None,
    test_data: np.ndarray | pd.DataFrame,
    scaling_method: ScalingMethod,
    scaling_range: tuple[float, float],
) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.DataFrame | None, np.ndarray | pd.DataFrame]:
    """Scale regular data and preserve original container type.

    Args:
        train_data: Training data.
        valid_data: Validation data (may be None).
        test_data: Test data.
        scaling_method: Scaling algorithm to use.
        scaling_range: Target range for min-max scaling.

    Returns:
        Scaled data in the same container type as input.
    """
    scaled_train, scaled_valid, scaled_test = _scale_regular_data(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        scaling_method=scaling_method,
        scaling_range=scaling_range,
    )

    if isinstance(train_data, pd.DataFrame):
        scaled_train = pd.DataFrame(scaled_train, columns=train_data.columns)
        if valid_data is not None and isinstance(valid_data, pd.DataFrame):
            scaled_valid = pd.DataFrame(scaled_valid, columns=valid_data.columns)
        if isinstance(test_data, pd.DataFrame):
            scaled_test = pd.DataFrame(scaled_test, columns=test_data.columns)
        else:
            scaled_test = pd.DataFrame(scaled_test)

    return scaled_train, scaled_valid, scaled_test


def _scale_multi_file_data(
    train_data: list[np.ndarray] | list[pd.DataFrame],
    valid_data: list[np.ndarray] | list[pd.DataFrame] | None,
    test_data: list[np.ndarray] | list[pd.DataFrame],
    scaling_method: ScalingMethod,
    scaling_range: tuple[float, float],
) -> tuple[list[np.ndarray], list[np.ndarray] | None, list[np.ndarray]]:
    """Scale a list of 1-D arrays using a single global scaler.

    The scaler is fit on the concatenated training data and applied
    element-wise to each array.

    Args:
        train_data: List of training arrays.
        valid_data: List of validation arrays (may be None).
        test_data: List of test arrays.
        scaling_method: Scaling algorithm to use.
        scaling_range: Target range for min-max scaling.

    Returns:
        Scaled (train, valid, test) lists of arrays.
    """
    train_arrays = [x.values if isinstance(x, pd.DataFrame) else x for x in train_data]
    test_arrays = [x.values if isinstance(x, pd.DataFrame) else x for x in test_data]

    if valid_data is not None:
        valid_arrays = [x.values if isinstance(x, pd.DataFrame) else x for x in valid_data]
    else:
        valid_arrays = None

    combined = flatten_list_of_np_arrays(train_arrays)
    scaler = _get_scaler(scaling_method=scaling_method, scaling_range=scaling_range)
    scaler.fit_transform(combined.reshape(-1, 1))

    scaled_train = [scaler.transform(x.reshape(-1, 1)).ravel() for x in train_arrays]
    scaled_test = [scaler.transform(x.reshape(-1, 1)).ravel() for x in test_arrays]

    if valid_arrays is not None:
        scaled_valid = [scaler.transform(x.reshape(-1, 1)).ravel() for x in valid_arrays]
    else:
        scaled_valid = None

    return scaled_train, scaled_valid, scaled_test


def _scale_nested_data_all_dimensions(
    train_data: np.ndarray,
    valid_data: np.ndarray,
    test_data: np.ndarray,
    scaling_method: ScalingMethod,
    scaling_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Scale 3-D nested data across all feature dimensions.

    Fits scaler on the last axis of the training data.

    Args:
        train_data: Training data shape (samples, timesteps, features).
        valid_data: Validation data (may be None).
        test_data: Test data shape (samples, timesteps, features).
        scaling_method: Scaling algorithm to use.
        scaling_range: Target range for min-max scaling.

    Returns:
        Scaled (train, valid, test) tuple preserving original shape.
    """
    scaler = _get_scaler(scaling_method=scaling_method, scaling_range=scaling_range)
    orig_shape = train_data.shape
    scaler.fit(train_data.reshape(-1, orig_shape[-1]))

    scaled_train = scaler.transform(train_data.reshape(-1, orig_shape[-1])).reshape(
        train_data.shape
    )

    if valid_data is not None:
        valid_shape = valid_data.shape
        scaled_valid = scaler.transform(valid_data.reshape(-1, valid_shape[-1])).reshape(
            valid_data.shape
        )
    else:
        scaled_valid = None

    test_shape = test_data.shape
    scaled_test = scaler.transform(test_data.reshape(-1, test_shape[-1])).reshape(test_data.shape)

    return scaled_train, scaled_valid, scaled_test

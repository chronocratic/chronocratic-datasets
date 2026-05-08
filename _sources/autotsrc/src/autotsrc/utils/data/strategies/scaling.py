__all__ = ['create_data_scaler']

from collections.abc import Callable
from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.autotsrc.utils import flatten_list_of_np_arrays
from src.autotsrc.utils.decorators.validation import validate_scaling_range


class DataFormEnum(Enum):
    """Enum for the form of the data."""

    REGULAR = 'regular'
    NESTED = 'nested'
    MULTI_FILES = 'multi_files'


@validate_scaling_range
def create_data_scaler(
    *,
    scale: bool,
    scaling_range: tuple[float, float],
    scaling_method: str = 'min_max',
    data_form: str = DataFormEnum.REGULAR.value,
) -> Callable:
    """
    Create a data scaler function.

    :param scale: if True, scale data
    :param scaling_range: the range to scale to
    :param scaling_method: the scaling method to use.
    :param data_form: the form of the data
    :return: a data scaler function
    """
    if scaling_method not in {'min_max', 'standardization'}:
        exception = ValueError(f'Unsupported scaling method: {scaling_method}')
        raise exception

    def scale_data(
        train_data: np.ndarray | pd.DataFrame | list[np.ndarray] | list[pd.DataFrame],
        valid_data: np.ndarray | pd.DataFrame | list[np.ndarray] | list[pd.DataFrame] | None,
        test_data: np.ndarray | pd.DataFrame | list[np.ndarray] | list[pd.DataFrame],
    ) -> tuple[Any, Any, Any]:
        """
        Scale data.

        :param train_data: the training data
        :param valid_data: the validation data
        :param test_data: the test data
        :return: a function that scales data
        """
        if scale:
            if data_form == DataFormEnum.REGULAR.value:
                return _scale_regular_data_and_return_the_same_input_type(
                    train_data=train_data
                    if isinstance(train_data, pd.DataFrame)
                    else np.array(train_data),
                    valid_data=valid_data
                    if isinstance(valid_data, pd.DataFrame)
                    else (np.array(valid_data) if valid_data is not None else None),
                    test_data=test_data
                    if isinstance(test_data, pd.DataFrame)
                    else np.array(test_data),
                    scaling_method=scaling_method,
                    scaling_range=scaling_range,
                )
            if data_form == DataFormEnum.MULTI_FILES.value:
                return _scale_multi_file_data(
                    train_data=cast(
                        'list[np.ndarray] | list[pd.DataFrame]',
                        train_data if isinstance(train_data, list) else [train_data],
                    ),
                    valid_data=cast(
                        'list[np.ndarray] | list[pd.DataFrame] | None',
                        valid_data
                        if isinstance(valid_data, list)
                        else ([valid_data] if valid_data is not None else None),
                    ),
                    test_data=cast(
                        'list[np.ndarray] | list[pd.DataFrame]',
                        test_data if isinstance(test_data, list) else [test_data],
                    ),
                    scaling_method=scaling_method,
                    scaling_range=scaling_range,
                )

            if data_form == DataFormEnum.NESTED.value:
                return _scale_nested_data_all_dimensions(
                    train_data=train_data
                    if isinstance(train_data, np.ndarray)
                    else np.array(train_data),
                    valid_data=valid_data
                    if isinstance(valid_data, np.ndarray)
                    else (np.array(valid_data) if valid_data is not None else None),
                    test_data=test_data
                    if isinstance(test_data, np.ndarray)
                    else np.array(test_data),
                    scaling_method=scaling_method,
                    scaling_range=scaling_range,
                )
            exception = ValueError(f'Unsupported data form: {data_form}')
            raise exception

        return train_data, valid_data, test_data

    return scale_data


def _get_scaler(
    scaling_method: str, scaling_range: tuple[float, float]
) -> MinMaxScaler | StandardScaler:
    """
    Get the appropriate scaler based on the scaling method.

    :param scaling_method: the method of scaling ('min_max' or 'standardization')
    :param scaling_range: the range to normalize to (used only for MinMaxScaler)
    :return: the scaler
    """
    if scaling_method == 'min_max':
        scaling_range_int = (int(scaling_range[0]), int(scaling_range[1]))
        return MinMaxScaler(feature_range=scaling_range_int)
    if scaling_method == 'standardization':
        return StandardScaler()
    exception = ValueError(f'Unsupported scaling method: {scaling_method}')
    raise exception


def _scale_regular_data(
    train_data: np.ndarray | pd.DataFrame,
    valid_data: np.ndarray | pd.DataFrame | None,
    test_data: np.ndarray | pd.DataFrame,
    scaling_method: str,
    scaling_range: tuple[float, float],
) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.DataFrame | None, np.ndarray | pd.DataFrame]:
    """
    Scale data.

    :param train_data: the training data
    :param valid_data: the validation data
    :param test_data: the test data
    :return: scaled data
    """
    scaler = _get_scaler(scaling_method=scaling_method, scaling_range=scaling_range)
    train_data = scaler.fit_transform(train_data)
    if valid_data is not None:
        valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)
    return train_data, valid_data, test_data


def _scale_regular_data_and_return_the_same_input_type(
    train_data: np.ndarray | pd.DataFrame,
    valid_data: np.ndarray | pd.DataFrame | None,
    test_data: np.ndarray | pd.DataFrame,
    scaling_method: str,
    scaling_range: tuple[float, float],
) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.DataFrame | None, np.ndarray | pd.DataFrame]:
    """
    Scale data.

    :param train_data: the training data
    :param valid_data: the validation data
    :param test_data: the test data
    :return: scaled data
    """
    scaled_train_data, scaled_valid_data, scaled_test_data = _scale_regular_data(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        scaling_method=scaling_method,
        scaling_range=scaling_range,
    )
    if isinstance(train_data, pd.DataFrame):
        scaled_train_data = pd.DataFrame(scaled_train_data, columns=train_data.columns)
        if valid_data is not None and isinstance(valid_data, pd.DataFrame):
            scaled_valid_data = pd.DataFrame(scaled_valid_data, columns=valid_data.columns)
        if test_data is not None and isinstance(test_data, pd.DataFrame):
            scaled_test_data = pd.DataFrame(scaled_test_data, columns=test_data.columns)

    return scaled_train_data, scaled_valid_data, scaled_test_data


def _scale_multi_file_data(
    train_data: list[np.ndarray] | list[pd.DataFrame],
    valid_data: list[np.ndarray] | list[pd.DataFrame] | None,
    test_data: list[np.ndarray] | list[pd.DataFrame],
    scaling_method: str,
    scaling_range: tuple[float, float],
) -> tuple[list[np.ndarray], list[np.ndarray] | None, list[np.ndarray]]:
    """
    Scale a list of per-file arrays by fitting on the concatenated training data.

    :param train_data: list of training arrays or DataFrames (one per file)
    :param valid_data: list of validation arrays or DataFrames, or None
    :param test_data: list of test arrays or DataFrames (one per file)
    :param scaling_method: the scaling method to use ('min_max' or 'standardization')
    :param scaling_range: the target range for min-max scaling
    :return: tuple of (scaled_train, scaled_valid, scaled_test) as lists of 1-D arrays
    """
    train_data_array_list = [x.to_numpy() if isinstance(x, pd.DataFrame) else x for x in train_data]
    test_data_array_list = [x.to_numpy() if isinstance(x, pd.DataFrame) else x for x in test_data]

    if valid_data is not None:
        valid_data_array_list = [
            x.to_numpy() if isinstance(x, pd.DataFrame) else x for x in valid_data
        ]
    else:
        valid_data_array_list = None

    train_data_combined = flatten_list_of_np_arrays(train_data_array_list)
    scaler = _get_scaler(scaling_method=scaling_method, scaling_range=scaling_range)
    scaler.fit(train_data_combined.reshape(-1, 1))

    scaled_train_data = [scaler.transform(x.reshape(-1, 1)).ravel() for x in train_data_array_list]
    scaled_test_data = [scaler.transform(x.reshape(-1, 1)).ravel() for x in test_data_array_list]

    if valid_data_array_list:
        scaled_valid_data = [
            scaler.transform(x.reshape(-1, 1)).ravel() for x in valid_data_array_list
        ]
    else:
        scaled_valid_data = None

    return scaled_train_data, scaled_valid_data, scaled_test_data


def _scale_nested_data_all_dimensions(
    train_data: np.ndarray,
    valid_data: np.ndarray | None,
    test_data: np.ndarray,
    scaling_method: str,
    scaling_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Scale 3-D nested arrays by fitting on the flattened training data across all dimensions.

    :param train_data: training array of shape (samples, time_steps, features)
    :param valid_data: validation array of the same shape, or None
    :param test_data: test array of the same shape
    :param scaling_method: the scaling method to use ('min_max' or 'standardization')
    :param scaling_range: the target range for min-max scaling
    :return: tuple of (scaled_train, scaled_valid, scaled_test) preserving original shapes
    """
    # prepare the scaler
    scaler = _get_scaler(scaling_method=scaling_method, scaling_range=scaling_range)

    scaler.fit(train_data.reshape(-1, train_data.shape[-1]))

    scaled_train_data = scaler.transform(train_data.reshape(-1, train_data.shape[-1])).reshape(
        train_data.shape
    )

    if valid_data is not None:
        scaled_valid_data = scaler.transform(valid_data.reshape(-1, valid_data.shape[-1])).reshape(
            valid_data.shape
        )
    else:
        scaled_valid_data = None

    scaled_test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(
        test_data.shape
    )

    return scaled_train_data, scaled_valid_data, scaled_test_data

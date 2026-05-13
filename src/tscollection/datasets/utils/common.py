"""Common utility functions for time series data processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    'FunctionComposer',
    'compose',
    'flatten_list_of_np_arrays',
    'get_num_samples_from_ts',
]


def get_num_samples_from_ts(ts: np.ndarray | list[np.ndarray]) -> int:
    """Get number of samples from a time series.

    Args:
        ts: A time series array or list of arrays.

    Returns:
        Number of samples (length) of the time series.
    """
    return len(ts)


class FunctionComposer:
    """Composes a list of callables into a single callable.

    Functions are applied in the order they were provided.
    None values in the list are filtered out.

    Args:
        functions: A list of callables to compose.
    """

    def __init__(self, functions: list[Callable]) -> None:
        self.functions = [f for f in functions if f is not None]

    def __call__(self, data: object) -> object:
        """Apply composed functions to data in order."""
        result = data
        for f in self.functions:
            result = f(result)
        return result


def compose(*functions: Callable) -> Callable:
    """Compose multiple functions into a single callable.

    Functions are applied in the order they are provided.

    Args:
        functions: Callables to compose.

    Returns:
        A callable that applies all functions in order.
    """
    return FunctionComposer(list(functions))


def flatten_list_of_np_arrays(
    list_of_np_arrays: list[np.ndarray],
) -> np.ndarray:
    """Flatten a list of numpy arrays into a single 1-D array.

    Args:
        list_of_np_arrays: A list of numpy arrays.

    Returns:
        A single flattened numpy array.
    """
    return np.concatenate(list_of_np_arrays).ravel()

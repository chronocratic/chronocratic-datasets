"""Common utility functions for time series data processing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ['FunctionComposer', 'compose', 'get_num_samples_from_ts']


def get_num_samples_from_ts(ts: np.ndarray) -> int:
    """Get number of samples from a time series.

    Args:
        ts: A time series array.

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

    def __call__(self, data: Any) -> Any:
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

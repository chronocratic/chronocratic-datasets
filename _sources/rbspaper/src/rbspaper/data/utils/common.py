"""Common data utility functions adapted from autotsaugment."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
import json
import math
from pathlib import Path
import time
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import lightning.pytorch as pl

__all__ = [
    'AccumulatingTimerCallback',
    'cartesian_product_dict',
    'closest_power_of_2',
    'compose',
    'find_project_root',
    'flatten_list',
    'flatten_list_of_np_arrays',
    'get_num_samples_from_ts',
    'load_json',
    'separate_target_feature_from_df',
]


def get_num_samples_from_ts(ts: np.ndarray) -> int:
    """Get number of samples from a time series.

    Args:
        ts: A time series array.

    Returns:
        Number of samples (length) of the time series.
    """
    return len(ts)


def load_json(json_file: Path | str) -> dict[str, Any]:
    """Load a JSON file and return its contents as a dictionary.

    Args:
        json_file: Path to the JSON file.

    Returns:
        Parsed JSON data as a dictionary.
    """
    with open(json_file) as f:
        return json.load(f)


class FunctionComposer:
    """Composes a list of callables into a single callable.

    Functions are applied in the order they were provided.

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


def flatten_list(list_of_lists: list[Any]) -> list[Any]:
    """Recursively flatten a nested list.

    Args:
        list_of_lists: A potentially nested list.

    Returns:
        A flattened list.
    """
    if not list_of_lists:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten_list(list_of_lists[0]) + flatten_list(list_of_lists[1:])
    return list_of_lists[:1] + flatten_list(list_of_lists[1:])


def flatten_list_of_np_arrays(list_of_np_arrays: list[np.ndarray]) -> np.ndarray:
    """Flatten a list of numpy arrays into a single 1-D array.

    Args:
        list_of_np_arrays: A list of numpy arrays.

    Returns:
        A single flattened numpy array.
    """
    return np.concatenate(list_of_np_arrays).ravel()


def separate_target_feature_from_df(
    df: pd.DataFrame, target_feature_name: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target feature from a DataFrame.

    Args:
        df: Source DataFrame containing the target column.
        target_feature_name: Name of the target column.

    Returns:
        A tuple of (features DataFrame, target Series).
    """
    target_feature = df[target_feature_name]
    features = df.drop(target_feature_name, axis=1)
    return features, target_feature


def find_project_root(current_path: Path, marker: str = '.project_root_marker') -> Path:
    """Recursively locate project root by searching for a marker file.

    Args:
        current_path: Starting directory.
        marker: Name of the marker file to search for.

    Returns:
        Path to the project root directory.

    Raises:
        FileNotFoundError: If the marker is not found up to filesystem root.
    """
    if (current_path / marker).exists():
        return current_path
    if current_path.parent != current_path:
        return find_project_root(current_path.parent, marker)
    raise FileNotFoundError(f"Project root marker '{marker}' not found.")


def closest_power_of_2(m: float, v: float) -> int:
    """Find the closest power of 2 to the division of m by v.

    Args:
        m: The dividend.
        v: The divisor.

    Returns:
        The nearest power of 2 to m / v.

    Raises:
        ZeroDivisionError: If v is zero.
    """
    if v == 0:
        raise ZeroDivisionError('The divisor v should not be zero.')

    div_result = m / v
    log_result = math.log2(div_result)
    lower_power = 2 ** math.floor(log_result)
    upper_power = 2 ** math.ceil(log_result)

    if abs(div_result - lower_power) < abs(upper_power - div_result):
        return int(lower_power)
    return int(upper_power)


def cartesian_product_dict(d: dict[Any, Any]) -> list[dict[Any, Any]]:
    """Generate all combinations of values in a dictionary.

    Each value can be a single value or a list. Returns a list of
    dictionaries, each representing one combination.

    Args:
        d: Input dictionary with scalar or list values.

    Returns:
        List of dictionaries, one per combination.

    Example:
        >>> cartesian_product_dict({"a": [1, 2], "b": "x"})
        [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
    """
    keys = d.keys()
    values = [v if isinstance(v, list) else [v] for v in d.values()]
    return [dict(zip(keys, combo, strict=False)) for combo in product(*values)]


class AccumulatingTimerCallback:
    """Lightning callback that tracks total training time across sessions.

    Records the duration of each training session and reports cumulative time.
    """

    def __init__(self) -> None:
        super().__init__()
        self.start_time: float | None = None
        self.sessions_count: int = 0
        self.sessions_times: dict[int, float] = {}

    def get_dict(self) -> dict[str, Any]:
        """Return timer state as a dictionary.

        Returns:
            Dictionary with session count and per-session times.
        """
        return {'sessions_count': self.sessions_count, 'sessions_times': self.sessions_times}

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.sessions_count += 1
        self.start_time = time.perf_counter()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        import logging

        total_time = sum(self.sessions_times.values())
        logging.info(
            'This model has been trained for %d sessions, with a total time of %.2f seconds.',
            self.sessions_count,
            total_time,
        )
        logging.info('Detailed session times: %s', self.sessions_times)

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        if self.start_time is None:
            return
        current_session_time = time.perf_counter() - self.start_time
        self.sessions_times.update({self.sessions_count: current_session_time})
        checkpoint['timer_callback_state'] = self.get_dict()

    def on_load_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        callback_state = checkpoint.get('timer_callback_state', {})
        self.sessions_count = callback_state.get('sessions_count', 0)
        self.sessions_times = callback_state.get('sessions_times', {})

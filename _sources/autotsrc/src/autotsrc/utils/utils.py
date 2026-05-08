__all__ = [
    'compose',
    'flatten_list_of_np_arrays',
    'get_num_samples_from_ts',
    'load_json',
    'separate_target_feature_from_df',
]

from collections.abc import Callable
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(json_file: Path | str) -> dict:
    """Load a JSON file and return its parsed content."""
    json_path = Path(json_file)
    with json_path.open() as f:
        return json.load(f)


def get_num_samples_from_ts(ts: np.ndarray | pd.DataFrame | list[np.ndarray]) -> int:
    """Return the number of samples in a time-series container."""
    return len(ts)


class FunctionComposer[T]:
    """Compose unary callables and apply them in reverse order."""

    def __init__(self, functions: list[Callable[[T], T]]) -> None:
        """Store non-null functions for later composition."""
        self.functions = [f for f in functions if f is not None]

    def __call__(self, data: T) -> T:
        """Apply stored functions from right to left."""
        result = data
        for f in reversed(self.functions):
            result = f(result)
        return result


def compose[T](*functions: Callable[[T], T]) -> Callable[[T], T]:
    """Compose unary callables and return a single callable."""
    return FunctionComposer(list(functions))


def flatten_list_of_np_arrays(list_of_np_arrays: list[np.ndarray]) -> np.ndarray:
    """Flatten a list of NumPy arrays into a 1D array."""
    return np.concatenate(list_of_np_arrays).ravel()


def separate_target_feature_from_df(
    df: pd.DataFrame, target_feature_name: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature DataFrame and target Series."""
    target_feature = df[target_feature_name]
    df = df.drop(target_feature_name, axis=1)
    return df, target_feature

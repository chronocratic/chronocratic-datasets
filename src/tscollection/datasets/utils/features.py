"""Time feature extraction utilities for forecasting datasets."""

import numpy as np
import pandas as pd

__all__ = ['TIME_FEATURE_COUNT', 'extract_time_features']

TIME_FEATURE_COUNT: int = 7


def extract_time_features(datetime_index: pd.DatetimeIndex) -> np.ndarray:
    """Extract cyclical time features from a DatetimeIndex.

    Produces a 2-D array with columns: minute, hour, dayofweek,
    day, dayofyear, month, week.

    Args:
        datetime_index: A pandas DatetimeIndex.

    Returns:
        2-D numpy array of shape (len(index), 7) with dtype float32.
    """
    series = datetime_index.to_series()
    return np.stack(
        [
            series.dt.minute.to_numpy(),
            series.dt.hour.to_numpy(),
            series.dt.dayofweek.to_numpy(),
            series.dt.day.to_numpy(),
            series.dt.dayofyear.to_numpy(),
            series.dt.month.to_numpy(),
            series.dt.isocalendar().week.to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)

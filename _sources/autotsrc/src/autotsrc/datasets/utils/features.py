__all__ = ['extract_time_features']

import numpy as np
import pandas as pd


def extract_time_features(datetime_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Extract various time-related features from a pandas DatetimeIndex.

    Parameters:
    -----------
    datetime_index : pd.DatetimeIndex
        The datetime index from which to extract time features.

    Returns:
    --------
    np.ndarray
        A 2D array with extracted time features, where each row corresponds to a datetime and each
        column represents a specific time feature.
        The features extracted are:
        - Minute of the hour
        - Hour of the day
        - Day of the week
        - Day of the month
        - Day of the year
        - Month of the year
        - Week of the year

    """
    return np.stack(
        [
            datetime_index.to_series().dt.minute.to_numpy(),
            datetime_index.to_series().dt.hour.to_numpy(),
            datetime_index.to_series().dt.dayofweek.to_numpy(),
            datetime_index.to_series().dt.day.to_numpy(),
            datetime_index.to_series().dt.dayofyear.to_numpy(),
            datetime_index.to_series().dt.month.to_numpy(),
            datetime_index.to_series().dt.isocalendar().week.to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)

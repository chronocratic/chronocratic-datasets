"""Tests for time feature extraction utilities.

Verifies that extract_time_features produces correct (N, 7) float32
arrays from pandas DatetimeIndex objects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronocratic.datasets.utils.features import extract_time_features

# --------------------------------------------------------------------------- #
# extract_time_features tests                                                  #
# --------------------------------------------------------------------------- #


def test_extract_time_features_shape() -> None:
    """extract_time_features returns (N, 7) float32 array."""
    dti = pd.date_range('2020-01-01', periods=10, freq='h')
    result = extract_time_features(dti)
    assert result.shape == (10, 7)
    assert result.dtype == np.float32


def test_extract_time_features_minute_hour() -> None:
    """First two columns are minute and hour."""
    dti = pd.date_range('2020-01-01 02:30:00', periods=1, freq='h')
    result = extract_time_features(dti)
    assert result[0, 0] == 30.0  # minute
    assert result[0, 1] == 2.0  # hour


def test_extract_time_features_dayofweek() -> None:
    """Third column is day of week (Monday=0)."""
    # 2020-01-01 is Wednesday
    dti = pd.date_range('2020-01-01', periods=1, freq='D')
    result = extract_time_features(dti)
    assert result[0, 2] == 2.0  # Wednesday = 2


def test_extract_time_features_multiple_dates() -> None:
    """Works with multiple dates at different intervals."""
    dti = pd.date_range('2020-01-01', periods=5, freq='D')
    result = extract_time_features(dti)
    assert result.shape == (5, 7)
    # Day of year should increment
    for i in range(4):
        assert result[i + 1, 4] > result[i, 4]


def test_extract_time_features_empty_index() -> None:
    """Empty DatetimeIndex returns (0, 7) array."""
    dti = pd.DatetimeIndex([])
    result = extract_time_features(dti)
    assert result.shape == (0, 7)
    assert result.dtype == np.float32


def test_extract_time_features_week() -> None:
    """Seventh column is ISO week number."""
    dti = pd.date_range('2020-01-01', periods=1, freq='D')
    result = extract_time_features(dti)
    # 2020-01-01 is in ISO week 1
    assert result[0, 6] == 1.0

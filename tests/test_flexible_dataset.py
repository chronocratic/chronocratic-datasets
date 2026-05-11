"""Tests for flexible (sliding-window) dataset classes (DST-02, DST-04).

Verifies that FlexibleTimeSeriesDataset accepts seq_len and step,
produces sliding-window sequences, and raises IndexError for
out-of-range indices.
"""

import numpy as np
import pytest
import torch

from tscollection.datasets.datasets.classes.strategies import (
    ForecastingStrategySingleFile,
)
from tscollection.datasets.enums import TimeSeriesDatasetMode

# --- Task 2 RED-phase tests (will fail until flexible.py is implemented) ---


def test_flexible_accepts_seq_len_step():
    """DST-04: FlexibleTimeSeriesDatasetSingleFile stores seq_len and step."""
    from tscollection.datasets.datasets.classes.flexible import (
        FlexibleTimeSeriesDatasetSingleFile,
    )

    data = np.random.default_rng().standard_normal((200, 7)).astype(np.float32)
    strategy = ForecastingStrategySingleFile(forecast_horizon=24)
    ds = FlexibleTimeSeriesDatasetSingleFile(
        data=data,
        labels=None,
        seq_len=96,
        step=1,
        mode=TimeSeriesDatasetMode.FORECASTING,
        sequence_handling_strategy=strategy,
        expand_dims_axis=None,
        transformations_sequence=(torch.from_numpy,),
    )
    assert ds._seq_len == 96
    assert ds._step == 1


def test_flexible_yields_windows():
    """DST-02: FlexibleTimeSeriesDatasetSingleFile yields sliding-window pairs."""
    from tscollection.datasets.datasets.classes.flexible import (
        FlexibleTimeSeriesDatasetSingleFile,
    )

    data = np.random.default_rng().standard_normal((200, 7)).astype(np.float32)
    strategy = ForecastingStrategySingleFile(forecast_horizon=24)
    ds = FlexibleTimeSeriesDatasetSingleFile(
        data=data,
        labels=None,
        seq_len=96,
        step=1,
        mode=TimeSeriesDatasetMode.FORECASTING,
        sequence_handling_strategy=strategy,
        expand_dims_axis=None,
        transformations_sequence=(torch.from_numpy,),
    )
    inp, tgt = ds[0]
    assert inp.shape[0] == 96  # seq_len
    assert tgt.shape[0] == 24  # forecast_horizon


def test_flexible_bounds_check():
    """FlexibleTimeSeriesDatasetSingleFile raises IndexError for out-of-range index."""
    from tscollection.datasets.datasets.classes.flexible import (
        FlexibleTimeSeriesDatasetSingleFile,
    )

    data = np.random.default_rng().standard_normal((200, 7)).astype(np.float32)
    strategy = ForecastingStrategySingleFile(forecast_horizon=24)
    ds = FlexibleTimeSeriesDatasetSingleFile(
        data=data,
        labels=None,
        seq_len=96,
        step=1,
        mode=TimeSeriesDatasetMode.FORECASTING,
        sequence_handling_strategy=strategy,
        expand_dims_axis=None,
        transformations_sequence=(torch.from_numpy,),
    )
    with pytest.raises(IndexError):
        _ = ds[len(ds)]

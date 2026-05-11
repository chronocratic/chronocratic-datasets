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


def test_flexible_multifile_boundary_indices():
    """FlexibleTimeSeriesDatasetMultipleFiles maps boundary indices correctly.

    Verify that global indices at file boundaries return data from the
    correct file, not the adjacent one (regression test for CR-04).
    """
    from tscollection.datasets.datasets.classes.flexible import (
        FlexibleTimeSeriesDatasetMultipleFiles,
    )
    from tscollection.datasets.datasets.classes.strategies import (
        ClassificationStrategyMultipleFiles,
    )

    # File 0: 100 samples, seq_len=50, step=10 -> 6 windows (indices 0..5)
    # File 1: 200 samples, seq_len=50, step=10 -> 16 windows (indices 6..21)
    data_list = [
        np.arange(100.0).astype(np.float32),
        np.arange(200.0).astype(np.float32) + 1000,  # offset to distinguish files
    ]
    strategy = ClassificationStrategyMultipleFiles()
    ds = FlexibleTimeSeriesDatasetMultipleFiles(
        data=data_list,
        labels=None,
        seq_len=50,
        step=10,
        mode=TimeSeriesDatasetMode.WITHOUT_LABELS,
        sequence_handling_strategy=strategy,
        expand_dims_axis=None,
        transformations_sequence=(torch.from_numpy,),
    )

    # Total: 6 + 16 = 22 sequences
    assert len(ds) == 22

    # Index 0: first window of file 0 -> data[0:50] of file 0 (values 0..49)
    sample = ds[0]
    np.testing.assert_array_equal(sample.numpy(), np.arange(50.0))

    # Index 5: last window of file 0 -> data[50:100] of file 0 (values 50..99)
    sample = ds[5]
    np.testing.assert_array_equal(sample.numpy(), np.arange(50.0, 100.0))

    # Index 6: first window of file 1 -> data[0:50] of file 1 (values 1000..1049)
    sample = ds[6]
    np.testing.assert_array_equal(sample.numpy(), np.arange(1000.0, 1050.0))

    # Index 21: last window of file 1 -> data[150:200] of file 1 (values 1150..1199)
    sample = ds[21]
    np.testing.assert_array_equal(sample.numpy(), np.arange(1150.0, 1200.0))

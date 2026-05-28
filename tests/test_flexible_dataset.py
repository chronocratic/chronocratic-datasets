"""Tests for flexible (sliding-window) dataset classes.

Verifies that FlexibleTimeSeriesDataset accepts seq_len and step,
produces sliding-window sequences, and raises IndexError for
out-of-range indices.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tscollection.datasets.datatypes._base.strategies import ForecastingStrategySingleFile
from tscollection.datasets.enums import TimeSeriesDatasetMode

# --- Task 2 RED-phase tests (will fail until flexible.py is implemented) ---


def test_flexible_accepts_seq_len_step():
    """FlexibleTimeSeriesDatasetSingleFile stores seq_len and step."""
    from tscollection.datasets.datatypes._base.flexible import FlexibleTimeSeriesDatasetSingleFile

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
    """FlexibleTimeSeriesDatasetSingleFile yields sliding-window pairs."""
    from tscollection.datasets.datatypes._base.flexible import FlexibleTimeSeriesDatasetSingleFile

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
    from tscollection.datasets.datatypes._base.flexible import FlexibleTimeSeriesDatasetSingleFile

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
    correct file, not the adjacent one.
    """
    from tscollection.datasets.datatypes._base.flexible import (
        FlexibleTimeSeriesDatasetMultipleFiles,
    )
    from tscollection.datasets.datatypes._base.strategies import ClassificationStrategyMultipleFiles

    # File 0: 100 samples, seq_len=50, step=10 -> 6 windows (global idx 0..5)
    # File 1: 200 samples, seq_len=50, step=10 -> 16 windows (global idx 6..21)
    # The dataset uses _n sequentially (step=1) for data access:
    # ds[i] -> file_data[i : i + seq_len] for file 0
    # accumulated boundaries: [6, 22]
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

    # Index 0: first window of file 0, _n=0 -> data[0:50] (values 0..49)
    sample = ds[0]
    np.testing.assert_array_equal(sample.numpy(), np.arange(50.0))

    # Index 5: last window of file 0, _n=5 -> data[5:55] (values 5..54)
    sample = ds[5]
    np.testing.assert_array_equal(sample.numpy(), np.arange(5.0, 55.0))

    # Index 6: first window of file 1, _n=0 -> data[0:50] (values 1000..1049)
    # CRITICAL: idx=6 must map to file 1 (bisect([6,22], 6) = 1), NOT file 0
    sample = ds[6]
    np.testing.assert_array_equal(sample.numpy(), np.arange(1000.0, 1050.0))

    # Index 21: last window of file 1, _n=15 -> data[15:65] (values 1015..1064)
    sample = ds[21]
    np.testing.assert_array_equal(sample.numpy(), np.arange(1015.0, 1065.0))

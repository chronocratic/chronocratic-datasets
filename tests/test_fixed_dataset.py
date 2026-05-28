"""Tests for fixed-length dataset classes.

Verifies that TimeSeriesDataset dispatches correctly by mode,
FixedTimeSeriesDataset exposes seq_len as a read-only property,
and univariate/multivariate subclasses return data in the expected format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tscollection.datasets.enums import TimeSeriesDatasetMode

# --- Task 1 RED-phase tests (will fail until fixed.py is implemented) ---


def test_fixed_yields_data_label():
    """FixedTimeSeriesDatasetUnivariate yields (torch.Tensor, int) in WITH_LABELS mode."""
    from tscollection.datasets.datatypes._base.fixed import FixedTimeSeriesDatasetUnivariate

    data = pd.DataFrame(np.random.default_rng().standard_normal((10, 50)).astype(np.float32))
    labels = pd.Series([0, 1] * 5)
    ds = FixedTimeSeriesDatasetUnivariate(
        data=data,
        labels=labels,
        mode=TimeSeriesDatasetMode.WITH_LABELS,
        expand_dims_axis=1,
        transformations_sequence=(torch.from_numpy,),
    )
    sample, label = ds[0]
    assert isinstance(sample, (np.ndarray, torch.Tensor))
    assert isinstance(label, (int, np.integer))


def test_fixed_seq_len_property():
    """FixedTimeSeriesDataset.seq_len returns int from data shape (read-only)."""
    from tscollection.datasets.datatypes._base.fixed import FixedTimeSeriesDatasetUnivariate

    data = pd.DataFrame(np.random.default_rng().standard_normal((10, 50)).astype(np.float32))
    labels = pd.Series([0, 1] * 5)
    ds = FixedTimeSeriesDatasetUnivariate(
        data=data,
        labels=labels,
        mode=TimeSeriesDatasetMode.WITH_LABELS,
        expand_dims_axis=1,
        transformations_sequence=(torch.from_numpy,),
    )
    assert ds.seq_len == 50
    # Verify read-only (no setter)
    with pytest.raises(AttributeError):
        ds.seq_len = 10  # type: ignore[assignment]


def test_fixed_length():
    """FixedTimeSeriesDataset.__len__ returns number of samples."""
    from tscollection.datasets.datatypes._base.fixed import FixedTimeSeriesDatasetUnivariate

    data = pd.DataFrame(np.random.default_rng().standard_normal((10, 50)).astype(np.float32))
    labels = pd.Series([0, 1] * 5)
    ds = FixedTimeSeriesDatasetUnivariate(
        data=data,
        labels=labels,
        mode=TimeSeriesDatasetMode.WITH_LABELS,
        expand_dims_axis=1,
        transformations_sequence=(torch.from_numpy,),
    )
    assert len(ds) == 10


def test_multivariate_get_current_data():
    """FixedTimeSeriesDatasetMultivariate._get_current_data returns 3D slice."""
    from tscollection.datasets.datatypes._base.fixed import FixedTimeSeriesDatasetMultivariate

    # 5 samples, 30 timesteps, 4 features
    data = np.random.default_rng().standard_normal((5, 30, 4)).astype(np.float32)
    labels = pd.Series([0, 1, 0, 1, 0])
    ds = FixedTimeSeriesDatasetMultivariate(
        data=data,
        labels=labels,
        mode=TimeSeriesDatasetMode.WITH_LABELS,
        expand_dims_axis=None,
        transformations_sequence=(torch.from_numpy,),
    )
    sample, label = ds[2]
    assert sample.shape == (30, 4)
    assert label == 0
    # Verify seq_len for ndarray
    assert ds.seq_len == 30

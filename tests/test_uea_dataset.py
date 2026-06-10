"""Tests for UEA multivariate classification wrapper.

Verifies that UEAClassificationMultivariateDataset correctly inherits from
FixedTimeSeriesDatasetMultivariate, applies default transforms, and yields
(data, label) tuples in WITH_LABELS mode with proper 3D shapes.
"""

import numpy as np
import pandas as pd
import torch

from chronocratic.datasets.enums import TimeSeriesDatasetMode


def test_uea_yields_data_label(synthetic_multivariate_data):
    """UEAClassificationMultivariateDataset yields (Tensor, int) in WITH_LABELS mode."""
    from chronocratic.datasets.datatypes.uea import UEAClassificationMultivariateDataset

    labels = pd.Series([0, 1, 0, 1, 0])
    ds = UEAClassificationMultivariateDataset(
        data=synthetic_multivariate_data, labels=labels, mode=TimeSeriesDatasetMode.SAMPLE_LABEL
    )
    sample, label = ds[0]

    # No expand_dims_axis by default, so shape is (timesteps, features)
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (30, 4)
    assert isinstance(label, (int, np.integer))


def test_uea_without_labels(synthetic_multivariate_data):
    """UEA yields single tensor in WITHOUT_LABELS mode."""
    from chronocratic.datasets.datatypes.uea import UEAClassificationMultivariateDataset

    ds = UEAClassificationMultivariateDataset(
        data=synthetic_multivariate_data, labels=None, mode=TimeSeriesDatasetMode.SAMPLE_ONLY
    )
    result = ds[0]
    assert not isinstance(result, tuple)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (30, 4)


def test_uea_length(synthetic_multivariate_data):
    """UEA dataset length equals number of samples in 3D array."""
    from chronocratic.datasets.datatypes.uea import UEAClassificationMultivariateDataset

    labels = pd.Series([0, 1, 0, 1, 0])
    ds = UEAClassificationMultivariateDataset(
        data=synthetic_multivariate_data, labels=labels, mode=TimeSeriesDatasetMode.SAMPLE_LABEL
    )
    assert len(ds) == 5


def test_uea_no_expand_dims_by_default(synthetic_multivariate_data):
    """UEA defaults to expand_dims_axis=None, preserving 2D sample shape."""
    from chronocratic.datasets.datatypes.uea import UEAClassificationMultivariateDataset

    labels = pd.Series([0, 1, 0, 1, 0])
    ds = UEAClassificationMultivariateDataset(
        data=synthetic_multivariate_data, labels=labels, mode=TimeSeriesDatasetMode.SAMPLE_LABEL
    )
    sample, _ = ds[0]
    # Shape should be (timesteps, features) = (30, 4), not expanded
    assert sample.ndim == 2
    assert sample.shape == (30, 4)

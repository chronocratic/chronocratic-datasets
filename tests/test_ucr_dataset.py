"""Tests for UCR univariate classification wrapper.

Verifies that UCRClassificationUnivariateDataset correctly inherits from
FixedTimeSeriesDatasetUnivariate, applies default transforms, and yields
(data, label) tuples in WITH_LABELS mode.
"""

from __future__ import annotations

import numpy as np
import torch

from tscollection.datasets.enums import TimeSeriesDatasetMode


def test_ucr_yields_data_label(synthetic_classification_df, synthetic_classification_labels):
    """UCRClassificationUnivariateDataset yields (Tensor, int) in WITH_LABELS mode."""
    from tscollection.datasets.datatypes.ucr import UCRClassificationUnivariateDataset

    ds = UCRClassificationUnivariateDataset(
        data=synthetic_classification_df,
        labels=synthetic_classification_labels,
        mode=TimeSeriesDatasetMode.WITH_LABELS,
    )
    sample, label = ds[0]

    # expand_dims_axis=1 on a (50,) array produces (50, 1).
    # Note: expand_data_dimensionality converts tensor back to numpy
    # (Pitfall 3 in research docs).
    assert isinstance(sample, (np.ndarray, torch.Tensor))
    assert sample.shape[0] == 50  # original timestep count
    assert sample.shape[1] == 1  # expanded dimension at axis=1
    assert isinstance(label, (int, np.integer))


def test_ucr_without_labels(synthetic_classification_df):
    """UCR yields single array in WITHOUT_LABELS mode."""
    from tscollection.datasets.datatypes.ucr import UCRClassificationUnivariateDataset

    ds = UCRClassificationUnivariateDataset(
        data=synthetic_classification_df, labels=None, mode=TimeSeriesDatasetMode.WITHOUT_LABELS
    )
    result = ds[0]
    assert not isinstance(result, tuple)


def test_ucr_length(synthetic_classification_df, synthetic_classification_labels):
    """UCR dataset length equals number of rows in DataFrame."""
    from tscollection.datasets.datatypes.ucr import UCRClassificationUnivariateDataset

    ds = UCRClassificationUnivariateDataset(
        data=synthetic_classification_df,
        labels=synthetic_classification_labels,
        mode=TimeSeriesDatasetMode.WITH_LABELS,
    )
    assert len(ds) == len(synthetic_classification_df)

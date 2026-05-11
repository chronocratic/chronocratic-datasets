"""Tests for conftest.py synthetic fixtures.

Validates that all fixture shapes and dtypes match the expected
values for downstream dataset tests.
"""

import numpy as np
import pandas as pd


def test_classification_df_shape(synthetic_classification_df):
    """Verify classification DataFrame has shape (10, 50) and dtype float32."""
    assert isinstance(synthetic_classification_df, pd.DataFrame)
    assert synthetic_classification_df.shape == (10, 50)
    assert all(d == np.float32 for d in synthetic_classification_df.dtypes)


def test_classification_labels_length(synthetic_classification_labels):
    """Verify classification labels have length 10."""
    assert isinstance(synthetic_classification_labels, pd.Series)
    assert len(synthetic_classification_labels) == 10


def test_forecast_data_shape(synthetic_forecast_data):
    """Verify forecast data has shape (200, 7) and dtype float32."""
    assert isinstance(synthetic_forecast_data, np.ndarray)
    assert synthetic_forecast_data.shape == (200, 7)
    assert synthetic_forecast_data.dtype == np.float32


def test_multivariate_data_shape(synthetic_multivariate_data):
    """Verify multivariate data has shape (5, 30, 4) and dtype float32."""
    assert isinstance(synthetic_multivariate_data, np.ndarray)
    assert synthetic_multivariate_data.shape == (5, 30, 4)
    assert synthetic_multivariate_data.dtype == np.float32

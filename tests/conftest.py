"""Shared pytest fixtures for dataset tests.

Provides synthetic numpy/pandas data matching real dataset shapes
for unit testing without file I/O or downloads.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_classification_df():
    """Return a DataFrame of shape (10, 50) with dtype float32.

    10 samples, 50 timesteps — typical UCR-style univariate classification data.
    """
    return pd.DataFrame(np.random.default_rng().standard_normal((10, 50)).astype(np.float32))


@pytest.fixture
def synthetic_classification_labels():
    """Return a Series of length 10 with binary labels [0, 1] * 5."""
    return pd.Series([0, 1] * 5)


@pytest.fixture
def synthetic_forecast_data():
    """Return an ndarray of shape (200, 7) with dtype float32.

    200 timesteps, 7 features — ETTh1-style multivariate forecasting data.
    """
    return np.random.default_rng().standard_normal((200, 7)).astype(np.float32)


@pytest.fixture
def synthetic_multivariate_data():
    """Return an ndarray of shape (5, 30, 4) with dtype float32.

    5 samples, 30 timesteps, 4 features — UEA-style multivariate classification data.
    """
    return np.random.default_rng().standard_normal((5, 30, 4)).astype(np.float32)

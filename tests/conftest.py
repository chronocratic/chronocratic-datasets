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


# ----------------------------------------------------------------------- #
# Config fixtures                                                          #
# ----------------------------------------------------------------------- #


@pytest.fixture
def sample_classification_config():
    """Return a valid ClassificationConfig instance for testing.

    Uses UCR-style settings: regular data form, two classes, ARFF-based
    file patterns.
    """
    from tscollection.datasets.config.base import (
        ArffFilePattern,
        ClassificationConfig,
        ClassificationFilePatterns,
    )
    from tscollection.datasets.enums import DatasetFamily

    return ClassificationConfig(
        name='TestDataset',
        family=DatasetFamily.UCR,
        url='https://example.com/test.zip',
        num_classes=2,
        data_form='regular',
        target_col_name='Class',
        file_patterns=ClassificationFilePatterns(
            train=ArffFilePattern(arff='{dataset_name}_train.arff'),
            test=ArffFilePattern(arff='{dataset_name}_test.arff'),
        ),
        tasks=('classification', 'representation'),
    )


@pytest.fixture
def sample_forecasting_config():
    """Return a valid ForecastingConfig instance with indexed splits.

    Uses ETT-style settings: absolute row indices for train/valid/test
    boundaries (8640, 11520, 14400).
    """
    from tscollection.datasets.config.base import ForecastingConfig
    from tscollection.datasets.enums import DatasetFamily, SplitMode

    return ForecastingConfig(
        name='TestForecast',
        family=DatasetFamily.ETT,
        url='https://example.com/test.csv',
        split_mode=SplitMode.INDEXED,
        split_bounds=(8640, 11520, 14400),
        default_seq_len=128,
        default_horizon=96,
        tasks=('forecasting', 'representation'),
    )


@pytest.fixture
def sample_fractional_config():
    """Return a valid ForecastingConfig instance with fractional splits.

    Uses Electricity-style settings: 60/20/20 proportional split
    fractions.
    """
    from tscollection.datasets.config.base import ForecastingConfig
    from tscollection.datasets.enums import DatasetFamily, SplitMode

    return ForecastingConfig(
        name='TestFractional',
        family=DatasetFamily.ELECTRICITY,
        url='https://example.com/test.csv',
        split_mode=SplitMode.FRACTIONAL,
        split_bounds=(0.6, 0.2, 0.2),
        default_seq_len=128,
        default_horizon=24,
        tasks=('forecasting', 'representation'),
    )

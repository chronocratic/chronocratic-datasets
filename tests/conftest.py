"""Shared pytest fixtures for dataset tests.

Provides synthetic numpy/pandas data matching real dataset shapes
for unit testing without file I/O or downloads.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from chronocratic.datasets.utils.cache import (
    atomic_save_metadata,
    atomic_save_npz,
    load_metadata,
    load_scaler,
    save_scaler,
)


@pytest.fixture
def synthetic_classification_df() -> pd.DataFrame:
    """Return a DataFrame of shape (10, 50) with dtype float32.

    10 samples, 50 timesteps — typical UCR-style univariate classification data.
    """
    return pd.DataFrame(np.random.default_rng().standard_normal((10, 50)).astype(np.float32))


@pytest.fixture
def synthetic_classification_labels() -> pd.Series:
    """Return a Series of length 10 with binary labels [0, 1] * 5."""
    return pd.Series([0, 1] * 5)


@pytest.fixture
def synthetic_forecast_data() -> np.ndarray:
    """Return an ndarray of shape (200, 7) with dtype float32.

    200 timesteps, 7 features — ETTh1-style multivariate forecasting data.
    """
    return np.random.default_rng().standard_normal((200, 7)).astype(np.float32)


@pytest.fixture
def synthetic_multivariate_data() -> np.ndarray:
    """Return an ndarray of shape (5, 30, 4) with dtype float32.

    5 samples, 30 timesteps, 4 features — UEA-style multivariate classification data.
    """
    return np.random.default_rng().standard_normal((5, 30, 4)).astype(np.float32)


@pytest.fixture
def synthetic_cache_dir(tmp_path: Path) -> Path:
    """Create a temp directory with pre-populated cache files.

    Writes an npz file (data + index), metadata.json (version 1),
    and a scaler.pt (fitted MinMaxScaler) so downstream tests can
    exercise ``setup()`` and ``prepare_dimensions()`` without calling
    ``prepare_data()``.

    Returns:
        The temporary directory path containing the cache files.
    """
    rng = np.random.default_rng(42)
    data = rng.standard_normal((500, 7)).astype(np.float32)
    time_index = pd.date_range('2016-01-01', periods=500, freq='h')
    index_ns = time_index.astype(np.int64).to_numpy()

    cache_key = 'synthetic.cache'

    # Write npz file with data and index arrays
    npz_path = tmp_path / f'{cache_key}.npz'
    atomic_save_npz(npz_path, data=data, index=index_ns)

    # Write metadata.json
    metadata_path = tmp_path / 'metadata.json'
    atomic_save_metadata(
        metadata_path,
        {
            'version': 1,
            'dataset_name': 'SyntheticETT',
            'n_features': 7,
            'seq_len': 128,
            'splits': {
                'train': [0, 300],
                'valid': [300, 400],
                'test': [400, 500],
            },
            'has_datetime_index': True,
            'data_scaling_method': 'MINMAX',
            'data_scaling_range': [0, 1],
        },
    )

    # Fit and save scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    scaler_path = tmp_path / f'{cache_key}_data_scaler.pt'
    save_scaler(scaler, scaler_path)

    return tmp_path


def test_synthetic_cache_dir_fixture(synthetic_cache_dir: Path) -> None:
    """Verify synthetic_cache_dir fixture produces valid cache files.

    Checks that the returned path contains the expected npz, json, and
    scaler.pt files with correct content and structure.
    """
    files = list(synthetic_cache_dir.iterdir())
    assert len(files) >= 3, 'Expected at least 3 cache files (npz, json, pt)'

    # Verify npz file
    npz_path = synthetic_cache_dir / 'synthetic.cache.npz'
    loaded = np.load(str(npz_path))
    assert loaded['data'].shape == (500, 7), 'Data shape mismatch'
    assert loaded['data'].dtype == np.float32, 'Data dtype mismatch'
    assert loaded['index'].shape == (500,), 'Index shape mismatch'
    assert loaded['index'].dtype == np.int64, 'Index dtype mismatch'

    # Verify metadata.json
    metadata_path = synthetic_cache_dir / 'metadata.json'
    metadata = load_metadata(metadata_path)
    assert metadata['version'] == 1, 'Metadata version mismatch'
    assert metadata['dataset_name'] == 'SyntheticETT', 'Dataset name mismatch'

    # Verify scaler.pt
    scaler_path = synthetic_cache_dir / 'synthetic.cache_data_scaler.pt'
    scaler = load_scaler(scaler_path)
    assert isinstance(scaler, MinMaxScaler), 'Scaler is not a MinMaxScaler'

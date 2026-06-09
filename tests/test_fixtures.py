"""Tests for shared pytest fixtures.

Verifies that fixtures defined in conftest.py produce valid outputs.
"""

from pathlib import Path

import numpy as np

from tscollection.datasets.utils.cache import load_metadata, load_scaler
from sklearn.preprocessing import MinMaxScaler


def test_synthetic_cache_dir_fixture(synthetic_cache_dir: Path) -> None:  # noqa: PT004 – fixture from conftest
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

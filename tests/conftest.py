"""Shared pytest fixtures for dataset tests.

Provides synthetic numpy/pandas data matching real dataset shapes
for unit testing without file I/O or downloads.
"""

import hashlib
import http.server
from http.server import SimpleHTTPRequestHandler
import io
from pathlib import Path
import threading
import zipfile

import numpy as np
import pandas as pd
import pytest

from tscollection.datasets.config.base import (
    ArffFilePattern,
    ClassificationConfig,
    ClassificationFilePatterns,
    ForecastingConfig,
)
from tscollection.datasets.enums import DatasetFamily, SplitMode

# ----------------------------------------------------------------------- #
# Synthetic data fixtures                                                  #
# ----------------------------------------------------------------------- #


@pytest.fixture
def synthetic_classification_df() -> pd.DataFrame:
    """Return a DataFrame of shape (10, 50) with dtype float32.

    10 samples, 50 timesteps — typical UCR-style univariate classification data.
    """
    return pd.DataFrame(
        np.random.default_rng().standard_normal((10, 50)).astype(np.float32)
    )


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


# ----------------------------------------------------------------------- #
# Config fixtures                                                          #
# ----------------------------------------------------------------------- #


@pytest.fixture
def sample_classification_config() -> ClassificationConfig:
    """Return a valid ClassificationConfig instance for testing.

    Uses UCR-style settings: regular data form, two classes, ARFF-based
    file patterns.
    """
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
def sample_forecasting_config() -> ForecastingConfig:
    """Return a valid ForecastingConfig instance with indexed splits.

    Uses ETT-style settings: absolute row indices for train/valid/test
    boundaries (8640, 11520, 14400).
    """
    return ForecastingConfig(
        name='TestForecast',
        family=DatasetFamily.ETT,
        url='https://example.com/test.csv',
        split_mode=SplitMode.INDEXED,
        split_bounds=(8640, 11520, 14400),
        tasks=('forecasting', 'representation'),
    )


@pytest.fixture
def sample_fractional_config() -> ForecastingConfig:
    """Return a valid ForecastingConfig instance with fractional splits.

    Uses Electricity-style settings: 60/20/20 proportional split
    fractions.
    """
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


# ----------------------------------------------------------------------- #
# Download test fixtures                                                   #
# ----------------------------------------------------------------------- #


def _compute_sha256(file_path: Path) -> str:
    """Compute the SHA256 hex digest of a file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        64-character hexadecimal SHA256 digest string.
    """
    digest = hashlib.sha256()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture
def mock_http_server(
    tmp_path: Path,
) -> tuple[str, dict[str, str]]:
    """Start a mock HTTP server for testing downloads without network access.

    Creates test files in a temporary directory, starts a threaded
    HTTP server on an OS-assigned port, and yields the base URL
    along with SHA256 hashes for each file.

    Yields:
        A tuple of (base_url, file_hashes) where base_url is the
        server's HTTP endpoint (e.g., 'http://127.0.0.1:12345') and
        file_hashes is a dict mapping filename to SHA256 hex digest.

    Teardown:
        Shuts down the server and cleans up the temp directory
        (handled by tmp_path fixture).
    """
    # Create test files

    # test_file.zip — valid ZIP with known content
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        zip_buffer, 'w', zipfile.ZIP_DEFLATED
    ) as zf:
        zf.writestr('data.txt', 'test file content for download\n')
    (tmp_path / 'test_file.zip').write_bytes(zip_buffer.getvalue())

    # test_file.csv — small CSV with header + 3 data rows
    csv_content = 'col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n'
    (tmp_path / 'test_file.csv').write_text(csv_content, encoding='utf-8')

    # bad_file.zip — different ZIP (for SHA256 mismatch tests)
    bad_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        bad_zip_buffer, 'w', zipfile.ZIP_DEFLATED
    ) as zf:
        zf.writestr('bad.txt', 'this is different content\n')
    (tmp_path / 'bad_file.zip').write_bytes(bad_zip_buffer.getvalue())

    # arff_file.zip — valid ZIP with ARFF files for UCR/UEA tests
    arff_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        arff_zip_buffer, 'w', zipfile.ZIP_DEFLATED
    ) as zf:
        zf.writestr(
            'TestDataset_train.arff',
            '@RELATION train\n@ATTRIBUTE x NUMERIC\n@DATA\n1\n',
        )
        zf.writestr(
            'TestDataset_test.arff',
            '@RELATION test\n@ATTRIBUTE x NUMERIC\n@DATA\n2\n',
        )
    (tmp_path / 'arff_file.zip').write_bytes(arff_zip_buffer.getvalue())

    # Compute SHA256 hashes for all files
    file_hashes: dict[str, str] = {}
    for filename in ('test_file.zip', 'test_file.csv', 'bad_file.zip', 'arff_file.zip'):
        file_hashes[filename] = _compute_sha256(tmp_path / filename)

    # Start the HTTP server
    server = http.server.HTTPServer(
        ('127.0.0.1', 0),
        lambda *args: SimpleHTTPRequestHandler(*args, directory=str(tmp_path)),
    )
    port = server.server_port  # type: ignore[attr-defined]
    base_url = f'http://127.0.0.1:{port}'

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield (base_url, file_hashes)
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated cache directory for download tests.

    Sets the TSCOLLECTION_CACHE_DIR environment variable to a
    temporary path, ensuring download/cache operations don't touch
    the real user cache.

    Args:
        tmp_path: pytest-provided temporary directory.
        monkeypatch: pytest fixture for environment variable manipulation.

    Yields:
        A Path object pointing to the empty cache directory.
    """
    cache_dir = tmp_path / 'tscollection_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('TSCOLLECTION_CACHE_DIR', str(cache_dir))
    return cache_dir

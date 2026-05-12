"""Tests for conftest.py synthetic fixtures.

Validates that all fixture shapes and dtypes match the expected
values for downstream dataset tests.
"""

import os

import numpy as np
import pandas as pd
import pytest
import requests

from tscollection.datasets.config.base import (
    ArffFilePattern,
    ClassificationConfig,
    ClassificationFilePatterns,
)
from tscollection.datasets.enums import DatasetFamily


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


class TestMockHttpServer:
    """Tests for the mock_http_server fixture."""

    def test_serves_file_content(
        self, mock_http_server
    ) -> None:
        """DL-01: mock_http_server serves a known file at a predictable URL."""
        base_url, file_hashes = mock_http_server
        response = requests.get(f'{base_url}/test_file.csv')
        assert response.status_code == 200
        assert len(response.content) > 0

    def test_content_length_header(
        self, mock_http_server
    ) -> None:
        """DL-01: mock_http_server returns correct Content-Length header."""
        base_url, file_hashes = mock_http_server
        response = requests.get(f'{base_url}/test_file.zip')
        assert int(response.headers['Content-Length']) == len(response.content)

    def test_file_hashes_provided(
        self, mock_http_server
    ) -> None:
        """DL-01: mock_http_server yields SHA256 hashes for all test files."""
        base_url, file_hashes = mock_http_server
        assert 'test_file.zip' in file_hashes
        assert 'test_file.csv' in file_hashes
        assert 'bad_file.zip' in file_hashes
        for hash_value in file_hashes.values():
            assert isinstance(hash_value, str)
            assert len(hash_value) == 64


class TestTmpCacheDir:
    """Tests for the tmp_cache_dir fixture."""

    def test_sets_env_var(
        self, tmp_cache_dir
    ) -> None:
        """DL-01: tmp_cache_dir sets TSCOLLECTION_CACHE_DIR environment variable."""
        env_value = os.environ.get('TSCOLLECTION_CACHE_DIR')
        assert env_value is not None
        assert env_value == str(tmp_cache_dir)

    def test_cache_dir_exists(
        self, tmp_cache_dir
    ) -> None:
        """DL-01: tmp_cache_dir creates the cache directory on disk."""
        assert tmp_cache_dir.exists()
        assert tmp_cache_dir.is_dir()

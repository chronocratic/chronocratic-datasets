"""Tests for forecasting data downloader (DL-01).

Verifies that download_forecasting correctly fetches, caches, and
returns ETT, electricity, and weather CSV files. Tests use
mock_http_server and sample_forecasting_config fixtures.

These tests fail on import until the forecasting downloader module is
implemented (in Plan 04-02).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from tscollection.datasets.download.forecasting import download_forecasting


class TestDownloadForecasting:
    """Tests for the download_forecasting function."""

    def test_returns_csv_path(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        sample_forecasting_config,
    ) -> None:
        """DL-01: download_forecasting returns Path to downloaded CSV file."""
        base_url, _file_hashes = mock_http_server

        cfg = sample_forecasting_config.model_copy(
            update={'url': base_url + '/test_file.csv'}
        )

        result = download_forecasting(
            config=cfg,
            overwrite_cache=False,
        )
        assert isinstance(result, Path)
        assert result.exists()

    def test_calls_download_with_config_url(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        sample_forecasting_config,
    ) -> None:
        """DL-01: download_forecasting calls download_file with correct URL."""
        base_url, _file_hashes = mock_http_server

        cfg = sample_forecasting_config.model_copy(
            update={'url': base_url + '/test_file.csv'}
        )

        with patch(
            'tscollection.datasets.download.forecasting.download_file'
        ) as mock_download:
            mock_download.return_value = Path('/cache/test_file.csv')
            download_forecasting(
                config=cfg,
                overwrite_cache=False,
            )
            assert mock_download.call_count >= 1
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs['url'] == str(cfg.url)

    def test_enforces_keyword_only_args(
        self,
        sample_forecasting_config,
    ) -> None:
        """DL-01: download_forecasting enforces keyword-only arguments."""
        with pytest.raises(TypeError):
            download_forecasting(sample_forecasting_config)  # type: ignore[call-arg]

"""Tests for UCR/UEA classification downloader (DL-01, DL-04).

Verifies that download_ucr_uea correctly fetches, caches, and extracts
UCR/UEA classification dataset archives. Tests use mock_http_server
and sample_classification_config fixtures.

These tests fail on import until the ucr_uea downloader module is
implemented (in Plan 04-02).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import zipfile

import pytest
from tscollection.datasets.download.ucr_uea import download_ucr_uea


class TestDownloadUcrUea:
    """Tests for the download_ucr_uea function."""

    def test_returns_arff_paths_dict(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        sample_classification_config,
        tmp_cache_dir: Path,
    ) -> None:
        """DL-01: download_ucr_uea returns dict with train and test keys."""
        base_url, _file_hashes = mock_http_server

        # Use the mock URL with ARFF files instead of the config's real URL
        cfg = sample_classification_config.model_copy(
            update={'url': base_url + '/arff_file.zip'}
        )

        result = download_ucr_uea(
            config=cfg,
            overwrite_cache=False,
        )
        assert isinstance(result, dict)
        assert 'train' in result
        assert 'test' in result
        assert result['train'].exists()
        assert result['test'].exists()

    def test_calls_download_with_config_url(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        sample_classification_config,
    ) -> None:
        """DL-01: download_ucr_uea calls download_file with correct URL."""
        base_url, _file_hashes = mock_http_server

        cfg = sample_classification_config.model_copy(
            update={'url': base_url + '/test_file.zip'}
        )

        with (
            patch(
                'tscollection.datasets.download.ucr_uea.download_file'
            ) as mock_download,
            patch(
                'tscollection.datasets.download.ucr_uea.extract_archive'
            ) as mock_extract,
        ):
            mock_download.return_value = Path('/cache/test_file.zip')
            mock_extract.return_value = Path('/cache/TestDataset/extracted')
            download_ucr_uea(
                config=cfg,
                overwrite_cache=False,
            )
            assert mock_download.call_count >= 1
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs['url'] == str(cfg.url)

    def test_enforces_keyword_only_args(
        self,
        sample_classification_config,
    ) -> None:
        """DL-01: download_ucr_uea enforces keyword-only arguments."""
        with pytest.raises(TypeError):
            download_ucr_uea(sample_classification_config)  # type: ignore[call-arg]

    def test_respects_overwrite_cache(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        sample_classification_config,
        tmp_cache_dir: Path,
    ) -> None:
        """DL-04: download_ucr_uea uses cache on second call (no re-download)."""
        base_url, file_hashes = mock_http_server

        # Use nested_arff_file.zip so rglob fallback finds the ARFF files
        cfg = sample_classification_config.model_copy(
            update={'url': base_url + '/nested_arff_file.zip'}
        )

        # First call: downloads and extracts
        result1 = download_ucr_uea(
            config=cfg,
            overwrite_cache=False,
        )
        assert result1['train'].exists()
        assert result1['test'].exists()

        # Second call: uses cached files (download_file skips network on hit)
        result2 = download_ucr_uea(
            config=cfg,
            overwrite_cache=False,
        )
        assert result2['train'].exists()
        assert result2['test'].exists()

        # Verify the cached archive is the one from the mock server
        archive_path = tmp_cache_dir / cfg.name / f'{cfg.name}.zip'
        assert archive_path.exists()
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
        assert any('train' in n for n in names)
        assert any('test' in n for n in names)

    def test_calls_clear_cache_on_overwrite(
        self,
        sample_classification_config,
    ) -> None:
        """DL-04: download_ucr_uea calls clear_cache_dir when overwrite=True."""
        cfg = sample_classification_config.model_copy(
            update={'url': 'https://example.com/test.zip'}
        )

        with (
            patch(
                'tscollection.datasets.download.ucr_uea.clear_cache_dir'
            ) as mock_clear,
            patch(
                'tscollection.datasets.download.ucr_uea.download_file'
            ) as mock_download,
            patch(
                'tscollection.datasets.download.ucr_uea.extract_archive'
            ) as mock_extract,
        ):
            mock_download.return_value = Path('/cache/test_file.zip')
            mock_extract.return_value = Path('/cache/TestDataset/extracted')
            download_ucr_uea(config=cfg, overwrite_cache=True)
            mock_clear.assert_called_once_with(dataset_name=cfg.name)

    def test_arff_rglob_fallback_finds_nested_files(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        sample_classification_config,
        tmp_cache_dir: Path,
    ) -> None:
        """IN-04: rglob fallback locates ARFF files inside subdirectories."""
        base_url, _file_hashes = mock_http_server

        # Use nested_arff_file.zip which has files in a subdirectory
        cfg = sample_classification_config.model_copy(
            update={'url': base_url + '/nested_arff_file.zip'}
        )

        result = download_ucr_uea(
            config=cfg,
            overwrite_cache=False,
        )
        assert isinstance(result, dict)
        assert 'train' in result
        assert 'test' in result
        assert result['train'].exists()
        assert result['test'].exists()


class TestHttpsValidator:
    """Tests for the HTTPS-only URL validator on DatasetConfig."""

    def test_rejects_http_url(self) -> None:
        """CR-02: ClassificationConfig rejects HTTP URLs at construction."""
        from pydantic import ValidationError

        from tscollection.datasets.config.base import (
            ArffFilePattern,
            ClassificationConfig,
            ClassificationFilePatterns,
        )
        from tscollection.datasets.enums import DatasetFamily

        with pytest.raises(ValidationError):
            ClassificationConfig(
                name='TestHttpReject',
                family=DatasetFamily.UCR,
                url='http://example.com/test.zip',
                num_classes=2,
                data_form='regular',
                target_col_name='Class',
                file_patterns=ClassificationFilePatterns(
                    train=ArffFilePattern(arff='{dataset_name}_train.arff'),
                    test=ArffFilePattern(arff='{dataset_name}_test.arff'),
                ),
                tasks=('classification',),
            )

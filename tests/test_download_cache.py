"""Tests for download cache primitives (DL-01 through DL-04).

Verifies that get_cache_dir, download_file, file_exists_in_cache, and
extract_archive work correctly without network access by using the
mock_http_server and tmp_cache_dir fixtures.

These tests fail on import until the cache module is implemented
(in Plans 04-01 and 04-02).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from tscollection.datasets.download.cache import (
    download_file,
    extract_archive,
    file_exists_in_cache,
    get_cache_dir,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestGetCacheDir:
    """Tests for the get_cache_dir function."""

    def test_default_path(self, tmp_cache_dir: Path) -> None:
        """DL-01: get_cache_dir returns the TSCOLLECTION_CACHE_DIR path."""
        result = get_cache_dir()
        assert str(tmp_cache_dir) in str(result)

    def test_env_override(self, tmp_cache_dir: Path) -> None:
        """DL-01: get_cache_dir respects TSCOLLECTION_CACHE_DIR env var."""
        result = get_cache_dir()
        assert result == tmp_cache_dir

    def test_creates_directory(self, tmp_path: Path) -> None:
        """DL-01: get_cache_dir creates directory if it does not exist."""
        import os

        new_dir = tmp_path / 'new_cache_dir'
        old_value = os.environ.get('TSCOLLECTION_CACHE_DIR')
        try:
            os.environ['TSCOLLECTION_CACHE_DIR'] = str(new_dir)
            result = get_cache_dir()
            assert result.exists()
            assert result.is_dir()
        finally:
            if old_value is None:
                os.environ.pop('TSCOLLECTION_CACHE_DIR', None)
            else:
                os.environ['TSCOLLECTION_CACHE_DIR'] = old_value


class TestDownloadFile:
    """Tests for the download_file function."""

    def test_downloads_to_cache(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        tmp_cache_dir: Path,
    ) -> None:
        """DL-01: download_file streams content and writes to cache."""
        base_url, file_hashes = mock_http_server
        url = f'{base_url}/test_file.zip'
        expected_hash = file_hashes['test_file.zip']
        result = download_file(
            url=url,
            sha256=expected_hash,
            cache_dir=tmp_cache_dir,
            filename='test_file.zip',
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_validates_sha256_match(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        tmp_cache_dir: Path,
    ) -> None:
        """DL-02: download_file validates SHA256 on success."""
        base_url, file_hashes = mock_http_server
        url = f'{base_url}/test_file.zip'
        expected_hash = file_hashes['test_file.zip']
        result = download_file(
            url=url,
            sha256=expected_hash,
            cache_dir=tmp_cache_dir,
            filename='test_file.zip',
        )
        assert result.exists()

    def test_raises_on_sha256_mismatch(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        tmp_cache_dir: Path,
    ) -> None:
        """DL-02: download_file raises ValueError on SHA256 mismatch."""
        base_url, file_hashes = mock_http_server
        url = f'{base_url}/test_file.zip'
        wrong_hash = file_hashes['bad_file.zip']
        with pytest.raises(ValueError, match='SHA256 mismatch'):
            download_file(
                url=url,
                sha256=wrong_hash,
                cache_dir=tmp_cache_dir,
                filename='test_file.zip',
            )

    def test_warns_when_no_sha256(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        tmp_cache_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """DL-02: download_file logs warning when sha256=None."""
        import logging

        base_url, _file_hashes = mock_http_server
        url = f'{base_url}/test_file.zip'
        with caplog.at_level(logging.WARNING):
            result = download_file(
                url=url,
                sha256=None,
                cache_dir=tmp_cache_dir,
                filename='test_file.zip',
            )
        assert result.exists()
        assert any('SHA256' in record.message for record in caplog.records)

    def test_cache_hit_skips_download(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        tmp_cache_dir: Path,
    ) -> None:
        """DL-03: download_file skips download on cache hit."""
        base_url, file_hashes = mock_http_server
        url = f'{base_url}/test_file.zip'
        expected_hash = file_hashes['test_file.zip']

        # First download
        result = download_file(
            url=url,
            sha256=expected_hash,
            cache_dir=tmp_cache_dir,
            filename='test_file.zip',
        )
        assert result.exists()

        # Second download: force network path to fail, verify cache is used
        with patch(
            'tscollection.datasets.download.cache._create_session',
            side_effect=RuntimeError('network should not be called on cache hit'),
        ):
            result2 = download_file(
                url=url,
                sha256=expected_hash,
                cache_dir=tmp_cache_dir,
                filename='test_file.zip',
                overwrite_cache=False,
            )
        assert result2.exists()


    def test_overwrite_cache_forces_redownload(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        tmp_cache_dir: Path,
    ) -> None:
        """DL-04: download_file redownloads when overwrite_cache=True."""
        base_url, file_hashes = mock_http_server
        url = f'{base_url}/test_file.zip'
        expected_hash = file_hashes['test_file.zip']

        # First download
        result = download_file(
            url=url,
            sha256=expected_hash,
            cache_dir=tmp_cache_dir,
            filename='test_file.zip',
        )
        assert result.exists()
        original_size = result.stat().st_size

        # Corrupt the cached file
        result.write_text('corrupted content', encoding='utf-8')

        # Redownload with overwrite
        result2 = download_file(
            url=url,
            sha256=expected_hash,
            cache_dir=tmp_cache_dir,
            filename='test_file.zip',
            overwrite_cache=True,
        )
        assert result2.stat().st_size == original_size


class TestFileExistsInCache:
    """Tests for the file_exists_in_cache function."""

    def test_returns_true_for_valid_cache(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        tmp_cache_dir: Path,
    ) -> None:
        """DL-03: file_exists_in_cache returns True for valid cached file."""
        base_url, file_hashes = mock_http_server
        url = f'{base_url}/test_file.zip'
        expected_hash = file_hashes['test_file.zip']

        # Download the file first
        download_file(
            url=url,
            sha256=expected_hash,
            cache_dir=tmp_cache_dir,
            filename='test_file.zip',
        )

        result = file_exists_in_cache(
            cache_dir=tmp_cache_dir,
            sha256=expected_hash,
        )
        assert result is True

    def test_returns_false_for_missing_cache(
        self,
        tmp_cache_dir: Path,
    ) -> None:
        """DL-03: file_exists_in_cache returns False for missing file."""
        result = file_exists_in_cache(
            cache_dir=tmp_cache_dir,
            sha256='a' * 64,
        )
        assert result is False


class TestExtractArchive:
    """Tests for the extract_archive function."""

    def test_extract_zip_archive(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        tmp_cache_dir: Path,
    ) -> None:
        """DL-01: extract_archive extracts zip contents safely."""
        base_url, file_hashes = mock_http_server
        url = f'{base_url}/test_file.zip'
        expected_hash = file_hashes['test_file.zip']

        # Download the archive
        archive_path = download_file(
            url=url,
            sha256=expected_hash,
            cache_dir=tmp_cache_dir,
            filename='test_file.zip',
        )

        # Extract it
        extract_dir = tmp_cache_dir / 'extracted'
        extract_archive(
            archive_path=archive_path,
            extract_to=extract_dir,
        )

        assert (extract_dir / 'data.txt').exists()

    def test_returns_extracted_directory(
        self,
        mock_http_server: tuple[str, dict[str, str]],
        tmp_cache_dir: Path,
    ) -> None:
        """DL-01: extract_archive returns extracted directory path."""
        base_url, file_hashes = mock_http_server
        url = f'{base_url}/test_file.zip'
        expected_hash = file_hashes['test_file.zip']

        # Download the archive
        archive_path = download_file(
            url=url,
            sha256=expected_hash,
            cache_dir=tmp_cache_dir,
            filename='test_file.zip',
        )

        # Extract it
        extract_dir = tmp_cache_dir / 'extracted2'
        result = extract_archive(
            archive_path=archive_path,
            extract_to=extract_dir,
        )

        assert result == extract_dir

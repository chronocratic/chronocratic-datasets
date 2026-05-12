"""UCR/UEA family downloader for classification datasets.

Downloads ZIP archives from UCR/UEA time series classification repos,
validates checksums, extracts ARFF files, and returns structured paths.
Consumes Phase 3 Pydantic ``ClassificationConfig`` instances.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from tscollection.datasets.download.cache import (
    clear_cache_dir,
    download_file,
    extract_archive,
    get_cache_dir,
)

if TYPE_CHECKING:
    from tscollection.datasets.config.base import ClassificationConfig

__all__ = ['download_ucr_uea']

logger = logging.getLogger(__name__)


def download_ucr_uea(
    *,
    config: ClassificationConfig,
    overwrite_cache: bool = False,
) -> dict[str, Path]:
    """Download and extract a UCR/UEA classification dataset archive.

    Fetches the ZIP archive specified by ``config.url``, validates the
    SHA256 checksum if provided, extracts contents to the cache directory,
    and returns resolved paths for the training and test ARFF files.

    When ``overwrite_cache`` is True the full per-dataset subdirectory
    is removed before downloading, ensuring a clean state.

    Args:
        config: A frozen ClassificationConfig instance providing the
            dataset name, URL, checksum, and file patterns.
        overwrite_cache: If True, delete the cached dataset directory
            and force a fresh download.

    Returns:
        A dict with ``'train'`` and ``'test'`` keys mapping to
        ``pathlib.Path`` objects pointing to the ARFF files.

    Raises:
        ValueError: If the archive checksum does not match the expected
            SHA256 digest.
        FileNotFoundError: If the extracted archive does not contain
            ARFF files matching the expected patterns.
    """
    cache_root = get_cache_dir()
    dataset_dir = cache_root / config.name

    # Clear cached data when forced refresh is requested
    if overwrite_cache:
        clear_cache_dir(dataset_name=config.name)

    # Download the archive
    archive_name = f'{config.name}.zip'
    archive_path = download_file(
        url=str(config.url),
        sha256=config.sha256,
        cache_dir=dataset_dir,
        filename=archive_name,
        overwrite_cache=False,
    )

    # Extract archive contents
    extract_dir = dataset_dir / 'extracted'
    extract_archive(archive_path=archive_path, extract_to=extract_dir)

    # Build expected ARFF paths from file patterns
    train_pattern = config.file_patterns.train.arff.replace(
        '{dataset_name}', config.name
    )
    test_pattern = config.file_patterns.test.arff.replace(
        '{dataset_name}', config.name
    )

    train_path = extract_dir / train_pattern
    test_path = extract_dir / test_pattern

    # Fallback: search recursively (handles nested subdirectories in UEA)
    if not train_path.exists() or not test_path.exists():
        logger.info(
            'Expected ARFF paths not found directly; searching recursively in %s',
            extract_dir,
        )
        arff_files = list(extract_dir.rglob('*.arff'))
        for arff_file in arff_files:
            if 'train' in arff_file.name.lower() and not train_path.exists():
                train_path = arff_file
            if 'test' in arff_file.name.lower() and not test_path.exists():
                test_path = arff_file

    logger.info(
        'UCR/UEA dataset %s ready — train: %s, test: %s',
        config.name,
        train_path,
        test_path,
    )

    return {'train': train_path, 'test': test_path}

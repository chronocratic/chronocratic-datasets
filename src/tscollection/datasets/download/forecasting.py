"""Forecasting family downloader for CSV-based datasets.

Downloads CSV files for forecasting datasets (ETT, electricity, weather),
validates checksums, and returns the cached file path. Consumes Phase 3
Pydantic ``ForecastingConfig`` instances.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from tscollection.datasets.download.cache import (
    clear_cache_dir,
    download_file,
    get_cache_dir,
)

if TYPE_CHECKING:
    from tscollection.datasets.config.base import ForecastingConfig

__all__ = ['download_forecasting']

logger = logging.getLogger(__name__)


def download_forecasting(
    *,
    config: ForecastingConfig,
    overwrite_cache: bool = False,
) -> Path:
    """Download a forecasting CSV dataset.

    Fetches the CSV file specified by ``config.url``, validates the SHA256
    checksum if provided, and caches it under the per-dataset subdirectory.

    When ``overwrite_cache`` is True the full per-dataset subdirectory
    is removed before downloading, ensuring a clean state.

    Args:
        config: A frozen ForecastingConfig instance providing the dataset
            name, URL, and optional checksum.
        overwrite_cache: If True, delete the cached dataset directory
            and force a fresh download.

    Returns:
        A ``pathlib.Path`` pointing to the cached CSV file.

    Raises:
        ValueError: If the file checksum does not match the expected
            SHA256 digest.
    """
    cache_root = get_cache_dir()
    dataset_dir = cache_root / config.name

    # Clear cached data when forced refresh is requested
    if overwrite_cache:
        clear_cache_dir(dataset_name=config.name)

    # Extract filename from URL path
    parsed = urlparse(str(config.url))
    filename = Path(parsed.path).name

    if not filename:
        raise ValueError(
            f"Cannot extract filename from URL '{config.url}'. "
            "URL must end with a filename component."
        )

    csv_path = download_file(
        url=str(config.url),
        sha256=config.sha256,
        cache_dir=dataset_dir,
        filename=filename,
        overwrite_cache=False,
    )

    logger.info('Forecasting dataset %s ready — %s', config.name, csv_path)
    return csv_path

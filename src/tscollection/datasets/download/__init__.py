"""Data download and caching utilities.

Provides cache primitives (download, extract, validate) and
family-specific downloaders for UCR/UEA classification and
forecasting CSV datasets.
"""

from tscollection.datasets.download.cache import (
    clear_cache_dir,
    download_file,
    extract_archive,
    file_exists_in_cache,
    get_cache_dir,
)
from tscollection.datasets.download.forecasting import download_forecasting
from tscollection.datasets.download.ucr_uea import download_ucr_uea

__all__ = [
    'clear_cache_dir',
    'download_file',
    'download_forecasting',
    'download_ucr_uea',
    'extract_archive',
    'file_exists_in_cache',
    'get_cache_dir',
]

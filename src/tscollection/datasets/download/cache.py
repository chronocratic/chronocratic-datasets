"""Download and caching utilities for time series datasets."""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    pass

__all__ = [
    'clear_cache_dir',
    'download_file',
    'extract_archive',
    'file_exists_in_cache',
    'get_cache_dir',
]

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 32 * 1024  # 32 KB -- torchvision default


def get_cache_dir() -> Path:
    """Return the tscollection cache directory, creating it if needed.

    Reads ``TSCOLLECTION_CACHE_DIR`` from the environment. Falls back to
    ``~/.cache/tscollection`` when the variable is not set.

    Returns:
        Path to the cache root directory.
    """
    default = str(Path.home() / '.cache' / 'tscollection')
    cache_root = Path(os.environ.get('TSCOLLECTION_CACHE_DIR', default))
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _create_session() -> requests.Session:
    """Create a requests session with exponential backoff retry.

    Returns:
        Configured requests.Session instance.
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=['GET'],
        raise_on_status=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


def _compute_file_hash(file_path: Path) -> str:
    """Compute the SHA256 hex digest of an existing file.

    Args:
        file_path: Path to the file to hash.

    Returns:
        64-character hexadecimal SHA256 digest string.
    """
    hash_obj = hashlib.sha256()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def download_file(
    *,
    url: str,
    sha256: str | None,
    cache_dir: Path,
    filename: str,
    overwrite_cache: bool = False,
) -> Path:
    """Download a file to cache_dir with SHA256 validation.

    Creates the cache directory if it does not exist. On cache hit
    (file exists, hash matches, and ``overwrite_cache`` is False) the
    HTTP request is skipped.

    Args:
        url: Download URL.
        sha256: Expected SHA256 hex digest, or None to skip validation.
        cache_dir: Directory to cache the file in.
        filename: Name to save the file as.
        overwrite_cache: If True, delete existing file and redownload.

    Returns:
        Path to the downloaded file.

    Raises:
        ValueError: If SHA256 validation fails.
        requests.HTTPError: If download fails after retries.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename

    # Cache hit check
    if destination.exists() and not overwrite_cache:
        if sha256 is not None:
            actual = _compute_file_hash(destination)
            if actual == sha256:
                logger.info('Cache hit: %s', destination)
                return destination
        else:
            logger.info('Cache hit (no hash): %s', destination)
            return destination

    # Remove cached file if overwriting
    if destination.exists():
        destination.unlink()

    logger.info('Downloading %s -> %s', url, destination)
    hash_obj = hashlib.sha256()

    with _create_session() as session:
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total = int(response.headers.get('Content-Length', 0))

        with (
            open(destination, 'wb') as fh,
            tqdm(
                total=total,
                unit='B',
                unit_scale=True,
                desc=f'Downloading {filename}',
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                fh.write(chunk)
                hash_obj.update(chunk)
                pbar.update(len(chunk))

    # Validate SHA256
    if sha256 is not None:
        actual = hash_obj.hexdigest()
        if actual != sha256:
            destination.unlink(missing_ok=True)
            raise ValueError(
                f'SHA256 mismatch for {url}: expected {sha256}, got {actual}'
            )
    else:
        logger.warning(
            'No SHA256 checksum provided -- skipping integrity verification for %s',
            filename,
        )

    return destination


def file_exists_in_cache(
    *,
    cache_dir: Path,
    sha256: str | None,
) -> bool:
    """Check whether a cached archive exists in the given directory.

    Searches for ``*.zip`` and ``*.csv`` files in ``cache_dir``. If
    ``sha256`` is provided, validates the hash of found archives.

    Args:
        cache_dir: Directory to search for cached archives.
        sha256: Expected SHA256 hex digest, or None to skip hash validation.

    Returns:
        True if a matching archive is found, False otherwise.
    """
    if not cache_dir.is_dir():
        return False

    archives = list(cache_dir.glob('*.zip')) + list(cache_dir.glob('*.csv'))
    if not archives:
        return False

    if sha256 is not None:
        for archive in archives:
            if _compute_file_hash(archive) == sha256:
                return True
        return False

    logger.warning('sha256 not set -- using existence check only')
    return True


def extract_archive(*, archive_path: Path, extract_to: Path) -> Path:
    """Extract a ZIP archive to the target directory.

    Uses ``zipfile.ZipFile.extractall()`` which is safe against zip-slip
    attacks in Python 3.12+ (rejects absolute paths and ``..`` components).

    Args:
        archive_path: Path to the ZIP archive file.
        extract_to: Directory to extract into. Created if it does not exist.

    Returns:
        The ``extract_to`` path.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as zf:
        zf.extractall(extract_to)
    return extract_to


def clear_cache_dir(*, dataset_name: str) -> None:
    """Remove the full cache subdirectory for a dataset.

    Deletes ``~/.cache/tscollection/{dataset_name}/`` and all its contents
    via ``shutil.rmtree``. This is the mechanism used by
    ``overwrite_cache=True`` to force a fresh download.

    Args:
        dataset_name: Dataset name (used as subdirectory in cache root).
    """
    target_dir = get_cache_dir() / dataset_name
    if target_dir.is_dir():
        shutil.rmtree(target_dir)
        logger.info('Cleared cache directory for %s: %s', dataset_name, target_dir)
    else:
        logger.debug('Cache directory not found for %s: %s', dataset_name, target_dir)

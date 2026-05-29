"""Cache utilities for DDP-safe DataModule persistence.

Provides deterministic key derivation, atomic file I/O, metadata
versioning, and scaler persistence for the Lightning DataModule
cache layer used across distributed training ranks.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

CACHE_SCHEMA_VERSION: int = 1

__all__ = [
    'CACHE_SCHEMA_VERSION',
    'atomic_save_metadata',
    'atomic_save_npz',
    'build_cache_key',
    'load_metadata',
    'load_scaler',
    'resolve_cache_dir',
    'save_scaler',
]


def build_cache_key(
    *,
    dataset_name: str,
    params: dict[str, Any],
) -> str:
    """Build a hybrid cache key: SHA-256 hash prefix plus readable suffix.

    The key format is ``<8-char-sha256>_<dataset>_<key-params>.cache``.
    Example: ``a3f8e1c2_ETTm1_seq_len=128_mode=UNIVARIATE.cache``.

    Args:
        dataset_name: Dataset identifier (e.g. ``"ETTm1"``).
        params: Parameters that affect data layout (seq_len, mode,
            scaling_method, etc.). Dict ordering does not affect the
            resulting key.

    Returns:
        A deterministic cache key string.
    """
    serialized = json.dumps(params, sort_keys=True)
    hash_prefix = hashlib.sha256(serialized.encode()).hexdigest()[:8]

    suffix_parts = [hash_prefix, dataset_name]
    for key, value in sorted(params.items()):
        suffix_parts.append(f'{key}={value}')

    return '_'.join(suffix_parts) + '.cache'


def resolve_cache_dir(
    *,
    cache_dir: Path | None,
    dataset_name: str,
) -> Path:
    """Resolve the absolute cache directory path.

    When ``cache_dir`` is ``None``, the default location
    ``~/.cache/tsdatasets/{dataset_name}`` is used.  A custom path
    is expanded (``~``) and resolved to an absolute path.

    Args:
        cache_dir: User-provided cache directory, or ``None`` for
            the default location.
        dataset_name: Dataset identifier appended to the default
            cache root.

    Returns:
        An absolute ``Path`` to the cache directory.
    """
    if cache_dir is not None:
        return cache_dir.expanduser().resolve()
    return Path.home().resolve() / '.cache' / 'tsdatasets' / dataset_name


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Save numpy arrays to a compressed ``.npz`` file atomically.

    Writes to a temporary ``.npz`` file in the same directory, then
    uses ``Path.replace()`` for POSIX atomicity.  The tmp file is
    created in the same directory as the target to guarantee same-
    filesystem rename.

    Args:
        path: Target ``.npz`` file path.
        **arrays: Named arrays to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez_compressed appends .npz to the given path, so we
    # write to a stem-named temp file and rename the resulting .npz.
    tmp_stem = path.with_name(path.stem + '_tmp')
    np.savez_compressed(str(tmp_stem), **arrays)
    actual_tmp = Path(str(tmp_stem) + '.npz')
    actual_tmp.replace(path)


def atomic_save_metadata(path: Path, data: dict[str, Any]) -> None:
    """Save a metadata dictionary to a JSON file atomically.

    Writes to a ``.json.tmp`` intermediate file then uses
    ``Path.replace()`` for POSIX atomicity.

    Args:
        path: Target ``.json`` file path.
        data: Metadata dictionary to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def load_metadata(path: Path) -> dict[str, Any]:
    """Load and validate metadata from a JSON file.

    Checks that the ``version`` field matches
    :data:`CACHE_SCHEMA_VERSION`.  Raises ``FileNotFoundError`` if the
    file does not exist and ``ValueError`` on version mismatch.

    Args:
        path: Metadata ``.json`` file path.

    Returns:
        The parsed metadata dictionary.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
        ValueError: If the schema version does not match
            :data:`CACHE_SCHEMA_VERSION`.
    """
    if not path.exists():
        msg = f'Metadata not found: {path}'
        raise FileNotFoundError(msg)

    with path.open() as f:
        data = json.load(f)

    actual = data.get('version')
    if actual != CACHE_SCHEMA_VERSION:
        msg = (
            f'Cache version {actual} does not match expected version '
            f'{CACHE_SCHEMA_VERSION}. Delete cache dir and re-run '
            f'prepare_data().'
        )
        raise ValueError(msg)

    return data


def save_scaler(scaler: Any, path: Path) -> None:  # noqa: ANN401
    """Persist a fitted sklearn scaler via ``torch.save``.

    Uses ``pickle_protocol=5`` and writes atomically through a
    ``.pt.tmp`` intermediate file.

    Args:
        scaler: Fitted scaler instance (e.g. ``MinMaxScaler``).
        path: Target ``.pt`` file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(scaler, str(tmp), pickle_protocol=5)
    tmp.replace(path)


def load_scaler(path: Path) -> Any:  # noqa: ANN401
    """Load a persisted sklearn scaler via ``torch.load``.

    Args:
        path: ``.pt`` file path containing a pickled scaler.

    Returns:
        The loaded scaler instance.
    """
    return torch.load(path, weights_only=False)

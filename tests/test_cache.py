"""Tests for cache utility functions.

Verifies deterministic key derivation, atomic file I/O, metadata
versioning, scaler persistence, and DatetimeIndex round-trips.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from tscollection.datasets.utils.cache import (
    CACHE_SCHEMA_VERSION,
    atomic_save_metadata,
    atomic_save_npz,
    build_cache_key,
    load_metadata,
    load_scaler,
    resolve_cache_dir,
    save_scaler,
)


# --------------------------------------------------------------------------- #
# CACHE_SCHEMA_VERSION                                                         #
# --------------------------------------------------------------------------- #


def test_cache_schema_version_is_one() -> None:
    """CACHE_SCHEMA_VERSION equals 1."""
    assert CACHE_SCHEMA_VERSION == 1


# --------------------------------------------------------------------------- #
# build_cache_key                                                              #
# --------------------------------------------------------------------------- #


def test_build_cache_key_produces_deterministic_output() -> None:
    """Same inputs always produce the same key."""
    params = {"seq_len": 128, "mode": "UNIVARIATE", "data_scaling_method": "MINMAX"}
    key1 = build_cache_key(dataset_name="ETTm1", params=params)
    key2 = build_cache_key(dataset_name="ETTm1", params=params)
    assert key1 == key2


def test_build_cache_key_format_matches_pattern() -> None:
    """Key matches <8-char-sha256>_<dataset>_<params>.cache."""
    pattern = re.compile(r'^[0-9a-f]{8}_[^.]+\.[cC]ache$')
    params = {"seq_len": 128, "mode": "UNIVARIATE"}
    key = build_cache_key(dataset_name="ETTm1", params=params)
    assert pattern.match(key), f"Key '{key}' does not match expected pattern"


def test_build_cache_key_includes_dataset_name() -> None:
    """Dataset name appears in the key suffix."""
    params = {"seq_len": 64}
    key = build_cache_key(dataset_name="Weather", params=params)
    assert "Weather" in key


def test_build_cache_key_invariant_to_param_order() -> None:
    """Key is the same regardless of param dict ordering."""
    params_a = {"seq_len": 128, "mode": "UNIVARIATE", "data_scaling_method": "MINMAX"}
    params_b = {"data_scaling_method": "MINMAX", "seq_len": 128, "mode": "UNIVARIATE"}
    key_a = build_cache_key(dataset_name="ETTm1", params=params_a)
    key_b = build_cache_key(dataset_name="ETTm1", params=params_b)
    assert key_a == key_b


def test_build_cache_key_different_params_produce_different_keys() -> None:
    """Different params produce different keys."""
    params_a = {"seq_len": 128}
    params_b = {"seq_len": 256}
    key_a = build_cache_key(dataset_name="ETTm1", params=params_a)
    key_b = build_cache_key(dataset_name="ETTm1", params=params_b)
    assert key_a != key_b


def test_build_cache_key_includes_hash_prefix() -> None:
    """Key starts with 8-character hex SHA-256 prefix."""
    params = {"seq_len": 128}
    key = build_cache_key(dataset_name="ETTm1", params=params)
    prefix = key.split("_")[0]
    assert len(prefix) == 8
    assert re.fullmatch(r'[0-9a-f]{8}', prefix) is not None


# --------------------------------------------------------------------------- #
# resolve_cache_dir                                                            #
# --------------------------------------------------------------------------- #


def test_resolve_cache_dir_default_returns_expected_path() -> None:
    """resolve_cache_dir(None) returns ~/.cache/tsdatasets/<name>."""
    result = resolve_cache_dir(cache_dir=None, dataset_name="ETTm1")
    expected = Path.home() / '.cache' / 'tsdatasets' / 'ETTm1'
    assert result == expected.resolve()


def test_resolve_cache_dir_custom_path_passes_through() -> None:
    """Custom path is resolved and expanded."""
    custom = Path('/tmp/custom_cache')
    result = resolve_cache_dir(cache_dir=custom, dataset_name="ETTm1")
    assert result == custom.resolve()


def test_resolve_cache_dir_expands_user_tilde() -> None:
    """Tilde in custom path is expanded."""
    custom = Path('~/my_cache')
    result = resolve_cache_dir(cache_dir=custom, dataset_name="ETTm1")
    assert str(result).startswith(str(Path.home()))


# --------------------------------------------------------------------------- #
# atomic_save_npz                                                              #
# --------------------------------------------------------------------------- #


def test_atomic_save_npz_creates_valid_file(tmp_path: Path) -> None:
    """atomic_save_npz writes a valid .npz file."""
    path = tmp_path / 'test.npz'
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    atomic_save_npz(path, data=data)
    loaded = np.load(str(path))
    assert 'data' in loaded
    assert np.allclose(loaded['data'], data)


def test_atomic_save_npz_no_tmp_file_after_save(tmp_path: Path) -> None:
    """No .tmp file remains after atomic_save_npz completes."""
    path = tmp_path / 'test.npz'
    data = np.array([1.0, 2.0, 3.0])
    atomic_save_npz(path, data=data)
    tmp_file = tmp_path / 'test_tmp.npz'
    assert not tmp_file.exists()


def test_atomic_save_npz_multiple_arrays(tmp_path: Path) -> None:
    """Multiple arrays saved and loaded correctly."""
    path = tmp_path / 'multi.npz'
    arr_a = np.array([1, 2, 3], dtype=np.int64)
    arr_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    atomic_save_npz(path, a=arr_a, b=arr_b)
    loaded = np.load(str(path))
    assert np.array_equal(loaded['a'], arr_a)
    assert np.array_equal(loaded['b'], arr_b)
    assert loaded['a'].dtype == np.int64
    assert loaded['b'].dtype == np.float32


# --------------------------------------------------------------------------- #
# atomic_save_metadata                                                         #
# --------------------------------------------------------------------------- #


def test_atomic_save_metadata_creates_valid_json(tmp_path: Path) -> None:
    """atomic_save_metadata writes a valid JSON file."""
    path = tmp_path / 'metadata.json'
    data = {"version": 1, "dataset_name": "ETTm1", "seq_len": 128}
    atomic_save_metadata(path, data)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == data


def test_atomic_save_metadata_no_tmp_file_after_save(tmp_path: Path) -> None:
    """No .tmp file remains after atomic_save_metadata completes."""
    path = tmp_path / 'metadata.json'
    data = {"version": 1}
    atomic_save_metadata(path, data)
    tmp_file = tmp_path / 'metadata.json.tmp'
    assert not tmp_file.exists()


# --------------------------------------------------------------------------- #
# load_metadata                                                                #
# --------------------------------------------------------------------------- #


def test_load_metadata_returns_dict(tmp_path: Path) -> None:
    """load_metadata returns the metadata dict for version 1."""
    path = tmp_path / 'metadata.json'
    data = {
        "version": 1,
        "dataset_name": "ETTm1",
        "n_features": 7,
        "seq_len": 128,
    }
    atomic_save_metadata(path, data)
    result = load_metadata(path)
    assert result["dataset_name"] == "ETTm1"
    assert result["seq_len"] == 128


def test_load_metadata_raises_value_error_on_version_mismatch(tmp_path: Path) -> None:
    """load_metadata raises ValueError when version != 1."""
    path = tmp_path / 'metadata.json'
    data = {"version": 2, "dataset_name": "ETTm1"}
    atomic_save_metadata(path, data)
    with pytest.raises(ValueError, match="Cache version"):
        load_metadata(path)


def test_load_metadata_raises_file_not_found(tmp_path: Path) -> None:
    """load_metadata raises FileNotFoundError for missing file."""
    path = tmp_path / 'nonexistent.json'
    with pytest.raises(FileNotFoundError):
        load_metadata(path)


# --------------------------------------------------------------------------- #
# save_scaler / load_scaler                                                    #
# --------------------------------------------------------------------------- #


def test_scaler_round_trip(tmp_path: Path) -> None:
    """save_scaler/load_scaler produces identical transform output."""
    path = tmp_path / 'scaler.pt'
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    scaler.fit(train_data)
    transform_before = scaler.transform(train_data)

    save_scaler(scaler=scaler, path=path)
    loaded_scaler = load_scaler(path)

    transform_after = loaded_scaler.transform(train_data)
    assert np.allclose(transform_before, transform_after)


def test_scaler_file_exists_after_save(tmp_path: Path) -> None:
    """Scaler file exists and no tmp remains after save."""
    path = tmp_path / 'scaler.pt'
    scaler = MinMaxScaler()
    save_scaler(scaler=scaler, path=path)
    assert path.exists()
    tmp_file = tmp_path / 'scaler.pt.tmp'
    assert not tmp_file.exists()


# --------------------------------------------------------------------------- #
# DatetimeIndex round-trip                                                     #
# --------------------------------------------------------------------------- #


def test_datetime_index_serialization_round_trip(tmp_path: Path) -> None:
    """DatetimeIndex serialized as int64 nanoseconds reconstructs."""
    path = tmp_path / 'index_test.npz'
    original = pd.date_range('2020-01-01', periods=100, freq='h')
    index_ns = original.as_unit('ns').view(np.int64)
    atomic_save_npz(path, index=index_ns)

    loaded = np.load(str(path))
    reconstructed = pd.DatetimeIndex(loaded['index'])
    assert len(reconstructed) == len(original)
    assert (reconstructed == original).all()


def test_datetime_index_preserves_timezone_naive(tmp_path: Path) -> None:
    """Round-trip preserves timezone-naive timestamps."""
    path = tmp_path / 'tz_test.npz'
    original = pd.DatetimeIndex(['2020-01-01', '2020-06-15', '2021-03-22'])
    assert original.tz is None
    index_ns = original.as_unit('ns').view(np.int64)
    atomic_save_npz(path, index=index_ns)

    loaded = np.load(str(path))
    reconstructed = pd.DatetimeIndex(loaded['index'])
    assert reconstructed.tz is None
    assert (reconstructed == original).all()


# --------------------------------------------------------------------------- #
# Metadata schema validation                                                   #
# --------------------------------------------------------------------------- #


def test_metadata_schema_version_field(tmp_path: Path) -> None:
    """Metadata with missing version field raises ValueError."""
    path = tmp_path / 'no_version.json'
    data = {"dataset_name": "ETTm1", "seq_len": 128}
    atomic_save_metadata(path, data)
    with pytest.raises(ValueError, match="Cache version"):
        load_metadata(path)

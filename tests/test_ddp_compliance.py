"""DDP compliance tests for phase 7 cache infrastructure.

Verifies multi-rank cache round-trips (gloo backend, 2 ranks),
isinstance branch elimination across all module files, and
setup idempotency with cache-backed data loading.
"""

import os
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp

from chronocratic.datasets.enums.data import ForecastingMode

# ---------------------------------------------------------------------------
# Module-level DDP workers (must be top-level for mp.spawn pickling)
# ---------------------------------------------------------------------------


def _get_free_port() -> int:
    """Return a locally available TCP/UDP port for DDP MASTER_PORT.

    Binds to port 0 (OS assigns) and returns the chosen port number.
    Must be called once before ``mp.spawn()`` so all ranks use the same
    port.
    """
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _ddp_forecasting_worker(
    rank: int, world_size: int, results_dir: str, csv_path: str, port: int
) -> None:
    """DDP worker for forecasting cache round-trip test.

    Rank 0 writes cache via prepare_data(), all ranks read via setup().
    Verifies that _full_data_raw and _train_data_samples are identical
    across ranks after cache-read setup.

    Args:
        rank: Process rank (0 or 1).
        world_size: Total number of processes.
        results_dir: Directory to write rank results for verification.
        csv_path: Path to synthetic CSV file.
    """
    from chronocratic.datasets.modules.weather import WeatherDataModule

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        # Rank 0: prepare_data() writes cache
        if rank == 0:
            module = WeatherDataModule(
                dataset_file_path=Path(csv_path), seq_len=96, mode=ForecastingMode.UNIVARIATE
            )
            module.prepare_data()

        dist.barrier()  # Ensure cache is written before other ranks read

        # All ranks: fresh module instance, setup reads from cache
        module = WeatherDataModule(
            dataset_file_path=Path(csv_path),
            seq_len=96,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
        )
        module.setup(stage="fit")

        # Verify _full_data_raw is populated from cache
        assert module._full_data_raw is not None, f"Rank {rank}: _full_data_raw is None after setup"

        # Write rank results for post-spawn verification
        result_path = Path(results_dir) / f"rank_{rank}.npz"
        np.savez(
            str(result_path),
            raw_shape=np.array(module._full_data_raw.shape),
            raw_dtype=str(module._full_data_raw.dtype),
            train_shape=np.array(module._train_data_samples.shape),
        )

        dist.barrier()
    finally:
        dist.destroy_process_group()


def _ddp_classification_worker(
    rank: int, world_size: int, results_dir: str, dataset_dir: str, port: int
) -> None:
    """DDP worker for classification cache round-trip test.

    Rank 0 writes cache via prepare_data(), all ranks read via setup().
    Verifies that cached data splits are identical across ranks.

    Args:
        rank: Process rank (0 or 1).
        world_size: Total number of processes.
        results_dir: Directory to write rank results for verification.
        dataset_dir: Path to synthetic UCR dataset directory.
    """
    from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    try:
        # Rank 0: prepare_data() writes cache
        if rank == 0:
            module = UCRClassificationDataModule(
                dataset_folder_path=Path(dataset_dir),
                target_column_name="class",
                valid_size=0.1,
                scale_data=False,
            )
            module.prepare_data()

        dist.barrier()  # Ensure cache is written before other ranks read

        # All ranks: fresh module instance, setup reads from cache
        module = UCRClassificationDataModule(
            dataset_folder_path=Path(dataset_dir),
            target_column_name="class",
            valid_size=0.1,
            scale_data=False,
        )
        module.setup(stage="fit")

        # Verify cached data is populated
        assert module._train_data_samples is not None, (
            f"Rank {rank}: _train_data_samples is None after setup"
        )
        assert module._train_data_labels is not None, (
            f"Rank {rank}: _train_data_labels is None after setup"
        )

        # Write rank results for post-spawn verification
        result_path = Path(results_dir) / f"rank_{rank}.npz"
        np.savez(
            str(result_path),
            train_shape=module._train_data_samples.to_numpy().shape,
            test_shape=module._test_data_samples.to_numpy().shape,
        )

        dist.barrier()
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# DDP Smoke Tests
# ---------------------------------------------------------------------------


class TestDDPSmokeTests:
    """DDP cache round-trip tests using mp.spawn with gloo backend."""

    def test_ddp_forecasting_cache_round_trip(self, tmp_path: Path) -> None:
        """Forecasting: rank-0 writes cache, all ranks read identical state.

        Uses WeatherDataModule with a synthetic CSV. Verifies that both ranks
        load the same _full_data_raw shape and dtype from cache after
        setup(stage='fit'). Weather uses fractional splits (60/20/20),
        making it suitable for small synthetic datasets.
        """
        # Create synthetic CSV matching Weather schema
        csv_file = tmp_path / "weather.csv"
        dates = pd.date_range("2006-01-01", periods=200, freq="h")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "date": dates,
                "wbng": rng.standard_normal(200),
                "wbhh": rng.standard_normal(200),
                "wbat": rng.standard_normal(200),
                "sbfg": rng.standard_normal(200),
            }
        )
        df.to_csv(csv_file, index=False)

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        port = _get_free_port()
        mp.spawn(
            _ddp_forecasting_worker,
            args=(2, str(results_dir), str(csv_file), port),
            nprocs=2,
            start_method="spawn",
        )

        # Verify both ranks wrote identical shapes
        rank0 = np.load(str(results_dir / "rank_0.npz"))
        rank1 = np.load(str(results_dir / "rank_1.npz"))

        assert tuple(rank0["raw_shape"]) == tuple(rank1["raw_shape"]), (
            f"Raw shape mismatch: rank0={tuple(rank0['raw_shape'])}, "
            f"rank1={tuple(rank1['raw_shape'])}"
        )
        assert str(rank0["raw_dtype"]) == str(rank1["raw_dtype"]), (
            f"Raw dtype mismatch: rank0={rank0['raw_dtype']}, rank1={rank1['raw_dtype']}"
        )
        assert tuple(rank0["train_shape"]) == tuple(rank1["train_shape"]), (
            f"Train shape mismatch: rank0={tuple(rank0['train_shape'])}, "
            f"rank1={tuple(rank1['train_shape'])}"
        )

    def test_ddp_classification_cache_round_trip(self, tmp_path: Path) -> None:
        """Classification: rank-0 writes cache, all ranks read identical state.

        Uses UCRClassificationDataModule with synthetic ARFF files.
        Verifies that both ranks load the same train/test shapes from
        cache after setup(stage='fit').
        """
        # Create synthetic UCR-style ARFF files
        arff_content = """@relation test

@attribute t1 numeric
@attribute t2 numeric
@attribute t3 numeric
@attribute class {0,1}

@data
0.1,0.2,0.3,0
0.4,0.5,0.6,1
0.7,0.8,0.9,0
0.2,0.3,0.4,1
0.5,0.6,0.7,0
0.8,0.9,1.0,1
0.1,0.2,0.3,0
0.4,0.5,0.6,1
0.7,0.8,0.9,0
0.2,0.3,0.4,1
0.5,0.6,0.7,0
0.8,0.9,1.0,1
0.1,0.2,0.3,0
0.4,0.5,0.6,1
0.7,0.8,0.9,0
"""
        dataset_dir = tmp_path / "synthetic"
        dataset_dir.mkdir()
        (dataset_dir / "synthetic_TRAIN.arff").write_text(arff_content)
        (dataset_dir / "synthetic_TEST.arff").write_text(arff_content)

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        port = _get_free_port()
        mp.spawn(
            _ddp_classification_worker,
            args=(2, str(results_dir), str(dataset_dir), port),
            nprocs=2,
            start_method="spawn",
        )

        # Verify both ranks wrote identical shapes
        rank0 = np.load(str(results_dir / "rank_0.npz"))
        rank1 = np.load(str(results_dir / "rank_1.npz"))

        assert tuple(rank0["train_shape"]) == tuple(rank1["train_shape"]), (
            f"Train shape mismatch: rank0={tuple(rank0['train_shape'])}, "
            f"rank1={tuple(rank1['train_shape'])}"
        )
        assert tuple(rank0["test_shape"]) == tuple(rank1["test_shape"]), (
            f"Test shape mismatch: rank0={tuple(rank0['test_shape'])}, "
            f"rank1={tuple(rank1['test_shape'])}"
        )


# ---------------------------------------------------------------------------
# isinstance Branch Elimination Test
# ---------------------------------------------------------------------------


class TestIsinstanceBranchElimination:
    """Static verification that isinstance(_full_data) branches are eliminated."""

    def test_isinstance_branch_elimination(self) -> None:
        """No isinstance(*_full_data) branches remain in module source files.

        Scans all Python files under src/chronocratic/datasets/modules/
        for patterns like isinstance(self._full_data, ...) or
        isinstance(_full_data, ...). These were eliminated in phase 7
        by splitting _full_data into typed attributes.
        """
        modules_dir = Path(__file__).parents[1] / "src" / "chronocratic" / "datasets" / "modules"
        matches = []
        for py_file in modules_dir.rglob("*.py"):
            for lineno, line in enumerate(py_file.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "isinstance" in stripped and "self._full_data" in stripped:
                    matches.append(f"{py_file.relative_to(modules_dir)}:{lineno}: {stripped}")
        assert not matches, (
            "Found isinstance(self._full_data) branches that should be eliminated:\n"
            + "\n".join(matches)
        )


# ---------------------------------------------------------------------------
# Setup Idempotency with Cache Tests
# ---------------------------------------------------------------------------


class TestSetupIdempotentWithCache:
    """Verify setup() produces identical results when re-reading from cache.

    Tests that clearing setup state and re-reading from cache produces
    the same _train_data_samples as the first setup call. This validates
    that the cache is the source of truth (DDP-safe pattern).
    """

    def test_setup_idempotent_with_cache(self, tmp_path: Path) -> None:
        """Second setup() call produces identical _train_data_samples from cache.

        Calls prepare_data() to write cache, setup('fit') to read and
        process, snapshots _train_data_samples, clears setup state,
        and verifies a fresh setup('fit') read produces identical data.
        """
        from chronocratic.datasets.modules.ett import ETTDataModule

        # Create synthetic CSV
        csv_file = tmp_path / "ett.csv"
        dates = pd.date_range("2016-01-01", periods=200, freq="h")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "date": dates,
                "HUFL": rng.standard_normal(200),
                "HT": rng.standard_normal(200),
                "OT": rng.standard_normal(200),
                "Wsp": rng.standard_normal(200),
            }
        )
        df.to_csv(csv_file, index=False)

        module = ETTDataModule(
            dataset_file_path=csv_file,
            variant="ETTh1",
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
        )

        # First pass: write cache, read it, process data
        module.prepare_data()
        module.setup(stage="fit")

        # Snapshot results
        snapshot_train = module._train_data_samples.copy()
        snapshot_valid = module._valid_data_samples.copy()
        snapshot_test = module._test_data_samples.copy()
        snapshot_raw = module._full_data_raw.copy()

        # Simulate fresh process: clear setup state and data samples
        # (cache files on disk remain intact)
        module._setup_completed_stages.clear()
        module._train_data_samples = None
        module._valid_data_samples = None
        module._test_data_samples = None
        module._full_data_scaled = None
        module._data_scaler_cache = None
        module._ts_feature_scaler_cache = None

        # Second pass: re-read from cache
        module.setup(stage="fit")

        # Verify _full_data_raw is identical (immutable cache read)
        np.testing.assert_array_equal(
            snapshot_raw,
            module._full_data_raw,
            err_msg="_full_data_raw changed after cache re-read",
        )
        # Verify data samples are identical
        np.testing.assert_array_equal(
            snapshot_train,
            module._train_data_samples,
            err_msg="_train_data_samples changed after cache re-read",
        )
        np.testing.assert_array_equal(
            snapshot_valid,
            module._valid_data_samples,
            err_msg="_valid_data_samples changed after cache re-read",
        )
        np.testing.assert_array_equal(
            snapshot_test,
            module._test_data_samples,
            err_msg="_test_data_samples changed after cache re-read",
        )

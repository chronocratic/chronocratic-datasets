"""Tests for opt-in stateful dataloaders on the base datamodule."""

from __future__ import annotations

import inspect
import subprocess
import sys
from typing import cast, TYPE_CHECKING

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from chronocratic.datasets.enums.data import ForecastingLoaderMode
from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule
from chronocratic.datasets.modules.electricity import ElectricityLoadDataModule
from chronocratic.datasets.modules.ett import ETTDataModule
from chronocratic.datasets.modules.ucr import UCRClassificationDataModule
from chronocratic.datasets.modules.uea import UEAClassificationDataModule
from chronocratic.datasets.modules.weather import WeatherDataModule

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def synthetic_ett_csv(tmp_path: Path) -> Path:
    """Create a synthetic ETT-style CSV with enough data for all slices."""
    csv_path = tmp_path / "synthetic_ett.csv"
    dates = pd.date_range("2016-01-01", periods=12_000, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "date": dates,
            "HUFL": rng.standard_normal(12_000).astype(np.float32),
            "HULL": rng.standard_normal(12_000).astype(np.float32),
            "MUFL": rng.standard_normal(12_000).astype(np.float32),
            "MULL": rng.standard_normal(12_000).astype(np.float32),
            "OT": rng.standard_normal(12_000).astype(np.float32),
            "T1": rng.standard_normal(12_000).astype(np.float32),
            "T2": rng.standard_normal(12_000).astype(np.float32),
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_ucr_folder(tmp_path: Path) -> Path:
    """Create a synthetic UCR dataset folder with ARFF files."""
    arff_content = """\
@relation test

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
    dataset_dir = tmp_path / "synthetic_ucr"
    dataset_dir.mkdir()
    (dataset_dir / "synthetic_ucr_TRAIN.arff").write_text(arff_content)
    (dataset_dir / "synthetic_ucr_TEST.arff").write_text(arff_content)
    return dataset_dir


class _StubDataModule(BaseTimeSeriesDataModule):
    def _do_prepare_data(self) -> None:
        pass


def _make_stub(*, loader_use_stateful: bool = False, batch_size: int = 4) -> _StubDataModule:
    return _StubDataModule(
        batch_size=batch_size,
        seq_len=None,
        valid_size=0.2,
        test_size=0.2,
        shuffle=False,
        scale_data=False,
        loader_use_stateful=loader_use_stateful,
    )


def _dataset() -> TensorDataset:
    return TensorDataset(torch.arange(20).float().unsqueeze(1))


class TestBaseLoaderUseStateful:
    def test_default_is_false_and_plain_loader(self) -> None:
        mod = _make_stub()
        assert mod.loader_use_stateful is False
        loader = mod._process_train_dataloader(dataset_object=_dataset())
        assert type(loader) is DataLoader

    def test_init_true_returns_stateful(self) -> None:
        mod = _make_stub(loader_use_stateful=True)
        assert isinstance(
            mod._process_train_dataloader(dataset_object=_dataset()), StatefulDataLoader
        )
        assert isinstance(
            mod._process_test_dataloader(dataset_object=_dataset()), StatefulDataLoader
        )
        assert isinstance(
            mod._process_valid_dataloader(dataset_object=_dataset()), StatefulDataLoader
        )

    def test_per_call_false_overrides_init_true(self) -> None:
        mod = _make_stub(loader_use_stateful=True)
        loader = mod._process_train_dataloader(dataset_object=_dataset(), loader_use_stateful=False)
        assert type(loader) is DataLoader

    def test_per_call_true_overrides_init_false(self) -> None:
        mod = _make_stub()
        loader = mod._process_test_dataloader(dataset_object=_dataset(), loader_use_stateful=True)
        assert isinstance(loader, StatefulDataLoader)

    @pytest.mark.parametrize(("bad", "type_name"), [("false", "str"), (1, "int")])
    def test_constructor_rejects_non_bool(self, bad: object, type_name: str) -> None:
        with pytest.raises(TypeError) as exc:
            _make_stub(loader_use_stateful=cast("bool", bad))

        assert str(exc.value) == f"loader_use_stateful must be bool, got {type_name}"

    @pytest.mark.parametrize(("bad", "type_name"), [("false", "str"), (1, "int")])
    def test_per_call_rejects_non_bool(self, bad: object, type_name: str) -> None:
        mod = _make_stub()
        with pytest.raises(TypeError) as exc:
            mod._process_train_dataloader(
                dataset_object=_dataset(),
                loader_use_stateful=cast("bool", bad),
            )

        assert str(exc.value) == f"loader_use_stateful must be bool, got {type_name}"

    def test_setter_rejects_non_bool(self) -> None:
        mod = _make_stub()
        attribute_name = "loader_use_stateful"
        for bad in ("true", 1, None):
            with pytest.raises(TypeError, match="loader_use_stateful must be bool"):
                setattr(mod, attribute_name, bad)

    def test_missing_torchdata_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torchdata.stateful_dataloader", None)
        mod = _make_stub(loader_use_stateful=True)
        with pytest.raises(
            ImportError,
            match=r"chronocratic-datasets\[stateful\]",
        ):
            mod._process_train_dataloader(dataset_object=_dataset())

    def test_missing_torchdata_ignored_when_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torchdata.stateful_dataloader", None)
        loader = _make_stub()._process_train_dataloader(dataset_object=_dataset())
        assert type(loader) is DataLoader

    def test_stateful_keeps_strict_collate(self) -> None:
        mod = _make_stub(loader_use_stateful=True, batch_size=6)
        batches = list(
            mod._process_test_dataloader(dataset_object=_dataset(), loader_strict_batch_size=True)
        )
        assert all(batch[0].shape[0] == 6 for batch in batches)

    def test_mid_epoch_round_trip(self) -> None:
        mod = _make_stub(loader_use_stateful=True)
        loader = mod._process_test_dataloader(dataset_object=_dataset())
        assert isinstance(loader, StatefulDataLoader)
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        state = loader.state_dict()
        expected = [batch[0] for batch in iterator]

        resumed = mod._process_test_dataloader(dataset_object=_dataset())
        assert isinstance(resumed, StatefulDataLoader)
        resumed.load_state_dict(state)
        assert all(
            torch.equal(expected_batch, batch[0])
            for expected_batch, batch in zip(expected, resumed, strict=True)
        )

    def test_shuffled_resume_matches_uninterrupted(self) -> None:
        mod = _make_stub(loader_use_stateful=True)
        torch.manual_seed(0)
        loader = mod._process_train_dataloader(dataset_object=_dataset(), shuffle=True)
        assert isinstance(loader, StatefulDataLoader)
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        state = loader.state_dict()
        expected = [batch[0] for batch in iterator]

        torch.manual_seed(0)
        resumed = mod._process_train_dataloader(dataset_object=_dataset(), shuffle=True)
        assert isinstance(resumed, StatefulDataLoader)
        resumed.load_state_dict(state)
        assert all(
            torch.equal(expected_batch, batch[0])
            for expected_batch, batch in zip(expected, resumed, strict=True)
        )

    def test_import_does_not_load_torchdata(self) -> None:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, chronocratic.datasets; assert 'torchdata' not in sys.modules",
            ],
            check=True,
        )


MODULES = (
    ETTDataModule,
    WeatherDataModule,
    ElectricityLoadDataModule,
    UCRClassificationDataModule,
    UEAClassificationDataModule,
)


@pytest.mark.parametrize("module_cls", MODULES)
def test_init_signature_has_flag(module_cls: type) -> None:
    parameter = inspect.signature(module_cls.__init__).parameters["loader_use_stateful"]
    assert parameter.default is False


@pytest.mark.parametrize("module_cls", MODULES)
@pytest.mark.parametrize("method", ["train_dataloader", "val_dataloader", "test_dataloader"])
def test_method_signature_has_override(module_cls: type, method: str) -> None:
    parameter = inspect.signature(getattr(module_cls, method)).parameters["loader_use_stateful"]
    assert parameter.default is None


def _make_ett(
    dataset_file_path: Path, *, loader_use_stateful: bool = True, num_workers: int = 0
) -> ETTDataModule:
    return ETTDataModule(
        dataset_file_path=dataset_file_path,
        variant="ETTh1",
        seq_len=24,
        forecast_horizon=12,
        batch_size=4,
        num_workers=num_workers,
        scale_data=False,
        loader_mode=ForecastingLoaderMode.INPUT_TARGET,
        loader_use_stateful=loader_use_stateful,
    )


def test_ett_init_flag_reaches_all_loaders(synthetic_ett_csv: Path) -> None:
    mod = _make_ett(synthetic_ett_csv)
    mod.prepare_data()
    mod.setup(stage="fit")
    mod.setup(stage="test")

    assert isinstance(mod.train_dataloader(), StatefulDataLoader)
    assert isinstance(mod.val_dataloader(), StatefulDataLoader)
    assert isinstance(mod.test_dataloader(), StatefulDataLoader)


def test_ett_per_call_override(synthetic_ett_csv: Path) -> None:
    mod = _make_ett(synthetic_ett_csv)
    mod.prepare_data()
    mod.setup(stage="fit")

    assert type(mod.train_dataloader(loader_use_stateful=False)) is DataLoader


def test_ucr_init_flag_and_override(synthetic_ucr_folder: Path) -> None:
    mod = UCRClassificationDataModule(
        dataset_folder_path=synthetic_ucr_folder,
        target_column_name="class",
        loader_use_stateful=True,
    )
    mod.prepare_data()
    mod.setup(stage="fit")

    assert isinstance(mod.train_dataloader(), StatefulDataLoader)
    assert type(mod.test_dataloader(loader_use_stateful=False)) is DataLoader


def test_resume_with_workers(synthetic_ett_csv: Path) -> None:
    mod = _make_ett(synthetic_ett_csv, num_workers=2)
    mod.prepare_data()
    mod.setup(stage="fit")
    loader = mod.train_dataloader()
    assert isinstance(loader, StatefulDataLoader)
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    state = loader.state_dict()
    expected = [batch[0] for batch in iterator]

    resumed = mod.train_dataloader()
    assert isinstance(resumed, StatefulDataLoader)
    resumed.load_state_dict(state)
    assert all(
        torch.equal(expected_batch, batch[0])
        for expected_batch, batch in zip(expected, resumed, strict=True)
    )


def test_lightning_reinstantiation_keeps_stateful_class(synthetic_ett_csv: Path) -> None:
    from lightning.pytorch.utilities.data import _update_dataloader
    from torch.utils.data import DistributedSampler

    mod = _make_ett(synthetic_ett_csv)
    mod.prepare_data()
    mod.setup(stage="fit")
    loader = mod.train_dataloader()
    sampler = DistributedSampler(loader.dataset, num_replicas=2, rank=0)
    new_loader = _update_dataloader(
        loader,
        sampler=sampler,
    )

    assert isinstance(new_loader, StatefulDataLoader)
    assert isinstance(new_loader.batch_sampler, BatchSampler)
    assert new_loader.batch_sampler.sampler is sampler


class _TinyForecastingModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(24, 12)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        predictions = self.linear(inputs.flatten(start_dim=1))
        return nn.functional.mse_loss(predictions, targets.flatten(start_dim=1))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_trainer_checkpoint_stores_loader_state(synthetic_ett_csv: Path, tmp_path: Path) -> None:
    mod = _make_ett(synthetic_ett_csv)
    trainer = pl.Trainer(
        max_steps=3,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
        default_root_dir=tmp_path,
    )
    trainer.fit(_TinyForecastingModule(), datamodule=mod)
    checkpoint_path = tmp_path / "c.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    assert checkpoint["loops"]["fit_loop"]["state_dict"]["combined_loader"]

"""Tests for opt-in stateful dataloaders on the base datamodule."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule


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

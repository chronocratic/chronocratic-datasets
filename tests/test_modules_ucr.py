"""Tests for UCRClassificationDataModule."""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from chronocratic.datasets.enums.data import (
    ClassificationLoaderMode,
    ClassificationSplitMode,
    DataForm,
)


def _make_arff(*, rows: list[tuple[float, float, float, int | str]], class_values: str) -> str:
    """Build ARFF text with three numeric features and a nominal target column."""
    header = (
        "@relation test\n\n"
        "@attribute t1 numeric\n@attribute t2 numeric\n@attribute t3 numeric\n"
        f"@attribute class {{{class_values}}}\n\n@data\n"
    )
    body = "\n".join(f"{t1},{t2},{t3},{label}" for t1, t2, t3, label in rows)
    return header + body + "\n"


class TestUCRClassificationDataModule:
    """Tests for UCRClassificationDataModule."""

    @pytest.fixture
    def module_class(self):
        """Lazy-import the module class."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        return UCRClassificationDataModule

    def test_constructor_accepts_path_and_target(self, module_class: type) -> None:
        """UCRClassificationDataModule accepts dataset_folder_path (Path) and target_column_name."""
        import inspect

        sig = inspect.signature(module_class.__init__)
        assert "dataset_folder_path" in sig.parameters
        assert "target_column_name" in sig.parameters

    def test_data_form_is_regular(self, module_class: type, tmp_path: Path) -> None:
        """data_form is hardcoded to DataForm.REGULAR."""
        mod = module_class(dataset_folder_path=tmp_path, target_column_name="class")
        assert mod._data_form == DataForm.REGULAR

    def test_prepare_data_raises_for_missing_folder(self, module_class: type) -> None:
        """prepare_data() raises FileNotFoundError for non-existent folder."""
        mod = module_class(
            dataset_folder_path=Path("/nonexistent/path"), target_column_name="class"
        )
        with pytest.raises(FileNotFoundError):
            mod.prepare_data()

    def test_extends_classification_base(self, module_class: type) -> None:
        """UCRClassificationDataModule extends BaseClassificationTimeSeriesDataModule."""
        from chronocratic.datasets.modules._base.classification import (
            BaseClassificationTimeSeriesDataModule,
        )

        assert issubclass(module_class, BaseClassificationTimeSeriesDataModule)

    def test_splitting_strategy_default(self, module_class: type, tmp_path: Path) -> None:
        """splitting_strategy defaults to AS_DEFINED."""
        mod = module_class(dataset_folder_path=tmp_path, target_column_name="class")
        assert mod.splitting_strategy == ClassificationSplitMode.AS_DEFINED

    def test_has_dataloader_methods(self, module_class: type) -> None:
        """Module has train_dataloader, val_dataloader, test_dataloader."""
        assert hasattr(module_class, "train_dataloader")
        assert hasattr(module_class, "val_dataloader")
        assert hasattr(module_class, "test_dataloader")

    @pytest.fixture
    def synthetic_ucr_folder(self, tmp_path: Path) -> Path:
        """Create a synthetic UCR dataset folder with ARFF files."""
        # Create simple ARFF content
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
        # Create a named subdirectory — the module derives dataset_name from this
        dataset_dir = tmp_path / "synthetic"
        dataset_dir.mkdir()
        (dataset_dir / "synthetic_TRAIN.arff").write_text(arff_content)
        (dataset_dir / "synthetic_TEST.arff").write_text(arff_content)
        return dataset_dir

    def test_prepare_data_loads_arff(self, module_class: type, synthetic_ucr_folder: Path) -> None:
        """prepare_data() reads ARFF files and sets internal state."""
        mod = module_class(
            dataset_folder_path=synthetic_ucr_folder,
            target_column_name="class",
            valid_size=0.1,
            splitting_strategy=ClassificationSplitMode.AS_DEFINED,
        )
        mod.prepare_data()
        mod.setup("fit")

        assert mod.name == "synthetic"
        assert mod.num_classes is not None
        assert mod.sequence_length == 3  # t1, t2, t3
        assert mod.num_features == 1
        assert mod.train_data_samples is not None
        assert mod.test_data_samples is not None
        assert mod.train_data_labels is not None

    def test_train_dataloader_returns_dataloader(
        self, module_class: type, synthetic_ucr_folder: Path
    ) -> None:
        """train_dataloader() returns a DataLoader instance."""
        mod = module_class(
            dataset_folder_path=synthetic_ucr_folder, target_column_name="class", valid_size=0.1
        )
        mod.prepare_data()
        mod.setup("fit")

        dl = mod.train_dataloader(loader_mode=ClassificationLoaderMode.SAMPLE_LABEL)
        assert isinstance(dl, DataLoader)

    def test_test_dataloader_returns_dataloader(
        self, module_class: type, synthetic_ucr_folder: Path
    ) -> None:
        """test_dataloader() returns a DataLoader instance."""
        mod = module_class(
            dataset_folder_path=synthetic_ucr_folder, target_column_name="class", valid_size=0.1
        )
        mod.prepare_data()
        mod.setup("fit")

        dl = mod.test_dataloader(loader_mode=ClassificationLoaderMode.SAMPLE_LABEL)
        assert isinstance(dl, DataLoader)

    def test_val_dataloader_returns_dataloader_or_none(
        self, module_class: type, synthetic_ucr_folder: Path
    ) -> None:
        """val_dataloader() returns DataLoader when valid_size > 0, None otherwise."""
        mod = module_class(
            dataset_folder_path=synthetic_ucr_folder, target_column_name="class", valid_size=0.1
        )
        mod.prepare_data()
        mod.setup("fit")

        dl = mod.val_dataloader(loader_mode=ClassificationLoaderMode.SAMPLE_LABEL)
        assert dl is None or isinstance(dl, DataLoader)

        # Test with valid_size=0
        mod_no_val = module_class(
            dataset_folder_path=synthetic_ucr_folder, target_column_name="class", valid_size=0.0
        )
        mod_no_val.prepare_data()
        mod_no_val.setup("fit")
        dl_none = mod_no_val.val_dataloader(loader_mode=ClassificationLoaderMode.SAMPLE_LABEL)
        assert dl_none is None


def test_setup_idempotent(tmp_path: Path) -> None:
    """UCR: setup('fit') called twice produces identical train samples.

    Creates a synthetic UCR folder with ARFF files, calls prepare_data() +
    setup('fit') to load and scale data, snapshots the train samples, then
    calls setup('fit') again and asserts data is unchanged (sentinel guard).
    """
    from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

    # Create synthetic UCR folder
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

    mod = UCRClassificationDataModule(
        dataset_folder_path=dataset_dir, target_column_name="class", valid_size=0.1
    )
    mod.prepare_data()
    mod.setup(stage="fit")

    snapshot_train = mod.train_data_samples.copy()
    mod.setup(stage="fit")

    pd.testing.assert_frame_equal(snapshot_train, mod.train_data_samples)


def test_setup_unknown_stage_raises(tmp_path: Path) -> None:
    """UCR: setup() raises ValueError for unknown stage.

    Verifies stage validation in the base class propagates to concrete modules.
    """
    from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

    dataset_dir = tmp_path / "synthetic"
    dataset_dir.mkdir()
    (dataset_dir / "synthetic_TRAIN.arff").write_text("")
    (dataset_dir / "synthetic_TEST.arff").write_text("")

    mod = UCRClassificationDataModule(
        dataset_folder_path=dataset_dir, target_column_name="class", valid_size=0.1
    )
    with pytest.raises(ValueError, match=r"Unknown stage"):
        mod.setup(stage="warmup")  # type: ignore[arg-type]


def test_setup_fit_then_test_reuses_scaler(tmp_path: Path) -> None:
    """UCR: setup('fit') then setup('test') creates scaler only once."""
    from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

    # Create synthetic UCR folder
    arff_content = """@relation test

@attribute t1 numeric
@attribute t2 numeric
@attribute class {0,1}

@data
0.1,0.2,0
0.4,0.5,1
0.7,0.8,0
0.2,0.3,1
0.5,0.6,0
0.8,0.9,1
"""
    dataset_dir = tmp_path / "synthetic"
    dataset_dir.mkdir()
    (dataset_dir / "synthetic_TRAIN.arff").write_text(arff_content)
    (dataset_dir / "synthetic_TEST.arff").write_text(arff_content)

    call_count = 0

    def counting_scaler(**_kwargs):
        nonlocal call_count
        call_count += 1
        return lambda t, v, te: (t, v, te)

    mod = UCRClassificationDataModule(
        dataset_folder_path=dataset_dir, target_column_name="class", valid_size=0.1
    )
    mod.prepare_data()
    with patch(
        "chronocratic.datasets.modules._base.base.create_data_scaler", side_effect=counting_scaler
    ):
        mod.setup(stage="fit")
        assert mod._scaler_cache is not None
        mod.setup(stage="test")
        assert call_count == 1


def test_cache_round_trip(tmp_path: Path) -> None:
    """UCR: prepare_data writes cache, setup reads it back correctly.

    Verifies the cache round-trip: after prepare_data() writes the npz
    and metadata.json, clearing in-memory state and calling setup()
    restores data from cache that matches the original.
    """
    from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

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

    mod = UCRClassificationDataModule(
        dataset_folder_path=dataset_dir,
        target_column_name="class",
        valid_size=0.1,
        scale_data=False,
    )
    mod.prepare_data()

    # Snapshot original data
    orig_train = mod._train_data_samples.copy()
    orig_test = mod._test_data_samples.copy()
    orig_train_labels = mod._train_data_labels.copy()
    orig_test_labels = mod._test_data_labels.copy()

    # Clear in-memory state (simulates fresh process reading from cache)
    mod._train_data_samples = None
    mod._test_data_samples = None
    mod._valid_data_samples = None
    mod._train_data_labels = None
    mod._test_data_labels = None
    mod._valid_data_labels = None

    # setup() should read from cache
    mod.setup(stage="fit")

    # Verify data matches (before scaling since scale_data=False)
    np.testing.assert_array_equal(mod._train_data_samples.to_numpy(), orig_train.to_numpy())
    np.testing.assert_array_equal(mod._test_data_samples.to_numpy(), orig_test.to_numpy())
    np.testing.assert_array_equal(mod._train_data_labels.to_numpy(), orig_train_labels.to_numpy())
    np.testing.assert_array_equal(mod._test_data_labels.to_numpy(), orig_test_labels.to_numpy())


# --------------------------------------------------------------------------- #
# Joint label encoding tests                                                  #
# --------------------------------------------------------------------------- #


def test_negative_binary_labels_encode_to_0_1(tmp_path: Path) -> None:
    """{-1,1} labels: num_classes == 2 and every split's labels are in {0, 1}."""
    from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

    rows: list[tuple[float, float, float, int | str]] = [
        (0.1 * i, 0.2 * i, 0.3 * i, -1 if i % 2 == 0 else 1) for i in range(1, 21)
    ]
    arff_content = _make_arff(rows=rows, class_values="-1,1")

    dataset_dir = tmp_path / "synthetic"
    dataset_dir.mkdir()
    (dataset_dir / "synthetic_TRAIN.arff").write_text(arff_content)
    (dataset_dir / "synthetic_TEST.arff").write_text(arff_content)

    mod = UCRClassificationDataModule(
        dataset_folder_path=dataset_dir, target_column_name="class", valid_size=0.1
    )
    mod.prepare_data()
    mod.setup("fit")

    assert mod.num_classes == 2
    for labels in (mod._train_data_labels, mod._valid_data_labels, mod._test_data_labels):
        if labels is not None:
            assert set(labels.unique()) <= {0, 1}

    metadata_path = mod._get_cache_dir() / f"{mod._cache_key}_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["num_classes"] == 2
    assert metadata["class_labels"] == ["-1", "1"]
    assert "label_encoding=joint_label_encoder_v1" in mod._cache_key


def test_class_absent_from_train_present_in_test(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Phoneme-like case: a class dropped as a train singleton stays in K and in test."""
    from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

    train_rows: list[tuple[float, float, float, int | str]] = [
        (0.1 * i, 0.2 * i, 0.3 * i, 1) for i in range(1, 11)
    ]
    train_rows += [(0.1 * i, 0.2 * i, 0.3 * i, 2) for i in range(1, 10)]
    train_rows += [(0.99, 0.98, 0.97, 3)]  # singleton class, dropped by the valid-split filter
    test_rows: list[tuple[float, float, float, int | str]] = [
        (0.1 * i, 0.2 * i, 0.3 * i, 1) for i in range(1, 6)
    ]
    test_rows += [(0.1 * i, 0.2 * i, 0.3 * i, 2) for i in range(1, 6)]
    test_rows += [(0.1 * i, 0.2 * i, 0.3 * i, 3) for i in range(1, 4)]

    dataset_dir = tmp_path / "synthetic"
    dataset_dir.mkdir()
    (dataset_dir / "synthetic_TRAIN.arff").write_text(
        _make_arff(rows=train_rows, class_values="1,2,3")
    )
    (dataset_dir / "synthetic_TEST.arff").write_text(
        _make_arff(rows=test_rows, class_values="1,2,3")
    )

    mod = UCRClassificationDataModule(
        dataset_folder_path=dataset_dir, target_column_name="class", valid_size=0.1
    )
    with caplog.at_level(logging.WARNING):
        mod.prepare_data()
    mod.setup("fit")

    assert mod.num_classes == 3
    # Class "3" -> code 2 (LabelEncoder sorts 1,2,3 -> 0,1,2)
    assert 2 in set(mod._test_data_labels.unique())
    assert 2 not in set(mod._train_data_labels.unique())
    assert any("absent from train" in message for message in caplog.messages)


def test_cache_round_trip_restores_num_classes(tmp_path: Path) -> None:
    """num_classes is restored from cache metadata after clearing in-memory state."""
    from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

    rows: list[tuple[float, float, float, int | str]] = [
        (0.1 * i, 0.2 * i, 0.3 * i, i % 2) for i in range(1, 21)
    ]
    arff_content = _make_arff(rows=rows, class_values="0,1")

    dataset_dir = tmp_path / "synthetic"
    dataset_dir.mkdir()
    (dataset_dir / "synthetic_TRAIN.arff").write_text(arff_content)
    (dataset_dir / "synthetic_TEST.arff").write_text(arff_content)

    mod = UCRClassificationDataModule(
        dataset_folder_path=dataset_dir, target_column_name="class", valid_size=0.1
    )
    mod.prepare_data()

    orig_labels = mod._train_data_labels.copy()
    mod._num_classes = None
    mod._train_data_samples = None
    mod._test_data_samples = None
    mod._valid_data_samples = None
    mod._train_data_labels = None
    mod._test_data_labels = None
    mod._valid_data_labels = None

    mod.setup(stage="fit")

    assert mod.num_classes == 2
    np.testing.assert_array_equal(mod._train_data_labels.to_numpy(), orig_labels.to_numpy())

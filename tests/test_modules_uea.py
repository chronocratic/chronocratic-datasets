"""Tests for UEAClassificationDataModule.

Covers constructor params, DataForm.NESTED, _process_stacked_data
byte-decoding and LabelEncoder, FileNotFoundError for missing folder,
and dataloader methods returning proper DataLoaders.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from chronocratic.datasets.enums.data import (
    ClassificationLoaderMode,
    ClassificationSplitMode,
    DataForm,
    ScalingMethod,
)


@pytest.fixture
def synthetic_uea_folder(tmp_path: Path) -> Path:
    """Create a minimal UEA-style folder for testing.

    Creates a named folder with placeholder ARFF files. The actual
    ARFF content is mocked in tests that exercise prepare_data()
    since scipy struggles with nested ARFF in test fixtures.
    """
    dataset_name = "synthetic_uea"
    folder = tmp_path / dataset_name
    folder.mkdir(parents=True, exist_ok=True)

    # Placeholders — real content injected via mocks in prepare_data tests
    train_arff = folder / f"{dataset_name}_TRAIN.arff"
    train_arff.write_text("@relation placeholder\n@data\n")
    test_arff = folder / f"{dataset_name}_TEST.arff"
    test_arff.write_text("@relation placeholder\n@data\n")

    return folder


def _make_mock_train_data() -> np.ndarray:
    """Create mock nested ARFF data for train split.

    Returns structured array mimicking scipy.loadarff output with
    5 samples of shape (3 timesteps, 2 features) and labels.
    """
    samples = [
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        np.array([[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]),
        np.array([[19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]),
        np.array([[25.0, 26.0], [27.0, 28.0], [29.0, 30.0]]),
    ]
    labels = [b"0", b"1", b"0", b"1", b"0"]
    data = np.array(list(zip(samples, labels, strict=False)), dtype=[("f0", "O"), ("f1", "O")])
    return data


def _make_mock_test_data() -> np.ndarray:
    """Create mock nested ARFF data for test split.

    Returns structured array mimicking scipy.loadarff output with
    2 samples of shape (3 timesteps, 2 features) and labels.
    """
    samples = [
        np.array([[31.0, 32.0], [33.0, 34.0], [35.0, 36.0]]),
        np.array([[37.0, 38.0], [39.0, 40.0], [41.0, 42.0]]),
    ]
    labels = [b"0", b"1"]
    data = np.array(list(zip(samples, labels, strict=False)), dtype=[("f0", "O"), ("f1", "O")])
    return data


class TestUEAClassificationDataModuleConstructor:
    """Tests for UEAClassificationDataModule constructor."""

    def test_import_uea_module(self) -> None:
        """UEAClassificationDataModule can be imported from uea module."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        assert UEAClassificationDataModule is not None

    def test_constructor_accepts_params(self, synthetic_uea_folder: Path) -> None:
        """Constructor accepts dataset_folder_path, target_column_name, and config params."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        module = UEAClassificationDataModule(
            dataset_folder_path=synthetic_uea_folder,
            target_column_name="class",
            batch_size=16,
            valid_size=0.2,
            shuffle=True,
            scale_data=False,
            data_scaling_method=ScalingMethod.STANDARD,
            data_scaling_range=(0, 1),
            splitting_strategy=ClassificationSplitMode.AS_DEFINED,
            test_size=0.3,
            num_workers=2,
        )
        assert module.batch_size == 16
        assert module.valid_size == 0.2
        assert module.shuffle is True
        assert module.scale_data is False
        assert module.data_scaling_method == ScalingMethod.STANDARD
        assert module.splitting_strategy == ClassificationSplitMode.AS_DEFINED
        assert module.num_workers == 2

    def test_data_form_is_nested(self, synthetic_uea_folder: Path) -> None:
        """data_form is hardcoded to DataForm.NESTED."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        module = UEAClassificationDataModule(
            dataset_folder_path=synthetic_uea_folder, target_column_name="class"
        )
        assert module._data_form == DataForm.NESTED


class TestUEAProcessStackedData:
    """Tests for _process_stacked_data method."""

    def test_process_stacked_data_returns_tuple(self, synthetic_uea_folder: Path) -> None:
        """_process_stacked_data returns (np.ndarray, np.ndarray) tuple."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        module = UEAClassificationDataModule(
            dataset_folder_path=synthetic_uea_folder, target_column_name="class"
        )
        # Build mock nested data that scipy.loadarff would return
        # Structure: each row is (sample_array, label)
        sample1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        sample2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        mock_data = np.array([(sample1, b"0"), (sample2, b"1")], dtype=[("f0", "O"), ("f1", "O")])

        samples, labels = module._process_stacked_data(mock_data)

        assert isinstance(samples, np.ndarray)
        assert isinstance(labels, np.ndarray)
        # Shape: (samples, features, timesteps) -> (samples, timesteps, features)
        assert samples.shape[0] == 2

    def test_process_stacked_data_decodes_bytes(self, synthetic_uea_folder: Path) -> None:
        """_process_stacked_data handles byte-decoded labels."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        module = UEAClassificationDataModule(
            dataset_folder_path=synthetic_uea_folder, target_column_name="class"
        )
        sample1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        sample2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        mock_data = np.array([(sample1, b"A"), (sample2, b"B")], dtype=[("f0", "O"), ("f1", "O")])

        _samples, labels = module._process_stacked_data(mock_data)

        # Labels should be encoded integers (0, 1)
        assert set(labels.tolist()).issubset({0, 1})


class TestUEAPrepareData:
    """Tests for prepare_data method."""

    def test_prepare_data_raises_file_not_found(self) -> None:
        """prepare_data raises FileNotFoundError for missing folder."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        module = UEAClassificationDataModule(
            dataset_folder_path=Path("/nonexistent/path"), target_column_name="class"
        )
        with pytest.raises(FileNotFoundError):
            module.prepare_data()

    @patch(
        "chronocratic.datasets.modules.uea.UEAClassificationDataModule._read_arff_data_file",
        side_effect=[_make_mock_train_data(), _make_mock_test_data()],
    )
    def test_prepare_data_loads_data(self, mock_read, synthetic_uea_folder: Path) -> None:
        """prepare_data loads train/test data and sets module state."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        with patch(
            "chronocratic.datasets.modules.uea.UEAClassificationDataModule._read_arff_data_file",
            side_effect=[_make_mock_train_data(), _make_mock_test_data()],
        ):
            module = UEAClassificationDataModule(
                dataset_folder_path=synthetic_uea_folder,
                target_column_name="class",
                scale_data=False,
            )
            module.prepare_data()

            assert module._train_data_samples is not None
            assert module._test_data_samples is not None
            assert module._train_data_labels is not None
            assert module._test_data_labels is not None
            assert module._num_classes is not None
            assert module._seq_len is not None
            assert module._num_features is not None
            # Labels should be pandas Series with category dtype
            assert isinstance(module._train_data_labels, pd.Series)
            assert module._train_data_labels.dtype.name == "category"


class TestUEADataLoaders:
    """Tests for dataloader methods."""

    @patch(
        "chronocratic.datasets.modules.uea.UEAClassificationDataModule._read_arff_data_file",
        side_effect=[_make_mock_train_data(), _make_mock_test_data()],
    )
    def test_train_dataloader_returns_dataloader(
        self, mock_read, synthetic_uea_folder: Path
    ) -> None:
        """train_dataloader returns a DataLoader instance."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        with patch(
            "chronocratic.datasets.modules.uea.UEAClassificationDataModule._read_arff_data_file",
            side_effect=[_make_mock_train_data(), _make_mock_test_data()],
        ):
            module = UEAClassificationDataModule(
                dataset_folder_path=synthetic_uea_folder,
                target_column_name="class",
                scale_data=False,
            )
            module.prepare_data()
            module.setup("fit")

            loader = module.train_dataloader(loader_mode=ClassificationLoaderMode.SAMPLE_ONLY)
            assert isinstance(loader, DataLoader)

    @patch(
        "chronocratic.datasets.modules.uea.UEAClassificationDataModule._read_arff_data_file",
        side_effect=[_make_mock_train_data(), _make_mock_test_data()],
    )
    def test_val_dataloader_returns_dataloader_or_none(
        self, mock_read, synthetic_uea_folder: Path
    ) -> None:
        """val_dataloader returns None when valid_size=0."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        with patch(
            "chronocratic.datasets.modules.uea.UEAClassificationDataModule._read_arff_data_file",
            side_effect=[_make_mock_train_data(), _make_mock_test_data()],
        ):
            module = UEAClassificationDataModule(
                dataset_folder_path=synthetic_uea_folder,
                target_column_name="class",
                valid_size=0.0,
                scale_data=False,
            )
            module.prepare_data()
            result = module.val_dataloader(loader_mode=ClassificationLoaderMode.SAMPLE_ONLY)
            assert result is None

    @patch(
        "chronocratic.datasets.modules.uea.UEAClassificationDataModule._read_arff_data_file",
        side_effect=[_make_mock_train_data(), _make_mock_test_data()],
    )
    def test_test_dataloader_returns_dataloader(
        self, mock_read, synthetic_uea_folder: Path
    ) -> None:
        """test_dataloader returns a DataLoader instance."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        with patch(
            "chronocratic.datasets.modules.uea.UEAClassificationDataModule._read_arff_data_file",
            side_effect=[_make_mock_train_data(), _make_mock_test_data()],
        ):
            module = UEAClassificationDataModule(
                dataset_folder_path=synthetic_uea_folder,
                target_column_name="class",
                scale_data=False,
            )
            module.prepare_data()
            module.setup("fit")

            loader = module.test_dataloader(loader_mode=ClassificationLoaderMode.SAMPLE_ONLY)
            assert isinstance(loader, DataLoader)


class TestUEAUsesScipyLoadarff:
    """Tests verifying scipy.io.arff.loadarff usage."""

    def test_uses_scipy_loadarff(self, synthetic_uea_folder: Path) -> None:
        """Module uses scipy.io.arff.loadarff directly (not utils/arff.py)."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        module = UEAClassificationDataModule(
            dataset_folder_path=synthetic_uea_folder, target_column_name="class", scale_data=False
        )

        with patch("scipy.io.arff.loadarff") as mock_load:
            # Return mock data that _process_stacked_data can handle
            sample1 = np.array([[1.0, 2.0], [3.0, 4.0]])
            sample2 = np.array([[5.0, 6.0], [7.0, 8.0]])
            train_data = np.array(
                [(sample1, b"0"), (sample2, b"1")], dtype=[("f0", "O"), ("f1", "O")]
            )
            mock_load.return_value = (train_data, None)

            module.prepare_data()
            mock_load.assert_called()

    def test_uses_labelencoder(self, synthetic_uea_folder: Path) -> None:
        """Module uses sklearn LabelEncoder for label processing."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        module = UEAClassificationDataModule(
            dataset_folder_path=synthetic_uea_folder, target_column_name="class"
        )
        # Build mock data with string labels
        sample1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        sample2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        mock_data = np.array(
            [(sample1, b"classA"), (sample2, b"classB")], dtype=[("f0", "O"), ("f1", "O")]
        )

        _, labels = module._process_stacked_data(mock_data)

        # LabelEncoder maps strings to integers
        encoder = LabelEncoder()
        expected = encoder.fit_transform(["classA", "classB"])
        assert list(labels) == list(expected)


def test_setup_idempotent(synthetic_uea_folder: Path) -> None:
    """UEA: setup('fit') called twice produces identical train samples.

    Uses mocked _read_arff_data_file to load synthetic nested ARFF data,
    calls prepare_data() + setup('fit'), snapshots the train samples,
    then calls setup('fit') again and asserts data is unchanged
    (sentinel guard).
    """
    from chronocratic.datasets.modules.uea import UEAClassificationDataModule

    with patch(
        "chronocratic.datasets.modules.uea.UEAClassificationDataModule._read_arff_data_file",
        side_effect=[_make_mock_train_data(), _make_mock_test_data()],
    ):
        module = UEAClassificationDataModule(
            dataset_folder_path=synthetic_uea_folder, target_column_name="class", scale_data=False
        )
        module.prepare_data()
        module.setup(stage="fit")

        snapshot_train = module._train_data_samples.copy()
        module.setup(stage="fit")

        np.testing.assert_array_equal(snapshot_train, module._train_data_samples)


def test_cache_round_trip(synthetic_uea_folder: Path) -> None:
    """UEA: prepare_data writes cache, setup reads it back correctly.

    Verifies the 3-D array cache round-trip: after prepare_data() writes
    the npz and metadata.json, clearing in-memory state and calling
    setup() restores data from cache that matches the original.
    """
    from chronocratic.datasets.modules.uea import UEAClassificationDataModule

    with patch(
        "chronocratic.datasets.modules.uea.UEAClassificationDataModule._read_arff_data_file",
        side_effect=[_make_mock_train_data(), _make_mock_test_data()],
    ):
        module = UEAClassificationDataModule(
            dataset_folder_path=synthetic_uea_folder, target_column_name="class", scale_data=False
        )
        module.prepare_data()

        # Snapshot original data
        orig_train = module._train_data_samples.copy()
        orig_test = module._test_data_samples.copy()
        orig_train_labels = module._train_data_labels.copy()
        orig_test_labels = module._test_data_labels.copy()

        # Clear in-memory state (simulates fresh process reading from cache)
        module._train_data_samples = None
        module._test_data_samples = None
        module._valid_data_samples = None
        module._train_data_labels = None
        module._test_data_labels = None
        module._valid_data_labels = None

        # setup() should read from cache
        module.setup(stage="fit")

        # Verify 3-D array preservation through cache round-trip
        np.testing.assert_array_equal(module._train_data_samples, orig_train)
        np.testing.assert_array_equal(module._test_data_samples, orig_test)
        np.testing.assert_array_equal(
            module._train_data_labels.to_numpy(), orig_train_labels.to_numpy()
        )
        np.testing.assert_array_equal(
            module._test_data_labels.to_numpy(), orig_test_labels.to_numpy()
        )

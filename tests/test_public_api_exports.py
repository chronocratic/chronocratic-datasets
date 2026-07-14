"""Tests for public API exports and package configuration."""

import importlib
import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).parent.parent / "src" / "chronocratic" / "datasets"
EXPECTED_INIT_FILES = [
    PACKAGE_ROOT / "__init__.py",
    PACKAGE_ROOT / "datatypes" / "__init__.py",
    PACKAGE_ROOT / "datatypes" / "_base" / "__init__.py",
    PACKAGE_ROOT / "modules" / "__init__.py",
    PACKAGE_ROOT / "modules" / "_base" / "__init__.py",
    PACKAGE_ROOT / "enums" / "__init__.py",
    PACKAGE_ROOT / "utils" / "__init__.py",
]


def test_version_defined():
    """Package exposes a non-empty __version__ following PEP 440."""
    import chronocratic.datasets

    assert hasattr(chronocratic.datasets, "__version__")
    assert isinstance(chronocratic.datasets.__version__, str)
    assert len(chronocratic.datasets.__version__) > 0


def test_import_chronocratic_datasets():
    """import chronocratic.datasets resolves without errors."""
    ts = importlib.import_module("chronocratic.datasets")
    assert hasattr(ts, "__all__")
    assert "__version__" in ts.__all__


def test_enum_exports_in_root():
    """Enum types are exported from package root."""
    from chronocratic.datasets import (
        ClassificationSplitMode,
        ForecastingMode,
        ScalingMethod,
        TimeSeriesDatasetMode,
    )

    assert TimeSeriesDatasetMode.SAMPLE_LABEL == "sample_label"
    assert ScalingMethod.MINMAX == "minmax"
    assert ClassificationSplitMode.AS_DEFINED == "as_defined"
    assert ForecastingMode.UNIVARIATE == "univariate"


def test_init_files_exist():
    """__init__.py files exist at all planned levels."""
    for init_path in EXPECTED_INIT_FILES:
        assert init_path.exists(), f"Missing __init__.py: {init_path.relative_to(PACKAGE_ROOT)}"


def test_submodule_all_declarations():
    """Each submodule __init__.py has an __all__ declaration."""
    submodules = [
        "chronocratic.datasets.datatypes",
        "chronocratic.datasets.datatypes._base",
        "chronocratic.datasets.modules",
        "chronocratic.datasets.modules._base",
        "chronocratic.datasets.enums",
        "chronocratic.datasets.utils",
    ]
    for module_name in submodules:
        mod = importlib.import_module(module_name)
        assert hasattr(mod, "__all__"), f"{module_name} missing __all__"


def test_enum_file_exists():
    """Verifies enums/data.py was created."""
    assert (PACKAGE_ROOT / "enums" / "data.py").exists()


def test_removed_symbols_not_in_root():
    """Removed utils and maps symbols are not importable from the root barrel.

    Per D-01, the root barrel exposes only modules, datatypes, enums, and __version__.
    Utils symbols (cache helpers, data processing) and loader maps are internal and
    must not be re-exported at the package root.
    """
    import chronocratic.datasets

    removed_from_root = [
        # Loader maps (D-01)
        "CLASSIFICATION_LOADER_MAP",
        "FORECASTING_LOADER_MAP",
        # Utils symbols (D-01)
        "atomic_save_metadata",
        "atomic_save_npz",
        "build_cache_key",
        "CACHE_SCHEMA_VERSION",
        "compose",
        "create_data_scaler",
        "custom_collate_fn",
        "extract_time_features",
        "flatten_list_of_np_arrays",
        "get_num_samples_from_ts",
        "load_metadata",
        "load_scaler",
        "process_data_with_varying_sequence_lengths_single",
        "process_df_according_to_dtypes",
        "read_arff_as_df",
        "resolve_cache_dir",
        "save_scaler",
        "separate_target_feature_from_df",
    ]
    for symbol in removed_from_root:
        assert symbol not in chronocratic.datasets.__all__, (
            f"{symbol} should not be in root __all__"
        )
        assert not hasattr(chronocratic.datasets, symbol), (
            f"{symbol} should not be accessible from root barrel"
        )


def test_removed_symbols_not_in_utils_barrel():
    """Removed utils symbols are not importable from the utils barrel.

    Per D-20, only compose, get_num_samples_from_ts, and separate_target_feature_from_df
    are exported from chronocratic.datasets.utils.
    """
    import chronocratic.datasets.utils

    retained = {"compose", "get_num_samples_from_ts", "separate_target_feature_from_df"}
    actual = set(chronocratic.datasets.utils.__all__)
    assert actual == retained, f"utils __all__ should be {sorted(retained)}, got {sorted(actual)}"


def test_root_barrel_symbol_count():
    """Root barrel exports only __version__, enums, datatypes, and modules."""
    from chronocratic.datasets import __all__

    expected = {
        "__version__",
        "ClassificationLoaderMode",
        "ClassificationSplitMode",
        "DataForm",
        "DataPartition",
        "ForecastingLoaderMode",
        "ForecastingMode",
        "ScalingMethod",
        "TimeSeriesDatasetMode",
        "ElectricityDataset",
        "ETTDataset",
        "FixedTimeSeriesDatasetMultivariate",
        "FixedTimeSeriesDatasetUnivariate",
        "FlexibleTimeSeriesDatasetSingleFile",
        "FlexibleTimeSeriesDatasetSingleFileMultipleSeries",
        "TimeSeriesDataset",
        "UCRClassificationUnivariateDataset",
        "UEAClassificationMultivariateDataset",
        "WeatherDataset",
        "BaseClassificationTimeSeriesDataModule",
        "BaseForecastingTimeSeriesDataModule",
        "BaseTimeSeriesDataModule",
        "ElectricityLoadDataModule",
        "ETTDataModule",
        "UCRClassificationDataModule",
        "UEAClassificationDataModule",
        "WeatherDataModule",
    }
    assert set(__all__) == expected, (
        f"Root __all__ mismatch. Extra: {set(__all__) - expected}, "
        f"Missing: {expected - set(__all__)}"
    )

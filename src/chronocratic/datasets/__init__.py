"""chronocratic.datasets -- Time series datasets for PyTorch Lightning."""

try:
    from chronocratic.datasets._version import __version__
except ImportError:  # not built yet -- fall back to installed metadata
    from importlib.metadata import version

    __version__ = version(__name__)

# Enums
# Datatypes
from chronocratic.datasets.datatypes import (
    ElectricityDataset,
    ETTDataset,
    FixedTimeSeriesDatasetMultivariate,
    FixedTimeSeriesDatasetUnivariate,
    FlexibleTimeSeriesDatasetSingleFile,
    FlexibleTimeSeriesDatasetSingleFileMultipleSeries,
    TimeSeriesDataset,
    UCRClassificationUnivariateDataset,
    UEAClassificationMultivariateDataset,
    WeatherDataset,
)
from chronocratic.datasets.enums import (
    ClassificationLoaderMode,
    ClassificationSplitMode,
    DataForm,
    DataPartition,
    ForecastingLoaderMode,
    ForecastingMode,
    ScalingMethod,
    TimeSeriesDatasetMode,
)

# Mappings
from chronocratic.datasets.maps import CLASSIFICATION_LOADER_MAP, FORECASTING_LOADER_MAP

# Modules
from chronocratic.datasets.modules import (
    BaseClassificationTimeSeriesDataModule,
    BaseForecastingTimeSeriesDataModule,
    BaseTimeSeriesDataModule,
    ElectricityLoadModule,
    ETTDataModule,
    UCRClassificationDataModule,
    UEAClassificationDataModule,
    WeatherModule,
)

# Utils (18 symbols from utils/__init__.py __all__)
from chronocratic.datasets.utils import (
    atomic_save_metadata,
    atomic_save_npz,
    build_cache_key,
    CACHE_SCHEMA_VERSION,
    compose,
    create_data_scaler,
    custom_collate_fn,
    extract_time_features,
    flatten_list_of_np_arrays,
    get_num_samples_from_ts,
    load_metadata,
    load_scaler,
    process_data_with_varying_sequence_lengths_single,
    process_df_according_to_dtypes,
    read_arff_as_df,
    resolve_cache_dir,
    save_scaler,
    separate_target_feature_from_df,
)

__all__ = [
    # Utils
    'CACHE_SCHEMA_VERSION',
    # Mappings
    'CLASSIFICATION_LOADER_MAP',
    'FORECASTING_LOADER_MAP',
    # Modules
    'BaseClassificationTimeSeriesDataModule',
    'BaseForecastingTimeSeriesDataModule',
    'BaseTimeSeriesDataModule',
    # Enums
    'ClassificationLoaderMode',
    'ClassificationSplitMode',
    'DataForm',
    'DataPartition',
    'ETTDataModule',
    # Datatypes
    'ETTDataset',
    'ElectricityDataset',
    'ElectricityLoadModule',
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'FlexibleTimeSeriesDatasetSingleFile',
    'FlexibleTimeSeriesDatasetSingleFileMultipleSeries',
    'ForecastingLoaderMode',
    'ForecastingMode',
    'ScalingMethod',
    'TimeSeriesDataset',
    'TimeSeriesDatasetMode',
    'UCRClassificationDataModule',
    'UCRClassificationUnivariateDataset',
    'UEAClassificationDataModule',
    'UEAClassificationMultivariateDataset',
    'WeatherDataset',
    'WeatherModule',
    '__version__',
    'atomic_save_metadata',
    'atomic_save_npz',
    'build_cache_key',
    'compose',
    'create_data_scaler',
    'custom_collate_fn',
    'extract_time_features',
    'flatten_list_of_np_arrays',
    'get_num_samples_from_ts',
    'load_metadata',
    'load_scaler',
    'process_data_with_varying_sequence_lengths_single',
    'process_df_according_to_dtypes',
    'read_arff_as_df',
    'resolve_cache_dir',
    'save_scaler',
    'separate_target_feature_from_df',
]

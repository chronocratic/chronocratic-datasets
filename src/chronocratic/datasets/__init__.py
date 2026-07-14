"""chronocratic.datasets -- Time series datasets for PyTorch Lightning."""

try:
    from chronocratic.datasets._version import __version__
except ImportError:  # not built yet -- fall back to installed metadata
    from importlib.metadata import version

    __version__ = version(__name__)

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

# Enums
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

# Modules
from chronocratic.datasets.modules import (
    BaseClassificationTimeSeriesDataModule,
    BaseForecastingTimeSeriesDataModule,
    BaseTimeSeriesDataModule,
    ElectricityLoadDataModule,
    ETTDataModule,
    UCRClassificationDataModule,
    UEAClassificationDataModule,
    WeatherDataModule,
)

__all__ = [
    "BaseClassificationTimeSeriesDataModule",
    "BaseForecastingTimeSeriesDataModule",
    "BaseTimeSeriesDataModule",
    "ClassificationLoaderMode",
    "ClassificationSplitMode",
    "DataForm",
    "DataPartition",
    "ETTDataModule",
    "ETTDataset",
    "ElectricityDataset",
    "ElectricityLoadDataModule",
    "FixedTimeSeriesDatasetMultivariate",
    "FixedTimeSeriesDatasetUnivariate",
    "FlexibleTimeSeriesDatasetSingleFile",
    "FlexibleTimeSeriesDatasetSingleFileMultipleSeries",
    "ForecastingLoaderMode",
    "ForecastingMode",
    "ScalingMethod",
    "TimeSeriesDataset",
    "TimeSeriesDatasetMode",
    "UCRClassificationDataModule",
    "UCRClassificationUnivariateDataset",
    "UEAClassificationDataModule",
    "UEAClassificationMultivariateDataset",
    "WeatherDataModule",
    "WeatherDataset",
    "__version__",
]

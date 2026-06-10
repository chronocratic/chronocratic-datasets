"""Loader-to-dataset mode mapping dictionaries.

Maps loader-level mode enums (ClassificationLoaderMode, ForecastingLoaderMode)
to internal dataset modes (TimeSeriesDatasetMode). RAW_SERIES maps to None
because it bypasses the dataset mode system entirely.
"""

from chronocratic.datasets.enums.data import (
    ClassificationLoaderMode,
    ForecastingLoaderMode,
    TimeSeriesDatasetMode,
)

CLASSIFICATION_LOADER_MAP: dict[ClassificationLoaderMode, TimeSeriesDatasetMode] = {
    ClassificationLoaderMode.SAMPLE_ONLY: TimeSeriesDatasetMode.SAMPLE_ONLY,
    ClassificationLoaderMode.SAMPLE_LABEL: TimeSeriesDatasetMode.SAMPLE_LABEL,
}

FORECASTING_LOADER_MAP: dict[ForecastingLoaderMode, TimeSeriesDatasetMode | None] = {
    ForecastingLoaderMode.RAW_SERIES: None,
    ForecastingLoaderMode.INPUT_TARGET: TimeSeriesDatasetMode.INPUT_OUTPUT,
    ForecastingLoaderMode.INPUT_ONLY: TimeSeriesDatasetMode.SAMPLE_ONLY,
}

__all__ = [
    'CLASSIFICATION_LOADER_MAP',
    'FORECASTING_LOADER_MAP',
]

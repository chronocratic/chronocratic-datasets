from enum import StrEnum


class TimeSeriesDownstreamTask(StrEnum):
    """Supported downstream tasks for time series data."""

    CLASSIFICATION = 'classification'
    FORECASTING = 'forecasting'
    CLUSTERING = 'clustering'

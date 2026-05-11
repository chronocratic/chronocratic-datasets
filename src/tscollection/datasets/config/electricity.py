"""Electricity load forecasting configuration.

Provides ElectricityConfig, a Pydantic model for the electricity load
forecasting dataset with fractional split boundaries. The CSV file uses
semicolon separator and comma decimal (European format), and the
univariate target column is MT_001.

Split boundaries are derived from rbspaper's electricity_load_datamodule:
- 60/20/20 fractional split (train, valid, test)
"""

from tscollection.datasets.config.base import ForecastingConfig, TaskType
from tscollection.datasets.enums import DatasetFamily, SplitMode

__all__ = [
    'ElectricityConfig',
    'ELECTRICITY_LOAD',
]


class ElectricityConfig(ForecastingConfig):
    """Configuration for electricity load forecasting datasets.

    Extends ForecastingConfig with electricity-specific fields: CSV parsing
    parameters (separator, decimal character) and forecast column for
    univariate mode. Uses fractional split boundaries (D-03) of 60/20/20.

    Attributes:
        name: Dataset name (e.g., 'Electricity').
        family: Always DatasetFamily.ELECTRICITY.
        url: Download URL for the CSV file.
        sha256: SHA256 checksum, or None if not yet available.
        split_mode: Always SplitMode.FRACTIONAL for electricity datasets.
        split_bounds: Train/valid/test fractions as (0.6, 0.2, 0.2).
        forecast_column: Column name for univariate mode ('MT_001').
        csv_sep: CSV field separator (';').
        csv_decimal: CSV decimal character (',').
        default_seq_len: Default input window length (>= 1).
        default_horizon: Default prediction horizon (>= 1).
        tasks: Tuple of supported task types.
    """

    family: DatasetFamily = DatasetFamily.ELECTRICITY
    split_mode: SplitMode = SplitMode.FRACTIONAL
    forecast_column: str = 'MT_001'
    csv_sep: str = ';'
    csv_decimal: str = ','


# ----------------------------------------------------------------------- #
# Dataset instances                                                        #
# ----------------------------------------------------------------------- #

ELECTRICITY_LOAD: ElectricityConfig = ElectricityConfig(
    name='Electricity',
    url='https://raw.githubusercontent.com/rashmitrivedi/NeuralForecast/master/dataset/electricity/electricity.csv',
    sha256=None,
    split_bounds=(0.6, 0.2, 0.2),
    forecast_column='MT_001',
    csv_sep=';',
    csv_decimal=',',
    default_seq_len=128,
    default_horizon=24,
    tasks=(TaskType.FORECASTING, TaskType.REPRESENTATION),
)

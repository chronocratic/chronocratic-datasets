"""Weather forecasting configuration.

Provides WeatherConfig, a Pydantic model for the weather forecasting
dataset with fractional split boundaries. The univariate target is
selected via the last column (iloc[:, -1:]) from the full CSV.

Split boundaries are derived from rbspaper's weather_datamodule:
- 60/20/20 fractional split (train, valid, test)
"""

from tscollection.datasets.config.base import ForecastingConfig, TaskType
from tscollection.datasets.enums import DatasetFamily, SplitMode

__all__ = [
    'WeatherConfig',
    'WEATHER',
]


class WeatherConfig(ForecastingConfig):
    """Configuration for weather forecasting datasets.

    Extends ForecastingConfig with weather-specific fields: univariate
    column selector for mode='univariate'. Uses fractional split
    boundaries (D-03) of 60/20/20.

    Attributes:
        name: Dataset name (e.g., 'weather').
        family: Always DatasetFamily.WEATHER.
        url: Download URL for the CSV file.
        sha256: SHA256 checksum, or None if not yet available.
        split_mode: Always SplitMode.FRACTIONAL for weather datasets.
        split_bounds: Train/valid/test fractions as (0.6, 0.2, 0.2).
        univariate_column: Column selector for univariate mode ('last').
        default_seq_len: Default input window length (>= 1).
        default_horizon: Default prediction horizon (>= 1).
        tasks: Tuple of supported task types.
    """

    family: DatasetFamily = DatasetFamily.WEATHER
    split_mode: SplitMode = SplitMode.FRACTIONAL
    univariate_column: str = 'last'


# ----------------------------------------------------------------------- #
# Dataset instances                                                        #
# ----------------------------------------------------------------------- #

WEATHER: WeatherConfig = WeatherConfig(
    name='weather',
    url='https://raw.githubusercontent.com/rashmitrivedi/NeuralForecast/master/dataset/weather/weather.csv',
    sha256=None,
    split_bounds=(0.6, 0.2, 0.2),
    univariate_column='last',
    default_seq_len=128,
    default_horizon=24,
    tasks=(TaskType.FORECASTING, TaskType.REPRESENTATION),
)

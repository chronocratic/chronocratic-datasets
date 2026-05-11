"""ETT (Electricity Transformer Temperature) forecasting configuration.

Provides ETTConfig, a Pydantic model for ETT family datasets with indexed
split boundaries. Includes pre-configured frozen instances for ETTh1,
ETTh2 (hourly) and ETTm1, ETTm2 (15-min) variants.

Split boundaries are derived from rbspaper's ETT datamodule:
- Hourly (ETTh1/ETTh2): 12*30*24=8640, 16*30*24=11520, 20*30*24=14400
- 15-min (ETTm1/ETTm2): 12*30*24*4=34560, 16*30*24*4=46080, 20*30*24*4=57600
"""

from pydantic import Field

from tscollection.datasets.config.base import ForecastingConfig, TaskType
from tscollection.datasets.enums import DatasetFamily, SplitMode

__all__ = [
    'ETTConfig',
    'ETT_H1',
    'ETT_H2',
    'ETT_M1',
    'ETT_M2',
]


class ETTConfig(ForecastingConfig):
    """Configuration for ETT forecasting datasets.

    Extends ForecastingConfig with ETT-specific fields: forecast column
    for univariate mode, frequency (1h or 15min), and number of features.
    Uses indexed split boundaries (D-03) derived from the original rbspaper
    datamodule.

    Attributes:
        name: Dataset name (ETTh1, ETTh2, ETTm1, ETTm2).
        family: Always DatasetFamily.ETT.
        url: Download URL for the CSV file.
        sha256: SHA256 checksum, or None if not yet available.
        split_mode: Always SplitMode.INDEXED for ETT datasets.
        split_bounds: Train/valid/test end indices as a 3-tuple of integers.
            For hourly: (8640, 11520, 14400). For 15-min: (34560, 46080, 57600).
        forecast_column: Column name for univariate mode (e.g., 'OT').
        frequency: Time step frequency ('1h' or '15min').
        num_features: Total number of CSV feature columns (excluding date).
        default_seq_len: Default input window length (>= 1).
        default_horizon: Default prediction horizon (>= 1).
        tasks: Tuple of supported task types.
    """

    family: DatasetFamily = DatasetFamily.ETT
    split_mode: SplitMode = SplitMode.INDEXED
    forecast_column: str
    frequency: str
    num_features: int = Field(ge=1)


# ----------------------------------------------------------------------- #
# Dataset instances                                                        #
# ----------------------------------------------------------------------- #

# Hourly datasets: 12*30*24=8640, 16*30*24=11520, 20*30*24=14400
_HOURLY_SPLIT_BOUNDS = (12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24)

# 15-min datasets: 12*30*24*4=34560, 16*30*24*4=46080, 20*30*24*4=57600
_15MIN_SPLIT_BOUNDS = (12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4)


ETT_H1: ETTConfig = ETTConfig(
    name='ETTh1',
    url='https://raw.githubusercontent.com/zhouhao118/ETD/main/ETTh1.csv',
    sha256=None,
    split_bounds=_HOURLY_SPLIT_BOUNDS,
    forecast_column='OT',
    frequency='1h',
    num_features=7,
    default_seq_len=128,
    default_horizon=24,
    tasks=(TaskType.FORECASTING, TaskType.REPRESENTATION),
)

ETT_H2: ETTConfig = ETTConfig(
    name='ETTh2',
    url='https://raw.githubusercontent.com/zhouhao118/ETD/main/ETTh2.csv',
    sha256=None,
    split_bounds=_HOURLY_SPLIT_BOUNDS,
    forecast_column='OT',
    frequency='1h',
    num_features=7,
    default_seq_len=128,
    default_horizon=24,
    tasks=(TaskType.FORECASTING, TaskType.REPRESENTATION),
)

ETT_M1: ETTConfig = ETTConfig(
    name='ETTm1',
    url='https://raw.githubusercontent.com/zhouhao118/ETD/main/ETTm1.csv',
    sha256=None,
    split_bounds=_15MIN_SPLIT_BOUNDS,
    forecast_column='OT',
    frequency='15min',
    num_features=7,
    default_seq_len=128,
    default_horizon=96,
    tasks=(TaskType.FORECASTING, TaskType.REPRESENTATION),
)

ETT_M2: ETTConfig = ETTConfig(
    name='ETTm2',
    url='https://raw.githubusercontent.com/zhouhao118/ETD/main/ETTm2.csv',
    sha256=None,
    split_bounds=_15MIN_SPLIT_BOUNDS,
    forecast_column='OT',
    frequency='15min',
    num_features=7,
    default_seq_len=128,
    default_horizon=96,
    tasks=(TaskType.FORECASTING, TaskType.REPRESENTATION),
)

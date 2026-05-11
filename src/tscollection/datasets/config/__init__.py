"""Pydantic configuration models for dataset metadata.

Re-exports all config classes, frozen instances, and factory functions
for convenient imports. Per D-02 (both module-level constants + registry
dict), this provides direct IDE support for all config instances.
"""

# Base types
from tscollection.datasets.config.base import (
    ArffFilePattern,
    ClassificationConfig,
    ClassificationFilePatterns,
    DatasetConfig,
    ForecastingConfig,
)

# Factory registry
from tscollection.datasets.config.factory import (
    CONFIGS,
    get_config,
    list_configs,
)

# UCR configs
from tscollection.datasets.config.ucr import (
    UCRConfig,
    UCR_COFFEE,
    UCR_ECG200,
    UCR_FACE_FOUR,
)

# UEA configs
from tscollection.datasets.config.uea import (
    UEAConfig,
    UEA_ATRIAL_FIBRILLATION,
    UEA_BASIC_MOTIONS,
)

# ETT configs
from tscollection.datasets.config.ett import (
    ETTConfig,
    ETT_H1,
    ETT_H2,
    ETT_M1,
    ETT_M2,
)

# Electricity configs
from tscollection.datasets.config.electricity import (
    ELECTRICITY_LOAD,
    ElectricityConfig,
)

# Weather configs
from tscollection.datasets.config.weather import (
    WEATHER,
    WeatherConfig,
)

__all__ = [
    # Base types
    'ArffFilePattern',
    'ClassificationConfig',
    'ClassificationFilePatterns',
    'DatasetConfig',
    'ForecastingConfig',
    # Config classes
    'ElectricityConfig',
    'ETTConfig',
    'UCRConfig',
    'UEAConfig',
    'WeatherConfig',
    # UCR instances
    'UCR_COFFEE',
    'UCR_ECG200',
    'UCR_FACE_FOUR',
    # UEA instances
    'UEA_ATRIAL_FIBRILLATION',
    'UEA_BASIC_MOTIONS',
    # ETT instances
    'ETT_H1',
    'ETT_H2',
    'ETT_M1',
    'ETT_M2',
    # Electricity instances
    'ELECTRICITY_LOAD',
    # Weather instances
    'WEATHER',
    # Factory
    'CONFIGS',
    'get_config',
    'list_configs',
]

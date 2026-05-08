__all__ = ['TimeSeriesEvaluationDownstreamTaskEnum']

from enum import Enum


class TimeSeriesEvaluationDownstreamTaskEnum(Enum):
    CLASSIFICATION = 'classification'
    FORECASTING = 'forecasting'
    CLUSTERING = 'clustering'

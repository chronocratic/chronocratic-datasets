__all__ = ['evaluate']

import logging
import random

import lightning.pytorch as pl
import numpy as np
from sklearn.model_selection import train_test_split

from src.rbspaper.evaluation.classification import classify_and_evaluate
from src.rbspaper.evaluation.enums import TimeSeriesEvaluationDownstreamTaskEnum
from src.rbspaper.evaluation.forecasting import forecast_and_evaluate

logger = logging.getLogger(name=__name__)


def _process_train_data_size(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    downstream_task: str,
    downstream_task_params: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    max_train_data_size_map = {
        TimeSeriesEvaluationDownstreamTaskEnum.FORECASTING.value: {'all': 100_000},
        TimeSeriesEvaluationDownstreamTaskEnum.CLASSIFICATION.value: {'svm': 10_000},
    }

    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.CLASSIFICATION.value:
        evaluation_protocol: str = str(downstream_task_params.get('evaluation_protocol'))
        max_train_data_size = max_train_data_size_map[downstream_task].get(evaluation_protocol)

    elif downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.FORECASTING.value:
        max_train_data_size = max_train_data_size_map[downstream_task]['all']

    else:
        max_train_data_size = None

    if max_train_data_size is not None and len(train_features) > max_train_data_size:
        message = (
            f'Training data size ({len(train_features)}) is larger than the maximum allowed '
            f'size ({max_train_data_size}) for downstream task: {downstream_task} with '
            f'params: {downstream_task_params}. Limiting the training data size to ensure '
            'faster evaluation and avoid memory issues.'
        )

        logger.info(message)

        if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.FORECASTING.value:
            max_train_data_size = max_train_data_size_map[downstream_task]['all']
            train_features, _, train_labels, _ = train_test_split(
                train_features, train_labels, train_size=max_train_data_size, random_state=42
            )

        else:
            train_features, _, train_labels, _ = train_test_split(
                train_features,
                train_labels,
                train_size=max_train_data_size,
                random_state=42,
                stratify=train_labels,
            )

    return train_features, train_labels


def evaluate(
    downstream_task: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    downstream_task_params: dict,
) -> dict:
    """Dispatch evaluation to classification or forecasting based on downstream_task.

    Sets random seeds for reproducibility, optionally limits training data size,
    then delegates to the appropriate evaluation function.

    Args:
        downstream_task: Task identifier (e.g. 'classification', 'forecasting').
        train_features: Training input features.
        train_labels: Training labels.
        valid_features: Validation input features.
        valid_labels: Validation labels.
        test_features: Test input features.
        test_labels: Test labels.
        downstream_task_params: Additional parameters for the evaluation protocol.

    Returns:
        Dictionary of evaluation metrics.

    Raises:
        ValueError: If the downstream task is unknown or missing required parameters.
    """
    evaluation_seed = 42
    messasge = f'Setting evaluation seed to {evaluation_seed} to ensure reproducibility'
    logger.info(messasge)
    np.random.default_rng(evaluation_seed)
    random.seed(evaluation_seed)
    pl.seed_everything(evaluation_seed)

    train_features, train_labels = _process_train_data_size(
        train_features=train_features,
        train_labels=train_labels,
        downstream_task=downstream_task,
        downstream_task_params=downstream_task_params,
    )

    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.FORECASTING.value:
        return forecast_and_evaluate(
            train_features=train_features,
            train_labels=train_labels,
            valid_features=valid_features,
            valid_labels=valid_labels,
            test_features=test_features,
            test_labels=test_labels,
        )

    if downstream_task == TimeSeriesEvaluationDownstreamTaskEnum.CLASSIFICATION.value:
        evaluation_protocol = str(downstream_task_params.get('evaluation_protocol'))
        if evaluation_protocol is None:
            message = 'evaluation_protocol is required for classification downstream task'
            logger.error(message)
            raise ValueError(message)

        return classify_and_evaluate(
            train_features=train_features,
            train_labels=train_labels,
            valid_features=valid_features,
            valid_labels=valid_labels,
            test_features=test_features,
            test_labels=test_labels,
            evaluation_protocol=evaluation_protocol,
        )

    messgae = (
        f'Unknown downstream task: {downstream_task}. '
        f'Valid options are: {[task.value for task in TimeSeriesEvaluationDownstreamTaskEnum]}'
    )
    logger.error(messgae)
    raise ValueError(messgae)

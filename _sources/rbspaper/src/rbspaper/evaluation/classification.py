__all__ = ['classify_and_evaluate']


import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.rbspaper.evaluation.protocols import get_evaluation_protocol


def calculate_classification_evaluation_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict:
    num_classes = len(np.unique(targets))
    average_strategy = 'binary' if num_classes == 2 else 'macro'  # noqa: PLR2004

    predictions = predictions.reshape(-1, 1)
    targets = targets.reshape(-1, 1)

    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average=average_strategy)
    recall = recall_score(targets, predictions, average=average_strategy)
    f1 = f1_score(targets, predictions, average=average_strategy)
    average_precision = average_precision_score(targets, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'average_precision': average_precision,
    }


def classify_and_evaluate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    evaluation_protocol: str,
) -> dict:
    """Fit a classification model, make predictions, and evaluate with classification metrics.

    Fits a classification model using the provided training and validation data,
    makes predictions on the test set, and evaluates using classification metrics.

    Args:
        train_features: Training input features.
        train_labels: Training labels.
        valid_features: Validation input features.
        valid_labels: Validation labels.
        test_features: Test input features.
        test_labels: Test labels.
        evaluation_protocol: Name of the evaluation protocol (e.g. 'svm', 'ridge').

    Returns:
        Dictionary of classification metrics.
    """
    evaluation_protocol_fn = get_evaluation_protocol(evaluation_protocol)

    cls = evaluation_protocol_fn(
        train_features=train_features,
        train_labels=train_labels,
        valid_features=valid_features,
        valid_labels=valid_labels,
    )

    predictions = cls.predict(test_features)

    return calculate_classification_evaluation_metrics(predictions=predictions, targets=test_labels)

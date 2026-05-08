__all__ = ['forecast_and_evaluate']


import numpy as np

from src.rbspaper.evaluation.protocols import get_evaluation_protocol


def calculate_forecasting_evaluation_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    mse_loss = np.mean((predictions - targets) ** 2)
    mae_loss = np.mean(np.abs(predictions - targets))
    mape_loss = np.mean(np.abs(predictions - targets) / np.abs(targets).clip(min=1e-8))

    return {'mse_loss': mse_loss, 'mae_loss': mae_loss, 'mape_loss': mape_loss}


def forecast_and_evaluate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Fit a forecasting model, make predictions, and evaluate with regression metrics.

    Fits a forecasting model using the provided training and validation data,
    makes predictions on the test set, and evaluates using regression metrics.

    Args:
        train_features: Training input features.
        train_labels: Training labels.
        valid_features: Validation input features.
        valid_labels: Validation labels.
        test_features: Test input features.
        test_labels: Test labels.

    Returns:
        Dictionary of forecasting metrics.
    """
    evaluation_protocol_fn = get_evaluation_protocol(evaluation_protocol='ridge')

    cls = evaluation_protocol_fn(
        train_features=train_features,
        train_labels=train_labels,
        valid_features=valid_features,
        valid_labels=valid_labels,
    )

    predictions = cls.predict(test_features)

    return calculate_forecasting_evaluation_metrics(predictions=predictions, targets=test_labels)

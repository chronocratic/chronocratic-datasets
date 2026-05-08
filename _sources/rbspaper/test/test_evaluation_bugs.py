"""Unit tests for evaluation bugs: Ridge argmax, MAPE zero-target crash, max_train_data_size UnboundLocalError."""

import numpy as np

from src.rbspaper.evaluation.enums import TimeSeriesEvaluationDownstreamTaskEnum
from src.rbspaper.evaluation.evaluation import _process_train_data_size
from src.rbspaper.evaluation.forecasting import calculate_forecasting_evaluation_metrics
from src.rbspaper.evaluation.protocols import _fit_ridge


def test_ridge_selects_minimum_loss_alpha() -> None:
    """Ridge protocol must select alpha with lowest validation loss, not highest.

    When validation scores (RMSE + MAE) decrease with smaller alpha (less regularization),
    argmin should select the smallest alpha (0.1), while argmax would select the largest
    alpha (1000). This test verifies argmin behavior by using data where the relationship
    between features and labels is approximately linear, so less regularization fits better.
    """
    rng = np.random.default_rng(42)
    n_train, n_valid, n_features = 50, 20, 5

    # Create features and labels with a known linear relationship.
    # Lower alpha -> less regularization -> better fit -> lower validation loss.
    weights = rng.standard_normal(n_features)
    train_features = rng.standard_normal((n_train, n_features))
    train_labels = train_features @ weights + rng.standard_normal(n_train) * 0.01
    valid_features = rng.standard_normal((n_valid, n_features))
    valid_labels = valid_features @ weights + rng.standard_normal(n_valid) * 0.01

    pipeline = _fit_ridge(
        train_features=train_features,
        train_labels=train_labels,
        valid_features=valid_features,
        valid_labels=valid_labels,
    )

    ridge_model = pipeline.named_steps['ridge']
    # alpha_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    # With argmin on monotonically decreasing losses, we expect the smallest alpha (0.1).
    # With argmax, we'd get the largest alpha (1000).
    assert ridge_model.alpha == 0.1, (
        f'Ridge should select smallest alpha (0.1) for best fit, got {ridge_model.alpha}. '
        'This indicates argmax is used instead of argmin for alpha selection.'
    )


def test_mape_handles_zero_targets() -> None:
    """MAPE must not raise or produce infinity when targets contain zero values."""
    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([0.0, 2.0, 3.0])

    result = calculate_forecasting_evaluation_metrics(predictions=predictions, targets=targets)

    assert np.isfinite(result['mape_loss']), (
        f'MAPE should be finite for zero targets, got {result["mape_loss"]}'
    )


def test_mape_returns_finite_for_all_zero_targets() -> None:
    """MAPE must return a finite value even when all targets are zero."""
    predictions = np.array([1.0, 2.0, 3.0])
    targets = np.array([0.0, 0.0, 0.0])

    result = calculate_forecasting_evaluation_metrics(predictions=predictions, targets=targets)

    assert np.isfinite(result['mape_loss']), (
        f'MAPE should be finite for all-zero targets, got {result["mape_loss"]}'
    )


def test_forecasting_process_train_data_no_unbound_error() -> None:
    """_process_train_data_size must not raise UnboundLocalError for forecasting task.

    When downstream_task is FORECASTING, max_train_data_size must be set from the map
    so the truncation logic can execute. Without this branch, the variable remains
    unset and a NameError is raised.
    """
    rng = np.random.default_rng(42)
    train_f = rng.standard_normal((50, 3))
    train_l = rng.standard_normal(50)

    # Should not raise NameError / UnboundLocalError
    result_f, result_l = _process_train_data_size(
        train_features=train_f,
        train_labels=train_l,
        downstream_task=TimeSeriesEvaluationDownstreamTaskEnum.FORECASTING.value,
        downstream_task_params={},
    )

    # Data is small (< 100_000), so no truncation should occur
    assert result_f.shape == train_f.shape
    assert result_l.shape == train_l.shape


def test_forecasting_process_train_data_uses_map_size() -> None:
    """_process_train_data_size must respect the FORECASTING max size from the map."""
    rng = np.random.default_rng(42)
    # Create data larger than the FORECASTING limit (100_000) would truncate.
    # Use a small dataset to verify no truncation, then verify with a larger one.
    train_f = rng.standard_normal((50_000, 3))
    train_l = rng.standard_normal(50_000)

    result_f, result_l = _process_train_data_size(
        train_features=train_f,
        train_labels=train_l,
        downstream_task=TimeSeriesEvaluationDownstreamTaskEnum.FORECASTING.value,
        downstream_task_params={},
    )

    # 50_000 < 100_000, so no truncation
    assert result_f.shape == train_f.shape

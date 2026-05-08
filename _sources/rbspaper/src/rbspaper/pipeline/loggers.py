"""Factory and helpers for W&B + TensorBoard experiment tracking loggers."""

from __future__ import annotations

__all__ = ['create_loggers']

import logging
from typing import Any, TYPE_CHECKING

from lightning.pytorch.loggers import TensorBoardLogger

if TYPE_CHECKING:
    from pathlib import Path

    import lightning.pytorch as pl
    import lightning.pytorch.loggers  # noqa: F401
    from lightning.pytorch.loggers import WandbLogger

logger = logging.getLogger(__name__)


def create_loggers(
    *, run_dir: Path, run_name: str, tracking_mode: str = 'online', persist_artifacts: bool = True
) -> tuple[WandbLogger | TensorBoardLogger, ...]:
    """Create Lightning loggers for experiment tracking.

    Builds a tuple of loggers based on the provided configuration:
    - Always creates TensorBoardLogger when persist_artifacts is True.
    - Creates WandbLogger when persist_artifacts is True and tracking_mode
      is not 'disabled'. WandbLogger is lazily imported to avoid hard
      dependencies in offline HPC environments.

    Args:
        run_dir: Directory for run-specific artifacts (checkpoints, logs).
        run_name: Hierarchical run identifier for the experiment.
        tracking_mode: W&B mode ('online' or 'offline'). Use 'disabled' to
            skip W&B logging entirely.
        persist_artifacts: When False, returns an empty tuple (no loggers).

    Returns:
        Tuple of created loggers. Empty when persist_artifacts is False.
        Contains only TensorBoardLogger when tracking_mode is 'disabled'.
        Contains both WandbLogger and TensorBoardLogger otherwise.

    Raises:
        ImportError: If tracking_mode requires W&B but it is not installed.
    """
    if not persist_artifacts:
        return ()

    loggers: list[WandbLogger | TensorBoardLogger] = []

    # TensorBoardLogger is always created (local-only, no network)
    try:
        loggers.append(TensorBoardLogger(save_dir=str(run_dir), name='tensorboard'))
    except ModuleNotFoundError:
        logger.warning(
            'TensorBoardLogger is not available (tensorboard/tensorboardX not installed). '
            'TensorBoard tracking will be skipped. Install with: uv sync --extra tracking'
        )

    # WandbLogger is optional (lazy import, graceful fallback)
    if tracking_mode != 'disabled':
        try:
            from lightning.pytorch.loggers import WandbLogger as _WandbLogger  # noqa: PLC0415

            loggers.append(
                _WandbLogger(
                    project='rbspaper',
                    name=run_name,
                    mode=tracking_mode,
                    log_model=False,
                    save_dir=str(run_dir),
                )
            )
        except (ImportError, ModuleNotFoundError) as exc:
            message = (
                f'WandbLogger requires wandb to be installed, but it is not '
                f'available ({exc}). Falling back to TensorBoardLogger only. '
                f'Install with: uv sync --extra tracking'
            )
            logger.warning(message)

    return tuple(loggers)


def _find_wandb_logger(*, loggers: tuple[pl.loggers.Logger, ...]) -> WandbLogger | None:
    """Return the first WandbLogger from the loggers tuple, or None.

    Uses ``type(logger).__name__`` instead of ``isinstance`` to avoid
    importing the wandb package at module level.

    Args:
        loggers: Tuple of Lightning loggers (may be empty).

    Returns:
        The WandbLogger instance if found, otherwise None.
    """
    for log in loggers:
        if type(log).__name__ == 'WandbLogger':
            return log  # ty: ignore[invalid-return-type]
    return None


def _flatten_dict(*, d: dict[str, Any], separator: str = '_', prefix: str = '') -> dict[str, Any]:
    """Recursively flatten a nested dictionary using *separator* as key joiner.

    Nested dicts produce dotted-style keys::

        {'a': {'b': 1}} -> {'a_b': 1}

    Lists produce index-based keys::

        {'a': [1, 2]} -> {'a_0': 1, 'a_1': 2}

    List items that are dicts are recursed with the index prefix::

        {'a': [{'x': 1}]} -> {'a_0_x': 1}

    Args:
        d: Dictionary to flatten.
        separator: String used to join nested keys.
        prefix: Current key prefix (used during recursion).

    Returns:
        A flat dictionary with compound keys.
    """
    result: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f'{prefix}{separator}{key}' if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_dict(d=value, separator=separator, prefix=new_key))
        elif isinstance(value, list | tuple):
            for idx, item in enumerate(value):
                item_key = f'{new_key}{separator}{idx}'
                if isinstance(item, dict):
                    result.update(_flatten_dict(d=item, separator=separator, prefix=item_key))
                else:
                    result[item_key] = item
        else:
            result[new_key] = value
    return result


def _log_config_to_wandb(
    *, config_data: dict[str, Any], loggers: tuple[pl.loggers.Logger, ...]
) -> None:
    """Update the W&B experiment config with flattened pipeline settings.

    Returns early if no WandbLogger is present in the loggers tuple
    (graceful fallback when W&B is disabled or unavailable).

    Args:
        config_data: Key-value pairs to write to the W&B config panel.
        loggers: Tuple of Lightning loggers produced by :func:`create_loggers`.
    """
    wandb_logger = _find_wandb_logger(loggers=loggers)
    if wandb_logger is None:
        return

    try:
        wandb_logger.experiment.config.update(config_data)
    except Exception:
        logger.warning(
            'Failed to update W&B config. Continuing without W&B config logging.', exc_info=True
        )


def _log_results_to_wandb(
    *,
    results_summary: dict[str, Any],
    timing: dict[str, float],
    loggers: tuple[pl.loggers.Logger, ...],
) -> None:
    """Log pipeline metrics and a summary comparison table to W&B.

    Writes two payloads:
    1. Flattened scalar metrics (results + timing) via ``run.log()``.
    2. A ``wandb.Table`` with one row of key representation-quality metrics.

    Returns early if no WandbLogger is present (graceful fallback).
    Does NOT call ``wandb.init()`` -- the logger's ``.experiment`` property
    already holds the active run.

    Args:
        results_summary: Nested dictionary of experiment outcomes
            (e.g. accuracy, F1, MAE, geometry metrics).
        timing: Mapping of stage names to wall-clock seconds.
        loggers: Tuple of Lightning loggers produced by :func:`create_loggers`.
    """
    wandb_logger = _find_wandb_logger(loggers=loggers)
    if wandb_logger is None:
        return

    try:
        import wandb  # noqa: PLC0415

        run = wandb_logger.experiment

        # Log flattened scalars
        flat_results = _flatten_dict(d=results_summary)
        flat_timing = {f'timing_{k}': v for k, v in timing.items()}
        run.log({**flat_results, **flat_timing})

        # Log comparison table (one-row summary of key metrics)
        table = wandb.Table(
            columns=[
                'experiment',
                'dataset',
                'seed',
                'model_name',
                'classification_clean_accuracy',
                'classification_clean_f1',
                'forecasting_clean_mae',
                'classification_clean_geometry_centroid_margin',
                'classification_attacked_shift_mean_l2',
                'total_seconds',
            ],
            data=[
                [
                    flat_results.get('experiment'),
                    flat_results.get('dataset'),
                    flat_results.get('seed'),
                    flat_results.get('model_name'),
                    flat_results.get('classification_clean_accuracy'),
                    flat_results.get('classification_clean_f1'),
                    flat_results.get('forecasting_clean_mae'),
                    flat_results.get('classification_clean_geometry_centroid_margin'),
                    flat_results.get('classification_attacked_shift_mean_l2'),
                    flat_results.get('total_seconds'),
                ]
            ],
        )
        run.log({'comparison': table})
    except Exception:
        logger.warning('Failed to log results to W&B. Results persisted to disk.', exc_info=True)

#!/usr/bin/env python3
"""CLI runner for robustness benchmark experiments.

Resolves an experiment instance and dataset, assembles the pipeline,
and executes it via `run_experiment_pipeline`.

Usage:
    uv run python runners/py/runner.py \
        --experiment_id ts2vec \
        --dataset_name Coffee \
        --data_root /path/to/data \
        --output_dir outputs
"""

import argparse
import copy
from dataclasses import asdict
from enum import Enum
import logging
import os
from pathlib import Path
import sys
from typing import cast, Protocol, runtime_checkable

from lightning.pytorch.loggers import Logger

from experiment_instances.data_utils import build_dataset_task_profile
from experiment_instances.instances import (
    ExperimentInstance,
    EXPERIMENTS_REGISTRY,
    get_experiment_instance,
    list_experiment_ids,
)
from src.rbspaper.attacks.enums import AttackFamily
from src.rbspaper.data.data_setup import get_all_datasets, get_datamodule_with_downstream_tasks
from src.rbspaper.pipeline.config import (
    build_hierarchical_run_name,
    DataConfig,
    DatasetTaskProfile,
    DownstreamTaskConfig,
    ExperimentPipelineConfig,
    PipelineArtifactConfig,
    RepresentationAnalysisConfig,
    RepresentationEncodingConfig,
    TrainingConfig,
)
from src.rbspaper.pipeline.core import run_experiment_pipeline
from src.rbspaper.pipeline.loggers import create_loggers
from src.rbspaper.pipeline.setup.model import build_model_from_parameters
from src.rbspaper.pipeline.state import compute_config_hash, load_pipeline_state, STATE_FILENAME

logger = logging.getLogger(__name__)


@runtime_checkable
class _ModelParamsWithSequenceLength(Protocol):
    """Protocol for model params that support set_sequence_length."""

    def set_sequence_length(self, length: int) -> None: ...


@runtime_checkable
class _ModelParamsWithMaxTrainLength(Protocol):
    """Protocol for model params that have max_train_length attribute."""

    max_train_length: int


def _make_json_serializable(obj: object) -> object:
    """Recursively convert Enum values to strings for JSON serialization.

    Used to prepare model_params dictionaries for compute_config_hash,
    which relies on json.dumps internally.

    Args:
        obj: Object to convert (dict, list, Enum, or primitive).

    Returns:
        JSON-serializable version of the object.
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    return obj


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run a robustness benchmark experiment.')
    parser.add_argument(
        '--experiment_id', type=str, default=None, help='Registered experiment ID (e.g. ts2vec).'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help='Registered dataset name (e.g. Coffee). Mutually exclusive with --dataset_index.',
    )
    parser.add_argument(
        '--dataset_index',
        type=int,
        default=None,
        help='Index into the registered dataset list (for HPC array jobs). '
        'Mutually exclusive with --dataset_name.',
    )
    parser.add_argument(
        '--data_root', type=Path, default=None, help='Root directory containing dataset files.'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('outputs'),
        help='Directory for experiment outputs (default: outputs).',
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Override run name (default: {experiment_id}_{dataset_name}).',
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Random seed for reproducibility (default: 42).'
    )
    parser.add_argument(
        '--max_epochs', type=int, default=None, help='Override max training epochs.'
    )
    parser.add_argument(
        '--encoding_batch_size', type=int, default=None, help='Override encoding batch size.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None, help='Override training batch size.'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help='DataLoader worker count (default: 0).'
    )
    parser.add_argument(
        '--forecasting_mode',
        type=str,
        default=None,
        choices=['univariate', 'multivariate'],
        help='Forecasting dataset mode override.',
    )
    parser.add_argument(
        '--list_experiments', action='store_true', help='List available experiment IDs and exit.'
    )
    parser.add_argument(
        '--dry_run', action='store_true', help='Assemble config and print summary without running.'
    )
    parser.add_argument(
        '--force', action='store_true', help='Ignore existing checkpoint state and start fresh.'
    )
    parser.add_argument(
        '--attack_family',
        type=str,
        default=None,
        choices=['white_box', 'black_box'],
        help='Filter attacks by family (white_box, black_box). Default: all families.',
    )
    parser.add_argument(
        '--tracking_mode',
        type=str,
        default=None,
        choices=['online', 'offline', 'disabled'],
        help='Experiment tracking mode. Default: auto-detect (offline on HPC, online locally).',
    )
    return parser.parse_args(argv)


def _resolve_tracking_mode(*, cli_mode: str | None) -> str:
    """Resolve effective tracking mode from CLI arg with HPC auto-detection.

    When the user does not provide --tracking_mode, the function auto-detects
    the execution environment: SLURM_JOB_ID indicates an HPC scheduler, so
    'offline' mode is used. Otherwise, 'online' is the default for local runs.

    Args:
        cli_mode: Value from --tracking_mode CLI argument. None when not set.

    Returns:
        Resolved tracking mode string ('online', 'offline', or 'disabled').
    """
    if cli_mode is not None:
        return cli_mode
    return 'offline' if os.environ.get('SLURM_JOB_ID') else 'online'


def _build_run_name(
    *,
    experiment_id: str,
    dataset_name: str,
    short_hash: str,
    seed: int,
    run_name_override: str | None,
) -> str:
    """Build run name using hierarchical structure (D-05)."""
    if run_name_override is not None:
        return run_name_override
    return build_hierarchical_run_name(
        experiment_id=experiment_id, short_hash=short_hash, seed=seed, dataset_name=dataset_name
    )


def _resolve_task_profile(*, dataset_name: str) -> DatasetTaskProfile | None:
    """Build task profile from registry, or None for forecasting-only datasets."""
    return build_dataset_task_profile(dataset_name=dataset_name)


def setup_logging(*, log_dir: Path, log_level: int = logging.INFO) -> None:
    """Configure root logger with file and stream handlers.

    File handler writes INFO+ to run_dir/pipeline.log.
    Stream handler writes WARNING+ to stdout (keeps tqdm unobstructed).

    Args:
        log_dir: Directory for the log file (the run_dir).
        log_level: Logging level for the file handler (default: INFO).
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Accept all; filters per handler

    # Avoid duplicate handlers on repeated calls
    if root.handlers:
        return

    # File handler -- INFO+, goes in run_dir
    file_handler = logging.FileHandler(log_dir / 'pipeline.log')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    )
    root.addHandler(file_handler)

    # Stream handler -- WARNING+, stdout (tqdm-friendly)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    root.addHandler(stream_handler)


def _build_pipeline_config(
    *,
    experiment_instance: ExperimentInstance,
    dataset_name: str,
    data_root: Path,
    output_dir: Path,
    run_name: str,
    seed: int,
    num_workers: int,
    forecasting_mode: str | None,
    max_epochs_override: int | None,
    encoding_batch_size_override: int | None,
    batch_size_override: int | None,
    attack_family: AttackFamily | None = None,
    loggers: tuple[Logger, ...] = (),
) -> ExperimentPipelineConfig:
    """Assemble all components into a pipeline configuration."""
    # --- Data ---
    extra_params: dict[str, str | None] = {}
    if forecasting_mode:
        extra_params['forecasting_mode'] = forecasting_mode
    if batch_size_override:
        extra_params['batch_size'] = str(batch_size_override)

    datamodule_result = get_datamodule_with_downstream_tasks(
        dataset_name=dataset_name,
        data_root=data_root,
        num_workers=num_workers,
        extra_params=extra_params or None,
    )
    data_module = datamodule_result.data_module
    downstream_task_names = datamodule_result.downstream_tasks

    # Resolve input dimensions from datamodule
    input_dims = data_module.n_features or 1
    sequence_len = data_module.sequence_len

    # --- Model (deepcopy to avoid mutating shared instance) ---
    model_params = copy.deepcopy(experiment_instance.model_params)
    model_params.set_input_dims(input_dims)
    if sequence_len is not None:
        if isinstance(model_params, _ModelParamsWithSequenceLength):
            model_params.set_sequence_length(sequence_len)
        if isinstance(model_params, _ModelParamsWithMaxTrainLength):
            model_params.max_train_length = max(
                sequence_len,
                model_params.max_train_length,
            )

    model = build_model_from_parameters(parameters=model_params)

    # --- Trainer kwargs ---
    trainer_kwargs = dict(experiment_instance.trainer_kwargs)
    effective_max_epochs = max_epochs_override or trainer_kwargs.get(
        'max_epochs', experiment_instance.max_epochs
    )
    trainer_kwargs['max_epochs'] = effective_max_epochs

    # --- Task profile ---
    task_profile = _resolve_task_profile(dataset_name=dataset_name)

    # --- Downstream tasks ---
    downstream_tasks = tuple(DownstreamTaskConfig(task_name=name) for name in downstream_task_names)

    # --- Attacks (filtered by family if specified) ---
    if attack_family is not None:
        attacks = experiment_instance.attack_families.get(attack_family, ())
    else:
        attacks = experiment_instance.attack_params

    # --- Encoding ---
    encoding_batch_size = encoding_batch_size_override or experiment_instance.encoding_batch_size

    # --- Artifacts ---
    run_dir = output_dir / run_name
    run_dir.parent.mkdir(parents=True, exist_ok=True)

    return ExperimentPipelineConfig(
        model=model,
        data=DataConfig(data_module=data_module, profile=task_profile),
        training=TrainingConfig(trainer_kwargs=trainer_kwargs),
        encoding=RepresentationEncodingConfig(batch_size=encoding_batch_size),
        attacks=attacks,
        downstream_tasks=downstream_tasks,
        analysis=RepresentationAnalysisConfig(),
        artifacts=PipelineArtifactConfig(output_dir=output_dir, run_name=run_name),
        seed=seed,
        attack_scope=experiment_instance.attack_scope,
        loggers=loggers,
    )


def _resolve_dataset(*, args: argparse.Namespace) -> str:
    """Validate dataset args and resolve the final dataset name.

    Checks that exactly one of --dataset_name or --dataset_index is provided.
    If --dataset_index is used, resolves it against the registry.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The resolved dataset name string.

    Raises:
        SystemExit: If both or neither dataset arg is provided, or if the
            index is out of range.
    """
    if args.dataset_name is not None and args.dataset_index is not None:
        parser = argparse.ArgumentParser(description='Run a robustness benchmark experiment.')
        parser.error(
            '--dataset_index and --dataset_name are mutually exclusive. Use one or the other.'
        )
    if args.dataset_name is None and args.dataset_index is None:
        parser = argparse.ArgumentParser(description='Run a robustness benchmark experiment.')
        parser.error('one of --dataset_name or --dataset_index is required')

    if args.dataset_index is not None:
        all_datasets_result = get_all_datasets(form='list')
        all_datasets: list[str] = (
            list(all_datasets_result)
            if isinstance(all_datasets_result, set)
            else all_datasets_result
        )
        if args.dataset_index < 0 or args.dataset_index >= len(all_datasets):
            parser = argparse.ArgumentParser(description='Run a robustness benchmark experiment.')
            parser.error(
                f'dataset_index {args.dataset_index} out of range (0-{len(all_datasets) - 1})'
            )
        return all_datasets[args.dataset_index]

    return args.dataset_name


def _log_summary(*, config: ExperimentPipelineConfig) -> None:
    """Log a human-readable run summary."""
    model_name = getattr(config.model, 'model_name', type(config.model).__name__)
    attack_names = ', '.join(a.name for a in config.attacks)
    task_names = ', '.join(t.task_name for t in config.downstream_tasks)

    logger.info('Model:        %s', model_name)
    logger.info('Input dims:   %s', config.data.data_module.n_features)
    logger.info('Seq length:   %s', config.data.data_module.sequence_len)
    logger.info('Attacks:      %s', attack_names)
    logger.info('Tasks:        %s', task_names)
    logger.info('Max epochs:   %s', config.training.trainer_kwargs.get('max_epochs'))
    logger.info('Output dir:   %s', config.artifacts.run_dir)
    logger.info('Seed:         %s', config.seed)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the experiment runner."""
    args = _parse_args(argv)

    if not args.list_experiments:
        if args.experiment_id is None or args.data_root is None:
            parser = argparse.ArgumentParser(description='Run a robustness benchmark experiment.')
            parser.error('the following arguments are required: --experiment_id, --data_root')

        dataset_name = _resolve_dataset(args=args)

    if args.list_experiments:
        # Configure consistent logging for --list_experiments path
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        )
        ids = list_experiment_ids()
        logger.info('Available experiments:')
        for exp_id in ids:
            inst = EXPERIMENTS_REGISTRY[exp_id]
            family_parts = []
            for family, attacks in inst.attack_families.items():
                attack_names = ', '.join(a.name for a in attacks)
                family_parts.append(f'{family.value}: {attack_names}')
            summary = '; '.join(family_parts)
            logger.info('  - %s (%s)', exp_id, summary)
        return

    # Resolve experiment instance (with family filter & alias support)
    attack_family: AttackFamily | None = None
    if args.attack_family:
        attack_family = AttackFamily(args.attack_family)

    experiment = get_experiment_instance(
        experiment_id=args.experiment_id, attack_family=attack_family
    )

    # Compute config hash for hierarchical paths (D-05, D-06)
    model_param_dict = asdict(experiment.model_params)
    serializable_params = _make_json_serializable(model_param_dict)
    short_hash = compute_config_hash(
        model_params=cast('dict[str, object]', serializable_params),
        seed=args.seed,
    )

    # Resolve tracking mode (D-03: auto-detect HPC via SLURM_JOB_ID)
    tracking_mode = _resolve_tracking_mode(cli_mode=args.tracking_mode)

    # Build run name (needed for logger factory run_dir)
    run_name = _build_run_name(
        experiment_id=args.experiment_id,
        dataset_name=dataset_name,
        short_hash=short_hash,
        seed=args.seed,
        run_name_override=args.run_name,
    )

    # Create loggers (D-01, D-02)
    loggers = create_loggers(
        run_dir=args.output_dir / run_name,
        run_name=f'{args.experiment_id}_{dataset_name}_{args.seed}',
        tracking_mode=tracking_mode,
        persist_artifacts=True,
    )

    # Assemble pipeline config
    config = _build_pipeline_config(
        experiment_instance=experiment,
        dataset_name=dataset_name,
        data_root=args.data_root,
        output_dir=args.output_dir,
        run_name=run_name,
        seed=args.seed,
        num_workers=args.num_workers,
        forecasting_mode=args.forecasting_mode,
        max_epochs_override=args.max_epochs,
        encoding_batch_size_override=args.encoding_batch_size,
        batch_size_override=args.batch_size,
        attack_family=attack_family,
        loggers=loggers,
    )

    # Ensure run_dir exists (for logging)
    config.artifacts.run_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging (before pipeline execution)
    setup_logging(log_dir=config.artifacts.run_dir)

    if args.dry_run:
        _log_summary(config=config)
        return

    _log_summary(config=config)
    logger.info('Running experiment...')

    # Load checkpoint state for resume (unless --force)
    previous_state = None
    if not args.force:
        state_path = config.artifacts.run_dir / STATE_FILENAME
        if state_path.exists():
            previous_state = load_pipeline_state(path=state_path)
            logger.info('Resuming from checkpoint: %s', state_path)

    results = run_experiment_pipeline(
        config=config, previous_state=previous_state, force=args.force
    )

    logger.info('Experiment complete. Results saved to: %s', results.run_dir)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.basicConfig(level=logging.INFO)
        logger.info('Interrupted by user.')
        sys.exit(130)
    except Exception as e:
        logging.basicConfig(level=logging.WARNING)
        logger.warning('Error: %s', e)
        sys.exit(1)

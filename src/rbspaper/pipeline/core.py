"""Core orchestration logic for robust time-series experiments."""

from __future__ import annotations

from dataclasses import asdict, fields
from enum import Enum
import json
from pathlib import Path
import shutil
from typing import Any, cast, TYPE_CHECKING

import lightning.pytorch as pl
import numpy as np
import torch
from torch import Tensor

from src.rbspaper.attacks import execute_attack
from src.rbspaper.attacks.config import AttackExecutionContext, AttackParameters
from src.rbspaper.attacks.registry import get_threat_model, validate_attack_support
from src.rbspaper.evaluation import evaluate
from src.rbspaper.evaluation.enums import TimeSeriesEvaluationDownstreamTaskEnum
from src.rbspaper.models.encoding import encode_data
from src.rbspaper.models.utils import extract_features_from_batch
from src.rbspaper.pipeline.analysis import (
    compute_geometry_metrics,
    compute_linear_separability,
    compute_low_dim_artifacts,
    compute_shift_metrics,
)
from src.rbspaper.pipeline.config import (
    AttackRepresentationBundle,
    AttackScopePolicy,
    ExperimentPipelineResults,
    PartitionRepresentations,
    TaskRepresentationBundle,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from src.rbspaper.pipeline.config import (
        AttackRunConfig,
        ExperimentPipelineConfig,
        PipelineArtifactConfig,
    )


MIN_LABELED_BATCH_LENGTH = 2
MIN_NDIMS_TO_MERGE = 3


def _json_default(obj: object) -> object:
    """Convert non-JSON-native values used in experiment artifacts."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        array_obj = cast('np.ndarray', obj)
        return array_obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.value
    message = f'Object of type {type(obj).__name__!r} is not JSON serializable'
    raise TypeError(message)


def run_experiment_pipeline(*, config: ExperimentPipelineConfig) -> ExperimentPipelineResults:
    """Execute the full robust representation experiment pipeline.

    Args:
        config: Pipeline configuration object.

    Returns:
        Structured experiment results with artifacts and metrics.
    """
    pl.seed_everything(seed=config.seed)
    _preflight_pipeline_config(config=config)

    run_dir = _prepare_run_directory(config=config)
    data_module = config.data.data_module
    checkpoint_path = _train_model(config=config, run_dir=run_dir, data_module=data_module)

    partition_tensors = _collect_partition_tensors(data_module=data_module)
    scope = config.attack_scope.scope

    # SHARED_INPUT: generate all attacked inputs once before the downstream task loop.
    # Each attack uses its own context.task (== anchor_task, validated in preflight) as
    # the adversarial objective. The perturbed inputs are then reused for every task.
    shared_attacked_inputs: dict[str, tuple[Tensor, dict[str, Any]]] = {}
    if scope == AttackScopePolicy.SHARED_INPUT:
        for attack_config in config.attacks:
            attacked_inputs, metadata = _generate_attacked_inputs(
                attack_config=attack_config,
                model=config.model,
                clean_test_inputs=partition_tensors['test']['inputs'],
                clean_test_labels=partition_tensors['test']['labels'],
            )
            shared_attacked_inputs[attack_config.name] = (attacked_inputs, metadata)

    clean_representations: dict[str, TaskRepresentationBundle] = {}
    attacked_representations: dict[str, dict[str, AttackRepresentationBundle]] = {}
    downstream_metrics: dict[str, list[dict[str, Any]]] = {}

    for task_config in config.downstream_tasks:
        task_name = task_config.task_name

        clean_bundle = _extract_clean_representations(
            partition_tensors=partition_tensors,
            model=config.model,
            task_name=task_name,
            batch_size=config.encoding.batch_size,
            num_workers=config.encoding.num_workers,
        )
        clean_representations[task_name] = clean_bundle

        if config.artifacts.persist_artifacts and config.artifacts.save_clean_representations:
            _persist_clean_representations(
                run_dir=run_dir, task_name=task_name, clean_bundle=clean_bundle
            )

        attacked_by_name = _build_attacked_reps_for_task(
            scope=scope,
            attacks=config.attacks,
            shared_attacked_inputs=shared_attacked_inputs,
            model=config.model,
            task_name=task_name,
            clean_test_partition=clean_bundle.test,
            clean_test_inputs=partition_tensors['test']['inputs'],
            clean_test_labels=partition_tensors['test']['labels'],
            batch_size=config.encoding.batch_size,
            num_workers=config.encoding.num_workers,
            run_dir=run_dir,
            artifacts=config.artifacts,
        )
        attacked_representations[task_name] = attacked_by_name

        task_metrics = _evaluate_downstream(
            task_name=task_name,
            task_hyperparam_cases=task_config.hyperparam_cases(),
            clean_bundle=clean_bundle,
            attacked_by_name=attacked_by_name,
        )
        downstream_metrics[task_name] = task_metrics

        if config.artifacts.persist_artifacts:
            _save_json(file_path=run_dir / 'metrics' / f'{task_name}.json', data=task_metrics)

    analysis_results = _run_representation_analysis(
        config=config,
        clean_representations=clean_representations,
        attacked_representations=attacked_representations,
    )

    if config.artifacts.persist_artifacts:
        _save_json(file_path=run_dir / 'analysis' / 'analysis.json', data=analysis_results)

    results = ExperimentPipelineResults(
        run_dir=run_dir,
        model_name=getattr(config.model, 'model_name', type(config.model).__name__),
        checkpoint_path=checkpoint_path,
        clean_representations=clean_representations,
        attacked_representations=attacked_representations,
        downstream_metrics=downstream_metrics,
        analysis=analysis_results,
    )

    if config.artifacts.persist_artifacts:
        _save_json(
            file_path=run_dir / 'results_summary.json',
            data={
                'run_dir': run_dir,
                'model_name': results.model_name,
                'checkpoint_path': checkpoint_path,
                'downstream_metrics': downstream_metrics,
                'analysis': analysis_results,
            },
        )

    return results


def _prepare_run_directory(*, config: ExperimentPipelineConfig) -> Path:
    run_dir = config.artifacts.run_dir
    if config.artifacts.persist_artifacts:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _train_model(
    *, config: ExperimentPipelineConfig, run_dir: Path, data_module: pl.LightningDataModule
) -> Path | None:
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    canonical_checkpoint = checkpoints_dir / config.training.checkpoint_filename

    if config.training.reuse_trained_checkpoint and canonical_checkpoint.exists():
        return canonical_checkpoint

    trainer = pl.Trainer(**config.training.trainer_kwargs)
    trainer.fit(
        model=config.model, datamodule=data_module, ckpt_path=config.training.resume_from_checkpoint
    )

    best_model_path = _resolve_best_checkpoint_path(trainer=trainer)
    if best_model_path is not None and best_model_path.exists():
        shutil.copyfile(src=best_model_path, dst=canonical_checkpoint)
        return canonical_checkpoint

    trainer.save_checkpoint(filepath=str(canonical_checkpoint))
    return canonical_checkpoint if canonical_checkpoint.exists() else None


def _resolve_best_checkpoint_path(*, trainer: pl.Trainer) -> Path | None:
    checkpoint_callback = getattr(trainer, 'checkpoint_callback', None)
    if checkpoint_callback is None:
        return None

    best_model_path = getattr(checkpoint_callback, 'best_model_path', None)
    if not best_model_path:
        return None

    return Path(best_model_path)


def _extract_labels_from_batch(*, batch: object) -> Tensor:
    if isinstance(batch, tuple | list):
        if len(batch) < MIN_LABELED_BATCH_LENGTH:
            msg = 'Expected batch to include labels as second element.'
            raise ValueError(msg)
        labels = batch[1]
        if not isinstance(labels, Tensor):
            msg = f'Unsupported label type: {type(labels)}'
            raise TypeError(msg)
        return labels

    msg = f'Unsupported labeled batch type: {type(batch)}'
    raise TypeError(msg)


def _collect_partition_tensor_from_dataloader(*, dataloader: DataLoader) -> dict[str, Tensor]:
    all_inputs: list[Tensor] = []
    all_labels: list[Tensor] = []

    for batch in dataloader:
        inputs = extract_features_from_batch(batch=batch)
        labels = _extract_labels_from_batch(batch=batch)
        all_inputs.append(inputs)
        all_labels.append(labels)

    return {
        'inputs': torch.cat(tensors=all_inputs, dim=0),
        'labels': torch.cat(tensors=all_labels, dim=0),
    }


def _collect_partition_tensors(
    *, data_module: pl.LightningDataModule
) -> dict[str, dict[str, Tensor]]:
    return {
        'train': _collect_partition_tensor_from_dataloader(
            dataloader=data_module.train_dataloader()
        ),
        'valid': _collect_partition_tensor_from_dataloader(dataloader=data_module.val_dataloader()),
        'test': _collect_partition_tensor_from_dataloader(dataloader=data_module.test_dataloader()),
    }


def _merge_first_two_dims(*, tensor: Tensor) -> Tensor:
    if tensor.ndim < MIN_NDIMS_TO_MERGE:
        return tensor
    new_shape = (tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
    return tensor.reshape(*new_shape)


def _normalize_partition_for_task(
    *, features: Tensor, labels: Tensor, task_name: str
) -> PartitionRepresentations:
    if task_name == TimeSeriesEvaluationDownstreamTaskEnum.FORECASTING.value:
        normalized_features = features.detach().cpu().numpy().reshape(features.shape[0], -1)
        normalized_labels = labels.detach().cpu().numpy().reshape(labels.shape[0], -1)
        return PartitionRepresentations(features=normalized_features, labels=normalized_labels)

    normalized_features = _merge_first_two_dims(tensor=features).detach().cpu().numpy()
    normalized_labels = labels.detach().cpu().numpy().reshape(-1)
    return PartitionRepresentations(features=normalized_features, labels=normalized_labels)


def _extract_clean_representations(
    *,
    partition_tensors: dict[str, dict[str, Tensor]],
    model: pl.LightningModule,
    task_name: str,
    batch_size: int,
    num_workers: int,
) -> TaskRepresentationBundle:
    train_features = encode_data(
        data=partition_tensors['train']['inputs'],
        model=model,  # ty: ignore[invalid-argument-type] # why: ExperimentPipelineConfig.model is pl.LightningModule; encode_data expects TS2Vec|AutoTCL|CoST; widening would cause call-non-callable on model.encode
        batch_size=batch_size,
        num_workers=num_workers,
        downstream_task=task_name,
    )
    valid_features = encode_data(
        data=partition_tensors['valid']['inputs'],
        model=model,  # ty: ignore[invalid-argument-type] # why: ExperimentPipelineConfig.model is pl.LightningModule; encode_data expects TS2Vec|AutoTCL|CoST; widening would cause call-non-callable on model.encode
        batch_size=batch_size,
        num_workers=num_workers,
        downstream_task=task_name,
    )
    test_features = encode_data(
        data=partition_tensors['test']['inputs'],
        model=model,  # ty: ignore[invalid-argument-type] # why: ExperimentPipelineConfig.model is pl.LightningModule; encode_data expects TS2Vec|AutoTCL|CoST; widening would cause call-non-callable on model.encode
        batch_size=batch_size,
        num_workers=num_workers,
        downstream_task=task_name,
    )

    return TaskRepresentationBundle(
        train=_normalize_partition_for_task(
            features=train_features,
            labels=partition_tensors['train']['labels'],
            task_name=task_name,
        ),
        valid=_normalize_partition_for_task(
            features=valid_features,
            labels=partition_tensors['valid']['labels'],
            task_name=task_name,
        ),
        test=_normalize_partition_for_task(
            features=test_features, labels=partition_tensors['test']['labels'], task_name=task_name
        ),
    )


def _build_attack_kwargs_from_parameters(*, parameters: AttackParameters) -> dict[str, Any]:
    ignored_fields = {'backend', 'clip_min', 'clip_max'}
    kwargs: dict[str, Any] = {}
    for field in fields(parameters):
        if field.name in ignored_fields:
            continue
        kwargs[field.name] = getattr(parameters, field.name)
    return kwargs


def _select_attacks_for_task(
    *, task_name: str, attacks: tuple[AttackRunConfig, ...]
) -> tuple[AttackRunConfig, ...]:
    return tuple(attack for attack in attacks if attack.context.task.value == task_name)


def _resolve_context(
    *, context: AttackExecutionContext, parameters: AttackParameters
) -> AttackExecutionContext:
    if context.clip_min is not None and context.clip_max is not None:
        return context

    return AttackExecutionContext(
        task=context.task,
        objective=context.objective,
        class_count=context.class_count,
        clip_min=parameters.clip_min,
        clip_max=parameters.clip_max,
        horizon_start=context.horizon_start,
        horizon_end=context.horizon_end,
    )


def _generate_attacked_inputs(
    *,
    attack_config: AttackRunConfig,
    model: pl.LightningModule,
    clean_test_inputs: Tensor,
    clean_test_labels: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    """Execute the attack and return perturbed inputs with metadata (no encoding step).

    Args:
        attack_config: Attack run configuration including parameters and context.
        model: Trained model used as the attack's white-box or query oracle.
        clean_test_inputs: Raw test inputs to perturb.
        clean_test_labels: Supervision signal for the attack objective.

    Returns:
        Tuple of (attacked_inputs tensor, metadata dict).
    """
    context = _resolve_context(context=attack_config.context, parameters=attack_config.parameters)
    attack_kwargs = _build_attack_kwargs_from_parameters(parameters=attack_config.parameters)

    if attack_config.query_budget is not None:
        attack_kwargs.setdefault('max_queries', attack_config.query_budget.max_queries)

    attacked_result = execute_attack(
        attack=attack_config.parameters.attack_method,
        inputs=clean_test_inputs,
        supervision=torch.as_tensor(clean_test_labels, device=clean_test_inputs.device),
        model=model,
        context=context,
        backend=attack_config.parameters.backend,
        return_metadata=True,
        **attack_kwargs,
    )
    if not isinstance(attacked_result, tuple):
        message = (
            'Attack execution must return (attacked_inputs, metadata) when return_metadata=True.'
        )
        raise TypeError(message)

    attacked_inputs, metadata = attacked_result
    metadata_dict = asdict(metadata) if hasattr(metadata, '__dataclass_fields__') else {}
    return attacked_inputs, metadata_dict


def _encode_attacked_inputs_for_task(
    *,
    attacked_inputs: Tensor,
    metadata: dict[str, Any],
    model: pl.LightningModule,
    task_name: str,
    clean_test_labels: Tensor,
    batch_size: int,
    num_workers: int,
) -> AttackRepresentationBundle:
    """Encode pre-generated attacked inputs for a specific downstream task.

    Args:
        attacked_inputs: Perturbed test inputs produced by a prior attack pass.
        metadata: Attack execution metadata dict.
        model: Trained model used to extract representations.
        task_name: Name of the downstream task driving the encoding mode.
        clean_test_labels: Labels corresponding to the test partition.
        batch_size: Encoding batch size.
        num_workers: DataLoader workers for encoding.

    Returns:
        AttackRepresentationBundle with encoded test representations and metadata.
    """
    attacked_features = encode_data(
        data=attacked_inputs,
        model=model,  # ty: ignore[invalid-argument-type] # why: ExperimentPipelineConfig.model is pl.LightningModule; encode_data expects TS2Vec|AutoTCL|CoST; widening would cause call-non-callable on model.encode
        batch_size=batch_size,
        num_workers=num_workers,
        downstream_task=task_name,
    )
    attacked_partition = _normalize_partition_for_task(
        features=attacked_features, labels=clean_test_labels, task_name=task_name
    )
    return AttackRepresentationBundle(test=attacked_partition, metadata=metadata)


def _extract_attacked_representations(
    *,
    attack_config: AttackRunConfig,
    model: pl.LightningModule,
    task_name: str,
    clean_test_partition: PartitionRepresentations,
    clean_test_inputs: Tensor,
    batch_size: int,
    num_workers: int,
) -> AttackRepresentationBundle:
    """Generate and encode attacked representations for one attack in TASK_CONDITIONED mode."""
    clean_test_labels = torch.as_tensor(clean_test_partition.labels)
    attacked_inputs, metadata = _generate_attacked_inputs(
        attack_config=attack_config,
        model=model,
        clean_test_inputs=clean_test_inputs,
        clean_test_labels=clean_test_labels,
    )
    return _encode_attacked_inputs_for_task(
        attacked_inputs=attacked_inputs,
        metadata=metadata,
        model=model,
        task_name=task_name,
        clean_test_labels=clean_test_labels,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def _build_attacked_reps_for_task(
    *,
    scope: AttackScopePolicy,
    attacks: tuple[AttackRunConfig, ...],
    shared_attacked_inputs: dict[str, tuple[Tensor, dict[str, Any]]],
    model: pl.LightningModule,
    task_name: str,
    clean_test_partition: PartitionRepresentations,
    clean_test_inputs: Tensor,
    clean_test_labels: Tensor,
    batch_size: int,
    num_workers: int,
    run_dir: Path,
    artifacts: PipelineArtifactConfig,
) -> dict[str, AttackRepresentationBundle]:
    """Build attacked representations for one downstream task under the configured scope.

    In TASK_CONDITIONED mode, only attacks whose context.task matches task_name are run.
    In SHARED_INPUT mode, pre-generated attacked inputs are encoded for this task.

    Args:
        scope: Attack scope policy controlling how attacks are executed.
        attacks: Full tuple of attack run configs (used in TASK_CONDITIONED mode).
        shared_attacked_inputs: Pre-computed attacked inputs keyed by attack name
            (populated before the task loop in SHARED_INPUT mode).
        model: Trained model used for encoding.
        task_name: Downstream task name for this evaluation pass.
        clean_test_partition: Clean test representations (labels used in TASK_CONDITIONED).
        clean_test_inputs: Raw test inputs used in TASK_CONDITIONED attack execution.
        clean_test_labels: Raw test labels used in SHARED_INPUT encoding.
        batch_size: Encoding batch size.
        num_workers: DataLoader workers for encoding.
        run_dir: Run output directory for artifact persistence.
        artifacts: Artifact configuration controlling persistence behaviour.

    Returns:
        Mapping of attack name to attacked representation bundle for this task.
    """
    should_persist = artifacts.persist_artifacts and artifacts.save_attacked_representations
    attacked_by_name: dict[str, AttackRepresentationBundle] = {}

    if scope == AttackScopePolicy.TASK_CONDITIONED:
        task_attacks = _select_attacks_for_task(task_name=task_name, attacks=attacks)
        for attack_config in task_attacks:
            bundle = _extract_attacked_representations(
                attack_config=attack_config,
                model=model,
                task_name=task_name,
                clean_test_partition=clean_test_partition,
                clean_test_inputs=clean_test_inputs,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            attacked_by_name[attack_config.name] = bundle
            if should_persist:
                _persist_attacked_representations(
                    run_dir=run_dir,
                    task_name=task_name,
                    attack_name=attack_config.name,
                    attacked_bundle=bundle,
                )
    else:  # SHARED_INPUT
        for attack_name, (attacked_inputs, metadata) in shared_attacked_inputs.items():
            bundle = _encode_attacked_inputs_for_task(
                attacked_inputs=attacked_inputs,
                metadata=metadata,
                model=model,
                task_name=task_name,
                clean_test_labels=clean_test_labels,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            attacked_by_name[attack_name] = bundle
            if should_persist:
                _persist_attacked_representations(
                    run_dir=run_dir,
                    task_name=task_name,
                    attack_name=attack_name,
                    attacked_bundle=bundle,
                )

    return attacked_by_name


def _validate_attack_scope_binding(
    *,
    attack_config: AttackRunConfig,
    scope: AttackScopePolicy,
    downstream_task_names: set[str],
    anchor_task: object,
) -> None:
    """Validate that an attack is correctly bound to the configured scope."""
    if scope == AttackScopePolicy.TASK_CONDITIONED:
        if attack_config.context.task.value not in downstream_task_names:
            message = (
                f'Attack run {attack_config.name!r} targets task '
                f'{attack_config.context.task.value!r} not present in downstream tasks.'
            )
            raise ValueError(message)
    elif attack_config.context.task != anchor_task:  # SHARED_INPUT
        message = (
            f'Attack run {attack_config.name!r} has context.task '
            f'{attack_config.context.task!r} which differs from anchor_task '
            f'{anchor_task!r}. In SHARED_INPUT mode all attacks must use the anchor_task objective.'
        )
        raise ValueError(message)


def _preflight_pipeline_config(*, config: ExperimentPipelineConfig) -> None:
    downstream_task_names = {task.task_name for task in config.downstream_tasks}
    if not downstream_task_names:
        message = 'Pipeline requires at least one downstream task configuration.'
        raise ValueError(message)

    scope = config.attack_scope.scope

    # Dataset profile validation: every configured downstream task must be in allowed_eval_tasks.
    if config.data.profile is not None:
        allowed_names = {t.value for t in config.data.profile.allowed_eval_tasks}
        for task_config in config.downstream_tasks:
            if task_config.task_name not in allowed_names:
                message = (
                    f'Downstream task {task_config.task_name!r} is not in '
                    'allowed_eval_tasks for the configured dataset profile.'
                )
                raise ValueError(message)

    seen_names: set[str] = set()
    for attack_config in config.attacks:
        if attack_config.name in seen_names:
            message = f'Duplicate attack run name: {attack_config.name!r}'
            raise ValueError(message)
        seen_names.add(attack_config.name)

        attack_config.validate()
        _validate_attack_scope_binding(
            attack_config=attack_config,
            scope=scope,
            downstream_task_names=downstream_task_names,
            anchor_task=config.attack_scope.anchor_task,
        )
        validate_attack_support(
            attack=attack_config.parameters.attack_method,
            task=attack_config.context.task,
            backend=attack_config.parameters.backend,
            has_supervision=True,
        )

        if attack_config.query_budget is not None:
            threat_model = get_threat_model(attack=attack_config.parameters.attack_method)
            if threat_model.value == 'white_box':
                message = (
                    f'Attack run {attack_config.name!r} sets query budget for white-box '
                    f'attack {attack_config.parameters.attack_method.value!r}. '
                    'Remove query_budget or choose a black-box/gray-box method.'
                )
                raise ValueError(message)


def _resolve_case_defaults(*, task_name: str, case: dict[str, Any]) -> dict[str, Any]:
    resolved_case = dict(case)
    if (
        task_name == TimeSeriesEvaluationDownstreamTaskEnum.CLASSIFICATION.value
        and 'evaluation_protocol' not in resolved_case
    ):
        resolved_case['evaluation_protocol'] = 'svm'
    return resolved_case


def _evaluate_downstream(
    *,
    task_name: str,
    task_hyperparam_cases: list[dict[str, Any]],
    clean_bundle: TaskRepresentationBundle,
    attacked_by_name: dict[str, AttackRepresentationBundle],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []

    for case in task_hyperparam_cases:
        resolved_case = _resolve_case_defaults(task_name=task_name, case=case)

        clean_metrics = evaluate(
            downstream_task=task_name,
            train_features=clean_bundle.train.features,
            train_labels=clean_bundle.train.labels,
            valid_features=clean_bundle.valid.features,
            valid_labels=clean_bundle.valid.labels,
            test_features=clean_bundle.test.features,
            test_labels=clean_bundle.test.labels,
            downstream_task_params=resolved_case,
        )
        metrics.append(
            {
                'scope': 'clean',
                'attack': None,
                'task_name': task_name,
                'hyperparams': resolved_case,
                'metrics': clean_metrics,
            }
        )

        for attack_name, attack_bundle in attacked_by_name.items():
            attacked_metrics = evaluate(
                downstream_task=task_name,
                train_features=clean_bundle.train.features,
                train_labels=clean_bundle.train.labels,
                valid_features=clean_bundle.valid.features,
                valid_labels=clean_bundle.valid.labels,
                test_features=attack_bundle.test.features,
                test_labels=attack_bundle.test.labels,
                downstream_task_params=resolved_case,
            )
            metrics.append(
                {
                    'scope': 'attacked',
                    'attack': attack_name,
                    'task_name': task_name,
                    'hyperparams': resolved_case,
                    'metrics': attacked_metrics,
                }
            )

    return metrics


def _run_representation_analysis(
    *,
    config: ExperimentPipelineConfig,
    clean_representations: dict[str, TaskRepresentationBundle],
    attacked_representations: dict[str, dict[str, AttackRepresentationBundle]],
) -> dict[str, Any]:
    analysis_results: dict[str, Any] = {}

    for task_name, clean_bundle in clean_representations.items():
        task_analysis: dict[str, Any] = {}
        attacked_by_name = attacked_representations.get(task_name, {})

        if config.analysis.enable_geometry:
            task_analysis['clean_geometry'] = compute_geometry_metrics(
                features=clean_bundle.test.features, labels=clean_bundle.test.labels
            )
            task_analysis['attacked_geometry'] = {
                attack_name: compute_geometry_metrics(
                    features=attack_bundle.test.features, labels=attack_bundle.test.labels
                )
                for attack_name, attack_bundle in attacked_by_name.items()
            }

        if config.analysis.enable_shift:
            task_analysis['attacked_shift'] = {
                attack_name: compute_shift_metrics(
                    clean_features=clean_bundle.test.features,
                    attacked_features=attack_bundle.test.features,
                )
                for attack_name, attack_bundle in attacked_by_name.items()
            }

        if (
            config.analysis.enable_linear_separability
            and task_name == TimeSeriesEvaluationDownstreamTaskEnum.CLASSIFICATION.value
        ):
            task_analysis['clean_linear_separability'] = compute_linear_separability(
                train_features=clean_bundle.train.features,
                train_labels=clean_bundle.train.labels,
                valid_features=clean_bundle.valid.features,
                valid_labels=clean_bundle.valid.labels,
                test_features=clean_bundle.test.features,
                test_labels=clean_bundle.test.labels,
            )
            task_analysis['attacked_linear_separability'] = {
                attack_name: compute_linear_separability(
                    train_features=clean_bundle.train.features,
                    train_labels=clean_bundle.train.labels,
                    valid_features=clean_bundle.valid.features,
                    valid_labels=clean_bundle.valid.labels,
                    test_features=attack_bundle.test.features,
                    test_labels=attack_bundle.test.labels,
                )
                for attack_name, attack_bundle in attacked_by_name.items()
            }

        if config.analysis.enable_low_dim_artifacts:
            task_analysis['clean_low_dim'] = compute_low_dim_artifacts(
                features=clean_bundle.test.features,
                labels=clean_bundle.test.labels,
                max_samples=config.analysis.max_visualization_samples,
                seed=config.seed,
            )
            task_analysis['attacked_low_dim'] = {
                attack_name: compute_low_dim_artifacts(
                    features=attack_bundle.test.features,
                    labels=attack_bundle.test.labels,
                    max_samples=config.analysis.max_visualization_samples,
                    seed=config.seed,
                )
                for attack_name, attack_bundle in attacked_by_name.items()
            }

        analysis_results[task_name] = task_analysis

    return analysis_results


def _persist_clean_representations(
    *, run_dir: Path, task_name: str, clean_bundle: TaskRepresentationBundle
) -> None:
    output_dir = run_dir / 'representations' / task_name / 'clean'
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_npz(
        file_path=output_dir / 'train.npz',
        features=clean_bundle.train.features,
        labels=clean_bundle.train.labels,
    )
    _save_npz(
        file_path=output_dir / 'valid.npz',
        features=clean_bundle.valid.features,
        labels=clean_bundle.valid.labels,
    )
    _save_npz(
        file_path=output_dir / 'test.npz',
        features=clean_bundle.test.features,
        labels=clean_bundle.test.labels,
    )


def _persist_attacked_representations(
    *, run_dir: Path, task_name: str, attack_name: str, attacked_bundle: AttackRepresentationBundle
) -> None:
    output_dir = run_dir / 'representations' / task_name / 'attacks' / attack_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_npz(
        file_path=output_dir / 'test.npz',
        features=attacked_bundle.test.features,
        labels=attacked_bundle.test.labels,
    )
    _save_json(file_path=output_dir / 'attack_metadata.json', data=attacked_bundle.metadata)


def _save_npz(*, file_path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(file=str(file_path), features=features, labels=labels)


def _save_json(*, file_path: Path, data: object) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open(mode='w', encoding='utf-8') as file_obj:
        json.dump(obj=data, fp=file_obj, indent=2, default=_json_default)

"""Typed configuration objects for the experiment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import product
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import lightning.pytorch as pl
    from lightning.pytorch.loggers import Logger
    import numpy as np

    from src.rbspaper.attacks.config import AttackExecutionContext, AttackParameters
    from src.rbspaper.enums.general import TimeSeriesDownstreamTask


class TargetSelectionStrategy(StrEnum):
    """Policy used to derive target labels for targeted attacks."""

    RANDOM = 'random'
    LEAST_LIKELY = 'least_likely'
    MOST_LIKELY = 'most_likely'
    FIXED = 'fixed'


class QueryBudgetPolicy(StrEnum):
    """Policy used when black-box query budget is exhausted."""

    STRICT = 'strict'
    SOFT = 'soft'
    SCALE = 'scale'


class AttackScopePolicy(StrEnum):
    """Policy controlling how attacks relate to downstream task evaluation.

    TASK_CONDITIONED: each attack is bound to a specific downstream task; only
        attacks whose context.task matches the current task are run for that task.
    SHARED_INPUT: all attacks run once using anchor_task as the adversarial objective;
        the resulting perturbed inputs are reused across all configured downstream tasks.
    """

    TASK_CONDITIONED = 'task_conditioned'
    SHARED_INPUT = 'shared_input'


@dataclass(frozen=True)
class AttackScopeConfig:
    """Controls the scope relationship between attacks and downstream task evaluation.

    In TASK_CONDITIONED mode (default), each attack is bound to a specific downstream
    task and only runs when that task is being evaluated. In SHARED_INPUT mode, all
    attacks run once using anchor_task as the adversarial objective, and the resulting
    perturbed inputs are reused across every configured downstream task.
    """

    scope: AttackScopePolicy = AttackScopePolicy.TASK_CONDITIONED
    anchor_task: TimeSeriesDownstreamTask | None = None

    def __post_init__(self) -> None:
        """Validate that anchor_task is provided when scope is SHARED_INPUT."""
        if self.scope == AttackScopePolicy.SHARED_INPUT and self.anchor_task is None:
            message = 'anchor_task is required when attack_scope is SHARED_INPUT.'
            raise ValueError(message)


@dataclass(frozen=True)
class DatasetTaskProfile:
    """Minimal contract defining which downstream tasks are valid for a dataset.

    This is a planning-time bridge: once the data registry is implemented, this profile
    can be derived automatically from dataset metadata instead of being specified manually.

    Args:
        primary_task: The main supervised task for this dataset (used as the attack
            anchor in SHARED_INPUT mode).
        allowed_eval_tasks: Set of downstream tasks that are scientifically valid for
            this dataset. Must include primary_task.
    """

    primary_task: TimeSeriesDownstreamTask
    allowed_eval_tasks: frozenset[TimeSeriesDownstreamTask]

    def __post_init__(self) -> None:
        """Validate that primary_task is included in allowed_eval_tasks."""
        if self.primary_task not in self.allowed_eval_tasks:
            message = f'primary_task {self.primary_task!r} must be included in allowed_eval_tasks.'
            raise ValueError(message)


@dataclass(frozen=True)
class AttackTargetingConfig:
    """Target-label derivation policy for targeted attacks."""

    strategy: TargetSelectionStrategy = TargetSelectionStrategy.RANDOM
    fixed_target: int | None = None
    avoid_correct: bool = True


@dataclass(frozen=True)
class QueryBudgetConfig:
    """Query budget policy for black-box attack runs."""

    max_queries: int
    policy: QueryBudgetPolicy = QueryBudgetPolicy.SOFT
    track_per_sample: bool = False


@dataclass(frozen=True)
class SurrogateAttackConfig:
    """Gray-box surrogate model details for transfer attacks."""

    name: str
    model: pl.LightningModule | None = None
    checkpoint_path: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttackMetadataConfig:
    """Optional runtime metadata controls for attack execution."""

    track_queries: bool = True
    track_per_sample_success: bool = False
    track_execution_time: bool = False
    extra_fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RepresentationEncodingConfig:
    """Controls representation extraction behavior."""

    batch_size: int = 256
    num_workers: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    """Controls model training behavior and trainer setup."""

    trainer_kwargs: dict[str, Any] = field(default_factory=dict)
    resume_from_checkpoint: str | None = None
    reuse_trained_checkpoint: bool = True
    checkpoint_filename: str = 'best.ckpt'


@dataclass(frozen=True)
class AttackRunConfig:
    """Defines one attack execution with attack parameters and context."""

    name: str
    parameters: AttackParameters
    context: AttackExecutionContext
    targeting: AttackTargetingConfig | None = None
    query_budget: QueryBudgetConfig | None = None
    surrogate: SurrogateAttackConfig | None = None
    metadata: AttackMetadataConfig = field(default_factory=AttackMetadataConfig)

    def validate(self) -> None:
        """Validate attack-run settings consistency."""
        if self.query_budget is not None and self.query_budget.max_queries <= 0:
            message = f'Attack run {self.name!r} query budget must be positive.'
            raise ValueError(message)

        if (
            self.surrogate is not None
            and self.surrogate.model is None
            and self.surrogate.checkpoint_path is None
        ):
            message = (
                f'Attack run {self.name!r} configured surrogate but neither model nor '
                'checkpoint_path was provided.'
            )
            raise ValueError(message)


@dataclass(frozen=True)
class DownstreamTaskConfig:
    """Defines downstream evaluation task and a hyperparameter search space."""

    task_name: str
    hyperparams: dict[str, Any] = field(default_factory=dict)

    def hyperparam_cases(self) -> list[dict[str, Any]]:
        """Expand the hyperparameter dictionary into cartesian-product cases."""
        if not self.hyperparams:
            return [{}]

        keys = list(self.hyperparams.keys())
        value_lists = [
            value if isinstance(value, list | tuple) else [value]
            for value in self.hyperparams.values()
        ]

        return [dict(zip(keys, values, strict=True)) for values in product(*value_lists)]


@dataclass(frozen=True)
class RepresentationAnalysisConfig:
    """Toggles representation analysis components."""

    enable_linear_separability: bool = True
    enable_geometry: bool = True
    enable_shift: bool = True
    enable_low_dim_artifacts: bool = True
    max_visualization_samples: int = 2000


@dataclass(frozen=True)
class PipelineArtifactConfig:
    """Controls disk artifact persistence and run identity."""

    output_dir: Path
    run_name: str
    persist_artifacts: bool = True
    save_clean_representations: bool = True
    save_attacked_representations: bool = True

    @property
    def run_dir(self) -> Path:
        """Return the run-specific output directory."""
        return self.output_dir / self.run_name


# ======== Run Identity ========


def build_hierarchical_run_name(
    *,
    experiment_id: str,
    short_hash: str,
    seed: int,
    dataset_name: str,
) -> str:
    """Build a hierarchical run name for deterministic output paths.

    Produces a path segment of the form
    '{experiment_id}/{short_hash}/seed_{seed}/{dataset_name}'.

    Args:
        experiment_id: Registered experiment identifier.
        short_hash: 8-character SHA-256 prefix of model params + seed.
        seed: Random seed for reproducibility.
        dataset_name: Registered dataset name.

    Returns:
        Slash-separated path string (e.g. 'ts2vec/a1b2c3d4/seed_42/Coffee').
    """
    return f'{experiment_id}/{short_hash}/seed_{seed}/{dataset_name}'


@dataclass(frozen=True)
class DataConfig:
    """Groups the dataset data source and its associated task profile.

    Bundles everything the pipeline needs to know about the data: the Lightning
    data module used during training and evaluation, and an optional task profile
    that constrains which downstream tasks are scientifically valid for this dataset.

    Args:
        data_module: Prepared LightningDataModule providing train/val/test splits.
        profile: Optional dataset task profile. When provided, preflight validation
            ensures configured downstream tasks are a subset of allowed_eval_tasks.
    """

    data_module: pl.LightningDataModule
    profile: DatasetTaskProfile | None = None


@dataclass(frozen=True)
class ExperimentPipelineConfig:
    """Top-level pipeline configuration."""

    model: pl.LightningModule
    data: DataConfig
    training: TrainingConfig
    encoding: RepresentationEncodingConfig
    attacks: tuple[AttackRunConfig, ...]
    downstream_tasks: tuple[DownstreamTaskConfig, ...]
    analysis: RepresentationAnalysisConfig
    artifacts: PipelineArtifactConfig
    seed: int = 42
    attack_scope: AttackScopeConfig = field(default_factory=AttackScopeConfig)
    loggers: tuple[Logger, ...] = field(default_factory=tuple)


# ======== Experiments Results Configuration Objects ========


@dataclass
class PartitionRepresentations:
    """Represents features and labels for one data partition."""

    features: np.ndarray
    labels: np.ndarray


@dataclass
class TaskRepresentationBundle:
    """Contains clean representations for train/val/test for one task."""

    train: PartitionRepresentations
    valid: PartitionRepresentations
    test: PartitionRepresentations


@dataclass
class AttackRepresentationBundle:
    """Contains attacked test representations and runtime metadata."""

    test: PartitionRepresentations
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentPipelineResults:
    """Top-level experiment pipeline output object."""

    run_dir: Path
    model_name: str
    checkpoint_path: Path | None
    clean_representations: dict[str, TaskRepresentationBundle] = field(default_factory=dict)
    attacked_representations: dict[str, dict[str, AttackRepresentationBundle]] = field(
        default_factory=dict
    )
    downstream_metrics: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)

"""Experiment instance definitions and registry.

An experiment instance bundles model parameters, attack configurations,
training settings, and downstream tasks into a single resolvable unit.
The runner resolves the instance by ID, then assembles a full pipeline.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any
import warnings

from src.rbspaper.attacks.config import (
    AttackExecutionContext,
    BimAttackParameters,
    FgsmAttackParameters,
    PgdAttackParameters,
)
from src.rbspaper.attacks.enums import AttackBackend, AttackFamily, AttackObjective
from src.rbspaper.enums.general import TimeSeriesDownstreamTask
from src.rbspaper.models.augmentation.config import (
    AutoTCLNeuralNetworkAugmentationParameters,
    CropShiftAugmentationParameters,
)
from src.rbspaper.models.augmentation.enums import AutoTCLAugmentationMode, TS2VecAugmentationMode
from src.rbspaper.models.config import (
    AutoTCLModelParameters,
    ModelParameters,
    TS2VecModelParameters,
)
from src.rbspaper.models.encoders.masking import MaskMode
from src.rbspaper.pipeline.config import AttackMetadataConfig, AttackRunConfig, AttackScopeConfig

__all__ = [
    'EXPERIMENTS_REGISTRY',
    '_EXPERIMENT_ID_ALIASES',
    'ExperimentInstance',
    'get_experiment_instance',
    'list_experiment_ids',
]


@dataclass
class ExperimentInstance:
    """One robustness experiment: model + attack(s) evaluated on dataset(s).

    Bundles everything needed to run a model through the pipeline:
    - Model hyperparameters (input_dims filled at runtime from datamodule)
    - Attack configurations organized by family
    - Encoding batch size
    - Training limits
    - Downstream evaluation tasks

    Args:
        id: Unique experiment identifier.
        model_params: Model configuration (input_dims resolved at runtime).
        attack_families: Named attack configs grouped by threat-model family.
        encoding_batch_size: Batch size for representation encoding.
        max_epochs: Upper bound for training epochs.
        downstream_tasks: Tasks to evaluate on the representations.
        attack_scope: Controls attack-to-task relationship.
        trainer_kwargs: Overrides passed directly to pl.Trainer.
    """

    id: str
    model_params: ModelParameters
    attack_families: dict[AttackFamily, tuple[AttackRunConfig, ...]]
    encoding_batch_size: int = 256
    max_epochs: int = 500
    downstream_tasks: tuple[str, ...] = ('classification',)
    attack_scope: AttackScopeConfig = field(default_factory=AttackScopeConfig)
    trainer_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def attack_params(self) -> tuple[AttackRunConfig, ...]:
        """Flatten all attack families into a single tuple."""
        return tuple(attack for family in self.attack_families.values() for attack in family)


# ---------------------------------------------------------------------------
# Default trainer kwargs
# ---------------------------------------------------------------------------

_DEFAULT_TRAINER_KWARGS: dict[str, Any] = {
    'max_epochs': 500,
    'enable_progress_bar': True,
    'enable_model_summary': True,
    'logger': False,
}


# ---------------------------------------------------------------------------
# Helper factories for common attack configs
# ---------------------------------------------------------------------------


def _fgsm_attack_run(
    *,
    name: str = 'fgsm',
    epsilon: float = 8.0 / 255.0,
    task: TimeSeriesDownstreamTask = TimeSeriesDownstreamTask.CLASSIFICATION,
) -> AttackRunConfig:
    """Build a standard FGSM attack run."""
    return AttackRunConfig(
        name=name,
        parameters=FgsmAttackParameters(backend=AttackBackend.ART, epsilon=epsilon),
        context=AttackExecutionContext(task=task, objective=AttackObjective.UNTARGETED),
        metadata=AttackMetadataConfig(),
    )


def _pgd_attack_run(
    *,
    name: str = 'pgd',
    epsilon: float = 8.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 10,
    task: TimeSeriesDownstreamTask = TimeSeriesDownstreamTask.CLASSIFICATION,
) -> AttackRunConfig:
    """Build a standard PGD attack run."""
    return AttackRunConfig(
        name=name,
        parameters=PgdAttackParameters(
            backend=AttackBackend.ART, epsilon=epsilon, alpha=alpha, steps=steps, random_start=True
        ),
        context=AttackExecutionContext(task=task, objective=AttackObjective.UNTARGETED),
        metadata=AttackMetadataConfig(),
    )


def _bim_attack_run(
    *,
    name: str = 'bim',
    epsilon: float = 8.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 10,
    task: TimeSeriesDownstreamTask = TimeSeriesDownstreamTask.CLASSIFICATION,
) -> AttackRunConfig:
    """Build a standard BIM attack run."""
    return AttackRunConfig(
        name=name,
        parameters=BimAttackParameters(
            backend=AttackBackend.ART, epsilon=epsilon, alpha=alpha, steps=steps
        ),
        context=AttackExecutionContext(task=task, objective=AttackObjective.UNTARGETED),
        metadata=AttackMetadataConfig(),
    )


# ---------------------------------------------------------------------------
# Helper factories for common model configs
# ---------------------------------------------------------------------------


def _ts2vec_params(
    *,
    augmentation_mode: TS2VecAugmentationMode = TS2VecAugmentationMode.CROP_SHIFT,
    hidden_dims: int = 64,
    output_dims: int = 320,
    depth: int = 10,
) -> TS2VecModelParameters:
    """Build default TS2Vec model parameters."""
    return TS2VecModelParameters(
        augmentation_mode=augmentation_mode,
        augmentation_method_params=CropShiftAugmentationParameters(),
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        depth=depth,
        dropout_rate=0.1,
        conv_kernel_size=3,
        mask_mode=MaskMode.BINOMIAL,
        learning_rate=1e-3,
    )


def _autotcl_params(
    *, hidden_dims: int = 64, output_dims: int = 320, depth: int = 10
) -> AutoTCLModelParameters:
    """Build default AutoTCL model parameters."""
    return AutoTCLModelParameters(
        augmentation_mode=AutoTCLAugmentationMode.NEURAL_NETWORK,
        augmentation_method_params=AutoTCLNeuralNetworkAugmentationParameters(),
        kernel_sizes=[3, 5, 7],
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        depth=depth,
        dropout_rate=0.1,
        conv_kernel_size=3,
        mask_mode=MaskMode.BINOMIAL,
        learning_rate=1e-3,
    )


# ---------------------------------------------------------------------------
# Alias map for backward compatibility
# ---------------------------------------------------------------------------

_EXPERIMENT_ID_ALIASES: dict[str, str] = {
    'ts2vec_fgsm': 'ts2vec',
    'ts2vec_pgd': 'ts2vec',
    'ts2vec_bim': 'ts2vec',
    'ts2vec_multi': 'ts2vec',
    'autotcl_fgsm': 'autotcl',
    'autotcl_pgd': 'autotcl',
    'autotcl_multi': 'autotcl',
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EXPERIMENTS_REGISTRY: dict[str, ExperimentInstance] = {
    'ts2vec': ExperimentInstance(
        id='ts2vec',
        model_params=_ts2vec_params(),
        attack_families={
            AttackFamily.WHITE_BOX: (_fgsm_attack_run(), _pgd_attack_run(), _bim_attack_run())
        },
        trainer_kwargs=_DEFAULT_TRAINER_KWARGS,
    ),
    'autotcl': ExperimentInstance(
        id='autotcl',
        model_params=_autotcl_params(),
        attack_families={
            AttackFamily.WHITE_BOX: (_fgsm_attack_run(), _pgd_attack_run(), _bim_attack_run())
        },
        trainer_kwargs=_DEFAULT_TRAINER_KWARGS,
    ),
}


def list_experiment_ids() -> list[str]:
    """Return all registered experiment IDs."""
    return list(EXPERIMENTS_REGISTRY)


def get_experiment_instance(
    *, experiment_id: str, attack_family: AttackFamily | None = None
) -> ExperimentInstance:
    """Resolve an experiment ID to its instance.

    Args:
        experiment_id: Registered experiment identifier.
        attack_family: Optional family filter. When provided, returns a copy
            containing only attacks from the specified family.

    Returns:
        The experiment instance (deepcopied if filtered).

    Raises:
        KeyError: If the experiment ID is not registered.
    """
    # Resolve aliases
    if experiment_id in _EXPERIMENT_ID_ALIASES:
        new_id = _EXPERIMENT_ID_ALIASES[experiment_id]
        warnings.warn(
            f"Experiment ID '{experiment_id}' is deprecated. "
            f"Use '{new_id}' instead. "
            f'The old ID will be removed in a future release.',
            UserWarning,
            stacklevel=2,
        )
        experiment_id = new_id

    if experiment_id not in EXPERIMENTS_REGISTRY:
        available = ', '.join(list_experiment_ids())
        message = f"Unknown experiment '{experiment_id}'. Available: {available}"
        raise KeyError(message)

    instance = EXPERIMENTS_REGISTRY[experiment_id]

    if attack_family is None:
        return instance

    # Return filtered deepcopy
    result = copy.deepcopy(instance)
    result.attack_families = {attack_family: instance.attack_families.get(attack_family, ())}
    return result

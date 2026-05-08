"""Tests for attack registry mappings."""

from src.rbspaper.attacks.enums import AttackBackend, AttackMethod, AttackThreatModel
from src.rbspaper.attacks.registry import (
    get_default_backend,
    get_supervision_requirement,
    get_threat_model,
    is_backend_supported,
    list_supported_attacks,
)
from src.rbspaper.enums.general import TimeSeriesDownstreamTask


def test_registry_contains_required_attack_set() -> None:
    """Required baseline attacks should be present in the registry."""
    methods = set(list_supported_attacks(task=TimeSeriesDownstreamTask.CLASSIFICATION))
    required = {
        AttackMethod.LBFGS,
        AttackMethod.FGSM,
        AttackMethod.DEEPFOOL,
        AttackMethod.CW,
        AttackMethod.BIM,
        AttackMethod.PGD,
        AttackMethod.UAP,
        AttackMethod.SPSA,
    }
    assert required.issubset(methods)


def test_default_backend_for_core_attacks() -> None:
    """Core attacks should route to expected default backends."""
    assert (
        get_default_backend(attack=AttackMethod.FGSM, task=TimeSeriesDownstreamTask.CLASSIFICATION)
        == AttackBackend.TORCHATTACKS
    )
    assert (
        get_default_backend(attack=AttackMethod.PGD, task=TimeSeriesDownstreamTask.CLASSIFICATION)
        == AttackBackend.TORCHATTACKS
    )
    assert (
        get_default_backend(attack=AttackMethod.UAP, task=TimeSeriesDownstreamTask.CLASSIFICATION)
        == AttackBackend.ART
    )
    assert (
        get_default_backend(attack=AttackMethod.PGD, task=TimeSeriesDownstreamTask.FORECASTING)
        == AttackBackend.ART
    )


def test_threat_model_mapping_for_core_attacks() -> None:
    """Threat-model mapping should match white-box/black-box expectations."""
    assert get_threat_model(attack=AttackMethod.PGD) == AttackThreatModel.WHITE_BOX
    assert get_threat_model(attack=AttackMethod.SPSA) == AttackThreatModel.BLACK_BOX


def test_registry_support_and_supervision_policy() -> None:
    assert is_backend_supported(
        attack=AttackMethod.PGD,
        task=TimeSeriesDownstreamTask.FORECASTING,
        backend=AttackBackend.ART,
    )
    assert not is_backend_supported(
        attack=AttackMethod.PGD,
        task=TimeSeriesDownstreamTask.FORECASTING,
        backend=AttackBackend.TORCHATTACKS,
    )
    assert (
        get_supervision_requirement(
            attack=AttackMethod.FGSM, task=TimeSeriesDownstreamTask.CLASSIFICATION
        ).value
        == 'optional'
    )

"""Tests for AttackFamily enum and registry grouping."""

from src.rbspaper.attacks.enums import AttackFamily, AttackMethod, AttackThreatModel
from src.rbspaper.attacks.registry import ATTACK_THREAT_MODEL, group_methods_by_family


def test_attack_family_values() -> None:
    assert AttackFamily.WHITE_BOX == 'white_box'
    assert AttackFamily.BLACK_BOX == 'black_box'


def test_attack_family_is_strenum() -> None:
    assert isinstance(AttackFamily.WHITE_BOX, str)
    assert isinstance(AttackFamily.BLACK_BOX, str)


def test_group_methods_by_family_returns_all_registered_methods() -> None:
    families = group_methods_by_family()
    grouped = set()
    for methods in families.values():
        grouped.update(methods)
    non_gray_methods = {
        method
        for method, threat in ATTACK_THREAT_MODEL.items()
        if threat != AttackThreatModel.GRAY_BOX
    }
    assert grouped == non_gray_methods


def test_fgsm_is_white_box_family() -> None:
    families = group_methods_by_family()
    assert AttackMethod.FGSM in families[AttackFamily.WHITE_BOX]


def test_spsa_is_black_box_family() -> None:
    families = group_methods_by_family()
    assert AttackMethod.SPSA in families[AttackFamily.BLACK_BOX]

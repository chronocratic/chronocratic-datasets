"""Enums for adversarial attacks scoped to the attacks package."""

from __future__ import annotations

from enum import StrEnum


class AttackBackend(StrEnum):
    """Supported attack implementation backends."""

    ART = 'art'
    TORCHATTACKS = 'torchattacks'


class AttackThreatModel(StrEnum):
    """Threat-model categories used in reporting and filtering."""

    WHITE_BOX = 'white_box'
    GRAY_BOX = 'gray_box'
    BLACK_BOX = 'black_box'


class AttackObjective(StrEnum):
    """Optimization objectives for attack execution."""

    UNTARGETED = 'untargeted'
    TARGETED = 'targeted'
    MAXIMIZE_LOSS = 'maximize_loss'


class AttackSupervisionRequirement(StrEnum):
    """Requirement level for target supervision per attack policy."""

    REQUIRED = 'required'
    OPTIONAL = 'optional'
    NOT_USED = 'not_used'


class AttackMethod(StrEnum):
    """Canonical attack identifiers used across the package."""

    LBFGS = 'lbfgs'
    FGSM = 'fgsm'
    DEEPFOOL = 'deepfool'
    CW = 'cw'
    BIM = 'bim'
    PGD = 'pgd'
    UAP = 'uap'
    SPSA = 'spsa'
    MI_FGSM = 'mi_fgsm'
    AUTOATTACK = 'autoattack'
    HOPSKIPJUMP = 'hopskipjump'
    BOUNDARY = 'boundary'
    JSMA = 'jsma'
    ONE_PIXEL = 'one_pixel'
    SIMBA = 'simba'
    EAD = 'ead'


class AttackFamily(StrEnum):
    """Family groupings for experiment-level attack selection.

    Maps 1:1 to AttackThreatModel but is the CLI-facing concept used by
    the experiment registry to organize attacks into named groups.
    """

    WHITE_BOX = 'white_box'
    BLACK_BOX = 'black_box'

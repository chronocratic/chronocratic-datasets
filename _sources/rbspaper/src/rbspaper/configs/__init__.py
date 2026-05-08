"""One-stop shop for all dataclasses used to configure RBSPaper experiments.

This package exposes structured dataclasses used to configure augmentation
methods and representation models in RBSPaper experiments.
"""

from .attacks import (
    AttackExecutionMetadata,
    AttackParameters,
    BimAttackParameters,
    CwAttackParameters,
    DeepFoolAttackParameters,
    FgsmAttackParameters,
    LbfgsAttackParameters,
    PgdAttackParameters,
    SpsaAttackParameters,
    UapAttackParameters,
)
from .augmentation_methods import (
    AugmentationMethodParameters,
    AutoTCLNeuralNetworkAugmentationParameters,
    CosTRandomFunctionAugmentationParameters,
    CropShiftAugmentationParameters,
)
from .models import (
    AutoTCLModelParameters,
    CoSTModelParameters,
    ModelParameters,
    TS2VecModelParameters,
)

PACKAGE_NAME: str = 'rbspaper.configs'
PACKAGE_DESCRIPTION: str = (
    'Dataclasses for augmentation and model parameters used across RBSPaper runs.'
)

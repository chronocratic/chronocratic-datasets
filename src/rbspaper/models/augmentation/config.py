__all__ = [
    'AutoTCLNeuralNetworkAugmentationParameters',
    'CosTRandomFunctionAugmentationParameters',
    'CropShiftAugmentationParameters',
]

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

from src.rbspaper.models.encoders.masking import MaskMode


@dataclass
class AugmentationMethodParameters(ABC):
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Stable identifier used for logging/debugging/dispatch."""


@dataclass
class CropShiftAugmentationParameters(AugmentationMethodParameters):
    @property
    @override
    def method_name(self) -> str:
        return 'crop_shift'


@dataclass
class AutoTCLNeuralNetworkAugmentationParameters(AugmentationMethodParameters):
    """Data class to store augmentation parameters for the neural network."""

    input_dims: int = field(init=False)
    output_dims: int = 16
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
    hidden_dims: int = 64
    feature_extractor_depth: int = 10
    dropout_rate: float = 0.1
    conv_kernel_size: int = 3
    mask_mode: MaskMode = MaskMode.BINOMIAL
    num_augmentation_channels: int = 1
    gumbel_bias: float = 0.001
    zeta: float = 1.0
    gamma_zeta: float = 0.05
    hard_mask: bool = True

    @property
    @override
    def method_name(self) -> str:
        return 'auto_tcl_neural_network'

    def set_input_dims(self, input_dims: int) -> None:
        """Set the input dimensions for the augmentation method."""
        self.input_dims = input_dims


@dataclass
class CosTRandomFunctionAugmentationParameters(AugmentationMethodParameters):
    """Data class to store augmentation parameters for the random function."""

    sigma: float = 0.1
    p: float = 0.5

    @property
    @override
    def method_name(self) -> str:
        return 'cost_random_function'

"""Configuration dataclasses for all model parameters."""

__all__ = [
    'AutoTCLModelParameters',
    'CoSTModelParameters',
    'ModelParameters',
    'TS2VecModelParameters',
]

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

from src.rbspaper.models.augmentation.config import (
    AutoTCLNeuralNetworkAugmentationParameters,
    CosTRandomFunctionAugmentationParameters,
    CropShiftAugmentationParameters,
)
from src.rbspaper.models.augmentation.enums import (
    AutoTCLAugmentationMode,
    CoSTAugmentationMode,
    TS2VecAugmentationMode,
)
from src.rbspaper.models.encoders.masking import MaskMode


@dataclass
class ModelParameters(ABC):
    input_dims: int = field(default=1, init=False)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable identifier used for logging/debugging/dispatch."""

    def set_input_dims(self, input_dims: int) -> None:
        """Set the input dimensions for the model."""
        self.input_dims = input_dims


@dataclass
class TS2VecModelParameters(ModelParameters):
    augmentation_mode: TS2VecAugmentationMode
    augmentation_method_params: CropShiftAugmentationParameters
    augmentation_mode_params: dict | None = None
    hidden_dims: int = 64
    output_dims: int = 320
    depth: int = 10
    dropout_rate: float = 0.1
    conv_kernel_size: int = 3
    mask_mode: MaskMode = MaskMode.BINOMIAL
    learning_rate: float = 1e-3
    max_train_length: int | None = None
    temporal_unit: int = 0
    sync_dist: bool = False

    @property
    @override
    def model_name(self) -> str:
        return 'TS2Vec'


@dataclass
class AutoTCLModelParameters(ModelParameters):
    augmentation_mode: AutoTCLAugmentationMode
    augmentation_method_params: AutoTCLNeuralNetworkAugmentationParameters
    augmentation_mode_params: dict | None = None
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
    hidden_dims: int = 64
    output_dims: int = 320
    depth: int = 10
    dropout_rate: float = 0.1
    conv_kernel_size: int = 3
    mask_mode: MaskMode = MaskMode.BINOMIAL
    learning_rate: float = 1e-3
    max_train_length: int | None = None
    sync_dist: bool = False

    @property
    @override
    def model_name(self) -> str:
        return 'AutoTCL'


@dataclass
class CoSTModelParameters(ModelParameters):
    sequence_length: int = field(init=False)
    augmentation_mode: CoSTAugmentationMode
    augmentation_method_params: CosTRandomFunctionAugmentationParameters
    augmentation_mode_params: dict | None = None
    kernel_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    max_train_length: int = 201
    hidden_dims: int = 64
    output_dims: int = 320
    depth: int = 10
    dropout_rate: float = 0.1
    mask_mode: MaskMode = MaskMode.BINOMIAL
    learning_rate: float = 1e-3
    seasonal_loss_weight: float = 0.1
    queue_size: int = 65536
    momentum: float = 0.999
    temperature: float = 0.07
    sync_dist: bool = False

    @property
    @override
    def model_name(self) -> str:
        return 'CoST'

    def set_sequence_length(self, sequence_length: int) -> None:
        """Set the sequence length for the CoST model."""
        self.sequence_length = sequence_length

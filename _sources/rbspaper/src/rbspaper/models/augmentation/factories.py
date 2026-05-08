"""Factory classes to instantiate augmentation methods by mode."""

__all__ = [
    'AutoTCLAugmentationMethodFactory',
    'CoSTAugmentationMethodFactory',
    'TS2VecAugmentationMethodFactory',
]

from src.rbspaper.models.augmentation.enums import (
    AutoTCLAugmentationMode,
    CoSTAugmentationMode,
    TS2VecAugmentationMode,
)
from src.rbspaper.models.augmentation.strategies import (
    AugmentationMethod,
    AutoTCLNeuralNetworkAugmentation,
    CosTRandomFunctionAugmentation,
    CropShiftAugmentation,
)


class AutoTCLAugmentationMethodFactory:
    """Factory to create augmentation methods for AutoTCL models."""

    @staticmethod
    def get_augmentation_method(mode: AutoTCLAugmentationMode, params: dict) -> AugmentationMethod:
        """Return an augmentation method instance for the given AutoTCL mode.

        Args:
            mode: The augmentation mode enum.
            params: Parameters to pass to the augmentation constructor.

        Returns:
            An AugmentationMethod instance.

        Raises:
            ValueError: If the mode is unsupported.
        """
        if mode == AutoTCLAugmentationMode.NEURAL_NETWORK:
            return AutoTCLNeuralNetworkAugmentation(params=params)
        msg = f'Unsupported augmentation mode: {mode}'
        raise ValueError(msg)


class TS2VecAugmentationMethodFactory:
    """Factory to create augmentation methods for TS2Vec models."""

    @staticmethod
    def get_augmentation_method(mode: TS2VecAugmentationMode) -> AugmentationMethod:
        """Return an augmentation method instance for the given TS2Vec mode.

        Args:
            mode: The augmentation mode enum.

        Returns:
            An AugmentationMethod instance.

        Raises:
            ValueError: If the mode is unsupported.
        """
        if mode == TS2VecAugmentationMode.CROP_SHIFT:
            return CropShiftAugmentation()
        msg = f'Unsupported augmentation mode: {mode}'
        raise ValueError(msg)


class CoSTAugmentationMethodFactory:
    """Factory to create augmentation methods for CoST models."""

    @staticmethod
    def get_augmentation_method(mode: CoSTAugmentationMode, params: dict) -> AugmentationMethod:
        """Return an augmentation method instance for the given CoST mode.

        Args:
            mode: The augmentation mode enum.
            params: Parameters to pass to the augmentation constructor.

        Returns:
            An AugmentationMethod instance.

        Raises:
            ValueError: If the mode is unsupported.
        """
        if mode == CoSTAugmentationMode.RANDOM_FUNCTIONS:
            return CosTRandomFunctionAugmentation(params=params)
        msg = f'Unsupported augmentation mode: {mode}'
        raise ValueError(msg)

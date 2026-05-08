"""Concrete augmentation strategies for time series data augmentation."""

__all__ = [
    'AugmentationMethod',
    'AutoTCLNeuralNetworkAugmentation',
    'CosTRandomFunctionAugmentation',
    'CropShiftAugmentation',
]

from abc import ABC, abstractmethod
import random

import numpy as np
import torch

from src.rbspaper.models.encoders import AutoTCLAugmentationTimeSeriesEncoder


class AugmentationMethod(ABC):
    """Abstract base class defining the interface for augmentation methods."""

    @abstractmethod
    def _setup(self) -> None:
        """Perform any initialization needed after construction."""

    @abstractmethod
    def augment(
        self, data: torch.Tensor, **kwargs: object
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | tuple[torch.Tensor, torch.Tensor, int]:
        """Augment the input data tensor.

        Args:
            data: Input tensor of shape (batch_size, sequence_length, features).
            **kwargs: Additional keyword arguments for subclass-specific augmentation.

        Returns:
            Augmented data (tensor or tuple of tensors).
        """

    @abstractmethod
    def get_model(self) -> torch.nn.Module | None:
        """Return the underlying model used for augmentation, if any.

        Returns:
            The augmentation model or None.
        """


class AutoTCLNeuralNetworkAugmentation(AugmentationMethod):
    """Neural network-based augmentation method following AutoTCL."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self.model: AutoTCLAugmentationTimeSeriesEncoder | None = None
        self._setup()

    def _build_model(self) -> None:
        self.model = AutoTCLAugmentationTimeSeriesEncoder(**self.params)

    def _setup(self) -> None:
        """Build the underlying augmentation model."""
        self._build_model()

    def augment(
        self, data: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        """Augment data using the trained neural network.

        Args:
            data: Input tensor to augment.
            **_kwargs: Additional keyword arguments (unused by this strategy).

        Returns:
            Augmented tensor.
        """
        if self.model is None:
            msg = 'Model not initialized'
            raise RuntimeError(msg)
        return self.model.augment(data)

    def get_model(self) -> AutoTCLAugmentationTimeSeriesEncoder | None:
        """Return the underlying AutoTCL augmentation encoder.

        Returns:
            The AutoTCLAugmentationTimeSeriesEncoder instance or None.
        """
        return self.model


class CropShiftAugmentation(AugmentationMethod):
    """Crop-and-shift augmentation: extracts randomized subsequences."""

    def _setup(self) -> None:
        """No setup required for crop-shift augmentation."""
        return

    def augment(
        self, data: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Crop and shift the input data to produce two augmented views.

        Args:
            data: Input tensor of shape (batch_size, sequence_length, features).
            **kwargs: Additional keyword arguments. ``temporal_unit`` (int, default 0) controls
                the power-of-two scale for crop length calculation.

        Returns:
            Tuple of (augmented_view_1, augmented_view_2, crop_length).
        """
        from src.rbspaper.models.ts2vec.utils import extract_subsequences_per_row  # noqa: PLC0415

        _temporal = kwargs.get('temporal_unit')
        temporal_unit: int = _temporal if isinstance(_temporal, int) else 0
        x = data
        rng = np.random.default_rng()

        total_length = x.size(1)

        # Randomly determine the length of the crop
        crop_length = rng.integers(low=2 ** (temporal_unit + 1), high=total_length + 1)

        # Randomly determine the starting and ending points for the crops
        crop_start = rng.integers(total_length - crop_length + 1)
        crop_end = crop_start + crop_length
        crop_extension_start = rng.integers(crop_start + 1)
        crop_extension_end = rng.integers(low=crop_end, high=total_length + 1)

        # Random offset for each sample in the batch
        crop_offsets = rng.integers(
            low=-crop_extension_start, high=total_length - crop_extension_end + 1, size=x.size(0)
        )

        # Generate augmented subsequences 1 by cropping and shifting
        augmented_subsequences_1 = extract_subsequences_per_row(
            array=x,
            indices=crop_offsets + crop_extension_start,
            num_elements=crop_end - crop_extension_start,
        )

        # Generate augmented subsequences 2 by cropping and shifting
        augmented_subsequences_2 = extract_subsequences_per_row(
            array=x, indices=crop_offsets + crop_start, num_elements=crop_extension_end - crop_start
        )

        return augmented_subsequences_1, augmented_subsequences_2, crop_length

    def get_model(self) -> None:
        """Crop-shift has no underlying model.

        Returns:
            None.
        """
        return


class CosTRandomFunctionAugmentation(AugmentationMethod):
    """Random function augmentation (jitter, scale, shift) for CoST."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self._setup()

    def _setup(self) -> None:
        """Extract sigma and probability parameters."""
        self._sigma = self.params['sigma']
        self._p = self.params.get('p', 0.5)

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add random Gaussian noise to the input."""
        if random.random() > self._p:  # noqa: S311 # research code, not cryptographic
            return x
        return x + (torch.randn(x.shape) * self._sigma)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each feature by a random factor."""
        if random.random() > self._p:  # noqa: S311 # research code, not cryptographic
            return x
        return x * (torch.randn(x.size(-1)) * self._sigma + 1)

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift each feature by a random offset."""
        if random.random() > self._p:  # noqa: S311 # research code, not cryptographic
            return x
        return x + (torch.randn(x.size(-1)) * self._sigma)

    def augment(
        self, data: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        """Apply jitter, shift, and scale augmentations in sequence.

        Args:
            data: Input tensor to augment.
            **_kwargs: Additional keyword arguments (unused by this strategy).

        Returns:
            Augmented tensor.
        """
        return self._jitter(self._shift(self._scale(data)))

    def get_model(self) -> None:
        """Random function augmentation has no underlying model.

        Returns:
            None.
        """
        return

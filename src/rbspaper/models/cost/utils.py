"""Utility functions for CoST model (amplitude/phase computation)."""

__all__ = ['compute_amplitude_and_phase']

import torch


def compute_amplitude_and_phase(
    x: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute amplitude and phase of a complex-valued tensor.

    The amplitude is computed as sqrt((real(x) + eps)^2 + (imag(x) + eps)^2).
    The phase is computed as arctan2(imag(x), real(x) + eps).

    Args:
        x: Complex-valued tensor (e.g., output of a Fourier transform).
        eps: Small epsilon for numerical stability (default 1e-6).

    Returns:
        Tuple of (amplitude, phase) tensors.

    Examples:
        >>> import torch
        >>> x = torch.tensor([1+2j, 3+4j], dtype=torch.cfloat)
        >>> amplitude, phase = compute_amplitude_and_phase(x)
    """
    amplitude = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
    phase = torch.atan2(x.imag, x.real + eps)
    return amplitude, phase

"""Utility functions for TS2Vec model (subsequence extraction)."""

__all__ = ['extract_subsequences_per_row']

import numpy as np
import torch


def extract_subsequences_per_row(
    array: torch.Tensor, indices: np.ndarray, num_elements: int
) -> torch.Tensor:
    """Extract subsequences from each row based on starting indices and length.

    Args:
        array: Input 2D tensor of shape (batch_size, sequence_length).
        indices: Starting indices for each row (1D array of length batch_size).
        num_elements: Number of elements to extract from each row.

    Returns:
        Tensor of extracted subsequences of shape (batch_size, num_elements).
    """
    all_indices = indices[:, None] + np.arange(num_elements)
    return array[torch.arange(all_indices.shape[0])[:, None], all_indices]

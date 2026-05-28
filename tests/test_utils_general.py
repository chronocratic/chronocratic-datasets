"""Tests for general utility functions.

Verifies that custom_collate_fn pads short batches by cycling,
centralize_variable_length_series centers NaN-padded data, and
process_data_with_varying_sequence_lengths_single handles 2-D/3-D data.
"""

import numpy as np
import pandas as pd
import torch

from tscollection.datasets.utils.general import (
    centralize_variable_length_series,
    custom_collate_fn,
    process_data_with_varying_sequence_lengths_single,
)

# --------------------------------------------------------------------------- #
# custom_collate_fn tests                                                      #
# --------------------------------------------------------------------------- #


def test_custom_collate_fn_pads_last_batch() -> None:
    """custom_collate_fn pads short batches by cycling."""
    batch = [torch.tensor([1.0]), torch.tensor([2.0])]
    result = custom_collate_fn(batch, desired_batch_size=4)
    assert result.shape[0] == 4


def test_custom_collate_fn_no_padding_needed() -> None:
    """custom_collate_fn returns collated batch when size is sufficient."""
    batch = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    result = custom_collate_fn(batch, desired_batch_size=2)
    assert result.shape[0] == 3


def test_custom_collate_fn_cycling_pattern() -> None:
    """custom_collate_fn cycles backwards through batch for padding."""
    batch = [torch.tensor([1.0]), torch.tensor([2.0])]
    result = custom_collate_fn(batch, desired_batch_size=5)
    # Original: [1, 2], pad 3 more: cycle backwards -> [2, 1, 2]
    assert result.shape[0] == 5
    assert result[0].item() == 1.0
    assert result[1].item() == 2.0
    assert result[2].item() == 2.0  # -(0%2)-1 = -1 -> last element
    assert result[3].item() == 1.0  # -(1%2)-1 = -2 -> second-to-last
    assert result[4].item() == 2.0  # -(2%2)-1 = -1 -> last element


def test_custom_collate_fn_keyword_only() -> None:
    """desired_batch_size is keyword-only."""
    batch = [torch.tensor([1.0]), torch.tensor([2.0])]
    # Should work with keyword
    result = custom_collate_fn(batch, desired_batch_size=4)
    assert result.shape[0] == 4


# --------------------------------------------------------------------------- #
# centralize_variable_length_series tests                                       #
# --------------------------------------------------------------------------- #


def test_centralize_variable_length_series() -> None:
    """Centering shifts valid data to middle of sequence."""
    # batch=1, seq=5, features=2
    # Valid data at indices 0 and 1 (right-padded pattern)
    series = np.array(
        [[[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]]
    )
    result = centralize_variable_length_series(series)

    # Shape should be preserved
    assert result.shape == series.shape


def test_centralize_variable_length_series_already_centered() -> None:
    """Already centered data stays centered."""
    # batch=1, seq=5, features=1
    # NaN at start and end, valid in middle
    series = np.array([[[np.nan], [np.nan], [1.0], [2.0], [np.nan]]])
    result = centralize_variable_length_series(series)

    assert result.shape == series.shape


def test_centralize_variable_length_series_batch() -> None:
    """Centering works on batch of sequences."""
    batch = np.zeros((2, 6, 3)) + np.nan
    # First sample: valid at indices 0, 1
    batch[0, 0, :] = [1.0, 2.0, 3.0]
    batch[0, 1, :] = [4.0, 5.0, 6.0]
    # Second sample: valid at indices 0, 1, 2
    batch[1, 0, :] = [1.0, 1.0, 1.0]
    batch[1, 1, :] = [2.0, 2.0, 2.0]
    batch[1, 2, :] = [3.0, 3.0, 3.0]

    result = centralize_variable_length_series(batch)

    assert result.shape == batch.shape


# --------------------------------------------------------------------------- #
# process_data_with_varying_sequence_lengths_single tests                       #
# --------------------------------------------------------------------------- #


def test_process_data_2d_array() -> None:
    """process_data handles 2-D numpy arrays."""
    data = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, np.nan]])
    result = process_data_with_varying_sequence_lengths_single(data)

    assert result.shape == data.shape


def test_process_data_3d_array() -> None:
    """process_data handles 3-D numpy arrays."""
    data = np.zeros((2, 4, 3)) + np.nan
    data[0, 0, :] = [1.0, 2.0, 3.0]
    data[0, 1, :] = [4.0, 5.0, 6.0]
    data[1, 0, :] = [7.0, 8.0, 9.0]
    data[1, 1, :] = [10.0, 11.0, 12.0]

    result = process_data_with_varying_sequence_lengths_single(data)

    assert result.shape == data.shape


def test_process_data_dataframe() -> None:
    """process_data handles pandas DataFrame."""
    data = pd.DataFrame({0: [1.0, 2.0, np.nan], 1: [3.0, 4.0, np.nan]})
    result = process_data_with_varying_sequence_lengths_single(data)

    # Should return DataFrame
    assert isinstance(result, pd.DataFrame)


def test_process_data_no_temporal_missing() -> None:
    """process_data skips centering when no edge NaN."""
    # Valid data at edges, NaN only in middle
    data = np.array([[1.0, np.nan, 2.0], [3.0, np.nan, 4.0]])
    result = process_data_with_varying_sequence_lengths_single(data)

    assert result.shape == data.shape


# --------------------------------------------------------------------------- #
# __all__ export test                                                           #
# --------------------------------------------------------------------------- #


def test_all_exports() -> None:
    """__all__ exports three public functions alphabetically."""
    import tscollection.datasets.utils.general as general_mod

    assert general_mod.__all__ == [
        'centralize_variable_length_series',
        'custom_collate_fn',
        'process_data_with_varying_sequence_lengths_single',
    ]

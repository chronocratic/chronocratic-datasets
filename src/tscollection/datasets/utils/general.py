"""General data utilities: collation, variable-length handling."""

from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data.dataloader import default_collate

__all__ = [
    'centralize_variable_length_series',
    'custom_collate_fn',
    'process_data_with_varying_sequence_lengths_single',
]


def custom_collate_fn(batch: list[Any], *, desired_batch_size: int) -> Any:
    """Collate function that pads the last batch by cycling samples.

    If the current batch is smaller than *desired_batch_size*, extra
    samples are appended by cycling backwards through the batch.

    Args:
        batch: A list of samples returned by the dataset.
        desired_batch_size: Target batch size.

    Returns:
        Standard collated tensor batch.
    """
    current_batch_size = len(batch)
    if current_batch_size < desired_batch_size:
        additional_needed = desired_batch_size - current_batch_size
        for i in range(additional_needed):
            sample_index = -(i % current_batch_size) - 1
            batch.append(batch[sample_index])

    return default_collate(batch)


def centralize_variable_length_series(series_batch: np.ndarray) -> np.ndarray:
    """Center variable-length time series within a fixed-length sequence.

    Shifts the valid (non-NaN) portion of each series to the centre
    of the sequence so the model receives centred rather than right-
    or left-padded data.

    Args:
        series_batch: 3-D array of shape
            (batch_size, sequence_length, feature_dim). Missing values
            must be represented as NaN.

    Returns:
        3-D array of the same shape with valid data centred.
    """
    first_valid_idx = np.argmax(~np.isnan(series_batch).all(axis=-1), axis=1)
    last_valid_idx = np.argmax(~np.isnan(series_batch[:, ::-1]).all(axis=-1), axis=1)

    sequence_length = series_batch.shape[1]
    offset = (first_valid_idx + last_valid_idx) // 2 - first_valid_idx
    offset[offset < 0] += sequence_length

    batch_indices, time_indices = np.ogrid[: series_batch.shape[0], :sequence_length]
    shifted_time_indices = time_indices - offset[:, np.newaxis]

    return series_batch[batch_indices, shifted_time_indices]


def process_data_with_varying_sequence_lengths_single(
    data: np.ndarray | pd.DataFrame,
) -> np.ndarray | pd.DataFrame:
    """Process data with varying sequence lengths by centering valid data.

    Handles both 2-D (samples, timesteps) and 3-D (samples, timesteps,
    features) arrays. If the original data is a DataFrame, the result
    is returned as a DataFrame.

    Args:
        data: Input array or DataFrame.

    Returns:
        Processed numpy array of the same shape.
    """
    original_data_shape = data.shape
    original_data_type = type(data)

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    if len(original_data_shape) == 2:
        data = np.expand_dims(data, axis=-1)

    temporal_missing = np.isnan(data).all(axis=-1).any(axis=0)
    # temporal_missing is a 1-D boolean array of shape (seq_len,)
    temporal_missing_flat = np.asarray(temporal_missing).flat
    if temporal_missing_flat[0] or temporal_missing_flat[-1]:
        data = centralize_variable_length_series(data)

    if len(original_data_shape) == 2:
        data = np.squeeze(data, axis=-1)

    if original_data_type == pd.DataFrame:
        data = pd.DataFrame(data)

    return data

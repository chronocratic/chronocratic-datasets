__all__ = [
    'centralize_variable_length_series',
    'custom_collate_fn',
    'process_data_with_varying_sequence_lengths_single',
]

import numpy as np
import pandas as pd
from torch.utils.data.dataloader import default_collate

_TWO_DIMS = 2


def custom_collate_fn(batch: list[object], desired_batch_size: int) -> object:
    """
    Collate a batch, padding it with reversed samples if it is smaller than desired_batch_size.

    :param batch: the batch of samples
    :param desired_batch_size: the target batch size
    :return: collated tensor batch of exactly desired_batch_size samples
    """
    current_batch_size = len(batch)
    if current_batch_size < desired_batch_size:
        # Calculate how many additional samples are needed
        additional_needed = desired_batch_size - current_batch_size

        # Create a list of additional samples, starting from the end and cycling backward
        additional_samples = []
        for i in range(additional_needed):
            # Cycle backwards through the batch without going out of bounds
            sample_index = -(i % current_batch_size) - 1  # -1, -2, ..., wrapping around
            additional_samples.append(batch[sample_index])

        # Add these to the original batch
        batch.extend(additional_samples)

    return default_collate(batch)


def centralize_variable_length_series(series_batch: np.ndarray) -> np.ndarray:
    """
    Centers variable-length time series data within a fixed-length sequence.

    :param series_batch: A 3D NumPy array of shape (batch_size, sequence_length, feature_dim)
    :return: A 3D NumPy array of the same shape as `series_batch`
    """
    # Identify the first and last indices of valid data in each time series
    first_valid_idx = np.argmax(~np.isnan(series_batch).all(axis=-1), axis=1)
    last_valid_idx = np.argmax(~np.isnan(series_batch[:, ::-1]).all(axis=-1), axis=1)

    # Compute the offset needed to center the valid data
    sequence_length = series_batch.shape[1]
    offset = (first_valid_idx + last_valid_idx) // 2 - first_valid_idx

    # Adjust negative offsets to ensure valid indexing
    offset[offset < 0] += sequence_length

    # Create index grids for advanced indexing
    batch_indices, time_indices = np.ogrid[: series_batch.shape[0], : series_batch.shape[1]]

    # Shift the time indices to center the valid data
    shifted_time_indices = time_indices - offset[:, np.newaxis]

    # Extract the centered time series using advanced indexing
    centered_series_batch = series_batch[batch_indices, shifted_time_indices]

    return centered_series_batch


def process_data_with_varying_sequence_lengths_single(
    data: np.ndarray | pd.DataFrame | None,
) -> np.ndarray | pd.DataFrame | None:
    """
    Process data with varying sequence lengths by centering the valid data within the sequence.

    :param data: the data to be processed
    :return: processed data
    """
    if data is None:
        return None
    original_data_shape = data.shape
    original_data_type = type(data)

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    # if data shape is not 3D then expand the dimensions
    if len(original_data_shape) == _TWO_DIMS:
        data = np.expand_dims(data, axis=-1)

    temporal_missing = np.atleast_1d(
        np.asarray(np.isnan(data).all(axis=-1).any(axis=0), dtype=bool)
    )
    if temporal_missing[0] or temporal_missing[-1]:
        data = centralize_variable_length_series(data)

    # return the data to its original shape
    if len(original_data_shape) == _TWO_DIMS:
        data = np.squeeze(data, axis=-1)

    if original_data_type == pd.DataFrame:
        data = pd.DataFrame(data)

    return data

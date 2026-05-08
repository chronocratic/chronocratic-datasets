"""Data utility subpackages."""

from src.rbspaper.data.utils.arff import process_df_according_to_dtypes, read_arff_as_df
from src.rbspaper.data.utils.common import (
    AccumulatingTimerCallback,
    cartesian_product_dict,
    closest_power_of_2,
    compose,
    find_project_root,
    flatten_list,
    flatten_list_of_np_arrays,
    get_num_samples_from_ts,
    load_json,
    separate_target_feature_from_df,
)
from src.rbspaper.data.utils.features import extract_time_features
from src.rbspaper.data.utils.general import (
    centralize_variable_length_series,
    custom_collate_fn,
    process_data_with_varying_sequence_lengths_single,
)
from src.rbspaper.data.utils.scaling import create_data_scaler, DataFormEnum

__all__ = [
    # common
    'AccumulatingTimerCallback',
    # scaling
    'DataFormEnum',
    'cartesian_product_dict',
    # general
    'centralize_variable_length_series',
    'closest_power_of_2',
    'compose',
    'create_data_scaler',
    'custom_collate_fn',
    # features
    'extract_time_features',
    'find_project_root',
    'flatten_list',
    'flatten_list_of_np_arrays',
    'get_num_samples_from_ts',
    'load_json',
    'process_data_with_varying_sequence_lengths_single',
    # arff
    'process_df_according_to_dtypes',
    'read_arff_as_df',
    'separate_target_feature_from_df',
]

# Utility API Reference

Utility functions are defined in {py:mod}`chronocratic.datasets.utils` and
re-exported from the package root. They handle caching, scaling, data processing,
and feature extraction.

## Cache Utilities

.. autofunction:: chronocratic.datasets.utils.build_cache_key

.. autofunction:: chronocratic.datasets.utils.resolve_cache_dir

.. autofunction:: chronocratic.datasets.utils.atomic_save_npz

.. autofunction:: chronocratic.datasets.utils.atomic_save_metadata

.. autofunction:: chronocratic.datasets.utils.load_metadata

.. autofunction:: chronocratic.datasets.utils.save_scaler

.. autofunction:: chronocratic.datasets.utils.load_scaler

.. autodata:: chronocratic.datasets.utils.CACHE_SCHEMA_VERSION

## Scaling Utilities

.. autofunction:: chronocratic.datasets.utils.create_data_scaler

## Common Utilities

.. autofunction:: chronocratic.datasets.utils.flatten_list_of_np_arrays

.. autofunction:: chronocratic.datasets.utils.get_num_samples_from_ts

.. autofunction:: chronocratic.datasets.utils.separate_target_feature_from_df

.. autofunction:: chronocratic.datasets.utils.compose

.. autoclass:: chronocratic.datasets.utils.FunctionComposer
   :members:
   :undoc-members:

## Feature Extraction

.. autofunction:: chronocratic.datasets.utils.extract_time_features

## General Utilities

.. autofunction:: chronocratic.datasets.utils.centralize_variable_length_series

.. autofunction:: chronocratic.datasets.utils.custom_collate_fn

.. autofunction:: chronocratic.datasets.utils.process_data_with_varying_sequence_lengths_single

## ARFF Utilities

.. autofunction:: chronocratic.datasets.utils.read_arff_as_df

.. autofunction:: chronocratic.datasets.utils.process_df_according_to_dtypes

"""ARFF file reading utilities for time series datasets."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ['process_df_according_to_dtypes', 'read_arff_as_df']


def read_arff_as_df(arff_file_path: Path | str) -> tuple[pd.DataFrame, Any]:
    """Read an ARFF file into a pandas DataFrame.

    Note:
        scipy.io.arff returns nominal (string) column values as bytes
        objects (e.g. ``b'a'`` not ``'a'``). The caller must provide a
        decode function in ``dtypes_functions_map`` when using
        :func:`process_df_according_to_dtypes`.

    Args:
        arff_file_path: Path to the ARFF file.

    Returns:
        A tuple of (DataFrame, ARFF metadata object).
    """
    from scipy.io import arff

    data, meta = arff.loadarff(arff_file_path)
    df_data = pd.DataFrame(data)
    return df_data, meta


def process_df_according_to_dtypes(
    df_data: pd.DataFrame, meta: Any, dtypes_functions_map: dict[str, Callable]
) -> pd.DataFrame:
    """Process DataFrame columns according to ARFF dtype mapping.

    Iterates over each column defined in the ARFF metadata, determines
    its type, and applies the corresponding transformation function from
    the provided mapping.

    Args:
        df_data: DataFrame to process.
        meta: ARFF metadata containing column type information.
        dtypes_functions_map: Mapping from dtype name to transformation
            function.

    Returns:
        Processed DataFrame with correctly typed columns.
    """
    for col_name in meta.names():
        col_type = str(meta[col_name][0])
        df_data[col_name] = dtypes_functions_map[col_type](df_data[col_name])
    return df_data

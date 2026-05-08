"""ARFF file reading utilities for UCR/UEA datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

__all__ = ['process_df_according_to_dtypes', 'read_arff_as_df']

if TYPE_CHECKING:
    from typing import Any


def read_arff_as_df(arff_file_path: Path | str) -> tuple[pd.DataFrame, Any]:
    """Read an ARFF file into a pandas DataFrame.

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

    Args:
        df_data: DataFrame to process.
        meta: ARFF metadata containing column type information.
        dtypes_functions_map: Mapping from dtype name to transformation function.

    Returns:
        Processed DataFrame with correctly typed columns.
    """
    for col_name in meta.names():
        col_type = str(meta[col_name][0])
        df_data[col_name] = dtypes_functions_map[col_type](df_data[col_name])
    return df_data

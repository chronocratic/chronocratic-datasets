__all__ = ['process_df_according_to_dtypes', 'read_arff_as_df']

from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd
from scipy.io import arff
from scipy.io.arff._arffread import MetaData


def read_arff_as_df(arff_file_path: Path | str) -> tuple[pd.DataFrame, MetaData]:
    """Read an ARFF file and return its DataFrame plus metadata."""
    data, meta = arff.loadarff(arff_file_path)
    df_data = pd.DataFrame(data)
    return (df_data, meta)


def process_df_according_to_dtypes(
    df_data: pd.DataFrame,
    meta: MetaData,
    dtypes_functions_map: Mapping[str, Callable[[pd.Series], pd.Series]],
) -> pd.DataFrame:
    """Transform DataFrame columns based on ARFF-declared column dtypes."""
    for col_name in meta.names():
        col_type = meta[col_name][0]
        df_data[col_name] = dtypes_functions_map[col_type](df_data[col_name])
    return df_data

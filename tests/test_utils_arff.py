"""Tests for ARFF utility functions.

Verifies that read_arff_as_df and process_df_according_to_dtypes
correctly parse ARFF files and transform DataFrame columns.
"""

import pandas as pd

from chronocratic.datasets.utils.arff import process_df_according_to_dtypes, read_arff_as_df

# --------------------------------------------------------------------------- #
# read_arff_as_df tests                                                        #
# --------------------------------------------------------------------------- #


def test_read_arff_as_df_returns_dataframe_and_metadata(tmp_path) -> None:
    """read_arff_as_df returns a DataFrame and ARFF metadata."""
    arff_content = """@relation test
@attribute numeric_col numeric
@attribute nominal_col {a, b}

@data
1.0, a
2.0, b
3.0, a
"""
    arff_file = tmp_path / "test.arff"
    arff_file.write_text(arff_content)

    df, meta = read_arff_as_df(arff_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3
    assert "numeric_col" in meta.names()
    assert "nominal_col" in meta.names()


def test_read_arff_as_df_string_path(tmp_path) -> None:
    """read_arff_as_df accepts string paths."""
    arff_content = """@relation test
@attribute x numeric

@data
5.0,
"""
    arff_file = tmp_path / "string_test.arff"
    arff_file.write_text(arff_content)

    df, _meta = read_arff_as_df(str(arff_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1


def test_read_arff_as_df_nominal_returns_bytes(tmp_path) -> None:
    """Nominal columns from scipy are bytes objects."""
    arff_content = """@relation test
@attribute label {cat, dog}

@data
cat,
dog,
"""
    arff_file = tmp_path / "bytes_test.arff"
    arff_file.write_text(arff_content)

    df, _ = read_arff_as_df(arff_file)
    # scipy returns bytes for nominal columns
    assert df["label"].dtype == object
    assert df["label"].iloc[0] == b"cat"


# --------------------------------------------------------------------------- #
# process_df_according_to_dtypes tests                                         #
# --------------------------------------------------------------------------- #


def test_process_df_according_to_dtypes_applies_transforms(tmp_path) -> None:
    """process_df_according_to_dtypes applies mapped functions."""
    arff_content = """@relation test
@attribute x numeric
@attribute y {a, b}

@data
1.0, a
2.0, b
"""
    arff_file = tmp_path / "process_test.arff"
    arff_file.write_text(arff_content)

    df, meta = read_arff_as_df(arff_file)

    def to_float(series):
        return series.astype(float)

    def decode_bytes(series):
        return series.apply(lambda x: x.decode() if isinstance(x, bytes) else x)

    dtypes_map = {"numeric": to_float, "nominal": decode_bytes}

    result = process_df_according_to_dtypes(df_data=df, meta=meta, dtypes_functions_map=dtypes_map)
    assert result["x"].dtype == float
    assert result["y"].iloc[0] == "a"

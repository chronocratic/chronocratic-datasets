"""Tests for sequence handling strategies (DST-05).

Verifies that Strategy pattern classes correctly compute window counts
and extract labels for forecasting and classification datasets.
"""

import numpy as np
import pytest

from tscollection.datasets.datasets.classes.strategies import (
    ClassificationStrategyMultipleFiles,
    ClassificationStrategySingleFile,
    ForecastingStrategySingleFile,
    SequenceHandlingStrategy,
)

# --------------------------------------------------------------------------- #
# ForecastingStrategySingleFile tests                                          #
# --------------------------------------------------------------------------- #


def test_forecasting_num_sequences() -> None:
    """DST-05: ForecastingStrategySingleFile returns correct window count.

    For a (200, 7) array with seq_len=96, step=1, forecast_horizon=24,
    exactly 80 valid windows exist: (200 - 96 - 24) = 80.
    """
    data = np.random.default_rng().standard_normal((200, 7)).astype(np.float32)
    strategy = ForecastingStrategySingleFile(forecast_horizon=24)
    count = strategy.get_num_sequences(data=data, seq_len=96, step=1)
    assert count == 80


def test_forecasting_label_slice() -> None:
    """DST-05: Forecasting label is the post-window data segment.

    For n=0 and seq_len=96, the label should be data[96:120]
    (96 + forecast_horizon=24 = 120).
    """
    data = np.arange(200).astype(np.float32)
    strategy = ForecastingStrategySingleFile(forecast_horizon=24)
    label = strategy.get_current_label(
        data=data, labels=None, n=0, seq_len=96
    )
    expected = data[96:120]
    np.testing.assert_array_equal(label, expected)


# --------------------------------------------------------------------------- #
# ClassificationStrategySingleFile tests                                       #
# --------------------------------------------------------------------------- #


def test_classification_num_sequences() -> None:
    """DST-05: ClassificationStrategySingleFile returns correct window count.

    For a (200,) array with seq_len=50 and step=10, count the valid windows.
    range(200 - 50, 0, -10) = range(150, 0, -10) = [150, 140, ..., 10]
    possible_ends = [200, 190, ..., 60]
    valid_ends = [e for e in possible_ends if e < 200] = [190, ..., 60] = 14
    """
    data = np.random.default_rng().standard_normal((200,)).astype(np.float32)
    strategy = ClassificationStrategySingleFile()
    count = strategy.get_num_sequences(data=data, seq_len=50, step=10)
    assert count == 14


def test_classification_label_with_labels() -> None:
    """DST-05: Classification returns label slice when labels are provided."""
    strategy = ClassificationStrategySingleFile()
    data = np.zeros(100)
    labels = np.arange(100)
    result = strategy.get_current_label(
        data=data, labels=labels, n=25, seq_len=20
    )
    expected = labels[25:45]
    np.testing.assert_array_equal(result, expected)


def test_classification_label_none() -> None:
    """DST-05: Classification returns None when labels is None."""
    strategy = ClassificationStrategySingleFile()
    result = strategy.get_current_label(
        data=np.zeros(100), labels=None, n=0, seq_len=50
    )
    assert result is None


# --------------------------------------------------------------------------- #
# ClassificationStrategyMultipleFiles tests                                    #
# --------------------------------------------------------------------------- #


def test_multifile_num_sequences() -> None:
    """DST-05: Multi-file strategy sums per-file sequence counts.

    Two arrays: one with 100 samples, one with 200 samples.
    seq_len=50, step=10.
    Array 1: range(100-50, 0, -10) = range(50, 0, -10) -> ends=[100,90,80,70,60] -> valid < 100 -> 4
    Array 2: range(200-50, 0, -10) = range(150, 0, -10) ->
    ends=[200,190,...,60] -> valid < 200 -> 14
    Total: 4 + 14 = 18
    """
    data_list = [
        np.random.default_rng().standard_normal((100,)).astype(np.float32),
        np.random.default_rng().standard_normal((200,)).astype(np.float32),
    ]
    strategy = ClassificationStrategyMultipleFiles()
    count = strategy.get_num_sequences(
        data=data_list, seq_len=50, step=10
    )
    assert count == 18


def test_multifile_per_file_counts() -> None:
    """DST-05: get_num_sequences_per_file returns per-file integer list.

    Same inputs as test_multifile_num_sequences.
    Array 1: 4 valid windows, Array 2: 14 valid windows.
    """
    data_list = [
        np.random.default_rng().standard_normal((100,)).astype(np.float32),
        np.random.default_rng().standard_normal((200,)).astype(np.float32),
    ]
    strategy = ClassificationStrategyMultipleFiles()
    counts = strategy.get_num_sequences_per_file(
        data=data_list, seq_len=50, step=10
    )
    assert counts == [4, 14]


# --------------------------------------------------------------------------- #
# ABC tests                                                                    #
# --------------------------------------------------------------------------- #


def test_abstract_cannot_instantiate() -> None:
    """DST-05: SequenceHandlingStrategy is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        SequenceHandlingStrategy()  # type: ignore[misc,call-arg]

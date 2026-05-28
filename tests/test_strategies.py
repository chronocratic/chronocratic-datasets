"""Tests for sequence handling strategies.

Verifies that Strategy pattern classes correctly compute window counts
and extract labels for forecasting and classification datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from tscollection.datasets.datatypes._base.strategies import (
    ClassificationStrategyMultipleFiles,
    ClassificationStrategySingleFile,
    ForecastingStrategySingleFile,
    SequenceHandlingStrategy,
)

# --------------------------------------------------------------------------- #
# ForecastingStrategySingleFile tests                                          #
# --------------------------------------------------------------------------- #


def test_forecasting_num_sequences() -> None:
    """ForecastingStrategySingleFile returns correct window count.

    For a (200, 7) array with seq_len=96, step=1, forecast_horizon=24,
    exactly 81 valid windows exist: indices 0..80 => 200 - 96 - 24 + 1 = 81.
    """
    data = np.random.default_rng().standard_normal((200, 7)).astype(np.float32)
    strategy = ForecastingStrategySingleFile(forecast_horizon=24)
    count = strategy.get_num_sequences(data=data, seq_len=96, step=1)
    assert count == 81


def test_forecasting_label_slice() -> None:
    """Forecasting label is the post-window data segment.

    For n=0 and seq_len=96, the label should be data[96:120]
    (96 + forecast_horizon=24 = 120).
    """
    data = np.arange(200).astype(np.float32)
    strategy = ForecastingStrategySingleFile(forecast_horizon=24)
    label = strategy.get_current_label(data=data, labels=None, n=0, seq_len=96)
    expected = data[96:120]
    np.testing.assert_array_equal(label, expected)


# --------------------------------------------------------------------------- #
# ClassificationStrategySingleFile tests                                       #
# --------------------------------------------------------------------------- #


def test_classification_num_sequences() -> None:
    """ClassificationStrategySingleFile returns correct window count.

    For a (200,) array with seq_len=50 and step=10, count the valid windows.
    range(200 - 50, -1, -10) = range(150, -1, -10) = [150, 140, ..., 10, 0]
    possible_ends = [200, 190, ..., 60, 50]
    valid_ends = [e for e in possible_ends if e <= 200] = [200, 190, ..., 50] = 16
    """
    data = np.random.default_rng().standard_normal((200,)).astype(np.float32)
    strategy = ClassificationStrategySingleFile()
    count = strategy.get_num_sequences(data=data, seq_len=50, step=10)
    assert count == 16


def test_classification_label_with_labels() -> None:
    """Classification returns label slice when labels are provided."""
    strategy = ClassificationStrategySingleFile()
    data = np.zeros(100)
    labels = np.arange(100)
    result = strategy.get_current_label(data=data, labels=labels, n=25, seq_len=20)
    expected = labels[25:45]
    np.testing.assert_array_equal(result, expected)


def test_classification_label_none() -> None:
    """Classification returns None when labels is None."""
    strategy = ClassificationStrategySingleFile()
    result = strategy.get_current_label(data=np.zeros(100), labels=None, n=0, seq_len=50)
    assert result is None


# --------------------------------------------------------------------------- #
# ClassificationStrategyMultipleFiles tests                                    #
# --------------------------------------------------------------------------- #


def test_multifile_num_sequences() -> None:
    """Multi-file strategy sums per-file sequence counts.

    Two arrays: one with 100 samples, one with 200 samples.
    seq_len=50, step=10.
    Array 1: range(100-50, -1, -10) -> ends=[100,90,...,50] -> valid <= 100 -> 6
    Array 2: range(200-50, -1, -10) -> ends=[200,190,...,50] -> valid <= 200 -> 16
    Total: 6 + 16 = 22
    """
    data_list = [
        np.random.default_rng().standard_normal((100,)).astype(np.float32),
        np.random.default_rng().standard_normal((200,)).astype(np.float32),
    ]
    strategy = ClassificationStrategyMultipleFiles()
    count = strategy.get_num_sequences(data=data_list, seq_len=50, step=10)
    assert count == 22


def test_multifile_per_file_counts() -> None:
    """get_num_sequences_per_file returns per-file integer list.

    Same inputs as test_multifile_num_sequences.
    Array 1: 6 valid windows, Array 2: 16 valid windows.
    """
    data_list = [
        np.random.default_rng().standard_normal((100,)).astype(np.float32),
        np.random.default_rng().standard_normal((200,)).astype(np.float32),
    ]
    strategy = ClassificationStrategyMultipleFiles()
    counts = strategy.get_num_sequences_per_file(data=data_list, seq_len=50, step=10)
    assert counts == [6, 16]


# --------------------------------------------------------------------------- #
# ABC tests                                                                    #
# --------------------------------------------------------------------------- #


def test_abstract_cannot_instantiate() -> None:
    """SequenceHandlingStrategy is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        SequenceHandlingStrategy()  # type: ignore[misc,call-arg]

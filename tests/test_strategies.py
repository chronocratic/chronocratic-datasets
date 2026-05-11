"""Tests for sequence handling strategies (DST-05).

Verifies that Strategy pattern classes correctly compute window counts
and extract labels for forecasting and classification datasets.
"""

import numpy as np
import pytest


class TestForecastingStrategySingleFile:
    """Tests for ForecastingStrategySingleFile (DST-05)."""

    def test_get_num_sequences(self) -> None:
        """DST-05: Forecasting strategy returns correct sequence count."""
        from tscollection.datasets.datasets.classes.strategies import (
            ForecastingStrategySingleFile,
        )

        data = np.random.randn(200, 7).astype(np.float32)
        strategy = ForecastingStrategySingleFile(forecast_horizon=24)
        count = strategy.get_num_sequences(data=data, seq_len=96, step=1)
        assert count == 80  # (200 - 96 - 24) = 80 valid windows

    def test_get_current_label(self) -> None:
        """DST-05: Forecasting label is the post-window data slice."""
        from tscollection.datasets.datasets.classes.strategies import (
            ForecastingStrategySingleFile,
        )

        data = np.arange(200).astype(np.float32)
        strategy = ForecastingStrategySingleFile(forecast_horizon=24)
        label = strategy.get_current_label(
            data=data, labels=None, n=0, seq_len=96
        )
        expected = data[96:120]
        np.testing.assert_array_equal(label, expected)


class TestClassificationStrategySingleFile:
    """Tests for ClassificationStrategySingleFile (DST-05)."""

    def test_get_current_label_none(self) -> None:
        """DST-05: Classification returns None when labels is None."""
        from tscollection.datasets.datasets.classes.strategies import (
            ClassificationStrategySingleFile,
        )

        strategy = ClassificationStrategySingleFile()
        result = strategy.get_current_label(
            data=np.zeros(100), labels=None, n=0, seq_len=50
        )
        assert result is None


class TestSequenceHandlingStrategy:
    """Tests for abstract SequenceHandlingStrategy (DST-05)."""

    def test_cannot_instantiate_abstract(self) -> None:
        """DST-05: SequenceHandlingStrategy is abstract."""
        from tscollection.datasets.datasets.classes.strategies import (
            SequenceHandlingStrategy,
        )

        with pytest.raises(TypeError):
            SequenceHandlingStrategy()  # type: ignore[misc,call-arg]

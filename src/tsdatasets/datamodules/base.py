"""Shared data-module abstractions."""

from __future__ import annotations

from dataclasses import dataclass

from tsdatasets.datasets import TimeSeriesDataset


@dataclass(frozen=True, slots=True)
class TimeSeriesDataModule:
    """Container for train/validation/test dataset splits."""

    train: TimeSeriesDataset
    validation: TimeSeriesDataset | None = None
    test: TimeSeriesDataset | None = None

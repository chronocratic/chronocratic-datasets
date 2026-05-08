"""Shared dataset abstractions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload


@dataclass(frozen=True, slots=True)
class TimeSeriesRecord:
    """One time-series sample."""

    values: tuple[float, ...]
    timestamps: tuple[str, ...] | None = None
    target: float | None = None
    series_id: str | None = None

    def __post_init__(self) -> None:
        if self.timestamps is not None and len(self.timestamps) != len(self.values):
            msg = "timestamps and values must have the same length"
            raise ValueError(msg)


class TimeSeriesDataset(Sequence[TimeSeriesRecord]):
    """Simple immutable sequence wrapper for time-series samples."""

    def __init__(self, records: Sequence[TimeSeriesRecord], name: str):
        self._records = tuple(records)
        self.name = name

    def __len__(self) -> int:
        return len(self._records)

    @overload
    def __getitem__(self, index: int) -> TimeSeriesRecord: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TimeSeriesRecord]: ...

    def __getitem__(self, index: int | slice) -> TimeSeriesRecord | Sequence[TimeSeriesRecord]:
        return self._records[index]

    def __iter__(self) -> Iterator[TimeSeriesRecord]:
        return iter(self._records)

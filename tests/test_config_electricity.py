"""Tests for Electricity forecasting configuration (CFG-02).

Verifies that ElectricityConfig correctly inherits from ForecastingConfig,
enforces fractional split mode, and provides valid frozen instances with
correct CSV parsing parameters from rbspaper source.
"""

import pytest
from pydantic import ValidationError

from tscollection.datasets.config.base import ForecastingConfig
from tscollection.datasets.enums import DatasetFamily, SplitMode


class TestElectricityConfigClass:
    """Tests for the ElectricityConfig class structure."""

    def test_inherits_from_forecasting_config(self) -> None:
        """CFG-02: ElectricityConfig inherits from ForecastingConfig."""
        from tscollection.datasets.config.electricity import ElectricityConfig

        assert issubclass(ElectricityConfig, ForecastingConfig)

    def test_split_mode_returns_fractional(self) -> None:
        """CFG-02: ElectricityConfig instances return SplitMode.FRACTIONAL."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        assert ELECTRICITY_LOAD.split_mode == SplitMode.FRACTIONAL

    def test_family_defaults_to_electricity(self) -> None:
        """CFG-02: ElectricityConfig defaults family to ELECTRICITY."""
        from tscollection.datasets.config.electricity import ElectricityConfig

        cfg = ElectricityConfig(
            name='Test',
            url='https://example.com/test.csv',
            split_bounds=(0.6, 0.2, 0.2),
            tasks=('forecasting',),
        )
        assert cfg.family == DatasetFamily.ELECTRICITY


class TestELECTRICITY_LOAD:
    """Tests for the ELECTRICITY_LOAD instance."""

    def test_name(self) -> None:
        """CFG-02: ELECTRICITY_LOAD has correct name."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        assert ELECTRICITY_LOAD.name == 'electricity'

    def test_split_bounds_fractional(self) -> None:
        """CFG-02: ELECTRICITY_LOAD uses 60/20/20 fractional splits."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        assert ELECTRICITY_LOAD.split_bounds == (0.6, 0.2, 0.2)

    def test_split_bounds_sum_to_one(self) -> None:
        """CFG-02: Fractional split bounds sum to 1.0."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        assert sum(ELECTRICITY_LOAD.split_bounds) == pytest.approx(1.0)

    def test_forecast_column(self) -> None:
        """CFG-02: ELECTRICITY_LOAD forecast column is MT_001."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        assert ELECTRICITY_LOAD.forecast_column == 'MT_001'

    def test_csv_separator(self) -> None:
        """CFG-02: ELECTRICITY_LOAD CSV separator is semicolon."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        assert ELECTRICITY_LOAD.csv_sep == ';'

    def test_csv_decimal(self) -> None:
        """CFG-02: ELECTRICITY_LOAD CSV decimal character is comma."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        assert ELECTRICITY_LOAD.csv_decimal == ','

    def test_default_horizon(self) -> None:
        """CFG-02: ELECTRICITY_LOAD default horizon is 24."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        assert ELECTRICITY_LOAD.default_horizon == 24

    def test_default_seq_len(self) -> None:
        """CFG-02: ELECTRICITY_LOAD default seq_len is 128."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        assert ELECTRICITY_LOAD.default_seq_len == 128

    def test_is_frozen(self) -> None:
        """CFG-02: ELECTRICITY_LOAD is frozen."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        with pytest.raises(ValueError, match='frozen'):
            ELECTRICITY_LOAD.name = 'Other'  # type: ignore[attr-defined]

    def test_model_copy(self) -> None:
        """CFG-02: ELECTRICITY_LOAD.model_copy produces a new instance."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        new_cfg = ELECTRICITY_LOAD.model_copy(update={'default_seq_len': 64})
        assert new_cfg.default_seq_len == 64
        assert ELECTRICITY_LOAD.default_seq_len == 128  # Original unchanged

"""Tests for ETT forecasting configuration (CFG-02).

Verifies that ETTConfig correctly inherits from ForecastingConfig,
enforces indexed split mode, and provides valid frozen instances for
ETTh1, ETTh2, ETTm1, and ETTm2.
"""

import pytest
from pydantic import ValidationError

from tscollection.datasets.config.base import ForecastingConfig
from tscollection.datasets.enums import DatasetFamily, SplitMode


class TestETTConfigClass:
    """Tests for the ETTConfig class structure."""

    def test_inherits_from_forecasting_config(self) -> None:
        """CFG-02: ETTConfig inherits from ForecastingConfig."""
        from tscollection.datasets.config.ett import ETTConfig

        assert issubclass(ETTConfig, ForecastingConfig)

    def test_split_mode_returns_indexed(self) -> None:
        """CFG-02: ETTConfig instances always return SplitMode.INDEXED."""
        from tscollection.datasets.config.ett import ETT_H1

        assert ETT_H1.split_mode == SplitMode.INDEXED

    def test_family_defaults_to_ett(self) -> None:
        """CFG-02: ETTConfig defaults family to DatasetFamily.ETT."""
        from tscollection.datasets.config.ett import ETTConfig

        # Create a minimal instance
        cfg = ETTConfig(
            name='Test',
            url='https://example.com/test.csv',
            split_bounds=(100, 200, 300),
            forecast_column='OT',
            frequency='1h',
            num_features=7,
            tasks=('forecasting',),
        )
        assert cfg.family == DatasetFamily.ETT


class TestETT_H1:
    """Tests for the ETT_H1 instance (hourly)."""

    def test_name(self) -> None:
        """CFG-02: ETT_H1 has correct name."""
        from tscollection.datasets.config.ett import ETT_H1

        assert ETT_H1.name == 'ETTh1'

    def test_split_bounds_hourly(self) -> None:
        """CFG-02: ETT_H1 split bounds match rbspaper (12*30*24, 16*30*24, 20*30*24)."""
        from tscollection.datasets.config.ett import ETT_H1

        assert ETT_H1.split_bounds == (8640, 11520, 14400)

    def test_forecast_column(self) -> None:
        """CFG-02: ETT_H1 forecast column is OT."""
        from tscollection.datasets.config.ett import ETT_H1

        assert ETT_H1.forecast_column == 'OT'

    def test_frequency_hourly(self) -> None:
        """CFG-02: ETT_H1 frequency is 1h."""
        from tscollection.datasets.config.ett import ETT_H1

        assert ETT_H1.frequency == '1h'

    def test_num_features(self) -> None:
        """CFG-02: ETT_H1 has 7 features."""
        from tscollection.datasets.config.ett import ETT_H1

        assert ETT_H1.num_features == 7

    def test_default_horizon_hourly(self) -> None:
        """CFG-02: ETT_H1 default horizon is 24 (1 day for hourly)."""
        from tscollection.datasets.config.ett import ETT_H1

        assert ETT_H1.default_horizon == 24

    def test_default_seq_len(self) -> None:
        """CFG-02: ETT_H1 default seq_len is 128."""
        from tscollection.datasets.config.ett import ETT_H1

        assert ETT_H1.default_seq_len == 128

    def test_tasks(self) -> None:
        """CFG-02: ETT_H1 has forecasting and representation tasks."""
        from tscollection.datasets.config.ett import ETT_H1

        assert ETT_H1.tasks == ('forecasting', 'representation')

    def test_is_frozen(self) -> None:
        """CFG-02: ETT_H1 is frozen."""
        from tscollection.datasets.config.ett import ETT_H1

        with pytest.raises(ValueError, match='frozen'):
            ETT_H1.name = 'Other'  # type: ignore[attr-defined]

    def test_model_copy(self) -> None:
        """CFG-02: ETT_H1.model_copy produces a new instance."""
        from tscollection.datasets.config.ett import ETT_H1

        new_cfg = ETT_H1.model_copy(update={'default_seq_len': 64})
        assert new_cfg.default_seq_len == 64
        assert ETT_H1.default_seq_len == 128  # Original unchanged


class TestETT_H2:
    """Tests for the ETT_H2 instance (hourly)."""

    def test_name(self) -> None:
        """CFG-02: ETT_H2 has correct name."""
        from tscollection.datasets.config.ett import ETT_H2

        assert ETT_H2.name == 'ETTh2'

    def test_split_bounds(self) -> None:
        """CFG-02: ETT_H2 split bounds match hourly pattern."""
        from tscollection.datasets.config.ett import ETT_H2

        assert ETT_H2.split_bounds == (8640, 11520, 14400)


class TestETT_M1:
    """Tests for the ETT_M1 instance (15-min)."""

    def test_name(self) -> None:
        """CFG-02: ETT_M1 has correct name."""
        from tscollection.datasets.config.ett import ETT_M1

        assert ETT_M1.name == 'ETTm1'

    def test_split_bounds_15min(self) -> None:
        """CFG-02: ETT_M1 split bounds match rbspaper (12*30*24*4, 16*30*24*4, 20*30*24*4)."""
        from tscollection.datasets.config.ett import ETT_M1

        assert ETT_M1.split_bounds == (34560, 46080, 57600)

    def test_frequency_15min(self) -> None:
        """CFG-02: ETT_M1 frequency is 15min."""
        from tscollection.datasets.config.ett import ETT_M1

        assert ETT_M1.frequency == '15min'

    def test_default_horizon_15min(self) -> None:
        """CFG-02: ETT_M1 default horizon is 96 (24h at 15-min intervals)."""
        from tscollection.datasets.config.ett import ETT_M1

        assert ETT_M1.default_horizon == 96


class TestETT_M2:
    """Tests for the ETT_M2 instance (15-min)."""

    def test_name(self) -> None:
        """CFG-02: ETT_M2 has correct name."""
        from tscollection.datasets.config.ett import ETT_M2

        assert ETT_M2.name == 'ETTm2'

    def test_split_bounds(self) -> None:
        """CFG-02: ETT_M2 split bounds match 15-min pattern."""
        from tscollection.datasets.config.ett import ETT_M2

        assert ETT_M2.split_bounds == (34560, 46080, 57600)

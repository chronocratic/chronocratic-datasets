"""Tests for Weather forecasting configuration (CFG-02).

Verifies that WeatherConfig correctly inherits from ForecastingConfig,
enforces fractional split mode, and provides valid frozen instances
with correct column selection from rbspaper source.
"""

import pytest
from pydantic import ValidationError

from tscollection.datasets.config.base import ForecastingConfig
from tscollection.datasets.enums import DatasetFamily, SplitMode


class TestWeatherConfigClass:
    """Tests for the WeatherConfig class structure."""

    def test_inherits_from_forecasting_config(self) -> None:
        """CFG-02: WeatherConfig inherits from ForecastingConfig."""
        from tscollection.datasets.config.weather import WeatherConfig

        assert issubclass(WeatherConfig, ForecastingConfig)

    def test_split_mode_returns_fractional(self) -> None:
        """CFG-02: WeatherConfig instances return SplitMode.FRACTIONAL."""
        from tscollection.datasets.config.weather import WEATHER

        assert WEATHER.split_mode == SplitMode.FRACTIONAL

    def test_family_defaults_to_weather(self) -> None:
        """CFG-02: WeatherConfig defaults family to WEATHER."""
        from tscollection.datasets.config.weather import WeatherConfig

        cfg = WeatherConfig(
            name='Test',
            url='https://example.com/test.csv',
            split_bounds=(0.6, 0.2, 0.2),
            tasks=('forecasting',),
        )
        assert cfg.family == DatasetFamily.WEATHER


class TestWEATHER:
    """Tests for the WEATHER instance."""

    def test_name(self) -> None:
        """CFG-02: WEATHER has correct name."""
        from tscollection.datasets.config.weather import WEATHER

        assert WEATHER.name == 'weather'

    def test_split_bounds_fractional(self) -> None:
        """CFG-02: WEATHER uses 60/20/20 fractional splits."""
        from tscollection.datasets.config.weather import WEATHER

        assert WEATHER.split_bounds == (0.6, 0.2, 0.2)

    def test_split_bounds_sum_to_one(self) -> None:
        """CFG-02: Fractional split bounds sum to 1.0."""
        from tscollection.datasets.config.weather import WEATHER

        assert sum(WEATHER.split_bounds) == pytest.approx(1.0)

    def test_univariate_column(self) -> None:
        """CFG-02: WEATHER univariate column is 'last' (iloc[:, -1:])."""
        from tscollection.datasets.config.weather import WEATHER

        assert WEATHER.univariate_column == 'last'

    def test_default_horizon(self) -> None:
        """CFG-02: WEATHER default horizon is 24."""
        from tscollection.datasets.config.weather import WEATHER

        assert WEATHER.default_horizon == 24

    def test_default_seq_len(self) -> None:
        """CFG-02: WEATHER default seq_len is 128."""
        from tscollection.datasets.config.weather import WEATHER

        assert WEATHER.default_seq_len == 128

    def test_is_frozen(self) -> None:
        """CFG-02: WEATHER is frozen."""
        from tscollection.datasets.config.weather import WEATHER

        with pytest.raises(ValueError, match='frozen'):
            WEATHER.name = 'Other'  # type: ignore[attr-defined]

    def test_model_copy(self) -> None:
        """CFG-02: WEATHER.model_copy produces a new instance."""
        from tscollection.datasets.config.weather import WEATHER

        new_cfg = WEATHER.model_copy(update={'default_seq_len': 64})
        assert new_cfg.default_seq_len == 64
        assert WEATHER.default_seq_len == 128  # Original unchanged

"""Tests for the config factory registry (CONFIGS, get_config, list_configs)."""

from __future__ import annotations

import pytest

from tscollection.datasets.config.base import DatasetConfig
from tscollection.datasets.config.factory import (
    CONFIGS,
    get_config,
    list_configs,
)
from tscollection.datasets.enums.data import DatasetFamily


class TestConfigsRegistry:
    """Test CONFIGS dict structure and contents."""

    def test_configs_has_11_entries(self) -> None:
        """CONFIGS must contain exactly 11 dataset configurations."""
        assert len(CONFIGS) == 11

    def test_configs_keys_are_strings(self) -> None:
        """All CONFIGS keys must be string dataset names."""
        for key in CONFIGS.keys():
            assert isinstance(key, str)

    def test_configs_values_are_frozen(self) -> None:
        """All CONFIGS values must be frozen Pydantic BaseModel instances."""
        from pydantic import BaseModel

        for cfg in CONFIGS.values():
            assert isinstance(cfg, BaseModel)
            assert cfg.model_config.get('frozen') is True

    def test_configs_are_dataset_config_subclass(self) -> None:
        """All CONFIGS values must be DatasetConfig subclasses."""
        for cfg in CONFIGS.values():
            assert isinstance(cfg, DatasetConfig)

    def test_contains_ucr_datasets(self) -> None:
        """CONFIGS must contain UCR dataset entries."""
        assert 'Coffee' in CONFIGS
        assert 'ECG200' in CONFIGS
        assert 'FaceFour' in CONFIGS

    def test_contains_uea_datasets(self) -> None:
        """CONFIGS must contain UEA dataset entries."""
        assert 'BasicMotions' in CONFIGS
        assert 'AtrialFibrillation' in CONFIGS

    def test_contains_forecasting_datasets(self) -> None:
        """CONFIGS must contain forecasting dataset entries."""
        assert 'ETTh1' in CONFIGS
        assert 'ETTh2' in CONFIGS
        assert 'ETTm1' in CONFIGS
        assert 'ETTm2' in CONFIGS
        assert 'Electricity' in CONFIGS
        assert 'weather' in CONFIGS


class TestGetConfig:
    """Test get_config lookup function."""

    def test_get_config_coffee_returns_ucr_instance(self) -> None:
        """get_config('Coffee') must return the UCR_COFFEE instance."""
        from tscollection.datasets.config.ucr import UCR_COFFEE

        result = get_config(name='Coffee')
        assert result is UCR_COFFEE

    def test_get_config_etth1_returns_ett_instance(self) -> None:
        """get_config('ETTh1') must return the ETT_H1 instance."""
        from tscollection.datasets.config.ett import ETT_H1

        result = get_config(name='ETTh1')
        assert result is ETT_H1

    def test_get_config_electricity_returns_instance(self) -> None:
        """get_config('Electricity') must return the ELECTRICITY_LOAD instance."""
        from tscollection.datasets.config.electricity import ELECTRICITY_LOAD

        result = get_config(name='Electricity')
        assert result is ELECTRICITY_LOAD

    def test_get_config_weather_returns_instance(self) -> None:
        """get_config('weather') must return the WEATHER instance."""
        from tscollection.datasets.config.weather import WEATHER

        result = get_config(name='weather')
        assert result is WEATHER

    def test_get_config_nonexistent_raises_keyerror(self) -> None:
        """get_config('NonExistent') must raise KeyError."""
        with pytest.raises(KeyError):
            get_config(name='NonExistent')

    def test_get_config_keyword_only(self) -> None:
        """get_config must require keyword arguments."""
        with pytest.raises(TypeError):
            get_config('Coffee')  # type: ignore


class TestListConfigs:
    """Test list_configs filtering function."""

    def test_list_configs_all(self) -> None:
        """list_configs() with no args must return all 11 configs."""
        result = list_configs()
        assert len(result) == 11

    def test_list_configs_ucr_family(self) -> None:
        """list_configs(family=UCR) must return 3 configs."""
        result = list_configs(family=DatasetFamily.UCR)
        assert len(result) == 3
        families = {cfg.family for cfg in result}
        assert families == {DatasetFamily.UCR}

    def test_list_configs_ett_family(self) -> None:
        """list_configs(family=ETT) must return 4 configs."""
        result = list_configs(family=DatasetFamily.ETT)
        assert len(result) == 4
        families = {cfg.family for cfg in result}
        assert families == {DatasetFamily.ETT}

    def test_list_configs_uea_family(self) -> None:
        """list_configs(family=UEA) must return 2 configs."""
        result = list_configs(family=DatasetFamily.UEA)
        assert len(result) == 2

    def test_list_configs_returns_list(self) -> None:
        """list_configs must return a list (not a tuple or generator)."""
        result = list_configs()
        assert isinstance(result, list)

    def test_list_configs_keyword_only(self) -> None:
        """list_configs must use keyword-only arguments."""
        with pytest.raises(TypeError):
            list_configs(DatasetFamily.UCR)  # type: ignore

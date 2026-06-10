"""Tests for public API exports and package configuration."""

from __future__ import annotations

import importlib
import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).parent.parent / 'src' / 'chronocratic' / 'datasets'
EXPECTED_INIT_FILES = [
    PACKAGE_ROOT / '__init__.py',
    PACKAGE_ROOT / 'datatypes' / '__init__.py',
    PACKAGE_ROOT / 'datatypes' / '_base' / '__init__.py',
    PACKAGE_ROOT / 'modules' / '__init__.py',
    PACKAGE_ROOT / 'modules' / '_base' / '__init__.py',
    PACKAGE_ROOT / 'enums' / '__init__.py',
    PACKAGE_ROOT / 'utils' / '__init__.py',
]


def test_version_defined():
    """Package exposes a non-empty __version__ following PEP 440."""
    import chronocratic.datasets

    assert hasattr(chronocratic.datasets, '__version__')
    assert isinstance(chronocratic.datasets.__version__, str)
    assert len(chronocratic.datasets.__version__) > 0


def test_import_chronocratic_datasets():
    """import chronocratic.datasets resolves without errors."""
    ts = importlib.import_module('chronocratic.datasets')
    assert hasattr(ts, '__all__')
    assert '__version__' in ts.__all__


def test_enum_exports_in_root():
    """Enum types are exported from package root."""
    from chronocratic.datasets import (
        ClassificationSplitMode,
        ForecastingMode,
        ScalingMethod,
        TimeSeriesDatasetMode,
    )

    assert TimeSeriesDatasetMode.SAMPLE_LABEL == 'sample_label'
    assert ScalingMethod.MINMAX == 'minmax'
    assert ClassificationSplitMode.AS_DEFINED == 'as_defined'
    assert ForecastingMode.UNIVARIATE == 'univariate'


def test_init_files_exist():
    """__init__.py files exist at all planned levels."""
    for init_path in EXPECTED_INIT_FILES:
        assert init_path.exists(), f'Missing __init__.py: {init_path.relative_to(PACKAGE_ROOT)}'


def test_submodule_all_declarations():
    """Each submodule __init__.py has an __all__ declaration."""
    submodules = [
        'chronocratic.datasets.datatypes',
        'chronocratic.datasets.datatypes._base',
        'chronocratic.datasets.modules',
        'chronocratic.datasets.modules._base',
        'chronocratic.datasets.enums',
        'chronocratic.datasets.utils',
    ]
    for module_name in submodules:
        mod = importlib.import_module(module_name)
        assert hasattr(mod, '__all__'), f'{module_name} missing __all__'


def test_enum_file_exists():
    """Verifies enums/data.py was created."""
    assert (PACKAGE_ROOT / 'enums' / 'data.py').exists()

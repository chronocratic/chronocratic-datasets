"""Tests for package foundation (PKG-01, PKG-02, PKG-03)."""

import importlib
import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).parent.parent / 'src' / 'tscollection' / 'datasets'
EXPECTED_INIT_FILES = [
    PACKAGE_ROOT / '__init__.py',
    PACKAGE_ROOT / 'datasets' / '__init__.py',
    PACKAGE_ROOT / 'datasets' / 'classes' / '__init__.py',
    PACKAGE_ROOT / 'modules' / '__init__.py',
    PACKAGE_ROOT / 'modules' / 'classes' / '__init__.py',
    PACKAGE_ROOT / 'download' / '__init__.py',
    PACKAGE_ROOT / 'config' / '__init__.py',
    PACKAGE_ROOT / 'enums' / '__init__.py',
    PACKAGE_ROOT / 'utils' / '__init__.py',
]


def test_version_defined():
    """PKG-02: Package exposes __version__."""
    import tscollection.datasets

    assert hasattr(tscollection.datasets, '__version__')
    assert tscollection.datasets.__version__ == '0.1.0'


def test_import_tscollection_datasets():
    """PKG-02: import tscollection.datasets resolves without errors."""
    ts = importlib.import_module('tscollection.datasets')
    assert hasattr(ts, '__all__')
    assert '__version__' in ts.__all__


def test_enum_exports_in_root():
    """PKG-02: Enum types are exported from package root."""
    from tscollection.datasets import (
        DistanceMetric,
        ForecastingMode,
        ScalingMethod,
        SplittingStrategy,
        TimeSeriesDatasetMode,
    )

    assert TimeSeriesDatasetMode.WITH_LABELS == 'with_labels'
    assert ScalingMethod.MINMAX == 'minmax'
    assert SplittingStrategy.AS_DEFINED == 'as_defined'
    assert ForecastingMode.UNIVARIATE == 'univariate'
    assert DistanceMetric.EUCLIDEAN == 'euclidean'


def test_init_files_exist():
    """PKG-03: __init__.py files exist at all planned levels."""
    for init_path in EXPECTED_INIT_FILES:
        assert init_path.exists(), f'Missing __init__.py: {init_path.relative_to(PACKAGE_ROOT)}'


def test_submodule_all_declarations():
    """PKG-03: Each submodule __init__.py has an __all__ declaration."""
    submodules = [
        'tscollection.datasets.datasets',
        'tscollection.datasets.datasets.classes',
        'tscollection.datasets.modules',
        'tscollection.datasets.modules.classes',
        'tscollection.datasets.download',
        'tscollection.datasets.config',
        'tscollection.datasets.enums',
        'tscollection.datasets.utils',
    ]
    for module_name in submodules:
        mod = importlib.import_module(module_name)
        assert hasattr(mod, '__all__'), f'{module_name} missing __all__'


def test_enum_file_exists():
    """Verifies enums/data.py was created."""
    assert (PACKAGE_ROOT / 'enums' / 'data.py').exists()

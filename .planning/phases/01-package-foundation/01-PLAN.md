---
wave: 1-3
depends_on: []
autonomous: true
requirements:
  - PKG-01
  - PKG-02
  - PKG-03
waves:
  - wave: 1
    tasks: [1, 2]
    parallel: true
  - wave: 2
    tasks: [3]
    blocked_on: "Wave 1 (needs enums for __init__.py imports)"
  - wave: 3
    tasks: [4, 5]
    blocked_on: "Wave 2 (needs all __init__.py files)"
files_modified:
  - pyproject.toml
  - src/tsdatasets/__init__.py
  - src/tsdatasets/enums/__init__.py
  - src/tsdatasets/enums/data.py
  - src/tsdatasets/datasets/__init__.py
  - src/tsdatasets/datasets/classes/__init__.py
  - src/tsdatasets/modules/__init__.py
  - src/tsdatasets/modules/classes/__init__.py
  - src/tsdatasets/download/__init__.py
  - src/tsdatasets/config/__init__.py
  - src/tsdatasets/utils/__init__.py
  - tests/__init__.py
  - tests/test_package.py
---

# Plan: Package Foundation

## Objective

Establish `tsdatasets` as an installable Python package with proper `__init__.py` exports at all levels, correct dependencies in `pyproject.toml`, and ported enum types from rbspaper source.

## Tasks

### Task 1: Update pyproject.toml

<action>
Update `pyproject.toml` with the following changes to `[project] dependencies`:

1. Add `"pydantic>=2.10,<3.0.0"` — required for typed config models
2. Add `"scikit-learn>=1.6,<2.0.0"` — required by data/scaling.py, modules/abstract.py (MinMaxScaler, StandardScaler, train_test_split, LabelEncoder)
3. Add `"requests>=2.31,<3.0.0"` — required for Phase 4 download/caching
4. Change `"lightning~=2.5.5"` to `"lightning>=2.5,<3.0"` — broaden for compatibility
5. Change `"torch>=2.4.0,<=2.8.0"` to `"torch>=2.4,<3.0"` — broaden for compatibility

Update `[tool.pytest.ini_options]`:
1. Change `testpaths = ["test"]` to `testpaths = ["tests"]` — follow pytest convention
2. Remove `"src"` from `pythonpath` so it becomes `pythonpath = ["."]` — avoid duplicate imports with src-layout editable install
</action>

<read_first>
- pyproject.toml (current state)
- .planning/phases/01-package-foundation/01-RESEARCH.md (dependency analysis)
</read_first>

<acceptance_criteria>
- `pyproject.toml` contains `"pydantic>=2.10,<3.0.0"` in dependencies
- `pyproject.toml` contains `"scikit-learn>=1.6,<2.0.0"` in dependencies
- `pyproject.toml` contains `"requests>=2.31,<3.0.0"` in dependencies
- `pyproject.toml` contains `"lightning>=2.5,<3.0"` in dependencies
- `pyproject.toml` contains `"torch>=2.4,<3.0"` in dependencies
- `pyproject.toml` has `testpaths = ["tests"]` in [tool.pytest.ini_options]
- `pyproject.toml` has `pythonpath = ["."]` in [tool.pytest.ini_options]
- Running `uv pip install -e .` completes without errors
</acceptance_criteria>

### Task 2: Port enums from rbspaper to tsdatasets

<action>
Create `src/tsdatasets/enums/data.py` with StrEnum classes ported from `_sources/rbspaper/src/rbspaper/enums/data_enums.py`. Rename to match project convention from PROJECT.md:

```python
from enum import StrEnum


class TimeSeriesDatasetMode(StrEnum):
    """Mode for how the dataset yields samples."""
    WITH_LABELS = 'with_labels'
    WITHOUT_LABELS = 'without_labels'
    FORECASTING = 'forecasting'


class SplittingStrategy(StrEnum):
    """Strategy for train/test data splitting."""
    AS_DEFINED = 'as_defined'
    MANUAL = 'manual'


class ScalingMethod(StrEnum):
    """Method for data scaling."""
    NONE = 'none'
    MINMAX = 'minmax'
    STANDARD = 'standard'


class ForecastingMode(StrEnum):
    """Whether forecasting is univariate or multivariate."""
    UNIVARIATE = 'univariate'
    MULTIVARIATE = 'multivariate'


class DistanceMetric(StrEnum):
    """Distance metric for time series comparison."""
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    SOFT_DTW = 'soft_dtw'
    COSINE = 'cosine'
```

Key renames from rbspaper:
- `TimeSeriesClassificationDatasetSplittingStrategy` -> `SplittingStrategy`
- `TimeSeriesDistanceMetric` -> `DistanceMetric`
- `ForecastingTimeSeriesDatasetMode` -> `ForecastingMode`
- `ScalingMethod` is NEW (not in rbspaper) — added for project constraint

Create `src/tsdatasets/enums/__init__.py`:
```python
"""Typed enumerations for dataset parameters."""

from tsdatasets.enums.data import (
    DistanceMetric,
    ForecastingMode,
    ScalingMethod,
    SplittingStrategy,
    TimeSeriesDatasetMode,
)

__all__ = [
    'DistanceMetric',
    'ForecastingMode',
    'ScalingMethod',
    'SplittingStrategy',
    'TimeSeriesDatasetMode',
]
```
</action>

<read_first>
- _sources/rbspaper/src/rbspaper/enums/data_enums.py (source enums)
- .planning/PROJECT.md (enum naming conventions)
- .planning/phases/01-package-foundation/01-RESEARCH.md (Open Question 3 about enum scope)
</read_first>

<acceptance_criteria>
- `src/tsdatasets/enums/data.py` exists with 5 StrEnum classes: TimeSeriesDatasetMode, SplittingStrategy, ScalingMethod, ForecastingMode, DistanceMetric
- `src/tsdatasets/enums/__init__.py` exists with all 5 enums in `__all__`
- Each enum class has a docstring
- `python -c "from tsdatasets.enums import TimeSeriesDatasetMode, ScalingMethod, SplittingStrategy, ForecastingMode, DistanceMetric"` exits 0
- No `src.rbspaper` imports — all relative imports use `tsdatasets`
</acceptance_criteria>

### Task 3: Create __init__.py skeleton at all levels

<action>
Create the following `__init__.py` files. They define `__all__` with intended symbols but do NOT import non-existent implementation modules yet (two-wave approach from research).

**`src/tsdatasets/__init__.py`:**
```python
"""tsdatasets -- Zero-config time series datasets for PyTorch Lightning."""

from __future__ import annotations

__version__ = '0.1.0'

# Enums available immediately
from tsdatasets.enums import (
    DistanceMetric,
    ForecastingMode,
    ScalingMethod,
    SplittingStrategy,
    TimeSeriesDatasetMode,
)

__all__ = [
    '__version__',
    # Enums
    'DistanceMetric',
    'ForecastingMode',
    'ScalingMethod',
    'SplittingStrategy',
    'TimeSeriesDatasetMode',
    # Datasets (populated in Phase 2)
    # Modules (populated in Phase 5)
    # Config (populated in Phase 3)
    # Factory (populated in Phase 6)
]
```

**`src/tsdatasets/datasets/__init__.py`:**
```python
"""Time series dataset classes (PyTorch Dataset)."""

__all__ = []  # Populated in Phase 2
```

**`src/tsdatasets/datasets/classes/__init__.py`:**
```python
"""Abstract base classes for time series datasets."""

__all__ = []  # Populated in Phase 2
```

**`src/tsdatasets/modules/__init__.py`:**
```python
"""LightningDataModule classes for time series datasets."""

__all__ = []  # Populated in Phase 5
```

**`src/tsdatasets/modules/classes/__init__.py`:**
```python
"""Abstract base classes for time series data modules."""

__all__ = []  # Populated in Phase 5
```

**`src/tsdatasets/download/__init__.py`:**
```python
"""Data download and caching utilities."""

__all__ = []  # Populated in Phase 4
```

**`src/tsdatasets/config/__init__.py`:**
```python
"""Pydantic configuration models for dataset metadata."""

__all__ = []  # Populated in Phase 3
```

**`src/tsdatasets/utils/__init__.py`:**
```python
"""Utility functions for data processing."""

__all__ = []  # Populated in Phase 2
```
</action>

<read_first>
- src/tsdatasets/ (current state — only .gitkeep)
- .planning/phases/01-package-foundation/01-RESEARCH.md (Pattern 1: __init__.py Export Hierarchy)
</read_first>

<acceptance_criteria>
- All 9 __init__.py files listed above exist
- `src/tsdatasets/__init__.py` has `__version__ = '0.1.0'` and exports all 5 enums
- Each submodule __init__.py has a docstring and an `__all__` list
- No import errors when running `python -c "import tsdatasets"`
- `python -c "import tsdatasets; print(tsdatasets.__version__)"` outputs `0.1.0`
</acceptance_criteria>

### Task 4: Create tests directory and package verification test

<action>
Create `tests/__init__.py` (empty file for test package marker).

Create `tests/test_package.py`:
```python
"""Tests for package foundation (PKG-01, PKG-02, PKG-03)."""

import importlib
import pathlib


PACKAGE_ROOT = pathlib.Path(__file__).parent.parent / 'src' / 'tsdatasets'
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
    import tsdatasets
    assert hasattr(tsdatasets, '__version__')
    assert tsdatasets.__version__ == '0.1.0'


def test_import_tsdatasets():
    """PKG-02: import tsdatasets resolves without errors."""
    ts = importlib.import_module('tsdatasets')
    assert hasattr(ts, '__all__')
    assert '__version__' in ts.__all__


def test_enum_exports_in_root():
    """PKG-02: Enum types are exported from package root."""
    from tsdatasets import (
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
        assert init_path.exists(), f"Missing __init__.py: {init_path.relative_to(PACKAGE_ROOT)}"


def test_submodule_all_declarations():
    """PKG-03: Each submodule __init__.py has an __all__ declaration."""
    submodules = [
        'tsdatasets.datasets',
        'tsdatasets.datasets.classes',
        'tsdatasets.modules',
        'tsdatasets.modules.classes',
        'tsdatasets.download',
        'tsdatasets.config',
        'tsdatasets.enums',
        'tsdatasets.utils',
    ]
    for module_name in submodules:
        mod = importlib.import_module(module_name)
        assert hasattr(mod, '__all__'), f"{module_name} missing __all__"


def test_enum_file_exists():
    """Verifies enums/data.py was created."""
    assert (PACKAGE_ROOT / 'enums' / 'data.py').exists()
```
</action>

<read_first>
- pyproject.toml (to verify testpaths = ["tests"])
- .planning/REQUIREMENTS.md (PKG-01, PKG-02, PKG-03 acceptance criteria)
</read_first>

<acceptance_criteria>
- `tests/__init__.py` exists (empty file)
- `tests/test_package.py` exists with 7 test functions
- `pytest tests/test_package.py -x` exits 0 with all 7 tests passing
- Test covers PKG-02 (import tsdatasets works, version defined, enum exports) and PKG-03 (__init__.py files exist, __all__ declarations)
</acceptance_criteria>

### Task 5: Reinstall package and verify

<action>
Run `uv pip install -e .` to reinstall with updated dependencies, then run tests.

Commands:
1. `uv pip install -e .` — reinstalls editable package with new deps (pydantic, scikit-learn, requests)
2. `uv pip install pydantic scikit-learn requests` — ensures new deps are resolved
3. `uv run pytest tests/test_package.py -x -v` — runs all package tests
4. `uv run python -c "import tsdatasets; print(tsdatasets.__version__)"` — smoke test
</action>

<read_first>
- pyproject.toml (updated deps)
- tests/test_package.py (the test to run)
</read_first>

<acceptance_criteria>
- `uv pip install -e .` completes without errors
- `uv run pytest tests/test_package.py -x -v` shows 7 passed, 0 failed
- `uv run python -c "import tsdatasets; print(tsdatasets.__version__)"` prints `0.1.0`
- No `ImportError` or `ModuleNotFoundError`
</acceptance_criteria>

## Verification Criteria

| Criteria | How to Verify |
|----------|--------------|
| PKG-01: pip install -e . works | `uv pip install -e .` exits 0 |
| PKG-02: import tsdatasets works | `python -c "import tsdatasets; print(tsdatasets.__version__)"` outputs `0.1.0` |
| PKG-02: Public API from root | Enums importable from `tsdatasets` root |
| PKG-03: __init__.py at all levels | All 9 __init__.py files exist with __all__ declarations |
| Tests pass | `pytest tests/test_package.py -x -v` — 7 passed, 0 failed |
| No src.rbspaper imports | `grep -r "src\.rbspaper" src/tsdatasets/` returns nothing |

## Must Haves

1. `pip install -e .` resolves all dependencies without errors
2. `import tsdatasets` works and exposes `__version__` and enum types
3. All `__init__.py` files have `__all__` declarations
4. Tests directory is `tests/` (not `test/`) with proper pytest config
5. No circular imports on `import tsdatasets`

## Threat Model

<threat_model>
**ASVS Level 1**

| Threat | Severity | Mitigation |
|--------|----------|-----------|
| Supply chain: compromised dependency | Medium | All new deps (pydantic, scikit-learn, requests) pinned with lower bounds to stable minor versions; upper bounds to major version |
| Directory traversal in future download paths | Low (Phase 4) | Phase 1 establishes `~/.cache/tsdatasets/` as the target path constant; Phase 4 will sanitize |

**No auth, session, or access control concerns — this is a data package, not a service.**
</threat_model>

# Phase 1: Package Foundation - Research

**Researched:** 2026-05-08
**Domain:** Python packaging, src-layout, pyproject.toml configuration
**Confidence:** HIGH

## Summary

This phase establishes the `tsdatasets` package as an installable Python package with proper `__init__.py` export hierarchy. The project already uses the src-layout pattern (`src/tsdatasets/`) with setuptools in `pyproject.toml`, but the package directory currently contains only a `.gitkeep` file and has no `__init__.py` files. The pyproject.toml is missing three critical runtime dependencies: `pydantic` (required by the project for configuration), `scikit-learn` (used directly by data/scaling.py and data/modules/abstract.py), and `requests` (needed for Phase 4 download/caching utilities).

The existing `pyproject.toml` has a partial test path misconfiguration (`test` instead of `tests/`) and the pytest section adds `src` to `pythonpath` which is redundant when using the src-layout with editable installs. Phase 1 will create the skeleton `__init__.py` files with placeholder exports (importing from submodules that will be filled in by later phases), add the missing dependencies, and clean up the pyproject.toml configuration.

**Primary recommendation:** Extend the existing src-layout pyproject.toml by adding missing dependencies (pydantic, scikit-learn, requests), create `__init__.py` files at all planned levels with forward-reference exports, and fix the test path configuration.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Package installation | Build system (setuptools) | — | pyproject.toml + src-layout |
| Public API surface | Package root (`__init__.py`) | Submodule `__init__.py` files | Top-level imports re-export from submodules |
| Dependency resolution | pyproject.toml | — | Declares all runtime and dev requirements |
| Test discovery | pytest config | — | Validates imports resolve correctly |

## User Constraints (from STATE.md Decisions)

### Locked Decisions
- rbspaper as primary source -- better docstrings, defensive code, existing registry
- Pydantic v2 for config -- typed, validated, frozen models
- Auto-download in prepare_data() -- torchtime pattern, user provides no file paths
- Family-prefixed imports -- UCRCoffeeModule disambiguates across families
- Classification seq_len from data -- intrinsic property, computed in prepare_data(), read-only
- Forecasting seq_len user-configurable -- with registry default, flexible for different use cases
- Cache-only download -- raw data in ~/.cache/tsdatasets/, SHA256 validated
- One config class per family -- UCRConfig with instances per dataset, not one class per dataset
- Enums for typed params -- ScalingMethod, SplittingStrategy, ForecastingMode -- no raw strings
- Modules return LightningDataModule -- not DataLoader, needed for Trainer integration

### Tech Stack Constraint (from PROJECT.md)
- Python 3.12, PyTorch, Lightning, Pydantic v2, numpy, pandas, scipy

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PKG-01 | User can install tsdatasets via pip with all dependencies resolved | Standard Stack table, pyproject.toml analysis, missing dependency inventory |
| PKG-02 | User can import the public API from `tsdatasets` package root | `__init__.py` patterns, export hierarchy design |
| PKG-03 | Package includes proper `__init__.py` exports at all levels | Source code analysis of rbspaper `__init__.py` patterns |

## Standard Stack

### Core Packaging
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| setuptools>=68 | 68+ (build-backend) | Build system, package discovery | Already configured in pyproject.toml, supports src-layout natively [VERIFIED: pyproject.toml] |
| pydantic>=2.10,<3.0.0 | 2.13.4 latest | Typed configuration models | Project constraint: "Pydantic v2 for config" locked decision; v2.10+ stabilizes frozen models and computed fields [VERIFIED: PyPI] |
| scikit-learn>=1.6,<2.0.0 | 1.8.0 latest | Data scaling (MinMaxScaler, StandardScaler), train_test_split, LabelEncoder | Directly imported by `data/utils/scaling.py`, `data/modules/abstract.py`, `data/modules/ucr_datamodule.py`, `data/modules/uea_datamodule.py` [VERIFIED: grep of rbspaper source] |
| requests>=2.31,<3.0.0 | 2.33.1 latest | HTTP downloads for data fetching | Required by Phase 4 download/caching; torchtime pattern uses requests for streaming downloads [VERIFIED: torchtime source / PyPI] |

### Runtime Dependencies (already in pyproject.toml)
| Library | Version Constraint | Current Installed | Latest on PyPI | Notes |
|---------|-------------------|-------------------|----------------|-------|
| torch | >=2.4.0,<=2.8.0 | 2.8.0 | 2.11.0 | Upper bound pins to 2.8.0; keep for now, revisit if compatibility issues arise [VERIFIED: pyproject.toml] |
| lightning | ~=2.5.5 | 2.5.6 | 2.6.1 | Compatible release constraint on 2.5.x [VERIFIED: pyproject.toml + uv pip list] |
| numpy | >=2.1,<3.0.0 | 2.4.4 | 2.4.4 | Already correct [VERIFIED: uv pip list] |
| pandas | >=2.2.0 | 3.0.2 | 3.0.2 | Already correct [VERIFIED: uv pip list] |
| scipy | >=1.13.0 | 1.17.1 | 1.17.1 | Already correct [VERIFIED: uv pip list] |
| tqdm | >=4.66.0 | 4.67.3 | 4.67.3 | Already correct [VERIFIED: uv pip list] |
| joblib | >=1.4.0 | 1.5.3 | 1.5.3 | Already correct [VERIFIED: uv pip list] |
| torchvision | >=0.19.0 | 0.23.0 | 0.23.0 | Present but not used by data/ modules; kept as existing dep [ASSUMED] |
| torchaudio | >=2.4.0 | 2.11.0 | 2.11.0 | Present but not used by data/ modules; kept as existing dep [ASSUMED] |
| openpyxl | ~=3.1.5 | 3.1.5 | 3.1.5 | Excel support; may not be needed by tsdatasets [ASSUMED] |
| h5py | ~=3.16.0 | 3.16.0 | 3.16.0 | HDF5 support; may not be needed by tsdatasets [ASSUMED] |

### Development Dependencies (already in pyproject.toml)
| Library | Version | Purpose |
|---------|---------|---------|
| pytest>=8.2 | 9.0.3 installed | Unit testing framework |
| pytest-cov>=5.0 | 7.1.0 installed | Coverage reporting |
| ruff>=0.15.9 | 0.15.12 installed | Linting and formatting |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| setuptools | hatchling, flit | setuptools already configured, works well with src-layout; switching adds migration risk |
| pydantic v1 | attr | Project locked on Pydantic v2; attr lacks model validation and frozen semantics |
| requests | urllib, httpx | requests is the standard in the ecosystem (torchtime uses it); httpx adds async complexity not needed yet |

**Installation (missing deps to add):**
```bash
uv pip install 'pydantic>=2.10,<3.0.0' 'scikit-learn>=1.6,<2.0.0' 'requests>=2.31,<3.0.0'
```

## Architecture Patterns

### System Architecture Diagram

```
pyproject.toml (build-system, dependencies)
       |
       v
  pip install -e .  ------>  src/tsdatasets/
                                   |
            +------------------+--+------+------------------+
            |                  |      |                  |
      tsdatasets/       tsdatasets/  tsdatasets/     tsdatasets/
      __init__.py       datasets/    modules/        enums/
      (public API)      __init__.py  __init__.py     __init__.py
                        |            |               |
                   classes/       classes/        data.py
                   __init__.py    __init__.py     (StrEnum types)
                   (ABC exports)  (ABC exports)
            +------+------+
            |             |
        [future:    [future:
        ucr.py      ett.py
        uea.py      ...
        ...]        ]
```

Phase 1 creates the `__init__.py` skeleton. Later phases fill in the implementation files.

### Recommended Project Structure

The structure from PROJECT.md is the target. Phase 1 creates `__init__.py` stubs:

```
src/tsdatasets/
+-- __init__.py              # Phase 1: Public API surface (re-exports)
+-- datasets/
|   +-- __init__.py          # Phase 1: empty stub, Phase 2: dataset exports
|   +-- classes/
|       +-- __init__.py      # Phase 1: empty stub, Phase 2: ABC exports
+-- modules/
|   +-- __init__.py          # Phase 1: empty stub, Phase 5: module exports
|   +-- classes/
|       +-- __init__.py      # Phase 1: empty stub, Phase 5: ABC exports
+-- download/
|   +-- __init__.py          # Phase 1: empty stub, Phase 4: download exports
+-- config/
|   +-- __init__.py          # Phase 1: empty stub, Phase 3: config exports
+-- enums/
|   +-- __init__.py          # Phase 1: enum exports (from rbspaper enums)
+-- utils/
    +-- __init__.py          # Phase 1: empty stub, Phase 2: utility exports
```

### Pattern 1: `__init__.py` Export Hierarchy

**What:** Submodule `__init__.py` files re-export public symbols from implementation files, and the top-level `__init__.py` re-exports from submodules. This creates a clean import path without exposing internal file structure.

**When to use:** All phases. Phase 1 creates the skeleton with forward-reference style imports (pointing to modules that will exist in later phases).

**Example (rbspaper pattern -- observed in source):**
```python
# _sources/rbspaper/src/rbspaper/data/datasets/__init__.py (existing pattern)
from src.rbspaper.data.datasets.abstract import (
    FixedTimeSeriesDataset,
    FixedTimeSeriesDatasetMultivariate,
    FixedTimeSeriesDatasetUnivariate,
    ...
)
from src.rbspaper.data.datasets.strategies import (
    ClassificationStrategyMultipleFiles,
    ...
)

__all__ = [
    'ClassificationStrategyMultipleFiles',
    'FixedTimeSeriesDataset',
    ...
]
```

**tsdatasets target pattern (after Phase 1):**
```python
# src/tsdatasets/__init__.py (Phase 1 skeleton)
"""tsdatasets -- Zero-config time series datasets for PyTorch Lightning."""

__version__ = '0.1.0'

# Enums (available from Phase 1, filled in later)
from tsdatasets.enums import (
    TimeSeriesDatasetMode,
    ScalingMethod,
    SplittingStrategy,
    ForecastingMode,
)

# Placeholders -- imports will resolve once submodule files exist
# (datasets, modules, config, download, utils are empty stubs in Phase 1)

__all__ = [
    '__version__',
    # enums
    'TimeSeriesDatasetMode',
    'ScalingMethod',
    'SplittingStrategy',
    'ForecastingMode',
]
```

**Note for Phase 1:** The `__init__.py` files created in this phase should use a two-wave approach:
- Wave A (Phase 1): Create `__init__.py` stubs that define `__all__` with the intended symbols but do NOT import non-existent implementation modules yet.
- Wave B (Phase 2+): Add the actual `from ... import ...` lines as implementation files are created.

This prevents `ImportError` at each intermediate step.

### Pattern 2: StrEnum-based Typed Parameters

**What:** Use Python `StrEnum` (available since 3.11) for all enumerated parameters instead of raw strings. This is already used in the rbspaper source (`_sources/rbspaper/src/rbspaper/enums/data_enums.py`).

**When to use:** Phase 1 for the enums package; inherited by all later phases.

**Example (from rbspaper source):**
```python
from enum import StrEnum

class TimeSeriesDatasetMode(StrEnum):
    WITH_LABELS = 'with_labels'
    WITHOUT_LABELS = 'without_labels'
    FORECASTING = 'forecasting'

class TimeSeriesClassificationDatasetSplittingStrategy(StrEnum):
    AS_DEFINED = 'as_defined'
    MANUAL = 'manual'
```

### Anti-Patterns to Avoid

- **`from src.rbspaper` absolute imports:** The rbspaper source uses `src.rbspaper` prefix imports. In tsdatasets, use clean `tsdatasets` imports. The src-layout with editable install makes `tsdatasets` available directly without the `src.` prefix.
- **Blanket `import *` in `__init__.py`:** Always use explicit named imports with `__all__` declarations. The rbspaper source correctly avoids this pattern.
- **Circular imports via eager top-level imports:** If `tsdatasets/__init__.py` imports from `tsdatasets.config` which imports from `tsdatasets.enums`, avoid circular chains. Use lazy imports or restructure the dependency order.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Package build system | Custom setup.py, Makefile-based builds | setuptools via pyproject.toml | Already configured, standard, PEP 517 compliant |
| Data scaling | Custom min-max/standard logic | sklearn.preprocessing.MinMaxScaler, StandardScaler | Already used by rbspaper source; battle-tested with edge cases |
| HTTP downloads | Raw urllib with manual retries | requests library | Standard ecosystem choice, handles redirects, streaming, timeouts; torchtime pattern uses it |
| Enum types | String constants or ints | enum.StrEnum (Python 3.11+) | Type-safe, hashable, string-compatible; already used in rbspaper source |
| Configuration validation | Manual type checks | pydantic v2 BaseModel | Locked decision; provides frozen models, Field validators, computed_fields |
| Dataset splits | Ad-hoc train/test logic | sklearn.model_selection.train_test_split | Already used by rbspaper ucr_datamodule.py, uea_datamodule.py |

**Key insight:** The rbspaper source already uses scikit-learn for scaling and splitting. The tsdatasets package must declare scikit-learn as an explicit dependency even though it is not currently in pyproject.toml.

## Runtime State Inventory

> This is a greenfield package structure phase -- no rename/refactor/migration of existing runtime state. The phase creates new files in an empty `src/tsdatasets/` directory.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None | -- |
| Live service config | None | -- |
| OS-registered state | None | -- |
| Secrets/env vars | None | -- |
| Build artifacts | `src/tsdatasets.egg-info/` exists from previous editable install | Reinstall package after pyproject.toml changes (`uv pip install -e .`) |

## Common Pitfalls

### Pitfall 1: Missing scikit-learn in dependencies

**What goes wrong:** pyproject.toml lists torch, lightning, numpy, pandas, scipy but not scikit-learn. The data/scaling.py and data/modules/abstract.py files directly import from sklearn. If someone installs tsdatasets without scikit-learn pre-installed, `import tsdatasets.utils` or `import tsdatasets.modules` will crash with `ModuleNotFoundError: No module named 'sklearn'`.

**Why it happens:** scikit-learn is a transitive dependency of `lightning` and `torchmetrics` in the current environment, so it is silently installed by uv. The pyproject.toml author may not have noticed it was missing.

**How to avoid:** Explicitly add `scikit-learn>=1.6,<2.0.0` to the `[project] dependencies` list in pyproject.toml during Phase 1.

**Warning signs:** grep for `sklearn` or `scikit` in all source files and verify the dependency is declared.

### Pitfall 2: `src.` prefix in imports

**What goes wrong:** Copying rbspaper imports directly results in `from src.rbspaper.data.utils` instead of `from tsdatasets.utils`. The `src.` prefix is a development-time artifact of the rbspaper project's structure. In the tsdatasets src-layout with editable install, the import path is just `tsdatasets`, not `src.tsdatasets`.

**Why it happens:** The rbspaper source files were developed inside the rbspaper project where `src.rbspaper` was the import path. When porting to tsdatasets, all imports must be rewritten.

**How to avoid:** Phase 1 only creates `__init__.py` stubs; the actual import rewrites happen in later phases. Document this clearly so the planner knows that Phase 1 does not need to fix imports in implementation files.

**Warning signs:** Any `from src.` import in `src/tsdatasets/` files is a bug.

### Pitfall 3: pytest `pythonpath` redundancy with src-layout

**What goes wrong:** The current pyproject.toml has `pythonpath = [".", "src"]` in `[tool.pytest.ini_options]`. With the src-layout and editable install (`pip install -e .`), `src` is already on the Python path via the installed package. Adding `src` to `pythonpath` can cause duplicate module imports and test discovery issues.

**Why it happens:** Developers often add `src` to pythonpath as a workaround before the package is installed. It becomes stale configuration.

**How to avoid:** Remove the `pythonpath` line or change it to `pythonpath = ["."]` only. The `src/` directory should be accessible via the installed package name (`tsdatasets`).

**Warning signs:** Tests that pass in dev mode but fail in CI (or vice versa) often indicate pythonpath confusion.

### Pitfall 4: `__init__.py` import order causing circular dependencies

**What goes wrong:** If `tsdatasets/__init__.py` imports from `tsdatasets.config`, and `tsdatasets.config` imports from `tsdatasets.enums`, the import chain must be carefully ordered. Phase 1 stubs should define `__all__` but defer actual cross-module imports.

**Why it happens:** Python executes `__init__.py` top-to-bottom, and importing a partially-initialized module raises `ImportError`.

**How to avoid:** Use the two-wave approach documented in Pattern 1. Phase 1 `__init__.py` files that have no downstream dependencies yet should only contain docstrings, `__all__` declarations, and version strings.

**Warning signs:** `ImportError` or `AttributeError: partially initialized module` on `import tsdatasets`.

### Pitfall 5: Version pinning too tightly

**What goes wrong:** The current pyproject.toml uses `~=2.5.5` for lightning (allows only 2.5.x) and `<=2.8.0` for torch. If Lightning 2.6+ has critical fixes or the torch upper bound blocks newer PyTorch releases, the package becomes hard to maintain.

**Why it happens:** Pinning to a specific minor version is safe for development but reduces compatibility.

**How to avoid:** Use broader compatible ranges: `lightning>=2.5,<3.0` and `torch>=2.4,<3.0` with the understanding that tests verify compatibility. The current constraints are acceptable for Phase 1 but should be reviewed.

**Warning signs:** `uv pip install` resolves slowly or fails due to conflicting version constraints.

## Code Examples

### pyproject.toml -- Updated dependencies section

```toml
[project]
name = "tsdatasets"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=2.1,<3.0.0",
    "pandas>=2.2.0",
    "scipy>=1.13.0",
    "scikit-learn>=1.6,<2.0.0",      # NEW: required by data/scaling.py, modules/abstract.py
    "lightning>=2.5,<3.0",            # BROADENED: was ~=2.5.5
    "torch>=2.4,<3.0",                # BROADENED: was <=2.8.0
    "torchvision>=0.19.0",
    "torchaudio>=2.4.0",
    "pydantic>=2.10,<3.0.0",          # NEW: project constraint, Pydantic v2
    "requests>=2.31,<3.0.0",          # NEW: required by Phase 4 download/caching
    "tqdm>=4.66.0",
    "joblib>=1.4.0",
    "openpyxl~=3.1.5",
    "h5py~=3.16.0",
]
```

### `__init__.py` -- Phase 1 skeleton pattern

```python
# src/tsdatasets/__init__.py
"""tsdatasets -- Zero-config time series datasets for PyTorch Lightning."""

__version__ = '0.1.0'

# Enum types -- these will be populated as Phase 1 creates the enums package
from tsdatasets.enums import (
    TimeSeriesDatasetMode,
    ScalingMethod,
    SplittingStrategy,
    ForecastingMode,
)

__all__ = [
    '__version__',
    'TimeSeriesDatasetMode',
    'ScalingMethod',
    'SplittingStrategy',
    'ForecastingMode',
]
```

### `__init__.py` -- Submodule stub pattern (empty, ready for later phases)

```python
# src/tsdatasets/datasets/__init__.py
"""Time series dataset classes (PyTorch Dataset)."""

__all__ = []  # Populated in Phase 2
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|-------------------|--------------|--------|
| setup.py for packaging | pyproject.toml (PEP 517/518) | PEP 518 (2020), standard now | No setup.py needed; declarative config |
| Flat layout (package in root) | src-layout (package in src/) | Community best practice | Prevents import-shadowing bugs during development |
| dependency-versions via pip freeze | Lock files (uv.lock) | Standard with uv/pip-tools | Reproducible environments |
| dataclasses for config | Pydantic v2 frozen models | Pydantic v2 (2023) | Runtime validation, type safety |

**Deprecated/outdated:**
- `setup.py`: Not needed when pyproject.toml + setuptools is configured. The current project correctly does not have a setup.py.
- `python_requires` as a string: The current pyproject.toml correctly uses `requires-python = ">=3.12,<3.13"`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `torchvision` and `torchaudio` are not directly used by tsdatasets data modules | Standard Stack (Runtime) | Low -- they are kept as existing deps; can be audited in a later cleanup phase |
| A2 | `openpyxl` and `h5py` are not directly used by the data/ modules being ported | Standard Stack (Runtime) | Low -- kept as existing deps; can be removed if confirmed unused after full port |
| A3 | `lightning~=2.5.5` should be broadened to `>=2.5,<3.0` for better compatibility | Code Examples (pyproject.toml) | Medium -- the current pin works but is restrictive; broadening may expose incompatibilities |
| A4 | The `test` directory referenced in `pyproject.toml` (`testpaths = ["test"]`) should be `tests/` | Common Pitfalls | Low -- standard convention; the current `test` path matches the rbspaper layout |
| A5 | `scikit-learn` is transitively installed via `lightning` in the current environment | Common Pitfalls (Pitfall 1) | Low -- verified by uv pip list showing sklearn present; removing it from explicit deps would be a bug |

## Open Questions

1. **Should `torchvision` and `torchaudio` be removed from dependencies?**
   - What we know: Neither is imported by any file in `_sources/rbspaper/src/rbspaper/data/`.
   - What's unclear: Whether they were added for future use or are left over from a broader project.
   - Recommendation: Keep them for Phase 1 (minimize diff). Audit and remove in a dedicated cleanup phase if confirmed unused.

2. **Should the `pythonpath` in pytest config be `["test"]` or `["tests"]`?**
   - What we know: pyproject.toml says `testpaths = ["test"]`. The rbspaper source has a `test/` directory. There is no `tests/` directory in the tsdatasets project root.
   - What's unclear: Whether the planner intends to create `test/` or `tests/`.
   - Recommendation: Create `tests/` following pytest convention and update `testpaths = ["tests"]` in pyproject.toml. The `test/` vs `tests/` naming does not affect functionality but `tests/` is the more common convention.

3. **What should the enums package contain in Phase 1 vs Phase 3?**
   - What we know: The rbspaper source has `TimeSeriesDatasetMode`, `SplittingStrategy`, `ForecastingMode` etc. in `enums/data_enums.py`. Phase 3 creates Pydantic configs which define `ScalingMethod`, `DatasetFamily` etc.
   - What's unclear: Whether the enums needed by `tsdatasets/__init__.py` (to satisfy PKG-02) should be created in Phase 1 or deferred to Phase 3.
   - Recommendation: Port the existing StrEnum types from rbspaper in Phase 1 (`TimeSeriesDatasetMode`, `TimeSeriesClassificationDatasetSplittingStrategy`, `TimeSeriesDistanceMetric`, `ForecastingTimeSeriesDatasetMode`). Rename them to match the project convention (`ScalingMethod`, `SplittingStrategy`, `ForecastingMode`) as defined in PROJECT.md. Phase 3 can add additional enums like `DatasetFamily`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All code | Yes | 3.12 (from requires-python) | -- |
| uv | Environment management | Yes | -- | pip |
| setuptools | Build system | Yes | >=68 (in pyproject.toml) | -- |
| ruff | Linting/formatting | Yes | 0.15.12 | -- |
| pytest | Tests | Yes | 9.0.3 | -- |
| torch | Dataset base classes | Yes | 2.8.0 | -- |
| lightning | DataModule base classes | Yes | 2.5.6 | -- |
| numpy | Data arrays | Yes | 2.4.4 | -- |
| pandas | DataFrames | Yes | 3.0.2 | -- |
| scipy | Signal processing | Yes | 1.17.1 | -- |
| pydantic | Config models | Not installed yet | -- | Must add to pyproject.toml |
| scikit-learn | Scaling, splitting | Yes (transitive) | -- | Must add to pyproject.toml |
| requests | Download (Phase 4) | Not checked | -- | Should add to pyproject.toml |

**Missing dependencies with no fallback:**
- None. pydantic and scikit-learn can be added to pyproject.toml and installed.

**Missing dependencies with fallback:**
- None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 (installed) |
| Config file | `pyproject.toml` section `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/ -x` |
| Full suite command | `pytest tests/ -x --cov=tsdatasets` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PKG-01 | `pip install -e .` resolves deps | smoke | `uv pip install -e . && echo SUCCESS` | N/A (manual) |
| PKG-02 | `import tsdatasets` works | unit | `python -c "import tsdatasets; print(tsdatasets.__version__)"` | Wave 0 |
| PKG-03 | `__init__.py` files exist at all levels | unit | `pytest tests/test_init_exports.py -x` | Wave 0 |

### Wave 0 Gaps
- [ ] `tests/__init__.py` -- test package marker
- [ ] `tests/test_package.py` -- covers PKG-01, PKG-02, PKG-03: verifies editable install works, imports resolve, `__init__.py` files exist with `__all__` declarations

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | pydantic v2 model validation on all config classes |
| V2 Authentication | no | -- |
| V3 Session Management | no | -- |
| V4 Access Control | no | -- |
| V6 Cryptography | partial | SHA256 for download validation (Phase 4); use `hashlib.sha256` (stdlib) |

### Known Threat Patterns for Python packaging

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Supply chain attack via compromised deps | Tampering | Pin dependency versions; review new deps before adding |
| Directory traversal in download paths | Spoofing | Validate and sanitize paths in `~/.cache/tsdatasets/` |

## Sources

### Primary (HIGH confidence)
- pyproject.toml at `/Users/skaf/VSCodeProjects/tsdatasets/pyproject.toml` -- current build configuration, dependency list
- `/Users/skaf/VSCodeProjects/tsdatasets/src/tsdatasets.egg-info/` -- current package metadata
- PyPI API (`https://pypi.org/pypi/{pkg}/json`) -- verified current versions for pydantic (2.13.4), scikit-learn (1.8.0), requests (2.33.1), torch (2.11.0), lightning (2.6.1)
- uv pip list -- verified installed versions in the project virtual environment

### Secondary (MEDIUM confidence)
- rbspaper source `__init__.py` files (`_sources/rbspaper/src/rbspaper/data/`) -- export patterns used as reference
- torchtime source (`/tmp/torchtime-extract/src/torchtime/utils.py`) -- download/caching pattern reference
- PROJECT.md -- package structure definition, locked decisions

### Tertiary (LOW confidence)
- Assumption that `torchvision`, `torchaudio`, `openpyxl`, `h5py` are unused by tsdatasets data modules -- needs verification after full port

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- versions verified against PyPI API and uv pip list; source code grep confirms actual imports
- Architecture: HIGH -- src-layout with setuptools is a well-established pattern; rbspaper export patterns provide concrete reference
- Pitfalls: HIGH -- all pitfalls identified by direct code analysis (grep, file reading) of the existing source

**Research date:** 2026-05-08
**Valid until:** 2026-06-08 (stable packaging ecosystem; Python packaging standards do not change rapidly)

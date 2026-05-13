# Technical Concerns

Last mapped: 2026-05-08

## Architecture Concerns

### Monorepo without Unification

**Severity:** Medium

The `src/tscollection/datasets/` package has the Phase 1 skeleton in place (__init__.py hierarchy, enums, pyproject.toml). Actual dataset and module code is still being extracted from `_sources/autotsrc/` and `_sources/rbspaper/` — independent packages with separate `pyproject.toml`, `uv.lock`, and `src/` layouts. The namespace is `tscollection.datasets` (PEP 420 implicit namespace at the `tscollection/` level).

**Impact:** `import tscollection.datasets` works and exposes enums + version, but dataset/module classes are not yet ported. Each source sub-project must still be run from its own directory for development purposes.

**Revisit when:** The project intends to publish as a single package or provide a unified API.

### Code Duplication Between Sub-Projects

**Severity:** Medium

autotsrc and rbspaper implement nearly identical dataset and data module hierarchies:
- Both have `ETT`, `Weather`, `Electricity`, `UCR`, `UEA` datasets
- Both have `LightningDataModule` wrappers for each
- Both have strategy patterns for sequence handling
- Both have ARFF parsing, scaling, and feature extraction utilities

rbspaper's versions have evolved (cleaner conventions, `from __future__ import annotations`, kwonly params), but the duplication remains.

**Impact:** Bug fixes and feature additions must be applied to two code paths. Divergence risk increases over time.

**Revisit when:** Consolidating into a shared data layer.

## Testing Concerns

### autotsrc Has No Tests

**Severity:** Medium

The autotsrc sub-project has zero test files. Its dataset and data module classes — which contain nontrivial logic (bisect-based indexing, strategy dispatch, sliding window math) — are untested.

**Impact:** Regressions in autotsrc go undetected. Unsafe to refactor without test coverage.

### No CI Pipeline

**Severity:** Low

No GitHub Actions, GitLab CI, or pre-commit hooks are configured. Linting and testing must be run manually.

**Impact:** Code quality enforcement relies on developer discipline.

## Complexity Concerns

### Pipeline Core is Monolithic

**Severity:** Medium

`_sources/rbspaper/src/rbspaper/pipeline/core.py` (830 lines) handles training, attack execution, encoding, evaluation, analysis, and artifact persistence in a single module. Ruff explicitly ignores `C901` (complexity) and `PLR0912` (branch count) for this file.

**Impact:** Changes to one pipeline stage risk unintended effects on others. Testing requires constructing full config objects.

### Type Narrowing Workarounds

**Severity:** Low

Several `ty: ignore[invalid-argument-type]` comments in pipeline/core.py suppress type checker complaints about `ExperimentPipelineConfig.model` being `pl.LightningModule` while `encode_data()` expects a union of concrete model types. The code works at runtime but the type contract is hand-waved.

**Impact:** Type checker cannot verify correctness of model encoding paths.

## Dependency Concerns

### Conditional PyTorch Indexes Not Yet Activated

**Severity:** Low

rbspaper's `pyproject.toml` has commented-out uv index configuration for CPU/CUDA routing. Until activated, users on GPU clusters must manually manage torch installation.

### Heavy Optional Dependencies

**Severity:** Low

The `attacks` group pulls in `adversarial-robustness-toolbox`, `torchattacks`, `foolbox`, and optionally `cleverhans`. These have overlapping dependencies and known compatibility constraints (hence the `conflicts` declaration with `notebooks`).

# RBSPaper Scaffold

This repository contains the initial scaffold for RBSPaper (Robust Time Series Representation Learning).

## Environment management

This project uses uv and pyproject.toml only.

1. Install uv.
2. Sync dependencies.
3. Run checks.

```bash
uv sync --group dev --group attacks
uv run pytest
uv run rbspaper-run --experiments_output_folder ./outputs --experiment_id 0 --dataset_index 0
```

Optional groups:

```bash
uv sync --group forecasting_foundation --group search --group notebooks --group legacy_io --group performance
uv sync --group attacks_extended
uv sync --group legacy_weka
```

Compatibility notes for Python 3.13:

- Some research dependencies may lag Python 3.13 wheels depending on platform and release timing.
- Search group is pinned to ConfigSpace 1.2.1 and SMAC 2.3.1.
- attacks and notebooks are intentionally declared as conflicting optional groups because torchattacks pins old requests versions while jupyterlab requires newer requests.
- Other likely candidates are numba in performance and sktime in forecasting_foundation.
- If one of these fails in your environment, keep the failing group optional and continue with core plus attacks until we pin compatible versions.
- Weka support requires system Java (openjdk) in addition to the optional legacy_weka group.

## Current implementation status

- Typed experiment registry with centralized instance definitions.
- Thin runner that selects experiment instance plus dataset index.
- Staged pipeline scaffold with deterministic artifact output.
- Adapter interfaces for models, attacks, and tasks.
- Smoke tests for instance resolution and pipeline execution.

## Next implementation chunk

- Add real model adapters and datamodules.
- Add attack backend adapters for torchattacks and ART.
- Add downstream evaluation and representation robustness metrics.

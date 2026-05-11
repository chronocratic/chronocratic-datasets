# Technology Stack

Last mapped: 2026-05-08

## Language and Runtime

- **Python 3.12** — pinned to `>=3.12,<3.13` in both sub-projects
- Package management via **uv** with virtual environment at `.venv/`
- **setuptools** build system with `src` layout (`package-dir = {"" = "src"}`)

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.4.0, <=2.8.0 | Deep learning framework |
| `lightning` | ==2.5.5 | Training orchestration (LightningModule, LightningDataModule) |
| `torchvision` | >=0.19.0 | Vision utilities |
| `torchaudio` | >=2.4.0 | Audio utilities |
| `numpy` | >=2.1, <3.0.0 | Numerical arrays |
| `pandas` | >=2.2.0 | Tabular data processing |
| `scipy` | >=1.13.0 | Scientific computing |
| `scikit-learn` | >=1.5.0 | ML evaluation (SVM, classifiers) |
| `einops` | >=0.8.2 (rbspaper) | Tensor rearrangements |
| `tqdm` | >=4.66.0 | Progress bars |
| `joblib` | >=1.4.0 | Parallelization, caching |

## rbspaper-specific Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `adversarial-robustness-toolbox` | ~=1.20 | ART attack backend |
| `torchattacks` | ~=3.5 | TorchAttacks attack backend |
| `foolbox` | ~=3.3 | Foolbox attack backend |
| `matplotlib` | >=3.9.0 | Plotting |
| `seaborn` | >=0.13.0 | Statistical visualization |
| `tenacity` | >=9.1.4 | Retry decorators |
| `pywavelets` | >=1.8.0 | Wavelet transforms (augmentations) |
| `numba` | >=0.60.0 | JIT compilation |
| `wandb` | >=0.24.2 | Experiment tracking (optional) |
| `tensorboard` | >=2.18.0 | Experiment tracking (optional) |

## autotsrc-specific Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openpyxl` | ~=3.1.5 | Excel file reading |
| `h5py` | ~=3.16.0 | HDF5 file I/O |

## Development Tooling

| Tool | Version | Purpose |
|------|---------|---------|
| `ruff` | >=0.15.9 | Linting + formatting (ALL rules, Google docstrings) |
| `pytest` | >=8.2 | Test runner |
| `pytest-cov` | >=5.0 | Coverage reporting |
| `ty` | >=0.0.28 | Static type checker |
| `jupyterlab` | >=4.3 | Notebooks (optional) |
| `notebook` | >=7.3 | Classic Jupyter (optional) |

## Configuration

- **Root:** `pyproject.toml` defines tscollection-datasets package (PEP 420 namespace, Phase 1 complete with enums, __init__.py hierarchy, tests)
- **rbspaper:** `_sources/rbspaper/pyproject.toml` with extras: `cpu`, `cu118`, `cu121`, `cu124`, `attacks`, `attacks_extended`, `tracking`, `notebooks`
- **autotsrc:** `_sources/autotsrc/pyproject.toml` with extras: `notebooks`
- **Ruff:** Root `ruff.toml` mirrors rbspaper config. Per-file ignores in rbspaper cover `pipeline/core.py` (C901, PLR0912), `runners/py/` (T201, BLE001), adapter ANN401/TC001, and extensive test relaxations
- **uv:** Default groups include `dev` (root), `dev+attacks+tracking` (rbspaper). Group conflicts declared between `attacks` and `notebooks`

## PyTorch Device Routing

rbspaper has commented-out uv index configuration for CPU / CUDA 11.8 / 12.1 / 12.4 routing via `--extra` flags. Currently uses standard PyPI wheels (Mac / MPS scenario).

# Technology Stack

**Analysis Date:** 2026-05-05

## Languages

**Primary:**
- Python 3.12 — Entire codebase. Strictly pinned to `>=3.12,<3.13` in `pyproject.toml`.

**Secondary:**
- None. The project is Python-only, with Rust used indirectly via PyTorch, numba, and other compiled extensions.

## Runtime

**Environment:**
- CPython 3.12

**Package Manager:**
- `uv` for environment management, dependency resolution, lockfile management, and task execution
- Lockfile: `uv.lock` present (2552 lines)
- Build backend: `setuptools.build_meta` (setuptools >= 68, wheel)

**Install Command:**
```bash
uv sync                    # Core + dev dependencies (default groups)
uv sync --extra cu124      # Core + dev + CUDA 12.4 PyTorch wheels
uv sync --extra cpu        # Core + dev + CPU-only PyTorch
```

## Frameworks

**Deep Learning:**
- PyTorch (>=2.4.0, <=2.8.0) — Core tensor computation and neural network definitions
- torchvision (>=0.19.0) — Auxiliary vision utilities
- torchaudio (>=2.4.0) — Audio utilities
- PyTorch Lightning (==2.5.5) — Training orchestration, `LightningModule` for model definition, `LightningDataModule` for data pipelines, `Trainer` for training loop management

**Scientific Computing:**
- NumPy (>=2.1, <3.0) — Numerical arrays
- SciPy (>=1.13.0) — Scientific computing, ARFF file reading (`scipy.io.arff`)
- pandas (>=2.2.0) — Tabular data manipulation, DataFrame operations
- scikit-learn (>=1.5.0) — Evaluation protocols (SVC, Ridge, PCA, t-SNE, GridSearchCV), metrics

**Visualization:**
- Matplotlib (>=3.9.0) — Plotting
- seaborn (>=0.13.0) — Statistical visualization

**Performance:**
- Numba (>=0.60.0) — JIT compilation for numerical kernels
- pyWavelets (>=1.8.0) — Wavelet transforms
- einops (>=0.8.2) — Tensor rearrangement and reshaping
- joblib (>=1.4.0) — Parallel execution, caching
- tqdm (>=4.66.0) — Progress bars

**Testing:**
- pytest (>=8.2) — Test runner
- pytest-cov (>=5.0) — Coverage reporting

## Key Dependencies

**Critical (core pipeline):**
- `lightning==2.5.5` — Pinned exact version. All models inherit from `pl.LightningModule`; all dataloaders use `pl.LightningDataModule`
- `torch>=2.4.0,<=2.8.0` — Upper-bounded to prevent API breakage. Models are TS2Vec, AutoTCL, CoST
- `numpy>=2.1,<3.0.0` — Used throughout for array operations, representation storage (`.npz`)
- `pandas>=2.2.0` — Data loading (UCR/UEA classification, forecasting CSVs)
- `scikit-learn>=1.5.0` — Downstream evaluation (SVC/Ridge classification, PCA/t-SNE analysis)

**Attack Libraries (optional group `attacks`):**
- `adversarial-robustness-toolbox~=1.20` — ART backend for adversarial attacks (FGSM, BIM, PGD, etc.)
- `torchattacks~=3.5` — Torchattacks backend for adversarial attacks
- `foolbox~=3.3` — Additional attack implementations

**Extended Attacks (optional group `attacks_extended`):**
- `cleverhans~=4.0` — Extended attack library

**Notebooks (optional group `notebooks`):**
- `notebook>=7.3` — Jupyter Notebook
- `jupyterlab>=4.3` — JupyterLab

**Note:** The `attacks` and `notebooks` groups are declared as conflicting in `pyproject.toml` due to incompatible `requests` version pinning.

## Configuration

**Environment:**
- No `.env` files or environment variables. All configuration is explicit via Python dataclasses, CLI arguments, or registry definitions.
- PyTorch index configuration for CUDA variants is commented out in `pyproject.toml` (lines 82-120). Setup is handled via `uv sync --extra <variant>`.

**Build:**
- `pyproject.toml` — Single source of truth for dependencies, build config, and tool settings
- `ruff.toml` — Linting and formatting configuration
- `uv.lock` — Deterministic dependency lockfile

**Key Config Files:**
- `pyproject.toml` — Project metadata, dependencies, optional groups, tool config
- `ruff.toml` — Ruff linter rules (`select = ["ALL"]` with targeted ignores), Google docstring convention
- `uv_lock.sh` — Retry wrapper script for `uv lock`

## Platform Requirements

**Development:**
- Python 3.12
- `uv` package manager
- macOS (Apple Silicon / MPS), Linux (CPU or GPU)
- CUDA toolkit (optional, for GPU training: cu118, cu121, cu124 variants)

**Production / HPC:**
- Linux with NVIDIA GPUs (CUDA 11.8 or 12.4 recommended)
- Sufficient RAM for large time series datasets
- Local filesystem for dataset storage (UCR/UEA ARFF, CSV)
- No external network dependency at runtime

**GPU Support Strategy:**
```bash
# Apple Silicon (MPS):
uv sync

# HPC with CUDA 11.8:
uv sync --extra cu118

# HPC with CUDA 12.4:
uv sync --extra cu124

# CPU-only (CI):
uv sync --extra cpu
```

## Tooling

**Linting/Formatting:**
- Ruff (>=0.15.9) — All-in-one linter and formatter
  - `select = ["ALL"]` with targeted ignores
  - Google-style docstrings (`convention = "google"`)
  - Line length: 100
  - Target: Python 3.12 (`py312`)
  - Single quotes, space indentation
  - Excludes: `data`, `dependencies`, `experiments_output`

**Type Checking:**
- ty (>=0.0.28) — Type checker

**Test Runner:**
```bash
uv run pytest              # Run all tests
```

## Entry Points

**CLI:**
- `rbspaper-run` — Installed via `[project.scripts]`, entry point at `runners/py/runner.py:main`
- Usage: `uv run rbspaper-run --experiment_id ts2vec_fgsm --dataset_name Coffee --data_root /path/to/data`

**Direct Script:**
- `runners/py/runner.py` — Can be invoked directly: `uv run python runners/py/runner.py ...`

---

*Stack analysis: 2026-05-05*

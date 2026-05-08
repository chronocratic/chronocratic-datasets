# External Integrations

**Analysis Date:** 2026-05-05

## APIs & External Services

**None.** This is a self-contained research benchmark with no external API calls. All computation is local.

## Data Storage

**Databases:**
- None. No database connections.

**File Storage:**
- Local filesystem only. Datasets are read from local directories.
- Data formats: CSV (`.csv`), TXT (`.txt`), ARFF (`.arff`), JSON (`.json`)
- Output artifacts: `.npz` (NumPy archives for representations), `.json` (metrics, analysis results), `.ckpt` (Lightning checkpoints)

**Data Sources (local paths only):**

| Dataset Family | Format | Expected Path Pattern |
|---|---|---|
| UCR (univariate classification) | ARFF | `{data_root}/ucr_classification_univariate/{dataset_name}/` |
| UEA (multivariate classification) | ARFF/NPY | `{data_root}/uea_classification_multivariate/{dataset_name}/` |
| ETT (forecasting) | CSV | `{data_root}/ett/{dataset_name}.csv` |
| Electricity (forecasting) | TXT | `{data_root}/electricity/LD2011_2014.txt` |
| Weather (forecasting) | CSV | `{data_root}/weather/weather.csv` |

**Caching:**
- None. No in-memory or disk cache beyond NumPy `.npz` artifact persistence.

**Output Directory Structure:**
```
{output_dir}/{run_name}/
├── checkpoints/
│   └── best.ckpt
├── representations/
│   └── {task_name}/
│       ├── clean/
│       │   ├── train.npz
│       │   ├── valid.npz
│       │   └── test.npz
│       └── attacks/
│           └── {attack_name}/
│               ├── test.npz
│               └── attack_metadata.json
├── metrics/
│   └── {task_name}.json
├── analysis/
│   └── analysis.json
└── results_summary.json
```

## Authentication & Identity

**Auth Provider:**
- None. No authentication required.

## Adversarial Attack Backends

The project integrates with external adversarial attack libraries through an adapter/dispatch pattern. These are optional dependencies.

**Adversarial Robustness Toolbox (ART):**
- Package: `adversarial-robustness-toolbox~=1.20`
- Backend identifier: `AttackBackend.ART`
- Used for: FGSM, BIM, PGD, DeepFool, CW, LBFGS, UAP, SPSA, MI-FGSM, AutoAttack, HopSkipJump, Boundary, JSMA, SIMBA, EAD, OnePixel
- Wrapper: `src/rbspaper/attacks/_backend.py` (`run_art()`)
- Wraps the model in `PyTorchClassifier` (classification) or `PyTorchRegressor` (forecasting)
- Requires: `adversarial-robustness-toolbox` installed (optional group `attacks`)

**Torchattacks:**
- Package: `torchattacks~=3.5`
- Backend identifier: `AttackBackend.TORCHATTACKS`
- Used for: FGSM, BIM, PGD, DeepFool, CW, LBFGS, SPSA, MI-FGSM, AutoAttack, OnePixel, JSMA
- Wrapper: `src/rbspaper/attacks/_backend.py` (`run_torchattacks()`)
- Direct attack invocation (no wrapper estimator)
- Requires: `torchattacks` installed (optional group `attacks`)

**Foolbox:**
- Package: `foolbox~=3.3`
- Declared in optional group `attacks` but no integration code found in the current codebase
- Planned/unused integration

**CleverHans:**
- Package: `cleverhans~=4.0`
- Declared in optional group `attacks_extended` but no integration code found
- Planned/unused integration

**Attack Dispatch:**
- Entry: `src/rbspaper/attacks/functional.py` — High-level functions (`fgsm_attack`, `pgd_attack`, `bim_attack`)
- Registry: `src/rbspaper/attacks/registry.py` — Method-to-backend mapping, support validation
- Backend: `src/rbspaper/attacks/_backend.py` — ART/Torchattacks dispatch logic
- Batch: `src/rbspaper/attacks/batch.py` — Batch-level attack execution
- Config: `src/rbspaper/attacks/config.py` — Attack parameter dataclasses (FGSM, PGD, BIM, etc.)

## Downstream Evaluation Libraries

**scikit-learn:**
- Used for: SVC classification, Ridge regression, GridSearchCV, PCA, t-SNE, StandardScaler, SimpleImputer
- Classification protocols: `svm` (SVC with RBF kernel), `ridge` (Ridge regression)
- Implementation: `src/rbspaper/evaluation/protocols.py`
- Metrics: accuracy, precision, recall, F1, average_precision (classification); MSE, MAE, MAPE (forecasting)

## Monitoring & Observability

**Error Tracking:**
- None. No external error tracking service.

**Logs:**
- Python `logging` module only. Loggers defined per-module (e.g., `src/rbspaper/evaluation/evaluation.py`, `src/rbspaper/evaluation/protocols.py`).
- Lightning's built-in logging can be enabled via `trainer_kwargs` (`logger=True`).

**Experiment Tracking:**
- Lightning Trainer logging (optional). Default experiments use `logger=False`.
- W&B integration is possible but not configured by default.

## CI/CD & Deployment

**Hosting:**
- Not applicable. Research codebase, no deployment target.

**CI Pipeline:**
- None. No GitHub Actions workflows or CI configuration found.

**Git:**
- Repository uses Git (branch `main`)
- No CI hooks or automated testing

## Environment Configuration

**Required env vars:**
- None. All configuration is passed explicitly through arguments, dataclasses, or registries.

**Secrets location:**
- No secrets file. No `.env` file exists.

## Data Format Integrations

**ARFF Reader:**
- Library: `scipy.io.arff` (via `scipy>=1.13.0`)
- Used by: `src/rbspaper/data/utils/arff.py` (`read_arff_as_df`)
- Purpose: Read UCR/UEA classification dataset files into pandas DataFrames

**CSV Reader:**
- Library: `pandas.read_csv`
- Used by: Forecasting datasets (ETT, Weather, Electricity, etc.)
- File patterns: `{dataset_name}.csv`, `LD2011_2014.txt` (semicolon-delimited)

**NPZ Storage:**
- Library: `numpy.savez` / `numpy.load`
- Used by: `src/rbspaper/pipeline/core.py` (`_save_npz`)
- Purpose: Persist representation arrays (features + labels) per partition

**JSON Storage:**
- Library: `json.dump` / `json.load` (Python stdlib)
- Used by: `src/rbspaper/pipeline/core.py` (`_save_json`)
- Purpose: Persist metrics, analysis results, attack metadata, results summary

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Design Pattern Summary

The project uses several adapter patterns for external integration:

| Pattern | Location | Purpose |
|---|---|---|
| Attack Backend Dispatch | `src/rbspaper/attacks/_backend.py` | Route attack methods to ART or Torchattacks |
| Model Adapter Interface | `src/rbspaper/adapters/model_adapter.py` | Abstract model contract for pipeline + attacks |
| Attack Adapter Interface | `src/rbspaper/adapters/attack_adapter.py` | Backend-agnostic attack interface |
| Evaluation Protocol Factory | `src/rbspaper/evaluation/protocols.py` | Protocol selection (SVM/Ridge) |
| DataModule Factory | `src/rbspaper/data/preparation.py` | Dataset-family-specific LightningDataModule creation |
| Experiment Registry | `experiment_instances/instances.py` | Named experiment instance lookup |
| Dataset Registry | `src/rbspaper/data/registry.py` | Named dataset metadata lookup |

---

*Integration audit: 2026-05-05*

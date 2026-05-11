# External Integrations

Last mapped: 2026-05-08

## Adversarial Attack Frameworks (rbspaper)

rbspaper integrates three adversarial-robustness backends, selectable per attack:

| Backend | Package | Attacks Supported |
|---------|---------|-------------------|
| `ART` (Adversarial Robustness Toolbox) | `adversarial-robustness-toolbox` | FGSM, BIM, PGD, DeepFool, CW, LBFGS, MI-FGSM, SPSA, UAP, JSMA |
| `TorchAttacks` | `torchattacks` | FGSM, BIM, PGD, DeepFool, CW, LBFGS, MI-FGSM, AutoAttack |
| `Foolbox` | `foolbox` | SPSA, HopSkipJump, Boundary, SIMBA, One-Pixel, EAD |

Backend selection is registry-driven: `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK` maps `(task, attack_method)` to available backends. See `_sources/rbspaper/src/rbspaper/attacks/registry.py`.

## Experiment Tracking (rbspaper, optional)

- **W&B** (`wandb>=0.24.2`) — enabled via `--extra tracking`
- **TensorBoard** (`tensorboard>=2.18.0`) — enabled via `--extra tracking`

Loggers are factory-created in `_sources/rbspaper/src/rbspaper/pipeline/loggers.py` via `create_loggers()`. Both are optional and gated by tracking extras group.

## Data Storage Formats

| Format | Library | Context |
|--------|---------|---------|
| Excel (`.xlsx`) | `openpyxl` | autotsrc dataset loading |
| HDF5 (`.h5`) | `h5py` | autotsrc dataset loading |
| ARFF | Custom parser | UCR/UEA classification datasets in `_sources/rbspaper/src/rbspaper/data/utils/arff.py` |
| NPZ | numpy | Pipeline artifact persistence (representations, metrics) |
| JSON | stdlib | Config serialization, metrics output |

## File System Layout for Data

Data is expected at externally mounted paths. No data files are committed to the repository. The `_sources/` directory contains source code only; raw datasets are fetched or placed by the user at runtime.

rbspaper datasets (`_sources/rbspaper/src/rbspaper/data/datasets/`) expect:
- ETT data at a configurable root path
- Electricity data at a configurable root path
- Weather data at a configurable root path
- UCR/UEA archives at a configurable root path

## No External APIs

The codebase does not call external HTTP APIs. All integrations are local (libraries, file I/O, CLI).

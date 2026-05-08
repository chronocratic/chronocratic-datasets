# RBSPaper — Code Quality & Experiment Infrastructure

## What This Is

A research benchmark for robust time series representation learning. The system trains self-supervised models (TS2Vec, AutoTCL, CoST), applies adversarial attacks, and evaluates representation quality across downstream tasks (classification, forecasting). This project hardens the codebase for large-scale HPC experiments.

## Core Value

The experiment pipeline must be resilient, resumable, and correct — so large HPC runs across 128+ datasets complete reliably without manual intervention.

## Requirements

### Validated

- Typed experiment registry with centralized instance definitions
- Thin runner that selects experiment instance + dataset index
- Staged pipeline scaffold with deterministic artifact output
- Adapter interfaces for models, attacks, and tasks
- Smoke tests for instance resolution and pipeline execution

### Active

- [ ] Refactor `EncodingFunctionalityMixin` from string-based dispatch (`self.model_name == 'CoST'`) to strategy methods (`_get_encoder()`, `_get_eval_method()`)
- [ ] Fix critical bugs: circular import in pipeline/config, Ridge argmax→argmin, MAPE zero-target crash, `max_train_data_size` UnboundLocalError
- [ ] Fix import inconsistency (`src.rbspaper.*` vs. `rbspaper.*`)
- [ ] Pipeline resume capability — checkpoint completed steps (training, encoding, attacks, evaluation), restart from interruption point
- [ ] Structured output folders with experiment metadata (unique filenames, deterministic paths)
- [ ] HPC SLURM runners — array job submission, QoS retry, proper logging
- [ ] Local testing runners — quick validation before HPC submission
- [ ] General code cleanup — ruff/ty clean, remove dead code, align conventions

### Out of Scope

- Deployment / production serving — this is research code, not a product
- New model architectures — existing models (TS2Vec, AutoTCL, CoST) are in scope for cleanup, not new additions
- New attack types — existing attacks (FGSM, PGD, BIM, DeepFool, CW) are in scope for cleanup, not new additions
- GUI / web interface — CLI and bash runners only

## Context

- Python 3.12, PyTorch + Lightning, uv package manager
- Codebase mapped on 2026-05-05 (see `.planning/codebase/`)
- Reference implementation exists in `_sources/autotsaugment/` — runners and experiment infrastructure should be adapted from these patterns, improved
- Current pipeline at `src/rbspaper/pipeline/core.py` has partial resume support (`reuse_trained_checkpoint`) but no step-level checkpointing
- Attack backends: ART, Torchattacks
- Evaluation: scikit-learn (SVC, Ridge, PCA), classification + forecasting
- HPC target: SLURM cluster (Kathleen), array jobs for 128+ datasets

## Constraints

- **Tech stack**: Python 3.12 only, PyTorch ≤2.8.0, Lightning ==2.5.5
- **Package manager**: uv only (no conda/pip outside uv)
- **Code style**: ruff for lint/format, ty for type checking, Google docstrings
- **HPC**: SLURM batch jobs, file-system multiprocessing sharing strategy
- **Design**: Functional patterns, type hints, frozen dataclass configs, registry-driven resolution

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Strategy methods for mixin | String-based dispatch (`model_name == 'CoST'`) is fragile and violates OCP | — Pending |
| Step-level pipeline checkpointing | HPC runs can take hours; training+encoding+attacks+evaluation should be independently resumable | — Pending |
| Adapt autotsaugment runners (not copy) | Reference runners work but use conda, have hardcoded paths, and lack resume logic | — Pending |

---
*Last updated: 2026-05-05 after initialization*

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

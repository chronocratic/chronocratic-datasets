# Roadmap: RBSPaper Code Quality & Experiment Infrastructure

**Created:** 2026-05-05
**Granularity:** Standard
**Mode:** YOLO

## Phases

| # | Phase | Goal | Success Criteria |
|---|-------|------|------------------|
| 1 | Bug Fixes & Import Consistency | Fix all critical bugs and unify imports | ruff + ty clean, no crashes, correct Ridge selection |
| 2 | Mixin Refactor | Replace string dispatch with strategy methods | No model_name checks, polymorphic encoding |
| 3 | Pipeline Hardening | Step-level resume + structured output | Resume from any step, deterministic artifacts |
| 4 | Experiment Registry Restructure | One experiment per model, attack families, declarative compatibility | Train once, apply N attacks on M tasks |
| 5 | Local Test Runners | Local validation before HPC | Single experiment runs end-to-end locally |
| 6 | HPC Runners | SLURM array job submission | Submit array jobs for 128+ datasets |
| 7 | Experiment Tracking | Integrate a tracking tool (W&B, MLflow, etc.) for experiment metadata and results | All runs logged with hyperparameters, metrics, artifacts |
| 8 | Code Quality Audit | Remove ty ignore band-aids, widen type contracts, clean dead code | Only genuine ty limitations remain |
| 9 | Mixin Refactor v2 | Split EncodingFunctionalityMixin into model-specific strategies for CoST | CoST no longer uses pooling-based mixin; decomposition strategy separate |

---

## Phase 1: Bug Fixes & Import Consistency

**Goal:** Fix all critical bugs that cause crashes or incorrect results, and unify import paths across the codebase.

**Deliverables:**
- Fix circular import chain in `src/rbspaper/pipeline/` (blocks `test_pipeline_core.py`)
- Fix Ridge alpha selection: `np.argmax` -> `np.argmin` (selects worst model)
- Fix MAPE crash on zero-valued targets
- Fix `max_train_data_size` UnboundLocalError for forecasting tasks
- Fix import inconsistency: `src.rbspaper.*` -> `rbspaper.*` (or vice versa, pick one)
- ruff format + lint clean
- ty type check clean

**Plans:** 4 plans in 2 waves

**Plan list:**
- [ ] 01-01-PLAN.md -- Fix circular import in augmentation/ts2vec modules
- [ ] 01-02-PLAN.md -- Fix evaluation bugs (Ridge argmin, MAPE zero, UnboundLocalError)
- [ ] 01-03-PLAN.md -- Unify imports to src.rbspaper.* across data/ and adapters/
- [ ] 01-04-PLAN.md -- Achieve ruff + ty clean across codebase

**Success Criteria:**
1. `uv run ruff check .` passes with no errors
2. `uv run ty` passes with no errors
3. `uv run pytest` passes all existing tests
4. Ridge evaluation returns minimum-loss model (verifiable by test)
5. MAPE handles zero targets gracefully (verifiable by test)
6. All imports use consistent prefix

---

## Phase 2: Mixin Refactor

**Goal:** Replace `self.model_name == 'CoST'` string checks in `EncodingFunctionalityMixin` with polymorphic strategy methods.

**Deliverables:**
- Add abstract methods to mixin: `_get_encoder()`, `_get_eval_method()`, `_get_slice()`
- Each model (TS2Vec, AutoTCL, CoST) overrides these methods
- Remove `model_name` string comparisons from mixin
- Remove `model_name` attribute from model classes (no longer needed)
- Update `src/rbspaper/models/encoding.py` if it references model names
- Update any tests that depend on `model_name`

**Success Criteria:**
1. Mixin has zero string comparisons against model names
2. TS2Vec, AutoTCL, CoST each implement strategy methods
3. Encoding produces identical outputs (regression check)
4. ruff + ty clean on changed files
5. No `model_name` attribute in model classes

---

## Phase 3: Pipeline Hardening

**Goal:** Make `run_experiment_pipeline()` resumable from any step with deterministic output structure.

**Requirements:** REQ-03, REQ-04, REQ-05, REQ-06, REQ-07

**Plans:** 15 plans in 5 waves

**Deliverables:**
- Step-level checkpoint: track completed steps (trained, encoded, attacked, evaluated) in `.pipeline_state.json`
- Resume logic: on restart, read checkpoint and skip completed steps
- Structured output: `{experiment_id}/{hash8}/seed_{seed}/{dataset_name}/` with deterministic paths
- Experiment metadata: `experiment_config.json`, `results_summary.json`
- Config hash: 8-char SHA-256 for drift detection
- Runner improvements: `--dataset_index`, `--force`, structured logging
- Tenacity retry with exponential backoff on pipeline steps
- Test resume flow and output structure

**Plan list:**
- [x] 03-01-PLAN.md -- PipelineState dataclass + builder + serialization
- [x] 03-02-PLAN.md -- Atomic write helper + state save/load
- [x] 03-03-PLAN.md -- Config hash computation (8-char SHA-256)
- [x] 03-04-PLAN.md -- Add tenacity dependency
- [ ] ~~03-05-PLAN.md~~ -- Hierarchical run name + runner wiring [DONE]
- [x] ~~03-06-PLAN.md~~ -- experiment_config.json at pipeline start [DONE]
- [x] ~~03-07-PLAN.md~~ -- State module import to core.py [DONE]
- [x] 03-08-PLAN.md -- Resume gates in pipeline core
- [x] 03-09-PLAN.md -- Tenacity retry decorator
- [x] 03-10-PLAN.md -- Auto-resume + force parameter
- [x] 03-11-PLAN.md -- --dataset_index + --force CLI args
- [x] 03-12-PLAN.md -- Structured logging setup
- [x] 03-13-PLAN.md -- Unit tests (state, hash, atomic write, paths)
- [x] 03-14-PLAN.md -- Integration tests (resume flow)
- [x] 03-15-PLAN.md -- Output structure + CLI tests

**Success Criteria:**
1. Pipeline resumes from interrupted encoding step without retraining
2. Pipeline skips already-completed steps on restart
3. Output folder structure is deterministic and hash-verifiable
4. Experiment config + metadata saved at run start
5. Runner accepts experiment_id + dataset_index + seed (matching autotsaugment interface)

---

## Phase 4: Experiment Registry Restructure

**Goal:** One experiment ID per model, attack families (whitebox/blackbox), declarative attack-task compatibility. Train model once, apply all selected attacks on compatible tasks.

**Requirements:** Restructure experiment registry from `model_attack` IDs (e.g., `ts2vec_fgsm`, `ts2vec_pgd`) to `model` IDs (e.g., `ts2vec`) with attack families and compatibility matrix.

**Deliverables:**
- Rewrite `experiment_instances/instances.py`: one ID per model, attacks organized by family (whitebox, blackbox)
- Add attack-task compatibility declarations: which attacks work on classification (need labels) vs. forecasting (no labels)
- Pipeline auto-filters: skips attacks incompatible with downstream task
- Runner `--attack_family` flag: select whitebox, blackbox, or default to all
- Remove redundant experiment IDs: `ts2vec_fgsm`, `ts2vec_pgd`, `ts2vec_bim`, `ts2vec_multi`, `autotcl_fgsm`, `autotcl_pgd`, `autotcl_multi` -> replaced by `ts2vec`, `autotcl`
- Update all existing tests that reference old experiment IDs
- Update `runners/py/runner.py` to handle new registry format

**Plans:** 1 plan in 1 wave

**Plan list:**
- [x] 04-01-PLAN.md -- AttackFamily enum, registry restructure, runner flag, preflight, tests

**Success Criteria:**
1. `uv run rbspaper-run --experiment_id ts2vec --dataset_name Coffee` trains TS2Vec once, applies all attacks
2. `uv run rbspaper-run --experiment_id ts2vec --attack_family whitebox --dataset_name Coffee` trains once, applies only whitebox attacks
3. Forecasting tasks auto-skip label-dependent attacks without errors
4. All existing tests pass with new registry format
5. `--list_experiments` shows model-based IDs (ts2vec, autotcl, cost)
6. ruff + ty clean on changed files

---

## Phase 5: Local Test Runners

**Goal:** Bash scripts for local validation before HPC submission.

**Plans:** 3 plans in 3 waves

**Plan list:**
- [x] 05-01-PLAN.md -- Python infrastructure: runners/__init__.py + print() to logging conversion
- [x] 05-02-PLAN.md -- Config template + local_single.sh (single experiment runner)
- [x] 05-03-PLAN.md -- local_batch.sh (batch runner with dataset expansion)

**Deliverables:**
- `runners/bash/config.sh.example` -- Template config file (gitignored `config.sh`)
- `runners/bash/local_single.sh` -- Run single experiment on single dataset locally
- `runners/bash/local_batch.sh` -- Run single experiment on multiple datasets sequentially
- Fix `runners/__init__.py` for proper package resolution (resolves Phase 3 UAT gap)
- Convert `print()` calls in `runners/py/runner.py` to `setup_logging()` infrastructure
- Scripts use `uv run` (no conda), detect project root for PYTHONPATH
- Config file auto-created from template on first run

**Success Criteria:**
1. `local_single.sh ts2vec 0` runs experiment ts2vec on dataset 0 locally
2. `local_batch.sh ts2vec 0-20` runs experiment on datasets 0 through 20 sequentially
3. Script outputs progress and aggregate pass/fail report
4. Logs written to `outputs/` hierarchy (same as HPC)
5. Can be run from any working directory
6. `--fraction 0.25` samples ~25% of datasets for quick smoke tests

---

## Phase 6: HPC Runners

**Goal:** SLURM array job submission for large-scale experiments across 128+ datasets.

**Plans:** 2 plans in 2 waves

**Plan list:**
- [x] 06-01-PLAN.md -- hpc_config.sh.example + hpc_submit.sh (batch array) + hpc_submit_single.sh (single dataset)
- [x] 06-02-PLAN.md -- Task/modality wrappers: classification/{univariate,multivariate} + forecasting/{univariate,multivariate}

**Deliverables:**
- `runners/bash/hpc_config.sh.example` -- Template HPC config (gitignored `hpc_config.sh`)
- `runners/bash/hpc_submit.sh` -- Generate and submit SLURM array job scripts (batch engine)
- `runners/bash/hpc_submit_single.sh` -- Generate and submit SLURM single job scripts (per-dataset)
- `runners/bash/classification/univariate/run_on.sh` -- Single dataset HPC launcher (family: ucr)
- `runners/bash/classification/univariate/run_all.sh` -- Batch launcher (loops over ucr datasets)
- `runners/bash/classification/multivariate/run_on.sh` -- Single dataset HPC launcher (family: uea)
- `runners/bash/classification/multivariate/run_all.sh` -- Batch launcher (loops over uea datasets)
- `runners/bash/forecasting/univariate/run_on.sh` -- Single dataset HPC launcher (--forecasting_mode univariate)
- `runners/bash/forecasting/univariate/run_all.sh` -- Batch launcher (loops over all forecasting datasets)
- `runners/bash/forecasting/multivariate/run_on.sh` -- Single dataset HPC launcher (--forecasting_mode multivariate)
- `runners/bash/forecasting/multivariate/run_all.sh` -- Batch launcher (loops over all forecasting datasets)
- QoS retry loop with exponential backoff on sbatch submission
- Configurable: experiment_id, dataset family, cores, memory, partition, time limit, account
- Single source of truth: HPC_OUTPUT_ROOT in hpc_config.sh eliminates repeated path typing
- All scripts use uv run (no conda), detect project root for PYTHONPATH
- Job files retained in outputs/ for post-hoc debugging
- Bash 3.2 compatible throughout (no declare -A, no (( )) on unbound vars, no [[ =~ ]])

**Success Criteria:**
1. `hpc_submit.sh ts2vec --families ucr --dry_run` generates valid SLURM array script
2. `hpc_submit_single.sh ts2vec Coffee --family ucr --dry_run` generates valid SLURM single script
3. Generated scripts use `uv run` for Python execution (no conda)
4. SLURM_ARRAY_TASK_ID resolves to family-scoped dataset name (not global index)
5. QoS retry loop on submission limits (10s, 20s, 40s backoff)
6. `classification/univariate/run_on.sh ts2vec Coffee` submits via hpc_submit_single.sh
7. `classification/univariate/run_all.sh` submits all experiments on all ucr datasets
8. `forecasting/univariate/run_on.sh ts2vec ETTh1` passes --forecasting_mode univariate
9. Configurable partition, time, memory, cores via hpc_config.sh
10. Experiment ID validated against registry before job generation
11. Dataset name validated against registry before job generation
12. `--dry_run` flag enables local validation without sbatch

---

## Dependencies

- Phase 2 requires Phase 1 (fix import consistency first)
- Phase 3 requires Phase 2 (pipeline uses encoding mixin)
- Phase 4 requires Phase 3 (registry touches pipeline + runner)
- Phase 5 requires Phase 4 (runners use new registry format)
- Phase 6 requires Phase 5 (HPC runs same runner, wrapped in SLURM)
- Phase 7 requires Phase 6 (tracking hooks into runner pipeline)
- Phase 8 can run anytime (independent type contract cleanup)
- Phase 9 requires Phase 2 (splits the mixin Phase 2 introduced)

## Phase 7: Experiment Tracking

**Goal:** Research and integrate a tracking tool (W&B, MLflow, or similar) for experiment metadata, metrics, and artifact logging.

**Requirements:** D-01, D-02, D-03, D-04, D-05, D-06, D-07

**Plans:** 3 plans in 3 waves

**Plan list:**
- [x] 07-01-PLAN.md -- Add tracking deps, loggers field on config, create logger factory module
- [x] 07-02-PLAN.md -- Wire loggers into Trainer, add timing instrumentation, hook W&B config/results, add --tracking_mode CLI
- [x] 07-03-PLAN.md -- Unit tests for logger factory, flatten utility, W&B helpers, --tracking_mode arg

**Success Criteria:**
1. All experiment runs automatically logged with hyperparameters, metrics, and artifacts
2. Tracking tool integrates with existing runner pipeline
3. Runs queryable via UI or API for comparison and filtering
4. No breaking changes to local/HPC runner interfaces

---

## Phase 8: Code Quality Audit

**Goal:** Replace `ty: ignore` band-aids with proper type fixes, remove dead code, and tighten type contracts.

**Plans:** 9 plans in 3 waves

**Plan list:**
- [ ] 08-01-PLAN.md — Remove lazy imports, widen _ModelType, fix LightningLogger
- [ ] 08-02-PLAN.md — Fix CoST._get_slice return type
- [ ] 08-03-PLAN.md — Widen AugmentationMethod.augment signature
- [ ] 08-04-PLAN.md — Add PipelineStateDict TypedDict
- [ ] 08-05-PLAN.md — Align attack_kwargs type, fix batch.py dataloader
- [ ] 08-06-PLAN.md — Widen DataConfig.data_module, clean core.py ty ignores
- [ ] 08-07-PLAN.md — Add ModelParamsProtocol, fix runner.py ty errors
- [ ] 08-08-PLAN.md — PLR2004 constant, BLE001 narrowing, deferred ty ignores
- [ ] 08-09-PLAN.md — Remove adapters, clean ruff.toml, final verification

**Deliverables:**
- Widen `_ModelType` in `encoding.py` from `TS2Vec | AutoTCL | CoST` to `pl.LightningModule` (4 ignores removed)
- Widen `DataConfig.data_module` from `BaseTimeSeriesDataModule` to `pl.LightningDataModule` (5 ignores removed)
- Add type params to `DataLoader` construction sites (2 ignores removed)
- Align `attack_kwargs` dict value type between `functional.py` and `_backend.py` (1 ignore removed)
- Add `isinstance` assertions in tests for union-typed returns (3 ignores removed)
- Refactor ternary to `if/else` in `data/datasets/strategies.py` for isinstance narrowing (1 ignore removed)
- Remove dead `adapters/` package (unused since creation)
- Address `CropShiftAugmentation.augment` Liskov violation via `**kwargs` or overloaded abstract method
- Keep 3 genuine `ty` limitation ignores: `__getitem__` override, `hasattr` narrowing (x2)
- **Remove unnecessary `noqa` directives from source code** -- fix the underlying issue (extract constants, refactor datetime calls, simplify long functions) rather than suppressing. Only keep `noqa` when structural: `N812` (torch.nn.functional convention), `PLC0415` (lazy imports), `SLF001` (PyTorch internals), `S311` (research random). Scope to specific rules, never broad.
- **Clean `ruff.toml` per-file-ignores** -- source files should carry their own scoped `noqa` directives. Remove `C901`/`PLR0915`/`PLR0912` ignores for `core.py` and refactor to eliminate complexity. Test files keep broader per-file-ignores (lazy imports, private access, fixtures).

**Success Criteria:**
1. `uv run ty check .` passes with only 4 documented limitation ignores
2. `uv run pytest` passes all tests
3. Dead adapters code removed
4. No `ty: ignore` for fixable type mismatches
5. No `noqa` in source code except for documented structural reasons (torch convention, lazy import, PyTorch internals, research random)
6. `ruff.toml` per-file-ignores contain only test file entries and `TYPE_CHECKING` adapter entries

## Phase 9: Mixin Refactor v2

**Goal:** Split `EncodingFunctionalityMixin` into model-specific encoding strategies so CoST no longer inherits a pooling-based interface it doesn't use.

**Problem:** CoST returns a `(trend, seasonality)` tuple, uses `_evaluate_with_feature_concatenation` (ignores slicing and encoding_window params), and overrides `_get_slice()` to always return `None` — all dummy workarounds because the mixin was designed around TS2Vec/AutoTCL's pooling pattern.

**Deliverables:**
- Extract a shared `BaseEncodingMixin` with only the common `encode()` entry point and `DataLoader` logic
- `PoolingEncodingMixin` for TS2Vec/AutoTCL — retains `_evaluate_with_pooling`, meaningful `_get_slice`, `_get_encoder`
- `DecompositionEncodingMixin` for CoST — owns `_evaluate_with_feature_concatenation`, no slicing, returns concatenated trend/seasonality
- Each mixin implements the same strategy interface: `_get_encoder()`, `_get_eval_method()`
- Remove `_evaluate_with_feature_concatenation` from the shared mixin
- Update `encoding.py` strategies module if affected
- Regression tests: encoding outputs unchanged for all three models

**Success Criteria:**
1. CoST has zero meaningless overrides (no dummy `_get_slice`)
2. TS2Vec and AutoTCL encoding outputs are identical to pre-refactor (regression check)
3. CoST encoding outputs are identical to pre-refactor (regression check)
4. `EncodingFunctionalityMixin` no longer contains CoST-specific logic
5. ruff + ty clean on changed files
6. All existing tests pass

---

## Success Criteria (Project-Level)

1. All critical bugs fixed, ruff + ty clean
2. Mixin uses polymorphism (no string dispatch)
3. Pipeline resumes from any step
4. One experiment ID per model, train once -> attack N times
5. Local runner validates full experiment end-to-end
6. HPC runner submits array jobs for 128+ datasets
7. Experiment runs logged with hyperparameters, metrics, and artifacts
8. Only genuine ty limitations remain as documented ignores
9. Encoding mixins separated by model pattern — no CoST-specific logic in shared code

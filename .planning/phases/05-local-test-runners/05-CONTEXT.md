# Phase 5: Local Test Runners - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Create bash wrapper scripts that invoke the Python runner for quick local validation before HPC submission. The scripts should be convenient to use, provide clear feedback, and work from any directory.

**In scope:**
- `runners/bash/config.sh.example` — Template config file (gitignored `config.sh`)
- `runners/bash/local_single.sh` — Run single experiment on single dataset locally
- `runners/bash/local_batch.sh` — Run single experiment on multiple datasets sequentially
- Fix `runners/__init__.py` for proper package resolution
- Convert `print()` calls in `runners/py/runner.py` to use `setup_logging()` infrastructure
- Scripts use `uv run` (no conda), detect project root for PYTHONPATH
- Config file auto-created from template on first run

**Out of scope:**
- HPC SLURM scripts (Phase 5)
- New model architectures or attack types
- Production deployment
- Parallel execution as default (sequential is fine for local testing)
</domain>

<decisions>
## Implementation Decisions

### Entry Point & Resolution
- **A-01:** Hybrid approach — create `runners/__init__.py` so `rbspaper-run` entry point works, AND bash scripts detect project root for PYTHONPATH (resolves Phase 3 UAT gap)
- **A-11:** Config file strategy — commit `config.sh.example`, gitignore `config.sh`, auto-copy on first run with DATA_ROOT validation

### Script Interface
- **A-03:** Hybrid arguments — positional `exp_id` + `dataset` (name or index), named flags for overrides (`--seed`, `--force`, `--max_epochs`, `--data_root`, `--output_dir`)
- **A-08:** Batch dataset specs — `0-20` (range), `0,3,7` (list), or `all` (registry)
- **A-09:** Sequential execution — datasets run one-by-one. `--parallel` flag available but not default
- **A-10:** Dataset fraction — `--fraction 0.25` samples ~25% of datasets for quick smoke tests

### Output & Logging
- **A-07:** Same output structure as HPC — uses standard `outputs/{experiment_id}/{short_hash}/seed_{seed}/{dataset_name}/`. No separate `local_outputs/`.
- **A-05:** Minimal bash logger — `[HH:MM:SS]` timestamps to stdout + log file per run
- **A-06:** Convert `runner.py` `print()` calls to `logger.info()`/`logger.warning()`/`logger.error()` — part of this phase

### Validation
- **A-04:** Essential checks — data_root exists, exit code per run, aggregate pass/fail report for batch. No uv sync or experiment_id pre-validation overhead.

### Claude's Discretion
- Exact format of the aggregate batch report (table vs. list)
- How `--fraction` samples datasets (random seed, deterministic first-N, etc.)
- Log file naming convention for bash-level logs vs. runner-level `pipeline.log`

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pipeline Core & Runner
- `runners/py/runner.py` — Current CLI runner: `_parse_args()`, `_build_pipeline_config()`, `main()`, `_print_summary()`. Has ~15 print() calls to convert to logging. `setup_logging()` already configured.
- `src/rbspaper/pipeline/core.py` — `run_experiment_pipeline()` main orchestration. Entry point invoked by runner.
- `src/rbspaper/pipeline/config.py` — Config dataclasses including `build_hierarchical_run_name()`, `PipelineArtifactConfig.output_dir`.

### Experiment & Data Registries
- `experiment_instances/instances.py` — `EXPERIMENTS_REGISTRY` dict with 2 registered experiments (`ts2vec`, `autotcl`). Model-scoped IDs with `attack_families` grouping. `get_experiment_instance(experiment_id, attack_family)` resolves with alias support and family filtering. `list_experiment_ids()` for validation.
- `src/rbspaper/data/data_setup.py` — `get_all_datasets(form='list')` returns all dataset names for index-to-name resolution.

### Reference Implementations
- `_sources/autotsaugment/runners/hpc_uni_bash/runner.sh` — Reference HPC script: per-dataset job generation, project root detection, PYTHONPATH setup, QoS retry loop.
- `_sources/autotsaugment/runners_foundation_models/run_encoder_training_wrapper.sh` — Reference wrapper: positional args, config defaults, logging patterns.

### Conventions
- `.planning/codebase/CONVENTIONS.md` — Naming conventions, error handling patterns, import organization.
- `.planning/codebase/STRUCTURE.md` — Directory layout, entry points, where to add new code.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `setup_logging()` in `runner.py` — Already configures file + stream handlers. `print()` calls at lines 335-342 (_print_summary), 358, 366 (--list_experiments), 421, 429, 435, 442, 445 should use logger instead.
- `_print_summary()` in `runner.py:335` — Config summary printer. Convert to structured log output.
- `build_hierarchical_run_name()` in `config.py` — Computes deterministic output paths. Scripts don't need to replicate this.

### Established Patterns
- `uv run` for all execution (no conda)
- `PYTHONPATH` needed for `experiment_instances` module (not under `src/`)
- Project root detection via script path traversal (autotsaugment reference uses `dirname` chain)
- Keyword-only arguments for all Python function calls

### Integration Points
- `runners/__init__.py` — Missing. Needs to exist for `rbspaper-run` entry point to resolve `runners.py.runner` module.
- `runners/bash/` — New directory. Scripts will live here alongside `runners/py/`.
- `_resolve_dataset()` in runner.py — Validates dataset name/index. Scripts may call this or replicate simpler validation.

</code_context>

<specifics>
## Specific Ideas

- User referenced autotsaugment HPC runner pattern: scripts should generate per-dataset job files, not just loop inline
- `local_single.sh b1 0` should be the simplest invocation — experiment ID + dataset index
- User wants `--fraction` for quick smoke tests on partial datasets
- Config file should auto-create from template on first run, blocking if DATA_ROOT is unset
- No parallelism needed for local testing — sequential is sufficient

</specifics>

<deferred>
## Deferred Ideas

- Parallel batch execution with configurable `MAX_JOBS` — useful for HPC (Phase 5), overkill for local validation
- Project-level output registry tracking all runs — deferred from Phase 3
- `local_batch.sh` as a job generator that writes per-dataset scripts — user initially wanted autotsaugment-style, but settled on sequential loop for simplicity. May revisit for Phase 5 HPC.

### Folded Todos
- **Replace print() with logging in runner** (from `.planning/todos/pending/phase04_print_to_logging.md`) — `runners/py/runner.py` has ~15 `print()` calls that should use `logger.info()`/`logger.warning()`/`logger.error()` now that `setup_logging()` is in place. Scope: `runners/py/runner.py`, `src/rbspaper/pipeline/core.py`.

</deferred>

---

*Phase: 5-Local Test Runners*
*Context gathered: 2026-05-07*

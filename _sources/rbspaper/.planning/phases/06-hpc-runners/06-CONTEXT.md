# Phase 6: HPC Runners - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Create SLURM-based HPC submission scripts that wrap the existing Python runner, enabling large-scale experiments across 128+ datasets on the Kathleen cluster. The scripts should submit clean array jobs, handle cluster QoS limits gracefully, and produce debuggable artifacts.

**In scope:**
- `runners/bash/hpc_config.sh.example` — Template HPC config file (gitignored `hpc_config.sh`)
- `runners/bash/hpc_submit.sh` — Generate and submit SLURM array job script
- `runners/bash/{ucr,uea,forecast,...}/run_on_{family}.sh` — Per-category convenience launchers
- `runners/bash/{ucr,uea,forecast,...}/run_all_on_{family}.sh` — Per-category experiment-set runners
- QoS retry loop with exponential backoff on `sbatch` submission
- Configurable: experiment_id, dataset family, cores, memory, partition, time limit, account
- Single source of truth: `HPC_OUTPUT_ROOT` in `hpc_config.sh` eliminates repeated path typing
- All scripts use `uv run` (no conda), detect project root for PYTHONPATH
- Job files retained in `outputs/` for post-hoc debugging

**Out of scope:**
- Local runners (Phase 5 — already done)
- Pipeline checkpointing/resume logic (Phase 3 — already done)
- New model architectures or attack types
- Production deployment / monitoring
- Non-SLURM schedulers (PBS, LSF, etc.)
</domain>

<decisions>
## Implementation Decisions

### Script Architecture
- **D-01:** SLURM array job — single `sbatch` with `--array=0-N%N`. One task per dataset via `SLURM_ARRAY_TASK_ID`. Matches autotsaugment reference pattern. Simplest and most HPC-idiomatic.

### Dataset Resolution — Family-Scoped Indexes
- **D-06:** Family-scoped indexes (Option A) — `SLURM_ARRAY_TASK_ID` maps to a position within the specified dataset family (e.g., `0-2` for 3 UCR datasets), not a global position across all datasets. The submit script queries the registry at generation time (`get_datasets_names(family, form='list')`) to compute the correct array range. The generated SLURM script resolves `family + array_id` → dataset name via a small Python one-liner, then passes `--dataset_name` to the runner. This eliminates index drift: if datasets are added or removed, ranges auto-adjust.
- **Why:** AutoTSAugment uses a global index (`--dataset_index = SLURM_ARRAY_TASK_ID`), which means hardcoded bash ranges like `"0-127"` rot silently when the registry changes. Family-scoped indexes are self-healing.

### CLI Interface
- **D-02:** Config-driven + `--family` flag — HPC-specific settings live in a separate config file. Submit script: `hpc_submit.sh <experiment_id> --family <ucr|uea|forecast,...> [--seed N] [--force]`. Per-category convenience scripts bake `--family` into short wrappers. Per-category experiment-set scripts loop over multiple experiment IDs.

### HPC Configuration
- **D-03:** Separate `hpc_config.sh` — clean separation between local (`config.sh`) and HPC settings. `hpc_submit.sh` sources both `config.sh` (for DATA_ROOT) and `hpc_config.sh` (for cluster-specifics). Auto-copies `hpc_config.sh.example` on first run if missing.
- **D-07:** `HPC_OUTPUT_ROOT` as single source of truth — `hpc_config.sh` defines `HPC_OUTPUT_ROOT` (e.g., `/storage/work/skaf/rbs_experiments`). Used across all runners instead of repeating the cluster path. Local runners default `OUTPUT_DIR` to `./outputs`. No need to type the storage path on every invocation.

### Script Hierarchy — AutoTSAugment Pattern
- **D-08:** Three-tier hierarchy per dataset family, matching AutoTSAugment structure:
  - Tier 1: `hpc_submit.sh` — core engine (SLURM job generation + submission + QoS retry)
  - Tier 2: `{family}/run_on_{family}.sh` — per-category launcher (calls hpc_submit.sh with `--family` baked in, user passes experiment_id)
  - Tier 3: `{family}/run_all_on_{family}.sh` — experiment-set runner (loops over an EXPERIMENTS array, calls Tier 2 for each)

### Job File Management
- **D-04:** Keep job files in `outputs/` — HPC debugging is harder than local; retaining submitted scripts aids diagnostics. `--delete_job_files` flag available for users who prefer the reference pattern.
- **D-05:** QoS retry loop — 3 attempts with exponential backoff (10s → 20s → 40s) before aborting on submission failures. Note: AutoTSAugment uses a 30-minute flat sleep; we improve this with shorter exponential backoff.

### Post-Execution Refinements (2026-05-07)
- **D-09:** Central config `runner_config.sh` — all scripts source a single shared config derived from `$PROJECT_ROOT` (env var). No more relative `../..` chains. Provides: `RUNNERS_BASE`, `PYTHONPATH`, `log()`, `get_experiment_list()`, `ensure_config()`, and all defaults (seed, retry backoff, max attempts, submit pause).
- **D-10:** Removed `--family` from `hpc_submit_single.sh` and classification wrappers — Python resolves family from dataset name. Only `hpc_submit.sh` (batch array engine) still uses `--families` to resolve dataset lists via `get_datasets_names(family)`.
- **D-11:** Stripped bash-side validation (numeric seed, experiment/dataset registry pre-checks) — Python dataclasses enforce these. Bash only validates environment-specific paths (DATA_ROOT, HPC_OUTPUT_ROOT).
- **D-12:** Experiment IDs resolved from `get_experiment_list()` at runtime instead of hardcoded `ts2vec`, `autotcl`.
- **D-13:** Magic numbers (seed=42, retry_backoff=10, max_attempts=3, submit_pause=0.5) moved to `runner_config.sh` with overridable env vars.

### Claude's Discretion
- Exact layout of the `outputs/` job file directory
- QoS retry backoff timing constants (10s → 20s → 40s is the baseline)
- Exact EXPERIMENTS array content per family (planner derives from registry)
- Whether `hpc_submit.sh` supports multi-seed loops (reference wraps seeds in bash `for`) or delegates to separate submissions
- Aggregate submission report format (job IDs, task ranges, estimated completion)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Reference HPC Runner (AutoTSAugment)
- `_sources/autotsaugment/runners/hpc_uni_bash/runner.sh` — Reference SLURM array job: per-seed job generation, project root detection, PYTHONPATH setup, `--array=0-$((total_datasets-1))%${max_concurrent_jobs}`, `sbatch` submission loop. Uses conda (replace with `uv run`). Note: uses global dataset index — we improve with family-scoped indexes.
- `_sources/autotsaugment/runners/bash/runner.sh` — Main bash runner: positional arg dispatch, QoS retry loop (30min flat sleep), mode-based naming. Note: hardcodes `extra_storage_dir="/storage/work/skaf"` — we fix with HPC_OUTPUT_ROOT.
- `_sources/autotsaugment/runners/bash/ucr/run_experiment_ucr.sh` — Tier 2 launcher: calls main runner.sh with family-specific params (range `"0-127"`, mode `"ucr"`). Pattern we adapt for per-category scripts.
- `_sources/autotsaugment/runners/bash/ucr/run_experiment_all_ucr.sh` — Tier 3 experiment-set: loops over EXPERIMENTS array, calls Tier 2. Pattern we adapt for run_all_on_{family}.sh scripts.

### AutoTSAugment Data Setup (Index Resolution)
- `_sources/autotsaugment/src/experiments/data_setup.py` — `get_all_datasets(form='list')` concatenates categories in fixed order: UCR → ETT → Other → UEA. Runner resolves `dataset_name = get_all_datasets(form='list')[args.dataset_index]`. Note: global index means hardcoded bash ranges rot when registry changes.

### Dataset Registry (RBSPaper)
- `src/rbspaper/data/data_setup.py` — `get_datasets_names(family, form='list')` returns dataset names per family. `get_classification_datasets()`, `get_forecasting_datasets()` for category filtering. These functions enable family-scoped index resolution.
- `src/rbspaper/data/registry.py` — `DATASET_REGISTRY` tuple of `DatasetMetadata(name, family, tasks)`. `list_dataset_names()` returns names in registration order. Families: `ucr`, `uea`, `ett`, `electricity`, `weather`, `exchange`, `traffic`, `illness`. `CLASSIFICATION_FAMILIES = frozenset({'ucr', 'uea'})`.

### Existing Local Runners (Phase 5)
- `runners/bash/local_single.sh` — Project root detection, config.sh sourcing, argument parsing, `uv run` dispatch pattern. Reuse log() helper, validation pattern, config management.
- `runners/bash/local_batch.sh` — Dataset spec expansion, fraction sampling, aggregate report. Pattern for batch loops.
- `runners/bash/config.sh.example` — Shared config template with DATA_ROOT. HPC scripts source this for data path.

### HPC Config Template (to be created)
- `runners/bash/hpc_config.sh.example` — New file. Defines: `HPC_OUTPUT_ROOT` (single source of truth for storage path), `HPC_PARTITION`, `HPC_ACCOUNT`, `HPC_TIME`, `HPC_MEM_PER_CPU`, `HPC_CPUS_PER_TASK`, `HPC_MAX_CONCURRENT`, `DELETE_JOB_FILES`.

### Python Runner
- `runners/py/runner.py` — Core execution entry point. Accepts `--dataset_name` or `--dataset_index` (mutually exclusive). SLURM script uses `--dataset_name` via family-scoped resolution.

### Pipeline Core
- `src/rbspaper/pipeline/core.py` — `run_experiment_pipeline()` with `reuse_trained_checkpoint` and `force` flag support.

### Experiment Registry
- `experiment_instances/instances.py` — `list_experiment_ids()`, `get_experiment_instance()`. Used for experiment ID validation.

### Phase Context
- `.planning/phases/05-local-test-runners/05-CONTEXT.md` — Phase 5 decisions (config strategy, CLI pattern, output structure, logging) that Phase 6 builds upon.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `log()` helper in `local_single.sh` — `[HH:MM:SS]` timestamped logger. Use as-is.
- Project root detection block (lines 20-28 in `local_single.sh`) — `BASH_SOURCE` traversal with `pyproject.toml` guard. Use as-is.
- `CONFIG_FILE` auto-copy pattern in `local_single.sh` — copy `.example` → live file on first run. Use for both config.sh and hpc_config.sh.
- AutoTSAugment tier 2/3 pattern — `run_on_{family}.sh` + `run_all_on_{family}.sh`. Adapt structure, rewrite content.

### Established Patterns
- `uv run python runners/py/runner.py --experiment_id X --dataset_name Y` — execution command (prefer --dataset_name for stability)
- `PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"` — module resolution
- `set -uo pipefail` — strict bash mode (all existing scripts use it)
- Keyword-only args for all CLI dispatch (`--flag value` pairs)
- Config files gitignored; `.example` templates committed

### Integration Points
- `hpc_submit.sh` generates SLURM job script → sources config.sh (DATA_ROOT) + hpc_config.sh (HPC_OUTPUT_ROOT, cluster params)
- `hpc_runner.sh` uses family-scoped `SLURM_ARRAY_TASK_ID` to resolve dataset name → dispatches `--dataset_name` to `runners/py/runner.py`
- `get_datasets_names(family, form='list')` in `data_setup.py` — resolves family-scoped index to dataset name at job time
- Output path: `${HPC_OUTPUT_ROOT}/{experiment_id}/{short_hash}/seed_{seed}/` on HPC
- Per-category tier 2 scripts: `{family}/run_on_{family}.sh` — call `hpc_submit.sh --family {family}` with experiment_id
- Per-category tier 3 scripts: `{family}/run_all_on_{family}.sh` — loop over EXPERIMENTS array, call tier 2

### Creative Options
- Multi-seed support: reference runner loops over seeds, generating one array job per seed. `hpc_submit.sh` could accept `--num_runs N` or default to 1.
- Job file naming: `hpc_exp_{experiment_id}_seed_{seed}.sh` mirrors reference naming convention.
- `HPC_OUTPUT_ROOT` replaces all hardcoded `/storage/work/skaf/` occurrences in autoTSAugment runners. Single variable, single source of truth.

### Anti-patterns from AutoTSAugment (Do NOT Copy)
- Global dataset index (`--dataset_index = SLURM_ARRAY_TASK_ID`) → ranges rot when registry changes. Use family-scoped instead.
- Hardcoded `extra_storage_dir="/storage/work/skaf"` repeated in multiple files → use `HPC_OUTPUT_ROOT` in config.
- Hardcoded `miniconda_path` in runner → use `uv run`, no conda needed.
- 30-minute flat QoS retry sleep → use shorter exponential backoff.
- Per-category index ranges baked into bash scripts (`"0-127"`, `"134-163"`) → compute dynamically from registry.

</code_context>

<specifics>
## Specific Ideas

- User referenced autotsaugment HPC runner pattern: SLURM array jobs, per-seed submission, three-tier script hierarchy per dataset category
- `hpc_submit.sh ts2vec --family ucr` should be the main invocation — experiment ID + dataset family
- `{family}/run_on_{family}.sh ts2vec` for quick per-category runs — experiment ID only
- `{family}/run_all_on_{family}.sh` to run all experiments on a dataset category
- `HPC_OUTPUT_ROOT` in `hpc_config.sh` is the single source of truth for storage path — eliminates repeating `/storage/work/skaf/` across scripts
- Kathleen cluster is the target SLURM environment
- `uv` replaces conda: `source` conda → direct `uv run`
- User prefers separate hpc_config.sh for cluster settings (not merged into config.sh)
- Job files should be kept for debugging (deviates from reference which deletes them)
- Family-scoped indexes: SLURM array ID maps to position within family, not global registry. Ranges computed dynamically from `get_datasets_names(family)`. Prevents index drift when datasets are added/removed.

</specifics>

<deferred>
## Deferred Ideas

- Non-SLURM scheduler support (PBS, LSF) — future phase if needed
- Parallel batch execution with configurable MAX_JOBS at the bash level — SLURM arrays handle concurrency natively
- Monitoring/dashboard for job status — deferred to separate observability phase
- Project-level output registry tracking all runs — deferred from Phase 3

---

*Phase: 6-HPC Runners*
*Context gathered: 2026-05-07*

</deferred>

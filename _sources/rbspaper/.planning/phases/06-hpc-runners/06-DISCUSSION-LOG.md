# Phase 6: HPC Runners - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-07
**Phase:** 6-hpc-runners
**Mode:** --all --analyze (auto-selected all gray areas, trade-off tables presented)
**Areas discussed:** Script Architecture, CLI Interface, SLURM Configuration, Job File Management, Output Path, Dataset Indexing, Invocation Pattern, Script Hierarchy

---

## HPC Script Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| SLURM array job | Single sbatch with --array=0-N. Matches autotsaugment reference. Simplest and most efficient. | ✓ |
| Per-dataset job files | Generate N separate scripts, submit individually. More control, more complexity. | |
| Hybrid (array + resume) | Array job for submission, but each task has independent checkpointing for selective retries. | |

**User's choice:** SLURM array job (Recommended)
**Notes:** Direct match with reference `runner.sh` pattern (`--array=0-$((total_datasets-1))%${max_concurrent_jobs}`). User accepted recommendation without deviation.

---

## CLI Interface

| Option | Description | Selected |
|--------|-------------|----------|
| Config-driven + overrides | Cluster settings in config file. CLI: hpc_submit.sh <exp_id> <dataset_spec> [flags]. Matches Phase 5 pattern. | |
| Named flags only | All params on command line. Explicit, self-documenting, verbose. | |
| Positional args | hpc_submit.sh <exp_id> <dataset_spec> <cores> <memory>. Minimal but rigid. | |
| Separate hpc_config.sh | (User free-text variant) Clean separation between local and HPC settings. | ✓ |

**User's choice:** Separate hpc_config.sh for cluster settings, config.sh for shared DATA_ROOT.
**Notes:** User deviated from recommendation (which was to extend config.sh with HPC_* prefixed vars). Prefers clean separation: local and HPC configs in different files. `hpc_submit.sh` will source both.

---

## Job File Lifecycle & QoS Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Keep job files + QoS retry | Store job files in outputs/ for debugging. Retry sbatch 3x with backoff on QoS limits. | ✓ |
| Delete after submit + QoS retry | Follow reference pattern: delete job files after sbatch. Add QoS retry loop. | |
| Keep job files + no QoS retry | Store job files for debugging. On QoS failure, abort and let user resubmit manually. | |

**User's choice:** Keep job files + QoS retry (Recommended)
**Notes:** Deviates from reference runner (which deletes job files). Rationale: HPC debugging is harder than local, having submitted scripts available aids post-hoc diagnostics.

---

## Output Path — Single Source of Truth

| Option | Description | Selected |
|--------|-------------|----------|
| HPC_OUTPUT_ROOT in hpc_config.sh | One variable stores `/storage/work/skaf/rbs_experiments`. All runners source it. Eliminates repeated typing. | ✓ |
| Shared path.sh file | Single file for all runners (local + HPC). More indirection. | |
| Environment variable | No config file. Set in ~/.zshrc. Less discoverable. | |

**User's choice:** HPC_OUTPUT_ROOT in hpc_config.sh.
**Notes:** User pointed out that AutoTSAugment repeats `/storage/work/skaf/` in many files (runner.sh, tier 2 scripts, etc.). Wanted a single source of truth to avoid this. `hpc_config.sh` stores the path once; all scripts source it.

---

## Dataset Indexing — Family-Scoped vs. Global

AutoTSAugment uses a global index: `SLURM_ARRAY_TASK_ID` maps to position in `get_all_datasets()` (UCR→ETT→Other→UEA). Bash scripts hardcode ranges like `"0-127"` for UCR. These ranges rot when the registry changes.

6 options were presented and compared:

| Option | Description | Selected |
|--------|-------------|----------|
| A: Family-scoped index | SLURM array ID maps to position within family. Submit script computes range dynamically from `get_datasets_names(family)`. Most robust. | ✓ |
| B: Name lookup file | Submit script writes names to text file. SLURM reads line N via `sed`. Simple but extra file. | |
| C: Global index with validation | Same as AutoTSAugment but validates at submit time. Still fragile over time. | |
| D: SLURM name-based array | `--array=Coffee,ECG200,FaceFour%32`. Eliminates indexes entirely. SLURM version-dependent. | |
| E: JSON mapping file | Submit writes index→name JSON. Bash parses with jq. Extra dependency. | |
| F: Per-dataset job files | N individual sbatch calls. No index needed. QoS burden, loses array convenience. | |

**User's choice:** Option A (family-scoped index).
**Notes:** User emphasized that SLURM arrays need integer task IDs (`SLURM_ARRAY_TASK_ID = 0, 1, 2...`) but these shouldn't be global dataset indexes. Family-scoped resolution via `get_datasets_names(family, form='list')[SLURM_ARRAY_TASK_ID]` prevents index drift. The submit script computes the correct range at generation time. Runner uses `--dataset_name` (stable), not `--dataset_index` (fragile).

---

## Script Hierarchy — AutoTSAugment Pattern

User pointed out the AutoTSAugment runner structure and wanted similar capabilities for running sets of experiments.

AutoTSAugment has a 3-tier hierarchy per dataset category:
- Tier 1: `runner.sh` — core engine (generates + submits SLURM job)
- Tier 2: `{category}/run_experiment_{category}.sh` — calls runner.sh with category defaults baked in
- Tier 3: `{category}/run_experiment_all_{category}.sh` — loops over EXPERIMENTS array, calls Tier 2

**Decision:** Adapt the same hierarchy, improved:
- Tier 1: `hpc_submit.sh` — core engine (same role as autotsaugment runner.sh)
- Tier 2: `{family}/run_on_{family}.sh` — per-category launcher (bakes `--family` flag)
- Tier 3: `{family}/run_all_on_{family}.sh` — experiment-set runner (loops over experiment IDs)

---

## Invocation — How to Specify Dataset Sets

| Option | Description | Selected |
|--------|-------------|----------|
| --family flag only | `hpc_submit.sh ts2vec --family ucr`. Simple, one category at a time. | |
| Per-category scripts only | `bash runners/bash/ucr/run_on_ucr.sh ts2vec`. Familiar, easy tab-complete. | |
| Both (Recommended) | Submit script accepts --family directly. Per-category convenience scripts also exist. Most flexible. | ✓ |

**User's choice:** Both — `hpc_submit.sh` takes `--family` directly, per-category scripts exist for convenience.

---

## Claude's Discretion

- Exact layout of the `outputs/` job file directory
- QoS retry backoff timing constants (10s → 20s → 40s suggested)
- Multi-seed loop support (reference wraps seeds in bash `for`) vs. separate submissions
- Aggregate submission report format (job IDs, task ranges, estimated completion)

## Deferred Ideas

- Non-SLURM scheduler support (PBS, LSF) — future phase if needed
- Monitoring/dashboard for job status — deferred to separate observability phase
- Project-level output registry tracking all runs — deferred from Phase 3

---

## Post-Execution Refinements (2026-05-07)

**Triggered by:** User review of completed phase 06 before verification step.

### Central Config (`runner_config.sh`)
All 14 runner scripts were using `SCRIPT_DIR + ../..` chains (up to 4 levels deep for wrappers) to find `PROJECT_ROOT`. Now all scripts source a single `runner_config.sh` that derives `RUNNERS_BASE` from `$PROJECT_ROOT` (env var). Provides shared `log()`, `ensure_config()`, `get_experiment_list()`, and all magic number defaults.

### Removed `--family` from Single-Submit Path
`hpc_submit_single.sh` required `--family` but never used it (parsed, validated, logged, but no effect on execution). The Python runner resolves everything from the dataset name. Classification wrappers no longer pass this dead arg.

### Stripped Bash-Pre-validation
Bash was re-validating experiment IDs, dataset names, and numeric seeds — all already enforced by Python dataclasses. Now only environment-specific checks remain (DATA_ROOT non-empty, HPC_OUTPUT_ROOT set).

### Experiment List from Registry
`run_all.sh` scripts had hardcoded `EXPERIMENTS=(ts2vec autotcl)`. Now resolved at runtime via `get_experiment_list()` from the Python registry.

**Net result:** -565 lines removed, +182 added across 14 scripts + 1 new config.

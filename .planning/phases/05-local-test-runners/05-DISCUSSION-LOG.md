# Phase 5: Local Test Runners - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-07
**Phase:** 4-Local Test Runners
**Areas discussed:** Entry Point, Data Root, Script Interface, Validation, Logging, Output Structure, Batch Handling, Error Recovery, Config Strategy

---

## Entry Point & PYTHONPATH

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid: fix runners/__init__.py + bash project root detection | One-time Python fix + robust bash detection | ✓ |
| Bash-only wrapper | Scripts set PYTHONPATH themselves | |
| Fix at source only | Just fix runners/__init__.py | |

**User's choice:** Hybrid (Recommended)
**Notes:** Resolves the Phase 3 UAT gap (blocked test #2). Scripts should detect project root reliably.

---

## Data Root Resolution

| Option | Description | Selected |
|--------|-------------|----------|
| Config file (runners/bash/config.sh) | Per-machine defaults, sourced by scripts | ✓ |
| $RBS_DATA_ROOT env var | Set in ~/.zshrc, optional --data_root override | |
| Always explicit arg | --data_root required every invocation | |

**User's choice:** Config file
**Notes:** Explicit and per-machine, easy for collaborators with different paths.

---

## Script Interface

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid: positional + named flags | Positional exp_id + dataset, named for overrides | ✓ |
| Minimal positional | Just exp_id + dataset, no overrides | |
| Full named flags | All args as --experiment_id, --dataset_index, etc. | |

**User's choice:** Hybrid (Recommended)
**Notes:** Ergonomic for common case, extensible for edge cases.

---

## Validation & Error Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Essential + batch report | Data_root check, exit codes, aggregate summary | ✓ |
| Full preflight | All checks including uv sync, experiment validation | |
| Minimal | Just exit codes | |

**User's choice:** Essential + batch report
**Notes:** Pragmatic. No overhead from uv sync or experiment validation subprocess calls.

---

## Logging

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal bash logger | [HH:MM:SS] timestamps to stdout + log file | ✓ |
| Let runner log everything | Bash silent, only runner produces output | |
| Echo + tee | Plain echo, piped through tee | |

**User's choice:** Minimal bash logger (Recommended)
**Notes:** Consistent format, captures bash-level events alongside runner logs.

---

## Runner print() to Logging

| Option | Description | Selected |
|--------|-------------|----------|
| Include in Phase 4 | Fix runner.py print() calls now | ✓ |
| Defer to Phase 6 | Code quality audit covers cleanup | |
| Skip entirely | print() works fine | |

**User's choice:** Include in Phase 4
**Notes:** Clean logging is a prerequisite for good local validation UX. From existing todo: phase04_print_to_logging.md.

---

## Output Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Same as HPC structure | Standard outputs/{experiment_id}/{short_hash}/... hierarchy | ✓ |
| Runner defaults + output_dir override | Scripts pass --output_dir local_outputs | |
| Simplified local structure | local_outputs/{experiment_id}/{dataset}/ | |

**User's choice:** Same as HPC structure
**Notes:** Local runs are pure convenience wrappers. No separate local_outputs needed. Keeps one output convention.

---

## Batch Dataset Range

| Option | Description | Selected |
|--------|-------------|----------|
| Range + list + all | 0-20, 0,3,7,15, or all | ✓ |
| Range only | Just 0-20 notation | |
| File-based | Read from text file | |

**User's choice:** Range + list + all (Recommended)
**Notes:** Most flexible for different use cases.

---

## Batch Execution Model

| Option | Description | Selected |
|--------|-------------|----------|
| Sequential + dataset fraction | One-by-one, with --fraction for sampling | ✓ |
| Job generator | Per-dataset scripts, concurrency limit | |
| Background processes | xargs -P, no job scripts | |

**User's choice:** Sequential by default, `--parallel` flag. Plus `--fraction 0.25` for sampling.
**Notes:** User corrected initial recommendation. No parallelism needed for local testing. Wants fraction limiter for quick smoke tests. User was initially ok with job-generator pattern (matching autotsaugment HPC runner) but decided sequential loop is sufficient for local.

---

## Config File Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Template + auto-create | config.sh.example committed, config.sh gitignored, auto-copied | ✓ |
| Template only | User manually copies | |
| Hardcoded defaults | No config file | |

**User's choice:** Template + auto-create (Recommended)
**Notes:** Best UX. Template shows available settings. Auto-create prevents first-run friction. Blocks if DATA_ROOT unset.

---

## Claude's Discretion

- Exact format of aggregate batch report (table vs. list)
- How `--fraction` samples datasets (random seed, deterministic first-N, etc.)
- Log file naming convention for bash-level logs vs. runner-level `pipeline.log`

---

## Deferred Ideas

- Parallel batch execution with configurable `MAX_JOBS` — useful for HPC (Phase 5), overkill for local validation
- Project-level output registry tracking all runs — deferred from Phase 3
- Job generator pattern for batches — user initially referenced autotsaugment style but settled on sequential for local

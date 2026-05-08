# Phase 3: Pipeline Hardening - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Make `run_experiment_pipeline()` resumable from any step with deterministic output structure, so large HPC runs across 128+ datasets complete reliably without manual intervention.

**In scope:**
- Step-level checkpoint with per-task granularity
- Automatic resume logic with `--force` override
- Structured output hierarchy: `{experiment_id}/{short_hash}/seed_{seed}/{dataset_name}/`
- Experiment config at start, results summary at end
- Runner improvements: `--dataset_index`, structured logging, log files in output dir
- Atomic state writes with retry with backoff

**Out of scope:**
- New model architectures or attack types
- HPC SLURM scripts (Phase 5)
- Local bash runners (Phase 4)
- Production deployment

</domain>

<decisions>
## Implementation Decisions

### Checkpoint Design
- **D-01:** Per-task granularity — state tracks completion per (step, task_name) pair. E.g., `{"encoding": ["classification"], "attacks": ["classification"]}`. More precise recovery with minimal complexity cost.
- **D-02:** State file at `run_dir/.pipeline_state.json` — self-contained, hidden file, travels with the run. Same output folder holds weights, logs, and state (for scratch/work on HPC).
- **D-03:** Atomic state writes — write to `.pipeline_state.json.tmp` then `os.rename()` for POSIX atomicity. Prevents corrupted state on crash.
- **D-04:** Retry with backoff — on step failure, retry N times with exponential backoff before marking as failed. Handles transient GPU memory fragmentation.

### Output Structure
- **D-05:** Hierarchical output: `output_dir/{experiment_id}/{short_hash}/seed_{seed}/{dataset_name}/` — grouped by experiment, then seed, then dataset. Deterministic and human-scannable.
- **D-06:** Short hash appendix — 8-char SHA-256 of serialized model params catches parameter drift while staying readable.
- **D-07:** `experiment_config.json` written at run start (model params, attack params, seed, dataset, trainer kwargs). `results_summary.json` written at end with metrics. Crash-safe config.

### Runner Interface
- **D-08:** Add `--dataset_index` for HPC array jobs, keep `--dataset_name` for local testing. Mutually exclusive.
- **D-09:** Structured logging with Python `logging` module + INFO-level step transitions. tqdm for inner loops. Log files saved in output dir, not runner dir.
- **D-10:** Automatic resume when `.pipeline_state.json` exists. `--force` flag to override and start fresh. HPC-friendly default.

### Claude's Discretion
- Exact structure of `.pipeline_state.json` (timestamps, config hash for integrity)
- Number of retries and backoff interval (suggest: 3 retries, 30s base)
- Whether to log to both stdout and file, or file only with stdout summary

### Deferred Ideas
- Project-level state registry across runs (deferred — overkill for now)
- JSON-structured logging (deferred — overkill for research code)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pipeline Core
- `src/rbspaper/pipeline/core.py` — Main orchestration: `run_experiment_pipeline()`, step functions, `_persist_*` helpers. Resume hook at `_train_model()` line 198 (`reuse_trained_checkpoint`).
- `src/rbspaper/pipeline/config.py` — Frozen dataclass configs: `ExperimentPipelineConfig`, `PipelineArtifactConfig`, `TrainingConfig`, `PipelineArtifactConfig.run_dir` property.

### Runner & Experiments
- `runners/py/runner.py` — Current CLI runner. Arg parsing, config assembly, `_print_summary()`. Entry point for all execution.
- `experiment_instances/instances.py` — Experiment registry: `ExperimentInstance`, `EXPERIMENTS_REGISTRY`, `get_experiment_instance()`.

### Reference Implementation
- `_sources/autotsaugment/runners/py/runner.py` — Reference runner with hash-based naming (`_generate_unique_filename`, line 64), `_convert_slurm_time_to_trainer_dict` (line 158), dataset index lookup (line 260).
- `_sources/autotsaugment/runners/bash/runner.sh` — Reference HPC script: SLURM array jobs, QoS retry loop, job file generation.

### Model Encoding
- `src/rbspaper/models/encoding.py` — `encode_data()` entry point used by pipeline for representation extraction.
- `src/rbspaper/models/abstract/encoding_functionality_mixin.py` — Mixin with `encode()` method, polymorphic strategy methods.

### Data Setup
- `src/rbspaper/data/data_setup.py` — `get_datamodule_with_downstream_tasks()`, dataset registry used for index-to-name mapping.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PipelineArtifactConfig.run_dir` property (config.py:224-226) — already computes `output_dir / run_name`. New hierarchy extends this.
- `_save_json()` helper (core.py:826-829) — JSON serialization with `_json_default()` for numpy/Path/Enum. Reuse for state file and config.
- `ExperimentPipelineResults` (config.py:292-303) — existing results container. Extend with state tracking fields if needed.
- tqdm already used in `encode()` and `_compute_sliding_representations()` — compatible with structured logging for outer steps.

### Established Patterns
- Frozen dataclasses for all config — new state/config objects should follow this pattern
- `src.rbspaper.*` import prefix — established in Phase 1
- Keyword arguments for all function calls
- Type hints on all functions including return types
- Google style docstrings

### Integration Points
- `run_experiment_pipeline()` (core.py:69) — main entry point. Resume logic wraps this function, checking state before each step.
- `_train_model()` (core.py:191) — existing `reuse_trained_checkpoint` check. Replace with state-driven resume.
- `_prepare_run_directory()` (core.py:184) — good place to write `experiment_config.json` and initialize state file.
- `main()` in `runners/py/runner.py` (line 214) — add `--dataset_index`, `--force`, logging setup.

</code_context>

<specifics>
## Specific Ideas

- User emphasized: all outputs (logs, weights, state) should live in the output folder, not the runner directory. This is for HPC scratch/work storage.
- Reference runners use conda; new runners use `uv run` only.
- User does not like unnecessary complexity — keep state file minimal, prefer readable folder names over hashes but use short hash for safety.

</specifics>

<deferred>
## Deferred Ideas

- Project-level state registry (`output_dir/.pipeline_registry.json`) — useful for cross-run queries but not needed now
- Full SHA-256 hash folder names — overkill when experiment registry already guarantees ID uniqueness
- JSON-structured logging — machine-parseable but noisy for research code
- Query budget and surrogate model support — already in config but not actively used

</deferred>

---

*Phase: 3-Pipeline Hardening*
*Context gathered: 2026-05-05*

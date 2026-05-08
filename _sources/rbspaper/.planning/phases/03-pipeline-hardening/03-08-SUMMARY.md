---
plan: 03-08
phase: 03-pipeline-hardening
status: complete
self_check: passed
tasks_completed: 3/3
key_files_created:
  - src/rbspaper/pipeline/core.py
  - test/test_pipeline_core.py
---

## Resume Gates

Inserted checkpoint-driven resume gates into `run_experiment_pipeline`. Each pipeline step is now guarded by an `is_step_complete` check against the `PipelineState`. When a step is skipped, its output is loaded from persisted disk artifacts (NPZ/JSON) via dedicated `_load_*` helpers -- never from placeholder values.

### What was built

1. **State parameter and resume detection** -- `run_experiment_pipeline` now accepts `previous_state` and `config_hash` parameters. The function initializes a `_PipelineStateBuilder` from either the previous state (resume mode) or a fresh builder (new run). A module-level `logging` logger was added for step transitions.

2. **Disk-load helpers** -- Four inverse functions for the persist helpers:
   - `_load_clean_representations` -- reads train/valid/test NPZ files
   - `_load_attacked_representations` -- reads per-attack test NPZ + metadata JSON
   - `_load_metrics_from_disk` -- reads task metrics JSON
   - `_load_analysis_from_disk` -- reads analysis JSON
   - `_load_shared_attacked_inputs_from_disk` -- reconstructs shared attacked inputs from persisted reps
   - `_resolve_existing_checkpoint` -- finds existing checkpoint file on disk

3. **Resume gates** -- Every step (train, shared_attacks, encoding, attacks, evaluate, analysis) is wrapped in `if not state.is_step_complete(...)` gates. The state is saved after each step completion via `save_pipeline_state`.

### Notable deviations

- The executor agent implemented Tasks 1 and 2 (state params + disk-load helpers) but missed Task 3 (wiring the gates into the pipeline). The orchestrator completed the wiring in a fix pass.
- `test_pipeline_skips_encoding_and_loads_from_disk` had a fragile assertion counting `encode_data` calls, which are also made by the attacks step. Fixed by marking attacks as complete in the test state.
- `run_experiment_pipeline` now exceeds the PLR0915 statement limit (98 > 50); a `noqa: PLR0915` was added as the complexity is required for the resume logic.

### Verification

- `uv run ruff check src/rbspaper/pipeline/core.py` -- clean
- `uv run ty check src/rbspaper/pipeline/core.py` -- only pre-existing warning
- 68 tests pass, including 3 new resume gate tests

---
status: partial
phase: 03-pipeline-hardening
source: [03-01-SUMMARY.md, 03-02-SUMMARY.md, 03-03-SUMMARY.md, 03-04-SUMMARY.md, 03-05-SUMMARY.md, 03-06-SUMMARY.md, 03-07-SUMMARY.md, 03-08-SUMMARY.md, 03-09-SUMMARY.md, 03-10-SUMMARY.md, 03-11-SUMMARY.md, 03-12-SUMMARY.md, 03-13-SUMMARY.md, 03-14-SUMMARY.md, 03-15-SUMMARY.md]
started: "2026-05-06T17:00:00Z"
updated: "2026-05-06T17:15:00Z"
---

## Current Test

[testing paused — 11 items outstanding]

## Tests

### 1. Cold Start Smoke Test
expected: Run `uv run pytest test/test_pipeline_core.py test/test_pipeline_state.py -v` — all 60 tests pass, no import errors, ruff and ty clean
result: pass

### 2. Hierarchical Output Directory Structure
expected: Runner produces output paths in format `outputs/{experiment_id}/{short_hash}/seed_{seed}/{dataset_name}/` — verified by dry run
result: blocked
blocked_by: other
reason: "runner requires PYTHONPATH to find experiment_instances module; rbspaper-run entry point also broken (missing runners/__init__.py)"

### 3. experiment_config.json Written
expected: After a pipeline run, `experiment_config.json` exists in the output directory with keys: model_name, seed, downstream_tasks, attack_names, trainer_kwargs, attack_scope, encoding_batch_size
result: [pending]

### 4. .pipeline_state.json Written
expected: After a pipeline run, `.pipeline_state.json` exists in the output directory with valid JSON containing step completion state
result: [pending]

### 5. Pipeline Resume Skips Completed Steps
expected: When `.pipeline_state.json` exists from a prior run and `--force` is NOT set, the pipeline skips steps marked complete in the state file and loads outputs from disk artifacts (NPZ/JSON files) instead of re-running
result: [pending]

### 6. --force Flag Resets Checkpoint
expected: Running with `--force` ignores any existing `.pipeline_state.json` and executes all steps from scratch, writing a fresh state file
result: [pending]

### 7. --dataset_index Resolves Correctly
expected: `--dataset_index 0` resolves to the first registered dataset name. Running `uv run python runners/py/runner.py --experiment_id ts2vec_fgsm --dataset_index 0 --data_root /nonexistent --dry_run` should print the resolved dataset name in the config summary
result: [pending]

### 8. Mutual Exclusivity: --dataset_name and --dataset_index
expected: Providing both `--dataset_name` and `--dataset_index` causes the runner to exit with an error message, not proceed
result: [pending]

### 9. Out-of-Range --dataset_index Rejected
expected: `--dataset_index 9999` (beyond registered dataset count) causes the runner to exit with a descriptive error
result: [pending]

### 10. Logging Creates pipeline.log File
expected: After a runner invocation, `run_dir/pipeline.log` exists and contains timestamped log entries with module names in format `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
result: [pending]

### 11. Retry Decorator Recovers from Transient Errors
expected: The `retry_step` decorator in core.py retries RuntimeError/MemoryError up to 3 times with exponential backoff. After 3 failures, it raises RetryError. Non-retryable exceptions (ValueError) pass through immediately
result: [pending]

### 12. Config Hash Determinism
expected: `compute_config_hash` produces identical 8-character hex strings for identical input dicts regardless of key ordering, and different hashes for different params or seeds
result: [pending]

## Summary

total: 12
passed: 1
issues: 0
pending: 10
skipped: 0
blocked: 1

## Gaps

[none yet]

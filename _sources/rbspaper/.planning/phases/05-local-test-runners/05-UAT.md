---
status: testing
phase: 05-local-test-runners
source: 05-01-SUMMARY.md, 05-02-SUMMARY.md, 05-03-SUMMARY.md
started: 2026-05-07T15:30:00Z
updated: 2026-05-07T15:30:00Z
---

## Current Test

number: 1
name: rbspaper-run CLI Entry Point
expected: |
  Running `uv run rbspaper-run --help` displays the runner help text with available arguments
  (--experiment, --dataset, --data_root, --list_experiments, etc.) and exits with code 0.
awaiting: user response

## Tests

### 1. rbspaper-run CLI Entry Point
expected: `uv run rbspaper-run --help` shows help text and exits 0
result: issue
reported: "ModuleNotFoundError: No module named 'runners'. The editable install .pth only adds 'src/' to sys.path, but 'runners/' lives at the project root."
severity: blocker

### 2. List Experiments
expected: `uv run rbspaper-run --list_experiments` prints registered experiments and exits 0
result: issue
reported: "Same ModuleNotFoundError as test 1 — entry point cannot resolve the runners package."
severity: blocker

### 3. runner.py Has No print() Calls
expected: All output from runner.py goes through logging (zero print() calls in AST)
result: pass

### 4. runner.py Module Logger and basicConfig
expected: Module-level logger = logging.getLogger(__name__) after imports. basicConfig() on --list_experiments, KeyboardInterrupt, and Exception paths.
result: pass

### 5. runners/__init__.py Package Marker
expected: File exists with docstring, enables `import runners.py.runner`
result: pass

### 6. config.sh.example Valid Bash Template
expected: File passes `bash -n` syntax check, contains DATA_ROOT placeholder, and has no merge conflict markers
result: pass

### 7. .gitignore Excludes config.sh
expected: `runners/bash/config.sh` appears in .gitignore to prevent committing local paths
result: pass

### 8. local_single.sh Runs Help
expected: `./runners/bash/local_single.sh --help` prints usage and exits 0
result: pending

### 9. local_single.sh Auto-Creates config.sh
expected: First run without config.sh present copies config.sh.example to config.sh and prompts to set DATA_ROOT
result: pending

### 10. local_single.sh Forwards to runner.py
expected: With valid config.sh (DATA_ROOT set), script constructs CMD array and runs `uv run python runners/py/runner.py <experiment> <dataset>`
result: blocked
blocked_by: server
reason: "Requires valid DATA_ROOT with actual dataset files to exercise the full pipeline. Tests 3-7 and 8 are pre-requisites."

### 11. local_batch.sh Runs Help
expected: `./runners/bash/local_batch.sh --help` prints usage and exits 0
result: pending

### 12. local_batch.sh Dataset Spec Expansion
expected: Running with dataset spec '0-2' expands to datasets [0, 1, 2]. Spec '1,5,10' expands to [1, 5, 10]. Spec 'all' queries registered datasets.
result: pending

### 13. local_batch.sh Fraction Sampling
expected: `--fraction 0.5` reduces dataset list to first half. `--fraction 0.0` or missing flag uses full list.
result: pending

### 14. local_batch.sh Aggregate Summary
expected: After running all datasets, prints BATCH RUN SUMMARY table showing pass/fail per dataset. Exits 1 if any run failed.
result: blocked
blocked_by: server
reason: "Requires valid DATA_ROOT with actual dataset files to exercise the full pipeline."

### 15. Test Suite Unchanged
expected: `uv run pytest` passes all 111 tests
result: pass

## Summary

total: 15
passed: 5
issues: 2
pending: 6
blocked: 2
skipped: 0

## Gaps

- truth: "rbspaper-run CLI entry point resolves and runs without ImportError"
  status: failed
  reason: "ModuleNotFoundError: No module named 'runners'. The editable install .pth file only adds 'src/' to sys.path, but 'runners/' is at the project root. Entry point script cannot find the module."
  severity: blocker
  test: 1
  artifacts: []
  missing: []

- truth: "rbspaper-run --list_experiments lists registered experiments"
  status: failed
  reason: "Same root cause as test 1 — entry point cannot import runners package."
  severity: blocker
  test: 2
  artifacts: []
  missing: []

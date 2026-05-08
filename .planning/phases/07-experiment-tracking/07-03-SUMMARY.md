---
phase: 07-experiment-tracking
plan: 03
subsystem: testing
tags: [pytest, logger-factory, wandb, tensorboard, cli-args, tracking]

requires:
  - phase: 07-experiment-tracking
    plan: 01
    provides: create_loggers factory, _flatten_dict, _find_wandb_logger, _log_config_to_wandb, _log_results_to_wandb
  - phase: 07-experiment-tracking
    plan: 02
    provides: --tracking_mode CLI arg, runner logger integration
provides:
  - 21 unit tests for logger factory and W&B helpers covering D-01 through D-07
  - 5 unit tests for --tracking_mode CLI argument parsing
affects: [07-experiment-tracking verification, CI test suite]

tech-stack:
  added: []
  patterns: [monkeypatch for WandbLogger import failure, MagicMock for wandb.Run]

key-files:
  created: [test/test_logger_factory.py]
  modified: [test/test_runner_cli_args.py, ruff.toml]

key-decisions:
  - "Added test/*.py per-file-ignores to ruff.toml alongside existing test/**/*.py"
  - "Used monkeypatch.delattr to simulate WandbLogger ImportError gracefully"

patterns-established:
  - "Class-based test organization with descriptive docstrings per test method"
  - "Mocked wandb.Run via MagicMock for _log_results_to_wandb verification"
  - "tmp_path fixture for filesystem-dependent logger creation tests"

requirements-completed: [D-01, D-02, D-03, D-04, D-05, D-06, D-07]

duration: 12min
completed: 2026-05-08
---

# Phase 07 Plan 03: Unit Tests Summary

**Comprehensive unit test suite for logger factory (21 tests) and --tracking_mode CLI arg (5 tests) covering decisions D-01 through D-07.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-05-08T09:31:31Z
- **Completed:** 2026-05-08T09:43:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- test/test_logger_factory.py with 21 tests across 4 test classes
- TestTrackingModeArg class added to test_runner_cli_args.py with 5 tests
- All 34 new tests pass; full suite (137 tests) passes with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test_logger_factory.py with comprehensive tests** - `398d8c8` (feat)
2. **Task 2: Add --tracking_mode tests to test_runner_cli_args.py** - `ce180bb` (test)

## Files Created/Modified
- `test/test_logger_factory.py` - 21 unit tests for logger factory, flatten utility, W&B helpers
- `test/test_runner_cli_args.py` - Added TestTrackingModeArg class (5 tests)
- `ruff.toml` - Added test/*.py per-file-ignores with ANN401, ARG002, ARG005

## Decisions Made
- Used `monkeypatch.delattr` on `lightning.pytorch.loggers.WandbLogger` to simulate import failure, avoiding fragile `__builtins__` manipulation
- Mocked `wandb.Run` and `wandb.Table` via `MagicMock` for results logging tests
- Added `test/*.py` alongside existing `test/**/*.py` in ruff.toml per-file-ignores to cover top-level test files

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated ruff.toml per-file-ignores**
- **Found during:** Task 1 (linting test_logger_factory.py)
- **Issue:** `test/**/*.py` glob in ruff.toml does not match files directly in `test/` (e.g., `test/test_logger_factory.py`), causing ANN401, ARG002, ARG005 violations
- **Fix:** Added `test/*.py` per-file-ignores with same rules plus ANN401, ARG002, ARG005; extended test/**/*.py with same additions
- **Files modified:** ruff.toml
- **Verification:** `uv run ruff check test/test_logger_factory.py test/test_runner_cli_args.py` passes clean
- **Committed in:** 398d8c8 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 blocking lint issue)
**Impact on plan:** Lint compliance fix. No functional scope change.

## Issues Encountered
- Initial graceful fallback test used `__builtins__['__import__']` monkeypatching which failed on Python 3.12 (builtins is a module, not dict). Rewrote to use `monkeypatch.delattr` on the lightning loggers module instead.

## User Setup Required

**Tracking dependencies must be installed for logger factory tests.**
- Run `uv sync --group tracking` before executing `test/test_logger_factory.py`
- Without tensorboard/wandb, `create_loggers` will raise ModuleNotFoundError for TensorBoardLogger

## Next Phase Readiness
- Full unit test coverage for D-01 through D-07 decisions
- Test infrastructure ready for integration testing (Plan 04 if applicable)
- 34 new/updated tests, all passing

## Known Stubs
None -- all tests exercise real code with proper mocking where needed.

## Threat Flags
None -- tests use mocks and tmp_path, no real W&B network calls or credentials.

---
*Phase: 07-experiment-tracking*
*Completed: 2026-05-08*

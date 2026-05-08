---
phase: 04-experiment-registry-restructure
plan: 01
subsystem: experiment registry
tags: [attack_family, registry, cli, preflight, warnings, dataclass]

requires:
  - phase: 03-pipeline-hardening
    provides: experiment instance registry, runner pipeline, attack config system
provides:
  - AttackFamily enum (WHITE_BOX, BLACK_BOX)
  - group_methods_by_family registry helper
  - Model-scoped experiment registry (ts2vec, autotcl) with attack_families grouping
  - Backward-compatible alias map (7 old IDs -> 2 new) with UserWarning
  - --attack_family CLI flag on runner
  - Warn-and-drop preflight for incompatible attacks
affects: [phase 5-local-test-runners, phase 6-hpc-runners]

tech-stack:
  added: []
  patterns:
    - StrEnum-based family grouping parallel to AttackThreatModel
    - Frozen dataclass with derived property (attack_families + attack_params property)
    - Alias map with deprecation warnings for backward compatibility
    - Warn-and-drop preflight (logging instead of raising for incompatibilities)

key-files:
  created:
    - test/test_attack_family.py
    - test/test_experiment_registry.py
    - test/test_preflight_compat.py
  modified:
    - src/rbspaper/attacks/enums.py
    - src/rbspaper/attacks/registry.py
    - experiment_instances/instances.py
    - runners/py/runner.py
    - src/rbspaper/pipeline/core.py
    - src/rbspaper/pipeline/config.py
    - test/test_runner_cli_args.py
    - test/test_hierarchical_run_name.py
    - test/test_pipeline_state.py

key-decisions:
  - "AttackFamily values match AttackThreatModel naming (white_box, black_box) for consistency"
  - "Alias map emits UserWarning with specific old->new ID for deprecation awareness"
  - "get_experiment_instance returns deepcopy when filtered by family (threat mitigation)"
  - "Preflight warn-and-drop is informational only; actual filtering at _select_attacks_for_task level"

requirements-completed: [D-01, D-02, D-03, D-04]

duration: 7min
completed: 2026-05-07
---

# Phase 4 Plan 01: Experiment Registry Restructure Summary

**Model-scoped experiment registry with AttackFamily grouping, --attack_family CLI flag, backward-compatible alias map, and warn-and-drop preflight**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-07T10:06:32Z
- **Completed:** 2026-05-07T10:14:03Z
- **Tasks:** 10/10 completed
- **Files modified:** 11 source/test files

## Accomplishments

- Added `AttackFamily` StrEnum and `group_methods_by_family()` registry helper
- Restructured `ExperimentInstance` from flat `attack_params` to family-grouped `attack_families` dict with derived property
- Replaced 7 model_attack registry entries with 2 model-scoped entries (ts2vec, autotcl)
- Implemented backward-compatible alias map with `UserWarning` deprecation notices
- Added `--attack_family` CLI flag with family-aware `--list_experiments` output
- Enhanced preflight with warn-and-drop for incompatible attacks (logging instead of raising)
- Full test coverage: 19 new tests + all 111 existing tests pass

## Task Commits

1. **Task 1: Add AttackFamily enum** - `c60d99e` (feat)
2. **Task 2: Add group_methods_by_family helper** - `4e49a1e` (feat)
3. **Task 3: Tests for AttackFamily and grouping** - `d0805da` (feat)
4. **Task 4: Restructure ExperimentInstance with attack_families** - `c561023` (feat)
5. **Task 5: Tests for restructured registry** - `f04be3e` (feat)
6. **Task 6: Add --attack_family CLI flag** - `5babdaa` (feat)
7. **Task 7: Enhance preflight with warn-and-drop** - `83f6e02` (feat)
8. **Task 8: Preflight compatibility tests** - `0d91d21` (feat)
9. **Task 9: Update existing tests with new IDs** - `f6d71db` (feat)
10. **Task 10: Final lint + format + test pass** - `ad687b6` (style)

## Files Created/Modified

- `src/rbspaper/attacks/enums.py` - Added `AttackFamily(StrEnum)` with WHITE_BOX/BLACK_BOX
- `src/rbspaper/attacks/registry.py` - Added `group_methods_by_family()` helper
- `experiment_instances/instances.py` - Restructured registry: 2 model-scoped entries, alias map, family filter
- `runners/py/runner.py` - Added `--attack_family` flag, enhanced `--list_experiments`
- `src/rbspaper/pipeline/core.py` - Warn-and-drop preflight for incompatible attacks
- `src/rbspaper/pipeline/config.py` - Updated docstring example
- `test/test_attack_family.py` - 5 tests for enum and grouping
- `test/test_experiment_registry.py` - 12 tests for registry structure, aliases, filtering
- `test/test_preflight_compat.py` - 2 tests for warn-and-drop behavior
- `test/test_runner_cli_args.py` - Updated experiment IDs
- `test/test_hierarchical_run_name.py` - Updated IDs and assertions
- `test/test_pipeline_state.py` - Updated IDs and assertions

## Decisions Made

- **AttackFamily StrEnum values:** Used `white_box`/`black_box` (snake_case lowercase) matching `AttackThreatModel` convention for consistency across the codebase.
- **Alias resolution:** Emitted `UserWarning` (not `DeprecationWarning`) with explicit old->new ID mapping. This ensures users see the warning by default.
- **Deepcopy on filter:** `get_experiment_instance` with `attack_family` returns a deepcopied instance to prevent mutation of the shared registry (threat mitigation from plan).
- **Preflight scope:** Warn-and-drop is informational (function returns `None`). Actual attack filtering already happens at `_select_attacks_for_task()`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed SPSA attack parameter names in preflight test**
- **Found during:** Task 8 (preflight compatibility tests)
- **Issue:** `SpsaAttackParameters` uses `nb_iter`, not `steps` as initially coded
- **Fix:** Corrected to `nb_iter=10` and used `FgsmAttackParameters` for the supported attack test
- **Files modified:** test/test_preflight_compat.py
- **Verification:** Tests pass after fix
- **Committed in:** `0d91d21` (Task 8 commit)

**2. [Rule 1 - Bug] Fixed downstream task mismatch in preflight test**
- **Found during:** Task 8 (preflight compatibility tests)
- **Issue:** Unsupported attack used `FORECASTING` task but downstream_tasks only had `classification`
- **Fix:** Added `forecasting` downstream task for the unsupported attack test config
- **Files modified:** test/test_preflight_compat.py
- **Committed in:** `0d91d21` (Task 8 commit)

**3. [Rule 2 - Missing Critical] Updated docstring examples and help text**
- **Found during:** Task 9 (existing test ID updates)
- **Issue:** Runner usage example and help text still referenced `ts2vec_fgsm`
- **Fix:** Updated runner.py docstring, help text, and pipeline/config.py docstring example
- **Files modified:** runners/py/runner.py, src/rbspaper/pipeline/config.py
- **Committed in:** `f6d71db` (Task 9 commit)

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 missing critical)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep.

## Issues Encountered

- None beyond the auto-fixed items documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Registry structure is stable for the local test runners (Phase 5) to consume
- AttackFamily enum available for HPC job scoping
- Backward compatibility ensures existing scripts still work (with warnings)

---
*Phase: 04-experiment-registry-restructure*
*Completed: 2026-05-07*

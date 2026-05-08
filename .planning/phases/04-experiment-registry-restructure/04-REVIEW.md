---
phase: 04-experiment-registry-restructure
reviewed: 2026-05-07T12:30:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - src/rbspaper/attacks/enums.py
  - src/rbspaper/attacks/registry.py
  - experiment_instances/instances.py
  - runners/py/runner.py
  - src/rbspaper/pipeline/core.py
  - src/rbspaper/pipeline/config.py
  - test/test_attack_family.py
  - test/test_experiment_registry.py
  - test/test_preflight_compat.py
  - test/test_runner_cli_args.py
  - test/test_hierarchical_run_name.py
  - test/test_pipeline_state.py
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 04: Code Review Report

**Reviewed:** 2026-05-07T12:30:00Z
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found

## Summary

This phase restructures the experiment registry from flat `model_attack` IDs to model-scoped IDs with `AttackFamily` grouping. It adds a backward-compatible alias map with deprecation warnings, a `--attack_family` CLI flag, and a warn-and-drop preflight mechanism for incompatible attacks.

The core registry restructuring (AttackFamily enum, `group_methods_by_family`, model-scoped instances, alias resolution, deepcopy-on-filter) is well designed and correctly implemented. Test coverage for the new features is thorough.

However, one critical bug was found: the "warn-and-drop" preflight change does not actually drop incompatible attacks, causing them to pass preflight with a logged warning but still execute during the pipeline, leading to runtime crashes.

## Critical Issues

### CR-01: Preflight warn-and-drop does not actually drop attacks

**File:** `src/rbspaper/pipeline/core.py:764-827`
**Issue:** The `_preflight_pipeline_config` function builds a `valid_attacks` list (line 784) to track attacks that pass backend support validation. Attacks that fail validation are logged as warnings (line 810) but the `valid_attacks` list is never returned or used to modify the pipeline. Since `ExperimentPipelineConfig` is frozen (`@dataclass(frozen=True)` at `config.py:274`), `config.attacks` cannot be modified in-place. The function returns `None`, and the pipeline proceeds with the original unfiltered attack list.

As a result, incompatible attacks (e.g., SPSA+FORECASTING, which has no backend support) pass preflight with a warning but are still attempted during execution. When `execute_attack` is called for an unsupported combination, it will crash at runtime rather than being gracefully skipped.

The test `test_prefilter_warns_on_unsupported_attack` in `test_preflight_compat.py` only verifies that a warning is logged. It does not verify that the attack is actually excluded from execution, because `_preflight_pipeline_config` has no observable effect on the attack list.

**Fix:** Either (a) have `_preflight_pipeline_config` return the filtered attack tuple and use it to construct a new config, or (b) raise an error for unsupported attacks instead of warning, restoring the pre-existing strict validation behavior. Option (b) is the minimal fix:

```python
# Replace the try/except warn-and-drop block (lines 800-815) with:
validate_attack_support(
    attack=attack_config.parameters.attack_method,
    task=attack_config.context.task,
    backend=attack_config.parameters.backend,
    has_supervision=True,
)
```

If warn-and-drop is truly desired (informational only), then the actual filtering must happen downstream. The `_select_attacks_for_task` function (line 522) filters by task name matching but does not check backend support, so it will not catch unsupported attacks either. The filtering logic must be added at the point where attacks are executed (e.g., in `run_experiment_pipeline`'s attack loops at lines 183 and 699), or the function must return a modified config.

### CR-02 (related): `_logger` variable introduced but `logger` already exists -- inconsistent usage

**File:** `src/rbspaper/pipeline/core.py:54,66,810`
**Issue:** Two logger instances are created at module level:
- `_logger = logging.getLogger(__name__)` (line 54) -- added by this phase
- `logger = logging.getLogger(__name__)` (line 66) -- pre-existing

While both resolve to the same Logger object (logging uses a registry by name), the naming inconsistency is confusing. More importantly, the preflight warning (line 810) uses `_logger.warning(...)` while all other logging in the file uses `logger.info(...)`, `logger.exception(...)`, etc. This signals that the code author may have intended `_logger` to be separate, creating maintenance confusion.

**Fix:** Remove the `_logger` definition at line 54 and use `logger` at line 810 for consistency:

```python
# Remove line 54: _logger = logging.getLogger(__name__)
# Change line 810 from:
_logger.warning(...)
# to:
logger.warning(...)
```

## Warnings

### WR-01: `ExperimentInstance` dataclass is not frozen

**File:** `experiment_instances/instances.py:45-79`
**Issue:** The `ExperimentInstance` class is defined as `@dataclass` without `frozen=True`. The registry instances in `EXPERIMENTS_REGISTRY` are shared mutable objects. While `get_experiment_family` correctly uses `copy.deepcopy` when filtering by family, nothing prevents external code from mutating a registry instance's `attack_families` dict or other mutable fields.

The project guidelines state "Frozen dataclasses for all config" as an established pattern (`pipeline/config.py` uses `frozen=True` for all config classes). The `ExperimentInstance` is a config-like object that should follow this pattern.

Additionally, the `attack_families` field is a `dict`, and the `model_params` and `trainer_kwargs` fields are also mutable. Without `frozen=True`, an accidental mutation to a shared instance would affect all subsequent uses of that experiment.

**Fix:** Add `frozen=True` and handle mutable defaults with `field(default_factory=...)`:

```python
@dataclass(frozen=True)
class ExperimentInstance:
    id: str
    model_params: ModelParameters
    attack_families: dict[AttackFamily, tuple[AttackRunConfig, ...]] = field(
        default_factory=dict
    )
    # ... other fields
```

Note: `model_params` and `trainer_kwargs` would also need `default_factory` treatment. This requires updating all instantiation sites in `EXPERIMENTS_REGISTRY` and tests.

### WR-02: `_EXPERIMENT_ID_ALIASES` exported despite private naming convention

**File:** `experiment_instances/instances.py:35`
**Issue:** The `__all__` list (line 36-42) exports `_EXPERIMENT_ID_ALIASES`, which has a leading underscore indicating it is a private/internal symbol. Exporting it in `__all__` signals public API status, creating a contradiction. The test file `test_experiment_registry.py` imports it directly (line 11), suggesting it is needed for testing.

**Fix:** Either rename to `EXPERIMENT_ID_ALIASES` (without underscore) if it is intended as a public API, or remove it from `__all__` if it is internal-only. If removed from `__all__`, tests should access it via the module directly rather than importing it.

### WR-03: Redundant double filtering of attacks by family

**File:** `runners/py/runner.py:370-376, 266-269`
**Issue:** When `--attack_family` is provided, attacks are filtered twice:
1. `get_experiment_instance` (line 374) returns a deepcopy with only the requested family in `attack_families`.
2. `_build_pipeline_config` (line 266) re-filters by accessing `experiment_instance.attack_families.get(attack_family, ())`.

The second filter is redundant because the instance was already filtered. This is not incorrect (the result is the same), but it adds unnecessary complexity and masks the fact that the filtered instance only has one key in `attack_families`.

**Fix:** Remove the `attack_family` parameter from `get_experiment_instance` call in `main()`, since `_build_pipeline_config` already handles the filtering. Or remove the filtering from `_build_pipeline_config` if the instance is expected to be pre-filtered.

### WR-04: `valid_attacks` membership test uses O(n) list search

**File:** `src/rbspaper/pipeline/core.py:818`
**Issue:** The line `if attack_config in valid_attacks` performs an O(n) membership test on a list. For the current codebase (at most a handful of attacks), this is negligible. However, it indicates a structural issue: the code must search the list to determine whether the current attack passed validation, rather than tracking this with a simple boolean.

**Fix:** Use a boolean flag to track validation outcome:

```python
passed_validation = False
try:
    validate_attack_support(...)
    passed_validation = True
except ValueError as exc:
    logger.warning(...)

if passed_validation and attack_config.query_budget is not None:
    ...
```

## Info

### IN-01: Unused `AttackMethod` import in preflight test

**File:** `test/test_preflight_compat.py:23`
**Issue:** `AttackMethod` is imported from `src.rbspaper.attacks.enums` but never used in the test file.

**Fix:** Remove `AttackMethod` from the import statement.

### IN-02: Unused `pytest` import in preflight test

**File:** `test/test_preflight_compat.py:8`
**Issue:** `pytest` is imported but none of the test functions use `pytest.raises`, `pytest.mark`, or any pytest fixture beyond `caplog` and `tmp_path` (which are provided by pytest automatically).

**Fix:** Remove the `import pytest` line.

### IN-03: No test for empty-attack pipeline execution

**File:** `test/test_experiment_registry.py` (gap)
**Issue:** `test_unknown_family_returns_empty` verifies that requesting `BLACK_BOX` for an experiment with no black-box attacks returns an instance with empty `attack_params`. However, there is no integration test verifying that the full pipeline handles an empty attack list gracefully. If `config.attacks == ()`, the pipeline should skip all attack-related steps and proceed to evaluation.

**Fix:** Add a test using `test_preflight_compat.py`'s `_build_preflight_config` helper or a minimal pipeline run with `attacks=()` to confirm no crashes occur.

---

_Reviewed: 2026-05-07T12:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_

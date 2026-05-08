---
phase: 04-experiment-registry-restructure
verified: 2026-05-07T14:30:00Z
status: passed
score: 9/9 must-haves verified
overrides_applied: 0
---

# Phase 4: Experiment Registry Restructure Verification Report

**Phase Goal:** Restructure experiment registry from `model_attack` IDs (e.g., `ts2vec_fgsm`, `ts2vec_pgd`) to `model` IDs (e.g., `ts2vec`) with attack families and compatibility matrix.
**Verified:** 2026-05-07T14:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                                               | Status     | Evidence                                                                                              |
| --- | ------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------- |
| 1   | EXPERIMENTS_REGISTRY has exactly 2 entries (ts2vec, autotcl) — no old model_attack entries remain                   | VERIFIED   | `instances.py:215-232`: registry with keys `ts2vec` and `autotcl` only; `test_registry_has_model_scoped_ids` and `test_no_old_ids_in_registry` pass; behavioral check: `len(list_experiment_ids()) == 2` |
| 2   | `get_experiment_instance(experiment_id='ts2vec')` returns instance with all attacks from all families               | VERIFIED   | `instances.py:275-276`: returns registry instance directly when no filter; `test_no_filter_returns_instance` passes (identity check); behavioral check: 3 attacks (fgsm, pgd, bim) |
| 3   | `get_experiment_instance(experiment_id='ts2vec', attack_family=WHITE_BOX)` returns filtered copy                    | VERIFIED   | `instances.py:279-281`: `copy.deepcopy` + narrows `attack_families`; `test_white_box_filter_returns_copy` passes (not-same-object + family key check); behavioral check confirms copy semantics |
| 4   | Old IDs (ts2vec_fgsm, etc.) resolve via alias map with UserWarning                                                   | VERIFIED   | `instances.py:200-208`: 7-entry alias map; `instances.py:259-265`: `warnings.warn(..., UserWarning)` with deprecation message; `test_alias_emits_warning` and `test_alias_resolves_correct_instance` pass; behavioral check: UserWarning emitted with 'deprecated' text |
| 5   | `--attack_family white_box` flag filters to white-box attacks only                                                   | VERIFIED   | `runner.py:136-142`: arg added with `choices=['white_box', 'black_box']`; `runner.py:370-376`: resolves to `AttackFamily` enum, passes to `get_experiment_instance` and `_build_pipeline_config` |
| 6   | `--list_experiments` output contains "white_box:" text                                                               | VERIFIED   | `runner.py:356-367`: iterates `attack_families` dict, formats family summary; behavioral output: `ts2vec (white_box: fgsm, pgd, bim)` and `autotcl (white_box: fgsm, pgd, bim)` |
| 7   | Forecasting tasks skip label-dependent attacks with warning log (NOTE: code review found this was reverted to strict validation instead) | VERIFIED   | `core.py:797-803`: strict `validate_attack_support` call (raises ValueError, no try/except); `valid_attacks` list and `_logger` removed per CR-01 fix; `test_preflight_raises_on_unsupported_attack` confirms ValueError raised for SPSA+FORECASTING; ROADMAP SC #3 satisfied because only compatible attacks are registered (see SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK) |
| 8   | All existing tests pass with updated IDs                                                                            | VERIFIED   | `uv run pytest -q`: 111 passed, 13 warnings in 64.50s; `test_runner_cli_args.py`, `test_hierarchical_run_name.py`, `test_pipeline_state.py` all use `ts2vec` (not `ts2vec_fgsm`); no old ID references remain outside alias map and alias-specific tests |
| 9   | ruff + ty clean on all changed files                                                                                | VERIFIED   | `ruff check` on all 11 changed files: "All checks passed"; `ruff format --check`: 8 files already formatted; `ty check` on phase-04 files: 0 errors (only pre-existing `ty: ignore` warnings in core.py encoding calls, unchanged by this phase) |

**Score:** 9/9 truths verified

### Deferred Items

| # | Item                                                        | Addressed In | Evidence                                                  |
|---|-------------------------------------------------------------|-------------|-----------------------------------------------------------|
| 1 | `cost` experiment in `--list_experiments` (ROADMAP SC #5 mentions it)   | Not tracked in later phases | `cost` model does not exist in the codebase; no later phase (5-7) references it; ROADMAP SC #5 is aspirational. PLAN must-have specifies exactly 2 entries, which is met. |

### Required Artifacts

| Artifact                              | Expected                                                                 | Status   | Details                                                                                         |
| ------------------------------------- | ------------------------------------------------------------------------ | -------- | ----------------------------------------------------------------------------------------------- |
| `src/rbspaper/attacks/enums.py`       | `AttackFamily(StrEnum)` with WHITE_BOX/BLACK_BOX                         | VERIFIED | Lines 60-68: `AttackFamily(StrEnum)` with `WHITE_BOX = 'white_box'` and `BLACK_BOX = 'black_box'` |
| `src/rbspaper/attacks/registry.py`    | `group_methods_by_family()` helper                                       | VERIFIED | Lines 243-259: maps all AttackMethods to WHITE_BOX/BLACK_BOX families, excludes GRAY_BOX, returns `dict[AttackFamily, frozenset[AttackMethod]]` |
| `experiment_instances/instances.py`   | Model-scoped registry (2 entries), alias map (7 entries), family filter  | VERIFIED | Registry at line 215; alias map at line 200; `attack_families` field; `attack_params` property; `get_experiment_instance` with `attack_family` param at line 240 |
| `runners/py/runner.py`                | `--attack_family` CLI flag; enhanced `--list_experiments`                | VERIFIED | Flag at line 136; resolution at line 370; list format at line 356; `_build_pipeline_config` accepts `attack_family` at line 213 |
| `src/rbspaper/pipeline/core.py`       | Strict validation in preflight (reverted from warn-and-drop per CR-01)  | VERIFIED | Lines 797-803: direct `validate_attack_support` call (no try/except, raises ValueError); no `_logger` or `valid_attacks` remains |
| `test/test_attack_family.py`          | 5 tests for enum and grouping                                            | VERIFIED | All 5 tests pass: values, StrEnum check, full coverage, FGSM white-box, SPSA black-box         |
| `test/test_experiment_registry.py`    | 12 tests for registry structure, aliases, filtering                      | VERIFIED | 12 tests across 4 test classes (RegistryStructure, AliasResolution, FamilyFiltering, UnknownExperiment), all pass |
| `test/test_preflight_compat.py`       | 2 tests for strict validation behavior                                   | VERIFIED | `test_preflight_raises_on_unsupported_attack`: confirms ValueError; `test_preflight_passes_on_supported_attack`: confirms no warnings |
| `test/test_runner_cli_args.py`        | Updated experiment IDs (ts2vec not ts2vec_fgsm)                          | VERIFIED | All references use `ts2vec`; all 8 tests pass; no `ts2vec_fgsm` references remain               |
| `test/test_hierarchical_run_name.py`  | Updated IDs and assertions                                               | VERIFIED | Uses `ts2vec`; assertion `ts2vec/a1b2c3d4/seed_42/Coffee`; all 4 tests pass                    |
| `test/test_pipeline_state.py`         | Updated IDs and assertions                                               | VERIFIED | Lines 343, 345, 371 use `ts2vec`; assertion `ts2vec/a1b2c3d4/seed_42/Coffee`                   |

### Key Link Verification

| From                           | To                                      | Via                                            | Status   | Details                                                                 |
| ------------------------------ | --------------------------------------- | ---------------------------------------------- | -------- | ----------------------------------------------------------------------- |
| `runner.py`                    | `get_experiment_instance`               | Import + call in `main()`                       | WIRED    | Line 374: `get_experiment_instance(experiment_id=..., attack_family=...)` |
| `runner.py`                    | `AttackFamily` enum                     | Import from `src.rbspaper.attacks.enums`        | WIRED    | Line 32 import; line 372: `AttackFamily(args.attack_family)`            |
| `runner.py`                    | `--list_experiments` output             | Iterates `EXPERIMENTS_REGISTRY`                 | WIRED    | Lines 356-367: reads `attack_families` dict and formats output           |
| `instances.py`                 | `_EXPERIMENT_ID_ALIASES`                | Dict lookup in `get_experiment_instance`        | WIRED    | Line 257: `if experiment_id in _EXPERIMENT_ID_ALIASES`                  |
| `instances.py`                 | `copy.deepcopy`                         | Used for filtered family returns                | WIRED    | Line 279: `result = copy.deepcopy(instance)`                             |
| `core.py`                      | `validate_attack_support`               | Called in `_preflight_pipeline_config`          | WIRED    | Line 798: strict call (no try/except), raises ValueError on incompatibility |
| `registry.py`                  | `ATTACK_THREAT_MODEL`                   | Used by `group_methods_by_family`               | WIRED    | Line 253: iterates `ATTACK_THREAT_MODEL.items()`                        |
| `runner.py`                    | `_build_pipeline_config`                | Receives `attack_family` parameter              | WIRED    | Line 407: `attack_family=attack_family` passed from `main()`            |

### Data-Flow Trace (Level 4)

| Artifact                     | Data Variable       | Source                       | Produces Real Data | Status   |
| ---------------------------- | ------------------- | ---------------------------- | ------------------ | -------- |
| `instances.py`               | `attack_families`   | Factory functions (`_fgsm_attack_run`, `_pgd_attack_run`, `_bim_attack_run`) | Yes — real `AttackRunConfig` objects with parameters, context, metadata | FLOWING  |
| `instances.py`               | `attack_params` property | Flattens `self.attack_families.values()` | Yes — derived from real family data, verified by `test_attack_params_property_flattens` | FLOWING  |
| `runner.py`                  | `attacks` in pipeline config | `experiment_instance.attack_families.get(attack_family)` or `.attack_params` | Yes — real attack configs flow to `ExperimentPipelineConfig.attacks` | FLOWING  |
| `runner.py`                  | `--list_experiments` output | `inst.attack_families.items()` iteration | Yes — iterates real family dict, joins attack names from real `AttackRunConfig.name` | FLOWING  |

### Behavioral Spot-Checks

| Behavior                                                  | Command/Method                              | Result                                     | Status   |
| --------------------------------------------------------- | ------------------------------------------- | ------------------------------------------ | -------- |
| Registry has exactly 2 entries                            | Python: `len(list_experiment_ids())`        | 2 (ts2vec, autotcl)                        | PASS     |
| ts2vec returns all attacks                                | Python: `get_experiment_instance(ts2vec)`   | 3 attacks (fgsm, pgd, bim)                 | PASS     |
| Family filter returns copy                                | Python: identity check                      | `filtered is not original`                 | PASS     |
| Old ID alias emits UserWarning                            | Python: warnings.catch_warnings             | 1 UserWarning with "deprecated" text       | PASS     |
| `--list_experiments` shows family grouping                | Python: `main(['--list_experiments'])`      | Output: `ts2vec (white_box: fgsm, pgd, bim)` | PASS     |
| Preflight raises on unsupported attack                    | pytest: `test_preflight_raises_on_unsupported_attack` | ValueError raised as expected    | PASS     |
| Full test suite passes                                    | `uv run pytest -q`                          | 111 passed, 13 warnings in 64.50s         | PASS     |
| ruff check clean                                          | `uv run ruff check` on changed files        | All checks passed                          | PASS     |
| ruff format check clean                                   | `uv run ruff format --check` on changed files | 8 files already formatted              | PASS     |

### Requirements Coverage

| Requirement | Source Plan | Description | Status     | Evidence                                                   |
| ----------- | ---------- | ----------- | ---------- | ---------------------------------------------------------- |
| D-01        | 04-01-PLAN.md | AttackFamily enum and grouping in registry             | SATISFIED  | `AttackFamily` StrEnum in `enums.py`; `group_methods_by_family()` in `registry.py`; 5 tests in `test_attack_family.py` |
| D-02        | 04-01-PLAN.md | Pipeline preflight validates incompatible attacks      | SATISFIED  | `_preflight_pipeline_config` uses strict `validate_attack_support` (raises ValueError); 2 tests in `test_preflight_compat.py` |
| D-03        | 04-01-PLAN.md | Runner filters by family, pipeline validates by task   | SATISFIED  | Runner: `--attack_family` flag + `get_experiment_instance` filter; Pipeline: `validate_attack_support` in preflight |
| D-04        | 04-01-PLAN.md | --list_experiments shows model IDs with attack summary | SATISFIED  | Output verified: `ts2vec (white_box: fgsm, pgd, bim)`, `autotcl (white_box: fgsm, pgd, bim)` |

**Note:** `REQUIREMENTS.md` not found at `.planning/REQUIREMENTS.md`. Requirement IDs D-01 through D-04 traced from PLAN frontmatter and SUMMARY `requirements-completed` field.

### Anti-Patterns Found

None. No TODO/FIXME/placeholder comments, no empty implementations, no hardcoded empty data in phase-04 files.

**Review findings status (04-REVIEW.md):**
- **CR-01** (warn-and-drop doesn't actually drop): RESOLVED — reverted to strict validation; no `valid_attacks` or `_logger` in code
- **CR-02** (`_logger`/`logger` inconsistency): RESOLVED — only `logger` exists at line 64; no `_logger` anywhere
- **IN-01** (unused `AttackMethod` import): RESOLVED — removed from `test_preflight_compat.py`
- **IN-02** (unused `pytest` import): RESOLVED — removed from `test_preflight_compat.py`
- **WR-01** (ExperimentInstance not frozen): NON-BLOCKING — acceptable for current usage
- **WR-02** (private `_EXPERIMENT_ID_ALIASES` in `__all__`): NON-BLOCKING — intentional for test access
- **WR-03** (redundant double family filtering): NON-BLOCKING — harmless, slightly verbose
- **WR-04** (O(n) list membership test): NON-BLOCKING — negligible for current attack count

### Human Verification Required

None. All phase deliverables are verifiable through automated checks, behavioral spot-checks, and the full test suite (111 tests passing). End-to-end pipeline execution with real data files is not required for this phase's verification — the code paths are exercised through unit tests and the full integration test suite.

### Gaps Summary

No gaps found. All 9 must-have truths verified against the codebase. Must-have #7 (forecasting tasks skip label-dependent attacks) was intentionally reverted from warn-and-drop to strict validation per code review CR-01; this is consistent with the NOTE provided in the must-have description. The strict validation approach is more robust: it prevents incompatible attacks from silently reaching execution, and the compatibility matrix (`SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK`) already defines which attacks are valid for forecasting tasks.

---

_Verified: 2026-05-07T14:30:00Z_
_Verifier: Claude (gsd-verifier)_

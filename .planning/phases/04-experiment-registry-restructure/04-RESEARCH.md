# Phase 4: Experiment Registry Restructure — Research

**Date:** 2026-05-07
**Phase:** 04 — Experiment Registry Restructure

## Current Architecture

### ExperimentInstance (experiment_instances/instances.py)

- **Dataclass** with flat `attack_params: tuple[AttackRunConfig, ...]` — a single ungrouped tuple of attacks.
- **7 registry entries**: `ts2vec_fgsm`, `ts2vec_pgd`, `ts2vec_bim`, `ts2vec_multi`, `autotcl_fgsm`, `autotcl_pgd`, `autotcl_multi`.
- Each entry trains the model once with a specific attack subset. The `*_multi` variants bundle all 3 white-box attacks (FGSM, PGD, BIM).
- `get_experiment_instance(experiment_id)` does a simple dict lookup. Old IDs are the only keys.
- Helper factories (`_fgsm_attack_run`, `_pgd_attack_run`, `_bim_attack_run`) are reused across instances.
- `_ts2vec_params()` and `_autotcl_params()` build shared model parameters.

### Attack Infrastructure (src/rbspaper/attacks/)

- **enums.py**: `AttackMethod` (17 values), `AttackThreatModel` (WHITE_BOX, GRAY_BOX, BLACK_BOX), `AttackBackend`, `AttackObjective`, `AttackSupervisionRequirement`.
- **registry.py**: `ATTACK_THREAT_MODEL` maps all 17 AttackMethods to WHITE_BOX/BLACK_BOX. `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK` declares (task, method) → backends. `SUPERVISION_REQUIREMENT_BY_TASK_AND_ATTACK` declares label requirements. Only classification has 17 attacks; forecasting has only FGSM, PGD, BIM.
- **config.py**: `AttackParameters` hierarchy (FgsmAttackParameters, PgdAttackParameters, etc.) + `AttackExecutionContext`. Only FGSM, PGD, BIM have parameters used in instances.

### Pipeline Integration (src/rbspaper/pipeline/)

- **core.py:779** `_preflight_pipeline_config()` validates:
  - Downstream tasks match dataset profile `allowed_eval_tasks`
  - Attack uniqueness (no duplicate names)
  - `validate_attack_support()` per attack (backend + supervision check)
  - Query budget only for black-box attacks
- **core.py:537** `_select_attacks_for_task()` filters attacks by `attack.context.task.value == task_name` — matches attack context to downstream task.
- **config.py**: `AttackScopeConfig` controls whether attacks are task-conditioned or shared-input.

### Runner (runners/py/runner.py)

- `_build_pipeline_config()` at line 275 passes `attacks = tuple(experiment_instance.attack_params)` directly to pipeline. No family filtering.
- `--list_experiments` prints raw IDs from `list_experiment_ids()`.
- No `--attack_family` flag exists.

### Test Impact

Old experiment IDs referenced in tests:
- **test_runner_cli_args.py** — 8 references to `ts2vec_fgsm` as default experiment_id in CLI argument tests
- **test_hierarchical_run_name.py** — 6 references to `ts2vec_fgsm` in run_name construction tests
- **test_pipeline_state.py** — 2 references to `ts2vec_fgsm` in state hash/run-name tests

All test references use `ts2vec_fgsm` as the "known-good" experiment. None use other old IDs.

## Proposed Changes by Area

### 1. AttackFamily Enum (attacks/enums.py)

Add `AttackFamily(StrEnum)` with values `WHITE_BOX` and `BLACK_BOX`. The existing `AttackThreatModel` enum has `WHITE_BOX` and `BLACK_BOX` — `AttackFamily` maps **directly** to `AttackThreatModel`. Consider whether `AttackFamily` is needed as a separate enum or if `AttackThreatModel` can serve this purpose directly.

**Recommendation**: Use `AttackThreatModel` directly as the family concept. The `ATTACK_THREAT_MODEL` registry already maps methods to threat models. A new `AttackFamily` enum would be redundant with `AttackThreatModel` unless the semantics diverge. Per CONTEXT.md D-01, create `AttackFamily` as the CLI-facing concept (user-friendly grouping) that maps internally to `AttackThreatModel`.

### 2. ExperimentInstance Restructure (experiment_instances/instances.py)

Replace flat `attack_params` with `attack_families: dict[AttackFamily, tuple[AttackRunConfig, ...]]`. Three model-scoped entries: `ts2vec`, `autotcl`, `cost` (partial, deferred per CONTEXT.md).

**Key decision**: `ExperimentInstance` needs `attack_families` as a new field. The flat `attack_params` becomes a derived property that flattens all families (or selected ones) for backward compatibility.

New structure:
```python
@dataclass
class ExperimentInstance:
    id: str
    model_params: ModelParameters
    attack_families: dict[AttackFamily, tuple[AttackRunConfig, ...]]
    encoding_batch_size: int = 256
    max_epochs: int = 500
    downstream_tasks: tuple[str, ...] = ('classification',)
    attack_scope: AttackScopeConfig = field(default_factory=AttackScopeConfig)
    trainer_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def attack_params(self) -> tuple[AttackRunConfig, ...]:
        """Flatten all families into a single tuple."""
        return tuple(a for fam in self.attack_families.values() for a in fam)
```

### 3. get_experiment_instance with Family Filter

Add optional `attack_family: AttackFamily | None = None` parameter. When provided, return a **copy** of the instance with only that family's attacks. When None, returns the full instance with all attacks flattened.

**Important**: Cannot mutate the registry instance in-place (shared reference). Must deepcopy when filtering.

### 4. Alias Map for Old IDs

```python
_EXPERIMENT_ID_ALIASES: dict[str, str] = {
    'ts2vec_fgsm': 'ts2vec',
    'ts2vec_pgd': 'ts2vec',
    'ts2vec_bim': 'ts2vec',
    'ts2vec_multi': 'ts2vec',
    'autotcl_fgsm': 'autotcl',
    'autotcl_pgd': 'autotcl',
    'autotcl_multi': 'autotcl',
}
```

In `get_experiment_instance()`, check aliases first. Emit `UserWarning` with deprecation message on alias hit.

### 5. Runner --attack_family Flag

Add `--attack_family` argument to `_parse_args()`. Accept `white_box`, `black_box`, or omit (all). Two filtering passes:
1. **Runner**: filters instance attacks by family (user choice)
2. **Pipeline preflight**: auto-drops attacks incompatible with dataset tasks (system check)

### 6. --list_experiments Output

Format: `ts2vec (white_box: fgsm, pgd, bim; black_box: —)` showing families inline.

### 7. Pipeline Preflight Enhancement

Extend `_preflight_pipeline_config()` to check each attack against `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK` for the dataset's downstream tasks. Currently calls `validate_attack_support()` which **raises** on mismatch. Per CONTEXT.md D-02, change to **warn + drop** instead of raise for auto-filtering.

## Cross-Phase Dependencies

- **Phase 1 (completed)**: Import consistency established — all imports use `src.rbspaper.*`. No impact.
- **Phase 2 (completed)**: Mixin refactor — no impact on registry structure.
- **Phase 3 (completed)**: Pipeline hardening added resume gates, config hash, structured logging. The `_preflight_pipeline_config()` is the same function; Phase 3 didn't restructure it. The `compute_config_hash()` in runner is used for run naming — new experiment IDs change the hash input string (experiment_id part of run_name), but hash computation is parameter-based, not ID-based. **Safe to proceed.**
- **Phase 5 (future)**: Local test runners will use new experiment IDs. Phase 4 must be complete first.
- **Phase 6 (future)**: HPC runners depend on Phase 5 which depends on Phase 4.

## File-by-File Impact

| File | Change | Effort |
|------|--------|--------|
| `src/rbspaper/attacks/enums.py` | Add `AttackFamily` StrEnum | Small |
| `experiment_instances/instances.py` | New `attack_families` field, alias map, filter in `get_experiment_instance()`, 3 model-scoped entries | Large |
| `runners/py/runner.py` | `--attack_family` arg, family filter in `_build_pipeline_config()`, `--list_experiments` formatting | Medium |
| `src/rbspaper/pipeline/core.py` | `_preflight_pipeline_config()` warn-and-drop for incompatible attacks | Small |
| `src/rbspaper/attacks/registry.py` | Add helper to group methods by family (optional) | Small |
| `test/test_runner_cli_args.py` | Update all `ts2vec_fgsm` → `ts2vec` references | Small |
| `test/test_hierarchical_run_name.py` | Update `ts2vec_fgsm` → `ts2vec` references | Small |
| `test/test_pipeline_state.py` | Update `ts2vec_fgsm` → `ts2vec` references | Small |

## Implementation Strategy (Task Ordering)

### Wave 1: Foundation (no dependencies)
1. Add `AttackFamily` enum to `attacks/enums.py`
2. Add family grouping helper to `attacks/registry.py` (optional, maps methods → families)

### Wave 2: Registry restructure (depends on Wave 1)
3. Restructure `ExperimentInstance`: add `attack_families`, `attack_params` derived property, alias map, update `get_experiment_instance()` with family filter
4. Rewrite `EXPERIMENTS_REGISTRY` with model-scoped entries (ts2vec, autotcl)

### Wave 3: Runner + Pipeline integration (depends on Wave 2)
5. Add `--attack_family` CLI flag to runner, wire family filter in `_build_pipeline_config()`
6. Update `--list_experiments` output format
7. Enhance `_preflight_pipeline_config()` to warn-and-drop incompatible attacks

### Wave 4: Tests (depends on Wave 2)
8. Update `test_runner_cli_args.py` — replace old IDs with `ts2vec`
9. Update `test_hierarchical_run_name.py` — replace old IDs
10. Update `test_pipeline_state.py` — replace old IDs
11. Add new tests: alias deprecation warning, family filtering, --list_experiments format

## Validation Architecture

### Unit Tests
- `AttackFamily` enum values align with `AttackThreatModel`
- `ExperimentInstance.attack_params` property flattens all families correctly
- `get_experiment_instance('ts2vec')` returns full instance with all attacks
- `get_experiment_instance('ts2vec', attack_family=AttackFamily.WHITE_BOX)` returns filtered copy
- Alias lookup emits `UserWarning` and resolves to correct new ID
- `--list_experiments` output contains family grouping
- `_preflight_pipeline_config()` warns on incompatible attacks for forecasting datasets
- `run_name` still builds correctly with new shorter IDs

### Integration Tests
- `runner --experiment_id ts2vec --attack_family white_box` → pipeline sees only white-box attacks
- `runner --experiment_id ts2vec` (no flag) → pipeline sees all attacks
- Forecasting dataset with label-dependent attacks → preflight warns + skips
- Old ID `ts2vec_fgsm` → deprecation warning + runs as `ts2vec`

### Regression Guarantees
- All existing tests pass with new IDs
- Config hash computation unaffected (based on model params + seed, not experiment ID)
- Pipeline state/resume unaffected (uses run_name which now has shorter experiment_id)

## Risk Assessment

1. **Low risk**: `ExperimentInstance` dataclass field change — only consumed by runner and tests. No external API dependency.
2. **Low risk**: `_preflight_pipeline_config()` change — warn-and-drop instead of raise is additive behavior. The `validate_attack_support()` call already checks support; we're just catching it earlier and logging.
3. **Medium risk**: Test updates — need to ensure all old ID references are found. Use grep to verify completeness.
4. **No risk**: `AttackFamily` enum — additive, doesn't break existing code.

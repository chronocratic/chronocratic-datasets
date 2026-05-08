# Phase 4: Experiment Registry Restructure - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Restructure the experiment registry from `model_attack` IDs (e.g., `ts2vec_fgsm`, `ts2vec_pgd`, `ts2vec_bim`, `ts2vec_multi`) to model-scoped IDs (e.g., `ts2vec`, `autotcl`, `cost`) with attack families and declarative compatibility. One training run, N attacks applied on M downstream tasks.

**In scope:**
- Rewrite `experiment_instances/instances.py`: one ID per model, attacks organized by family
- Add `AttackFamily` enum (white_box, black_box) to `src/rbspaper/attacks/enums.py`
- Pipeline preflight auto-filters attacks incompatible with downstream task, logs skips
- Runner `--attack_family` CLI flag to select whitebox, blackbox, or default all
- Remove redundant experiment IDs: `ts2vec_fgsm`, `ts2vec_pgd`, `ts2vec_bim`, `ts2vec_multi`, `autotcl_fgsm`, `autotcl_pgd`, `autotcl_multi`
- Update `--list_experiments` to show model IDs with attack family summary
- Update runner to resolve model-scoped instances and filter by family
- Update all tests referencing old experiment IDs

**Out of scope:**
- New model architectures or attack types
- HPC SLURM integration (Phase 6)
- Local bash runners (Phase 5)
- CoST experiment instance (no attack params defined yet — may be Phase 7 cleanup)

</domain>

<decisions>
## Implementation Decisions

### Experiment Instance Structure (D-01)
- **Attack families as named groups** — Add `attack_families: dict[AttackFamily, tuple[AttackRunConfig, ...]]` field to `ExperimentInstance`. Replace flat `attack_params` tuple. Groups map directly to `--attack_family` CLI flag. Extensible for future families (quantitative, transfer).

### Attack-Task Compatibility (D-02)
- **Use existing attack registry tables** — Extend pipeline `_preflight_pipeline_config()` to auto-drop attacks not in `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK` for the dataset's downstream tasks. Log a warning per skipped attack. No new declarations needed — the registry already has the compatibility info.

### Filtering Logic Location (D-03)
- **Split concerns** — Runner filters attacks by `--attack_family` (user intent, explicit choice). Pipeline preflight filters by task compatibility (system capability, auto-skip). Two passes with clear separation: runner decides "which family", pipeline decides "which survive for this task".

### List Experiments Output (D-04)
- **Model IDs with attack summary** — `--list_experiments` shows each model ID with inline attack families, e.g., `ts2vec (white_box: fgsm, pgd, bim; black_box: spsa, uap)`. Self-documenting for users discovering available experiments.

### Claude's Discretion
- Exact type of `AttackFamily` enum values (e.g., `white_box` vs `whitebox` — follow existing `AttackThreatModel` convention which uses `WHITE_BOX`)
- Whether to keep `attack_params` as legacy field or remove entirely (recommend: replace, remove)
- How to handle the case where `--attack_family` is not provided (recommend: default to all attacks across all families)
- Test structure for new registry format (recommend: parametrize by model × family × task)
- Migration strategy for old IDs (recommend: alias map with deprecation warning)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Experiment Registry
- `experiment_instances/instances.py` — Current registry: `ExperimentInstance` dataclass with flat `attack_params`, `EXPERIMENTS_REGISTRY` dict (7 model-attack entries), `get_experiment_instance()`. Must be restructured to model-scoped with family groups.
- `experiment_instances/data_utils.py` — `build_dataset_task_profile()` derives task profile from dataset registry. Used by runner to resolve dataset→task mapping.

### Attack Infrastructure
- `src/rbspaper/attacks/registry.py` — `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK`, `SUPERVISION_REQUIREMENT_BY_TASK_AND_ATTACK`, `ATTACK_THREAT_MODEL`, `list_supported_attacks()`, `validate_attack_support()`. The compatibility data source for preflight filtering.
- `src/rbspaper/attacks/enums.py` — `AttackMethod`, `AttackBackend`, `AttackThreatModel`, `AttackObjective`, `AttackSupervisionRequirement`. New `AttackFamily` enum added here.
- `src/rbspaper/attacks/config.py` — `AttackRunConfig`, `AttackExecutionContext`, `AttackParameters` hierarchy. Attack configs that will be grouped into families.

### Pipeline
- `src/rbspaper/pipeline/core.py` — `run_experiment_pipeline()` (line 57), `_preflight_pipeline_config()` (line 591), `_select_attacks_for_task()` (line 537). Preflight is the integration point for task-compatibility filtering.
- `src/rbspaper/pipeline/config.py` — `AttackScopeConfig` (line 52), `AttackRunConfig` re-use, `DatasetTaskProfile` (line 71). Frozen dataclass patterns.

### Runner
- `runners/py/runner.py` — CLI entry point. `--experiment_id`, `--list_experiments`, `_resolve_dataset`, `_print_summary`, `_build_pipeline_config`. Must add `--attack_family`, update resolution logic, and format list output.

### Models
- `src/rbspaper/models/config.py` — `TS2VecModelParameters`, `AutoTCLModelParameters`. Current instances use TS2Vec and AutoTCL; CoST not registered yet.

### Tests
- `test/test_pipeline_core.py` — Tests referencing old experiment IDs that need updating.
- `test/test_attacks_registry.py` — Tests for attack registry, likely stable.
- `test/test_attacks_functional.py` — Functional attack tests, likely stable.
- `test/test_attacks_batch.py` — Batch attack tests, likely stable.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ATTACK_THREAT_MODEL` dict (`attacks/registry.py:13`) — maps `AttackMethod → AttackThreatModel`. Direct basis for `AttackFamily` grouping.
- `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK` (`attacks/registry.py:32`) — already declares which attacks work on which tasks. The preflight filter just needs to consult it.
- `_select_attacks_for_task()` (`pipeline/core.py:537`) — existing helper that filters attacks by task name. New preflight filter complements this.
- `_preflight_pipeline_config()` (`pipeline/core.py:591`) — validates attack scope, uniqueness, task binding. Natural place to add compatibility filtering.

### Established Patterns
- Frozen dataclasses for all config (`pipeline/config.py`) — new `AttackFamily` should follow this.
- `src.rbspaper.*` import prefix established in Phase 1.
- Attack params use `AttackRunConfig(name, parameters, context, metadata)` — family grouping wraps these, doesn't replace.
- Registry dicts are static module-level constants — new aliases for old IDs should follow this.

### Integration Points
- `ExperimentInstance` — add `attack_families` field, deprecate/remove `attack_params`.
- `get_experiment_instance()` — add optional `attack_family` filter parameter.
- `main()` in runner — add `--attack_family` arg, pass to instance resolution.
- `--list_experiments` handler — iterate registry, format family summary per model.
- `_preflight_pipeline_config()` — iterate attacks, check against `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK`, drop with warning.
- `test/test_pipeline_core.py` — update any fixtures using old IDs (`ts2vec_fgsm`, etc.) to new format.

### Existing Threat Model Mapping (for AttackFamily derivation)
The `ATTACK_THREAT_MODEL` registry already classifies all 17 attack methods:
- **White-box:** FGSM, BIM, PGD, DeepFool, CW, LBFGS, MI-FGSM, AutoAttack, JSMA, One-Pixel, EAD
- **Black-box:** SPSA, UAP, HopSkipJump, Boundary, Simba

Only FGSM, PGD, BIM are registered in current experiment instances. Only FGSM, PGD, BIM have forecasting support.

</code_context>

<specifics>
## Specific Ideas

- Success criteria from ROADMAP.md requires: `uv run rbspaper-run --experiment_id ts2vec --dataset_name Coffee` trains TS2Vec once, applies all attacks. The new instance must default to all families when `--attack_family` is omitted.
- Success criteria: `uv run rbspaper-run --experiment_id ts2vec --attack_family whitebox --dataset_name Coffee` trains once, applies only whitebox attacks.
- Success criteria: forecasting tasks auto-skip label-dependent attacks without errors.
- User prefers minimal complexity — keep the alias map small with deprecation warnings, not silent forwarding.

</specifics>

<deferred>
## Deferred Ideas

- CoST experiment instance registration — no attack params defined yet. Deferred to Phase 7 cleanup or when CoST attack evaluation is needed.
- Factory-based attack construction (attack methods declared as enum set, runner builds params) — considered but rejected; families with full configs are simpler and co-locate params with instance.
- Quantitative/transfer attack families — future extensions beyond white_box/black_box. Not needed now.
- Full backward compatibility layer for old experiment IDs — alias map is sufficient; no need to keep old instances as first-class entries.

</deferred>

---

*Phase: 4-Experiment Registry Restructure*
*Context gathered: 2026-05-07*

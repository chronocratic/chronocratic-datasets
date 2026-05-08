# Phase 4: Experiment Registry Restructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-07
**Phase:** 4-experiment-registry-restructure
**Areas discussed:** Experiment Instance Structure, Attack-Task Compatibility, Filtering Logic Location, List Experiments Output
**Mode:** --all --analyze with recommended defaults accepted

---

## Experiment Instance Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Full attack param tuples (default set) | Single flat tuple, filter at pipeline | |
| Named attack families | `attack_families: dict[AttackFamily, tuple[AttackRunConfig, ...]]` | ✓ |
| Factory-based attacks | Instance declares methods, runner constructs configs | |

**User's choice:** Named attack families (recommended)
**Notes:** Maps directly to `--attack_family` CLI flag. Extensible for future families. Keeps configs co-located with instances.

---

## Attack-Task Compatibility

| Option | Description | Selected |
|--------|-------------|----------|
| Extend attack registry with compatible_tasks field | New declaration in `AttackRunConfig.context` | |
| Pipeline preflight filter (existing registry) | Use `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK` to auto-drop | ✓ |
| Compatibility matrix in experiment instance | Per-experiment compat dict | |

**User's choice:** Pipeline preflight filter using existing registry (recommended)
**Notes:** Registry already has task-attack compatibility info. No duplication needed.

---

## Filtering Logic Location

| Option | Description | Selected |
|--------|-------------|----------|
| Filter at runner level | Runner drops all incompatible attacks | |
| Filter at pipeline preflight | Preflight does all filtering | |
| Split: runner filters family, preflight filters compat | User intent vs system capability | ✓ |

**User's choice:** Split concerns (recommended)
**Notes:** Runner = `--attack_family` (explicit user choice), preflight = task compatibility (auto-skip with warning).

---

## List Experiments Output

| Option | Description | Selected |
|--------|-------------|----------|
| Model-only IDs | `ts2vec`, `autotcl`, `cost` | |
| Model IDs with attack summary | `ts2vec (white_box: fgsm, pgd, bim)` | ✓ |

**User's choice:** Model IDs with attack summary (recommended)
**Notes:** Self-documenting output helps users discover what each experiment does.

---

## Claude's Discretion

- Attack family enum value naming convention (follow existing `AttackThreatModel.WHITE_BOX`)
- Whether to keep or remove legacy `attack_params` field (recommend: remove)
- Default when `--attack_family` not provided (recommend: all families)
- Test structure for new format (recommend: parametrize model × family × task)
- Migration strategy for old IDs (recommend: alias map with deprecation warning)

## Deferred Ideas

- CoST experiment instance — deferred to Phase 7 or later
- Quantitative/transfer attack families — future extension
- Factory-based attack construction — considered, rejected for complexity
- Full backward compatibility layer — alias map sufficient

---

*User accepted all 4 recommended options without modification.*

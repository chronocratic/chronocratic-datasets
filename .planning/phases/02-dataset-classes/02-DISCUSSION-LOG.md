# Phase 2: Dataset Classes - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-11
**Phase:** 02-Dataset Classes
**Areas discussed:** Dataset constructor API, Transformation strategy, Utility porting scope, Concrete dataset + test design

---

## Dataset Constructor API

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-loaded data (module-only) | Datasets take data/labels, no I/O. Modules own loading. | ✓ |
| File paths + lazy load | Datasets accept path, handle parsing. Standalone usable. | |
| Both data OR path | Flexible but adds validation complexity. | |

**User's choice:** Pre-loaded data (module-only)
**Notes:** User clarified "why preload? isn't this handled in the module?" — confirmed that modules load data and instantiate datasets. Datasets are pure iterators. This aligns with rbspaper pattern and keeps datasets testable.

## Transformation Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed per class | Baked-in transforms, no override. | |
| Defaults + optional override | Sensible defaults, `transformations_sequence` kwarg for customization. | ✓ |
| Full compose() pipeline | Maximum flexibility, requires compose/partial knowledge. | |

**User's choice:** Defaults + optional override
**Notes:** User wanted balance — out-of-the-box works, power users can customize.

## Utility Porting Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Port arff.py + general.py now (roadmap) | Follow roadmap table, Phase 2 self-contained. | |
| Defer to Phase 5 | Port only transformations.py + common.py. arff/general where modules use them. | ✓ |
| You decide | — | |

**User's choice:** Defer to Phase 5
**Notes:** Since datasets are module-only (no I/O), arff.py is logically a module concern. Keeps Phase 2 smaller and focused.

## Concrete Dataset Design

| Option | Description | Selected |
|--------|-------------|----------|
| Thin wrappers (rbspaper pattern) | 10-line classes that set domain defaults. | ✓ |
| Self-configuring with defaults | Simpler API, more code per class. | |
| Factory functions | No class ceremony, breaks PyTorch convention. | |

**User's choice:** Thin wrappers

## Testing Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Synthetic + minimal real fixtures | Fast unit tests + format validation. | ✓ |
| Synthetic only | Pure numpy/pandas, no extra files. | |
| Property-based (hypothesis) | Generates edge cases, extra dependency. | |

**User's choice:** Synthetic + minimal real fixtures

## Claude's Discretion

- Internal cursor management (`_n`, `_go_to_idx`) follows rbspaper pattern verbatim
- Type hints use `from __future__ import annotations` where rbspaper source already uses it

## Deferred Ideas

- `arff.py`, `general.py`, `scaling.py`, `features.py` utility ports — deferred to Phase 5 (modules)
- UEA multivariate format validation — deferred to Phase 5 with real ARFF fixtures
- Dynamic class generation from registry — Phase 6 (factory API)

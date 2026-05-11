# Phase 3: Pydantic Registry - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-11
**Phase:** 3-Pydantic Registry
**Areas discussed:** Config class hierarchy, instance exposure, forecasting splits, validation strategy, config immutability, config scope

---

## Config Class Hierarchy

| Option | Description | Selected |
|--------|-------------|----------|
| Layered inheritance | DatasetConfig base (abstract) + family-specific subclasses with own fields | ✓ |
| All-in-base | Single DatasetConfig with all fields as Optional | |
| Flat with model_config | Single class, vary validation per instance | |

**User's choice:** Layered inheritance (Recommended)
**Notes:** The roadmap already specifies per-family classes. Layered inheritance keeps configs type-safe and avoids sparse None defaults.

## Instance Exposure

| Option | Description | Selected |
|--------|-------------|----------|
| Both constants + registry | Module-level constants for direct import + registry dict for lookup | ✓ |
| Registry only | All configs in a dict, access via get_config() | |
| Constants only | Just module-level constants, no registry | |

**User's choice:** Both constants + registry (Recommended)
**Notes:** Best DX — IDE auto-complete for imports, iterable registry for factory API.

## Forecasting Splits

| Option | Description | Selected |
|--------|-------------|----------|
| Dual-mode | SplitMode enum (INDEXED vs FRACTIONAL), validated with model_validator | ✓ |
| Index-based only | Always use absolute indices | |
| Fraction-based only | Always use fractions | |

**User's choice:** Dual-mode (Recommended)
**Notes:** ETT uses intrinsic time-based splits (16/4/4 months), electricity uses 60/20/20 fractions. Neither representation alone covers both.

## Validation Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Both field + model validators | @field_validator for single fields, @model_validator for cross-field | ✓ |
| Model validator only | Single @model_validator(mode='after') for everything | |
| Field validators + runtime | @field_validator + assert/log in modules | |

**User's choice:** Both field + model validators (Recommended)
**Notes:** Best error messages, catches issues at config creation time.

## Config Immutability

| Option | Description | Selected |
|--------|-------------|----------|
| Class-level frozen + model_copy | ConfigDict(frozen=True) on base, model_copy() for variants | ✓ |
| Instance-level freeze | Freeze after __init__ | |
| No freeze, convention | Skip frozen=True | |

**User's choice:** Class-level frozen + model_copy (Recommended)
**Notes:** Most idiomatic Pydantic v2. Prevents accidental mutation of shared instances.

## Config Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Complete | All fields modules need: url, sha256, file_patterns, split_bounds, ... | ✓ |
| Minimal + defaults | Registry metadata + download info, module-specific fields in code | |
| Registry-only | Just name, family, tasks | |

**User's choice:** Complete (Recommended)
**Notes:** Enables get_module("Coffee") to work from config alone. No external JSON files needed.

---

## Claude's Discretion

- Exact field names for split boundaries (train_end_index vs train_slice_end)
- Config auto-registration approach (__all__ iteration vs decorator)
- Whether to add computed properties (cache_key from url + sha256)

## Deferred Ideas

- Dynamic class generation from registry — Phase 6 (factory API)
- Conda package distribution — v2 requirement
- Full UCR archive (120+ datasets) — v2 requirement

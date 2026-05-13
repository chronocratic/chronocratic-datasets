# Phase 3: Utility Modules - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-13
**Phase:** 03-utility-modules
**Areas discussed:** File organization, Enum vs string for scaling, Dependency on Phase 4 modules, Style and typing improvements

---

## File Organization

| Option | Description | Selected |
|--------|-------------|----------|
| Option A | Merge everything into common.py | |
| Option B | Keep separate files per concern | ✓ |
| Option C | Rename common.py to general.py | |

**User's choice:** Option B — separate files per concern (arff.py, scaling.py, features.py, general.py). Keep existing common.py as-is (Phase 2 functions).

## Enum vs String for Scaling

| Option | Description | Selected |
|--------|-------------|----------|
| Option A | Use ScalingMethod and DataForm enums | ✓ |
| Option B | Keep strings to match source | |
| Option C | Literal for scaling_method, enum for data_form | |

**User's choice:** Option A — use enums from our enums package for type safety.

## ScalingMethod Value Mismatch

| Option | Description | Selected |
|--------|-------------|----------|
| Rename both | Use 'minmax'/'standard' everywhere | ✓ |
| Adjust enum values | Match rbspaper source values | |
| Adapt the function | Map enum to internal strings | |

**User's choice:** Rename both — keep clean enum values ('minmax'/'standard') and update the ported function to compare against enum values, not rbspaper source strings.

## DataFormEnum Location

| Option | Description | Selected |
|--------|-------------|----------|
| Add to enums/data.py | Co-locate with other data enums | ✓ |
| Keep in scaling.py | Define locally where used | |

**User's choice:** Add to enums/data.py — import from there in scaling.py.

## Dependency on Phase 4 modules

| Option | Description | Selected |
|--------|-------------|----------|
| Option A | Port all 4 utility files now | |
| Option B | Port only scaling.py now | |
| Option C | Port all but defer tests | |

**User's choice:** "port all now" — all 4 files, all ported and tested in Phase 3.

## Style and Typing

| Option | Description | Selected |
|--------|-------------|----------|
| Option A | Full CLAUDE.md compliance | ✓ |
| Option B | Minimal style changes | |

**User's choice:** Option A — keyword-only args, full type hints, clean imports.

---

## Claude's Discretion

None — user made explicit choices on all areas.

## Deferred Ideas

None — discussion stayed within phase scope.

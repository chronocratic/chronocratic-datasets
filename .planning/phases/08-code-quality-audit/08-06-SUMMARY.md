---
phase: 08-code-quality-audit
plan: 06
subsystem: code_quality
tags: [ty, type_annotations, lightning, data_module, config]

requires:
  - phase: 08-code-quality-audit
    plan: 01
    provides: ty-clean encoding.py and core.py Wave 1 cleanup
provides:
  - DataConfig.data_module widened to pl.LightningDataModule
  - Documented ty: ignore comments for encode_data type gap
affects: [08-code-quality-audit, all plans using DataConfig]

tech-stack:
  added: []
  patterns: [Documented ty: ignore with # why: for genuine type checker limitations]

key-files:
  created: []
  modified:
    - src/rbspaper/pipeline/config.py
    - src/rbspaper/pipeline/core.py

key-decisions:
  - "Widened DataConfig.data_module to pl.LightningDataModule per D-02"
  - "Added documented ty: ignore at encode_data call sites; widening _ModelType in encoding.py would cause call-non-callable errors (discovered 08-01)"

requirements-completed: []

duration: 15min
completed: 2026-05-08
---

# Phase 08 Plan 06: Widen DataConfig.data_module Type and Clean core.py Ty Ignores Summary

**DataConfig.data_module widened to pl.LightningDataModule; documented ty: ignore comments added at encode_data call sites for unfixable type gap**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-08T15:00:00Z
- **Completed:** 2026-05-08T15:15:00Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- DataConfig.data_module type widened from BaseTimeSeriesDataModule to pl.LightningDataModule (D-02)
- Removed unused BaseTimeSeriesDataModule import from config.py TYPE_CHECKING block
- Added 4 documented ty: ignore[invalid-argument-type] comments in core.py at encode_data call sites
- Both files pass ty check and ruff check cleanly

## Task Commits

Each task was committed atomically:

1. **Task 1: Widen data_module type in config.py and clean core.py ty: ignore** - `5e5bf17` (feat)

## Files Created/Modified

- `src/rbspaper/pipeline/config.py` - Widened DataConfig.data_module to pl.LightningDataModule; removed unused BaseTimeSeriesDataModule import
- `src/rbspaper/pipeline/core.py` - Added documented ty: ignore comments at 4 encode_data call sites; applied ruff formatting

## Decisions Made

- **Documented ty: ignore pattern:** The original plan assumed Wave 1 would widen `_ModelType` in encoding.py to `pl.LightningModule`, eliminating the need for ty: ignore comments in core.py. However, 08-01 discovered that widening `_ModelType` causes 3 `call-non-callable` errors because `pl.LightningModule.encode` is typed as `Tensor | Module` (not callable). Therefore, the type gap between `ExperimentPipelineConfig.model: pl.LightningModule` and `encode_data(model: TS2Vec | AutoTCL | CoST)` requires documented suppression comments.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added documented ty: ignore comments to core.py instead of removing them**

- **Found during:** Task 1 (ty verification)
- **Issue:** Plan stated "Zero `ty: ignore` comments in core.py" as a must-have. However, the pre-requisite widening of `_ModelType` in encoding.py (assumed done by Wave 1) was not applied because it causes `call-non-callable` errors on `model.encode()` calls (discovered in 08-01 execution). Without the widening, core.py passes `pl.LightningModule` to `encode_data()` which expects the narrower `TS2Vec | AutoTCL | CoST` union, producing 4 `invalid-argument-type` errors.
- **Fix:** Added 4 documented `ty: ignore[invalid-argument-type]` comments with `# why:` explanations at the encode_data call sites in core.py (lines 300, 307, 314, 441). The `# why:` comments explain this is a genuine limitation: widening `_ModelType` to `pl.LightningModule` would break `model.encode()` calls.
- **Files modified:** src/rbspaper/pipeline/core.py
- **Verification:** `uv run ty check src/rbspaper/pipeline/core.py` reports zero diagnostics.
- **Committed in:** 5e5bf17

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking issue requiring documented suppression instead of clean removal)
**Impact on plan:** The config.py widening (D-02) was applied exactly as planned. The core.py ty: ignore situation required a documented limitation instead of clean removal, due to the 08-01 discovery about `_ModelType` widening side effects. This is consistent with the CONTEXT.md genuine limitations approach.

## Issues Encountered

- The plan referenced specific line numbers (530, 537, 544, 671) for ty: ignore removal in core.py. Those ignores were already removed by Wave 1 (plan 08-01). The current ty errors are at different call sites (298-318, 439-445) caused by the type mismatch between `pl.LightningModule` and `TS2Vec | AutoTCL | CoST`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- config.py: DataConfig.data_module is correctly typed as pl.LightningDataModule
- core.py: ty check clean with documented limitations
- Both files pass ruff check and ruff format
- Subsequent phase 08 plans can proceed independently

---
*Phase: 08-code-quality-audit*
*Completed: 2026-05-08*

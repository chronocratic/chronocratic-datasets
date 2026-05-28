---
plan: 06-07
phase: 06-lightning-lifecycle
status: complete
self_check: passed
updated: '2026-05-28T14:30:00Z'
---

## Description

Integration tests for all 5 concrete modules verifying lifecycle idempotency, prepare_data sentinel, finalize hook, dimension API, and stage gating. Plus forecasting setup stage gating implementation to match classification.

## Changes

- 17 new integration tests across 3 test files
- Rewrote forecasting `setup()` with stage validation, idempotency guard, fitted scaler caching, and stage branching
- TestSetupIdempotency: ETT, Weather, Electricity — setup('fit') twice produces identical data
- TestPrepareDataIdempotency: prepare_data() twice runs I/O once (all 3 modules)
- TestFinalizePrepareData: prepare_data() sets slices via base wrapper (all 3 modules)
- TestPrepareDimensions: pre-setup dims == post-setup dims (all 3 modules)
- TestSetupStageGating: scaler cache populated by fit, reused by test, validate no-op
- test_setup_idempotent: UCR and UEA classification setup idempotency

## Key Files

- src/tscollection/datasets/modules/_base/forecasting.py
- tests/test_modules_forecasting.py
- tests/test_modules_ucr.py
- tests/test_modules_uea.py

## Verification

214 tests pass. 17 new tests added for integration coverage.

## Self-Check: PASSED

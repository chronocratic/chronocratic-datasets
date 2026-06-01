---
phase: 07
slug: ddp-compliance
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-29
updated: 2026-05-29
---

# Phase 07 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8.2 |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` (`pythonpath = ["."]`, `testpaths = ["tests"]`) |
| **Quick run command** | `uv run pytest tests/test_cache.py tests/test_modules_base.py tests/test_modules_forecasting.py -x` |
| **Full suite command** | `uv run pytest tests/ -x --cov=src/tscollection` |
| **Estimated runtime** | ~5 seconds (quick), ~60 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_cache.py tests/test_modules_base.py tests/test_modules_forecasting.py -x` (~5 seconds; covers cache module + touched bases).
- **After every plan wave:** Run `uv run pytest tests/ -x` (full suite).
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 07-01 | 01 | 1 | DDP-01 | — | DDP smoke test (gloo, 2 ranks): identical state across ranks after `setup('fit')` | integration | `uv run pytest tests/test_ddp_compliance.py::TestDDPSmokeTests::test_ddp_forecasting_cache_round_trip -x` | Exists | | green |
| 07-01 | 01 | 1 | DDP-02 | — | No `isinstance` branches on `_full_data` consumers | static analysis | `uv run pytest tests/test_ddp_compliance.py::TestIsinstanceBranchElimination -x` | Exists | | green |
| 07-01 | 01 | 1 | D-03 | — | `prepare_dimensions()` reads metadata without loading arrays | unit | `uv run pytest tests/test_modules_base.py::TestPrepareDimensionsCache::test_reads_metadata_when_num_features_none -x` | Exists | | green |
| 07-01 | 01 | 1 | D-07 | — | `prepare_dimensions()` raises when metadata missing | unit | `uv run pytest tests/test_modules_base.py::TestPrepareDimensionsCache::test_raises_file_not_found_when_metadata_missing -x` | Exists | | green |
| 07-01 | 01 | 1 | D-08 | — | `metadata.json` schema matches spec | unit | `uv run pytest tests/test_cache.py -k "schema" -x` | Exists | | green |
| 07-02 | 02 | 1 | MOD-01 | — | `synthetic_cache_dir` fixture returns Path with npz + metadata.json + scaler.pt | unit | `uv run pytest tests/conftest.py::test_synthetic_cache_dir_fixture -x --tb=short` | Exists | | green |
| 07-03 | 03 | 2 | D-02 | — | `resolve_cache_dir(None)` returns `~/.cache/tsdatasets/<name>` | unit | `uv run pytest tests/test_cache.py::test_resolve_cache_dir_default_returns_expected_path -x` | Exists | | green |
| 07-03 | 03 | 2 | D-02 | — | `resolve_cache_dir(custom_path)` returns custom path | unit | `uv run pytest tests/test_cache.py::test_resolve_cache_dir_custom_path_passes_through -x` | Exists | | green |
| 07-03 | 03 | 2 | D-03 | — | `build_cache_key` produces deterministic hybrid key | unit | `uv run pytest tests/test_cache.py -k "build_cache_key" -x` | Exists | | green |
| 07-04 | 04 | 3 | D-04 | — | npz round-trip preserves dtype and shape | unit | `uv run pytest tests/test_cache.py -k "atomic_save_npz" -x` | Exists | | green |
| 07-04 | 04 | 3 | D-06 | — | torch.save/load sklearn scaler produces identical transform | unit | `uv run pytest tests/test_cache.py -k "scaler_round_trip" -x` | Exists | | green |
| 07-05 | 05 | 4 | D-01 | — | `_full_data_raw` immutable after `setup()` load | unit | `uv run pytest tests/test_modules_forecasting.py -k "test_setup_reads_cache_and_sets_raw" -x` | Exists | | green |
| 07-05 | 05 | 4 | D-04 | — | metadata.json version mismatch raises ValueError | unit | `uv run pytest tests/test_cache.py::test_load_metadata_raises_value_error_on_version_mismatch -x` | Exists | | green |
| 07-05 | 05 | 4 | ROADMAP DDP-04 | — | Second `setup()` call produces identical output | unit | `uv run pytest tests/test_modules_forecasting.py::TestSetupIdempotency -x` | Exists | | green |
| 07-06 | 06 | 5 | MOD-01 | — | Weather/Electricity cache round-trip (prepare_data writes, setup reads) | integration | `uv run pytest tests/test_modules_forecasting.py -x --tb=short -k "Weather or Electricity"` | Exists | | green |
| 07-06 | 06 | 5 | MOD-05 | — | Classification UCR/UEA cache round-trip | integration | `uv run pytest tests/test_modules_ucr.py tests/test_modules_uea.py -k cache -x` | Exists | | green |

*Status: | pending · | green · | red · | flaky*

---

## Wave 0 Requirements

- [x] `tests/test_cache.py` — Unit tests for `build_cache_key()`, `resolve_cache_dir()`, `atomic_save_npz()`, `atomic_save_metadata()`, `load_metadata()`, `save_scaler()`, `load_scaler()`, DatetimeIndex round-trip, metadata version mismatch, schema validation.
- [x] `tests/test_ddp_compliance.py` — DDP smoke test using `mp.spawn` with gloo backend, 2 ranks. Tests: forecasting cache round-trip (rank 0 writes, all ranks read), classification cache round-trip, identical `_train_data_samples` across ranks.
- [x] `tests/test_modules_forecasting.py` — Cache integration tests verify `_full_data_raw` immutability, `_time_index` persistence, `_full_data_scaled` rebuild from raw, no `isinstance` branches.
- [x] `tests/test_modules_base.py::TestPrepareDimensionsCache` — Tests `prepare_dimensions()` reads from `metadata.json` (mocked via `patch`), raises `FileNotFoundError` when metadata missing, raises `ValueError` on version mismatch.
- [x] `tests/conftest.py` — `synthetic_cache_dir` fixture: creates temp dir with pre-populated cache files (npz + metadata.json + scaler.pt) for tests that need cache without calling `prepare_data()`.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 5s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** complete — 2026-05-29

## Validation Audit 2026-05-29

| Metric | Count |
|--------|-------|
| Requirements audited | 16 |
| Gaps found | 0 (behavioral) / 8 (stale commands) |
| Resolved | 8 (command corrections only; all behavior already tested) |
| Escalated | 0 |
| Tests passing | 185 (full phase-07 suite) |

### Command Corrections

All `test_cache.py` automated commands referenced non-existent class names (e.g., `TestMetadataSchema`). Cache tests use standalone functions, not classes. Commands updated to use `-k` keyword selection and direct function references. `TestFullDataSplit` was planned but behavior is covered by `TestETTCacheIntegration`, `TestWeatherCacheIntegration`, and `TestElectricityCacheIntegration`. `DDP-02` static analysis command replaced with `TestIsinstanceBranchElimination` for automated verification.

## Manual-Only Verifications

None — all phase behaviors have automated verification. DDP-02 (isinstance branch elimination) is now tested by `TestIsinstanceBranchElimination` in `test_ddp_compliance.py`.

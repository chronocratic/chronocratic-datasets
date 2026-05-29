---
phase: 07
slug: ddp-compliance
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-29
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
| 07-01 | 01 | 1 | DDP-01 | — | DDP smoke test (gloo, 2 ranks): identical state across ranks after `setup('fit')` | integration | `uv run pytest tests/test_ddp_compliance.py::test_ddp_forecasting_cache_round_trip -x` | W0 | | pending |
| 07-01 | 01 | 1 | DDP-02 | — | No `isinstance` branches on `_full_data` consumers | static analysis | `uv run grep -rn "isinstance.*_full_data" src/ --include="*.py"` (should return 0) | Post-implementation | | pending |
| 07-01 | 01 | 1 | D-03 | — | `prepare_dimensions()` reads metadata without loading arrays | unit | `uv run pytest tests/test_modules_base.py::TestPrepareDimensionsCache -x` | W0 | | pending |
| 07-01 | 01 | 1 | D-07 | — | `prepare_dimensions()` raises when metadata missing | unit | `uv run pytest tests/test_modules_base.py::TestPrepareDimensionsCache -x` | W0 | | pending |
| 07-01 | 01 | 1 | D-08 | — | `metadata.json` schema matches spec | unit | `uv run pytest tests/test_cache.py::TestMetadataSchema -x` | W0 | | pending |
| 07-02 | 02 | 1 | MOD-01 | — | `synthetic_cache_dir` fixture returns Path with npz + metadata.json + scaler.pt | unit | `uv run pytest tests/conftest.py::test_synthetic_cache_dir_fixture -x --tb=short` | W0 | | pending |
| 07-03 | 03 | 2 | D-02 | — | `resolve_cache_dir(None)` returns `~/.cache/tsdatasets/<name>` | unit | `uv run pytest tests/test_cache.py::TestResolveCacheDir -x` | W0 | | pending |
| 07-03 | 03 | 2 | D-02 | — | `resolve_cache_dir(custom_path)` returns custom path | unit | `uv run pytest tests/test_cache.py::TestResolveCacheDir -x` | W0 | | pending |
| 07-03 | 03 | 2 | D-03 | — | `build_cache_key` produces deterministic hybrid key | unit | `uv run pytest tests/test_cache.py::TestBuildCacheKey -x` | W0 | | pending |
| 07-04 | 04 | 3 | D-04 | — | npz round-trip preserves dtype and shape | unit | `uv run pytest tests/test_cache.py::TestAtomicSaveNpz -x` | W0 | | pending |
| 07-04 | 04 | 3 | D-06 | — | torch.save/load sklearn scaler produces identical transform | unit | `uv run pytest tests/test_cache.py::TestScalerPersistence -x` | W0 | | pending |
| 07-05 | 05 | 4 | D-01 | — | `_full_data_raw` immutable after `setup()` load | unit | `uv run pytest tests/test_modules_forecasting.py::TestFullDataSplit -x` | W0 | | pending |
| 07-05 | 05 | 4 | D-04 | — | metadata.json version mismatch raises ValueError | unit | `uv run pytest tests/test_cache.py::TestLoadMetadataVersion -x` | W0 | | pending |
| 07-05 | 05 | 4 | ROADMAP DDP-04 | — | Second `setup()` call produces identical output | unit | `uv run pytest tests/test_modules_forecasting.py::TestSetupIdempotency -x` | Exists (phase 6) | | pending |
| 07-06 | 06 | 5 | MOD-01 | — | Weather/Electricity cache round-trip (prepare_data writes, setup reads) | integration | `uv run pytest tests/test_modules_forecasting.py -x --tb=short -k "Weather or Electricity"` | Exists | | pending |
| 07-06 | 06 | 5 | MOD-05 | — | Classification UCR/UEA cache round-trip | integration | `uv run pytest tests/test_modules_ucr.py tests/test_modules_uea.py -k cache -x` | W0 | | pending |

*Status: | pending · | green · | red · | flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_cache.py` — Unit tests for `build_cache_key()`, `resolve_cache_dir()`, `atomic_save_npz()`, `atomic_save_metadata()`, `load_metadata()`, `save_scaler()`, `load_scaler()`, DatetimeIndex round-trip, metadata version mismatch, schema validation.
- [ ] `tests/test_ddp_compliance.py` — DDP smoke test using `mp.spawn` with gloo backend, 2 ranks. Tests: forecasting cache round-trip (rank 0 writes, all ranks read), classification cache round-trip, identical `_train_data_samples` across ranks.
- [ ] `tests/test_modules_forecasting.py::TestFullDataSplit` — Tests `_full_data_raw` immutability, `_time_index` persistence, `_full_data_scaled` rebuild from raw, no `isinstance` branches.
- [ ] `tests/test_modules_base.py::TestPrepareDimensionsCache` — Tests `prepare_dimensions()` reads from `metadata.json` (mocked via `patch`), raises `FileNotFoundError` when metadata missing, raises `ValueError` on version mismatch.
- [ ] `tests/conftest.py` — ADD `synthetic_cache_dir` fixture: creates temp dir with pre-populated cache files (npz + metadata.json + scaler.pt) for tests that need cache without calling `prepare_data()`.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| No `isinstance` branches on `_full_data` consumers | ROADMAP DDP-02 | Static analysis gate, runs post-implementation | `grep -rn "isinstance.*_full_data" src/ --include="*.py"` — must return 0 matches |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

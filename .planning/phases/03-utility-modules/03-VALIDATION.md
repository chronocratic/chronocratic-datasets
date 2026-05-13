---
phase: 03
slug: utility-modules
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-13
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=8.2 |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] section |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ -x --cov=tscollection.datasets` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_utils_*.py -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -x -q` (full suite including existing tests)
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | UTI-01 | — | Type-checked ARFF DataFrames | unit | `pytest tests/test_utils_arff.py::test_read_arff_as_df -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | UTI-01 | — | dtype transformation map applied | unit | `pytest tests/test_utils_arff.py::test_process_df_according_to_dtypes -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | UTI-02 | — | Scaler returns correct arrays | unit | `pytest tests/test_utils_scaling.py::test_create_data_scaler_regular -x` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | UTI-02 | — | NESTED preserves 3-D shape | unit | `pytest tests/test_utils_scaling.py::test_create_data_scaler_nested -x` | ❌ W0 | ⬜ pending |
| 03-02-03 | 02 | 1 | UTI-02 | — | MULTI_FILES scales list | unit | `pytest tests/test_utils_scaling.py::test_create_data_scaler_multi_files -x` | ❌ W0 | ⬜ pending |
| 03-02-04 | 02 | 1 | UTI-02/D-05 | — | ScalingMethod enum wired | unit | `pytest tests/test_utils_scaling.py::test_scaling_method_enum -x` | ❌ W0 | ⬜ pending |
| 03-03-01 | 03 | 1 | UTI-03 | — | Returns (N, 7) float32 | unit | `pytest tests/test_utils_features.py::test_extract_time_features -x` | ❌ W0 | ⬜ pending |
| 03-03-02 | 03 | 1 | UTI-04 | — | custom_collate_fn pads batch | unit | `pytest tests/test_utils_features.py::test_custom_collate_fn -x` | ❌ W0 | ⬜ pending |
| 03-04-01 | 04 | 1 | UTI-05 | — | All modules export via __all__ | unit | `pytest tests/test_utils_general.py::test_all_exports -x` | ❌ W0 | ⬜ pending |
| 03-04-02 | 04 | 1 | UTI-05/D-07 | — | DataForm enum importable | unit | `pytest tests/test_utils_general.py::test_dataform_enum -x` | ❌ W0 | ⬜ pending |
| 03-04-03 | 04 | 1 | D-04 | — | flatten_list_of_np_arrays | unit | `pytest tests/test_utils_scaling.py::test_flatten_list_of_np_arrays -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_utils_arff.py` — ARFF reading tests (UTI-01)
- [ ] `tests/test_utils_scaling.py` — Scaling tests (UTI-02, D-04, D-05)
- [ ] `tests/test_utils_features.py` — Feature extraction tests (UTI-03, UTI-04)
- [ ] `tests/test_utils_general.py` — General utils & export tests (UTI-04, UTI-05, D-07)
- [ ] Synthetic ARFF fixture in `tests/conftest.py` — temporary ARFF file for `read_arff_as_df` tests
- [ ] `DataForm` enum export test

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | — | — | All phase behaviors have automated verification. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

---
phase: 02
slug: dataset-classes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-11
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml (existing pytest section) |
| **Quick run command** | `uv run pytest tests/ -q --tb=short` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -q --tb=short`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | DST-01 | — | N/A | unit | `uv run pytest tests/test_fixed_dataset.py -q` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | DST-02 | — | N/A | unit | `uv run pytest tests/test_flexible_dataset.py -q` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | DST-03 | — | N/A | unit | `uv run pytest tests/test_fixed_dataset.py -k seq_len -q` | ❌ W0 | ⬜ pending |
| 02-01-04 | 01 | 1 | DST-04 | — | N/A | unit | `uv run pytest tests/test_flexible_dataset.py -k seq_len -q` | ❌ W0 | ⬜ pending |
| 02-01-05 | 01 | 1 | DST-05 | — | N/A | unit | `uv run pytest tests/test_strategies.py -q` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | DST-01 | — | N/A | unit | `uv run pytest tests/test_ucr_dataset.py -q` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 2 | DST-02 | — | N/A | unit | `uv run pytest tests/test_ett_dataset.py -q` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | — | — | N/A | unit | `uv run pytest tests/test_transformations.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_fixed_dataset.py` — stubs for DST-01, DST-03
- [ ] `tests/test_flexible_dataset.py` — stubs for DST-02, DST-04
- [ ] `tests/test_strategies.py` — stubs for DST-05
- [ ] `tests/test_ucr_dataset.py` — stubs for UCR wrapper
- [ ] `tests/test_ett_dataset.py` — stubs for ETT wrapper
- [ ] `tests/test_transformations.py` — stubs for transform utilities
- [ ] `tests/conftest.py` — synthetic numpy/pandas fixtures

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Transform ordering (expand_dims before/after tensor conversion) | D-02 | Runtime type behavior depends on pipeline ordering | Inspect output type of `dataset[0]` — should be `torch.Tensor` not `np.ndarray` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

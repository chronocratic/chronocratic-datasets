---
phase: 04
slug: data-modules
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-13
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/ -x -q` |
| **Full suite command** | `uv run pytest tests/ --cov=tscollection.datasets -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/ -x -q`
- **After every plan wave:** Run `uv run pytest tests/ -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-T1 | 01 | 1 | MOD-01, MOD-05 | — | Path-only params (D-07) | unit | `uv run pytest tests/ -x -q` | ✅ exists | ⬜ pending |
| 04-01-T2 | 01 | 1 | MOD-01, MOD-05 | — | Path existence checks (D-16) | unit | `uv run pytest tests/ -x -q` | ✅ exists | ⬜ pending |
| 04-02-T1 | 02 | 2 | MOD-03, MOD-04 | — | No authorization layer | unit | `uv run pytest tests/ -x -q` | ✅ exists | ⬜ pending |
| 04-02-T2 | 02 | 2 | MOD-01, MOD-06 | — | FileNotFoundError on missing files | unit | `uv run pytest tests/ -x -q` | ✅ exists | ⬜ pending |
| 04-03-T1 | 03 | 3 | MOD-01, MOD-06 | — | FileNotFoundError on missing files | unit | `uv run pytest tests/ -x -q` | ✅ exists | ⬜ pending |
| 04-03-T2 | 03 | 3 | MOD-01, MOD-04, MOD-06 | — | Input validation via Path-only params | unit | `uv run pytest tests/ -x -q` | ✅ exists | ⬜ pending |
| 04-04-T1 | 04 | 4 | MOD-02 | — | No security impact | unit | `uv run pytest tests/ -x -q` | ✅ exists | ⬜ pending |
| 04-04-T2 | 04 | 4 | MOD-02 | — | No security impact | integration | `uv run pytest tests/test_modules.py -v` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_modules.py` — covers MOD-01 through MOD-06
- [ ] `tests/conftest.py` — needs synthetic ARFF fixtures for classification module tests
- [ ] Framework: No additional install needed — pytest already in dev dependencies

*Note: Wave 0 test files are deferred to Phase 5 (Tests). Current plans verify build and type check pass.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Module instantiation with real data files | MOD-01 | Requires real UCR/ETT datasets on disk | Run `python -c "from tscollection.datasets.modules import UCRClassificationDataModule; m = UCRClassificationDataModule(...)"` with real paths |

*All other phase behaviors have automated verification via import checks and type assertions.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

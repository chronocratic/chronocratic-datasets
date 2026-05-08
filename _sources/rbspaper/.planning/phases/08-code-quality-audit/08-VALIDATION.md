---
phase: 08
slug: code-quality-audit
status: draft
nyquist_compliant: false
wave_0_complete: true
created: 2026-05-08
---

# Phase 08 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | None (uses pytest defaults) |
| **Quick run command** | `uv run pytest -x -q` |
| **Full suite command** | `uv run pytest -x` |
| **Test count** | 143 tests |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** `uv run ty check src/rbspaper runners/py/runner.py --quiet` + `uv run ruff check src/rbspaper runners/py/runner.py`
- **After every plan wave:** `uv run pytest -x` (full suite)
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Req | Threat | Secure | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-----|--------|--------|-----------|-------------------|-------------|--------|
| 08-01-01 | 08-01 | 1 | D-01 | — | N/A | Import smoke | `uv run python -c "from src.rbspaper.models.ts2vec.model import TS2Vec"` | Existing | ⬜ pending |
| 08-01-02 | 08-01 | 1 | D-01 | — | N/A | Static | `uv run ty check src/rbspaper/models/encoding.py` | Existing | ⬜ pending |
| 08-01-03 | 08-01 | 1 | Research | — | N/A | Static | `uv run ty check src/rbspaper/pipeline/loggers.py` | Existing | ⬜ pending |
| 08-02-01 | 08-02 | 2 | D-09 | — | N/A | Static | `uv run ty check src/rbspaper/models/cost/model.py` | Existing | ⬜ pending |
| 08-03-01 | 08-03 | 2 | D-10 | — | N/A | Static | `uv run ty check src/rbspaper/models/augmentation/strategies.py` | Existing | ⬜ pending |
| 08-04-01 | 08-04 | 2 | D-04 | — | N/A | Static | `uv run ty check src/rbspaper/pipeline/state.py` | Existing | ⬜ pending |
| 08-05-01 | 08-05 | 2 | D-03 | — | N/A | Static | `uv run ty check src/rbspaper/attacks/` | Existing | ⬜ pending |
| 08-06-01 | 08-06 | 2 | D-02 | — | N/A | Static | `uv run ty check src/rbspaper/pipeline/core.py` | Existing | ⬜ pending |
| 08-07-01 | 08-07 | 2 | D-05 | — | N/A | Static | `uv run ty check runners/py/runner.py` | Existing | ⬜ pending |
| 08-08-T1 | 08-08 | 3 | D-07 | — | N/A | Static | `uv run ruff check --select PLR2004 src/rbspaper/evaluation/classification.py` | Existing | ⬜ pending |
| 08-08-T2 | 08-08 | 3 | D-07 | — | N/A | Static | `uv run ruff check --select BLE001 src/rbspaper/pipeline/loggers.py` | Existing | ⬜ pending |
| 08-08-T3 | 08-08 | 3 | Deferred | — | N/A | Static | `uv run ty check src/rbspaper/models/autotcl/model.py src/rbspaper/models/ts2vec/model.py` | Existing | ⬜ pending |
| 08-08-T4 | 08-08 | 3 | D-12 | — | N/A | Static | `uv run ty check src/rbspaper/data/datasets/strategies.py` | Existing | ⬜ pending |
| 08-09-T1 | 08-09 | 3 | D-06 | — | N/A | File check | `test ! -d src/rbspaper/adapters` | Existing | ⬜ pending |
| 08-09-T2 | 08-09 | 3 | All | — | N/A | Integration | `uv run pytest -x` | Existing | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

None. Existing test infrastructure covers all phase requirements. This phase is static analysis cleanup, not behavioral change.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Import chain after lazy import removal | D-01 | No existing test for direct imports | `uv run python -c "from src.rbspaper.models.encoding import encode_data"` and verify no import errors |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

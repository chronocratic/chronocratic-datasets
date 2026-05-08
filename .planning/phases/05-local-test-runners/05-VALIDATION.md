---
phase: 5
slug: local-test-runners
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-07
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (via `uv run pytest`) |
| **Config file** | `pyproject.toml` inline (`[tool.pytest.ini_options]`) |
| **Quick run command** | `uv run pytest test/test_runner_cli_args.py test/test_runner_logging.py -x` |
| **Full suite command** | `uv run pytest test/ -x` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest test/test_runner_cli_args.py test/test_runner_logging.py -x`
- **After every plan wave:** Run `uv run pytest test/ -x`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | A-01 | — | `__init__.py` makes runners a proper package | unit | `uv run pytest test/test_runner_logging.py -x` | Yes | ⬜ pending |
| 05-01-02 | 01 | 1 | A-06 | — | `runner.py` logging conversion | unit | `uv run pytest test/test_runner_logging.py -x` | Yes | ⬜ pending |
| 05-02-01 | 02 | 2 | A-11 | T-5-01 | config.sh.example secure defaults | shell | `bash runners/bash/local_single.sh --help` | Wave 0 | ⬜ pending |
| 05-02-02 | 02 | 2 | A-03, A-04, A-05, A-07 | T-5-02 | Script input validation + PATH safety | shell | `bash runners/bash/local_single.sh ts2vec 0 --dry_run` | Wave 0 | ⬜ pending |
| 05-03-01 | 03 | 3 | A-08, A-09, A-10 | T-5-03 | Batch sequential execution | shell | `bash runners/bash/local_batch.sh ts2vec all --dry_run` | Wave 0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Shell script tests — no existing infrastructure for testing Bash scripts. Add minimal test harness or manual verification checklist.
- [ ] `test_runner_logging.py` — verify existing imports still work after adding `runners/__init__.py`.
- [ ] `test_runner_cli_args.py` — verify `--dry_run` flag added for testable bash scripts.

*Existing pytest infrastructure covers Python tasks (05-01). Wave 0 gap: shell script testing.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `--fraction` sampling | A-10 | Shell scripts lack test fixtures; validates dataset expansion | Run `bash runners/bash/local_batch.sh ts2vec all --fraction 0.25 --dry_run` and inspect which datasets are selected |
| Config auto-creation | A-11 | File system state; tests `config.sh` copy from `.example` | Remove `config.sh`, run `local_single.sh ts2vec 0 --dry_run`, verify `config.sh` created |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending 2026-05-07

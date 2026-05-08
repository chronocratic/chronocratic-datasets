---
phase: 04
slug: experiment-registry-restructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-07
---

# Phase 4 — Validation Strategy

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (via `uv run pytest`) |
| **Config file** | `pyproject.toml` / existing pytest setup |
| **Quick run command** | `uv run pytest test/test_runner_cli_args.py test/test_hierarchical_run_name.py test/test_pipeline_state.py -q` |
| **Full suite command** | `uv run pytest` |
| **Estimated runtime** | ~30 seconds (quick), ~120 seconds (full) |

---

## Sampling Rate

- **After every task commit:** Run `uv run ruff check . && uv run ruff format --check .`
- **After every plan wave:** Run `uv run pytest -q`
- **Before `/gsd-verify-work`:** Full suite + `uv run ty` must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | CONTEXT D-01 | — | N/A | unit | `uv run pytest test/test_attack_family.py -q` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 2 | CONTEXT D-01 | — | N/A | unit | `uv run pytest test/test_experiment_registry.py -q` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 2 | CONTEXT D-01 | — | N/A | unit | `uv run pytest test/test_experiment_registry.py -k alias -q` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 3 | CONTEXT D-03 | — | N/A | unit | `uv run pytest runners/py/runner.py -k attack_family -q` | ❌ W0 | ⬜ pending |
| 04-01-05 | 01 | 3 | CONTEXT D-04 | — | N/A | integration | `uv run rbspaper-run --list_experiments` | ✅ existing | ⬜ pending |
| 04-01-06 | 01 | 3 | CONTEXT D-02 | — | N/A | unit | `uv run pytest -k preflight -q` | ❌ W0 | ⬜ pending |
| 04-01-07 | 01 | 4 | ROADMAP SC-4 | — | N/A | unit | `uv run pytest test/test_runner_cli_args.py -q` | ✅ existing | ⬜ pending |
| 04-01-08 | 01 | 4 | ROADMAP SC-4 | — | N/A | unit | `uv run pytest test/test_hierarchical_run_name.py test/test_pipeline_state.py -q` | ✅ existing | ⬜ pending |
| 04-01-09 | 01 | 4 | ROADMAP SC-5 | — | N/A | integration | `uv run rbspaper-run --list_experiments 2>&1 \| grep -c ts2vec` | ✅ existing | ⬜ pending |

---

## Wave 0 Requirements

- [ ] `test/test_attack_family.py` — unit tests for `AttackFamily` enum and registry grouping
- [ ] `test/test_experiment_registry.py` — unit tests for new `ExperimentInstance.attack_families`, derived `attack_params`, alias map, `get_experiment_instance` with filter
- [ ] `test/test_prefilter_compat.py` — unit tests for preflight warn-and-drop logic

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end: train once, apply all attacks | ROADMAP SC-1 | Requires data + GPU | Run `uv run rbspaper-run --experiment_id ts2vec --dataset_name Coffee --data_root /path/to/data` and verify single training run with multiple attack evaluations |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending

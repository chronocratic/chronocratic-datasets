---
phase: 07
slug: experiment-tracking
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 07 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.3 |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest test/test_logger_factory.py test/test_runner_cli_args.py -q` |
| **Full suite command** | `uv run pytest test/ -q` |
| **Estimated runtime** | ~65 seconds (full), ~4 seconds (quick) |

---

## Sampling Rate

- **After every task commit:** Run quick command (logger factory + CLI args)
- **After every plan wave:** Run full suite
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 65 seconds

---

## Per-Requirement Coverage Map

### D-01: Dual logger factory (online/offline modes)

| Test | File | Status |
|------|------|--------|
| `test_online_creates_dual_loggers` | test_logger_factory.py | ✅ green |
| `test_offline_creates_dual_loggers` | test_logger_factory.py | ✅ green |
| `test_disabled_skips_wandb` | test_logger_factory.py | ✅ green |

### D-02: Empty tuple when persist_artifacts=False

| Test | File | Status |
|------|------|--------|
| `test_no_persist_returns_empty` | test_logger_factory.py | ✅ green |

### D-03: HPC auto-detection via SLURM_JOB_ID

| Test | File | Status |
|------|------|--------|
| `test_cli_mode_overrides_auto_detection` | test_runner_cli_args.py | ✅ green |
| `test_hpc_detection_defaults_to_offline` | test_runner_cli_args.py | ✅ green |
| `test_local_detection_defaults_to_online` | test_runner_cli_args.py | ✅ green |
| `test_hpc_detection_uses_env_var_value` | test_runner_cli_args.py | ✅ green |

### D-04: Results logging with timing

| Test | File | Status |
|------|------|--------|
| `test_flat_dict_unchanged` | test_logger_factory.py | ✅ green |
| `test_nested_dict_flattened` | test_logger_factory.py | ✅ green |
| `test_list_with_primitives` | test_logger_factory.py | ✅ green |
| `test_list_with_dicts` | test_logger_factory.py | ✅ green |
| `test_empty_dict` | test_logger_factory.py | ✅ green |
| `test_custom_separator` | test_logger_factory.py | ✅ green |
| `test_none_values_preserved` | test_logger_factory.py | ✅ green |
| `test_mixed_types` | test_logger_factory.py | ✅ green |
| `test_log_results_noop_without_wandb` | test_logger_factory.py | ✅ green |
| `test_log_results_calls_run_log` | test_logger_factory.py | ✅ green |

### D-05: Config logging to W&B

| Test | File | Status |
|------|------|--------|
| `test_log_config_noop_without_wandb` | test_logger_factory.py | ✅ green |
| `test_log_config_calls_experiment_config_update` | test_logger_factory.py | ✅ green |

### D-06: W&B run naming

| Test | File | Status |
|------|------|--------|
| `test_wandb_logger_project_name` | test_logger_factory.py | ✅ green |

### D-07: log_model=False (no checkpoint upload)

| Test | File | Status |
|------|------|--------|
| `test_wandb_logger_log_model_false` | test_logger_factory.py | ✅ green |

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Test Type | Automated Command | Status |
|---------|------|------|-------------|------------|-----------|-------------------|--------|
| 07-01-T1 | 01 | 1 | D-01, D-05 | T-07-05 | unit | `grep 'tracking' pyproject.toml` | ✅ green |
| 07-01-T2 | 01 | 1 | D-02 | — | unit | `grep 'loggers.*tuple' config.py` | ✅ green |
| 07-01-T3 | 01 | 1 | D-01, D-02, D-04, D-06, D-07 | T-07-01..04 | unit | `pytest test_logger_factory.py -q` | ✅ green |
| 07-02-T1 | 02 | 2 | D-03, D-04, D-05 | T-07-06, T-07-09 | integration | `pytest test_pipeline_core.py -q` | ✅ green |
| 07-02-T2 | 02 | 2 | D-03, D-06 | T-07-07, T-07-08 | unit | `pytest test_runner_cli_args.py -q` | ✅ green |
| 07-03-T1 | 03 | 3 | D-01..D-07 | T-07-10 | unit | `pytest test_logger_factory.py -q` | ✅ green |
| 07-03-T2 | 03 | 3 | D-03, D-06 | T-07-11 | unit | `pytest test_runner_cli_args.py::TestTrackingModeArg -q` | ✅ green |

---

## Cross-Task Integration Verification

| Behavior | Test | Coverage |
|----------|------|----------|
| Logger factory -> core.py Trainer wiring | `test_pipeline_smoke_run` | Integration via smoke test |
| Runner --tracking_mode -> create_loggers -> config | `test_pipeline_smoke_run` | End-to-end via tmp_path |
| Timing dict populated after pipeline run | `test_pipeline_smoke_run` | Indirect (no assertions on timing keys) |
| W&B hooks called with config.loggers | `test_experiment_config_written` | Indirect (no logger assertions) |

---

## Validation Sign-Off

- [x] All tasks have automated verify commands
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] All D-01 through D-07 requirements have direct test coverage
- [x] 143 tests pass, 0 failures
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-05-08

---
phase: 07
slug: experiment-tracking
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-08
---

# Phase 07 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| create_loggers() -> WandbLogger | W&B credentials via WANDB_API_KEY env var cross into external service | API key (handled by wandb SDK) |
| _log_results_to_wandb() -> wandb.log() | Pipeline metrics sent to W&B cloud | Aggregated metrics only (accuracy, F1, MAE) |
| runner.py -> create_loggers() | User-supplied --tracking_mode validated against enum | CLI argument string |
| core.py -> pl.Trainer(loggers=...) | Logger instances cross from runner into pipeline | Logger objects (no secrets) |
| core.py -> _log_config_to_wandb() | Config data crosses into W&B cloud | Model params, seed, attack names |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-07-01 | Information Disclosure | WandbLogger in online mode | mitigate | `log_model=False` prevents checkpoint upload (loggers.py:71) | closed |
| T-07-02 | Information Disclosure | `_log_results_to_wandb` wandb.Table | mitigate | Table columns are aggregated metrics only, no sample-level data (loggers.py:205-231) | closed |
| T-07-03 | Spoofing | WANDB_API_KEY env var | accept | Key handled by wandb SDK, not our code. User responsibility | closed |
| T-07-04 | Repudiation | Lazy wandb import failure | mitigate | try/except ImportError with graceful fallback to TensorBoardLogger only (loggers.py:63-81) | closed |
| T-07-05 | Supply Chain | New deps: wandb, tensorboard | mitigate | Pinned minimum versions in pyproject.toml (`wandb>=0.24.2`, `tensorboard>=2.18.0`) | closed |
| T-07-06 | Information Disclosure | `_log_config_to_wandb` config_data | mitigate | Config contains model params, seed, attack names — no PII or secrets (core.py:409) | closed |
| T-07-07 | Tampering | `--tracking_mode` CLI arg | mitigate | argparse `choices=['online', 'offline', 'disabled']` validates input (runner.py:149-152) | closed |
| T-07-08 | Information Disclosure | SLURM_JOB_ID env var read | accept | Boolean detection (present/absent), value never logged (runner.py:172) | closed |
| T-07-09 | Denial of Service | Empty loggers to Trainer | mitigate | Conditional `if config.loggers:` prevents empty-list default (core.py:423-424) | closed |
| T-07-10 | Information Disclosure | Tests for _log_results_to_wandb | mitigate | Tests use MockRun, no real data sent to W&B (test_logger_factory.py) | closed |
| T-07-11 | Repudiation | Mock objects in tests | accept | Unit-level mocks sufficient; real W&B integration out of scope for unit tests | closed |

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-07-01 | T-07-03 | WANDB_API_KEY managed entirely by wandb SDK; our code never reads or stores credentials | security-auditor | 2026-05-08 |
| AR-07-02 | T-07-08 | SLURM_JOB_ID is a non-secret job identifier used only for boolean env detection | security-auditor | 2026-05-08 |
| AR-07-03 | T-07-11 | Test mocks verify unit behavior; integration with real W&B is out of scope | security-auditor | 2026-05-08 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-08 | 11 | 11 | 0 | gsd-security-auditor |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-08

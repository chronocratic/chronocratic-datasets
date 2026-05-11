# Testing

Last mapped: 2026-05-08

## Framework

- **pytest** >=8.2 with pytest-cov >=5.0
- Test paths configured in `pyproject.toml`: `testpaths = ["test"]`
- Python path includes `.`, `src`, and `experiment_instances` (rbspaper)

## Test Locations

| Project | Test Directory | Files |
|---------|---------------|-------|
| rbspaper | `_sources/rbspaper/test/` | ~18 test files |
| autotsrc | None | No test directory present |
| Root | None | No test directory present |

## rbspaper Test Coverage

Tests cover pipeline internals, attack system, configuration, and utilities:

| File | Scope |
|------|-------|
| `test_pipeline_core.py` | Pipeline orchestration, retry logic, checkpoint recovery |
| `test_pipeline_state.py` | State machine, config hashing |
| `test_attacks_registry.py` | Attack method → backend mappings |
| `test_attacks_functional.py` | Attack execution end-to-end |
| `test_attacks_batch.py` | Batch attack processing |
| `test_attack_family.py` | AttackFamily enum behavior |
| `test_evaluation_bugs.py` | Evaluation edge cases |
| `test_experiment_registry.py` | Experiment preset registration |
| `test_runner_cli_args.py` | CLI argument parsing |
| `test_runner_logging.py` | Logger configuration |
| `test_logger_factory.py` | `create_loggers()` factory |
| `test_preflight_compat.py` | Preflight validation compatibility |
| `test_hierarchical_run_name.py` | Run name construction |

## Test Patterns

- **Smoke tests:** `test_pipeline_core.py` constructs minimal fake models and datasets to validate pipeline flow
- **Unit tests:** Test individual functions with parameterized fixtures — `@pytest.mark.parametrize`
- **Assertions:** Standard `assert` with pytest magic methods
- **No mocking framework:** Tests use real objects (small tensors, minimal configs) rather than `unittest.mock`
- **tenacity integration:** Tests verify retry behavior — check for `RetryError`

## Test Configuration in ruff

Extensive per-file ignores for tests:

```
"test/*.py" = ["D", "E501", "PLR2004", "S101", "N812", "ANN001", "ANN003", ...]
```

Key relaxations:
- `D` — No docstring requirements
- `S101` — `assert` statements allowed
- `PLR2004` — Magic numbers allowed
- `ANN001/ANN003` — Type hints on parameters/returns relaxed
- `ARG001/ARG002/ARG005` — Unused argument warnings suppressed

## Gaps

- **autotsrc has no tests** — dataset and data module classes are untested
- **Root package has tests** — `tests/test_package.py` validates imports, version, enum exports, __init__.py hierarchy, and __all__ declarations (6 tests, Phase 1)
- **No integration tests** — rbspaper tests are unit-level; no full pipeline E2E with real data
- **No CI configuration** — no GitHub Actions, GitLab CI, or similar

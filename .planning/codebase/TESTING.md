# Testing Patterns

**Analysis Date:** 2026-05-05

## Test Framework

**Runner:**
- `pytest` (version >= 8.2)
- Config: Inline in `pyproject.toml` under `[tool.pytest.ini_options]`
  ```ini
  [tool.pytest.ini_options]
  pythonpath = [".", "src", "experiment_instances"]
  testpaths = ["test"]
  ```
- No separate `conftest.py` file exists

**Assertion Library:**
- Built-in `assert` statements only (no `pytest.assert` helpers)
- `torch.allclose()` for tensor comparison
- `torch.equal()` for exact tensor comparison

**Coverage Tool:**
- `pytest-cov` (version >= 5.0) available as dev dependency

**Run Commands:**
```bash
uv run pytest                        # Run all tests
uv run pytest -v                     # Verbose output
uv run pytest --cov=src              # With coverage
uv run pytest test/test_attacks_registry.py  # Single file
```

**Important:** One test file (`test/test_pipeline_core.py`) currently fails to collect due to a circular import in `src/rbspaper/models/augmentation/strategies.py`. The working test surface is the attacks subpackage only.

## Test File Organization

**Location:**
- Dedicated `test/` directory at project root (not co-located with source)
- No nested test directories

**Naming:**
- `test_<module_name>.py` pattern
- Current test files:
  - `test/test_attacks_batch.py` — Tests for batched attack helpers
  - `test/test_attacks_functional.py` — Tests for function-based attack wrappers
  - `test/test_attacks_registry.py` — Tests for attack registry mappings
  - `test/test_pipeline_core.py` — Smoke tests for pipeline orchestration (currently fails to collect)

**Structure:**
```
test/
├── test_attacks_batch.py       # 3 tests — batch attack utilities
├── test_attacks_functional.py  # 2 tests — functional attack wrappers
├── test_attacks_registry.py    # 4 tests — registry lookups
└── test_pipeline_core.py       # 8 tests — pipeline integration (broken)
```

## Test Structure

**Suite Organization:**
Tests are flat functions (no `unittest.TestCase` or `pytest.Class`). Each test file has:
1. Module-level docstring describing the test scope
2. Optional `# ruff: noqa` header for rule suppression
3. `from __future__ import annotations` (in most files)
4. Import section
5. Test functions

**Pattern:**
```python
"""Tests for attack registry mappings."""

# ruff: noqa: S101

from src.rbspaper.attacks.enums import AttackBackend, AttackMethod
from src.rbspaper.attacks.registry import get_default_backend, list_supported_attacks
from src.rbspaper.enums.general import TimeSeriesDownstreamTask


def test_registry_contains_required_attack_set() -> None:
    """Required baseline attacks should be present in the registry."""
    methods = set(list_supported_attacks(task=TimeSeriesDownstreamTask.CLASSIFICATION))
    required = {
        AttackMethod.LBFGS,
        AttackMethod.FGSM,
        AttackMethod.DEEPFOOL,
        ...
    }
    assert required.issubset(methods)
```

**Test Function Naming:**
- `test_<description>` with full sentence descriptions
- Examples: `test_registry_contains_required_attack_set`, `test_pipeline_rejects_query_budget_for_white_box_attack`, `test_batched_attack_with_metadata`

**Return Type:**
- All test functions declare `-> None` return type

## Mocking

**Framework:** `pytest.monkeypatch` fixture only

**Patterns:**
- `monkeypatch.setattr()` used to replace functions at module level with fakes:
  ```python
  def test_pipeline_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
      monkeypatch.setattr('src.rbspaper.pipeline.core.encode_data', _fake_encode_data)
      monkeypatch.setattr('src.rbspaper.pipeline.core.execute_attack', _fake_execute_attack)
      monkeypatch.setattr('src.rbspaper.pipeline.core.evaluate', _fake_evaluate)
  ```
- `monkeypatch.setattr()` used to replace internal functions within same module:
  ```python
  def test_fgsm_wrapper_calls_backend(monkeypatch) -> None:
      monkeypatch.setattr(functional_attacks, "run_attack_backend", _fake_run_attack_backend)
  ```

**Fake Functions Pattern:**
Minimal stub functions that validate inputs and return controlled outputs:
```python
def _fake_encode_data(**kwargs: object) -> torch.Tensor:
    data = kwargs['data']
    if not isinstance(data, torch.Tensor):
        message = 'Expected tensor inputs for encoding stub.'
        raise TypeError(message)
    return torch.mean(data, dim=1)
```

**What to Mock:**
- External dependencies: attack backends (`run_attack_backend`), encoding (`encode_data`), evaluation (`evaluate`)
- I/O operations: nothing yet (no file system mocking pattern established)

**What NOT to Mock:**
- Pure functions and registry lookups are tested directly without mocking
- Data structures (dataclasses, enums) are tested as-is

**No `unittest.mock` usage.** The project uses `monkeypatch` exclusively.

## Fixtures and Factories

**Test Data:**
- Tensors created inline with `torch.zeros()` or `torch.randn()`:
  ```python
  inputs = torch.zeros(size=(8, 16, 1), dtype=torch.float32)
  supervision = torch.zeros(size=(8,), dtype=torch.long)
  ```
- Small datasets created with `TensorDataset`:
  ```python
  dataset = TensorDataset(inputs, supervision)
  dataloader = DataLoader(dataset=dataset, batch_size=3, shuffle=False)
  ```

**Mini-models for Testing:**
Inline `nn.Module` subclasses used as test doubles:
```python
class _SimplePredictionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=4, out_features=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = torch.mean(input=inputs, dim=1)
        return self.linear(pooled)
```
Named with leading underscore to indicate test-local scope.

**Factory Functions for Config:**
Helper functions build common test configurations:
```python
def _minimal_no_analysis() -> RepresentationAnalysisConfig:
    return RepresentationAnalysisConfig(
        enable_linear_separability=False,
        enable_geometry=False,
        enable_shift=False,
        enable_low_dim_artifacts=False,
    )
```

**Module-level Constants:**
Named constants at module level for expected counts:
```python
CASE_COUNT = 4
EXPECTED_CLASSIFICATION_METRICS = 2
EXPECTED_SHARED_INPUT_METRICS = 2
```

## Coverage

**Requirements:** No enforced coverage threshold configured in pyproject.toml

**View Coverage:**
```bash
uv run pytest --cov=src --cov-report=term-missing
```

**Current Coverage Gaps:**
- `src/rbspaper/pipeline/` — No passing tests due to circular import
- `src/rbspaper/models/` — No tests
- `src/rbspaper/data/` — No tests
- `src/rbspaper/evaluation/` — No tests
- `src/rbspaper/adapters/` — No tests
- `experiment_instances/` — No tests
- `runners/py/runner.py` — No tests

Only the `src/rbspaper/attacks/` subpackage has test coverage.

## Test Types

**Unit Tests:**
- Scope: Pure functions, registry lookups, batch processing utilities
- Approach: Direct function calls with inline test data
- Examples: `test_registry_contains_required_attack_set`, `test_batched_attack_without_metadata`

**Integration Tests:**
- Scope: Pipeline orchestration with mocked dependencies
- Approach: Full `run_experiment_pipeline()` call with monkeypatched internals
- Examples: `test_pipeline_smoke_run`, `test_pipeline_shared_input_scope`

**E2E Tests:**
- Not used. No end-to-end tests that train real models or run real attacks.

## Common Patterns

**Async Testing:**
- Not applicable. All code is synchronous.

**Error Testing:**
- `pytest.raises()` with `match=` for regex pattern matching on error messages:
  ```python
  def test_pipeline_rejects_query_budget_for_white_box_attack(tmp_path: Path) -> None:
      with pytest.raises(ValueError, match='query budget'):
          run_experiment_pipeline(config=config)
  ```
- Validates that specific configuration combinations raise errors at preflight

**Counting Side Effects:**
- Closure-based counter pattern for verifying call counts:
  ```python
  attack_call_count = 0
  def _counting_attack(**kwargs: object) -> tuple[torch.Tensor, AttackExecutionMetadata]:
      nonlocal attack_call_count
      attack_call_count += 1
      return _fake_execute_attack(**kwargs)
  # ...
  assert attack_call_count == 1
  ```

**Tensor Comparison:**
- `torch.allclose()` for approximate equality:
  ```python
  assert torch.allclose(adversarial, inputs + 0.25)
  ```
- `torch.equal()` for exact equality:
  ```python
  assert torch.equal(attacked_supervision, supervision)
  ```

**tmp_path Fixture:**
- Used for isolated file system operations:
  ```python
  def test_pipeline_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
      config = ExperimentPipelineConfig(
          artifacts=PipelineArtifactConfig(output_dir=tmp_path, run_name='smoke'),
          ...
      )
      assert (tmp_path / 'smoke' / 'results_summary.json').exists()
  ```

## Test File Ruff Configuration

Test files have special ruff ignores configured:
```toml
[lint.per-file-ignores]
"tests/**/*.py" = ["D", "PLR2004", "S101"]
```
- `D` — Docstring requirements waived
- `PLR2004` — Magic number checks waived
- `S101` — Assertion usage allowed

However, some test files add their own `# ruff: noqa` header for broader suppression:
```python
# ruff: noqa: D103, S101, PLR2004
```

---

*Testing analysis: 2026-05-05*

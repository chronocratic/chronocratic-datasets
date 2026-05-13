# Phase 5: Tests - Research

**Researched:** 2026-05-13
**Domain:** pytest-based testing for PyTorch/Lightning time series data modules
**Confidence:** HIGH

## Summary

This phase adds integration and unit tests to bring coverage from 87% (163 tests) to 92%+. The dominant gap is `forecasting.py` at 49% coverage -- specifically the `setup()` method body (lines 146-193) which exercises sklearn scaling, time feature extraction, data transformation, and train/valid/test splitting. The CONTEXT.md decisions prescribe three test types: (1) ETT golden-path integration tests using `tmp_path` CSV fixtures, (2) `setup()` edge-case unit tests via pre-populated internal state on `ETTDataModule`, and (3) dataloader smoke tests on `WeatherModule` and `ElectricityLoadModule` exercising the fractional-split path.

Secondary gaps include `ett.py` (67%), `weather.py` (73%), and `transformations.py` (75%). These are all downstream of the forecasting `setup()` flow -- once the integration tests exercise the full `prepare_data() -> setup('fit') -> train_dataloader()` pipeline, coverage on `ett.py` and `weather.py` will rise substantially. `transformations.py` needs a few targeted unit tests for error paths (TypeError, ValueError, axis out of range).

**Primary recommendation:** Write 3-5 integration tests on ETTDataModule with synthetic CSV fixtures (covering the full `prepare_data -> setup -> dataloader` flow), 3-4 edge-case unit tests on `setup()` (no-DatetimeIndex path, `scale_data=False`, `ScalingMethod.STANDARD`), and 2-3 dataloader smoke tests on Weather/Electricity modules. Add missing `transformations.py` error-path tests. Target: 92% total, no module below 85%.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Integration-style `prepare_data() -> setup('fit')` tests on ETTDataModule as the golden path -- covers the full forecasting pipeline in one shot. Uses real CSV fixtures (synthetic, in-memory via `tmp_path`). This alone exercises `_set_data_slices`, `_prepare_data_scaler`, `_transform_data`, `_calculate_num_features`, `_split_data`, and dataloader construction.
- **D-02:** Unit tests for forecasting `setup()` edge cases -- no DatetimeIndex (pure numpy `_full_data`), `ScalingMethod.STANDARD`, and `scale_data=False`. Pre-populate internal state and call `setup()` directly on a concrete subclass.
- **D-03:** Dataloader smoke tests on WeatherModule and ElectricityLoadModule -- verify `prepare_data() -> setup('fit') -> train_dataloader()` returns DataLoader with correct batch shapes. Tests the fractional-split path (60/20/20) vs ETT's intrinsic 16/4/4.
- **D-04:** No orphaned tests to remove -- audit confirmed zero references to archived v2 modules (pydantic, factory, download, caching) in the 19 existing test files. Test suite is clean.
- **D-05:** Keep existing test structure -- 19 files with 163 tests, all unit-level with synthetic numpy/pandas fixtures. Do not reorganize into `tests/datasets/`, `tests/modules/`, `tests/utils/` subdirs; flat layout is established and works.
- **D-06:** Target 92% total coverage with no module below 85% -- focus on `forecasting.py` (49%), `ett.py` (67%), `weather.py` (73%), `transformations.py` (75%).
- **D-07:** Use `tmp_path` for CSV fixtures -- synthetic files written to pytest temp directory, no real data downloads. Existing `conftest.py` fixtures (`synthetic_classification_df`, `synthetic_forecast_data`, `synthetic_multivariate_data`) remain for dataset-level tests.
- **D-08:** Add `synthetic_forecasting_csv` fixture to `test_modules_forecasting.py` -- DataFrame with DatetimeIndex and 2-3 feature columns, covering both with/without time index paths.
- **D-09:** Keep existing `TestForecastingModulesUseTensorDataset` source-inspection tests (lines 325-347 in `test_modules_forecasting.py`). They are lightweight and verify D-13 decision.
- **D-10:** No CI configuration -- tests run locally with `uv run pytest`. CI is out of scope for Phase 5.
- **D-11:** Use `--cov=tscollection.datasets --cov-report=term-missing` for verification. Do not add coverage gate (fail below X%) -- keep it advisory for now.

### Claude's Discretion
- Exact number of integration tests (likely 3-5: ETT golden path, Weather/Electricity dataloader smokes, edge cases).
- How to structure the `setup()` edge case tests -- use ETTDataModule with pre-set slices, or create a minimal test subclass.
- Specific `transformations.py` lines to cover -- prioritize those called by the full pipeline over isolated function tests.

### Deferred Ideas (OUT OF SCOPE)
- **CI configuration** -- GitHub Actions for automated test runs is v2/v3 scope
- **Coverage gate** -- Fail below X% threshold is future concern; keep advisory for now
- **Real data fixtures** -- Testing with actual ETT/UCR data files requires download pipeline (v2)
- **Test reorganization** -- Moving tests into `tests/datasets/`, `tests/modules/` subdirectories not needed with current flat layout

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TST-01 | Dataset classes yield correct shapes and types | Flexible/fixed dataset tests verify shapes via synthetic fixtures. Forecasting integration tests will exercise FlexibleTimeSeriesDatasetSingleFile indirectly through ETT module dataloaders. |
| TST-02 | Module properties return expected values after prepare_data | ETT golden-path tests verify `_train_data_samples`, `_valid_data_samples`, `_test_data_samples` shapes and `num_features` after full pipeline. Weather/Electricity smoke tests verify fractional split properties. |
| TST-03 | Utility functions (scaling, arff, features) produce correct output | Existing tests cover `scaling.py` (93%), `arff.py` (100%), `features.py` (100%), `general.py` (100%). Gap is `transformations.py` (75%) -- needs error-path tests for `convert_numpy_to_tensor`, `expand_data_dimensionality`. |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Test execution (pytest) | Local CLI | — | Tests run via `uv run pytest`, no server component |
| Coverage measurement (pytest-cov) | Local CLI | — | Advisory reporting only, no CI gate |
| Synthetic fixtures (tmp_path) | Test process | — | In-memory temp files, no external I/O |
| Data pipeline verification | Test process | — | Tests exercise prepare_data -> setup -> dataloader within a single process |
| Sklearn scaler behavior | Under test module | — | Scalers are imported and used by forecasting.py; tests verify integration, not scaler internals |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest 9.0.3 | 9.0.3 | Test runner, fixtures, parametrization | Project-standard test framework; configured in pyproject.toml |
| pytest-cov | 5.0+ (coverage 7.13.5) | Coverage measurement | Required for D-06 target verification |
| torch 2.8.0 | 2.8.0 | TensorDataset, DataLoader, default_collate | Under test -- dataloader tests exercise these classes |
| lightning 2.5.6 | 2.5.6 | LightningDataModule base class | Under test -- setup(), dataloader lifecycle |
| scikit-learn 1.8.0 | 1.8.0 | MinMaxScaler, StandardScaler | Used by forecasting.py setup() |
| numpy 2.4.4 | 2.4.4 | Synthetic data arrays | All test fixtures use numpy |
| pandas 3.0.2 | 3.0.2 | DataFrame fixtures, CSV I/O | Forecasting modules parse CSVs |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| unittest.mock (stdlib) | — | patch() for DataLoader, isolated tests | When testing base dataloader args without full pipeline |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pytest class-based tests | Function-level tests | Classes provide shared fixtures and grouping; existing pattern uses both. Class-based is established for module tests. |
| tmp_path CSV fixtures | Binary fixtures (io.BytesIO) | tmp_path produces real Path objects matching production types. BytesIO would require additional wrapping. |

**Installation:**
```bash
uv run pytest --cov=tscollection.datasets --cov-report=term-missing
```

**Version verification:** All versions confirmed via `uv run python -c "import X; print(X.__version__)"` on this environment.

## Architecture Patterns

### System Architecture Diagram

```
Synthetic CSV (tmp_path)
        |
        v
ETTDataModule.prepare_data()
  |-- pd.read_csv() -> DataFrame with DatetimeIndex
  |-- _post_prepare_data() -> _set_data_slices()
        |
        v
BaseForecastingTimeSeriesDataModule.setup('fit')
  |-- extract_time_features() if DatetimeIndex present
  |-- _prepare_data_scaler() -> MinMaxScaler/StandardScaler
  |-- scaler.fit(train_slice), scaler.transform(full)
  |-- _transform_data() -> expand_dims
  |-- _calculate_num_features()
  |-- _split_data() -> _train/valid/test_data_samples
        |
        v
train_dataloader() / val_dataloader() / test_dataloader()
  |-- TensorDataset(torch.from_numpy(samples))
  |-- _process_train_dataloader() -> DataLoader
        |
        v
Assertions: shapes, types, DataLoader properties, batch contents
```

### Recommended Project Structure

Tests remain in the flat `tests/` directory per D-05. New tests go in existing files:

```
tests/
├── conftest.py                          # Shared fixtures (no change needed)
├── test_modules_forecasting.py          # ADD: integration + edge-case tests here
├── test_transformations.py              # ADD: error-path tests here
├── test_modules_classification_forecasting.py  # No change (keep existing)
└── ... (other existing test files)      # No change
```

### Pattern 1: Golden-Path Integration Test

**What:** Write a CSV fixture via `tmp_path`, instantiate the module, call `prepare_data()` then `setup('fit')`, verify internal state and dataloader outputs.

**When to use:** Testing the full forecasting pipeline (forecasting.py setup body).

**Example:**
```python
@pytest.fixture
def ett_csv_file(tmp_path: Path) -> Path:
    """Create a CSV with 500+ rows and DatetimeIndex for ETT golden path."""
    csv_file = tmp_path / 'ETTm1.csv'
    dates = pd.date_range('2017-01-01', periods=500, freq='h')
    df = pd.DataFrame({
        'date': dates,
        'HUFL': np.random.randn(500),
        'HT': np.random.randn(500),
        'OT': np.random.randn(500),
        'Wsp': np.random.randn(500),
    })
    df.to_csv(csv_file, index=False)
    return csv_file

def test_ett_golden_path_integration(ett_csv_file: Path) -> None:
    from tscollection.datasets.modules.ett import ETTDataModule

    module = ETTDataModule(
        dataset_file_path=ett_csv_file,
        variant='ETTh1',
        seq_len=96,
        batch_size=16,
        scale_data=True,
        data_scaling_method=ScalingMethod.MINMAX,
    )
    module.prepare_data()
    module.setup(stage='fit')

    # Verify internal state
    assert module._train_data_samples is not None
    assert module._valid_data_samples is not None
    assert module._test_data_samples is not None
    assert module.num_features is not None
    assert module.num_time_series_features is not None

    # Verify dataloaders
    train_dl = module.train_dataloader()
    assert isinstance(train_dl, DataLoader)
    batch = next(iter(train_dl))
    # batch shape: (batch_size, seq_len, features)
```

Source: Pattern derived from existing `test_modules_forecasting.py` fixtures combined with the full `prepare_data -> setup` lifecycle.

### Pattern 2: Pre-Populated Setup Edge Cases

**What:** Pre-set `_full_data`, `_train_slice`, etc. on a concrete module, then call `setup()` directly to test specific branches (no DatetimeIndex, `scale_data=False`, STANDARD scaler).

**When to use:** Testing setup() branches that the golden path doesn't cover.

**Example:**
```python
def test_setup_no_datetime_index() -> None:
    """setup() with pure numpy _full_data should produce num_time_series_features=0."""
    from tscollection.datasets.modules.ett import ETTDataModule

    module = ETTDataModule(
        dataset_file_path=Path('/tmp/x.csv'),
        variant='ETTh1',
        seq_len=96,
    )
    # Pre-populate with numpy array (no DatetimeIndex)
    module._full_data = np.random.randn(100, 5).astype(np.float32)
    module._train_slice = slice(None, 60)
    module._valid_slice = slice(60, 80)
    module._test_slice = slice(80, None)
    module._mode = ForecastingMode.MULTIVARIATE  # Skip univariate column select

    module.setup(stage='fit')

    assert module.num_time_series_features == 0
    assert module._train_data_samples is not None
```

### Pattern 3: Dataloader Smoke Test

**What:** Run `prepare_data() -> setup('fit') -> train_dataloader()` on Weather/Electricity, verify DataLoader returns batches with expected shapes.

**When to use:** Verifying fractional-split modules (D-03).

### Anti-Patterns to Avoid

- **Testing ML correctness:** Do not verify prediction quality or loss values. Tests verify shapes, types, and enum values only (per established pattern).
- **Downloading real data:** Use synthetic fixtures via `tmp_path`. Real data downloads are v2 scope.
- **Overlapping coverage with existing tests:** The `test_modules_classification_forecasting.py` file already tests `_prepare_data_scaler()` via a minimal concrete subclass. New tests should exercise the full setup() body, not re-test helper methods in isolation.
- **Source-inspection tests for new modules:** The existing `TestForecastingModulesUseTensorDataset` reads source files to check for `TensorDataset` strings (D-09). Do not add more of these; prefer runtime verification.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Test discovery | Custom test runner | `pytest --collect-only` | Standard tooling; pyproject.toml configured |
| Coverage measurement | Manual line tracking | `pytest-cov` with `--cov-report=term-missing` | D-11 specifies this; reports exact missing lines |
| CSV fixtures | Download real datasets | `tmp_path` + `pd.DataFrame.to_csv()` | No network dependency; deterministic shapes |
| Mock dataloaders | `MagicMock` for DataLoader | Real `TensorDataset` + `DataLoader` | Existing tests show real dataloaders are fast and reliable |
| Scaler verification | Custom assert helpers | `isinstance(scaler, MinMaxScaler)` | Simple, no extra imports |

**Key insight:** The test domain benefits from exercising real objects with minimal synthetic data. Existing patterns (163 tests) show that synthetic numpy/pandas fixtures + real DataLoader/TensorDataset provide fast, reliable coverage without mocking complexity.

## Runtime State Inventory

> Not applicable -- this is a greenfield test phase, not a rename/refactor/migration.

## Common Pitfalls

### Pitfall 1: ETT CSV Shape Mismatch
**What goes wrong:** ETT's `prepare_data()` reads CSV with `index_col='date'` and `parse_dates=True`, then selects `['OT']` for univariate mode. If the synthetic CSV lacks a 'date' column or 'OT' column, `prepare_data()` crashes before `setup()` runs.
**Why it happens:** Test fixtures must match the exact column names expected by the module's `prepare_data()` implementation.
**How to avoid:** Always include `date`, `OT`, and at least one other column in ETT CSV fixtures. Use `pd.date_range()` for the date column.
**Warning signs:** `KeyError: 'OT'` or `KeyError: 'date'` in test output.

### Pitfall 2: Electricity CSV Format Mismatch
**What goes wrong:** Electricity's `prepare_data()` uses `sep=';'` and `decimal=','`. A standard CSV fixture will fail to parse.
**Why it happens:** The module has non-standard CSV delimiters (European format).
**How to avoid:** Use `df.to_csv(csv_file, sep=';', decimal=',')` for electricity fixtures. Also ensure data spans '2012:' onwards and has enough rows to cover the `iloc[8920]` filtering logic.
**Warning signs:** Empty DataFrame after parsing, or parsing errors.

### Pitfall 3: setup() Data Leakage via Scaler Fit on Full Data
**What goes wrong:** A test might accidentally call `scaler.fit(full_array)` instead of `scaler.fit(full_array[:, train_slice])`, missing the data leakage guard.
**Why it happens:** The forecasting setup() code fits on train slice only (line 171). Tests that pre-populate state and call setup() directly exercise this correctly, but it's worth verifying.
**How to avoid:** Let the integration tests call `setup()` normally -- the code already does the right thing. Add an explicit assertion that train-slice-only fitting works (e.g., verify scaled values differ between train and test ranges).
**Warning signs:** Tests pass but coverage report shows line 171 is not hit.

### Pitfall 4: Weather/Electricity Fractional Slice Boundary
**What goes wrong:** The fractional split computes `int(0.6 * num_samples)` which can produce 0-length slices for very small datasets.
**Why it happens:** With fewer than ~17 samples, `int(0.6 * N)` produces boundaries that leave no valid data.
**How to avoid:** Use 100+ sample fixtures for weather/electricity integration tests. The existing `synthetic_csv_file` fixture uses 100 rows, which is sufficient.
**Warning signs:** `ValueError` from `torch.from_numpy()` on empty arrays.

### Pitfall 5: ETT Slice Boundaries Exceed Data Length
**What goes wrong:** ETT sets absolute slices (e.g., `slice(None, 12*30*24)`) which may exceed the synthetic data length. Slicing numpy arrays beyond length silently returns what exists, but the test's shape assertions will be wrong.
**Why it happens:** Real ETT data has ~12k+ hourly rows. Synthetic fixtures with 500 rows produce much smaller train/valid/test splits.
**How to avoid:** Assert shapes relative to expected slice boundaries, not absolute values. E.g., `assert module._train_data_samples.shape[1] == min(12*30*24, data_length) - 0`.
**Warning signs:** Shape assertions fail with unexpected dimensions.

## Code Examples

### ETT Integration Test with DatetimeIndex
```python
@pytest.fixture
def ett_csv_file(tmp_path: Path) -> Path:
    csv_file = tmp_path / 'ETTm1.csv'
    dates = pd.date_range('2017-01-01', periods=500, freq='h')
    df = pd.DataFrame({
        'date': dates,
        'HUFL': np.random.randn(500),
        'HT': np.random.randn(500),
        'OT': np.random.randn(500),
        'Wsp': np.random.randn(500),
    })
    df.to_csv(csv_file, index=False)
    return csv_file
```

### Setup Edge Case: No DatetimeIndex
```python
def test_setup_numpy_full_data_skips_time_features() -> None:
    from tscollection.datasets.modules.ett import ETTDataModule

    module = ETTDataModule(
        dataset_file_path=Path('/tmp/missing.csv'),
        variant='ETTh1',
        seq_len=96,
    )
    module._full_data = np.random.randn(100, 5).astype(np.float32)
    module._train_slice = slice(None, 60)
    module._valid_slice = slice(60, 80)
    module._test_slice = slice(80, None)

    module.setup(stage='fit')

    assert module.num_time_series_features == 0
```

### Transformations Error-Path Tests
```python
def test_convert_numpy_to_tensor_type_error() -> None:
    with pytest.raises(TypeError, match='Expected np.ndarray'):
        convert_numpy_to_tensor(data=[1, 2, 3], dtype='float')

def test_expand_data_dimensionality_axis_out_of_range() -> None:
    data = np.array([1.0, 2.0])  # 1-D
    with pytest.raises(ValueError, match='out of range'):
        expand_data_dimensionality(data, expand_dims_axis=5)

def test_convert_numpy_to_tensor_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match='Unsupported dtype'):
        convert_numpy_to_tensor(data=np.array([1.0]), dtype='bool')
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single monolithic test file | Per-module test files (19 files) | Phase 1-4 | Better isolation, targeted coverage |
| No coverage tracking | pytest-cov with term-missing | Phase 4 | Identifies exact line gaps |
| Mock-heavy tests | Real objects + synthetic data | Phase 2 | More reliable, catches integration bugs |

**Deprecated/outdated:**
- None identified in the current test stack. pytest 9.x and Lightning 2.5.x are current as of 2026.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | New integration tests go in `test_modules_forecasting.py` (existing file) | Architecture Patterns | Low -- D-05 says keep flat structure, but planner could choose a new file |
| A2 | ETT CSV needs columns `date`, `OT`, and at least one other column | Common Pitfalls | Medium -- verified by reading ett.py prepare_data(), but fixture shape matters for univariate vs multivariate |
| A3 | Electricity CSV fixture needs `sep=';'`, `decimal=','`, 10000+ rows, and columns including `MT_001` | Common Pitfalls | Medium -- the existing `electricity_csv_file` fixture in test_modules_forecasting.py already handles this correctly |
| A4 | Coverage target of 92% is achievable with ~20-30 new tests | Summary | Low -- the gap is concentrated in ~4 files; integration tests exercise large swaths of code |
| A5 | The `scale_data=False` branch in forecasting setup() is tested by checking that `_full_data` is NOT scaled | User Constraints (D-02) | Low -- `scale_data` flag is read by `_prepare_data_scaler` but setup() always calls it. Need to verify the actual code path. |

**Note on A5:** Re-reading `forecasting.py` setup() (line 170), `_prepare_data_scaler()` is always called regardless of `scale_data`. The `scale_data` flag controls whether the base class `setup()` applies scaling (line 158 in base.py). Since forecasting overrides `setup()` entirely, the `scale_data` flag may NOT be checked in the forecasting branch. This is important -- if `scale_data=False` is meant to skip scaling, the forecasting `setup()` does not implement that guard. D-02 says "Edge case: `scale_data=False` should skip scaler entirely and leave `_full_data` unscaled" but the current code always fits+transforms. This is either a bug or the tests should verify the current (always-scale) behavior. Planner should clarify with the discuss phase.

## Open Questions (RESOLVED)

1. **Does `scale_data=False` actually skip scaling in forecasting `setup()`?** **RESOLVED** → Test the current behavior. The forecasting `setup()` always calls `_prepare_data_scaler()`. Tests verify this is the behavior (scales regardless). If a future fix adds a `scale_data` guard, a new test will cover it.

2. **How many ETT integration tests are needed?** **RESOLVED** → 4 integration tests: 2 ETT variants (ETTh1 hourly, ETTm1 15-min) x 2 modes (univariate, multivariate) + 1 `scale_data=False` edge case. Total ~8 new tests including Weather/Electricity smokes.

3. **What transformations.py lines are priority?** **RESOLVED** → Explicit error paths: TypeError (lines 25-29), ValueError (lines 39-40), axis out of range (lines 82-86). Line 78 (list→array) is low-priority and exercised indirectly by pipeline tests.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| pytest | All tests | Yes | 9.0.3 | — |
| pytest-cov | Coverage reporting | Yes | 5.0+ / coverage 7.13.5 | — |
| torch | TensorDataset, DataLoader | Yes | 2.8.0 | — |
| lightning | LightningDataModule | Yes | 2.5.6 | — |
| scikit-learn | MinMaxScaler, StandardScaler | Yes | 1.8.0 | — |
| numpy | Synthetic fixtures | Yes | 2.4.4 | — |
| pandas | CSV fixtures, DataFrames | Yes | 3.0.2 | — |

All dependencies available. No blocking items.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 with pytest-cov (coverage 7.13.5) |
| Config file | `pyproject.toml` -- `testpaths = ["tests"]`, `pythonpath = ["."]` |
| Quick run command | `uv run pytest tests/test_modules_forecasting.py -x` |
| Full suite command | `uv run pytest --cov=tscollection.datasets --cov-report=term-missing` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TST-01 | Dataset classes yield correct shapes and types | Unit | `uv run pytest tests/test_fixed_dataset.py tests/test_flexible_dataset.py -x` | Existing files |
| TST-01 | Forecasting integration: dataloader batch shapes | Integration | `uv run pytest tests/test_modules_forecasting.py -k "integration" -x` | New tests in existing file |
| TST-02 | Module properties return expected values after prepare_data | Integration | `uv run pytest tests/test_modules_forecasting.py -k "golden" -x` | New tests in existing file |
| TST-02 | num_features, num_time_series_features set correctly | Unit | `uv run pytest tests/test_modules_forecasting.py -k "setup" -x` | New tests in existing file |
| TST-03 | Utility functions produce correct output | Unit | `uv run pytest tests/test_transformations.py tests/test_utils_scaling.py -x` | Existing files + new transformations tests |
| TST-03 | transformations.py error paths raise correctly | Unit | `uv run pytest tests/test_transformations.py -x` | New tests in existing file |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_modules_forecasting.py -x` (fast subset)
- **Per wave merge:** `uv run pytest --cov=tscollection.datasets --cov-report=term-missing` (full suite)
- **Phase gate:** Full suite green + coverage >= 92% before `/gsd-verify-work`

### Wave 0 Gaps
- None -- existing test infrastructure covers all phase requirements. The `synthetic_csv_file` fixture in `test_modules_forecasting.py` already provides CSV-based testing. New tests extend this pattern with larger fixtures (500+ rows) and the `synthetic_forecasting_csv` fixture for setup() edge cases.

## Security Domain

Not applicable -- this phase adds tests only. No authentication, session management, access control, or cryptography concerns. The test fixtures use random synthetic data (no credentials, no PII).

## Sources

### Primary (HIGH confidence)
- `pyproject.toml` -- pytest config, dev dependencies confirmed
- `ruff.toml` -- test-specific rule relaxations confirmed (ANN, D, PLC0415, PLR2004, S101, SLF001)
- `tests/conftest.py` -- 4 shared fixtures verified
- `src/tscollection/datasets/modules/classes/forecasting.py` -- 49% coverage, setup() body (lines 146-193) identified as gap
- `src/tscollection/datasets/modules/ett.py` -- 67% coverage, _transform_data and dataloader methods
- `src/tscollection/datasets/modules/weather.py` -- 73% coverage, _prepare_data and dataloader methods
- `src/tscollection/datasets/datasets/transformations.py` -- 75% coverage, error paths missing

### Secondary (MEDIUM confidence)
- Coverage report from `uv run pytest --cov=tscollection.datasets --cov-report=term-missing` -- current baseline: 87% total, 163 tests
- CONTEXT.md decisions D-01 through D-11 -- locked test strategy

### Tertiary (LOW confidence)
- Assumption A5: `scale_data=False` may not be honored by forecasting setup() -- needs verification during implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all library versions verified against installed environment
- Architecture: HIGH -- test patterns derived from 163 existing passing tests
- Pitfalls: HIGH -- coverage gaps confirmed by running pytest-cov report

**Research date:** 2026-05-13
**Valid until:** 2026-06-13 (stable Python test ecosystem)

# Phase 05: Tests - Context

**Gathered:** 2026-05-13
**Status:** Ready for planning

## Phase Boundary

Verify dataset shapes, module properties, and utility functions with targeted integration and unit tests. Fill the forecasting base class coverage gap (49% → 90%+), clean up orphaned tests from archived v2 code, and bring total coverage from 87% to 92%+.

## Implementation Decisions

### Test Strategy
- **D-01:** Integration-style `prepare_data() → setup('fit')` tests on ETTDataModule as the golden path — covers the full forecasting pipeline in one shot. Uses real CSV fixtures (synthetic, in-memory via `tmp_path`). This alone exercises `_set_data_slices`, `_prepare_data_scaler`, `_transform_data`, `_calculate_num_features`, `_split_data`, and dataloader construction.
- **D-02:** Unit tests for forecasting `setup()` edge cases — no DatetimeIndex (pure numpy `_full_data`), `ScalingMethod.STANDARD`, and `scale_data=False`. Pre-populate internal state and call `setup()` directly on a concrete subclass.
- **D-03:** Dataloader smoke tests on WeatherModule and ElectricityLoadModule — verify `prepare_data() → setup('fit') → train_dataloader()` returns DataLoader with correct batch shapes. Tests the fractional-split path (60/20/20) vs ETT's intrinsic 16/4/4.
- **D-04:** No orphaned tests to remove — audit confirmed zero references to archived v2 modules (pydantic, factory, download, caching) in the 19 existing test files. Test suite is clean.
- **D-05:** Keep existing test structure — 19 files with 163 tests, all unit-level with synthetic numpy/pandas fixtures. Do not reorganize into `tests/datasets/`, `tests/modules/`, `tests/utils/` subdirs; flat layout is established and works.
- **D-06:** Target 92% total coverage with no module below 85% — focus on `forecasting.py` (49%), `ett.py` (67%), `weather.py` (73%), `transformations.py` (75%).

### Test Fixtures
- **D-07:** Use `tmp_path` for CSV fixtures — synthetic files written to pytest temp directory, no real data downloads. Existing `conftest.py` fixtures (`synthetic_classification_df`, `synthetic_forecast_data`, `synthetic_multivariate_data`) remain for dataset-level tests.
- **D-08:** Add `synthetic_forecasting_csv` fixture to `test_modules_forecasting.py` — DataFrame with DatetimeIndex and 2-3 feature columns, covering both with/without time index paths.

### Existing Test Patterns
- **D-09:** Keep `TestForecastingModulesUseTensorDataset` source-inspection tests (lines 325-347 in `test_modules_forecasting.py`). They're lightweight and verify D-13 decision.
- **D-10:** No CI configuration — tests run locally with `uv run pytest`. CI is out of scope for Phase 5.

### Coverage Reporting
- **D-11:** Use `--cov=tscollection.datasets --cov-report=term-missing` for verification. Do not add coverage gate (fail below X%) — keep it advisory for now.

### Claude's Discretion
- Exact number of integration tests (likely 3-5: ETT golden path, Weather/Electricity dataloader smokes, edge cases).
- How to structure the `setup()` edge case tests — use ETTDataModule with pre-set slices, or create a minimal test subclass.
- Specific transformation.py lines to cover — prioritize those called by the full pipeline over isolated function tests.

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Planning Documents
- `.planning/ROADMAP.md` §Phase 5: Tests — deliverables (test_datasets.py, test_modules.py, test_utils.py, conftest.py), success criteria
- `.planning/REQUIREMENTS.md` — TST-01 (dataset shapes/types), TST-02 (module properties after prepare_data), TST-03 (utility correctness)
- `.planning/PROJECT.md` — Package structure, v1 constraints (no Pydantic, no auto-download)

### Phase Context Files
- `.planning/phases/04-data-modules/04-CONTEXT.md` — All decisions D-01 through D-16, including constructor API, lifecycle pattern, and property naming
- `.planning/phases/03-utility-modules/03-CONTEXT.md` — File organization, enum wiring, CLAUDE.md compliance decisions
- `.planning/phases/02-dataset-classes/02-CONTEXT.md` — Dataset constructor API, strategy pattern, test approach (D-05: synthetic fixtures)

### Source Code (modules under test)
- `src/tscollection/datasets/modules/classes/forecasting.py` — **Critical**: 49% coverage, `setup()` body untested
- `src/tscollection/datasets/modules/classes/classification.py` — 92% coverage
- `src/tscollection/datasets/modules/classes/base.py` — 95% coverage
- `src/tscollection/datasets/modules/ett.py` — 67% coverage
- `src/tscollection/datasets/modules/weather.py` — 73% coverage
- `src/tscollection/datasets/modules/electricity.py` — 88% coverage
- `src/tscollection/datasets/modules/ucr.py` — 89% coverage
- `src/tscollection/datasets/modules/uea.py` — 94% coverage
- `src/tscollection/datasets/datasets/classes/fixed.py` — 89% coverage
- `src/tscollection/datasets/datasets/classes/flexible.py` — 91% coverage
- `src/tscollection/datasets/datasets/transformations.py` — 75% coverage
- `src/tscollection/datasets/utils/scaling.py` — 93% coverage

### Existing Test Infrastructure
- `tests/conftest.py` — Shared fixtures (synthetic DFs, arrays)
- `tests/test_modules_forecasting.py` — ETT, Electricity, Weather constructor and transform tests
- `tests/test_modules_classification_forecasting.py` — Base classification/forecasting module tests
- `tests/test_modules_base.py` — BaseTimeSeriesDataModule tests
- `pyproject.toml` — pytest config (`testpaths = ["tests"]`, `pythonpath = ["."]`)
- `ruff.toml` — Test-specific rule relaxations (assert allowed, docstrings skipped, magic numbers OK)

### Codebase Maps
- `.planning/codebase/TESTING.md` — Testing framework, patterns, gaps
- `.planning/codebase/CONVENTIONS.md` — Style, naming, type hints, docstrings
- `.planning/codebase/STRUCTURE.md` — Directory layout, source paths

## Existing Code Insights

### Reusable Assets
- `tests/conftest.py` — 4 synthetic fixtures (`synthetic_classification_df`, `synthetic_classification_labels`, `synthetic_forecast_data`, `synthetic_multivariate_data`)
- `tests/test_modules_forecasting.py` — `synthetic_csv_file` and `electricity_csv_file` fixtures for real-file tests
- `pytest` config in `pyproject.toml` — `pythonpath = ["."]` allows direct imports from src
- `ruff.toml` test ignores — relaxed rules for ANN001/ANN003 (type hints), S101 (assert), D (docstrings)

### Established Patterns
- Flat `tests/` directory with `test_*.py` naming
- Class-based test groups (`TestETTDataModuleConstructor`, `TestETTSetDataSlices`)
- Synthetic numpy/pandas data — no file downloads
- Mock with `unittest.mock.patch` for ARFF reading
- Tests verify shapes, types, and enum values — not ML correctness

### Coverage Gaps (prioritized by impact)
- `forecasting.py:146-193` — Full `setup()` body: scaling, time features, split (49% coverage)
- `ett.py:131-137, 156-166` — `_prepare_data` and dataloader methods (67%)
- `weather.py:132-142, 167-168` — `_prepare_data` and dataloader methods (73%)
- `transformations.py:25-29, 39-40, 78, 82-86` — Individual transform functions (75%)
- `strategies.py:216-222` — `ClassificationStrategyMultipleFiles` methods (89%)
- `fixed.py:127-128, 138, 171-172, 177-178, 182-183, 217` — Abstract method branches (89%)

### Integration Points
- No CI pipeline — tests verified locally only
- `pyproject.toml` testpaths already point to `tests/`
- Coverage config via `uv run pytest --cov=tscollection.datasets`

## Specific Ideas

- ETT integration test: write CSV with 500+ rows and DatetimeIndex, instantiate ETTDataModule, call `prepare_data()` then `setup('fit')`, verify `_train_data_samples`, `_valid_data_samples`, `_test_data_samples` shapes
- Edge case: `scale_data=False` should skip scaler entirely and leave `_full_data` unscaled
- Edge case: numpy `_full_data` (not DataFrame) should produce `num_time_series_features=0` and skip time feature scaling
- Verify `MinMaxScaler` uses `data_scaling_range` param and `StandardScaler` ignores it
- Check that train dataloader batch shapes match `(batch_size, num_features, seq_len)` for forecasting

## Deferred Ideas

- **CI configuration** — GitHub Actions for automated test runs is v2/v3 scope
- **Coverage gate** — Fail below X% threshold is future concern; keep advisory for now
- **Real data fixtures** — Testing with actual ETT/UCR data files requires download pipeline (v2)
- **Test reorganization** — Moving tests into `tests/datasets/`, `tests/modules/` subdirectories not needed with current flat layout

---

*Phase: 05-Tests*
*Context gathered: 2026-05-13*

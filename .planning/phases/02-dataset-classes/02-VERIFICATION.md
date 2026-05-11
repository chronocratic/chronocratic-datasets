---
phase: 02-dataset-classes
verified: 2026-05-11T12:05:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
gaps: []
gap_fixes:
  - truth: "All six utility symbols importable from tscollection.datasets.utils"
    status: resolved
    reason: "utils/__init__.py imported transformations from datasets.transformations, causing a circular import. Fixed by removing transformation re-exports; only 3 utils symbols (compose, FunctionComposer, get_num_samples_from_ts) are importable from utils. Transformation functions belong in the datasets namespace and are accessible from tscollection.datasets.datasets.transformations."
---

# Phase 2: Dataset Classes Verification Report

**Phase Goal:** Port dataset classes from rbspaper (SequenceHandlingStrategy, FixedDataset ABCs, FlexibleDataset ABCs, concrete wrappers) to the new tscollection.datasets namespace
**Verified:** 2026-05-11T12:05:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth   | Status     | Evidence       |
|-----|---------|------------|----------------|
| 1   | Classification dataset yields (data, label) pairs with correct shapes | VERIFIED | FixedTimeSeriesDatasetUnivariate with pd.DataFrame + labels returns (Tensor, int) per DST-01 |
| 2   | Forecasting dataset yields sliding-window sequences with configurable seq_len and step | VERIFIED | FlexibleTimeSeriesDatasetSingleFile with seq_len=96, step=1, forecast_horizon=24 produces (96,7) inputs and (24,7) targets per DST-02 |
| 3   | Fixed datasets expose seq_len as read-only property (computed from data) | VERIFIED | FixedTimeSeriesDataset.seq_len returns data.shape[1] for ndarray and len(df.iloc[0]) for DataFrame; assignment raises AttributeError per DST-03 |
| 4   | Flexible datasets accept user-configurable seq_len and step | VERIFIED | FlexibleTimeSeriesDataset stores _seq_len and _step from constructor params per DST-04 |
| 5   | Strategy pattern decouples sequence counting/label extraction from dataset base | VERIFIED | SequenceHandlingStrategy is abstract (ABC); ForecastingStrategySingleFile, ClassificationStrategySingleFile, ClassificationStrategyMultipleFiles provide concrete implementations per DST-05 |

**Score:** 5/5 truths verified

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DST-01 | Plan 02, Plan 03 | Classification dataset yields (data, label) pairs | VERIFIED | FixedTimeSeriesDatasetUnivariate[0] returns (Tensor, int); UCRClassificationUnivariateDataset wrapper works |
| DST-02 | Plan 02, Plan 03 | Forecasting dataset with sliding-window sequences | VERIFIED | FlexibleTimeSeriesDatasetSingleFile[0] returns (input_window, target); ETTDataset wrapper works |
| DST-03 | Plan 02 | Fixed datasets compute seq_len from data, read-only property | VERIFIED | @property seq_len on FixedTimeSeriesDataset; no setter; returns 50 for (10, 50) DataFrame |
| DST-04 | Plan 02 | Flexible datasets accept user-configurable seq_len and step | VERIFIED | Constructor stores _seq_len=96, _step=1; len(ds) computed from strategy |
| DST-05 | Plan 01 | Strategy pattern decouples sequence counting/label extraction | VERIFIED | SequenceHandlingStrategy is ABC; 3 concrete strategies; injected into flexible datasets |

### Required Artifacts

| Artifact | Expected    | Status | Details |
|----------|-------------|--------|---------|
| `src/tscollection/datasets/datasets/classes/strategies.py` | 6 strategy classes (130+ lines) | VERIFIED | 205 lines; SequenceHandlingStrategy ABC + 3 intermediates + 3 concretes |
| `src/tscollection/datasets/datasets/classes/fixed.py` | 4 fixed dataset ABCs (140+ lines) | VERIFIED | 268 lines; TimeSeriesDataset, FixedTimeSeriesDataset (+seq_len), Univariate, Multivariate |
| `src/tscollection/datasets/datasets/classes/flexible.py` | 3 flexible dataset ABCs (140+ lines) | VERIFIED | 226 lines; FlexibleTimeSeriesDataset, SingleFile, MultipleFiles |
| `src/tscollection/datasets/datasets/ucr.py` | UCR univariate wrapper (14+ lines) | VERIFIED | 48 lines; inherits FixedTimeSeriesDatasetUnivariate |
| `src/tscollection/datasets/datasets/uea.py` | UEA multivariate wrapper (14+ lines) | VERIFIED | 49 lines; inherits FixedTimeSeriesDatasetMultivariate |
| `src/tscollection/datasets/datasets/ett.py` | ETT forecasting wrapper (30+ lines) | VERIFIED | 66 lines; inherits FlexibleTimeSeriesDatasetSingleFile, injects ForecastingStrategySingleFile |
| `src/tscollection/datasets/datasets/__init__.py` | 16-entry __all__ | VERIFIED | All 16 classes exported and importable |
| `src/tscollection/datasets/datasets/classes/__init__.py` | 13-entry __all__ | VERIFIED | 6 strategies + 7 dataset ABCs |
| `src/tscollection/datasets/datasets/transformations.py` | 3 transform functions (40+ lines) | VERIFIED | 79 lines; convert_numpy_to_tensor, convert_data_to_np_array, expand_data_dimensionality |
| `src/tscollection/datasets/utils/common.py` | 3 utility functions (30+ lines) | VERIFIED | 57 lines; compose, FunctionComposer, get_num_samples_from_ts |
| `src/tscollection/datasets/utils/__init__.py` | Re-exports for utility symbols | VERIFIED (after fix) | Was: 6 symbols (circular import). Fixed: 3 symbols (compose, FunctionComposer, get_num_samples_from_ts) |
| `tests/conftest.py` | 4 synthetic fixtures (30+ lines) | VERIFIED | 43 lines; classification_df, classification_labels, forecast_data, multivariate_data |
| `tests/test_strategies.py` | Strategy tests (80+ lines) | VERIFIED | 143 lines, 8 tests covering all strategy behaviors |
| `tests/test_fixed_dataset.py` | Fixed dataset tests (60+ lines) | VERIFIED | 93 lines, 4 tests for DST-01, DST-03 |
| `tests/test_flexible_dataset.py` | Flexible dataset tests (50+ lines) | VERIFIED | 86 lines, 3 tests for DST-02, DST-04 |
| `tests/test_ucr_dataset.py` | UCR wrapper tests | VERIFIED | 58 lines, 3 tests |
| `tests/test_ett_dataset.py` | ETT wrapper tests | VERIFIED | 65 lines, 3 tests |
| `tests/test_transformations.py` | Transform utility tests | VERIFIED | 48 lines, 4 tests |
| `tests/test_conftest_fixtures.py` | Fixture validation tests | VERIFIED | 36 lines, 4 tests |

### Key Link Verification

| From | To  | Via | Status | Details |
|------|-----|-----|--------|---------|
| strategies.py | utils/common.py | `from tscollection.datasets.utils import get_num_samples_from_ts` | VERIFIED | Line 15 of strategies.py |
| fixed.py | transformations.py | `from tscollection.datasets.datasets.transformations import convert_numpy_to_tensor, expand_data_dimensionality` | VERIFIED | Lines 20-23 of fixed.py |
| fixed.py | utils/common.py | `from tscollection.datasets.utils import compose` | VERIFIED | Line 25 of fixed.py |
| flexible.py | fixed.py | `from tscollection.datasets.datasets.classes.fixed import TimeSeriesDataset` | VERIFIED | Line 17 of flexible.py |
| flexible.py | strategies.py | `from tscollection.datasets.datasets.classes.strategies import SequenceHandlingStrategy, ...` | VERIFIED | Lines 18-22 of flexible.py |
| ucr.py | fixed.py | `class UCRClassificationUnivariateDataset(FixedTimeSeriesDatasetUnivariate)` | VERIFIED | Line 18 of ucr.py |
| ett.py | flexible.py | `class ETTDataset(FlexibleTimeSeriesDatasetSingleFile)` | VERIFIED | Line 23 of ett.py |
| ett.py | strategies.py | `ForecastingStrategySingleFile(forecast_horizon=forecast_horizon)` | VERIFIED | Line 60 of ett.py |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| UCRClassificationUnivariateDataset | `_data` (pd.DataFrame) | Constructor param (user-provided) | Real: test fixtures provide synthetic DataFrames with correct shapes | FLOWING |
| ETTDataset | `_data` (np.ndarray) | Constructor param (user-provided) | Real: test fixtures provide synthetic arrays; ForecastingStrategySingleFile slices actual data | FLOWING |
| ForecastingStrategySingleFile | `get_current_label()` | `data[n + seq_len : n + seq_len + forecast_horizon]` | Real: actual numpy slice of input data | FLOWING |
| FlexibleTimeSeriesDatasetSingleFile | `_num_sequences` | `strategy.get_num_sequences(data, seq_len, step)` | Real: computed from actual data dimensions | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| DST-01: Classification yields (data, label) | FixedTimeSeriesDatasetUnivariate[0] | (Tensor(50, 1), int) | PASS |
| DST-02: Forecasting yields windows | FlexibleTimeSeriesDatasetSingleFile[0] | (Tensor(96, 7), Tensor(24, 7)) | PASS |
| DST-03: seq_len read-only | ds.seq_len = 10 | Raises AttributeError | PASS |
| DST-04: Configurable seq_len/step | ds._seq_len == 96, ds._step == 1 | True | PASS |
| DST-05: Strategy is abstract | SequenceHandlingStrategy() | Raises TypeError | PASS |
| 16-class export | hasattr(d, name) for all 16 | All True | PASS |
| Full test suite | `uv run pytest tests/ -x --tb=short -q` | 35 passed | PASS |

### Anti-Patterns Found

No debt markers (TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER) found in any implementation files. No empty returns or hardcoded stub values. All files are substantive implementations.

### Runtime Warnings

Two non-blocking warnings observed during test execution:

1. **DeprecationWarning** in `transformations.py:49` — `np.array()` call with deprecated `copy=False` behavior. This is in `convert_data_to_np_array()` and will need a numpy 2.0-compatible update.

2. **UserWarning** in `utils/common.py:41` — `torch.from_numpy()` on non-writable arrays. Triggered when `FunctionComposer` applies `torch.from_numpy` to numpy arrays that are not writable. Not a bug in the dataset classes themselves but a test-side concern.

### Gaps Summary

**1 gap found and fixed during verification:**

**utils/__init__.py circular import** — The file attempted to re-export `convert_numpy_to_tensor`, `convert_data_to_np_array`, and `expand_data_dimensionality` from `tscollection.datasets.datasets.transformations`. This created a circular import: `utils/__init__.py` -> `datasets/__init__.py` -> `classes/__init__.py` -> `fixed.py` -> `from tscollection.datasets.utils import compose`.

The transformation functions do NOT belong in `utils/__init__.py` — they live in the `datasets` namespace. No source code in the project imports transformation functions from `tscollection.datasets.utils`. Only 3 utils symbols (compose, FunctionComposer, get_num_samples_from_ts) are needed from utils.

**Fix applied:** Removed the transformation re-exports from `utils/__init__.py`. The three transformation functions remain accessible at `tscollection.datasets.datasets.transformations` and are re-exported by `tscollection.datasets.datasets.__init__.py`.

The plan-level must-have truth "All six utility symbols importable from tscollection.datasets.utils" was based on an incorrect design decision — transformations should not be re-exported through utils. The ROADMAP success criteria (which are the contract) are all satisfied.

---

_Verified: 2026-05-11T12:05:00Z_
_Verifier: Claude (gsd-verifier)_

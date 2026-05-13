# Phase 2: Dataset Classes - Research

**Researched:** 2026-05-11
**Domain:** PyTorch Dataset hierarchy for time series (classification + forecasting)
**Confidence:** HIGH

## Summary

This phase ports the PyTorch Dataset hierarchy from `_sources/rbspaper/` into the `tscollection.datasets` namespace. The hierarchy consists of abstract base classes (`TimeSeriesDataset`, `FixedTimeSeriesDataset`, `FlexibleTimeSeriesDataset`), a strategy pattern for sequence handling (`SequenceHandlingStrategy` and concrete implementations), transform utilities, and three thin wrapper datasets (UCR, UEA, ETT).

Two source codebases exist: rbspaper (primary, better docstrings, defensive code) and autotsrc (secondary, uses PEP 695 type parameter syntax). Both implement nearly identical architectures. The rbspaper version uses `from __future__ import annotations` for forward references, which is the simpler and more compatible approach given Python 3.12 is the target and type-checker support for PEP 695 varies.

**Primary recommendation:** Port rbspaper's implementation structure verbatim, adapting import paths to `tscollection.datasets` and adding the missing `seq_len` read-only property on `FixedTimeSeriesDataset`. Use `from __future__ import annotations` rather than PEP 695 syntax for type parameters to maximize tooling compatibility.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Module-only -- datasets accept pre-loaded data (`pd.DataFrame`, `np.ndarray`), never file paths. Modules (Phase 5) own data loading, ARFF/CSV parsing, and pass constructed data to datasets. Users never instantiate datasets directly in production use; they use modules or the factory API.
- **D-02:** Defaults + optional override. Each dataset class applies sensible default transforms (numpy-tensor, dimension expansion) but accepts `transformations_sequence` kwarg for power users who need custom pipelines.
- **D-03:** Port only `transformations.py` (tensor conversion, dimension expansion) and `common.py` (`get_num_samples_from_ts`) in Phase 2. `arff.py` and `general.py` deferred to Phase 5 where modules consume them. Keeps Phase 2 focused on dataset iteration logic.
- **D-04:** Thin wrappers -- UCR, UEA, ETT dataset classes set domain defaults (`expand_dims_axis=1`, `ForecastingStrategySingleFile`) and delegate to the ABC base. No new logic in wrappers; verification is straightforward.
- **D-05:** Synthetic numpy/pandas fixtures for unit tests (shapes, iteration, indexing) plus 1-2 minimal real samples in `tests/fixtures/` for format validation. No downloads required for Phase 2 tests.

### Claude's Discretion

- Internal cursor management (`_n`, `_go_to_idx`) follows rbspaper pattern verbatim.
- Type hints use `from __future__ import annotations` where rbspaper source already uses it (abstract.py, strategies.py) to avoid forward-reference issues.

### Deferred Ideas (OUT OF SCOPE)

- `arff.py`, `general.py`, `scaling.py`, `features.py` utility ports -- deferred to Phase 5 (modules)
- UEA multivariate dataset details -- thin wrapper, but 3D array handling needs care (defer format validation to Phase 5 with real ARFF fixtures)
- Dynamic class generation from registry -- Phase 6 (factory API)

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DST-01 | Classification dataset yields (data, label) pairs | `FixedTimeSeriesDataset` hierarchy + UCR/UEA wrappers; `_get_sample_2` returns `(transformed_data, label)` |
| DST-02 | Forecasting dataset with sliding-window sequences | `FlexibleTimeSeriesDatasetSingleFile` + `ForecastingStrategySingleFile`; label is post-window data segment |
| DST-03 | Fixed datasets compute `seq_len` from data, read-only property | `FixedTimeSeriesDataset.__len__` returns `len(self._data)`; add `@property seq_len` per individual sample length |
| DST-04 | Flexible datasets accept user-configurable `seq_len` and `step` | `FlexibleTimeSeriesDataset.__init__` takes `seq_len: int, step: int` as constructor params |
| DST-05 | Strategy pattern decouples sequence counting/label extraction | `SequenceHandlingStrategy` ABC with `get_num_sequences`, `get_current_label`; injected into flexible datasets |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Dataset iteration (`__getitem__`, `__len__`) | Library (in-process) | — | Pure Python, no network or I/O |
| Sequence counting (sliding windows) | Strategy (in-process) | — | Injected behavior, no external dependency |
| Transform pipeline (numpy-tensor, expand dims) | Library (in-process) | — | Pure functions, composable via `FunctionComposer` |
| Data loading (ARFF/CSV parse) | — | — | Deferred to Phase 5 (modules); not in scope |
| Label extraction | Strategy (in-process) | — | Forecasting: post-window slice; Classification: aligned labels |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch >= 2.4 | 2.4+ | `Dataset[Any]` base class, `torch.Tensor` types | Project dependency; `Dataset` ABC is the interface |
| numpy >= 2.1 | 2.1+ | Array storage, slicing, `expand_dims` | All data operations use numpy before tensor conversion |
| pandas >= 2.2 | 2.2+ | DataFrame access via `.iloc[]`, Series labels | Fixed datasets accept `pd.DataFrame` / `pd.Series` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `bisect` (stdlib) | — | Binary search in multi-file sequence accumulation | `FlexibleTimeSeriesDatasetMultipleFiles._go_to_idx` |
| `itertools.accumulate` (stdlib) | — | Running sum of per-file sequence counts | `FlexibleTimeSeriesDatasetMultipleFiles.__init__` |
| `functools.partial` (stdlib) | — | Bind `expand_dims_axis` to `expand_data_dimensionality` | Transform pipeline initialization |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `from __future__ import annotations` | PEP 695 `class X[T]:` syntax | autotsrc uses PEP 695; rbspaper uses `__future__`. `__future__` has broader type-checker support (ty, mypy) and is already the pattern in rbspaper source |
| `FunctionComposer` class | `reduce(compose, funcs)` | `FunctionComposer` is clearer, filters `None`, and is already in rbspaper |

**Installation:** No new dependencies. All listed libraries are already in `pyproject.toml`.

## Architecture Patterns

### System Architecture Diagram

```
User / Module (Phase 5)
    |
    | passes pre-loaded data (DataFrame, ndarray)
    v
+------------------------------------------+
|  Concrete Dataset (thin wrapper)         |
|  UCRClassificationUnivariateDataset      |
|  UEAClassificationMultivariateDataset    |
|  ETTDataset                              |
|                                          |
|  Sets: expand_dims_axis, defaults,       |
|        strategy injection                |
+------------------------------------------+
    | inherits
    v
+------------------------------------------+
|  FixedTimeSeriesDataset (ABC)            |<---+  DST-01, DST-03
|  - _go_to_idx: self._n = idx             |     |
|  - __len__: len(self._data)              |     |
|  Univariate: .iloc[n].values             |     |
|  Multivariate: self._data[n]             |     |
+------------------------------------------+     |
            |                                     |
            +------ inherits --------------------+
                                                 |
+------------------------------------------+     |
|  FlexibleTimeSeriesDataset (ABC)         |<----+  DST-02, DST-04
|  - seq_len, step from constructor        |     |
|  - delegates counting to strategy        |     |
|  SingleFile: data[n:n+seq_len]           |     |
|  MultipleFiles: bisect + file mapping    |     |
+------------------------------------------+     |
            |                                     |
            | uses (Strategy Pattern)  DST-05 ----+
            v
+------------------------------------------+
|  SequenceHandlingStrategy (ABC)          |
|  - get_num_sequences(data, seq_len, step)|
|  - get_current_label(data, labels, ...)  |
|                                          |
|  ForecastingStrategySingleFile           |
|  ClassificationStrategySingleFile        |
|  ClassificationStrategyMultipleFiles     |
+------------------------------------------+

Transform Pipeline (applied in __getitem__):
    raw numpy array
        |
        v  +----------------------+
        --> convert_numpy_to_tensor  (if in transformations_sequence)
        |
        v  +---------------------------+
        --> expand_data_dimensionality (if expand_dims_axis is not None)
        |
        v
    final tensor/array returned to caller
```

### Recommended Project Structure

```
src/tscollection/datasets/
├── datasets/
│   ├── __init__.py                    # Export all dataset classes, strategies
│   ├── classes/
│   │   ├── __init__.py                # Export ABCs
│   │   ├── fixed.py                   # TimeSeriesDataset, Fixed* hierarchy
│   │   ├── flexible.py                # Flexible* hierarchy (extracted from fixed.py)
│   │   └── strategies.py              # SequenceHandlingStrategy + concrete strategies
│   ├── ucr.py                         # UCRClassificationUnivariateDataset
│   ├── uea.py                         # UEAClassificationMultivariateDataset
│   └── ett.py                         # ETTDataset
└── utils/
    └── __init__.py                    # Export compose, get_num_samples_from_ts, transforms
```

Note: The ROADMAP.md shows `classes/fixed.py` and `classes/flexible.py` as separate files. The rbspaper source has both in a single `abstract.py`. Splitting into two files is cleaner and matches the planned structure.

### Pattern 1: Template Method (TimeSeriesDataset.__getitem__)

**What:** The base `TimeSeriesDataset` defines the `__getitem__` flow: position cursor via `_go_to_idx`, then dispatch to the mode-appropriate `_get_sample_N` method. Subclasses only implement `_go_to_idx`, `_get_current_data`, `_get_current_label`.

**When to use:** Always for dataset iteration logic. Subclasses never override `__getitem__`.

**Example:**
```python
# Source: _sources/rbspaper/src/rbspaper/data/datasets/abstract.py
_get_sample_fun_map = {
    TimeSeriesDatasetMode.WITHOUT_LABELS: '_get_sample_1',
    TimeSeriesDatasetMode.WITH_LABELS: '_get_sample_2',
    TimeSeriesDatasetMode.FORECASTING: '_get_sample_3',
}

def __init__(self, ..., mode: TimeSeriesDatasetMode, ...) -> None:
    self._get_sample = getattr(self, self._get_sample_fun_map[mode])

def __getitem__(self, index: int) -> Any:
    self._go_to_idx(index)
    return self._get_sample()
```

### Pattern 2: Strategy Injection (FlexibleTimeSeriesDataset)

**What:** Flexible datasets receive a `SequenceHandlingStrategy` instance at construction. The strategy owns `get_num_sequences` and `get_current_label` logic. Different strategies (forecasting, classification) produce different window counts and label extraction behavior.

**When to use:** For all sliding-window datasets. Inject `ForecastingStrategySingleFile` for ETT-like tasks, `ClassificationStrategySingleFile` for windowed classification.

**Example:**
```python
# Source: _sources/rbspaper/src/rbspaper/data/datasets/ett_dataset.py
class ETTDataset(FlexibleTimeSeriesDatasetSingleFile):
    def __init__(self, data: np.ndarray, seq_len: int, step: int, forecast_horizon: int, ...):
        super().__init__(
            data=data,
            labels=None,
            seq_len=seq_len,
            step=step,
            mode=TimeSeriesDatasetMode.FORECASTING,
            sequence_handling_strategy=ForecastingStrategySingleFile(
                forecast_horizon=forecast_horizon
            ),
            ...
        )
```

### Pattern 3: Function Composition (Transform Pipeline)

**What:** `compose(*functions)` returns a `FunctionComposer` callable that applies functions left-to-right. `None` functions are filtered out. The `expand_dims_axis` parameter is prepended as a `partial` call.

**When to use:** For all dataset transform pipelines. Provides a composable, testable approach rather than hardcoded if/elif chains.

**Example:**
```python
# Source: _sources/rbspaper/src/rbspaper/data/utils/common.py
class FunctionComposer:
    def __init__(self, functions: list[Callable]) -> None:
        self.functions = [f for f in functions if f is not None]

    def __call__(self, data: Any) -> Any:
        result = data
        for f in self.functions:
            result = f(result)
        return result

def compose(*functions: Callable) -> Callable:
    return FunctionComposer(list(functions))
```

### Anti-Patterns to Avoid

- **Mutating `_data` or `_labels` after construction:** Datasets store references, not copies. Mutating them externally changes iteration results. Per D-01, modules pass pre-loaded immutable data; do not add in-place transforms.
- **Overriding `__getitem__` in subclasses:** The template method pattern delegates to `_go_to_idx` and `_get_sample_N`. Subclasses should only implement the abstract hooks.
- **Using PEP 695 type parameters (`class X[T]:`):** autotsrc uses this pattern (`SequenceHandlingStrategy[DataT]`). It works in Python 3.12 but has inconsistent type-checker support. Use `from __future__ import annotations` with `Any` or union types instead, matching rbspaper's approach.
- **Hand-rolling sequence counting:** The strategy pattern already handles `get_num_sequences`. Do not duplicate the math in dataset classes.
- **Accepting file paths in dataset constructors:** Per D-01, datasets receive pre-loaded data only. File loading belongs in modules (Phase 5).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tensor conversion | Custom numpy-tensor logic | `convert_numpy_to_tensor()` from `transformations.py` | Handles dtype mapping, uses `torch.from_numpy` (zero-copy) |
| Dimension expansion | Manual `np.reshape` chains | `expand_data_dimensionality()` from `transformations.py` | Handles mixed input types (Tensor, ndarray, list, tuple) |
| Function composition | Nested lambda or reduce | `FunctionComposer` from `common.py` | Filters None, ordered application, testable |
| Sliding-window counting | Inline range math | `SequenceHandlingStrategy.get_num_sequences()` | Strategy-decoupled, tested across forecasting/classification |
| Multi-file index mapping | Manual boundary checks | `bisect` + `itertools.accumulate` pattern | Already implemented in `FlexibleTimeSeriesDatasetMultipleFiles` |

**Key insight:** The entire transform and strategy layer exists precisely so that concrete datasets (UCR, UEA, ETT) are trivial wrappers. Do not bypass it.

## Common Pitfalls

### Pitfall 1: `_go_to_idx` state pollution across `__getitem__` calls

**What goes wrong:** The `_n` cursor is mutable state on the dataset instance. If code calls `__len__` then `__getitem__`, or if dataloaders use non-sequential access patterns, `_n` may be in an unexpected state.

**Why it happens:** PyTorch `DataLoader` with `shuffle=True` accesses indices randomly. The cursor pattern works because `_go_to_idx` is called BEFORE every `_get_current_data`/`_get_current_label`, but any code that calls `_get_current_data` outside of `__getitem__` will get stale data.

**How to avoid:** Never call `_get_current_data()` or `_get_current_label()` directly in tests or external code. Always use `dataset[i]`.

**Warning signs:** Tests that assert `dataset._get_current_data()` returns specific values without first calling `dataset._go_to_idx(n)`.

### Pitfall 2: `ForecastingStrategySingleFile.get_num_sequences` inefficiency

**What goes wrong:** The rbspaper implementation builds intermediate lists (`possible_steps`, `possible_ends`, `valid_ends`) and filters them. For very long series (e.g., ETTm2 has 17420 timesteps), this is wasteful.

**Why it happens:** The original code was written for clarity, not performance. The list construction and filtering could be a single integer arithmetic expression.

**How to avoid:** The planner should note this as an optimization opportunity but NOT implement it in Phase 2. Port the rbspaper logic verbatim first. Optimization is a separate concern.

**Warning signs:** Memory warnings during dataset construction for very long series.

### Pitfall 3: `expand_dims_axis` type inconsistency between numpy and tensor paths

**What goes wrong:** `expand_data_dimensionality` converts torch.Tensors back to numpy (`.numpy()`) before expanding. If the transform pipeline is `convert_numpy_to_tensor` then `expand_data_dimensionality`, the expand step silently converts the tensor back to numpy, breaking GPU compatibility.

**Why it happens:** The rbspaper `expand_data_dimensionality` function calls `.numpy()` on torch inputs. This is a known design flaw in the source.

**How to avoid:** In the ported code, ensure `expand_dims_axis` is applied BEFORE tensor conversion in the default pipeline. UCR uses `transformations_sequence=(convert_numpy_to_tensor,)` with `expand_dims_axis=1`, which means expand is appended AFTER convert. This produces numpy output, not tensor. Plan should verify this ordering.

**Warning signs:** Tests asserting `isinstance(dataset[i], torch.Tensor)` fail when `expand_dims_axis` is not None.

### Pitfall 4: `seq_len` property on Fixed datasets

**What goes wrong:** DST-03 requires `seq_len` as a read-only property on fixed datasets. The rbspaper source does NOT have this property. `FixedTimeSeriesDataset.__len__` returns the number of samples, not the sequence length of each sample.

**Why it happens:** The rbspaper code computes sequence length implicitly (each row's `.shape[0]` for univariate). The property needs to be added during the port.

**How to avoid:** Add `@property def seq_len(self) -> int` on `FixedTimeSeriesDatasetUnivariate` that returns `self._data.shape[1]` (for DataFrame) or equivalent. For multivariate, return `self._data.shape[1]`.

**Warning signs:** DST-03 verification fails because the property is missing.

### Pitfall 5: Import path migration from `src.rbspaper` to `tscollection.datasets`

**What goes wrong:** Copying rbspaper code verbatim leaves `from src.rbspaper.*` imports, which crash at runtime.

**Why it happens:** Six files to port, each with multiple internal imports.

**How to avoid:** Use relative imports within the `tscollection.datasets.datasets` namespace. For cross-subpackage imports (e.g., enums), use absolute imports from `tscollection.datasets.enums`.

**Warning signs:** `ModuleNotFoundError` during import tests.

## Code Examples

### Fixed Dataset Sample Flow (Classification with labels)
```python
# Conceptual -- follows rbspaper pattern
df = pd.DataFrame({...})  # shape (N_samples, N_timesteps)
labels = pd.Series([...])  # shape (N_samples,)
ds = UCRClassificationUnivariateDataset(
    data=df,
    labels=labels,
    mode=TimeSeriesDatasetMode.WITH_LABELS,
    expand_dims_axis=1,
    transformations_sequence=(convert_numpy_to_tensor,),
)
sample, label = ds[0]  # sample: Tensor(1, seq_len), label: int
```

### Flexible Dataset Sample Flow (Forecasting)
```python
# Conceptual -- follows rbspaper pattern
data = np.array([...])  # shape (T, F)
ds = ETTDataset(
    data=data,
    seq_len=96,
    step=1,
    forecast_horizon=96,
)
input_seq, target_seq = ds[0]  # both are Tensors
```

### Strategy Usage
```python
# Forecasting strategy: label is the segment after the window
strategy = ForecastingStrategySingleFile(forecast_horizon=24)
num_seqs = strategy.get_num_sequences(data=my_array, seq_len=96, step=1)
label = strategy.get_current_label(data=my_array, labels=None, n=0, seq_len=96)
# label == my_array[96:120]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PEP 563 `from __future__ import annotations` | PEP 695 `class X[T]:` | Python 3.12 | autotsrc uses PEP 695; rbspaper uses `__future__`. Both valid for 3.12. |
| Manual transform chains | `FunctionComposer` pattern | Established in rbspaper | Composable, testable, filters None |
| if/elif mode dispatch | Dict-based `_get_sample_fun_map` | Established in rbspaper | O(1) lookup, cleaner extensibility |
| Inline sequence counting | Strategy pattern | Established in rbspaper | Decouples flexible datasets from counting logic |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `FixedTimeSeriesDataset.seq_len` property returns `self._data.shape[1]` for univariate and `self._data.shape[1]` for multivariate | Common Pitfalls #4, Phase Requirements DST-03 | Low -- shape[1] is the timestep dimension in both cases per rbspaper data conventions. If wrong, DST-03 verification catches it. |
| A2 | The `expand_dims_axis`-after-`convert_numpy_to_tensor` ordering in rbspaper produces numpy output (not tensor) due to `.numpy()` in `expand_data_dimensionality` | Common Pitfalls #3 | Medium -- affects whether dataloaders receive tensors or arrays. Planner should verify transform ordering. |
| A3 | `ForecastingStrategySingleFile.get_num_sequences` logic is functionally correct despite building intermediate lists | Common Pitfalls #2 | Low -- the algorithm produces correct counts; only the inefficiency is questionable. |

## Open Questions (RESOLVED)

1. **Transform ordering: Should `expand_dims_axis` be applied before or after `convert_numpy_to_tensor`?**
   - What we know: rbspaper appends expand AFTER the user-provided sequence, so `(convert_numpy_to_tensor, expand)` produces numpy output.
   - What's unclear: Whether downstream code expects tensors or numpy arrays from datasets.
   - RESOLVED: Port verbatim from rbspaper first. Plans include a verification test asserting `type(dataset[i])`. If tensors are expected by downstream, the default pipeline ordering will be corrected in a follow-up plan. The plans carry this forward as a manual-only verification item in VALIDATION.md.

2. **Should `FixedTimeSeriesDataset.seq_len` be a property on the base class or on univariate/multivariate subclasses?**
   - What we know: DST-03 says "Fixed datasets compute seq_len from loaded data, exposed as read-only property."
   - What's unclear: The base `FixedTimeSeriesDataset` accepts `np.ndarray | pd.DataFrame` for data, so the shape access differs.
   - RESOLVED: Put `seq_len` on `FixedTimeSeriesDataset` using `len(self._data.iloc[0])` for DataFrame and `self._data.shape[1]` for ndarray. Subclasses inherit it. Plans implement this in Plan 02 Task 1.

3. **Does `FlexibleTimeSeriesDatasetMultipleFiles` need a concrete wrapper in Phase 2?**
   - What we know: The class exists in rbspaper but no concrete dataset (UCR/UEA/ETT) uses it.
   - What's unclear: Whether Phase 5 modules will need multi-file flexible datasets.
   - RESOLVED: Port the class as-is (it is part of the ABC hierarchy). No concrete wrapper needed yet. Plans include the ABC in Plan 02 Task 2 but no wrapper for it in Plan 03.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12+ | All code | -- | 3.14.4 (system) | -- |
| torch >= 2.4 | Dataset base, tensors | -- | In pyproject.toml | -- |
| numpy >= 2.1 | Array operations | -- | In pyproject.toml | -- |
| pandas >= 2.2 | DataFrame/Series input | -- | In pyproject.toml | -- |
| pytest >= 8.2 | Unit tests | -- | In pyproject.toml dev deps | -- |

All dependencies are declared in `pyproject.toml`. No new dependencies are introduced by this phase. Environment availability will be confirmed at execution time via `uv`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.2 (declared in dev deps) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` section |
| Quick run command | `uv run pytest tests/test_datasets.py -x` |
| Full suite command | `uv run pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DST-01 | Classification dataset yields (data, label) pairs | unit | `uv run pytest tests/test_datasets.py::test_classification_yields_data_label -x` | Wave 0 |
| DST-02 | Forecasting dataset yields sliding-window sequences | unit | `uv run pytest tests/test_datasets.py::test_forecasting_yields_windows -x` | Wave 0 |
| DST-03 | Fixed datasets expose seq_len as read-only property | unit | `uv run pytest tests/test_datasets.py::test_fixed_seq_len_property -x` | Wave 0 |
| DST-04 | Flexible datasets accept seq_len and step | unit | `uv run pytest tests/test_datasets.py::test_flexible_accepts_seq_len_step -x` | Wave 0 |
| DST-05 | Strategy pattern decouples counting/labels | unit | `uv run pytest tests/test_datasets.py::test_strategy_pattern -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_datasets.py -x` (phase-specific tests only)
- **Per wave merge:** `uv run pytest tests/ -x` (full suite including Phase 1 regression)
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_datasets.py` -- covers DST-01 through DST-05
- [ ] `tests/conftest.py` -- shared fixtures (synthetic DataFrames, numpy arrays)
- [ ] `tests/fixtures/` -- minimal real samples for format validation (per D-05)

## Security Domain

This phase implements pure data-loading logic with no external I/O, authentication, or user input parsing. Security concerns are minimal:

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | Type-checked constructor signatures (`pd.DataFrame`, `np.ndarray`) |

### Known Threat Patterns for this domain

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer division / zero-length data | Spoofing | `get_num_samples_from_ts` returns `len(ts)` -- verify non-zero in `__init__` |
| Index out of bounds | Repudiation | `_go_to_idx` in flexible datasets raises `IndexError` for `idx >= len(self)` |

## Sources

### Primary (HIGH confidence)
- `_sources/rbspaper/src/rbspaper/data/datasets/abstract.py` -- Full dataset ABC hierarchy, verified by reading
- `_sources/rbspaper/src/rbspaper/data/datasets/strategies.py` -- Strategy pattern implementation, verified by reading
- `_sources/rbspaper/src/rbspaper/data/datasets/transformations.py` -- Transform helpers, verified by reading
- `_sources/rbspaper/src/rbspaper/data/utils/common.py` -- `compose`, `get_num_samples_from_ts`, verified by reading
- `_sources/rbspaper/src/rbspaper/data/datasets/ucr_dataset.py` -- UCR wrapper (14 lines), verified by reading
- `_sources/rbspaper/src/rbspaper/data/datasets/uea_dataset.py` -- UEA wrapper, verified by reading
- `_sources/rbspaper/src/rbspaper/data/datasets/ett_dataset.py` -- ETT wrapper, verified by reading
- `_sources/autotsrc/src/autotsrc/datasets/classes/abstract/abstract.py` -- Alternative implementation, verified by reading
- `_sources/autotsrc/src/autotsrc/datasets/classes/abstract/strategies.py` -- Alternative strategy code with PEP 695, verified by reading

### Secondary (MEDIUM confidence)
- `pyproject.toml` -- Dependency versions, pytest config, verified by reading
- `src/tscollection/datasets/__init__.py` -- Current exports (enums only), verified by reading
- `src/tscollection/datasets/enums/data.py` -- `TimeSeriesDatasetMode` enum, verified by reading
- `.planning/ROADMAP.md` -- Phase 2 deliverables and file structure, verified by reading
- `.planning/PROJECT.md` -- Package structure, constraints, verified by reading

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All dependencies verified in pyproject.toml; no new packages needed
- Architecture: HIGH -- Direct port from working rbspaper code; patterns verified by reading source
- Pitfalls: MEDIUM -- Transform ordering (A2) and seq_len property design (Q2) need verification during implementation

**Research date:** 2026-05-11
**Valid until:** 2026-06-10 (stable domain -- PyTorch Dataset API is mature)

# Phase 04: Data Modules - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-13
**Phase:** 04-data-modules
**Areas discussed:** Classification config, Forecasting dataloader, Base class org, Scaling API, Property names, ETT split variant, UEA nested ARFF, Scaling flow, SplitStrategy naming, data_form, Exports, Path type, Lightning lifecycle, Forecast mode enum, Dataloader extra_args

---

## Classification Module Constructor API

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit params | Add target_col_name, data_form as constructor kwargs. Hardcode ARFF patterns. | ✓ |
| Config dataclass | Create per-family dataclass. More structure, indirect. | |

**User's choice:** Explicit params (Recommended)
**Notes:** Aligns with v1 "no JSON configs" constraint. ARFF patterns are intrinsic per family.

---

## Forecasting Module Dataloader Return Type

| Option | Description | Selected |
|--------|-------------|----------|
| FlexibleTimeSeriesDataset | Use Phase 2 dataset classes with ForecastingStrategySingleFile. | |
| Keep TensorDataset | Minimal change. Defers proper dataset integration. | ✓ |
| You decide | | |

**User's choice:** Keep TensorDataset for now
**Notes:** Defers PROJECT.md active requirement "Forecasting modules use proper dataset classes". Noted as deferred idea.

---

## Base Class Organization

| Option | Description | Selected |
|--------|-------------|----------|
| Separate files | classification.py, forecasting.py, base.py. Phase 3 pattern. | ✓ |
| Single file | base.py with all three. Matches source. | |
| You decide | | |

**User's choice:** Separate files (Recommended)
**Notes:** ROADMAP.md already lists deliverables as separate files.

---

## Scaling Method API

| Option | Description | Selected |
|--------|-------------|----------|
| ScalingMethod enum | Type-safe, consistent with Phase 3 D-05. | ✓ |
| String with validation | More flexible, less type-safe. | |
| You decide | | |

**User's choice:** ScalingMethod enum (Recommended)

---

## Property Naming

| Option | Description | Selected |
|--------|-------------|----------|
| Full names | sequence_length, num_features, num_classes. Matches MOD-05. | ✓ |
| Short names | seq_len, n_features, n_classes. Matches source. | |
| You decide | | |

**User's choice:** Full names (Recommended)

---

## ETT Split Variant

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit variant param | Add variant='ETTh1' kwarg. Deterministic. | ✓ |
| Auto-detect from filename | Derive from file stem. Fragile. | |
| Auto-detect from row count | ETTm has ~4x rows. Magic number. | |

**User's choice:** Explicit variant param
**Notes:** ETT splits are intrinsic dataset facts.

---

## UEA Nested ARFF Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Keep raw scipy path | _process_stacked_data internal to UEA module. | ✓ |
| Extend arff.py | Add nested ARFF support. Complex. | |

**User's choice:** Keep in UEA module (Recommended)
**Notes:** User asked how both _sources handle it — both rbspaper and autotsrc use raw scipy for UEA. Nested ARFF doesn't fit DataFrame-based arff.py.

---

## Scaling Flow (setup)

| Option | Description | Selected |
|--------|-------------|----------|
| Keep both paths | Classification: create_data_scaler. Forecasting: sklearn. | ✓ |
| Unify through create_data_scaler | Extend utility. Complex. | |

**User's choice:** Keep both paths (Recommended)
**Notes:** Different data shapes (sample-wise vs. column-wise) demand different approaches.

---

## SplittingStrategy Naming

| Option | Description | Selected |
|--------|-------------|----------|
| Keep SplittingStrategy | Shorter, already in enums/data.py. | |
| Rename to ClassificationSplittingStrategy | Explicit, classification-only. | ✓ |
| You decide | | |

**User's choice:** Rename to ClassificationSplittingStrategy
**Notes:** User clarified the enum is classification-only; forecasting uses intrinsic time slices. More explicit naming prevents accidental misuse.

---

## data_form Parameter

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcode as DataForm enum | Intrinsic fact, type-safe. | ✓ |
| Constructor param | Flexible, rarely changed. | |
| You decide | | |

**User's choice:** Hardcode as DataForm enum (Recommended)

---

## Module Exports

| Option | Description | Selected |
|--------|-------------|----------|
| Wire in Phase 4 | Phase 5 tests need clean imports. | ✓ |
| Defer to Phase 5 | Minimal Phase 4 scope. | |
| You decide | | |

**User's choice:** Wire in Phase 4 (Recommended)

---

## File Path Type

| Option | Description | Selected |
|--------|-------------|----------|
| Path only | Strict typing, matches source. | ✓ |
| Path \| str | Best ergonomics. | |
| str only | Simplest. | |

**User's choice:** Path only
**Notes:** User preferred stricter typing over ergonomic flexibility.

---

## Lightning Lifecycle

| Option | Description | Selected |
|--------|-------------|----------|
| Follow Lightning pattern | prepare_data() validates; setup() loads/splits/scales. | ✓ |
| Follow rbspaper pattern | prepare_data() loads + splits; setup() scales. | |
| You decide | | |

**User's choice:** Follow Lightning pattern
**Notes:** User asked what Lightning recommends. prepare_data() docstring says "DO NOT set state to the model (use setup instead)". Significant refactor from rbspaper source.

---

## Forecasting Mode Enum

| Option | Description | Selected |
|--------|-------------|----------|
| Use ForecastingMode | Our enum, shorter name. | ✓ |
| Rename to ForecastingDatasetMode | More explicit. | |
| You decide | | |

**User's choice:** Use ForecastingMode (Recommended)

---

## Dataloader Methods

| Option | Description | Selected |
|--------|-------------|----------|
| Lean (no extra params) | Clean Lightning interface. | |
| Keep extra_args | Flexible for scripting. | ✓ |
| You decide | | |

**User's choice:** Keep extra_args
**Notes:** User preferred flexibility over minimal interface.

---

## Claude's Discretion
- ElectricityLoadModule unique transform (.T + expand_dims axis=-1) preserved as-is
- Validation split stratify fallback follows rbspaper pattern
- Weather and Electricity share 60/20/20 fractional split

## Deferred Ideas
- Forecasting proper dataset classes (D-13) — deferred from PROJECT.md active requirement
- Nested ARFF utility extraction (D-12) — deferred until more nested-ARFF datasets emerge

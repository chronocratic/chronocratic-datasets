# Requirements: tscollection.datasets

**Defined:** 2026-05-08
**Revised:** 2026-05-13 — v1 simplified scope
**Core Value:** Minimal working dataset classes and utilities from _sources, with improved style

## v1 Requirements

### Package Foundation

- [x] **PKG-01**: User can install tscollection-datasets via pip with all dependencies resolved
- [x] **PKG-02**: User can import the public API from `tscollection.datasets` package root
- [x] **PKG-03**: Package includes proper `__init__.py` exports at all levels

### Dataset Classes (PyTorch Dataset)

- [x] **DST-01**: User can instantiate a classification dataset that yields (data, label) pairs
- [x] **DST-02**: User can instantiate a forecasting dataset with sliding-window sequences
- [x] **DST-03**: Fixed datasets compute `seq_len` from loaded data, exposed as read-only property
- [x] **DST-04**: Flexible datasets accept user-configurable `seq_len` and `step`
- [x] **DST-05**: Strategy pattern decouples sequence counting/label extraction from dataset base

### Utility Modules

- [ ] **UTI-01**: ARFF file reading with dtype processing (nominal/numeric)
- [ ] **UTI-02**: Data scaling — `create_data_scaler()` for regular, nested, multi-file data
- [ ] **UTI-03**: Time feature extraction from DatetimeIndex
- [ ] **UTI-04**: Variable-length series processing — centering, collation
- [ ] **UTI-05**: Each utility is in a separate file with proper `__all__` exports

### Data Modules (LightningDataModule)

- [ ] **MOD-01**: Module loads data from provided file paths in `prepare_data()`
- [ ] **MOD-02**: User passes module to Lightning Trainer with explicit file paths
- [ ] **MOD-03**: Classification modules support `AS_DEFINED` and `MANUAL` splitting strategies
- [ ] **MOD-04**: Forecasting modules use dataset-intrinsic split boundaries (e.g., ETT 16/4/4 months)
- [ ] **MOD-05**: Modules expose `sequence_length`, `num_classes`, `num_features` as read-only properties
- [ ] **MOD-06**: Dataloader methods return `DataLoader` instances (module returns `LightningDataModule`)

### Tests

- [ ] **TST-01**: Dataset classes yield correct shapes and types
- [ ] **TST-02**: Module properties return expected values after prepare_data
- [ ] **TST-03**: Utility functions (scaling, arff, features) produce correct output

## v2 Requirements (Deferred)

Archived code on `archive/v2-full-implementation` branch for future reintegration.

### Pydantic Registry

- **CFG-01**: Registry stores intrinsic dataset facts (name, classes, URL, splits, data_form)
- **CFG-02**: One Pydantic config class per family, instances per dataset
- **CFG-03**: Enums for typed parameters: `ScalingMethod`, `SplittingStrategy`, `ForecastingMode`

### Download and Caching

- **DL-01**: Data downloads to `~/.cache/tscollection/` on first use
- **DL-02**: Downloaded data is validated via SHA256 checksums
- **DL-03**: Cached data is reused without re-downloading
- **DL-04**: User can force cache refresh via `overwrite_cache=True`

### Factory API

- **FCT-01**: `get_module("Coffee")` returns a configured LightningDataModule instance
- **FCT-02**: `get_module` accepts bare name, qualified name, or name + family
- **FCT-03**: `get_dataset("Coffee")` returns a configured Dataset instance
- **FCT-04**: `list_modules(family="ucr")` returns all available module names for a family
- **FCT-05**: User can import family-prefixed modules directly (`from tscollection.datasets.modules import UCRCoffeeModule`)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Model architectures | Data-only package |
| Attack/robustness pipeline | Belongs in rbspaper |
| Training runners | Belongs in consuming projects |
| Conda recipe | PyPI first |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PKG-01 | Phase 1 | Done |
| PKG-02 | Phase 1 | Done |
| PKG-03 | Phase 1 | Done |
| DST-01 | Phase 2 | Done |
| DST-02 | Phase 2 | Done |
| DST-03 | Phase 2 | Done |
| DST-04 | Phase 2 | Done |
| DST-05 | Phase 2 | Done |
| UTI-01 | Phase 3 | Pending |
| UTI-02 | Phase 3 | Pending |
| UTI-03 | Phase 3 | Pending |
| UTI-04 | Phase 3 | Pending |
| UTI-05 | Phase 3 | Pending |
| MOD-01 | Phase 4 | Pending |
| MOD-02 | Phase 4 | Pending |
| MOD-03 | Phase 4 | Pending |
| MOD-04 | Phase 4 | Pending |
| MOD-05 | Phase 4 | Pending |
| MOD-06 | Phase 4 | Pending |
| TST-01 | Phase 5 | Pending |
| TST-02 | Phase 5 | Pending |
| TST-03 | Phase 5 | Pending |

**v1 Coverage:**
- v1 requirements: 22 total
- Done: 8 (Phase 1 + Phase 2)
- Pending: 14 (Phase 3-5)

**v2 Coverage:**
- v2 requirements: 11 total (CFG + DL + FCT)
- Code archived on `archive/v2-full-implementation`

---
*Requirements defined: 2026-05-08*
*Revised: 2026-05-13 — v1 simplified, pydantic/download/factory moved to v2*

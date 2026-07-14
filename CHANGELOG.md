# Changelog

All notable changes to **chronocratic-datasets** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Entries are managed with [towncrier](https://towncrier.readthedocs.io/); see
`changelog.d/` for unreleased changes.

<!-- towncrier release notes start -->

## v0.1.0a5 (2026-07-14)

### Added

- feat: add loader_strict_batch_size for uniform batch sizes; clean code and update CI: Added `loader_strict_batch_size` as an `__init__` parameter on `BaseTimeSeriesDataModule` (propagated through all 5 concrete modules) so dataloaders always produce uniform batch sizes via cycle-padding. Cleaned barrel exports and updated all CI workflows (`uv` tooling, `ty` type checking, protected files). ([#42](https://github.com/chronocratic/chronocratic-datasets/issues/42))


## v0.1.0a4 (2026-07-02)

### Added

- feat(dataloader): unified loader_mode injection for forecasting and classification modules: Added instance-level `loader_mode` property with validated setters to both forecasting and classification base classes; all 5 concrete modules propagate `loader_mode` through `super().__init__()` with None-to-self fallback in dataloader methods. 27 new TDD tests; 6 code-review findings fixed. ([#39](https://github.com/chronocratic/chronocratic-datasets/issues/39))


## v0.1.0a3 (2026-06-11)

### Changed

- Renamed `WeatherModule` to `WeatherDataModule` and `ElectricityLoadModule` to `ElectricityLoadDataModule` for naming consistency. ([#34](https://github.com/chronocratic/chronocratic-datasets/issues/34))
- Auto-create towncrier fragment on feature→dev PRs. ([#36](https://github.com/chronocratic/chronocratic-datasets/issues/36))

### Fixed

- Use `copy=True` in `DataFrame.to_numpy()` to avoid a non-writable tensor warning from PyTorch. ([#34](https://github.com/chronocratic/chronocratic-datasets/issues/34))


## v0.1.0a1 (2026-06-10) — First Alpha Release

The first pre-release of chronocratic-datasets. This alpha introduces the complete set of time series datasets, a clean and type-safe API, and full PyTorch Lightning integration.

Expect breaking changes before the 1.0 release. Feedback is welcome.

### Added

- **Forecasting datasets:** ETT, Weather, Electricity with LightningDataModule integration
- **Classification datasets:** UCR (univariate) and UEA (multivariate) benchmarks
- **ForecastingLoaderMode** enum: `RAW_SERIES`, `INPUT_TARGET`, `INPUT_ONLY`
- **ClassificationLoaderMode** enum: `SAMPLE_ONLY`, `SAMPLE_LABEL`
- **ForecastingMode** enum: `UNIVARIATE`, `MULTIVARIATE`
- **Data caching:** Automatic NPZ caching for downloaded and preprocessed data
- **Data scaling:** Configurable normalization via scikit-learn scalers
- **DDP compliance:** All data modules work with distributed training strategies
- **Utility functions:** Cache management, feature extraction, ARFF parsing, collation
- **Package structure:** Full `__init__.py` with 49 re-exported public symbols
- **BSD 3-Clause license**
- **Sphinx documentation** with autodoc-generated API reference

### Notes

- Namespace is `chronocratic.datasets` (installed via `chronocratic-datasets` on PyPI).
- Requires Python 3.12+.
- Uses PyTorch Lightning as the primary training framework integration.

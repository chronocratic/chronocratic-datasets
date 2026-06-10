# Changelog

All notable changes to **chronocratic-datasets** will be documented in this file.

## v0.1.0 (2026-06-09)

### Added

- **Forecasting datasets:** ETT, Weather, Electricity with LightningDataModule integration
- **Classification datasets:** UCR (univariate) and UEA (multivariate) benchmarks
- **ForecastingLoaderMode** enum: `SAMPLE_ONLY`, `SAMPLE_LABEL`, `INPUT_OUTPUT`
- **ClassificationLoaderMode** enum: `FULL_SERIES`, `SLIDING_WINDOW`
- **ForecastingMode** enum: `UNIVARIATE`, `MULTIVARIATE`
- **Data caching:** Automatic NPZ caching for downloaded and preprocessed data
- **Data scaling:** Configurable normalization via scikit-learn scalers
- **DDP compliance:** All data modules work with distributed training strategies
- **Utility functions:** Cache management, feature extraction, ARFF parsing, collation
- **Package structure:** Full `__init__.py` with 49 re-exported public symbols
- **BSD 3-Clause license**
- **CITATION.cff** for academic citation support
- **Sphinx documentation** with autodoc-generated API reference

### Notes

- Namespace is `chronocratic.datasets` (installed via `chronocratic-datasets` on PyPI).
- Requires Python 3.12+.
- Uses PyTorch Lightning as the primary training framework integration.

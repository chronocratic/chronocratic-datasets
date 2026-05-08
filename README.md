# tsdatasets

A reusable Python package for time-series datasets and data modules.

## Development setup

This repository is configured for:

- **uv** for dependency management
- **Ruff** for linting
- **ty** for static type checking

Typical workflow:

```bash
uv sync
uv run ruff check .
uv run ty check
```

## Package layout

- `src/tsdatasets/datasets`: dataset building blocks
- `src/tsdatasets/datamodules`: data-module abstractions and split helpers

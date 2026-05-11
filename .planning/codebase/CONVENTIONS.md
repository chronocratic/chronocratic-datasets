# Code Conventions

Last mapped: 2026-05-08

## Style and Formatting

- **Line length:** 100 characters
- **Quotes:** Single quotes for strings (ruff format `quote-style = "single"`)
- **Indentation:** Spaces (4-space standard)
- **Import sorting:** isort via ruff — `combine-as-imports=true`, `force-sort-within-sections=true`, `order-by-type=false`
- **Trailing commas:** Skipped (`skip-magic-trailing-comma = true`)

## Type Hints

- **All functions** have full type hints for parameters and return types
- Uses `from __future__ import annotations` for forward references (rbspaper pipeline, attacks, strategies)
- `TYPE_CHECKING` guard for import-heavy types (e.g., `DataLoader`, config classes) to avoid circular imports
- Uses `typing.cast()` and `ty: ignore[...]` comments for narrow type assertions where static checker is too strict
- Protocol classes with `@runtime_checkable` for duck-typed interfaces (`_ModelParamsWithSequenceLength`, `_ModelParamsWithMaxTrainLength`)

## Docstrings

- **Google style** convention enforced by ruff (`convention = "google"`)
- Module-level docstrings describe purpose briefly (e.g., `"Core orchestration logic for robust time-series experiments."`)
- Function docstrings include Args/Returns sections
- `D101` (public class docstrings) and `D107` (`__init__` docstrings) are ignored — class purpose is inferred from name and parent
- `D100` (public module docstrings) ignored

## Naming

- **Modules:** `snake_case` — `pipeline/core.py`, `data/datasets/abstract.py`
- **Classes:** `PascalCase` — `TimeSeriesDataset`, `BaseTimeSeriesDataModule`
- **Functions/variables:** `snake_case` — `run_experiment_pipeline`, `partition_tensors`
- **Private methods/attributes:** prefixed with `_` — `_go_to_idx`, `_data`, `_get_sample`
- **Constants:** `UPPER_SNAKE_CASE` — `MIN_LABELED_BATCH_LENGTH`, `CASE_COUNT`
- **Enums:** `PascalCase` — `TimeSeriesDatasetMode`, `AttackBackend`

## Module Exports

- Uses `__all__` lists at the top of modules to define public API
- `__init__.py` files re-export selectively; `F401` (unused imports) ignored on all `__init__.py` files

## Functional Patterns

- **Keyword-only calls:** Functions use `*` in signature to enforce kwargs — `_extract_clean_representations(*, partition_tensors, model, ...)`
- **compose() utility:** Transformations are chained functionally — `compose(*transformations_sequence)`
- **partial() for currying:** Pre-configure callables — `partial(expand_data_dimensionality, expand_dims_axis=...)`
- **Dict dispatch:** Strategy selection via dicts — `_get_sample_fun_map = {mode: handler}`

## Error Handling

- Raises descriptive `ValueError`, `TypeError`, `IndexError` with message strings assigned to variables first (`msg = ...; raise ValueError(msg)`)
- No broad `except:` — specific exception types
- Preflight validation in `pipeline/core.py` catches config errors before training starts

## Code Organization

- **Private helpers:** Internal functions prefixed with `_` are colocated with their public counterparts
- **Config dataclasses:** Separate `config.py` per domain (pipeline, attacks, models)
- **Enums:** Grouped in dedicated `enums/` packages
- **Abstract first:** Base ABCs live in `abstract/` subdirectories; concrete implementations at the parent level

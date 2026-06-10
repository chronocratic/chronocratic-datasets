# Contributing

Thank you for your interest in contributing to **chronocratic-datasets**!

## Development Setup

This project uses `uv` for environment management and package installation.

### Prerequisites

- Python 3.12+
- `uv` — see [docs.astral.sh/uv](https://docs.astral.sh/uv/) for installation

### Clone and Install

```bash
git clone https://github.com/chronocratic/datasets.git
cd datasets

# Install with development dependencies
uv sync --all-extras
```

## Code Style

The project follows these conventions:

- **Type hints:** All functions must have type hints for parameters and return types
- **Docstrings:** Google-style docstrings for all public functions and classes
- **Naming:** `snake_case` for functions and variables, `PascalCase` for classes
- **Imports:** Use keyword arguments for all function calls
- **Organization:** Functional programming patterns preferred; pure functions where possible

### Linting and Formatting

We use `ruff` for linting and formatting:

```bash
# Check for issues
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# Check formatting without modifying
uv run ruff format --check src/ tests/
```

## Testing

Tests are written using `pytest`. Run the test suite with:

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=src/chronocratic/datasets

# Run specific test file
uv run pytest tests/test_public_api_exports.py -v
```

### Writing Tests

- Place test files in the `tests/` directory with `test_` prefix
- Test imports from the package root: `from chronocratic.datasets import ForecastingMode`
- Keep tests focused on one behavior per test function
- Use fixtures for common setup

## Documentation

Documentation is built with Sphinx using MyST Parser for Markdown source files.

```bash
# Build documentation
uv run sphinx-build -b html docs/ docs/_build/
```

### Adding Documentation

- Write in Markdown (`.md` files) with MyST directives
- Use `.. autoclass::` for API reference pages
- Use `{doc}` for cross-references between pages
- Update `docs/index.md` to add new pages to the TOC

## Adding New Datasets

To add a new dataset:

1. Create a dataset class in `src/chronocratic/datasets/datatypes/`
2. Create a data module in `src/chronocratic/datasets/modules/`
3. Register exports in the submodule `__init__.py`
4. Update the root `src/chronocratic/datasets/__init__.py` to re-export
5. Add tests in `tests/`
6. Document in the appropriate guide page

## Pull Requests

- Write clear commit messages following conventional commits
- Ensure all tests pass before submitting
- Update documentation for user-facing changes
- Reference any related issues in the PR description

## License

By contributing, you agree that your contributions will be licensed under the
BSD 3-Clause License. See the
[LICENSE](https://github.com/chronocratic/datasets/blob/main/LICENSE) for details.

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

## Branching Strategy

This project uses a two-line branching model with `dev` and `main` as the only long-lived branches. Both maintain strictly linear histories.

### Philosophy

A linear history is not cosmetic. It makes every commit independently deployable in theory, trivial to bisect, and easy to reason about during code review. Merge commits obscure causality: did bug X come from branch A, B, or the three-way merge itself? Squash-merge and fast-forward policies eliminate that ambiguity.

`dev` is the integration branch. It collects feature work, may be unstable, and is the source for all releases. `main` is the release branch. It tracks published versions only — every commit on `main` corresponds to a tag on PyPI.

### Branch Rules

| Rule | `dev` | `main` |
| --- | --- | --- |
| Source for feature branches | Yes | No |
| Who can open PRs | Everyone | Maintainers only |
| Merge strategy | Squash only | Fast-forward only |
| Rebase allowed | Yes (before PR) | No |
| Force-push allowed | Own branches only | Never |

### Contributing Workflow

All contribution branches must be created from `dev`. All PRs from contributors target `dev`.

```bash
# 1. Sync with remote dev
git fetch origin
git checkout dev
git pull

# 2. Create feature branch from dev
git checkout -b feat/your-feature

# 3. Commit, push, open PR against dev
git push -u origin feat/your-feature
```

PRs into `dev` are **squash-merged**. This collapses all intermediate commits into a single clean commit on `dev`. The commit message is rewritten at merge time to follow conventional commits format. Your local branch may have twenty exploratory commits; `dev` sees one.

Rebase your feature branch onto `dev` before opening a PR — or immediately after reviewers request changes — so the squash target is clean and CI runs against the latest code.

### Release Workflow

PRs from `dev` into `main` are **restricted to maintainers** and must be **fast-forward merged**. No squash, no merge commit. Fast-forward means every commit on `main` was already reviewed and integrated into `dev`; the act of merging to `main` is a release assertion, not a code change.

Because `dev` squash-merges feature work and `main` fast-forwards from `dev`, both branches stay linear. `git log --oneline main` reads as a chronological changelog. `git bisect` works without navigating merge diamonds.

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): summary

[optional body]
```

Types: `feat`, `fix`, `docs`, `ci`, `refactor`, `test`, `chore`. Scope is the affected submodule. Summary is imperative mood, no period. Example:

```
feat(classification): add UCRElectricalGM12 dataset loader
```

Because contributor PRs are squash-merged, the commit on `dev` uses the PR title as the subject line. Write PR titles in conventional commits format. Local commits on your feature branch need not follow the format — they are development notes, not release history.

## Pull Requests

- Write clear commit messages following conventional commits
- Ensure all tests pass before submitting
- Update documentation for user-facing changes
- Reference any related issues in the PR description

## License

By contributing, you agree that your contributions will be licensed under the
BSD 3-Clause License. See the
[LICENSE](https://github.com/chronocratic/datasets/blob/main/LICENSE) for details.

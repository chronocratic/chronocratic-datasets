# Contributing

Thank you for your interest in contributing to **chronocratic-datasets**!

## Development Setup

This project uses `uv` for environment management and package installation.

### Prerequisites

- Python 3.12+
- `uv` — see [docs.astral.sh/uv](https://docs.astral.sh/uv/) for installation

### Clone and Install

```bash
git clone https://github.com/chronocratic/chronocratic-datasets.git
cd chronocratic-datasets

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

## Changelog

User-facing changes are recorded with [towncrier](https://towncrier.readthedocs.io/).
Instead of editing `CHANGELOG.md` directly (which causes merge conflicts), each PR
adds a small **news fragment** to `changelog.d/`. They are assembled into
`CHANGELOG.md` at release time.

### Fragments are created automatically

When you open a PR into `dev`, CI writes a fragment for you: it takes your PR
title, strips the Conventional Commits prefix (`fix:`, `feat(x):`, …) for the
body text, and commits `changelog.d/<pr>.<type>.md` to your branch. Pull before
pushing again so you don't lose it. Most PRs need nothing more — just give the
PR a clear, user-facing title with the right prefix.

The fragment type is inferred from that prefix:

| Title prefix              | Fragment type |
| ------------------------- | ------------- |
| `fix:`                    | `fixed`       |
| `feat:`                   | `added`       |
| `refactor:`, `perf:`      | `changed`     |
| `revert:`, `remove…`      | `removed`     |
| `deprecate…`              | `deprecated`  |
| `security:`               | `security`    |
| anything else             | `changed`     |

A `changelog:<type>` label, if present, overrides the inferred type.

Two cases need manual action:

- **Wrong type.** If the inferred type is off, rename the file's type (e.g.
  `42.changed.md` → `42.fixed.md`) or replace it with one you create yourself
  (see below).
- **No user-facing change.** Chores, refactors, and internal docs don't belong
  in the changelog — add the `skip-changelog` label and CI skips the fragment
  entirely.
- **Fork PRs.** CI can't push to a fork, so it only *checks* that a fragment
  exists. Add one by hand (below) before the check will pass.

### Creating a fragment by hand

Filename format is `<issue-or-pr>.<type>.md`, where `<type>` is one of `added`,
`changed`, `deprecated`, `removed`, `fixed`, or `security`:

```bash
# tied to issue/PR #42
uv run towncrier create -c "Add hourly variant of the ETT dataset." 42.added.md

# no issue number — prefix with '+'
uv run towncrier create -c "Fix NPZ cache invalidation on scaler change." +cache.fixed.md
```

If a fragment already exists for your PR number, CI leaves it untouched — your
hand-written one wins.

The body is a single sentence written for end users. Preview the assembled notes
with `uv run towncrier build --draft --version <next>`. CI also runs
`towncrier check` on PRs into `dev` as a safety net. See `changelog.d/README.md`
for details.

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

To cut a release, assemble the accumulated news fragments **on `dev`** before fast-forwarding to `main`:

```bash
git checkout dev && git pull
uv run towncrier build --version 0.1.0a2   # writes CHANGELOG.md, removes fragments
git commit -am "chore(release): v0.1.0a2"
```

Fast-forward `dev` into `main`, then create the release/tag (GitHub UI or CLI). Publishing the GitHub release triggers the PyPI upload and syncs the release notes from the matching `CHANGELOG.md` section automatically.

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

## Protected Project Files

Certain files affect the entire project and require maintainer review before merging:

| File | Why Protected |
| --- | --- |
| `pyproject.toml` | Dependencies, build settings, project metadata |
| `.vscode/**` | Shared editor config (linting, formatting, type-checking) |

When a PR modifies any of these paths, the **Protected Files Check** workflow posts an automated warning comment requesting maintainer verification. Contributors are free to suggest changes, but the PR should not be merged until a maintainer confirms:

- New dependencies are necessary and version-pinned
- Build/tool configurations don't break CI
- Changes are backward-compatible
- Editor settings benefit all contributors (not local preferences)

## Pull Requests

- Write clear commit messages following conventional commits
- Ensure all tests pass before submitting
- Update documentation for user-facing changes
- Reference any related issues in the PR description

## License

By contributing, you agree that your contributions will be licensed under the
BSD 3-Clause License. See the
[LICENSE](https://github.com/chronocratic/chronocratic-datasets/blob/main/LICENSE) for details.

# Phase 1 Context: Bug Fixes & Import Consistency

## Decisions

### Import Prefix: `src.rbspaper.*`
All internal imports use the `src.rbspaper.*` prefix. The `data/` and `adapters/` subpackages currently use bare `rbspaper.*` — these will be unified to `src.rbspaper.*` to match `pipeline/`, `models/`, `attacks/`, and `evaluation/`.

### Bug Fixes (Exact Locations)

| Bug | File | Line | Fix |
|-----|------|------|-----|
| Circular import: `augmentation/__init__` → `strategies` → `ts2vec/utils` → `ts2vec/model` → `augmentation/factories` → `augmentation/__init__` | `src/rbspaper/models/augmentation/__init__.py:3` | 3 | Lazy-import in `__init__.py` or break cycle via TYPE_CHECKING guard |
| Ridge argmax → argmin | `src/rbspaper/evaluation/protocols.py` | 119 | `np.argmax(validation_scores)` → `np.argmin(validation_scores)` (lower loss = better) |
| MAPE zero-target crash | `src/rbspaper/evaluation/forecasting.py` | 12 | Add epsilon floor: `np.abs(targets).clip(min=1e-8)` |
| `max_train_data_size` UnboundLocalError | `src/rbspaper/evaluation/evaluation.py` | 29-33 | Set `max_train_data_size` for forecasting path (line 29 only sets for classification) |
| Import inconsistency: `rbspaper.*` → `src.rbspaper.*` | `src/rbspaper/data/`, `src/rbspaper/adapters/` | various | Global find-replace, update all affected imports |

### Circular Import Chain
```
augmentation/__init__.py (imports strategies)
  → strategies.py (imports ts2vec.utils.extract_subsequences_per_row)
    → ts2vec/__init__.py (imports model)
      → ts2vec/model.py (imports augmentation.factories)
        → augmentation/__init__.py (CYCLE)
```

**Fix:** Change `augmentation/__init__.py` to lazy-import `strategies` (only export the symbol, don't import at module level). Or move the `strategies` import from `__init__.py` to a TYPE_CHECKING block and re-export via `__all__` only.

### ruff/ty Configuration
- ruff config: `ruff.toml` in project root
- ty: run via `uv run ty`
- Python 3.12 target

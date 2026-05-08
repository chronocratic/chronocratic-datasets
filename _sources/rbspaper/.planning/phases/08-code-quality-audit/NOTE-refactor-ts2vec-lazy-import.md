---
note: refactor-ts2vec-lazy-import
created: "2026-05-07"
related: src/rbspaper/models/ts2vec/__init__.py
---

## Revisit TS2Vec lazy `__getattr__` import

`src/rbspaper/models/ts2vec/__init__.py` uses a lazy `__getattr__` to avoid a circular dependency:

```
ts2vec/model.py  →  augmentation/factories.py, augmentation/strategies.py
augmentation/strategies.py  →  models/encoders.py
models/encoders.py  →  (potentially back to ts2vec modules)
```

The lazy deferral works but is a hack. Better solutions:
1. **Extract shared types** — move the interfaces both sides depend on to `models/types.py` or `models/shared.py`.
2. **Dependency injection** — pass augmentation methods into the model rather than importing factories inside.
3. **Protocol-based decoupling** — define an `@runtime_checkable` protocol for augmentation, so TS2Vec depends on a shape, not a concrete class.

Do not keep `__getattr__` as the permanent solution.

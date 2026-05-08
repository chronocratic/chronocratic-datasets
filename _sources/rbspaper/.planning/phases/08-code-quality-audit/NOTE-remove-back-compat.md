---
phase: 07-code-quality-audit
source: User request, 2026-05-07
---

## Remove All Backwards-Compatibility Code

The codebase is research-only and has not been deployed. All back-compat shims are dead weight — remove them.

### Targets

- **`src/rbspaper/configs/attacks.py`** — Entire file is "Backward-compatible re-exports" from `src.rbspaper.attacks.config`. Zero importers found anywhere in the codebase. Delete.
- **`class_count_alias` in `src/rbspaper/attacks/functional.py:129`** — Legacy alias for `num_classes` → `class_count` resolution chain. Remove the alias fallback.

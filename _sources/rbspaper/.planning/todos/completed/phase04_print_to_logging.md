---
name: Replace print() with logging in runner + pipeline
description: Convert print statements in runner.py and pipeline core to use logger for consistent structured output
type: project
resolves_phase: 04
---

`runners/py/runner.py` has ~15 `print()` calls (config summary, progress updates, errors) that should use `logger.info()` / `logger.warning()` / `logger.error()` now that `setup_logging()` is in place (Plan 03-12). Same likely applies to `src/rbspaper/pipeline/core.py`.

Files to audit: `runners/py/runner.py`, `src/rbspaper/pipeline/core.py`

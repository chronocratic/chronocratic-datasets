# Codebase Concerns

**Analysis Date:** 2026-05-05

## Tech Debt

**Circular import in models package:**
- Issue: Circular import chain prevents `test_pipeline_core.py` from loading. The chain is:
  `models/augmentation/__init__.py` -> `augmentation/strategies.py` -> `models/ts2vec/utils.py` ->
  `models/ts2vec/__init__.py` -> `models/ts2vec/model.py` -> `augmentation/factories.py` ->
  `augmentation/strategies.py` (partially initialized).
- Files: `src/rbspaper/models/augmentation/__init__.py`, `src/rbspaper/models/augmentation/strategies.py`,
  `src/rbspaper/models/augmentation/factories.py`, `src/rbspaper/models/ts2vec/model.py`
- Impact: `test_pipeline_core.py` crashes at import time with `ImportError: cannot import name 'AugmentationMethod' from partially initialized module`.
  5 of 10 tests are unreachable. This blocks full CI execution.
- Fix approach: Break the cycle by lazy-importing `extract_subsequences_per_row` inside `augmentation/strategies.py`
  using a function-local import, or by extracting the shared utilities into a separate module not tied to `ts2vec/__init__.py`.

**Duplicate config definitions across packages:**
- Issue: Attack parameters and model parameters exist in two locations with re-export wrappers.
  `src/rbspaper/attacks/config.py` is the canonical source, re-exported via `src/rbspaper/configs/attacks.py`.
  Similarly `src/rbspaper/models/config.py` is canonical, re-exported via `src/rbspaper/configs/models.py`.
- Files: `src/rbspaper/configs/attacks.py`, `src/rbspaper/configs/models.py`
- Impact: Confusion about canonical import path. Some code uses `src.rbspaper.attacks.config`,
  some uses `src.rbspaper.configs.attacks`. Risk of divergent changes if both are edited independently.
- Fix approach: Remove the `configs/` re-export layer entirely. Update all consumers to import from the canonical
  location. Mark `src/rbspaper/configs/` as deprecated.

**Inconsistent import style (absolute vs. package-relative):**
- Issue: The `pipeline/` and `models/` packages use `from src.rbspaper.*` imports.
  The `data/` and `adapters/` packages use `from rbspaper.*` imports.
- Files: `src/rbspaper/pipeline/core.py` (uses `src.rbspaper`),
  `src/rbspaper/data/modules/abstract.py` (uses `rbspaper`)
- Impact: Different import paths within the same package. Fragile to project structure changes.
  `src.rbspaper.*` imports require `src/` to be on `sys.path`; `rbspaper.*` requires the package to be installed.
- Fix approach: Standardize on one style across the codebase. If the package is installed (editable),
  prefer `from rbspaper.*` (relative to the package root). Add `ruff` or import-linter rules to enforce consistency.

**Three `ty: ignore` directives masking type issues:**
- Issue: Three locations suppress `ty` type checker errors as workarounds for unresolved type issues.
- Files: `src/rbspaper/data/datasets/abstract.py:117` (`__getitem__` override),
  `src/rbspaper/data/datasets/strategies.py:120` (union assignment),
  `src/rbspaper/models/augmentation/strategies.py:94` (method override)
- Impact: Type safety is reduced at these points. Future changes may introduce bugs that `ty` would otherwise catch.
- Fix approach: Resolve the underlying typing issues. For `__getitem__`, define a proper `__getitem__` protocol.
  For the union assignment, use explicit type narrowing.

## Known Bugs

**Ridge regression selects worst alpha (argmax instead of argmin):**
- Issue: In `_fit_ridge()`, `best_alpha = alpha_values[np.argmax(validation_scores)]` selects the alpha with
  the HIGHEST combined RMSE+MAE score. Higher scores mean worse model performance. This should be `np.argmin`.
- Files: `src/rbspaper/evaluation/protocols.py:119`
- Impact: All forecasting evaluations use the worst-performing Ridge alpha instead of the best one.
  This systematically biases forecasting accuracy results downward.
- Fix approach: Replace `np.argmax` with `np.argmin` on line 119.

**MAPE division by zero on zero-valued targets:**
- Issue: `mape_loss = np.mean(np.abs(predictions - targets) / np.abs(targets))` produces `inf` or `NaN`
  whenever any target value is exactly zero. No guard or epsilon is used.
- Files: `src/rbspaper/evaluation/forecasting.py:12`
- Impact: Any forecasting dataset containing zero-valued targets will crash or produce invalid MAPE results.
  Many real-world time series (e.g., load, demand) contain zero values.
- Fix approach: Add a small epsilon to the denominator: `np.maximum(np.abs(targets), 1e-8)`.
  Alternatively, exclude zero targets from MAPE or use a symmetric MAPE (sMAPE).

**Evaluation seed RNG created but never used:**
- Issue: `np.random.default_rng(evaluation_seed)` creates a new Generator but does not store or use it.
  Subsequent numpy random operations (e.g., in `train_test_split`) may use the legacy global RNG instead.
- Files: `src/rbspaper/evaluation/evaluation.py:63`
- Impact: Evaluation may not be fully deterministic despite the apparent seeding. The `random.seed()` and
  `pl.seed_everything()` calls are still effective, but the numpy Generator is wasted.
- Fix approach: Either pass the Generator to operations that support it, or remove the line and rely on
  `pl.seed_everything()` which seeds numpy's global state.

**`max_train_data_size` variable scope leak for forecasting:**
- Issue: In `_process_train_data_size()`, if `downstream_task` is FORECASTING, the variable
  `max_train_data_size` is never set in the classification branch (line 29-31). When the code reaches
  line 33, `max_train_data_size` may be undefined, causing an `UnboundLocalError`. The forecasting branch
  at line 39-40 sets `max_train_data_size` after the check, so the guard condition on line 33 is evaluated
  against an undefined variable.
- Files: `src/rbspaper/evaluation/evaluation.py:33`
- Impact: Forecasting evaluation crashes with `UnboundLocalError: cannot access local variable 'max_train_data_size'`
  when the downstream task is 'forecasting'.
- Fix approach: Initialize `max_train_data_size = None` before the conditional block, or restructure the logic
  so that `max_train_data_size` is set for both task types before the check.

## Security Considerations

**No sandboxing of external attack backends:**
- Issue: The ART and Torchattacks backends are invoked dynamically using `import_module()` and
  `call_with_supported_kwargs()`. Untrusted attack configurations could theoretically invoke arbitrary
  ART/Torchattacks methods.
- Files: `src/rbspaper/attacks/_backend.py`
- Impact: Low risk in a research setting. The `validate_attack_support()` check in the registry limits
  execution to known attack/method combinations.
- Current mitigation: Whitelist-based validation via `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK` registry.
- Recommendations: Maintain strict input validation. Consider adding execution timeouts for long-running attacks
  like CW or LBFGS.

**`_sources/` directory committed to repository (134 MB, 462 files):**
- Issue: The `_sources/` directory contains 134 MB of code from external projects (autotsaugment, tscar_thesis,
  tscar_jesse) committed to the git repository. It is listed in `.gitignore` but the directory still exists
  and is tracked (or was previously tracked).
- Files: `_sources/`
- Impact: Large repository size. Potential license compliance issues if these external projects have restrictive
  licenses. Code rot -- the sources may diverge from upstream.
- Fix approach: Remove from git history if previously tracked. Verify license compatibility. Consider vendoring
  or submodules instead of raw copies.

## Performance Bottlenecks

**Full partition materialization in memory:**
- Issue: `_collect_partition_tensors()` loads all train, val, and test data from DataLoaders into concatenated
  tensors. For large datasets (e.g., full ElectricityLoad or Weather), this means the entire dataset is
  held in GPU/CPU memory as raw tensors AND as encoded representations.
- Files: `src/rbspaper/pipeline/core.py:258-267`
- Impact: Out-of-memory errors on large datasets. The `_process_train_data_size()` function attempts to mitigate
  this by limiting training data to 10,000 samples for classification and 100,000 for forecasting, but raw
  tensors are still fully materialized.
- Fix approach: Use on-the-fly encoding with streaming dataloaders instead of full materialization.
  Alternatively, chunk large datasets and encode in batches.

**SVM GridSearchCV with `n_jobs=-1`:**
- Issue: The evaluation SVM protocol uses `GridSearchCV(..., n_jobs=-1)` which uses all CPU cores.
  When combined with pipeline parallelism or HPC resource constraints, this can cause oversubscription.
- Files: `src/rbspaper/evaluation/protocols.py:42`
- Impact: Resource contention on multi-user systems. Potential slowdown from context switching.
- Fix approach: Make `n_jobs` configurable. Default to a conservative value (e.g., 4) or use `os.cpu_count() // 2`.

**Redundant dual keyword arguments in attack functions:**
- Issue: Attack wrapper functions pass duplicate keyword arguments for compatibility with both ART and
  Torchattacks backends. For example, `fgsm_attack()` passes both `eps=epsilon` and `epsilon=epsilon`.
  The `call_with_supported_kwargs()` function filters to supported args, but both are computed and passed.
- Files: `src/rbspaper/attacks/functional.py` (e.g., lines 188-189, 216-217, 247-248)
- Impact: Minor performance overhead. Mainly a maintainability concern -- if either backend changes parameter
  names, the wrapper needs updating.
- Fix approach: Build the kwargs dict once in `execute_attack()` based on the resolved backend, rather than
  passing all variants.

## Fragile Areas

**Pipeline import chain is fragile to model additions:**
- Issue: The pipeline imports encode_data, which imports all three model classes (TS2Vec, AutoTCL, CoST).
  Each model class imports from the augmentation module. Adding a new model type that has additional
  dependencies risks re-creating the circular import that currently blocks test collection.
- Files: `src/rbspaper/models/encoding.py:8-10`, `src/rbspaper/models/augmentation/strategies.py:17`
- Why fragile: The import graph is tight. Any new import in `augmentation/strategies.py` that references
  `models/` will likely trigger the same circular import issue.
- Safe modification: Always use TYPE_CHECKING guards for cross-module type imports. Avoid runtime imports
  of `models/` from `augmentation/`.
- Test coverage: Only the pipeline smoke tests exercise this path, and they are currently blocked.

**Attack epsilon defaults are image-domain values (8/255 ~ 0.031):**
- Issue: Default epsilon values across attack parameter dataclasses and functional wrappers are `8.0/255.0`
  (~0.031). This is the standard perturbation budget for image-based adversarial attacks (pixel values 0-255).
  Time series data typically has different scales (normalized sensor readings, prices, etc.).
- Files: `src/rbspaper/attacks/config.py` (lines 54, 66, 80, 124, 137, 153),
  `src/rbspaper/attacks/functional.py` (multiple locations)
- Impact: Default attacks may be too weak (if data has large variance) or too strong (if data is tightly scaled).
  Results may not be comparable across datasets with different scales.
- Safe modification: Use data-aware epsilon selection. For normalized data [0,1], use absolute epsilon.
  For standardized data, use epsilon relative to the data standard deviation. Document the expected data scale.
- Test coverage: Not tested. No tests verify epsilon appropriateness for time-series scale.

**Forecasting horizon bounds defined but unused:**
- Issue: `AttackExecutionContext` has `horizon_start` and `horizon_end` fields for restricting
  adversarial perturbation to the forecast horizon. These fields are never referenced during attack execution
  or representation encoding.
- Files: `src/rbspaper/attacks/config.py:32-33`
- Impact: For forecasting attacks, perturbations may affect the entire input window instead of just the
  forecast-relevant portion, reducing scientific validity.
- Safe modification: Implement horizon-aware perturbation masking in `execute_attack()` or the backend layer.

**Encoding relies on isinstance dispatch (not extensible):**
- Issue: `encode_data()` uses `isinstance(model, TS2Vec)`, `isinstance(model, AutoTCL)`, etc. to dispatch
  to model-specific encoding. Adding a new model requires modifying the function.
- Files: `src/rbspaper/models/encoding.py:33-58`
- Impact: Violates open-closed principle. Every new model type requires editing shared encoding logic.
  Risk of forgetting to add a new branch, resulting in cryptic `NotImplementedError`.
- Safe modification: Use a registry pattern similar to the attacks registry. Each model class registers its
  own encoding strategy. Or use a protocol/interface with model-specific `encode()` methods.

## Scaling Limits

**Memory scaling with dataset size and attack count:**
- Issue: The pipeline materializes representations for all splits (train, val, test) for each downstream
  task, plus attacked representations for each attack. For N tasks, M attacks, and a dataset of size D,
  memory usage is roughly O(D * N * (1 + M)).
- Files: `src/rbspaper/pipeline/core.py:102-105`
- Current capacity: The `_process_train_data_size()` function limits training data to 10K/100K samples,
  but clean test representations are not limited.
- Limit: On a machine with 32 GB RAM, large forecasting datasets (e.g., Weather: ~365K samples, 21 features)
  with multiple attacks may exceed available memory.
- Scaling path: Implement lazy encoding (on-demand per evaluation) and discard intermediate representations.
  Use memory-mapped arrays (e.g., `np.memmap`) for large representation stores.

**SVM evaluation scales quadratically with sample count:**
- Issue: SVM training is O(N^2) to O(N^3) in the number of samples. The 10,000-sample cap in
  `_process_train_data_size()` is a mitigation but is hardcoded.
- Files: `src/rbspaper/evaluation/evaluation.py:24`
- Current capacity: 10,000 samples (classification), 100,000 samples (forecasting).
- Limit: GridSearchCV with 9 C values and a single SVM can take minutes on 10K samples.
  Adding more evaluation protocols or hyperparameter values increases time linearly.
- Scaling path: Use linear SVM or SGD-based classifiers for large datasets. Make sample caps configurable.

## Dependencies at Risk

**Optional attack backends (ART, Torchattacks):**
- Issue: Both ART (`adversarial-robustness-toolbox`) and Torchattacks are optional dependencies.
  The backend resolver uses lazy imports with fallback error messages, but the registry defines support
  matrices that assume both are available.
- Files: `src/rbspaper/attacks/_backend.py:138-147`, `src/rbspaper/attacks/_backend.py:175-183`
- Impact: If either backend is missing, attacks that depend exclusively on it will fail at runtime.
  E.g., UAP, HopSkipJump, Boundary are ART-only.
- Migration plan: Document required backends per attack in the registry. Consider a pre-flight check
  that warns about missing backends.

**`_sources/` directory contains untracked external code:**
- Issue: The `_sources/` directory contains 462 Python files from three external projects
  (autotsaugment, tscar_thesis, tscar_jesse). These are not installed as packages and have no
  dependency management.
- Files: `_sources/`
- Impact: These files appear to be reference implementations or legacy code. They are not part of the
  main `rbspaper` package but add 134 MB to the repository. Import paths are unclear.
- Migration plan: Audit usage. If unused, remove. If used as reference, move to documentation.
  If they contain needed code, integrate properly with version pinning.

## Missing Critical Features

**Clustering downstream task not implemented:**
- Issue: `TimeSeriesEvaluationDownstreamTaskEnum` includes `CLUSTERING`, and `encode_data()` has a
  clustering encoding path. However, `evaluate()` raises `ValueError` for clustering tasks.
  No clustering evaluation protocol (e.g., silhouette score, adjusted rand index) exists.
- Files: `src/rbspaper/evaluation/evaluation.py:96-98`, `src/rbspaper/evaluation/enums.py:9`
- Impact: Experiments configured with `clustering` as a downstream task will crash at evaluation time.
  The encoding infrastructure supports it, but evaluation does not.
- Fix approach: Implement `cluster_and_evaluate()` using sklearn clustering metrics (silhouette, ARI, NMI).
  Alternatively, remove CLUSTERING from the enum until implementation is ready.

**No model checkpoint versioning:**
- Issue: The pipeline saves a single `best.ckpt` per run. If an experiment is re-run with the same
  `run_name` and `reuse_trained_checkpoint=True`, the old checkpoint is reused silently.
- Files: `src/rbspaper/pipeline/core.py:198-199`
- Impact: Stale checkpoint reuse can produce incorrect results if the model or data has changed.
  No mechanism to detect checkpoint-config mismatch.
- Fix approach: Add checkpoint hash or metadata (model class, config hash) for validation.
  Support checkpoint rotation or versioned naming.

**No experiment result comparison or aggregation:**
- Issue: Results are saved as individual JSON files per run. There is no tooling to compare runs,
  aggregate metrics across experiments, or generate summary tables.
- Files: `src/rbspaper/pipeline/core.py:170-179`
- Impact: Manual effort required to analyze results across multiple experiments.
  No automated reporting or statistical testing of attack impact.
- Fix approach: Add a results aggregation CLI tool. Support loading multiple run JSONs and computing
  statistical summaries (mean, CI, paired tests).

## Test Coverage Gaps

**Pipeline core tests are non-functional (blocked by circular import):**
- What's not tested: The pipeline orchestration logic, including attack scope handling,
  artifact persistence, preflight validation, and downstream evaluation integration.
- Files: `test/test_pipeline_core.py`
- Risk: The core `run_experiment_pipeline()` function is the most complex piece of code in the project.
  Regression in this function would go undetected.
- Priority: High. Fix the circular import first (see Tech Debt section), then verify all 8 tests pass.

**Data modules have zero test coverage:**
- What's not tested: Data loading, splitting, scaling, DataLoader construction, and the
  DataModule hierarchy (classification, forecasting, UCR, UEA, ETT, etc.).
- Files: `src/rbspaper/data/modules/`, `src/rbspaper/data/datasets/`
- Risk: Data bugs (incorrect splitting, scaling leaks, shape mismatches) are common and hard to diagnose.
  Incorrect data splits can silently invalidate all experiment results.
- Priority: High. Add unit tests for each DataModule class with synthetic data.
  Test edge cases: single-sample datasets, zero-variance features, variable-length sequences.

**Model training and encoding have no tests:**
- What's not tested: TS2Vec, AutoTCL, CoST training loops, encoding pipelines, loss functions.
- Files: `src/rbspaper/models/losses.py`, `src/rbspaper/models/encoding.py`,
  `src/rbspaper/models/ts2vec/`, `src/rbspaper/models/autotcl/`, `src/rbspaper/models/cost/`
- Risk: Loss function bugs (e.g., numerical instability, incorrect gradients) are difficult to detect.
  Encoding bugs can produce silently incorrect representations.
- Priority: Medium. Add smoke tests for each model's training step and encoding path.
  Test loss functions with known inputs (e.g., identical pairs should have zero contrastive loss).

**Attack backends tested only with mocks:**
- What's not tested: Actual ART and Torchattacks backend integration. The tests mock `run_attack_backend()`.
- Files: `test/test_attacks_functional.py`, `test/test_attacks_batch.py`
- Risk: Backend API changes (ART/Torchattacks updates) may break the adapters without detection.
  Parameter mapping bugs between the config dataclasses and backend APIs are untested.
- Priority: Medium. Add integration tests with minimal models and both backends installed.
  Test parameter passthrough for each backend/attack combination.

**Evaluation logic has no tests:**
- What's not tested: Classification evaluation, forecasting evaluation, ridge fitting,
  train data size limiting, metric calculation.
- Files: `src/rbspaper/evaluation/evaluation.py`, `src/rbspaper/evaluation/protocols.py`,
  `src/rbspaper/evaluation/classification.py`, `src/rbspaper/evaluation/forecasting.py`
- Risk: The known bugs (argmax bug, MAPE division by zero, UnboundLocalError) could have been caught
  by unit tests. Metric calculation errors produce scientifically invalid results.
- Priority: High. Add unit tests for each evaluation function. Include edge cases: zero targets,
  two-class classification, single-sample evaluation.

---

*Concerns audit: 2026-05-05*

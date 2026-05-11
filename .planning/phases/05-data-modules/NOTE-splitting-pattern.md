# Note: Multi-GPU Splitting Pattern

> **Added:** 2026-05-11 (from Phase 3 discussion)
> **For:** Phase 5 planner — read before designing module architecture

## Problem

The rbspaper source does all splitting (train/val/test) inside `prepare_data()`.
In DDP, `prepare_data()` runs **only on rank 0**. Split data stored as instance
attributes never reaches the other GPUs.

## Lightning Best Practice (per docs)

- **`prepare_data()`** — rank-0-only: download, cache, write raw data to disk
- **`setup()`** — runs on **every GPU**: load cached data, split into train/val/test,
  create `Dataset` objects
- **`train_dataloader()`** — no setup logic: return `DataLoader` from datasets created in `setup()`

## Decision Needed

Split data should be created in `setup()`, not `prepare_data()`. `prepare_data()` should
only handle download + caching (delegated to Phase 4 download module).

## Impact on Config

No change to Phase 3 config — the config still provides split boundaries (indices/fractions),
these are consumed in `setup()` instead of `prepare_data()`.

## Reference

Lightning docs: https://lightning.ai/docs/pytorch/stable/data/split.html

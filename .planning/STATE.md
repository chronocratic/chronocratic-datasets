# State: tsdatasets

**Current Phase:** 1 (Phase 1 complete — Package Foundation)
**Last Updated:** 2026-05-08

## Planning Artifacts

| File | Status |
|------|--------|
| PROJECT.md | Complete |
| REQUIREMENTS.md | Complete (29 v1 requirements) |
| ROADMAP.md | Complete (7 phases) |
| CLAUDE.md | Complete (project guidelines) |

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-08 | rbspaper as primary source | Better docstrings, defensive code, existing registry |
| 2026-05-08 | Pydantic v2 for config | Typed, validated, frozen models, no extra deps for ML projects |
| 2026-05-08 | Auto-download in prepare_data() | torchtime pattern, user provides no file paths |
| 2026-05-08 | Family-prefixed imports | UCRCoffeeModule disambiguates across families |
| 2026-05-08 | Classification seq_len from data | Intrinsic property, computed in prepare_data(), read-only |
| 2026-05-08 | Forecasting seq_len user-configurable | With registry default, flexible for different use cases |
| 2026-05-08 | Cache-only download | Raw data in ~/.cache/tsdatasets/, SHA256 validated, avoids legal issues |
| 2026-05-08 | One config class per family | UCRConfig with instances per dataset, not one class per dataset |
| 2026-05-08 | Enums for typed params | ScalingMethod, SplittingStrategy, ForecastingMode — no raw strings |
| 2026-05-08 | Modules return LightningDataModule | Not DataLoader, needed for Trainer integration |

## Next Steps

Run `/gsd-plan-phase 2` to start Phase 2: Dataset Classes.

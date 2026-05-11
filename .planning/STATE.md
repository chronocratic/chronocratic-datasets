# State: tsdatasets

**Current Phase:** 2 (Phase 2 not started — Dataset Classes)
**Last Updated:** 2026-05-11

## Planning Artifacts

| File | Status |
|------|--------|
| PROJECT.md | Complete |
| REQUIREMENTS.md | Complete (29 v1 requirements) |
| ROADMAP.md | Complete (7 phases) |
| CLAUDE.md | Complete (project guidelines) |

## Progress

```
Phase 1: Package Foundation    [████████] Done
Phase 2: Dataset Classes       [        ] Not started
Phase 3: Pydantic Registry     [        ] Not started
Phase 4: Download & Caching    [        ] Not started
Phase 5: Data Modules          [        ] Not started
Phase 6: Factory API           [        ] Not started
Phase 7: Tests                 [        ] Not started
```

## Session Continuity

Last session: 2026-05-08
Stopped at: Phase 1 complete, all 5 tasks done (commit 61524ef)
Resume: Ready to plan Phase 2
Current session: 2026-05-11 — Renamed tsdatasets → tscollection.datasets (PEP 420 namespace)

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-08 | rbspaper as primary source | Better docstrings, defensive code, existing registry |
| 2026-05-08 | Pydantic v2 for config | Typed, validated, frozen models, no extra deps for ML projects |
| 2026-05-08 | Auto-download in prepare_data() | torchtime pattern, user provides no file paths |
| 2026-05-08 | Family-prefixed imports | UCRCoffeeModule disambiguates across families |
| 2026-05-08 | Classification seq_len from data | Intrinsic property, computed in prepare_data(), read-only |
| 2026-05-08 | Forecasting seq_len user-configurable | With registry default, flexible for different use cases |
| 2026-05-08 | Cache-only download | Raw data in ~/.cache/tscollection/, SHA256 validated, avoids legal issues |
| 2026-05-08 | One config class per family | UCRConfig with instances per dataset, not one class per dataset |
| 2026-05-08 | Enums for typed params | ScalingMethod, SplittingStrategy, ForecastingMode — no raw strings |
| 2026-05-08 | Modules return LightningDataModule | Not DataLoader, needed for Trainer integration |

## Next Steps

Phase 2 has no CONTEXT.md yet — discuss phase vision or plan directly.

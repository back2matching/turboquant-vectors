# Docs — turboquant-vectors

## Structure

```
docs/
├── research/          — deep research, specs, threat models
│   ├── PRIVACY-PRESERVING-EMBEDDINGS-SPEC.md    — full API + math spec
│   ├── EMBEDDING-INVERSION-THREAT-MODEL.md      — attack/defense analysis
│   ├── PRIVACY-PRESERVING-EMBEDDINGS-LANDSCAPE.md — competitive landscape
│   ├── PRIVATE-ENCODER-INTEGRATIONS.md          — 7 DBs + 5 pipeline integrations
│   ├── TURBOQUANT-VECTORS-REAL-BENCHMARKS.md    — benchmark design
│   ├── TURBOQUANT-NEXT-MOVES.md                 — ecosystem strategy (from FlockRun)
│   └── TURBOQUANT-COMBINATIONS.md               — technique combination analysis
├── marketing/         — launch materials (dev-only, not on main)
│   ├── LAUNCH-PLAN.md — sequence, timing, platforms
│   ├── REDDIT-POSTS.md — draft posts for r/LocalLLaMA, r/MachineLearning
│   └── HN-TWITTER.md  — HN submission + Twitter thread
└── plans/
    ├── EXECPLAN.md                — ACTIVE plan (0.4 roadmap)
    ├── PLAN-private-embeddings.md — completed plan (0.1-0.3)
    └── archive/                   — historical plans from FlockRun
```

## Active Plan

**[EXECPLAN.md](plans/EXECPLAN.md)** — the current forward-looking plan. Four phases:
- **A: Launch Marketing** (this week — TurboQuant trending)
- **B: Harden for 0.3.1** (this week)
- **C: Credibility** (next 2 weeks — SIFT1M, VIBE, blog)
- **D: 0.4 Features** (next month — CLI privacy, strict mode, LangChain)

## Related repos

Part of the TurboQuant family under [back2matching](https://github.com/back2matching). The ecosystem strategy docs (`TURBOQUANT-NEXT-MOVES.md`, `TURBOQUANT-COMBINATIONS.md`) live in `research/` in this repo.

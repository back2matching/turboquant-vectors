# ExecPlan: TurboQuant Ecosystem — 4 Repos Status + Actions

> Consolidated plan for the 4 public TurboQuant-related repos.
> Created: 2026-03-25. Replaces scattered PLAN-turboquant-*, PLAN-quant-sim, PLAN-turboquant-vectors.

## The 4 Repos

| Repo | PyPI | Status | Niche |
|------|------|--------|-------|
| **turboquant** | ✅ 0.1.0 | Live on PyPI | KV cache compression for HuggingFace |
| **quantsim-bench** (was quant-sim) | ✅ 0.1.0 | Live on PyPI (renamed, name collision) | "Which quantization should I use?" CLI |
| **kvcache-bench** | ✅ 0.1.0 | Live on PyPI | Benchmark Ollama KV cache types |
| **turboquant-vectors** | ✅ 0.1.0b1 | Live on PyPI (beta) | TurboQuant for embeddings/RAG |

**Deleted:** llama.cpp fork (burned PR, no value)

## What's Done

- [x] ✅ turboquant core: published to PyPI 0.1.0, CUDA acceleration, inference server
- [x] ✅ turboquant "first" claim dropped (53-star competitor exists, multiple impls same day)
- [x] ✅ quant-sim: core CLI + leaderboard feature built (11 commits, 15 tests)
- [x] ✅ kvcache-bench: v0.1.0 built (4 commits, benchmarks Ollama KV cache types)
- [x] ✅ turboquant-vectors: core + benchmarks + 16 tests (beats FAISS PQ by 7.4pp at 4-bit)

## What Needs Doing

### Priority 1: PyPI Publishes ✅ ALL DONE
- [x] ✅ turboquant 0.1.0 on PyPI
- [x] ✅ quantsim-bench 0.1.0 on PyPI (renamed from quant-sim — name collision)
- [x] ✅ kvcache-bench 0.1.0 on PyPI
- [x] ✅ turboquant-vectors 0.1.0b1 on PyPI
- [x] ✅ quant-sim README updated with `pip install quantsim-bench`
- [x] ✅ llama.cpp fork deleted
- [x] ✅ turboquant "first" claim dropped

### Priority 2: README Fixes
- [x] ✅ turboquant-vectors: README rewritten with fair claims, matched-budget benchmarks, CLI docs
- [ ] ⬜ kvcache-bench: embed chart screenshots in README, note hardcoded perplexity claims
- [ ] ⬜ quant-sim: fix misleading `pip install` hero line

### Priority 3: Real Benchmarks
- [ ] ⬜ turboquant: Llama-3.1-8B + WikiText-2 perplexity (required before llama.cpp resubmit)
- [ ] ⬜ turboquant-vectors: real-world embedding benchmark (MTEB or text-embedding-3-small)
- [ ] ⬜ kvcache-bench: add perplexity measurement (currently hardcoded)

### Priority 4: Promotion (after PyPI)
- [ ] ⬜ Post quant-sim to r/LocalLLaMA ("one command tells you which quant to use")
- [ ] ⬜ Post kvcache-bench to Ollama Discord/Reddit
- [ ] ⬜ Post turboquant-vectors to HN (TurboQuant is trending, 121-comment thread active)

### NOT Doing
- llama.cpp PR resubmit — not until Phase 3 benchmarks done + community warming
- turboquant-vectors stable release — not until real-data benchmark replaces synthetic (beta is live)
- Any "first" claims on any repo

## Relationship to FlockRun

These are **standalone tools**, not FlockRun features. They demonstrate technical credibility and could drive traffic to the GitHub profile where FlockRun lives. But they're separate projects with separate repos, separate audiences, and separate marketing.

FlockRun's CLAUDE.md and competitive analysis should NOT reference these as FlockRun features. They're portfolio pieces.

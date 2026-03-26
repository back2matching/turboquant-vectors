# ExecPlan: turboquant-vectors 0.4

> What to do next, in what order, and why.
> This is the ACTIVE plan. See `archive/PLAN-private-embeddings.md` for the completed 0.1-0.3 plan.

**Updated:** 2026-03-25
**Current:** 0.3.0 on PyPI, 123 tests, all passing
**Branch:** `dev` (6 commits ahead of `main`)

---

## Context

TurboQuant paper went viral today (March 25). TechCrunch, VentureBeat, 500+ HN points. The name is trending. We have the only pip-installable embedding privacy tool using orthogonal rotation. No competitors on PyPI. Window of attention: ~1-2 weeks.

**Priority stack:**
1. Ship marketing while attention exists (days, not weeks)
2. Harden the product (bugs, tests, 0.3.1 release)
3. Expand credibility (benchmarks, Colab, VIBE submission)
4. New features (CLI privacy commands, strict mode, 0.4)

---

## Phase A: Launch Marketing (THIS WEEK)

**Why now:** TurboQuant trending. Every day of delay loses attention.

| # | Task | Effort | Deliverable |
|---|------|--------|-------------|
| A1 | Finalize Reddit posts | 1h | Post to r/LocalLLaMA and r/MachineLearning |
| A2 | Post Show HN | 1h | HN submission linking to GitHub |
| A3 | Twitter/X thread | 1h | 7-tweet thread with code examples |
| A4 | Colab notebook | 4h | Runnable proof: rotate, attack fails, search identical |

**A4 is the highest ROI.** Every platform post should link to the Colab. A runnable demo beats 10 pages of docs.

**Colab notebook contents:**
1. `pip install turboquant-vectors`
2. Generate embeddings with sentence-transformers
3. Train a category classifier on originals (shows ~89% accuracy)
4. Rotate with PrivateEncoder
5. Same classifier on rotated: drops to ~11%
6. Search recall: identical (1.000)
7. Compression demo: 8x smaller, recall > 96%

**Gate:** A4 done before A1-A3 go live.

---

## Phase B: Harden for 0.3.1 (THIS WEEK)

**Why:** People will pip install after seeing posts. First impression matters.

| # | Task | Effort | Deliverable |
|---|------|--------|-------------|
| B1 | Fix B8 — IP/L2 metric issue in CompressedPrivateVectors | 2h | Either fix or document cosine-only |
| B2 | Fix B9 — NaN validation in rekey/unrotate | 30m | Consistency with rotate() |
| B3 | Add CHANGELOG.md | 30m | 0.1.0 through 0.3.1 |
| B4 | Bump to 0.3.1, publish to PyPI | 30m | `pip install turboquant-vectors==0.3.1` |
| B5 | Merge dev -> main (strip docs/) | 1h | Clean public release |

**B1 decision:** The simplest correct fix is to document `CompressedPrivateVectors.search()` as cosine-only and raise `ValueError` for ip/l2. These metrics on compressed+rotated data with mixed normalization are misleading anyway.

---

## Phase C: Credibility (NEXT 2 WEEKS)

| # | Task | Effort | Deliverable |
|---|------|--------|-------------|
| C1 | Benchmark on SIFT1M + GloVe (ann-benchmarks datasets) | 2d | Standard numbers everyone compares against |
| C2 | Benchmark against RaBitQ / SAQ at matched bit budgets | 2d | Head-to-head with current SOTA |
| C3 | Submit to VIBE benchmark | 3d | Listed on public benchmark |
| C4 | Blog post: "Your RAG Embeddings Are Not Private" | 1d | Dev.to / Hashnode |

**C1-C2 are the foundation for C3.** VIBE submission requires competitive numbers on their datasets.

**Key differentiator vs SAQ/RaBitQ:** TurboQuant is data-oblivious (no PCA, no training). This means:
- No information about the data distribution leaks through the quantization
- Instant compression (no training step)
- Works identically on any domain without tuning

---

## Phase D: 0.4 Features (NEXT MONTH)

| # | Task | Effort | Deliverable |
|---|------|--------|-------------|
| D1 | CLI privacy commands: `keygen`, `rotate`, `keyinfo` | 2d | Full CLI for privacy workflows |
| D2 | Per-tenant key rotation guide | 1d | docs/ guide for multi-tenant deployments |
| D3 | Investigate SPARSE "strict mode" | 3d | Optional noise layer on top of rotation |
| D4 | LangChain integration PR | 2d | `PrivateEmbeddings` wrapper in LangChain community |

**D3 rationale:** SPARSE (ICLR 2026) applies concept-aware elliptical noise to privacy-sensitive dimensions. Combined with rotation, this would provide defense-in-depth against known-plaintext attacks. Rotation handles the zero-loss base case; SPARSE handles the paranoid case at ~1-3% recall cost.

**D4 rationale:** LangChain is the #1 RAG framework. A community integration PR would put turboquant-vectors in front of every LangChain user who cares about privacy.

---

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-03-25 | Marketing before features | TurboQuant trending, attention window closing |
| 2026-03-25 | Cosine-only for CompressedPrivateVectors.search() | IP/L2 on mixed-norm compressed data is misleading |
| 2026-03-25 | Keep legacy RNG in core.py | Changing would break existing compressed indexes |
| 2026-03-25 | Don't fix C5 (4-bit codebook) | Matches TurboQuant paper implementation, changing breaks reproducibility |

---

## References

- [ISSUES.md](../../ISSUES.md) — bug tracker, open items
- [CLAUDE.md](../../CLAUDE.md) — repo operating instructions
- [Completed plan](archive/PLAN-private-embeddings.md) — phases 1-7 history
- [Threat model](../research/EMBEDDING-INVERSION-THREAT-MODEL.md)
- [Competitive landscape](../research/PRIVACY-PRESERVING-EMBEDDINGS-LANDSCAPE.md)
- [Benchmark research](../research/TURBOQUANT-VECTORS-REAL-BENCHMARKS.md)

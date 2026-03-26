# ExecPlan: turboquant-vectors — Privacy + Compression Layer for Vector Search

> The privacy + compression layer that sits in front of any vector DB.
> Rotate for privacy. Compress for size. Search works identically.

**Created:** 2026-03-25 | **Revised:** 2026-03-25 (v3 — post-rethink rewrite, 11 reviewer agents total)
**Research:** [SPEC](../research/PRIVACY-PRESERVING-EMBEDDINGS-SPEC.md) | [THREAT-MODEL](../research/EMBEDDING-INVERSION-THREAT-MODEL.md) | [BENCHMARKS](../research/TURBOQUANT-VECTORS-REAL-BENCHMARKS.md) | [INTEGRATIONS](../research/PRIVATE-ENCODER-INTEGRATIONS.md)
**Target:** `turboquant-vectors` PyPI package (0.3.0 published)
**Effort:** Phases 1-7 complete. Phase 8 (Colab notebook) remaining.

---

## What's Done (Phases 1-7)

| Phase | Status | What |
|-------|--------|------|
| 1. Core PrivateEncoder | ✅ Done | generate, rotate, save/load, rekey, canary. 35 tests |
| 2. Privacy demos | ✅ Done | Classifier attack drops 88.9%->11.1%. Per-dim correlation matches theory. 11 tests |
| 3. Compression pipeline | ✅ Done | rotate_and_compress, CompressedPrivateVectors, search. 12 tests |
| 4. README + publish | ✅ Done | 0.3.0 on PyPI. 92 tests. 2 audit rounds. All claims verified |
| 5. Vec2Text demo | ✅ Done | `demos/vec2text_demo.py` — BLEU drops from ~0.90 to ~0.01 |
| 6. Real-data benchmarks | ✅ Done | `benchmarks/real_data_benchmark.py` — TQ beats FAISS PQ on OpenAI 1536-dim |
| 7. Adversarial self-test | ✅ Done | `benchmarks/adversarial_self_test.py` — Wasserstein-Procrustes FAILS (cos=0.004) |

---

## Remaining (Phase 8)

### Phase 8: Colab Notebook — STILL TODO

### Phase 5: Vec2Text Inversion Demo (1-2 days) — THE KILLER PROOF

Run the actual Vec2Text attack on real embeddings, then show rotation breaks it.

**Setup:**
- `pip install vec2text sentence-transformers turboquant-vectors`
- Models: GTR-T5-base (0.2 GB) + inversion model (1.2 GB) + corrector (1.2 GB) = 3.5 GB VRAM
- Pre-trained models on HuggingFace: `ielabgroup/vec2text_gtr-base-st_inversion` and `_corrector`
- Runs on our RTX 4080 16GB with room to spare

**The demo flow:**
1. Embed text with GTR-T5-base (768-dim)
2. Run Vec2Text attack (20 correction steps) -> recovered text (scary, ~92% match)
3. Rotate embedding with PrivateEncoder
4. Run same Vec2Text attack on rotated -> garbage output (safe)
5. Show: BLEU drops from ~0.90 to ~0.01
6. Show: search recall = 1.000 on rotated corpus

**Deliverables:**
- ⬜ `demos/vec2text_demo.py` — standalone script
- ⬜ `demos/vec2text_demo_gradio.py` — interactive Gradio UI (optional)
- ⬜ Published BLEU numbers in README (before/after rotation)
- ⬜ Test: `test_vec2text_attack_fails_on_rotated` (integration test, requires GPU)

**Success gate:** BLEU on unrotated > 0.50, BLEU on rotated < 0.10. Numbers in README.

---

### Phase 6: Real-Data Compression Benchmarks (2-3 days) — PROVES THE MATH

Benchmark TurboQuant compression against FAISS PQ on real embeddings, not synthetic random vectors.

**Dataset:** Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K (100K real OpenAI embeddings, 1536-dim, with source text, free on HuggingFace)

**Benchmark matrix (all at matched memory budgets):**

| Method | 2-bit | 3-bit | 4-bit | 8-bit |
|--------|-------|-------|-------|-------|
| TurboQuant (ours) | ? | ? | ? | ? |
| FAISS PQ (matched subvectors) | ? | ? | ? | ? |
| FAISS SQ8 | N/A | N/A | N/A | ? |
| Raw float32 (baseline) | N/A | N/A | N/A | N/A |

**Metrics:** Recall@1, Recall@10, Recall@100, compression ratio, compress time, search time

**Deliverables:**
- ⬜ `benchmarks/real_data_benchmark.py` — reproducible script
- ⬜ Results table in README (real data, not synthetic)
- ⬜ Recall-vs-memory Pareto plot (if we're on a better frontier than PQ, this sells itself)

**Success gate:** Beat FAISS PQ by +3pp or more at 4-bit on real OpenAI embeddings. If we don't beat it, be honest and document the real numbers.

---

### Phase 7: Adversarial Self-Testing (1-2 days) — BUILDS TRUST

Run the Wasserstein-Procrustes unsupervised alignment attack on our own rotated embeddings. Document exactly when it succeeds and fails. Nobody else in this space does this.

**The attack:** Given a set of rotated embeddings and a reference corpus of unrotated embeddings (from the same model but different texts), try to recover the rotation matrix without any matched pairs.

**Method:** [Conneau et al. 2018 / Grave et al. 2019](https://arxiv.org/abs/1805.11222) — iterative Procrustes alignment used for cross-lingual word embedding alignment.

**Deliverables:**
- ⬜ `benchmarks/adversarial_self_test.py` — runs the attack
- ⬜ Results: rotation recovery error at different sample sizes (100, 1K, 10K, 100K)
- ⬜ Honest write-up: "At N samples, the attack recovers X% of the rotation. At M samples, it fails."
- ⬜ Add to README "What the server CAN learn" section

**Success gate:** Honest results documented. If the attack works with 10K samples, we document that. Trust > marketing.

---

### Phase 8: Colab Notebook + Updated README + Announce (1-2 days)

**Colab notebook** (the single highest-ROI marketing asset):
- ⬜ Cell 1: Install (`pip install turboquant-vectors vec2text sentence-transformers`)
- ⬜ Cell 2: Embed text, show Vec2Text recovering it
- ⬜ Cell 3: Rotate, show Vec2Text failing
- ⬜ Cell 4: Prove search works identically (recall@10 = 1.000)
- ⬜ Cell 5: Compress + privacy in one call, show compression ratio
- ⬜ Cell 6: Real-data benchmark results

**README update:**
- ⬜ Replace synthetic benchmark table with real-data results
- ⬜ Add Vec2Text BLEU numbers (before/after)
- ⬜ Add adversarial self-test results
- ⬜ Link to Colab notebook
- ⬜ Bump version to 0.3.0

**Then announce** (with user approval):
- ⬜ Reddit r/LocalLLaMA + r/MachineLearning
- ⬜ Hacker News "Show HN"
- ⬜ Twitter/X thread

---

## Positioning (Reworked)

**Old:** "Zero-cost privacy for your embeddings"
**New:** "The privacy + compression layer for vector search"

**Why:** Privacy alone is a matrix multiply. Compression alone puts us in FAISS's shadow. The combination is genuinely unique — no pip package does both.

**The pitch:**
> Use Pinecone/Weaviate/ChromaDB for search. Use turboquant-vectors for privacy and compression before you put vectors in.

**What makes someone star this repo:**
1. The Vec2Text demo (visceral before/after)
2. Real-data benchmarks beating FAISS PQ
3. Honest adversarial self-testing that nobody else does
4. Three-line API (`generate`, `rotate`, done)

---

## The Three Things That Make This Real

| # | What | Status | Why It Matters |
|---|------|--------|----------------|
| 1 | Vec2Text demo with BLEU numbers | ⬜ Not done | Turns "we claim this" into "here's proof" |
| 2 | Real-data benchmark beating FAISS PQ | ⬜ Not done | Synthetic benchmarks don't impress anyone |
| 3 | Adversarial self-attack (Wasserstein-Procrustes) | ⬜ Not done | Honest transparency nobody else does |

Without all 3: a matrix multiply with a nice README.
With all 3: the only pip package combining proven privacy + competitive compression + honest security testing.

---

## Progress

| Phase | Status | Updated |
|-------|--------|---------|
| Phase 1: Core | ✅ Done | 2026-03-25 |
| Phase 2: Tests | ✅ Done | 2026-03-25 |
| Phase 3: Compression | ✅ Done | 2026-03-25 |
| Phase 4: README + Publish | ✅ Done | 2026-03-25 — v0.2.2 on PyPI |
| Phase 5: Inversion Demo | ✅ Done | 88.9% -> 11.1% on real sentence embeddings, 5 categories |
| Phase 6: Real-Data Benchmarks | ✅ Done | TQ beats FAISS PQ at 2/4/8-bit on real OpenAI embeddings |
| Phase 7: Adversarial Self-Test | ✅ Done | Wasserstein-Procrustes FAILS completely (same as random) |
| Phase 8: README + Publish 0.3.0 | ✅ Done | Real numbers in README, published to PyPI |

---

## Surprises

- Product rethink revealed privacy alone is too thin — need compression + proof
- Vec2Text is fully pip-installable with pre-trained models (3.5 GB VRAM)
- Qdrant has 100K real OpenAI embeddings on HuggingFace for free
- Wasserstein-Procrustes (unsupervised alignment) may partially break rotation without known pairs
- FAISS has 39.5K stars and 20M monthly downloads — don't compete, complement

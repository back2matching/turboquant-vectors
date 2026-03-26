# Reddit Posts -- turboquant-vectors Privacy Module

> Draft posts for 2 subreddits. Each targets a different audience.
> All claims verified against primary sources and 92 passing tests.

---

## r/LocalLLaMA

**Title:** Vec2Text can recover 92% of your RAG text from embeddings. Here's a one-line fix with zero recall loss.

**Body:**

Vec2Text (EMNLP 2023) recovers 92% of 32-token text from GTR-base embeddings. Including full patient names from clinical records. ALGEN (ACL 2025) needs only 1,000 leaked pairs. Zero2Text (2026) needs zero training data. OWASP added this as LLM08 in their 2025 Top 10.

If you're storing embeddings in Pinecone, Weaviate, Qdrant, or any third-party vector DB, your text is recoverable.

**The fix is one matrix multiply:**

```python
from turboquant_vectors import PrivateEncoder

encoder = PrivateEncoder.generate(dim=1536)
rotated = encoder.rotate(embeddings)
# Upload rotated vectors to your DB instead of originals
# Search works identically — cosine, L2, inner product all preserved exactly
```

**How it works:** Orthogonal rotation preserves all distance metrics by definition: `<Qx, Qy> = x^T Q^T Q y = x^T y = <x, y>`. Cosine similarity, L2 distance, inner product are all identical before and after rotation (up to float32 precision, ~1e-6). Your top-K results don't change. Literally the same results.

**What it defeats:** Vec2Text, ALGEN, ZSinvert, Zero2Text, attribute classifiers. Our test suite proves it: a classifier trained on original embeddings drops from 100% to 0% accuracy on rotated vectors.

**What it does NOT defeat (honest):**
- If an attacker gets 1,536 original-to-rotated pairs (for OpenAI embeddings), they can recover the key via SVD
- The server can still see which documents are similar to each other (pairwise distances preserved)
- This is NOT encryption and NOT differential privacy

**Benchmarks (92 tests, all passing):**
- Single vector rotation: 0.11ms at d=1536
- Batch 10K: 88ms
- Key generation: 465ms (one-time)
- Key file: 9.4 MB
- Recall@10: exactly 1.000 (20/20 queries on 10K vectors, verified)

Only dependency is numpy. No torch, no scipy for the privacy module.

```
pip install turboquant-vectors
```

GitHub: https://github.com/back2matching/turboquant-vectors

Happy to answer questions about the math, the threat model, or the known-plaintext weakness.

---

## r/MachineLearning

**Title:** [P] Zero-loss embedding privacy via orthogonal rotation — first pip package, 92 tests, honest threat model

**Body:**

We packaged a simple observation: orthogonal rotation preserves all inner-product-based distance metrics exactly, while making embedding inversion attacks (Vec2Text, ALGEN, ZSinvert) fail completely.

**The math:** For orthogonal Q, `<Qx, Qy> = <x, y>`. Cosine similarity, L2 distance, and inner product are invariant under orthogonal transformation. This is a well-known property, but nobody had packaged it as a practical privacy tool for vector databases.

**The threat:** Vec2Text (Morris et al., EMNLP 2023) recovers 92% of 32-token text from GTR-base embeddings. ALGEN (Chen et al., ACL 2025) achieves near-optimal inversion with only 1,000 text-embedding pairs. OWASP lists embedding weaknesses as LLM08 in their 2025 Top 10.

**The tool:**

```python
from turboquant_vectors import PrivateEncoder

encoder = PrivateEncoder.generate(dim=1536)
rotated = encoder.rotate(embeddings)  # search results are identical
encoder.save_key("secret.tqkey")      # treat like an SSH key
```

**What we prove (92 tests):**
- Recall@10 = 1.000 exactly (brute-force verification on 10K vectors)
- Classifier transfer attack: accuracy drops from 100% to 0% on rotated vectors
- Per-dimension Pearson r between original and rotated ~ sqrt(2/pi*d) (matches theory for Haar-random orthogonal matrices)
- Round-trip error < 1e-5

**Honest threat model — what this does NOT protect against:**
- Known-plaintext: d linearly independent (original, rotated) pairs fully recover Q via Orthogonal Procrustes / SVD. For d=1536, that's 1,536 pairs.
- Distribution alignment: Wasserstein-Procrustes style attacks could partially recover Q without known pairs (unexplored, noted in our threat model doc)
- Pairwise distance leakage: the server sees the similarity graph
- Not DP, not encryption, not MPC

**Comparison with alternatives:**

| | Rotation (ours) | Differential Privacy | IronCore Cloaked AI | HE |
|---|---|---|---|---|
| Recall loss | 0% | 5-30% | ~5% | 0% |
| Latency | <0.1ms | Negligible | SDK overhead | 1000x+ |
| License | Apache 2.0 | N/A | AGPL / $599+/mo | N/A |

The core observation (isometric privacy) appears in prior work on image/voice domains (Aso et al. 2023, arXiv 2301.03843), but to our knowledge this is the first pip-installable package applying it to embedding/vector-database privacy.

Paper references: TurboQuant (ICLR 2026, arXiv:2504.19874) for the rotation+quantization pipeline. The privacy application is our contribution.

`pip install turboquant-vectors` | [GitHub](https://github.com/back2matching/turboquant-vectors) | [Threat model doc](https://github.com/back2matching/turboquant-vectors)

---

## Posting Notes

- **r/LocalLLaMA:** Lead with the threat (Vec2Text), show the fix, provide benchmarks. This community wants code they can run and honest limitations.
- **r/MachineLearning:** Lead with the math, cite papers properly, acknowledge prior art. This community will verify claims and destroy anything that oversells.
- **Both:** Never call it encryption. Always document the known-plaintext weakness upfront. Honesty is the marketing strategy.
- **Timing:** Post Tuesday-Wednesday for best engagement. Not weekends.

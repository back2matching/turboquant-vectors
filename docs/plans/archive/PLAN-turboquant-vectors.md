# ExecPlan: TurboQuant Vector Compressor

> Compress embeddings 8x instantly. No training needed. First pip package using TurboQuant for vector search.

**Created:** 2026-03-25
**Research:** Google TurboQuant blog explicitly covers vector search, not just KV cache.
**Core insight:** TurboQuant is data-oblivious (no k-means training like FAISS PQ). Instant compression.

---

## The Problem

RAG on consumer hardware is memory-constrained:
- 1M documents at 1536-dim float32 = **6.1 GB** just for the embedding index
- Add a 30B model (12GB) + KV cache (4GB) = exceeds 24GB GPU
- FAISS Product Quantization helps but requires slow k-means training per dataset

**Nobody has packaged TurboQuant's instant vector compression as a pip tool.**

## The Product

```python
from turboquant_vectors import compress, search

compressed = compress(embeddings, bits=4)  # 307 MB -> 38 MB
indices, scores = search(compressed, query, top_k=10)
```

```bash
pip install turboquant-vectors
tq-vectors compress embeddings.npy --bits 4
tq-vectors search embeddings.tqv.npz query.npy --top-k 10
```

---

## Phase 1: Core Library ✅

### 1.1 Compression API
- ✅ `compress(vectors, bits=4)` — numpy array in, CompressedVectors out
- ✅ `decompress(compressed)` — restore to float32
- ✅ Support 1, 2, 3, 4, 8 bit compression
- ✅ Batched quantization (handles 100K+ vectors without OOM)

### 1.2 Search API
- ✅ `search(compressed, query, top_k)` — cosine similarity nearest neighbor
- ✅ Batch query support (queries as 2D array)
- ✅ Cached decompression for repeated search calls

### 1.3 Save/Load + CLI
- ✅ Save/load compressed vectors to .npz files
- ✅ CLI: `tq-vectors compress`, `tq-vectors search`, `tq-vectors info`

---

## Phase 2: Fair Benchmarks ✅

### 2.1 Reproducible FAISS Comparison
- ✅ Matched storage budgets (same bytes/vector for both methods)
- ✅ Fixed seeds, fully reproducible: `python benchmarks/compare_faiss.py`
- ✅ 50K x 1536-dim results:
  - 4-bit: TurboQuant **83.8%** vs FAISS PQ 75.8% (**+8.0pp**)
  - 2-bit: TurboQuant **52.8%** vs FAISS PQ 45.7% (**+7.1pp**)
  - 1-bit: FAISS PQ **29.0%** vs TurboQuant 26.0% (FAISS wins by 3.0pp)
- ✅ README updated with honest claims (no inflated numbers)

### 2.2 Tests
- ✅ 16 tests passing (13 core + 3 benchmark reproducibility)

---

## Phase 3: Publish ✅

- ✅ Agent team review (Eng Ops: "go for beta publish", toned down claims)
- ✅ User approval (logged in via Playwright, created API token)
- ✅ PyPI: `pip install turboquant-vectors==0.1.0b1` — https://pypi.org/project/turboquant-vectors/0.1.0b1/

---

## Future (post-publish)

- ⬜ Inner product distance mode
- ⬜ FAISS index wrapper (`compress_faiss_index`)
- ⬜ ChromaDB integration
- ⬜ GloVe dataset benchmark (match paper directly)
- ⬜ Search latency optimization (avoid full decompression)

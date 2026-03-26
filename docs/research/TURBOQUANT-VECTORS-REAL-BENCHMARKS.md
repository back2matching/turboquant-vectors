# turboquant-vectors: Real Benchmark Research

> What datasets, metrics, protocols, and privacy demos we need to graduate from synthetic benchmarks to credible real-world claims.

**Date:** 2026-03-25
**Current state:** 0.3.0 on PyPI. Real-data benchmarks shipped (`benchmarks/real_data_benchmark.py`). TQ beats FAISS PQ on 10K OpenAI embeddings at 2/4/8-bit.

---

## 1. Standard ANN Benchmark Datasets

### The Big 5 (what everyone uses)

| Dataset | Dimensions | Vectors | Queries | Distance | Size | Source |
|---------|-----------|---------|---------|----------|------|--------|
| **SIFT1M** | 128 | 1,000,000 | 10,000 | Euclidean | 501 MB (HDF5) | ann-benchmarks.com |
| **GloVe-100** | 100 | 1,183,514 | 10,000 | Angular | 463 MB (HDF5) | ann-benchmarks.com |
| **GloVe-200** | 200 | 1,183,514 | 10,000 | Angular | 918 MB (HDF5) | ann-benchmarks.com |
| **GIST** | 960 | 1,000,000 | 1,000 | Euclidean | ~3.6 GB (HDF5) | ann-benchmarks.com |
| **Deep1M** | 96 | 9,990,000 | 10,000 | Angular | ~3.8 GB (HDF5) | ann-benchmarks.com |

All datasets from ann-benchmarks are HDF5 format with pre-split train/test and ground truth top-100 nearest neighbors.

**Download:** `python -c "from ann_benchmarks.datasets import get_dataset; get_dataset('sift-128-euclidean')"`
Or direct: `http://ann-benchmarks.com/sift-128-euclidean.hdf5`

### Standard Metrics (what gets plotted)

- **Recall@K** (K=1, 10, 100) -- the primary metric. What fraction of true top-K neighbors appear in your approximate top-K?
- **QPS** (queries per second) -- throughput at a given recall level
- **Build time** -- index construction time
- **Memory footprint** -- compressed index size in bytes
- **1-Recall@1** -- used by FAISS wiki (lower is better)

### What matters for us (compression library, not ANN index)

We are a **codec**, not a full ANN index. The right comparison is:
- Our compressed brute-force vs. FAISS PQ compressed brute-force (same storage budget)
- Recall@10 at matched compression ratio (e.g., 4 bits/dim for both)
- This is what the FAISS "Vector codec benchmarks" wiki page does

---

## 2. Real RAG Embedding Datasets on HuggingFace

### Tier 1: OpenAI embeddings (the killer datasets)

| Dataset | Model | Dim | Vectors | Text? | Download |
|---------|-------|-----|---------|-------|----------|
| **Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K** | text-embedding-3-small | 1536 | 100K | Yes (title+text) | `datasets.load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K")` |
| **Qdrant/dbpedia-entities-openai3-text-embedding-3-small-512-100K** | text-embedding-3-small | 512 | 100K | Yes | Same pattern |
| **Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M** | text-embedding-3-large | 1536 | 1M | Yes | Same pattern |
| **Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M** | text-embedding-3-large + ada-002 | 3072 + 1536 | 1M | Yes | ~31 GB |

**Why these are gold:** Real OpenAI embeddings from real DBpedia text. 1M vectors at 1536-dim is exactly the RAG use case. The 3072-dim dataset also includes ada-002 embeddings. Columns: `_id`, `title`, `text`, `text-embedding-*-embedding`.

### Tier 2: Cohere Wikipedia embeddings

| Dataset | Model | Dim | Vectors | Text? | Download |
|---------|-------|-----|---------|-------|----------|
| **Cohere/wikipedia-22-12-en-embeddings** | multilingual-22-12 | 768 | 35.2M | Yes (title+text) | `datasets.load_dataset("Cohere/wikipedia-22-12-en-embeddings")` |
| **Cohere/wikipedia-22-12-simple-embeddings** | multilingual-22-12 | 768 | ~500K | Yes | Same pattern |
| **Cohere/wikipedia-2023-11-embed-multilingual-v3** | embed-multilingual-v3 | 1024 | ~250M | Yes | Massive, stream only |

**Why these matter:** Cohere embed-v3 is widely deployed. 35M vectors at 768-dim is a realistic production workload. Columns: `id`, `title`, `text`, `emb`.

### Tier 3: Other real embeddings

| Dataset | Model | Dim | Vectors | Notes |
|---------|-------|-----|---------|-------|
| **Supabase/wikipedia-en-embeddings** | OpenAI (ada-002) + MiniLM + GTE-small | 1536/384/384 | 224K | Simple English Wikipedia |
| **Supabase/dbpedia-openai-3-large-1M** | text-embedding-3-large | 3072 | 1M | Alternate source for same DBpedia data |
| **VIBE datasets** | ArXiv/ImageNet embeddings | Various | Various | From vector-index-bench/vibe on HuggingFace |

### Recommended benchmark dataset order

1. **Qdrant/dbpedia-openai3-text-embedding-3-small-1536-100K** -- start here. 100K vectors, 1536-dim, small download, real OpenAI embeddings.
2. **Cohere/wikipedia-22-12-simple-embeddings** -- 500K vectors, 768-dim, different model family.
3. **Qdrant/dbpedia-openai3-text-embedding-3-large-1536-1M** -- scale test. 1M vectors.
4. **SIFT1M** (from ann-benchmarks) -- traditional benchmark for apples-to-apples with published numbers.
5. **GloVe-100** (from ann-benchmarks) -- another standard reference point.

---

## 3. What Would Be Impressive Numbers?

### Known FAISS PQ performance on real data

From FAISS wiki "Indexing 1M vectors" and "Vector codec benchmarks":

**SIFT1M (128-dim, Euclidean):**
- IVF64+PQ8x8 (8 bytes/vector): recall@1 = 0.92 at 20K QPS
- IndexIVFPQFastScan, nbits=4, M=64, nprobe=16: recall@10 ~ 0.83
- IndexIVFPQFastScan, nbits=4, M=64, nprobe=32: recall@10 ~ 0.85
- Pure PQ (no IVF, no re-ranking): recall@1 varies by code size

**At 4 bits per dimension (our target):**
- FAISS PQ4 on Deep1M: recall@1 ~ 0.35-0.45 (pure codec, no IVF)
- FAISS PQ8 on Deep1M: recall@1 ~ 0.50-0.60 (pure codec)
- With IVF + re-ranking, recall goes up significantly

### Competing approaches at 4-bit

| Method | Recall@10 (approx) | Notes |
|--------|-------------------|-------|
| FAISS PQ (4-bit subquantizers) | 0.40-0.55 | Pure codec, no IVF |
| FAISS PQ (8-bit, 8 bytes/vec) | 0.65-0.80 | The standard baseline |
| ScaNN (4-bit PQ + anisotropic) | 0.50-0.65 | Better than vanilla PQ on GloVe |
| RaBitQ (4-bit) | 0.60-0.75 | SIGMOD 2025, strong on Euclidean |
| Extended-RaBitQ (4-bit) | 0.65-0.80 | SIGMOD 2025, multi-bit generalization |
| Weaviate BQ (binary, 1-bit) | 0.60-0.80 | 32x compression, requires rescoring |
| HuggingFace int8 SQ | ~0.99 | 4x compression only, near-lossless |

### What we need to show

| Claim | Threshold | Why it matters |
|-------|-----------|---------------|
| Beat FAISS PQ at 4-bit on real OpenAI embeddings | +3-5pp recall@10 | Core claim validation |
| Beat FAISS PQ at 2-bit on real embeddings | +5pp recall@10 | Nobody else does 2-bit well |
| Competitive with RaBitQ at 4-bit | Within 2pp | RaBitQ is the new SOTA |
| Compression speed < 2x FAISS PQ build time | Same order of magnitude | Can't be 100x slower |
| Works on 1536-dim OpenAI embeddings | Any reasonable recall | Proves it works on RAG dimensions |

### Is +5pp on real data significant?

**Yes, very.** In the ANN benchmark world, algorithms fight over 1-2pp differences at the Pareto front. A consistent +5pp at matched compression ratio would be a notable result. For context:
- ScaNN's anisotropic quantization was a major paper for ~3-5pp improvement over PQ
- RaBitQ (SIGMOD 2024/2025) gained attention for ~5-10pp over PQ at same bit budget
- If we match or beat RaBitQ, that's publishable

### Compression speed matters too

- FAISS PQ training on 1M vectors: seconds to minutes
- If our rotation + quantization takes similar time: fine
- If it takes 10x longer: need to justify with recall gains
- Build time is plotted on ann-benchmarks, people care

---

## 4. Privacy Benchmarks That Would Actually Impress

### Vec2Text: The gold standard attack

**What it is:** Given an embedding vector, reconstruct the original text. Published results:
- GTR-base on Wikipedia: 92% exact match, 97.3 BLEU on 32-token texts
- OpenAI ada-002 on MSMARCO: 60.9% exact match, 83.4 BLEU on 32-token texts
- ada-002 on 128-token texts: 8.0% exact match, 55.0 BLEU

**Can we run it?** Yes, with constraints:

| Component | Requirement |
|-----------|-------------|
| **pip install** | `pip install vec2text` |
| **GTR-base models** | `ielabgroup/vec2text_gtr-base-st_inversion` + `ielabgroup/vec2text_gtr-base-st_corrector` on HuggingFace |
| **ada-002 models** | `jxm/vec2text__openai_ada002__msmarco__msl128__corrector` + `jxm/vec2text__openai_ada002__msmarco__msl128__hypothesizer` on HuggingFace |
| **GPU** | Single GPU with ~8-16 GB VRAM for inference (beam search). Our RTX 4080 16GB should work |
| **Inference speed** | ~5 seconds per passage on H100; expect ~10-15s on RTX 4080 |
| **API for ada-002** | Need OpenAI API key to re-embed during correction steps |

**The demo we should run:**
```python
import vec2text

# Load corrector for GTR-base (free, no API needed)
corrector = vec2text.load_pretrained_corrector("gtr-base")

# Embed some test texts
embeddings = embed_texts(["The patient was diagnosed with..."])

# Attack unprotected embeddings
recovered_unprotected = vec2text.invert_embeddings(embeddings, corrector, num_steps=20)
# Result: recovers ~92% of text

# Attack rotated embeddings (our PrivateEncoder)
rotated = encoder.rotate(embeddings)
recovered_rotated = vec2text.invert_embeddings(rotated, corrector, num_steps=20)
# Expected result: garbage text, BLEU ~0
```

**Why this is the killer demo:** Visual proof that Vec2Text recovers text from unprotected embeddings but returns garbage from rotated embeddings. Screenshot-able, shareable, undeniable.

### Simpler inversion attacks (no Vec2Text needed)

**MLP-based inversion:** Train a simple MLP on (embedding, text) pairs. Test on rotated embeddings. Shows that even simple statistical attacks fail.

**Classifier transfer attack (we already have this):**
- Train topic classifier on original embeddings
- Test on rotated embeddings
- Accuracy drops to random chance
- Already implemented in Phase 2 of PLAN-private-embeddings

### Privacy metrics to report

| Metric | Unprotected | After Rotation | Why it matters |
|--------|-------------|----------------|---------------|
| **Vec2Text BLEU** | 83-97 | ~0-5 | Gold standard attack fails |
| **Vec2Text Exact Match** | 60-92% | ~0% | Complete inversion failure |
| **Per-dimension Pearson r** | 1.0 | < 0.02 | Statistical decorrelation |
| **Topic classifier accuracy** | 85-95% | ~random (10-20%) | Attribute inference fails |
| **Token F1 (reconstruction)** | 0.70-0.95 | ~0 | Token-level recovery fails |
| **Mutual information** | High | ~0 | Information-theoretic guarantee (Eguard metric) |

### What we do NOT need

- Differential privacy epsilon/delta (we're not DP, don't pretend to be)
- Formal security proofs (we're obfuscation, not encryption)
- MPC/HE comparisons (different threat model entirely)

---

## 5. Getting Listed on ann-benchmarks.com (or equivalent)

### How to submit to ann-benchmarks

**Repository:** github.com/erikbern/ann-benchmarks

**Required files for a PR:**

```
ann_benchmarks/algorithms/turboquant_vectors/
    module.py       # Python wrapper, subclass BaseANN
    Dockerfile      # Build environment
    config.yml      # Hyperparameter grid to test
```

**BaseANN interface to implement:**

```python
class TurboQuantVectors(BaseANN):
    def fit(self, X):
        """Build index from training data (numpy array)"""
        # Apply rotation + quantization
        pass

    def query(self, q, n):
        """Return n nearest neighbor indices for query q"""
        pass

    def batch_query(self, X, n):
        """Optional: batch mode for GPU acceleration"""
        pass

    def get_memory_usage(self):
        """Return bytes used by index"""
        pass
```

**config.yml example:**

```yaml
float:
  turboquant_vectors:
    docker-tag: ann-benchmarks-turboquant-vectors
    module: ann_benchmarks.algorithms.turboquant_vectors
    constructor: TurboQuantVectors
    base-args: ["@metric"]
    run-groups:
      tq4:
        arg-groups:
          - {"bits": 4}
        query-args: [[10, 50, 100, 200]]
      tq3:
        arg-groups:
          - {"bits": 3}
        query-args: [[10, 50, 100, 200]]
      tq2:
        arg-groups:
          - {"bits": 2}
        query-args: [[10, 50, 100, 200]]
```

**Process:**
1. Fork github.com/erikbern/ann-benchmarks
2. Add algorithm directory with module.py, Dockerfile, config.yml
3. Add entry to `.github/workflows/benchmarks.yml`
4. Submit PR
5. Results auto-generated by CI

**Would being listed drive adoption?**

Absolutely. ann-benchmarks.com is THE reference. Every vector DB engineer checks it. Being listed with competitive results would:
- Validate the approach on standard datasets
- Generate backlinks and citations
- Put us alongside FAISS, ScaNN, HNSW, Annoy
- Drive GitHub stars and pip installs

### But: ann-benchmarks tests FULL ANN indexes, not codecs

**Problem:** ann-benchmarks measures recall vs. QPS for complete search systems (index + search). We're a compression library, not a full index. We'd need to pair our codec with an index structure (e.g., brute-force scan on compressed vectors, or IVF + our PQ replacement).

**Options:**
1. Submit as brute-force on compressed vectors (shows codec quality, QPS will be lower than indexed approaches)
2. Submit as IVF + TurboQuant PQ (more competitive, but more engineering)
3. Skip ann-benchmarks, publish on VIBE instead (newer, more codec-friendly)

### Alternative: VIBE benchmark

**VIBE** (Vector Index Benchmark for Embeddings, arXiv 2505.17810) is newer and more relevant:
- Uses real embedding datasets (ArXiv, ImageNet embeddings)
- Tests both index methods and quantization approaches
- GitHub: github.com/vector-index-bench/vibe
- HuggingFace: vector-index-bench/vibe
- Website: vector-index-bench.github.io
- Evaluated 21 implementations on 12 datasets

**Submitting to VIBE may be easier and more relevant** since they explicitly test quantization methods.

---

## 6. Concrete Benchmark Protocol

### Phase 1: Quick validation on real data (1-2 days)

```bash
# Download Qdrant OpenAI dataset
pip install datasets numpy faiss-cpu turboquant-vectors

python -c "
from datasets import load_dataset
import numpy as np

# Load 100K OpenAI text-embedding-3-small vectors
ds = load_dataset('Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K')
embeddings = np.array(ds['train']['text-embedding-3-small-1536-embedding'], dtype='float32')
np.save('openai_small_100k.npy', embeddings)
print(f'Shape: {embeddings.shape}')  # Expected: (100000, 1536)
"
```

**Run head-to-head:**
1. FAISS PQ at 4-bit: `faiss.IndexPQ(1536, M, 4)` (M = 1536/subvec_dim)
2. Our TurboQuant at 4-bit: `compress(embeddings, bits=4)`
3. Measure recall@10 on 1000 random queries, brute-force on compressed
4. Report: recall@10, compression ratio, encode time, search time

### Phase 2: Standard benchmarks (2-3 days)

```bash
# Download ann-benchmarks datasets
pip install h5py
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
wget http://ann-benchmarks.com/glove-100-angular.hdf5
```

Run same head-to-head on SIFT1M and GloVe-100. These are the numbers people compare.

### Phase 3: Privacy demo (1-2 days)

```bash
pip install vec2text
# Load GTR-base corrector (no API key needed)
# Run Vec2Text on unprotected vs rotated embeddings
# Report BLEU, exact match, token F1
```

### Phase 4: Scale test (1 day)

Load Qdrant 1M dataset. Test at scale. Report memory, build time, query latency.

---

## 7. Summary: What Gets Us Taken Seriously

| Deliverable | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| Recall@10 on Qdrant OpenAI 100K dataset | 1 day | High -- first real-data number | **P0** |
| Head-to-head vs FAISS PQ on SIFT1M | 1 day | High -- apples-to-apples with published | **P0** |
| Vec2Text privacy demo (BLEU before/after) | 1-2 days | Very High -- the screenshot everyone shares | **P0** |
| Recall@10 on Cohere Wikipedia 768-dim | 1 day | Medium -- second model family | P1 |
| VIBE benchmark submission | 2-3 days | High -- listed on public benchmark | P1 |
| ann-benchmarks PR | 3-5 days | Very High -- listed on THE benchmark | P2 (needs index wrapper) |
| 1M scale test (Qdrant large dataset) | 1 day | Medium -- proves scalability | P1 |
| RaBitQ comparison | 2 days | High -- current SOTA comparison | P2 |

**Bottom line:** Three things get us from "interesting beta" to "credible library":
1. Beat FAISS PQ on real OpenAI embeddings (not synthetic)
2. Vec2Text BLEU dropping from 83 to ~0 after rotation
3. Listed on VIBE or ann-benchmarks with competitive numbers

---

## Sources

### ANN Benchmarks
- [ANN-Benchmarks (official)](http://ann-benchmarks.com/)
- [ann-benchmarks GitHub](https://github.com/erikbern/ann-benchmarks)
- [Big ANN Benchmarks (NeurIPS)](https://big-ann-benchmarks.com/neurips21.html)
- [FAISS Vector Codec Benchmarks Wiki](https://github.com/facebookresearch/faiss/wiki/Vector-codec-benchmarks)
- [FAISS Indexing 1M Vectors Wiki](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)
- [VIBE: Vector Index Benchmark for Embeddings (arXiv 2505.17810)](https://arxiv.org/abs/2505.17810)
- [VIBE GitHub](https://github.com/vector-index-bench/vibe)
- [VIBE Website](https://vector-index-bench.github.io/)

### Real Embedding Datasets
- [Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K)
- [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M)
- [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M)
- [Cohere/wikipedia-22-12-en-embeddings (35M vectors)](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings)
- [Cohere/wikipedia-2023-11-embed-multilingual-v3 (250M vectors)](https://huggingface.co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3)
- [Supabase/wikipedia-en-embeddings (224K, ada-002)](https://huggingface.co/datasets/Supabase/wikipedia-en-embeddings)
- [Qdrant Practice Datasets](https://qdrant.tech/documentation/datasets/)

### Quantization Research
- [Product Quantization Deep Dive (Pinecone)](https://www.pinecone.io/learn/series/faiss/product-quantization/)
- [Binary and Scalar Embedding Quantization (HuggingFace blog)](https://huggingface.co/blog/embedding-quantization)
- [RaBitQ (SIGMOD 2024)](https://github.com/gaoj0017/RaBitQ)
- [Extended-RaBitQ (SIGMOD 2025)](https://github.com/VectorDB-NTU/Extended-RaBitQ)
- [Weaviate Binary Quantization](https://weaviate.io/blog/binary-quantization)
- [Weaviate PQ Rescoring](https://weaviate.io/blog/pq-rescoring)
- [The FAISS Library (arXiv 2401.08281)](https://arxiv.org/abs/2401.08281)

### Privacy / Embedding Inversion
- [Vec2Text GitHub](https://github.com/vec2text/vec2text)
- [Vec2Text Paper: Text Embeddings Reveal (Almost) As Much As Text](https://aclanthology.org/2023.emnlp-main.765.pdf)
- [Vec2Text ada-002 Corrector Model](https://huggingface.co/jxm/vec2text__openai_ada002__msmarco__msl128__corrector)
- [Vec2Text GTR-base Inversion Model](https://huggingface.co/ielabgroup/vec2text_gtr-base-st_inversion)
- [Vec2Text GTR-base Corrector Model](https://huggingface.co/ielabgroup/vec2text_gtr-base-st_corrector)
- [Eguard: Mutual Information Defense (arXiv 2411.05034)](https://arxiv.org/abs/2411.05034)
- [Transferable Embedding Inversion Attack (arXiv 2406.10280)](https://arxiv.org/abs/2406.10280)
- [Vec2Text Reproducibility Study (arXiv 2507.07700)](https://arxiv.org/html/2507.07700)
- [Concept-Aware Privacy Mechanisms (arXiv 2602.07090)](https://arxiv.org/html/2602.07090v1)

### Benchmark Submission
- [ann-benchmarks PR #596 (example submission)](https://github.com/erikbern/ann-benchmarks/pull/596)
- [VIBE HuggingFace datasets](https://huggingface.co/datasets/vector-index-bench/vibe)
- [Milvus/Zilliz ANN Benchmarks Explainer](https://zilliz.com/glossary/ann-benchmarks)

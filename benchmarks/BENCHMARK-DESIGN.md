# Benchmark Design: turboquant-vectors

> Prove four claims with real numbers: lossless recall from rotation, negligible latency, privacy against inversion attacks, and better quality-at-compression than competitors.

**Target:** Run all benchmarks, publish results in README and a `benchmarks/results/` directory with reproducible scripts.

---

## 1. Datasets

### 1.1 Standard ANN-Benchmarks Datasets (HDF5, pre-split train/test/ground-truth)

These are the industry standard. Every ANN paper uses them. Pre-built HDF5 files with ground truth top-100 neighbors included.

| Dataset | Dims | Train Vectors | Test Vectors | Metric | Size | Download |
|---------|------|--------------|-------------|--------|------|----------|
| GloVe-100 | 100 | 1,183,514 | 10,000 | Angular | 485 MB | `https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/glove-100-angular.hdf5` |
| GloVe-200 | 200 | 1,183,514 | 10,000 | Angular | 918 MB | `http://ann-benchmarks.com/glove-200-angular.hdf5` |
| SIFT-128 | 128 | 1,000,000 | 10,000 | Euclidean | 501 MB | `http://ann-benchmarks.com/sift-128-euclidean.hdf5` |
| Fashion-MNIST | 784 | 60,000 | 10,000 | Euclidean | 228 MB | `http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5` |

**Why these:** GloVe-200 matches the TurboQuant paper directly. SIFT is the classic million-scale benchmark. Fashion-MNIST tests higher dimensionality on real data. GloVe-100 is the most commonly cited ANN-benchmarks dataset.

### 1.2 Real RAG Embeddings (OpenAI, 1536-dim)

| Dataset | Dims | Vectors | Source | Download |
|---------|------|---------|--------|----------|
| DBpedia-OpenAI-100K | 1,536 | 100,000 | text-embedding-3-small on DBpedia entities | `huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K` |
| DBpedia-OpenAI-1M | 3,072 | 1,000,000 | text-embedding-3-large on DBpedia entities | `huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M` |

**Why these:** Real production embeddings from OpenAI's current model. 1536-dim is the most common RAG dimension. The 1M set tests scale. Precomputed (no API cost).

### 1.3 Domain-Specific: Medical (HIPAA use case)

| Dataset | Dims | Documents | Source | Access |
|---------|------|-----------|--------|--------|
| NFCorpus (BEIR) | embed yourself | 3,633 docs | Biomedical nutrition/health IR | `huggingface.co/datasets/BeIR/nfcorpus` (open, CC-BY-SA-4.0) |
| SciFact (BEIR) | embed yourself | 5,183 docs | Scientific fact verification | `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip` |

**Why these:** NFCorpus is biomedical IR, directly relevant to HIPAA. SciFact is scientific claims verification. Both are open access (no MIMIC credentialing needed). We embed them ourselves with a model (e.g., `all-MiniLM-L6-v2` at 384-dim or `text-embedding-3-small` at 1536-dim) to get the vectors, then compress.

**Note on MIMIC:** MIMIC-IV-Note (331K discharge summaries) would be the gold standard HIPAA demo, but requires PhysioNet credentialing and a data use agreement. Mention it as a "validated compatible" dataset in docs but don't include in automated benchmarks.

### 1.4 Domain-Specific: Financial

| Dataset | Dims | Documents | Source | Access |
|---------|------|-----------|--------|--------|
| FiQA-2018 (BEIR) | embed yourself | 57,638 docs | Financial opinion QA from StackExchange/Reddit | `huggingface.co/datasets/BeIR/fiqa` (open) |

**Why this:** FiQA is the standard financial IR dataset in BEIR. Real financial questions and answers. Demonstrates the use case of "compress your financial RAG index without leaking client data."

### 1.5 Synthetic Controlled Data (already have this)

Keep the existing `generate_clustered_data()` for controlled experiments where we need exact reproducibility and parameter sweeps. This is what `benchmarks/compare_faiss.py` already uses.

---

## 2. Recall Benchmarks

### 2.1 Exact Recall (Rotation is Lossless) -- THE critical claim

**Claim:** Random orthogonal rotation preserves all pairwise distances and inner products exactly. Therefore Recall@K on rotated vectors vs original vectors is exactly 1.000 for brute-force search.

**Test: `benchmarks/bench_rotation_lossless.py`**

```python
"""
For each dataset:
  1. Load vectors V (float32)
  2. Generate rotation matrix Q (orthogonal, from seed)
  3. Compute V_rot = V @ Q.T
  4. For 1000 random queries:
     a. Exact brute-force top-K on V
     b. Exact brute-force top-K on V_rot (query also rotated: q_rot = q @ Q.T)
  5. Assert Recall@K == 1.000 for K in {1, 5, 10, 50, 100}
  6. Assert max |cosine(v_i, v_j) - cosine(v_rot_i, v_rot_j)| < 1e-6
  7. Assert max |norm(v_i) - norm(v_rot_i)| < 1e-6
"""
```

**Expected results:**

| Dataset | K=1 | K=10 | K=100 | Max Cosine Error |
|---------|-----|------|-------|-----------------|
| GloVe-100 (1.2M) | 1.000 | 1.000 | 1.000 | < 1e-6 |
| SIFT-128 (1M) | 1.000 | 1.000 | 1.000 | < 1e-6 |
| DBpedia-1536 (100K) | 1.000 | 1.000 | 1.000 | < 1e-6 |

**Why this works:** Orthogonal matrices preserve inner products by definition: `<Qx, Qy> = x^T Q^T Q y = x^T y = <x, y>`. The benchmark just verifies our implementation is numerically correct.

### 2.2 Recall After Compression (rotate + quantize)

**Test: `benchmarks/bench_recall_compressed.py`**

```python
"""
For each dataset x each bit width {1, 2, 3, 4, 8}:
  1. Ground truth: brute-force top-K on original float32 vectors
  2. Compress with TurboQuant at given bits
  3. Decompress and brute-force top-K on decompressed vectors
  4. Compute Recall@K for K in {1, 5, 10, 50, 100}
"""
```

**Expected results (approximate, GloVe-200, 1.2M vectors):**

| Bits | Recall@1 | Recall@10 | Recall@100 | Bytes/Vec |
|------|----------|-----------|------------|-----------|
| 8 | ~0.98 | ~0.97 | ~0.96 | 200 |
| 4 | ~0.85 | ~0.82 | ~0.80 | 100 |
| 3 | ~0.70 | ~0.68 | ~0.65 | 75 |
| 2 | ~0.50 | ~0.48 | ~0.45 | 50 |
| 1 | ~0.25 | ~0.22 | ~0.20 | 25 |

### 2.3 FAISS Index on Rotated Vectors (ANN integration)

**Test: `benchmarks/bench_faiss_index.py`**

```python
"""
Prove: building a FAISS IVF or HNSW index on rotated vectors gives
identical recall to building it on original vectors.

  1. Build FAISS IndexIVFFlat on original vectors, search, record recall
  2. Build FAISS IndexIVFFlat on rotated vectors (same nprobe), search, record recall
  3. Assert recalls match within noise (same random seed for index training)
  4. Repeat with IndexHNSWFlat
"""
```

**Expected:** Identical recall (within index build randomness). This proves rotation is transparent to downstream ANN indexes.

### 2.4 ANN-Benchmarks Integration

**Not a priority for v0.1.** The ann-benchmarks framework expects a Docker container implementing a specific Python interface. We can add this later. For now, we benchmark against FAISS PQ directly (already done) and add the standard HDF5 datasets above.

---

## 3. Latency Benchmarks

### 3.1 Rotation Latency vs Embedding Latency

**Test: `benchmarks/bench_latency.py`**

```python
"""
Measure wall-clock time for:
  1. Rotation matrix generation (QR decomposition of d x d Gaussian)
  2. Rotating N vectors: V @ Q.T
  3. Quantization: find nearest centroid per coordinate
  4. Full compress() call (rotation + quantization)
  5. Decompress() call
  6. search() call (decompress + cosine + top-k)

Compare rotation time vs typical embedding API latency (~200ms for OpenAI).
Run 10 trials each, report mean +/- std.

Dimensions: 128, 384, 768, 1536, 3072
N vectors: 1K, 10K, 100K
"""
```

**Expected results (single CPU core, approximate):**

| Operation | d=384 | d=1536 | d=3072 |
|-----------|-------|--------|--------|
| QR decomposition (one-time) | ~5ms | ~200ms | ~1.5s |
| Rotate 10K vectors | ~15ms | ~250ms | ~1s |
| Quantize 10K vectors (4-bit) | ~50ms | ~200ms | ~400ms |
| Full compress 10K vectors | ~70ms | ~450ms | ~1.4s |
| Decompress 10K vectors | ~5ms | ~20ms | ~40ms |
| Search 100 queries over 10K | ~10ms | ~40ms | ~80ms |

**Key claim:** Rotation adds < 1% overhead vs the embedding API call itself. If OpenAI takes 200ms to embed one text, rotating a 1536-dim vector takes < 1ms.

### 3.2 Search Latency: Rotated vs Original

**Test:** Same as above but compare cosine search on original float32 vs decompressed float32. Should be identical since the vectors are same dtype and dimension after decompression.

### 3.3 Key Generation and Load Time

```python
"""
  1. Time to generate key (= rotation matrix): QR on d x d
  2. Time to save key to disk (.npy)
  3. Time to load key from disk
  4. Key size on disk: d*d*4 bytes (float32)

  d=1536: key = 1536*1536*4 = 9.4 MB, generation ~200ms
"""
```

---

## 4. Privacy Benchmarks (The Killer Demo)

### 4.1 Vec2Text Inversion Attack

**Test: `benchmarks/bench_privacy_vec2text.py`**

This is the flagship privacy benchmark. We use the published Vec2Text models to attempt text recovery from rotated embeddings.

```python
"""
Prerequisites:
  pip install vec2text

Steps:
  1. Take 100 short texts (32 tokens each) from a held-out dataset
  2. Embed with text-embedding-ada-002 (or GTR-base for the open model)
  3. Run Vec2Text inversion on ORIGINAL embeddings -> measure BLEU/ROUGE
  4. Rotate embeddings with random orthogonal Q
  5. Run Vec2Text inversion on ROTATED embeddings -> measure BLEU/ROUGE
  6. Show BLEU drops from ~0.95 to ~0.01 (random baseline)

Vec2Text corrector models available:
  - "gtr-base" (open, no API key needed)
  - "text-embedding-ada-002" (needs OpenAI key)
"""
```

**Expected results:**

| Condition | BLEU | ROUGE-L | Exact Match % |
|-----------|------|---------|---------------|
| Original embeddings (no rotation) | ~0.90 | ~0.92 | ~85% |
| Rotated embeddings (unknown key) | ~0.01 | ~0.03 | 0% |
| Random vectors (baseline) | ~0.01 | ~0.02 | 0% |

**Why this works:** Vec2Text learns a mapping from embedding space to text. Rotation shuffles the coordinate system entirely. Without knowing Q, the corrector model's learned mapping is useless. The rotated embeddings look indistinguishable from random to Vec2Text.

### 4.2 Classifier Transfer Attack

**Test: `benchmarks/bench_privacy_classifier.py`**

A simpler, faster privacy test that doesn't need Vec2Text.

```python
"""
Train a sentiment/topic classifier on ORIGINAL embeddings.
Test it on ROTATED embeddings. Should fail completely.

  1. Use IMDb sentiment dataset (or 20 Newsgroups for topic)
  2. Embed all texts with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
  3. Train a 2-layer MLP: embedding -> sentiment (accuracy ~88% on originals)
  4. Test MLP on ROTATED embeddings (same texts, same model, different key)
  5. Accuracy should drop to ~50% (random chance for binary classification)

  Also test: logistic regression, SVM, random forest (all should fail on rotated)
"""
```

**Expected results:**

| Classifier | Original Embeddings | Rotated (unknown key) | Rotated (known key) |
|-----------|-------------------|---------------------|-------------------|
| MLP | ~88% | ~50% (chance) | ~88% |
| Logistic Regression | ~85% | ~50% (chance) | ~85% |
| SVM | ~86% | ~50% (chance) | ~86% |

**Why this works:** All learned classifiers depend on the coordinate system. Rotation changes every coordinate. Without the key, no learned function of coordinates transfers.

### 4.3 Statistical Correlation Attack

**Test: `benchmarks/bench_privacy_correlation.py`**

```python
"""
Show that rotated embeddings have zero correlation with originals.

  1. Generate or load N vectors (10K+)
  2. Rotate with random Q
  3. Compute:
     a. Pearson correlation per dimension: corr(v[:,i], v_rot[:,i]) for each i
     b. Cross-correlation matrix: corrcoef between all original dims and all rotated dims
     c. Mutual information between original and rotated (per dimension)
  4. Show all correlations are ~0 (within sampling noise: |r| < 0.02 for N=10K)
  5. Plot: heatmap of cross-correlation matrix (should be uniform noise)
"""
```

**Expected results:**
- Mean |Pearson r| per dimension: < 0.02
- Max |Pearson r| across all dim pairs: < 0.05
- Mutual information: near zero (within estimation error)

### 4.4 Known-Plaintext Attack Resistance

**Test: `benchmarks/bench_privacy_known_pairs.py`**

```python
"""
An attacker has n known (original, rotated) vector pairs.
Can they recover the rotation matrix Q?

  d = dimension of vectors
  Q is d x d orthogonal matrix (d^2 unknowns, but d*(d-1)/2 degrees of freedom)

  If attacker has n >= d linearly independent pairs, they can solve for Q exactly.
  If n < d, they get only a partial solution.

Test:
  1. For d in {128, 384, 768, 1536}:
  2. For n in {1, 10, d//4, d//2, d-1, d, d+10}:
     a. Give attacker n known (v_i, Q @ v_i) pairs
     b. Attacker solves least-squares: Q_hat = argmin ||V_rot - V @ Q_hat||
     c. Measure recovery error: ||Q - Q_hat||_F / ||Q||_F
     d. Measure recall degradation: search with Q_hat vs true Q
  3. Show: full recovery requires n >= d pairs (1536 known pairs for 1536-dim!)
"""
```

**Expected results (d=1536):**

| Known Pairs (n) | Q Recovery Error | Recall@10 with Q_hat |
|-----------------|-----------------|---------------------|
| 1 | ~1.00 (no recovery) | ~0.001 |
| 10 | ~0.99 | ~0.001 |
| 100 | ~0.97 | ~0.01 |
| 384 (d/4) | ~0.87 | ~0.05 |
| 768 (d/2) | ~0.71 | ~0.15 |
| 1535 (d-1) | ~0.03 | ~0.90 |
| 1536 (d) | ~0.00 | 1.000 |

**Key takeaway:** An attacker needs ALL d dimensions worth of known pairs to recover the key. For 1536-dim OpenAI embeddings, that's 1536 known plaintexts. This is a strong security guarantee for most threat models.

### 4.5 Privacy vs IronCore Cloaked AI Comparison

**Test: `benchmarks/bench_vs_cloaked_ai.py`**

We can't run Cloaked AI (proprietary SDK), but we can compare published numbers.

IronCore's published benchmark results (from their docs):
- Approximation factor 1.5: average -4.9% NDCG loss across BEIR datasets
- Approximation factor 2.0: larger losses
- NFCorpus with gte-base: -19% precision (outlier)

We run our rotation on the same BEIR datasets with the same embedding models and report:
- NDCG@10 on original vs rotated (should be identical -- 0% loss)
- NDCG@10 on original vs rotated+compressed at 4-bit (some loss from quantization)

**Expected comparison:**

| Method | Privacy Level | NDCG@10 Loss | Compression | Training Needed |
|--------|-------------|-------------|-------------|----------------|
| TurboQuant rotation only | Full (coordinate shuffle) | 0% | 1:1 (no compression) | None |
| TurboQuant rotate+4bit | Full + 8x compression | ~5-15% | 8x | None |
| Cloaked AI (factor 1.5) | Approximate distance-preserving | ~5% | 1:1 | None |
| FAISS PQ (matched bytes) | None | ~15-25% | 8x | k-means training |

---

## 5. Compression Quality Benchmarks

### 5.1 Recall@K at Different Bit Widths

**Test: `benchmarks/bench_compression_quality.py`**

```python
"""
For each dataset x each bit width:
  1. Compress with TurboQuant
  2. Compress with FAISS PQ at matched byte budget
  3. Compute Recall@K and NDCG@K against float32 ground truth
  4. Also compute: MSE, cosine similarity of decompressed vs original

Bit widths: 1, 2, 3, 4, 8
Datasets: GloVe-200, SIFT-128, DBpedia-1536, synthetic-50K
"""
```

### 5.2 Head-to-Head vs FAISS PQ (extended from existing benchmark)

Extend `compare_faiss.py` to run on real datasets (not just synthetic):

```python
"""
For GloVe-200 (1.2M vectors) at 4-bit:
  TurboQuant: 200*4/8 = 100 bytes/vec
  FAISS PQ:   m=100, nbits=8 -> 100 bytes/vec (matched)

For DBpedia-1536 (100K vectors) at 4-bit:
  TurboQuant: 1536*4/8 = 768 bytes/vec
  FAISS PQ:   m=768, nbits=8 -> 768 bytes/vec (matched)

Report: Recall@{1,5,10,50,100}, compression time, search time
"""
```

### 5.3 Quality Metrics Beyond Recall

```python
"""
Additional metrics for each compression method:
  1. Mean Squared Error (MSE) of decompressed vs original
  2. Mean cosine similarity between decompressed and original vectors
  3. Rank correlation (Spearman) of pairwise distances
  4. NDCG@K (weighted recall, penalizes wrong ordering)
"""
```

---

## 6. Scalability Benchmarks

### 6.1 Compression Throughput

**Test: `benchmarks/bench_scalability.py`**

```python
"""
Measure time and memory for:
  N in {1_000, 10_000, 100_000, 1_000_000, 10_000_000}
  d in {384, 768, 1536}
  bits = 4

Report:
  - Total compression time (seconds)
  - Throughput (vectors/second)
  - Peak memory (RSS via tracemalloc or psutil)
  - Memory per vector during compression

For N=10M, d=1536:
  - Original: 10M * 1536 * 4 = 61.4 GB (won't fit in memory)
  - Need streaming compression (batch by batch)
"""
```

**Expected throughput (single CPU, d=1536, 4-bit):**

| N | Time | Throughput | Peak Memory |
|---|------|-----------|-------------|
| 1K | ~0.3s | ~3K vec/s | ~20 MB |
| 10K | ~2.5s | ~4K vec/s | ~100 MB |
| 100K | ~25s | ~4K vec/s | ~950 MB |
| 1M | ~250s | ~4K vec/s | ~9.4 GB |

### 6.2 Streaming Compression

```python
"""
Test batch-by-batch compression to avoid loading all vectors at once.

  1. Generate 1M vectors on disk (memmap)
  2. Compress in batches of 10K
  3. Verify: search results are identical to compressing all at once
  4. Report: peak memory should stay at ~100MB regardless of total N
"""
```

### 6.3 Scaling of QR Decomposition

```python
"""
The one-time cost: generating the rotation matrix via QR decomposition.

d in {64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096}
Report: QR time, matrix size on disk

d=1536: QR takes ~200ms, matrix = 9.4 MB (amortized over millions of vectors)
d=4096: QR takes ~5s, matrix = 67 MB
"""
```

---

## 7. Code Structure

```
benchmarks/
  BENCHMARK-DESIGN.md          # This document
  compare_faiss.py             # Existing: TQ vs FAISS PQ (synthetic data)
  datasets/                    # Downloaded datasets (gitignored)
    glove-200-angular.hdf5
    sift-128-euclidean.hdf5
    dbpedia-openai-100k/
    ...
  download_datasets.py         # Script to download all benchmark datasets
  results/                     # Generated results (gitignored)
    rotation_lossless.json
    recall_compressed.json
    latency.json
    privacy_vec2text.json
    ...

  # Core benchmarks
  bench_rotation_lossless.py   # Prove rotation preserves exact recall
  bench_recall_compressed.py   # Recall@K at different bit widths
  bench_faiss_index.py         # FAISS IVF/HNSW on rotated vectors
  bench_latency.py             # Timing of all operations
  bench_compression_quality.py # MSE, cosine, NDCG, Spearman rank

  # Privacy benchmarks
  bench_privacy_vec2text.py    # Vec2Text inversion on rotated embeddings
  bench_privacy_classifier.py  # Classifier transfer attack
  bench_privacy_correlation.py # Statistical correlation analysis
  bench_privacy_known_pairs.py # Known-plaintext attack resistance

  # Scalability benchmarks
  bench_scalability.py         # N and d scaling, throughput, memory

  # Comparison benchmarks
  bench_vs_cloaked_ai.py       # Compare with IronCore published numbers

  # Utilities
  utils.py                     # Shared: data loading, recall calc, timing
  run_all.py                   # Run entire benchmark suite, generate report
  plot_results.py              # Generate charts from results JSON files
```

---

## 8. Dependencies

```toml
[project.optional-dependencies]
bench = [
    "faiss-cpu",
    "h5py",              # For ANN-benchmark HDF5 files
    "datasets",          # For HuggingFace datasets
    "sentence-transformers",  # For embedding BEIR text datasets
    "scikit-learn",      # For classifier attack benchmarks
    "matplotlib",        # For result plots
    "psutil",            # For memory measurement
    "tqdm",              # Progress bars
]
bench-privacy = [
    "vec2text",          # For embedding inversion attack
    "torch",             # Vec2Text dependency
]
```

---

## 9. Reproducibility

### 9.1 Hardware

All benchmarks should run on commodity hardware. Minimum spec:
- **CPU:** Any x86-64 with AVX2 (2015+)
- **RAM:** 16 GB minimum (32 GB for 1M-scale GloVe)
- **Disk:** 10 GB free for datasets
- **GPU:** Not required (NumPy CPU only). Vec2Text privacy benchmark benefits from GPU.

We report results on two configurations:
1. **Consumer laptop:** AMD Ryzen 7 / Intel i7, 32 GB RAM, no GPU
2. **Cloud instance:** AWS r6i.xlarge (4 vCPU, 32 GB RAM) for reproducibility

### 9.2 Fixed Seeds Everywhere

Every benchmark uses fixed random seeds. The `run_all.py` script produces deterministic results on the same hardware/numpy version.

```python
SEED = 42
QUERY_SEED = 99
ROTATION_SEED = 7  # Different from compression seed to simulate separate key
```

### 9.3 Versioning

Each results JSON includes:
```json
{
  "turboquant_vectors_version": "0.1.0",
  "numpy_version": "1.26.4",
  "python_version": "3.12.0",
  "platform": "Linux-6.1-x86_64",
  "cpu": "AMD Ryzen 7 5800X",
  "ram_gb": 32,
  "timestamp": "2026-03-25T12:00:00Z",
  "seed": 42
}
```

### 9.4 Publishing Results

Results go in three places:

1. **README.md** -- headline numbers only (recall table, one privacy number, one latency number)
2. **benchmarks/RESULTS.md** -- full tables for all datasets and configurations
3. **benchmarks/results/*.json** -- raw machine-readable results for anyone to verify

The `run_all.py` script generates all three automatically.

---

## 10. Priority Order

Phase 1 (ship with v0.2.0):
1. `bench_rotation_lossless.py` -- proves the core mathematical claim
2. `bench_privacy_classifier.py` -- quick, no external deps beyond sklearn
3. `bench_privacy_correlation.py` -- quick, pure numpy
4. `bench_latency.py` -- answers "is this fast enough?"
5. `bench_recall_compressed.py` on real datasets (GloVe-200, SIFT-128)

Phase 2 (ship with v0.3.0):
6. `bench_privacy_vec2text.py` -- the flagship demo (needs vec2text install)
7. `bench_privacy_known_pairs.py` -- quantifies key security
8. `bench_compression_quality.py` -- extended metrics
9. `bench_faiss_index.py` -- proves ANN compatibility

Phase 3 (v0.4.0+):
10. `bench_scalability.py` -- million/10M scale
11. `bench_vs_cloaked_ai.py` -- competitive comparison
12. `download_datasets.py` + BEIR domain datasets (medical, financial)
13. `run_all.py` + `plot_results.py` -- full automation

---

## 11. Key Claims and Evidence Map

| Claim | Benchmark | Expected Evidence |
|-------|-----------|-------------------|
| Rotation is lossless | `bench_rotation_lossless` | Recall@K = 1.000 for all K, all datasets |
| Rotation is fast | `bench_latency` | < 1ms per vector at d=1536 |
| Vec2Text fails on rotated | `bench_privacy_vec2text` | BLEU drops from ~0.90 to ~0.01 |
| Classifiers fail on rotated | `bench_privacy_classifier` | Accuracy drops to random chance |
| No statistical leakage | `bench_privacy_correlation` | |Pearson r| < 0.02 across all dims |
| Key recovery needs d pairs | `bench_privacy_known_pairs` | Error > 0.5 until n >= d/2 |
| Better than FAISS PQ | `bench_recall_compressed` | +5-8pp at 4-bit on real data |
| Comparable to Cloaked AI privacy, better compression | `bench_vs_cloaked_ai` | 0% NDCG loss (rotation only) vs ~5% |
| Scales to millions | `bench_scalability` | Linear time, bounded memory |

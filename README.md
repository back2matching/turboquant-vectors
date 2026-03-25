# turboquant-vectors

Compress embeddings 8x instantly. No training needed.

First open-source implementation of Google's TurboQuant ([ICLR 2026](https://arxiv.org/abs/2504.19874)) for vector search and embedding compression.

```python
from turboquant_vectors import compress, search

compressed = compress(embeddings, bits=4)  # 307 MB -> 38 MB
indices, scores = search(compressed, query, top_k=10)
```

## Why

RAG on consumer hardware is memory-constrained. 1M documents at 1536-dim = 6.1 GB just for the embedding index. Add a 30B model and you're out of VRAM.

FAISS Product Quantization helps but requires k-means training per dataset. TurboQuant is instant (data-oblivious), compresses 2-2.5x faster, and gets up to +8pp better recall at the same storage budget on our synthetic matched-budget benchmark.

## Benchmarks

All benchmarks are fully reproducible with fixed seeds. Run them yourself:

```bash
python benchmarks/compare_faiss.py --n 50000 --dim 1536 --top-k 10
```

### Recall@10 at matched storage (50K vectors, 1536-dim)

Same bytes per vector for both methods. Fair comparison.

| Budget | TurboQuant | FAISS PQ | Delta | Compress Time |
|--------|-----------|----------|-------|---------------|
| 1-bit (192 B/vec) | 26.0% | **29.0%** | -3.0pp | 3.4s vs 4.9s |
| 2-bit (384 B/vec) | **52.8%** | 45.7% | **+7.1pp** | 3.8s vs 8.5s |
| 4-bit (768 B/vec) | **83.8%** | 75.8% | **+8.0pp** | 6.5s vs 16.0s |

On synthetic clustered data, TurboQuant wins at 2-bit and 4-bit. FAISS PQ is slightly better at ultra-compact 1-bit. TurboQuant compresses 2-2.5x faster because it skips k-means training.

### Compression ratios (50K vectors, 1536-dim)

| Bits | Original | Compressed | Ratio |
|------|----------|-----------|-------|
| 2 | 307 MB | 19 MB | **16x** |
| 4 | 307 MB | 38 MB | **8x** |
| 8 | 307 MB | 77 MB | **4x** |

## Install

```bash
pip install turboquant-vectors

# With FAISS comparison benchmark support:
pip install turboquant-vectors[faiss]
```

## Usage

```python
import numpy as np
from turboquant_vectors import compress, decompress, search

# Your embeddings (any source: OpenAI, sentence-transformers, etc.)
embeddings = np.load("my_embeddings.npy")  # shape: (n, dim)

# Compress (instant, no training)
compressed = compress(embeddings, bits=4)
print(f"Compressed {compressed.original_bytes/1e6:.0f} MB -> {compressed.packed_memory_bytes/1e6:.0f} MB")

# Search on compressed vectors
query = np.random.randn(embeddings.shape[1]).astype(np.float32)
indices, scores = search(compressed, query, top_k=10)

# Save/load compressed vectors
compressed.save("my_index.tqv.npz")
from turboquant_vectors.core import CompressedVectors
loaded = CompressedVectors.load("my_index.tqv.npz")

# Or decompress back to float32
restored = decompress(compressed)
```

### CLI

```bash
# Compress a numpy embedding file
tq-vectors compress embeddings.npy --bits 4

# Search compressed vectors
tq-vectors search embeddings.tqv.npz query.npy --top-k 10

# Show compression stats
tq-vectors info embeddings.tqv.npz
```

## How It Works

TurboQuant applies a random orthogonal rotation to vectors before scalar quantization. The rotation makes coordinates approximately independent, enabling near-optimal per-coordinate quantization without any training data.

1. **Rotate** vectors by a random orthogonal matrix (generated once from a seed)
2. **Quantize** each coordinate using optimal centroids for the resulting distribution
3. **Search** by decompressing and computing cosine similarity

No k-means. No calibration data. Works on any embedding set instantly.

## Paper

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Zandieh, Daliri, Hadian, Mirrokni (Google Research)
ICLR 2026 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

This is an independent implementation, not affiliated with Google Research.

## License

Apache 2.0

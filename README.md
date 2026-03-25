# turboquant-vectors

Compress embeddings 6x instantly. No training needed.

First open-source implementation of Google's TurboQuant ([ICLR 2026](https://arxiv.org/abs/2504.19874)) for vector search and embedding compression.

```python
from turboquant_vectors import compress, search

compressed = compress(embeddings, bits=4)  # 586 MB -> 74 MB
indices, scores = search(compressed, query, top_k=10)
```

## Why

RAG on consumer hardware is memory-constrained. 1M documents at 1536-dim = 6.1 GB just for the embedding index. Add a 30B model and you're out of VRAM.

FAISS Product Quantization helps but requires slow k-means training per dataset. TurboQuant is instant (data-oblivious) and achieves **5.4x better recall**.

## Benchmarks

### Compression (100K vectors, 1536-dim, OpenAI embedding size)

| Bits | Original | Compressed | Ratio | Time |
|------|----------|-----------|-------|------|
| 2 | 586 MB | 37 MB | **15.8x** | 7.4s |
| 3 | 586 MB | 55 MB | **10.6x** | 8.9s |
| 4 | 586 MB | 74 MB | **8.0x** | 12.7s |

### Recall: TurboQuant vs FAISS PQ (10K vectors, 768-dim)

| Method | Recall@10 | Index Time |
|--------|----------|------------|
| **TurboQuant 4-bit** | **71.9%** | **0.7s (no training)** |
| FAISS PQ (m=48) | 13.3% | 0.6s |

5.4x better recall. Zero training. Data-oblivious.

## Install

```bash
pip install turboquant-vectors  # coming soon
# For now:
git clone https://github.com/back2matching/turboquant-vectors
cd turboquant-vectors && pip install -e .
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
query = np.random.randn(dim).astype(np.float32)
indices, scores = search(compressed, query, top_k=10)

# Or decompress back to float32
restored = decompress(compressed)
```

## How It Works

TurboQuant applies a random orthogonal rotation to vectors before scalar quantization. The rotation makes coordinates approximately independent (via Johnson-Lindenstrauss), enabling near-optimal per-coordinate quantization without any training data.

1. **Rotate** vectors by a random orthogonal matrix (generated once from a seed)
2. **Quantize** each coordinate using optimal centroids for the resulting distribution
3. **Search** by decompressing and computing cosine similarity (v1)

No k-means. No calibration data. Works on any embedding set instantly.

## Paper

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Zandieh, Daliri, Hadian, Mirrokni (Google Research)
ICLR 2026 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

This is an independent implementation, not affiliated with Google Research.

## License

Apache 2.0

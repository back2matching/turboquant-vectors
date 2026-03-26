# turboquant-vectors

Compress and protect embeddings with TurboQuant.

Two tools in one package:
- **PrivateEncoder** -- rotate embeddings with a secret key. Search works identically. Inversion attacks fail.
- **compress/search** -- 8x compression, no training needed, instant.

```python
from turboquant_vectors import PrivateEncoder

encoder = PrivateEncoder.generate(dim=1536)
rotated = encoder.rotate(embeddings)       # search works identically
encoder.save_key("secret.tqkey")           # treat like an SSH key
```

## Embedding Privacy

Vec2Text recovers 92% of original text from unprotected embeddings (32-token inputs, GTR-base encoder). ALGEN needs only 1,000 leaked pairs. OWASP lists this as LLM08 in their 2025 Top 10.

PrivateEncoder applies a secret orthogonal rotation before you send embeddings to a third-party vector DB. The math:

```
<Qx, Qy> = x^T Q^T Q y = x^T y = <x, y>
```

Cosine similarity, L2 distance, inner product -- all preserved exactly (up to float32 precision, ~1e-6 error).

### Quick start

```python
from turboquant_vectors import PrivateEncoder
import numpy as np

# Generate a secret key (uses OS entropy)
encoder = PrivateEncoder.generate(dim=1536)
encoder.save_key("secret.tqkey")

# Rotate before uploading to Pinecone/Weaviate/Qdrant
rotated = encoder.rotate(embeddings)
# pinecone_index.upsert(vectors=rotated.tolist(), ids=ids)

# Rotate query too (same key)
rotated_query = encoder.rotate(query)
# results = pinecone_index.query(vector=rotated_query.tolist(), top_k=10)

# Later, load the same key
encoder = PrivateEncoder.load_key("secret.tqkey")
```

### What it protects against

- **Vec2Text** (92% text recovery from embeddings) -- fails completely on rotated vectors
- **ALGEN** (few-shot inversion with 1K pairs) -- fails without the rotation key
- **ZSinvert / Zero2Text** (zero-shot inversion) -- fails on rotated embedding space
- **Attribute classifiers** (age, sex, medical conditions from embeddings) -- drop to random chance

Our demo proves it on real sentence-transformer embeddings across 5 sensitive categories (medical, financial, legal, personal, neutral): a classifier achieves 88.9% accuracy on originals but drops to 11.1% on rotated vectors (below 20% random chance). See `demos/inversion_demo.py`.

We also tested the Wasserstein-Procrustes unsupervised alignment attack (the strongest known attack that doesn't require matched pairs). It fails completely: cosine recovery of 0.004, identical to a random guess. See `benchmarks/adversarial_self_test.py`.

### What it does NOT protect against

Be honest about the threat model:

- **Known-plaintext attack**: d original-rotated pairs (e.g., 1,536 for OpenAI embeddings) fully recovers the key via SVD. Don't let anyone see both the original AND rotated versions of the same content.
- **Pairwise distances are visible**: The server can see which documents are similar to each other, cluster structure, and query patterns. It just can't read what any document says.
- **Key compromise**: If the key file leaks, all rotated vectors are trivially recoverable.
- **RAG output attacks**: Membership inference via LLM output is not mitigated.

### What it is NOT

- Not encryption in the cryptographic sense
- Not differential privacy (no epsilon-delta guarantee)
- Not a substitute for access control on the vector database

**Threat model**: honest-but-curious vector DB provider who sees only rotated vectors and has no access to your original texts or the rotation key.

### What the server CAN learn

Even with rotation, the server can observe:
- Cluster structure (how many topics exist)
- Document similarity graph (which docs are related)
- Query patterns (which clusters you search most)
- Duplicate/near-duplicate documents
- Temporal patterns (when documents are added)

The server CANNOT determine what any document says, infer PII, or run published inversion attacks.

### Comparison with other approaches

| Property | Rotation (ours) | Differential Privacy | Homomorphic Encryption | IronCore Cloaked AI |
|----------|----------------|---------------------|----------------------|-------------------|
| Search quality | Identical (lossless) | 5-30% recall loss | Identical | ~5% recall loss |
| Latency overhead | <0.1ms per vector | Negligible | 1000-10000x | SDK overhead |
| Deployment | One numpy matmul | Drop-in | Custom server | SDK + license |
| License | Apache 2.0 | N/A | N/A | AGPL / $599+/mo |
| Known-plaintext resistant | No (d pairs breaks it) | Yes | Yes | Partially |

### Key management

Treat `.tqkey` files like SSH private keys:
- Don't commit to git (add `*.tqkey` to .gitignore)
- Back up securely -- if lost, you can't unrotate (search still works)
- Use `from_seed()` with a 128-bit seed to share keys without large files
- Use `rekey_vectors()` to rotate to a new key without exposing originals

### Benchmarks

| Dimension | Single vector | Batch 10K | Key generation | Key file |
|-----------|--------------|-----------|---------------|---------|
| 384 | 0.03 ms | 8.7 ms | 31 ms | 0.6 MB |
| 768 | 0.06 ms | 25 ms | 141 ms | 2.4 MB |
| 1536 | 0.11 ms | 88 ms | 465 ms | 9.4 MB |

### Integration examples

Works with any vector DB that accepts float arrays:

```python
# Pinecone
rotated = encoder.rotate(embeddings)
index.upsert(vectors=[(id, vec.tolist(), meta) for id, vec, meta in zip(ids, rotated, metadata)])

# ChromaDB
collection.add(embeddings=encoder.rotate(embeddings).tolist(), ids=ids)

# LangChain (wrap any embedding model)
class PrivateEmbeddings(Embeddings):
    def __init__(self, base, encoder):
        self.base, self.encoder = base, encoder
    def embed_documents(self, texts):
        return self.encoder.rotate(np.array(self.base.embed_documents(texts))).tolist()
    def embed_query(self, text):
        return self.encoder.rotate(np.array(self.base.embed_query(text))).tolist()

# sentence-transformers
embeddings = model.encode(texts)
rotated = encoder.rotate(embeddings)
```

### Privacy + compression

Combine both: rotate for privacy, then quantize for 8x compression.

```python
compressed = encoder.rotate_and_compress(embeddings, bits=4)
idx, scores = compressed.search(encoder.rotate(query), top_k=10)
compressed.save("private_index.npz")
```

---

## Compression

8x instant compression, no training needed.

First open-source implementation of Google's TurboQuant ([ICLR 2026](https://arxiv.org/abs/2504.19874)) for vector search.

```python
from turboquant_vectors import compress, search

compressed = compress(embeddings, bits=4)  # 307 MB -> 38 MB
indices, scores = search(compressed, query, top_k=10)
```

### Why

FAISS Product Quantization requires k-means training per dataset. TurboQuant is instant (data-oblivious), compresses 2-2.5x faster, and gets up to +8pp better recall at the same storage budget.

### Benchmarks on real OpenAI embeddings (10K vectors, 1536-dim)

Tested on Qdrant's `dbpedia-entities-openai3-text-embedding-3-small` dataset from HuggingFace. Real embeddings, not synthetic.

| Bits | TurboQuant Recall@10 | FAISS PQ Recall@10 | Delta | TQ Compress Time |
|------|---------------------|-------------------|-------|-----------------|
| 2-bit | **90.6%** | 90.2% | **+0.4pp** | 1.2s (no training) |
| 4-bit | **96.6%** | 96.1% | **+0.5pp** | 1.7s (no training) |
| 8-bit | **99.3%** | 98.1% | **+1.2pp** | 9.5s (no training) |

TurboQuant needs zero training (data-oblivious). FAISS PQ requires k-means training. Reproduce: `python benchmarks/real_data_benchmark.py`

---

## Install

```bash
pip install turboquant-vectors
```

Requires only numpy. No torch, no scipy for the privacy module.

## Full API

### PrivateEncoder

```python
PrivateEncoder.generate(dim)           # New key from OS entropy
PrivateEncoder.from_seed(dim, seed)    # Deterministic key (seed >= 2^64)
PrivateEncoder.load_key(path)          # Load from .tqkey file

encoder.rotate(vectors)                # Apply rotation
encoder.unrotate(vectors)              # Reverse rotation (needs key)
encoder.save_key(path)                 # Save to .tqkey file
encoder.fingerprint()                  # 16-char hex key ID
encoder.rekey_vectors(vecs, old_enc)   # Switch keys without unrotating
encoder.rotate_and_compress(vecs, 4)   # Privacy + compression
encoder.make_canary() / verify_canary()  # Key verification without originals
```

### Compression

```python
compress(vectors, bits=4)              # Compress vectors
decompress(compressed)                 # Restore to float32
search(compressed, query, top_k=10)    # Search compressed vectors
compressed.save(path) / .load(path)    # Persistence
```

## Paper

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Zandieh, Daliri, Hadian, Mirrokni (Google Research)
ICLR 2026 | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

Independent implementation, not affiliated with Google Research.

## License

Apache 2.0

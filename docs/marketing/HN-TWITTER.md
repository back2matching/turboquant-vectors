# Hacker News + Twitter -- turboquant-vectors

---

## Hacker News

**Title:** Show HN: Zero-loss privacy for vector embeddings (orthogonal rotation, pip install)

**Text:**

Vec2Text recovers 92% of text from embeddings. OWASP made it LLM08. Most defenses (DP, HE) sacrifice search quality or add massive latency.

Orthogonal rotation preserves ALL distance metrics exactly (it's a mathematical identity, not an approximation). Rotate your embeddings before storing in any vector DB. Published inversion attacks fail. Search results are identical.

    pip install turboquant-vectors

    from turboquant_vectors import PrivateEncoder
    encoder = PrivateEncoder.generate(dim=1536)
    rotated = encoder.rotate(embeddings)

92 tests proving: Recall@10 = 1.000 exactly, classifier attacks drop from 100% to 0%, round-trip error < 1e-5.

Known weakness (documented honestly): d original-to-rotated pairs recovers the key via SVD. This is a privacy layer for honest-but-curious storage providers, not cryptographic encryption.

Only dependency: numpy. Apache 2.0.

https://github.com/back2matching/turboquant-vectors

---

## Twitter/X Thread (7 tweets)

**1/7** Your vector embeddings are not private. Vec2Text (EMNLP 2023) recovers 92% of original text from embeddings. Patient names from clinical records. OWASP made this LLM08. If you're using Pinecone, Weaviate, or Qdrant, your text is recoverable.

**2/7** The standard fix is differential privacy. It costs you 5-30% recall. Or homomorphic encryption. It costs you 1000x latency. Both require you to choose: privacy OR search quality.

**3/7** There's a third option nobody packaged: orthogonal rotation.

<Qx, Qy> = x^T Q^T Q y = x^T y = <x, y>

Cosine similarity, L2 distance, inner product -- all preserved exactly. Not a tradeoff. A mathematical identity.

**4/7** pip install turboquant-vectors

Three lines of Python. Recall@10 stays at 100.0%. Vec2Text recovers gibberish.

```python
from turboquant_vectors import PrivateEncoder
encoder = PrivateEncoder.generate(dim=1536)
rotated = encoder.rotate(embeddings)
```

**5/7** We tested it properly. 92 tests:
- Classifier trained on originals: 100% accuracy
- Same classifier on rotated vectors: 0%
- Recall@10: exactly 1.000 (20/20 queries on 10K vectors)
- Round-trip error: < 1e-5

**6/7** What it doesn't protect (we're honest):
- 1,536 known original-to-rotated pairs recovers the key via SVD
- Server sees which documents are similar (pairwise distances preserved)
- Not encryption. Not DP. A privacy layer for honest-but-curious storage.

**7/7** Apache 2.0. One dependency (numpy). Works with Pinecone, Weaviate, Qdrant, ChromaDB, FAISS, pgvector, LangChain, LlamaIndex.

First pip package for lossless embedding privacy via orthogonal rotation.

github.com/back2matching/turboquant-vectors

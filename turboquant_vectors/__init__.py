"""
turboquant-vectors: Compress embeddings 6x instantly with TurboQuant.

First pip package applying Google's TurboQuant (ICLR 2026) to vector search.
No training needed. Drop-in for numpy, FAISS, ChromaDB.

Usage:
    from turboquant_vectors import compress, decompress, search

    compressed = compress(embeddings, bits=4)  # 6x smaller
    results = search(compressed, query, top_k=10)  # search on compressed
"""

__version__ = "0.1.0"

from turboquant_vectors.core import compress, decompress, search, TurboQuantVectors

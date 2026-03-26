"""
turboquant-vectors: Compress and protect embeddings with TurboQuant.

First pip package applying Google's TurboQuant (ICLR 2026) to vector search.
No training needed. Drop-in for numpy, FAISS, ChromaDB.

Compression:
    from turboquant_vectors import compress, decompress, search
    compressed = compress(embeddings, bits=4)  # 8x smaller

Privacy:
    from turboquant_vectors import PrivateEncoder
    encoder = PrivateEncoder.generate(dim=1536)
    rotated = encoder.rotate(embeddings)  # search works identically
"""

__version__ = "0.3.0"

from turboquant_vectors.core import compress, decompress, search, TurboQuantVectors
from turboquant_vectors.private import PrivateEncoder, CompressedPrivateVectors

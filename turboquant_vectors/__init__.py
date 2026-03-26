"""
turboquant-vectors: Zero-loss embedding privacy + compression.

Privacy:
    from turboquant_vectors import PrivateEncoder
    encoder = PrivateEncoder.generate(dim=1536)
    rotated = encoder.rotate(embeddings)  # search works identically

Compression:
    from turboquant_vectors import compress, decompress, search
    compressed = compress(embeddings, bits=4)  # 8x smaller
"""

__version__ = "0.3.1"

from turboquant_vectors.core import compress, decompress, search, TurboQuantVectors
from turboquant_vectors.private import PrivateEncoder, CompressedPrivateVectors
from turboquant_vectors._types import DimensionError, KeyMismatchError

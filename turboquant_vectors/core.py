"""
Core TurboQuant vector compression and search.

Applies random orthogonal rotation + optimal scalar quantization to compress
high-dimensional vectors (embeddings, search indices) with minimal quality loss.

The rotation makes coordinates approximately independent, enabling near-optimal
per-coordinate quantization without any training data (data-oblivious).
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class CompressedVectors:
    """Compressed vector representation."""
    indices: np.ndarray    # uint8 quantized indices, shape (n, dim)
    norms: np.ndarray      # float32 vector norms, shape (n,)
    rotation: np.ndarray   # float32 rotation matrix, shape (dim, dim)
    codebook: np.ndarray   # float32 centroids, shape (2^bits,)
    dim: int
    bits: int
    n_vectors: int

    def save(self, path: str):
        """Save compressed vectors to .npz file."""
        np.savez_compressed(path,
            indices=self.indices, norms=self.norms,
            rotation=self.rotation, codebook=self.codebook,
            meta=np.array([self.dim, self.bits, self.n_vectors]))

    @staticmethod
    def load(path: str) -> 'CompressedVectors':
        """Load compressed vectors from .npz file."""
        data = np.load(path)
        meta = data['meta']
        return CompressedVectors(
            indices=data['indices'], norms=data['norms'],
            rotation=data['rotation'], codebook=data['codebook'],
            dim=int(meta[0]), bits=int(meta[1]), n_vectors=int(meta[2]))

    @property
    def memory_bytes(self) -> int:
        """Memory used by compressed representation (packed bits)."""
        # Indices: bits per element, packed
        index_bits = self.n_vectors * self.dim * self.bits
        index_bytes = (index_bits + 7) // 8
        # Norms + rotation + codebook are overhead (amortized over many vectors)
        return index_bytes + self.norms.nbytes + self.rotation.nbytes + self.codebook.nbytes

    @property
    def packed_memory_bytes(self) -> int:
        """Memory with packed bit storage (what you'd get in production)."""
        index_bits = self.n_vectors * self.dim * self.bits
        index_bytes = (index_bits + 7) // 8
        return index_bytes + self.norms.nbytes

    @property
    def original_bytes(self) -> int:
        """Memory that would be used by uncompressed float32 vectors."""
        return self.n_vectors * self.dim * 4  # float32

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / max(self.memory_bytes, 1)


class TurboQuantVectors:
    """
    TurboQuant vector compressor.

    Compresses high-dimensional vectors using random rotation + optimal
    scalar quantization. No training data needed (data-oblivious).
    """

    def __init__(self, dim: int, bits: int = 4, seed: int = 42):
        self.dim = dim
        self.bits = bits
        self.n_centroids = 2 ** bits

        # Generate random orthogonal rotation via QR decomposition
        rng = np.random.RandomState(seed)
        gaussian = rng.randn(dim, dim).astype(np.float32)
        Q, R = np.linalg.qr(gaussian)
        # Ensure proper rotation (det = +1)
        Q = Q * np.sign(np.diag(R))
        self.rotation = Q.astype(np.float32)
        self.rotation_t = Q.T.astype(np.float32)

        # Compute optimal codebook for Beta distribution
        self.codebook = self._compute_codebook(dim, bits)

    def _compute_codebook(self, dim: int, bits: int) -> np.ndarray:
        """Optimal codebook for Gaussian-like distribution after rotation."""
        sigma = 1.0 / math.sqrt(dim)
        # Lloyd-Max optimal centroids for Gaussian(0, sigma^2)
        if bits == 1:
            c = math.sqrt(2.0 / (math.pi * dim))
            return np.array([-c, c], dtype=np.float32)
        elif bits == 2:
            lloyd = [0.4528, 1.5104]
        elif bits == 3:
            lloyd = [0.1284, 0.3882, 0.6568, 0.9423]
        elif bits == 4:
            lloyd = [0.1284, 0.3882, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326]
        else:
            # For higher bits, use uniform quantization
            n = 2 ** (bits - 1)
            lloyd = [(i + 0.5) / n * 3.0 for i in range(n)]

        centroids = []
        for v in reversed(lloyd):
            centroids.append(-v * sigma)
        for v in lloyd:
            centroids.append(v * sigma)
        return np.array(centroids, dtype=np.float32)

    def compress(self, vectors: np.ndarray) -> CompressedVectors:
        """
        Compress vectors using TurboQuant rotation + quantization.

        Args:
            vectors: float32 array of shape (n, dim)

        Returns:
            CompressedVectors with quantized indices and metadata
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim

        n = vectors.shape[0]

        # Compute norms
        norms = np.linalg.norm(vectors, axis=1)

        # Normalize to unit vectors
        safe_norms = np.maximum(norms, 1e-10)
        unit_vectors = vectors / safe_norms[:, np.newaxis]

        # Rotate: y = x @ rotation_t (batch matrix multiply)
        rotated = unit_vectors @ self.rotation_t

        # Quantize: find nearest centroid per coordinate (batched to avoid OOM)
        batch_size = max(1, min(10000, 500_000_000 // (self.dim * self.n_centroids * 4)))
        indices = np.empty((n, self.dim), dtype=np.uint8)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = rotated[start:end]
            dists = np.abs(batch[:, :, np.newaxis] - self.codebook[np.newaxis, np.newaxis, :])
            indices[start:end] = dists.argmin(axis=2).astype(np.uint8)

        return CompressedVectors(
            indices=indices,
            norms=norms.astype(np.float32),
            rotation=self.rotation,
            codebook=self.codebook,
            dim=self.dim,
            bits=self.bits,
            n_vectors=n,
        )

    def decompress(self, compressed: CompressedVectors) -> np.ndarray:
        """Decompress back to float32 vectors."""
        # Look up centroids
        y_hat = compressed.codebook[compressed.indices]  # (n, dim)

        # Inverse rotation
        x_hat = y_hat @ compressed.rotation  # (n, dim)

        # Rescale by norms
        x_hat = x_hat * compressed.norms[:, np.newaxis]

        return x_hat

    def decompress_normalized(self, compressed: CompressedVectors) -> np.ndarray:
        """Decompress and L2-normalize (cached for repeated search)."""
        if not hasattr(compressed, '_cache_normalized'):
            decompressed = self.decompress(compressed)
            norms = np.linalg.norm(decompressed, axis=1, keepdims=True)
            compressed._cache_normalized = decompressed / np.maximum(norms, 1e-10)
        return compressed._cache_normalized

    def search(self, compressed: CompressedVectors, query: np.ndarray,
               top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search compressed vectors for nearest neighbors to query.

        First call decompresses vectors (cached for subsequent queries).
        Supports single query (dim,) or batch queries (n_queries, dim).

        Returns:
            (indices, scores) — top_k nearest neighbors and their scores
        """
        query = np.asarray(query, dtype=np.float32)
        single = query.ndim == 1
        if single:
            query = query[np.newaxis, :]

        # Normalize query
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query_unit = query / np.maximum(query_norm, 1e-10)

        # Get normalized vectors (cached after first call)
        vec_unit = self.decompress_normalized(compressed)

        # Cosine similarity
        scores = query_unit @ vec_unit.T  # (n_queries, n_vectors)

        # Top-k
        if scores.shape[1] <= top_k:
            top_indices = np.argsort(-scores, axis=1)
        else:
            # Use argpartition for speed on large sets
            top_indices = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
            # Sort the top-k by score
            for i in range(len(top_indices)):
                order = np.argsort(-scores[i, top_indices[i]])
                top_indices[i] = top_indices[i, order]

        top_scores = np.take_along_axis(scores, top_indices, axis=1)

        if single:
            return top_indices.squeeze(0), top_scores.squeeze(0)
        return top_indices, top_scores


# Convenience functions

def compress(vectors: np.ndarray, bits: int = 4, seed: int = 42) -> CompressedVectors:
    """Compress vectors using TurboQuant. One-liner API."""
    dim = vectors.shape[1]
    tq = TurboQuantVectors(dim=dim, bits=bits, seed=seed)
    return tq.compress(vectors)


def decompress(compressed: CompressedVectors) -> np.ndarray:
    """Decompress TurboQuant vectors back to float32."""
    tq = TurboQuantVectors(dim=compressed.dim, bits=compressed.bits)
    return tq.decompress(compressed)


def search(compressed: CompressedVectors, query: np.ndarray,
           top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Search compressed vectors for nearest neighbors."""
    tq = TurboQuantVectors(dim=compressed.dim, bits=compressed.bits)
    return tq.search(compressed, query, top_k)

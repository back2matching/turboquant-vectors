"""
Low-level shared operations: rotation matrices and codebooks.

Shared by both the privacy module (PrivateEncoder) and compression module (TurboQuantVectors).
Generates random orthogonal matrices from the Haar measure on O(d) via QR decomposition.
"""

import hashlib
import math
import numpy as np
from typing import Optional


def generate_rotation_matrix(dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random orthogonal matrix from the Haar measure on O(d).

    Uses QR decomposition of a Gaussian random matrix (Stewart 1980).
    The matrix is generated in float64 for numerical stability, then
    cast to float32 for storage and computation.

    Args:
        dim: Matrix dimension.
        seed: RNG seed. If None, uses OS entropy via numpy default_rng().

    Returns:
        Orthogonal matrix Q, shape (dim, dim), dtype float32.
        Satisfies Q^T Q = I (up to float32 precision).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Generate in float64 for numerical stability of QR decomposition
    gaussian = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(gaussian)
    # Ensure proper rotation (det = +1) by correcting sign ambiguity
    Q = Q * np.sign(np.diag(R))
    return Q.astype(np.float32)


def derive_seed_from_key(seed: int, dim: int) -> int:
    """
    Derive an RNG seed from a user seed + dimension using HMAC-SHA256.

    This prevents cross-dimension correlation: the same user seed with
    different dimensions produces completely unrelated rotation matrices.

    Args:
        seed: User-provided seed (must be >= 2^64).
        dim: Embedding dimension.

    Returns:
        Derived 128-bit integer seed for numpy RNG.
    """
    import hmac
    seed_bytes = seed.to_bytes(max(16, (seed.bit_length() + 7) // 8), 'big')
    info = f"tqkey-d{dim}".encode()
    # HMAC-SHA256 key derivation (seed as key, dimension info as message)
    derived = hmac.new(seed_bytes, info, hashlib.sha256).digest()
    return int.from_bytes(derived[:16], 'big')


def validate_orthogonal(matrix: np.ndarray, tol: float = 1e-4) -> bool:
    """Check if a matrix is approximately orthogonal (Q^T Q ≈ I)."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    identity_approx = matrix.T @ matrix
    identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    return np.allclose(identity_approx, identity, atol=tol)


def fingerprint(matrix: np.ndarray) -> str:
    """Compute a short hex fingerprint of a rotation matrix (first 16 chars of SHA-256)."""
    return hashlib.sha256(matrix.tobytes()).hexdigest()[:16]


def compute_codebook(dim: int, bits: int) -> np.ndarray:
    """
    Optimal codebook for Gaussian-like distribution after rotation.

    Uses Lloyd-Max optimal centroids for bits 1-4, uniform quantization for 5-8.
    Shared by both TurboQuantVectors (compression) and PrivateEncoder (rotate_and_compress).
    """
    sigma = 1.0 / math.sqrt(dim)
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
        n = 2 ** (bits - 1)
        lloyd = [(i + 0.5) / n * 3.0 for i in range(n)]

    centroids = []
    for v in reversed(lloyd):
        centroids.append(-v * sigma)
    for v in lloyd:
        centroids.append(v * sigma)
    return np.array(centroids, dtype=np.float32)

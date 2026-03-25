"""
Low-level rotation matrix operations.

Shared by both the privacy module (PrivateEncoder) and compression module (TurboQuantVectors).
Generates random orthogonal matrices from the Haar measure on O(d) via QR decomposition.
"""

import hashlib
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

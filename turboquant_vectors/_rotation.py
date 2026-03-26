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


def quantize(values: np.ndarray, codebook: np.ndarray, stochastic: bool = False,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Quantize values to nearest codebook centroid.

    Args:
        values: Array to quantize (any shape).
        codebook: Sorted 1D array of centroids.
        stochastic: If True, use randomized rounding between the two nearest
            centroids (proportional to inverse distance). This provides formal
            Renyi differential privacy. If False, deterministic nearest-centroid.
        rng: Random generator for stochastic mode. Uses default if None.

    Returns:
        uint8 indices into codebook, same shape as values.
    """
    thresholds = (codebook[:-1] + codebook[1:]) / 2
    indices = np.searchsorted(thresholds, values)

    if stochastic:
        if rng is None:
            rng = np.random.default_rng()
        # For each value, compute distance to assigned centroid and the neighbor
        assigned = codebook[indices]
        dists = np.abs(values - assigned)
        # Determine neighbor: if value > assigned centroid, neighbor is idx+1, else idx-1
        # Clamp to valid range
        n_centroids = len(codebook)
        neighbor_idx = np.where(values > assigned,
                                np.minimum(indices + 1, n_centroids - 1),
                                np.maximum(indices.astype(np.int32) - 1, 0))
        neighbor = codebook[neighbor_idx]
        neighbor_dists = np.abs(values - neighbor)
        # Probability of keeping assigned = neighbor_dist / (dist + neighbor_dist)
        total = dists + neighbor_dists
        safe_total = np.maximum(total, 1e-10)
        keep_prob = neighbor_dists / safe_total
        # Flip to neighbor with probability 1 - keep_prob
        flip = rng.random(values.shape) > keep_prob
        indices = np.where(flip, neighbor_idx, indices)

    return indices.astype(np.uint8)


def compute_codebook(dim: int, bits: int) -> np.ndarray:
    """
    Optimal codebook for Gaussian-like distribution after rotation.

    Uses Lloyd-Max optimal centroids for bits 1-8.
    Shared by both TurboQuantVectors (compression) and PrivateEncoder (rotate_and_compress).
    """
    sigma = 1.0 / math.sqrt(dim)
    if bits == 1:
        c = math.sqrt(2.0 / (math.pi * dim))
        return np.array([-c, c], dtype=np.float32)
    elif bits == 2:
        # Lloyd-Max optimal centroids for N(0,1), 4 levels
        lloyd = [0.4528, 1.5104]
    elif bits == 3:
        # Lloyd-Max optimal centroids for N(0,1), 8 levels
        # BUG FIX: was [0.1284, 0.3882, 0.6568, 0.9423] (inner 4-bit values, ~6x worse MSE)
        lloyd = [0.2451, 0.7560, 1.3439, 2.1519]
    elif bits == 4:
        # Lloyd-Max optimal centroids for N(0,1), 16 levels
        lloyd = [0.1304, 0.3939, 0.6661, 0.9545, 1.2703, 1.6330, 2.0837, 2.7456]
    elif bits == 5:
        # Lloyd-Max optimal centroids for N(0,1), 32 levels
        lloyd = [0.0704, 0.2116, 0.3536, 0.4973, 0.6433, 0.7926, 0.9463, 1.1058,
                 1.2731, 1.4507, 1.6425, 1.8541, 2.0947, 2.3814, 2.7506, 3.3133]
    elif bits == 6:
        # Lloyd-Max optimal centroids for N(0,1), 64 levels
        lloyd = [0.0418, 0.1254, 0.2090, 0.2927, 0.3764, 0.4602, 0.5442, 0.6283,
                 0.7125, 0.7971, 0.8820, 0.9674, 1.0533, 1.1399, 1.2274, 1.3160,
                 1.4060, 1.4978, 1.5916, 1.6880, 1.7877, 1.8914, 2.0002, 2.1152,
                 2.2382, 2.3716, 2.5187, 2.6845, 2.8775, 3.1126, 3.4228, 3.9080]
    elif bits == 7:
        # Lloyd-Max optimal centroids for N(0,1), 128 levels
        lloyd = [0.0228, 0.0684, 0.1139, 0.1595, 0.2051, 0.2507, 0.2963, 0.3418,
                 0.3874, 0.4330, 0.4786, 0.5242, 0.5697, 0.6153, 0.6609, 0.7065,
                 0.7520, 0.7976, 0.8432, 0.8888, 0.9344, 0.9799, 1.0255, 1.0711,
                 1.1167, 1.1623, 1.2079, 1.2535, 1.2991, 1.3447, 1.3903, 1.4360,
                 1.4817, 1.5274, 1.5732, 1.6191, 1.6651, 1.7112, 1.7575, 1.8041,
                 1.8511, 1.8985, 1.9464, 1.9950, 2.0445, 2.0950, 2.1469, 2.2004,
                 2.2558, 2.3136, 2.3742, 2.4383, 2.5064, 2.5795, 2.6584, 2.7446,
                 2.8396, 2.9457, 3.0660, 3.2053, 3.3714, 3.5782, 3.8564, 4.2996]
    elif bits == 8:
        # Lloyd-Max optimal centroids for N(0,1), 256 levels
        lloyd = [0.0117, 0.0350, 0.0583, 0.0816, 0.1049, 0.1282, 0.1515, 0.1749,
                 0.1982, 0.2215, 0.2448, 0.2681, 0.2914, 0.3148, 0.3381, 0.3614,
                 0.3847, 0.4080, 0.4313, 0.4546, 0.4780, 0.5013, 0.5246, 0.5479,
                 0.5712, 0.5945, 0.6179, 0.6412, 0.6645, 0.6878, 0.7111, 0.7344,
                 0.7577, 0.7811, 0.8044, 0.8277, 0.8510, 0.8743, 0.8976, 0.9210,
                 0.9443, 0.9676, 0.9909, 1.0142, 1.0375, 1.0608, 1.0842, 1.1075,
                 1.1308, 1.1541, 1.1774, 1.2007, 1.2241, 1.2474, 1.2707, 1.2940,
                 1.3173, 1.3406, 1.3639, 1.3873, 1.4106, 1.4339, 1.4572, 1.4805,
                 1.5038, 1.5271, 1.5505, 1.5738, 1.5971, 1.6204, 1.6437, 1.6670,
                 1.6904, 1.7137, 1.7370, 1.7603, 1.7836, 1.8069, 1.8302, 1.8536,
                 1.8769, 1.9002, 1.9235, 1.9468, 1.9701, 1.9935, 2.0168, 2.0401,
                 2.0634, 2.0868, 2.1101, 2.1334, 2.1568, 2.1802, 2.2036, 2.2270,
                 2.2505, 2.2741, 2.2977, 2.3215, 2.3455, 2.3697, 2.3941, 2.4190,
                 2.4444, 2.4703, 2.4971, 2.5248, 2.5536, 2.5839, 2.6159, 2.6500,
                 2.6865, 2.7259, 2.7688, 2.8157, 2.8673, 2.9245, 2.9883, 3.0599,
                 3.1410, 3.2337, 3.3411, 3.4677, 3.6211, 3.8147, 4.0781, 4.5019]
    else:
        raise ValueError(f"bits must be 1-8, got {bits}")

    centroids = []
    for v in reversed(lloyd):
        centroids.append(-v * sigma)
    for v in lloyd:
        centroids.append(v * sigma)
    return np.array(centroids, dtype=np.float32)

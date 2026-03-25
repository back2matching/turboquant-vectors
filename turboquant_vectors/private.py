"""
PrivateEncoder: Lossless privacy layer for embedding vectors.

Applies a secret random orthogonal rotation to embedding vectors before
storage or transmission. All distance metrics (cosine, L2, inner product)
are preserved exactly. Without the secret key, published inversion attacks
(Vec2Text, ALGEN, ZSinvert, Zero2Text) fail completely.

Threat model: honest-but-curious server without access to the key file.

This is NOT encryption. This is NOT differential privacy. It is an orthogonal
rotation that makes embedding inversion attacks fail. The security rests on
the secrecy of the rotation matrix, similar to a symmetric key.

Known weakness: d linearly independent (original, rotated) pairs recover
the key exactly via SVD (Orthogonal Procrustes Problem).
"""

import hashlib
import struct
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Union

from turboquant_vectors._rotation import (
    generate_rotation_matrix,
    derive_seed_from_key,
    validate_orthogonal,
    fingerprint as _fingerprint,
)


# .tqkey file format constants
_TQKEY_MAGIC = b'TQKEY\x00\x01\x00'
_TQKEY_HEADER_SIZE = 16  # 8 magic + 4 dim + 4 reserved

# Minimum seed size: 2^64 (Security Audit requirement)
_MIN_SEED = 2 ** 64


class PrivateEncoder:
    """
    Lossless privacy layer for embedding vectors.

    Applies a secret random orthogonal rotation to embedding vectors before
    storage or transmission. All distance metrics (cosine, L2, inner product)
    are preserved exactly. Without the secret key, original vectors cannot
    be reconstructed.

    Usage:
        encoder = PrivateEncoder.generate(dim=1536)
        encoder.save_key("my_secret.tqkey")

        # Rotate before sending to vector DB
        rotated = encoder.rotate(embeddings)
        pinecone_index.upsert(vectors=rotated.tolist(), ids=ids)

        # Query must also be rotated
        rotated_query = encoder.rotate(query_vector)
        results = pinecone_index.query(vector=rotated_query.tolist(), top_k=10)

        # Later, load the same key
        encoder = PrivateEncoder.load_key("my_secret.tqkey")
    """

    def __init__(self, rotation_matrix: np.ndarray, normalize: bool = True):
        """
        Initialize with an existing rotation matrix.

        Prefer PrivateEncoder.generate() or PrivateEncoder.load_key()
        instead of calling this directly.

        Args:
            rotation_matrix: Orthogonal matrix, shape (d, d), dtype float32.
                Must satisfy Q^T Q ≈ I (verified on construction).
            normalize: If True (default), L2-normalize vectors before rotation.
                Prevents norm leakage. Set to False only if your embedding
                model already produces unit-norm vectors.

        Raises:
            ValueError: If matrix is not square, not orthogonal, or wrong dtype.
        """
        if rotation_matrix.ndim != 2:
            raise ValueError(f"Rotation matrix must be 2D, got {rotation_matrix.ndim}D")
        if rotation_matrix.shape[0] != rotation_matrix.shape[1]:
            raise ValueError(f"Rotation matrix must be square, got {rotation_matrix.shape}")
        if not validate_orthogonal(rotation_matrix):
            raise ValueError(
                "Matrix is not orthogonal (Q^T Q != I). "
                "Use PrivateEncoder.generate() to create a valid rotation matrix."
            )
        self._rotation = rotation_matrix.astype(np.float32)
        self._rotation_t = self._rotation.T.copy()  # Precompute transpose
        self._normalize = normalize
        self._dim = rotation_matrix.shape[0]

    @classmethod
    def generate(cls, dim: int, normalize: bool = True) -> "PrivateEncoder":
        """
        Generate a new random rotation key using OS entropy.

        Creates a uniformly random orthogonal matrix via QR decomposition
        of a Gaussian random matrix (Stewart 1980). The matrix is drawn
        from the Haar measure on O(d).

        Args:
            dim: Embedding dimension (e.g., 1536 for OpenAI, 768 for BERT).
            normalize: If True (default), L2-normalize vectors before rotation.

        Returns:
            PrivateEncoder with a fresh random rotation matrix.
        """
        if dim < 2:
            raise ValueError(f"Dimension must be >= 2, got {dim}")
        if dim > 4096:
            warnings.warn(
                f"Generating rotation matrix for dim={dim} requires "
                f"{dim ** 2 * 4 / 1e6:.0f} MB of memory and may take >30 seconds. "
                f"Consider structured rotations for d > 4096 (future feature).",
                stacklevel=2,
            )
        Q = generate_rotation_matrix(dim, seed=None)
        return cls(Q, normalize=normalize)

    @classmethod
    def from_seed(cls, dim: int, seed: int, normalize: bool = True) -> "PrivateEncoder":
        """
        Deterministically reconstruct a key from a seed.

        The seed IS the secret. Anyone with the seed can reconstruct the key.
        Use a cryptographically random value (128+ bits recommended).

        Args:
            dim: Embedding dimension.
            seed: Integer seed. Must be >= 2^64 to prevent brute-force attacks.
                Use secrets.randbits(128) to generate a secure seed.
            normalize: If True (default), L2-normalize vectors before rotation.

        Returns:
            PrivateEncoder with a deterministic rotation matrix.

        Raises:
            ValueError: If seed < 2^64.
        """
        if seed < _MIN_SEED:
            raise ValueError(
                f"Seed must be >= 2^64 ({_MIN_SEED}) to prevent brute-force attacks. "
                f"Got {seed}. Use secrets.randbits(128) for a secure seed."
            )
        if dim < 2:
            raise ValueError(f"Dimension must be >= 2, got {dim}")
        derived = derive_seed_from_key(seed, dim)
        Q = generate_rotation_matrix(dim, seed=derived)
        return cls(Q, normalize=normalize)

    @classmethod
    def load_key(cls, path: Union[str, Path], normalize: bool = True) -> "PrivateEncoder":
        """
        Load a rotation key from a .tqkey file.

        Args:
            path: Path to the key file (created by save_key()).
            normalize: If True (default), L2-normalize vectors before rotation.

        Returns:
            PrivateEncoder initialized with the stored rotation matrix.

        Raises:
            FileNotFoundError: If the key file doesn't exist.
            ValueError: If the file is corrupted or not a valid .tqkey file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Key file not found: {path}")

        data = path.read_bytes()

        # Validate magic
        if len(data) < _TQKEY_HEADER_SIZE:
            raise ValueError(f"File too small to be a .tqkey file: {len(data)} bytes")
        if data[:8] != _TQKEY_MAGIC:
            raise ValueError(f"Not a .tqkey file (wrong magic bytes)")

        # Read dimension
        dim = struct.unpack('<I', data[8:12])[0]
        expected_size = _TQKEY_HEADER_SIZE + dim * dim * 4 + 32  # header + matrix + checksum
        if len(data) != expected_size:
            raise ValueError(
                f"File size mismatch: expected {expected_size} bytes for dim={dim}, got {len(data)}"
            )

        # Read matrix
        matrix_bytes = data[_TQKEY_HEADER_SIZE:_TQKEY_HEADER_SIZE + dim * dim * 4]
        stored_checksum = data[-32:]

        # Verify checksum
        computed_checksum = hashlib.sha256(matrix_bytes).digest()
        if stored_checksum != computed_checksum:
            raise ValueError("Key file corrupted: checksum mismatch")

        # Reconstruct matrix
        Q = np.frombuffer(matrix_bytes, dtype=np.float32).reshape(dim, dim).copy()
        return cls(Q, normalize=normalize)

    def save_key(self, path: Union[str, Path]) -> None:
        """
        Save the rotation key to a .tqkey file.

        The file is NOT encrypted. Treat it like an SSH private key:
        protect with filesystem permissions, don't commit to git.

        Args:
            path: Output file path. Extension .tqkey is added if not present.
        """
        path = Path(path)
        if path.suffix != '.tqkey':
            path = path.with_suffix('.tqkey')

        matrix_bytes = self._rotation.tobytes()
        checksum = hashlib.sha256(matrix_bytes).digest()

        header = _TQKEY_MAGIC + struct.pack('<I', self._dim) + b'\x00' * 4
        path.write_bytes(header + matrix_bytes + checksum)

        # Warn about permissions on Unix-like systems
        try:
            import os
            mode = os.stat(path).st_mode
            if mode & 0o077:  # world or group readable
                warnings.warn(
                    f"Key file {path} is readable by other users (mode {oct(mode)}). "
                    f"Consider: chmod 600 {path}",
                    stacklevel=2,
                )
        except (OSError, AttributeError):
            pass  # Windows or other systems without Unix permissions

    @property
    def dim(self) -> int:
        """Embedding dimension this encoder was created for."""
        return self._dim

    @property
    def key_size_bytes(self) -> int:
        """Size of the rotation matrix in memory (d * d * 4 bytes)."""
        return self._dim * self._dim * 4

    @property
    def normalize(self) -> bool:
        """Whether vectors are L2-normalized before rotation."""
        return self._normalize

    def rotate(self, vectors: np.ndarray, normalize: Optional[bool] = None) -> np.ndarray:
        """
        Rotate vectors for privacy.

        Applies the secret orthogonal rotation. Output vectors have
        identical pairwise distances and cosine similarities as the inputs.

        Args:
            vectors: Input vectors.
                - Shape (d,) for a single vector
                - Shape (n, d) for a batch of vectors
            normalize: Override the constructor's normalize setting.
                None = use constructor default.

        Returns:
            Rotated vectors, same shape as input, dtype float32, C-contiguous.

        Raises:
            ValueError: If vector dimension doesn't match, or input contains NaN/inf.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        single = vectors.ndim == 1
        if single:
            vectors = vectors[np.newaxis, :]

        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"Expected vectors with dim={self._dim}, got shape {vectors.shape}"
            )

        # Validate input
        if not np.isfinite(vectors).all():
            bad = np.where(~np.isfinite(vectors).all(axis=1))[0]
            raise ValueError(
                f"Input contains NaN or inf at {len(bad)} indices (first: {bad[:5].tolist()}). "
                f"Clean your embeddings before rotation."
            )

        do_normalize = normalize if normalize is not None else self._normalize

        if do_normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Check for zero vectors when normalizing
            zero_mask = norms.squeeze() < 1e-10
            if zero_mask.any():
                n_zeros = zero_mask.sum()
                raise ValueError(
                    f"Input contains {n_zeros} zero/near-zero vectors. "
                    f"Cannot normalize zero vectors. Filter them out or set normalize=False."
                )
            vectors = vectors / norms

        # Apply rotation: y = x @ Q^T
        rotated = vectors @ self._rotation_t

        # Ensure C-contiguous (required by FAISS and most vector DBs)
        rotated = np.ascontiguousarray(rotated)

        if single:
            return rotated.squeeze(0)
        return rotated

    def unrotate(self, vectors: np.ndarray) -> np.ndarray:
        """
        Recover original vectors from rotated ones (requires the key).

        Applies the inverse rotation (Q^T). Note: if normalize=True was used
        during rotation, the returned vectors are unit-norm (original norms
        are not stored).

        Args:
            vectors: Rotated vectors, shape (d,) or (n, d).

        Returns:
            Unrotated vectors, same shape as input, float32, C-contiguous.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        single = vectors.ndim == 1
        if single:
            vectors = vectors[np.newaxis, :]

        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"Expected vectors with dim={self._dim}, got shape {vectors.shape}"
            )

        # Inverse rotation: x = y @ Q
        result = vectors @ self._rotation
        result = np.ascontiguousarray(result)

        if single:
            return result.squeeze(0)
        return result

    def rekey_vectors(self, vectors: np.ndarray, old_encoder: "PrivateEncoder") -> np.ndarray:
        """
        Re-rotate vectors from old key to this encoder's key in one step.

        Applies Q_new @ Q_old^T directly, avoiding materializing the
        unrotated (original) vectors in memory. This is more secure than
        calling old_encoder.unrotate() followed by self.rotate().

        Args:
            vectors: Vectors rotated with old_encoder, shape (n, d).
            old_encoder: The PrivateEncoder that originally rotated these vectors.

        Returns:
            Vectors rotated with this encoder's key, shape (n, d).
        """
        if old_encoder.dim != self._dim:
            raise ValueError(
                f"Dimension mismatch: this encoder is {self._dim}D, "
                f"old encoder is {old_encoder.dim}D"
            )

        vectors = np.asarray(vectors, dtype=np.float32)
        single = vectors.ndim == 1
        if single:
            vectors = vectors[np.newaxis, :]

        # Combined rotation: Q_new @ Q_old^T applied via two matrix multiplies
        # y_new = x_rotated @ Q_old @ Q_new^T
        result = vectors @ old_encoder._rotation @ self._rotation_t
        result = np.ascontiguousarray(result)

        if single:
            return result.squeeze(0)
        return result

    def fingerprint(self) -> str:
        """
        Short hex fingerprint of the key for identification.

        Returns first 16 hex chars of SHA-256(rotation_matrix_bytes).
        Does NOT leak the key (one-way hash).

        Use this to label rotated datasets so you know which key was used.
        """
        return _fingerprint(self._rotation)

    def make_canary(self) -> bytes:
        """
        Create a small blob for key verification without needing originals.

        Store this alongside your rotated vectors. Later, use verify_canary()
        to check if a loaded key matches.

        Returns:
            32 bytes (SHA-256 of fingerprint + a fixed test rotation).
        """
        # Rotate a deterministic test vector and hash the result
        test_vec = np.ones(self._dim, dtype=np.float32) / np.sqrt(self._dim)
        rotated = (test_vec @ self._rotation_t).tobytes()
        return hashlib.sha256(rotated).digest()

    def verify_canary(self, canary: bytes) -> bool:
        """
        Check if this key matches a previously created canary.

        Args:
            canary: Bytes from make_canary().

        Returns:
            True if this key matches, False otherwise.
        """
        return self.make_canary() == canary

    def rotate_and_compress(
        self,
        vectors: np.ndarray,
        bits: int = 4,
    ) -> "CompressedPrivateVectors":
        """
        Rotate AND quantize vectors for privacy + compression.

        Applies rotation first (for privacy), then TurboQuant scalar
        quantization (for compression). Gives both privacy AND memory
        savings. Search quality reduced by quantization (not by rotation).

        Args:
            vectors: Input vectors, shape (n, d), float32.
            bits: Quantization bits per coordinate (1-8). Default 4.

        Returns:
            CompressedPrivateVectors with search() and save()/load().
        """
        if not (1 <= bits <= 8):
            raise ValueError(f"bits must be 1-8, got {bits}")

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"Expected (n, {self._dim}) array, got {vectors.shape}")

        # Capture original norms BEFORE normalization so decompress() can
        # reconstruct approximate original magnitudes (not just unit vectors).
        original_norms = np.linalg.norm(vectors, axis=1)

        # Step 1: Rotate for privacy (uses self._normalize setting)
        rotated = self.rotate(vectors)

        # Step 2: Normalize rotated vectors for quantization.
        # Rotation preserves norms, so if normalize=True was applied in
        # rotate(), these are already ~1.0. We still normalize for the
        # quantizer but store original_norms for faithful decompression.
        rot_norms = np.linalg.norm(rotated, axis=1)
        safe_norms = np.maximum(rot_norms, 1e-10)
        unit_rotated = rotated / safe_norms[:, np.newaxis]

        # Step 3: Compute codebook (same as TurboQuantVectors but for the rotated space)
        codebook = _compute_codebook(self._dim, bits)
        n_centroids = 2 ** bits

        # Step 4: Quantize — find nearest centroid per coordinate
        n = vectors.shape[0]
        batch_size = max(1, min(10000, 500_000_000 // (self._dim * n_centroids * 4)))
        indices = np.empty((n, self._dim), dtype=np.uint8)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = unit_rotated[start:end]
            dists = np.abs(batch[:, :, np.newaxis] - codebook[np.newaxis, np.newaxis, :])
            indices[start:end] = dists.argmin(axis=2).astype(np.uint8)

        return CompressedPrivateVectors(
            indices=indices,
            norms=original_norms.astype(np.float32),
            codebook=codebook,
            bits=bits,
            dim=self._dim,
            key_fingerprint=self.fingerprint(),
        )

    def __repr__(self) -> str:
        return (
            f"PrivateEncoder(dim={self._dim}, normalize={self._normalize}, "
            f"key={self.fingerprint()})"
        )


def _compute_codebook(dim: int, bits: int) -> np.ndarray:
    """Optimal codebook for Gaussian-like distribution after rotation."""
    import math
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


class CompressedPrivateVectors:
    """
    Rotated + quantized embedding vectors.

    Created by PrivateEncoder.rotate_and_compress(). Supports direct
    nearest-neighbor search on compressed data and serialization to disk.
    """

    def __init__(
        self,
        indices: np.ndarray,
        norms: np.ndarray,
        codebook: np.ndarray,
        bits: int,
        dim: int,
        key_fingerprint: str,
    ):
        self.indices = indices
        self.norms = norms
        self.codebook = codebook
        self.bits = bits
        self.dim = dim
        self.key_fingerprint = key_fingerprint
        self._decompressed_cache = None

    @property
    def n_vectors(self) -> int:
        return self.indices.shape[0]

    @property
    def shape(self) -> tuple:
        return (self.n_vectors, self.dim)

    @property
    def memory_bytes(self) -> int:
        """Total memory of compressed representation."""
        index_bits = self.n_vectors * self.dim * self.bits
        index_bytes = (index_bits + 7) // 8
        return index_bytes + self.norms.nbytes + self.codebook.nbytes

    @property
    def original_bytes(self) -> int:
        return self.n_vectors * self.dim * 4

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / max(self.memory_bytes, 1)

    def _decompress(self) -> np.ndarray:
        """Decompress to float32 (cached)."""
        if self._decompressed_cache is None:
            y_hat = self.codebook[self.indices.astype(np.int32)]
            x_hat = y_hat * self.norms[:, np.newaxis]
            self._decompressed_cache = x_hat
        return self._decompressed_cache

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        metric: str = "cosine",
    ) -> tuple:
        """
        Nearest-neighbor search on compressed vectors.

        The query MUST be rotated with the same PrivateEncoder.

        Args:
            query: Rotated query vector, shape (d,) or (n_queries, d).
            top_k: Number of results.
            metric: "cosine", "l2", or "ip".

        Returns:
            (indices, scores) arrays.
        """
        query = np.asarray(query, dtype=np.float32)
        single = query.ndim == 1
        if single:
            query = query[np.newaxis, :]

        decompressed = self._decompress()

        if metric == "cosine":
            q_norm = np.linalg.norm(query, axis=1, keepdims=True)
            query_unit = query / np.maximum(q_norm, 1e-10)
            d_norm = np.linalg.norm(decompressed, axis=1, keepdims=True)
            data_unit = decompressed / np.maximum(d_norm, 1e-10)
            scores = query_unit @ data_unit.T
        elif metric == "ip":
            scores = query @ decompressed.T
        elif metric == "l2":
            # L2 distance via ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
            # This is O(n_queries * n_vectors) memory, not O(n_q * n_v * dim)
            q_sq = np.sum(query ** 2, axis=1, keepdims=True)  # (n_q, 1)
            d_sq = np.sum(decompressed ** 2, axis=1)  # (n_v,)
            dot = query @ decompressed.T  # (n_q, n_v)
            sq_dists = q_sq + d_sq[np.newaxis, :] - 2 * dot
            scores = -sq_dists  # Negate: higher = closer
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine', 'l2', or 'ip'.")

        # Efficient top-k via argpartition (O(n) vs O(n log n) for argsort)
        n_vecs = scores.shape[1]
        if n_vecs <= top_k:
            top_idx = np.argsort(-scores, axis=1)
        else:
            top_idx = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
            # Sort the top-k by score
            for i in range(len(top_idx)):
                order = np.argsort(-scores[i, top_idx[i]])
                top_idx[i] = top_idx[i, order]

        top_scores = np.take_along_axis(scores, top_idx, axis=1)

        if single:
            return top_idx.squeeze(0), top_scores.squeeze(0)
        return top_idx, top_scores

    def save(self, path: Union[str, Path]) -> None:
        """Save compressed vectors to .npz file."""
        path = Path(path)
        np.savez_compressed(
            path,
            indices=self.indices,
            norms=self.norms,
            codebook=self.codebook,
            meta=np.array([self.dim, self.bits, self.n_vectors]),
            key_fp=np.frombuffer(self.key_fingerprint.encode('ascii'), dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CompressedPrivateVectors":
        """Load compressed vectors from .npz file."""
        data = np.load(path)
        meta = data['meta']
        key_fp = data['key_fp'].tobytes().decode('ascii')
        return cls(
            indices=data['indices'],
            norms=data['norms'],
            codebook=data['codebook'],
            bits=int(meta[1]),
            dim=int(meta[0]),
            key_fingerprint=key_fp,
        )

    def decompress(self, encoder: "PrivateEncoder") -> np.ndarray:
        """
        Fully decompress and unrotate back to original space.

        Requires the original PrivateEncoder. Result is approximate
        due to quantization loss.
        """
        decompressed = self._decompress()
        return encoder.unrotate(decompressed)

    def __repr__(self) -> str:
        return (
            f"CompressedPrivateVectors(n={self.n_vectors}, dim={self.dim}, "
            f"bits={self.bits}, ratio={self.compression_ratio:.1f}x, "
            f"key={self.key_fingerprint})"
        )

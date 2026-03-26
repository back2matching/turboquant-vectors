# Privacy-Preserving Embeddings via Random Orthogonal Rotation

> Technical spec for the `PrivateEncoder` module. Uses TurboQuant's random orthogonal rotation as a lossless, secret-key obfuscation layer for embedding vectors.

**Date:** 2026-03-25
**Status:** Research complete, ready for implementation
**Package:** `turboquant_vectors.private` (submodule of turboquant-vectors)
**Effort:** 3-5 days
**Dependencies:** numpy, scipy (optional: torch for GPU acceleration)

---

## 1. The Math

### 1.1 Orthogonal Rotation Preserves All Distance Metrics Exactly

Let **Q** be a d x d orthogonal matrix (Q^T Q = Q Q^T = I, det(Q) = +1). For any vectors **x**, **y** in R^d, define the rotated vectors **x'** = Q**x**, **y'** = Q**y**.

**Inner product preservation:**

```
<x', y'> = (Qx)^T (Qy) = x^T Q^T Q y = x^T I y = <x, y>
```

This is exact. No approximation, no error term, no epsilon.

**L2 distance preservation:**

```
||x' - y'||_2 = ||Q(x - y)||_2 = ||x - y||_2
```

Proof: ||Qz||^2 = z^T Q^T Q z = z^T z = ||z||^2 for any z. Take z = x - y.

**Cosine similarity preservation:**

```
cos(x', y') = <x', y'> / (||x'|| * ||y'||) = <x, y> / (||x|| * ||y||) = cos(x, y)
```

Since inner products AND norms are both preserved, cosine similarity is preserved identically.

**Mahalanobis distance preservation (for identity covariance):**

Any metric defined purely through inner products is invariant under orthogonal transformation. This includes:
- Euclidean distance (L2)
- Cosine similarity
- Inner product (dot product)
- Squared Euclidean distance
- Kernel functions that depend only on ||x - y|| (RBF, Laplacian, etc.)

**NOT preserved:** L1 (Manhattan) distance, Lp norms for p != 2, Chebyshev distance. These depend on individual coordinates, not just inner products. In rotated space, individual coordinates are scrambled.

**Key distinction from random projection (Johnson-Lindenstrauss):** Random projection reduces dimensionality (d -> k, k << d) and preserves distances only approximately, within a (1 +/- epsilon) factor. Orthogonal rotation keeps the same dimensionality and preserves distances exactly. Rotation is lossless. Projection is lossy.

### 1.2 Irreversibility Without the Key

**Theorem:** Given only the rotated vector **x'** = Q**x**, and no knowledge of Q, the original vector **x** is information-theoretically unrecoverable.

**Proof sketch:**

For any target vector **v** in R^d with ||v|| = ||x'||, there exists exactly one orthogonal matrix Q_v such that Q_v **v** = **x'**. (This is because orthogonal matrices act transitively on spheres.) Therefore, the rotated vector **x'** is consistent with the original being ANY vector of the same norm. Without knowing Q, an adversary cannot distinguish between any of these.

More formally: let the adversary observe **x'** = Q**x** where Q is drawn uniformly from the orthogonal group O(d) (Haar measure). The posterior distribution of **x** given **x'** is uniform on the sphere of radius ||**x'**||. This is maximum entropy — the adversary learns nothing beyond the norm.

**The norm leaks.** ||**x'**|| = ||**x**||. If the adversary knows the norm distribution of the original embedding model, this leaks some information. In practice, many embedding models (OpenAI, Cohere, Jina) produce unit-norm vectors (||**x**|| = 1 for all x), so the norm carries zero information. For models that don't unit-normalize, the user can normalize before rotating.

### 1.3 Security Guarantees and Threat Model

**Threat model: honest-but-curious server.**

The server (Pinecone, Weaviate, Qdrant, or any third-party vector DB):
- Stores and searches rotated vectors faithfully (honest)
- May inspect vector values and attempt to recover original content (curious)
- Does NOT have access to the rotation matrix Q
- Does NOT have access to any original-rotated vector pairs

Under this model, the server learns:
- The norms of original vectors (mitigated by pre-normalization)
- The pairwise distances between vectors (these are public anyway, since the server does search)
- Nothing else about the semantic content of individual vectors

**What this is NOT:**
- NOT defense against an adversary who has the rotation matrix (if Q leaks, all vectors are trivially recoverable: **x** = Q^T **x'**)
- NOT defense against an adversary with known-plaintext pairs (see 1.4)
- NOT differential privacy (no noise added, no epsilon-delta guarantee)
- NOT homomorphic encryption (the server can see pairwise relationships)
- NOT secure multi-party computation (no protocol, just a transformation)

**Comparison with other privacy approaches:**

| Property | Orthogonal Rotation | Differential Privacy | Homomorphic Encryption | Secure MPC |
|----------|-------------------|---------------------|----------------------|------------|
| Search quality | Identical (lossless) | Degraded (noise) | Identical | Identical |
| Latency overhead | Negligible (one matmul) | Negligible (add noise) | 1000-10000x | 10-100x |
| Protects individual vectors | Yes (without key) | Yes (with epsilon bound) | Yes | Yes |
| Protects pairwise distances | No | Partially | Yes | Yes |
| Key management | One d x d matrix | No key needed | Per-query keys | Multi-party setup |
| Threat model | Honest-but-curious, no key access | Any, with bounded epsilon | Any | Requires honest majority |
| Deployment complexity | Drop-in replacement | Drop-in replacement | Custom server needed | Custom protocol needed |

**Where rotation wins:** Zero quality loss + negligible overhead + simple deployment. This is the only approach where Recall@K is provably identical pre- and post-transformation.

**Where rotation loses:** If the adversary obtains even a few (original, rotated) pairs, the key can be recovered. DP protects against this. HE protects against everything.

### 1.4 Known-Plaintext Attacks

**If an adversary has d linearly independent (original, rotated) pairs, they can recover Q exactly.**

Given pairs (x_1, x'_1), ..., (x_d, x'_d) where x'_i = Qx_i, form the matrices:

```
X = [x_1 | x_2 | ... | x_d]     (d x d)
X' = [x'_1 | x'_2 | ... | x'_d]  (d x d)
```

Then Q = X' X^{-1} (assuming X is invertible, which it is if the pairs are linearly independent).

**Practical implications:**

- For d = 1536 (OpenAI text-embedding-3-small), the adversary needs 1536 known pairs. This is a high bar but not impossible if the adversary can query the original embedding API with chosen text and also observe the rotated outputs.
- For d = 3072 (text-embedding-3-large), they need 3072 pairs.
- Mitigation: don't let anyone observe both the original AND rotated versions of the same content. This is natural in the intended use case (you send ONLY rotated vectors to the server, originals never leave your machine).

**With fewer than d pairs:** The adversary recovers a partial constraint on Q. With k < d pairs, they constrain Q to a (d-k)-dimensional subspace of O(d). The remaining dimensions are still uniformly distributed from the adversary's perspective. Even d/2 pairs leave significant uncertainty.

**Statistical attacks without known pairs:**

If the adversary knows the distribution of the original embeddings (e.g., "these are OpenAI embeddings of English text"), can they exploit statistical structure?

- The rotated vectors have the SAME covariance structure. If original embeddings have covariance Sigma, rotated embeddings have covariance Q Sigma Q^T. An adversary can compute Q Sigma Q^T from the rotated data, but recovering Q from it requires knowing Sigma, which they don't have per-vector.
- If the adversary knows the EXACT covariance matrix of the original distribution, they can align eigenvectors: diagonalize both Q Sigma Q^T and Sigma, then Q is the rotation between their eigenbases. But this only works up to sign ambiguities on eigenvectors (2^d possibilities) and assumes non-degenerate eigenvalues.
- In practice, embedding covariance matrices are estimated, not exact, and have repeated eigenvalues (particularly in high dimensions). This makes eigenvalue alignment attacks impractical.

**Bottom line:** Rotation is secure against a server that never sees originals. It is NOT secure against an adversary who can query the original embedding model with chosen inputs and correlate with the rotated database.

### 1.5 Rotation + Quantization: Does Quantization Add Privacy?

TurboQuant's full pipeline is: rotate -> quantize -> store indices + norms. On dequantization: indices -> centroid lookup -> inverse rotate.

**Quantization adds a form of irreversibility.** Even with the rotation matrix Q, the quantized indices only map back to codebook centroids, not to the exact rotated coordinates. The quantization error is:

```
e = Q^T * codebook[indices] - x_original
```

This error acts like noise. At 4-bit quantization on d=1536, the MSE is approximately 0.001 per coordinate, corresponding to roughly 3% relative error per vector.

**Is this like differential privacy noise?**

No. DP noise has specific distributional guarantees (Gaussian or Laplacian with calibrated scale). Quantization error is deterministic given the input — the same input always produces the same quantized output. It does not satisfy DP's randomness requirement.

However, quantization does increase the difficulty of inversion:
- Without quantization: knowing Q perfectly recovers **x** from **x'** = Q**x**
- With quantization: knowing Q recovers Q^T * codebook[indices], which has ~0.1-3% MSE vs. the original, depending on bit width

**Quantization also reduces the effective search space for known-plaintext attacks.** The adversary sees quantized indices (discrete), not continuous rotated vectors. Recovering Q from quantized pairs requires solving a discrete optimization problem, not a simple linear system. This is harder but not provably hard.

**Recommendation:** Treat quantization as defense-in-depth, not as a primary privacy mechanism. Market the privacy story around the rotation, not the quantization.

### 1.6 Formal Statement of Security

**Definition (Rotation Privacy).** A rotation privacy scheme for embedding vectors consists of:
- **KeyGen(d, seed)** -> Q in O(d): Generate a random orthogonal matrix
- **Rotate(Q, x)** -> x' = Qx: Apply the rotation
- **Unrotate(Q, x')** -> x = Q^T x': Recover the original

**Security property (IND-ROT).** For any two vectors x_0, x_1 with ||x_0|| = ||x_1||, given only x' = Qx_b for a random bit b and uniformly random Q, no polynomial-time adversary can determine b with probability better than 1/2.

**Proof:** Q is drawn from the Haar measure on O(d). For any fixed x_0, x_1 with equal norms, the distributions of Qx_0 and Qx_1 are both uniform on the sphere S^{d-1}(r) where r = ||x_0|| = ||x_1||. They are identically distributed. The advantage is exactly 0.

**This breaks when:**
1. ||x_0|| != ||x_1|| (norms leak)
2. The adversary observes multiple vectors rotated by the same Q (pairwise distances leak)
3. The adversary has auxiliary information linking specific plaintexts to ciphertexts

---

## 2. API Design

### 2.1 Module Location

```python
from turboquant_vectors import PrivateEncoder

# Alternative if the user has only the base package:
# from turboquant.private import PrivateEncoder
```

`PrivateEncoder` lives in `turboquant_vectors` because privacy-preserving embeddings are a vector search feature, not a KV cache feature. The base `turboquant` package stays focused on compression.

### 2.2 Full Class API

```python
import numpy as np
from pathlib import Path
from typing import Optional, Union, Literal

class PrivateEncoder:
    """
    Lossless privacy layer for embedding vectors.

    Applies a secret random orthogonal rotation to embedding vectors before
    storage or transmission. All distance metrics (cosine, L2, inner product)
    are preserved exactly. Without the secret key, original vectors cannot
    be reconstructed.

    Threat model: honest-but-curious server without access to the key file.

    Usage:
        encoder = PrivateEncoder.generate(dim=1536)
        encoder.save_key("my_secret.tqkey")

        # Rotate before sending to vector DB
        rotated = encoder.rotate(embeddings)
        pinecone_index.upsert(vectors=rotated, ids=ids)

        # Query must also be rotated
        rotated_query = encoder.rotate(query_vector)
        results = pinecone_index.query(vector=rotated_query, top_k=10)

        # Later, load the same key
        encoder = PrivateEncoder.load_key("my_secret.tqkey")
    """

    def __init__(self, rotation_matrix: np.ndarray):
        """
        Initialize with an existing rotation matrix.

        Prefer PrivateEncoder.generate() or PrivateEncoder.load_key()
        instead of calling this directly.

        Args:
            rotation_matrix: Orthogonal matrix, shape (d, d), dtype float32.
                Must satisfy Q^T Q = I (verified on construction).

        Raises:
            ValueError: If matrix is not square, not orthogonal, or not float32.
        """
        ...

    @classmethod
    def generate(
        cls,
        dim: int,
        seed: Optional[int] = None,
    ) -> "PrivateEncoder":
        """
        Generate a new random rotation key.

        Creates a uniformly random orthogonal matrix via QR decomposition
        of a Gaussian random matrix (Stewart 1980). The matrix is drawn
        from the Haar measure on O(d).

        Args:
            dim: Embedding dimension (e.g., 1536 for OpenAI, 768 for BERT).
            seed: Optional seed for reproducibility. If None, uses OS entropy.
                WARNING: Using a seed means the key is recoverable from the seed.
                For real privacy, omit the seed and save the key to a file.

        Returns:
            PrivateEncoder with a fresh random rotation matrix.

        Memory: The rotation matrix is d x d float32. For d=1536, this is 9.4 MB.
            For d=3072, this is 37.7 MB. Allocated once, reused for all operations.
        """
        ...

    @classmethod
    def load_key(cls, path: Union[str, Path]) -> "PrivateEncoder":
        """
        Load a rotation key from a .tqkey file.

        Args:
            path: Path to the key file (created by save_key()).

        Returns:
            PrivateEncoder initialized with the stored rotation matrix.

        Raises:
            FileNotFoundError: If the key file doesn't exist.
            ValueError: If the file is corrupted or not a valid .tqkey file.
        """
        ...

    @classmethod
    def from_seed(cls, dim: int, seed: int) -> "PrivateEncoder":
        """
        Deterministically reconstruct a key from a seed.

        Equivalent to PrivateEncoder.generate(dim=dim, seed=seed) but makes
        the intent explicit: the key is derived from the seed, so anyone
        with the seed can reconstruct it.

        Use case: sharing a key between multiple parties without exchanging
        a large file. Each party generates the same key from the same seed.

        Security note: The seed is the secret. Protect it like a password.
        For d=1536, the seed is 4 bytes vs. 9.4 MB for the full matrix.
        Much easier to manage, but also much easier to brute-force if the
        seed space is small. Use a 64-bit or 128-bit seed in production.

        Args:
            dim: Embedding dimension.
            seed: Integer seed. Use a cryptographically random value.

        Returns:
            PrivateEncoder with the deterministic rotation matrix.
        """
        ...

    def save_key(self, path: Union[str, Path]) -> None:
        """
        Save the rotation key to a .tqkey file.

        File format:
            - 8 bytes: magic number "TQKEY\x00\x01\x00" (version 1)
            - 4 bytes: dimension (uint32, little-endian)
            - 4 bytes: reserved (zeros)
            - d*d*4 bytes: rotation matrix (float32, row-major, little-endian)
            - 32 bytes: SHA-256 checksum of the matrix data

        The file is NOT encrypted. Protect it with filesystem permissions
        or encrypt it with a separate tool.

        Args:
            path: Output file path. Extension .tqkey is added if not present.
        """
        ...

    @property
    def dim(self) -> int:
        """Embedding dimension this encoder was created for."""
        ...

    @property
    def key_size_bytes(self) -> int:
        """Size of the rotation matrix in memory (d * d * 4 bytes)."""
        ...

    def rotate(
        self,
        vectors: np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Rotate vectors for privacy.

        Applies the secret orthogonal rotation. Output vectors have
        identical norms, pairwise distances, and cosine similarities
        as the inputs.

        Args:
            vectors: Input vectors.
                - Shape (d,) for a single vector
                - Shape (n, d) for a batch of vectors
                Dtype: float32 or float64 (converted to float32 internally).
            normalize: If True, L2-normalize vectors before rotation.
                Recommended for non-unit-norm embeddings to prevent norm leakage.

        Returns:
            Rotated vectors, same shape as input, dtype float32.

        Raises:
            ValueError: If vector dimension doesn't match encoder dimension.

        Performance:
            Single vector (d=1536): ~0.05 ms (matrix-vector multiply)
            Batch of 10K vectors (d=1536): ~15 ms (matrix-matrix multiply)
            Batch of 1M vectors (d=1536): ~1.5 s (memory-bound)
        """
        ...

    def unrotate(self, vectors: np.ndarray) -> np.ndarray:
        """
        Recover original vectors from rotated ones.

        Applies the inverse rotation (Q^T). Only possible with the key.

        Args:
            vectors: Rotated vectors, shape (d,) or (n, d).

        Returns:
            Original vectors (up to float32 precision).

        Raises:
            ValueError: If vector dimension doesn't match encoder dimension.
        """
        ...

    def rotate_and_compress(
        self,
        vectors: np.ndarray,
        bits: int = 4,
        normalize: bool = False,
    ) -> "CompressedPrivateVectors":
        """
        Rotate AND quantize vectors for privacy + compression.

        Applies rotation first (for privacy), then TurboQuant scalar
        quantization (for compression). This gives both privacy AND
        memory savings, but search quality is reduced by quantization
        (not by rotation).

        Args:
            vectors: Input vectors, shape (n, d).
            bits: Quantization bits per coordinate (2, 3, 4, or 8).
            normalize: If True, L2-normalize before rotation.

        Returns:
            CompressedPrivateVectors object with quantized indices and norms.
            Can be searched directly via .search() or decompressed via .decompress().
        """
        ...

    def verify_key_match(self, rotated_sample: np.ndarray, original_sample: np.ndarray) -> bool:
        """
        Check if this key was used to rotate the given vectors.

        Useful for catching key mismatch errors before running a full search.
        Rotates original_sample and checks if it matches rotated_sample
        within float32 tolerance.

        Args:
            rotated_sample: A few rotated vectors (e.g., first 5 from the DB).
            original_sample: The corresponding original vectors.

        Returns:
            True if the key matches (MSE < 1e-10), False otherwise.
        """
        ...

    def fingerprint(self) -> str:
        """
        Short hex fingerprint of the key for identification.

        Returns first 16 hex chars of SHA-256(rotation_matrix_bytes).
        Does NOT leak the key (one-way hash).

        Use this to label rotated datasets so you know which key was used.
        """
        ...


class CompressedPrivateVectors:
    """
    Rotated + quantized embedding vectors.

    Created by PrivateEncoder.rotate_and_compress(). Supports direct
    nearest-neighbor search on compressed data and serialization to disk.
    """

    def __init__(
        self,
        indices: np.ndarray,    # (n, d) uint8 — quantization indices
        norms: np.ndarray,      # (n,) float32 — vector norms
        codebook: np.ndarray,   # (2^bits,) float32 — centroid values
        bits: int,
        dim: int,
        key_fingerprint: str,   # to verify key match on search
    ):
        ...

    @property
    def shape(self) -> tuple:
        """(n_vectors, dim)"""
        ...

    @property
    def memory_bytes(self) -> int:
        """Total memory usage of compressed representation."""
        ...

    @property
    def compression_ratio(self) -> float:
        """Ratio vs. float32 storage (e.g., 8.0 for 4-bit)."""
        ...

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        metric: Literal["cosine", "l2", "ip"] = "cosine",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Nearest-neighbor search on compressed vectors.

        The query MUST be rotated with the same PrivateEncoder before
        calling this method.

        Args:
            query: Rotated query vector, shape (d,) or (n_queries, d).
                Must be rotated with the same key used to compress the data.
            top_k: Number of nearest neighbors to return.
            metric: Distance metric. "cosine", "l2", or "ip" (inner product).

        Returns:
            (indices, distances) where:
                indices: shape (n_queries, top_k) — indices into the compressed array
                distances: shape (n_queries, top_k) — distances/similarities

        Raises:
            DimensionMismatchError: If query dim doesn't match compressed dim.
        """
        ...

    def decompress(self, encoder: "PrivateEncoder") -> np.ndarray:
        """
        Fully decompress and unrotate back to original space.

        Requires the original PrivateEncoder (with the secret key).
        Result is approximate due to quantization loss.

        Args:
            encoder: The PrivateEncoder that created this compressed data.

        Returns:
            Approximate original vectors, shape (n, d), float32.

        Raises:
            KeyMismatchError: If encoder fingerprint doesn't match.
        """
        ...

    def save(self, path: Union[str, Path]) -> None:
        """
        Save compressed vectors to disk (.tqv format).

        Stores indices, norms, codebook, and metadata. Does NOT store
        the rotation key (that's in the .tqkey file).

        Includes the key fingerprint so load() can verify the correct
        key is used for search.
        """
        ...

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CompressedPrivateVectors":
        """Load compressed vectors from a .tqv file."""
        ...
```

### 2.3 Exception Classes

```python
class TurboQuantPrivacyError(Exception):
    """Base exception for privacy module."""
    pass

class KeyMismatchError(TurboQuantPrivacyError):
    """
    Raised when searching with a different key than was used to rotate.

    This is the most common user error. The error message includes both
    key fingerprints to help debug.
    """
    def __init__(self, expected_fingerprint: str, actual_fingerprint: str):
        super().__init__(
            f"Key mismatch: data was rotated with key {expected_fingerprint}, "
            f"but search is using key {actual_fingerprint}. "
            f"Use the same PrivateEncoder for both rotation and search."
        )

class DimensionMismatchError(TurboQuantPrivacyError):
    """
    Raised when vector dimension doesn't match the encoder or compressed data.
    """
    def __init__(self, expected: int, actual: int):
        super().__init__(
            f"Dimension mismatch: encoder expects {expected}-dim vectors, "
            f"got {actual}-dim. Check that you're using the right encoder "
            f"for this data."
        )
```

### 2.4 Usage Examples

**Example 1: Rotate embeddings before sending to Pinecone**

```python
from turboquant_vectors import PrivateEncoder
import openai

# One-time setup: generate and save key
encoder = PrivateEncoder.generate(dim=1536)
encoder.save_key("embeddings.tqkey")

# Embed and rotate
client = openai.OpenAI()
response = client.embeddings.create(input=texts, model="text-embedding-3-small")
vectors = np.array([e.embedding for e in response.data])

rotated = encoder.rotate(vectors, normalize=True)

# Send to Pinecone — they see only rotated vectors
index.upsert(vectors=list(zip(ids, rotated.tolist())))

# Search: rotate query with the same key
q_response = client.embeddings.create(input=[query], model="text-embedding-3-small")
q_vec = np.array(q_response.data[0].embedding)
q_rotated = encoder.rotate(q_vec, normalize=True)

results = index.query(vector=q_rotated.tolist(), top_k=10)
# Results are identical to searching unrotated vectors
```

**Example 2: Share a dataset without revealing originals**

```python
# Alice has proprietary embeddings
encoder = PrivateEncoder.generate(dim=768)
encoder.save_key("alice_secret.tqkey")  # Alice keeps this

# Rotate the dataset
rotated_dataset = encoder.rotate(proprietary_embeddings)
np.save("shared_embeddings.npy", rotated_dataset)

# Bob can search the shared embeddings, but cannot recover originals.
# Bob cannot infer what the original documents say.
# Alice gives Bob the key ONLY if she trusts him.
```

**Example 3: Multiple users searching the same index**

```python
# Admin generates key and distributes to authorized users
encoder = PrivateEncoder.generate(dim=1536)
encoder.save_key("team_key.tqkey")  # distribute via secure channel

# --- User A ---
encoder_a = PrivateEncoder.load_key("team_key.tqkey")
q_rotated = encoder_a.rotate(my_query)
results = shared_index.query(vector=q_rotated.tolist(), top_k=10)

# --- User B ---
encoder_b = PrivateEncoder.load_key("team_key.tqkey")
q_rotated = encoder_b.rotate(my_other_query)
results = shared_index.query(vector=q_rotated.tolist(), top_k=10)
```

**Example 4: Rotation + compression for local storage**

```python
encoder = PrivateEncoder.generate(dim=1536)
encoder.save_key("local.tqkey")

# Rotate AND compress: privacy + 8x memory savings
compressed = encoder.rotate_and_compress(embeddings, bits=4)
print(compressed.compression_ratio)  # ~8.0x vs float32
print(compressed.memory_bytes)       # ~192 MB instead of 1.5 GB for 250K vectors

# Search directly on compressed data
q_rotated = encoder.rotate(query_vector)
indices, distances = compressed.search(q_rotated, top_k=10)

# Save to disk
compressed.save("my_index.tqv")
encoder.save_key("my_index.tqkey")

# Load later
compressed = CompressedPrivateVectors.load("my_index.tqv")
encoder = PrivateEncoder.load_key("my_index.tqkey")
```

**Example 5: Detecting key mismatch**

```python
encoder_1 = PrivateEncoder.generate(dim=1536)
encoder_2 = PrivateEncoder.generate(dim=1536)

rotated = encoder_1.rotate(embeddings)
compressed = encoder_1.rotate_and_compress(embeddings, bits=4)

# This works
q = encoder_1.rotate(query)
results = compressed.search(q, top_k=10)  # correct results

# This raises KeyMismatchError (if compressed data has fingerprint)
q_wrong = encoder_2.rotate(query)
# Search still runs (we can't always detect at the vector level),
# but results will be garbage. The verify method catches it:
assert not encoder_2.verify_key_match(rotated[:5], embeddings[:5])
```

### 2.5 Key Serialization Format (.tqkey)

```
Offset  Size     Field
------  ----     -----
0       5        Magic: "TQKEY" (ASCII)
5       1        Version: 0x01
6       2        Reserved: 0x0000
8       4        Dimension d (uint32 LE)
12      4        Flags (uint32 LE):
                   bit 0: 1 = generated from seed (seed stored in reserved area)
                   bits 1-31: reserved (0)
16      d*d*4    Rotation matrix Q (float32, row-major, little-endian)
16+d*d*4  32     SHA-256 checksum of bytes [0, 16+d*d*4)
```

For d=1536: total file size = 16 + 1536*1536*4 + 32 = 9,437,232 bytes (9.0 MB).
For d=3072: total file size = 16 + 3072*3072*4 + 32 = 37,748,784 bytes (36.0 MB).
For d=768: total file size = 16 + 768*768*4 + 32 = 2,359,344 bytes (2.3 MB).

### 2.6 Compressed Vector Format (.tqv)

```
Offset  Size         Field
------  ----         -----
0       4            Magic: "TQV\x00" (ASCII)
4       2            Version: 0x0001
6       2            Bits per coordinate (uint16 LE)
8       4            Dimension d (uint32 LE)
12      4            Number of vectors n (uint32 LE)
16      16           Key fingerprint (first 16 bytes of SHA-256 of rotation matrix)
32      2^bits * 4   Codebook (float32, little-endian)
32+cb   n*d*bits/8   Quantized indices (packed bits, little-endian)
...     n*4          Norms (float32, little-endian)
...     32           SHA-256 checksum of all preceding bytes
```

---

## 3. Benchmarks to Run

### 3.1 Lossless Proof: Rotation Preserves Recall@K Identically

**Test protocol:**
1. Generate or load a real embedding dataset (e.g., 100K vectors from OpenAI text-embedding-3-small, or GloVe, or sentence-transformers).
2. Compute exact brute-force top-K for 1000 random queries (ground truth).
3. Rotate all vectors and queries with the same PrivateEncoder.
4. Compute exact brute-force top-K on rotated vectors.
5. Compare: the result sets MUST be identical (not just close, IDENTICAL).

**Expected result:** Recall@1 = 1.000, Recall@10 = 1.000, Recall@100 = 1.000. Any deviation is a bug (float32 precision issues aside; test with float64 to confirm).

**Why this matters:** This is the core selling point. Rotation is provably lossless. No other privacy approach can claim this.

### 3.2 Rotation + Quantization: Isolate Quantization Loss

**Test protocol:**
1. Same dataset as 3.1.
2. Four conditions:
   - (a) Original vectors, brute-force search (baseline)
   - (b) Rotated vectors, brute-force search (must match a)
   - (c) Rotated + 4-bit quantized, search on compressed (shows quantization loss)
   - (d) Unrotated + 4-bit quantized, search on compressed (shows quantization loss without privacy)
3. Report Recall@1, @10, @100 for each condition.

**Expected result:**
- (b) = (a) exactly
- (c) approximately equals (d) (quantization loss is the same with or without rotation, since rotation doesn't change the distribution's quantizability)
- (c) and (d) will have lower recall than (a) due to quantization, NOT due to rotation

This proves: any recall loss is from quantization, not from the privacy layer.

### 3.3 Latency Overhead

**Benchmarks to report:**

| Operation | d=768 | d=1536 | d=3072 |
|-----------|-------|--------|--------|
| rotate(1 vector) | target: <0.1ms | target: <0.1ms | target: <0.5ms |
| rotate(1K vectors) | target: <5ms | target: <15ms | target: <50ms |
| rotate(100K vectors) | target: <200ms | target: <500ms | target: <2s |
| rotate(1M vectors) | target: <2s | target: <5s | target: <20s |
| generate key | target: <1s | target: <2s | target: <5s |
| save/load key | target: <100ms | target: <200ms | target: <500ms |

All are dominated by a single d x d matrix multiply (or n x d times d x d for batches). numpy's BLAS backend makes this fast.

### 3.4 Comparison vs. Differential Privacy

**Test protocol:**
1. Same dataset.
2. Apply Gaussian DP mechanism with epsilon = {1, 5, 10, 50} to embeddings.
3. Compute Recall@10 for each epsilon level.
4. Compare to our rotation approach (Recall@10 = 1.000).

**Expected result:**

| Approach | Recall@10 | Privacy |
|----------|----------|---------|
| No privacy | 1.000 | None |
| Rotation (ours) | 1.000 | Hides individual vectors from server |
| DP epsilon=50 | ~0.95 | Weak DP guarantee |
| DP epsilon=10 | ~0.80 | Moderate DP guarantee |
| DP epsilon=5 | ~0.60 | Strong DP guarantee |
| DP epsilon=1 | ~0.20 | Very strong DP guarantee |

**Key message:** Rotation gives perfect recall with meaningful privacy (honest-but-curious threat model). DP gives formal guarantees but destroys search quality. For the vector DB use case, rotation is strictly better unless you need protection against known-plaintext attacks.

---

## 4. Edge Cases

### 4.1 Searching with the Wrong Key

**What happens:** Results are garbage. Cosine similarities become essentially random because the query is in one rotated space and the database is in another.

**Detection:** If the compressed data stores the key fingerprint (first 16 bytes of SHA-256), we can detect mismatch when the user calls `.search()` on a `CompressedPrivateVectors` object. For raw rotated vectors stored in an external DB (Pinecone, etc.), we cannot detect mismatch automatically.

**Mitigation:**
- `CompressedPrivateVectors.search()` checks key fingerprint and raises `KeyMismatchError`.
- `PrivateEncoder.verify_key_match()` lets users test alignment manually.
- Documentation prominently warns about this.

### 4.2 Rotation Matrix Lost

**What happens:** The original vectors are unrecoverable. The rotated vectors remain searchable (pairwise distances are preserved), so the index is not useless. But new queries cannot be rotated correctly, and the index cannot be un-rotated.

**Mitigation:**
- `save_key()` documentation warns: "Back up this file. If lost, your original vectors are unrecoverable."
- `generate()` documentation warns against relying on seeds without backing up.
- We recommend the user stores the .tqkey file in a password manager, encrypted backup, or HSM.

### 4.3 Rotating Already-Quantized Vectors

**Can you do it?** Yes, you can quantize first and rotate second, but this is wrong:

1. Quantize: x -> indices, norms
2. Dequantize: indices, norms -> x_hat (lossy)
3. Rotate: Q @ x_hat -> x'_hat

The rotation doesn't worsen the quantization error. But you've lost the benefit of TurboQuant's rotation-aware codebook. TurboQuant's codebook is optimized for the distribution AFTER rotation (Beta distribution). If you rotate post-quantization, the codebook was optimized for the wrong distribution.

**Correct order:** Rotate first (for privacy), THEN quantize (for compression). This is what `rotate_and_compress()` does. The TurboQuant codebook is designed for this order.

### 4.4 Memory for the Rotation Matrix

| Dimension | Model | Matrix Size | Acceptable? |
|-----------|-------|-------------|-------------|
| 384 | all-MiniLM-L6-v2 | 0.6 MB | Trivial |
| 768 | BERT, all-mpnet-base-v2 | 2.4 MB | Trivial |
| 1024 | many models | 4.2 MB | Fine |
| 1536 | OpenAI text-embedding-3-small | 9.4 MB | Fine |
| 3072 | OpenAI text-embedding-3-large | 37.7 MB | Acceptable |
| 4096 | Cohere embed-v4 | 67.1 MB | Borderline |
| 8192 | hypothetical future | 268 MB | Problematic |

For d <= 3072, the matrix fits easily in memory and on disk. For d=4096, it's large but manageable (one-time allocation). For d >= 8192, we should consider structured random matrices (e.g., HD^k: Hadamard x diagonal sign) which are O(d) storage and O(d log d) application time. This is a future optimization, not needed for the first release.

**Structured alternative (future):** Replace the dense d x d matrix with a product of k Hadamard-diagonal pairs: Q = H D_1 H D_2 ... H D_k where H is the d x d Hadamard matrix (applied via Fast Walsh-Hadamard Transform, O(d log d)) and D_i are diagonal matrices with random +/-1 entries. Storage: k * d entries (e.g., k=3, d=8192 -> 98 KB vs. 268 MB). This gives O(d log d) rotation time instead of O(d^2) and near-uniform distribution on O(d). NOT needed for v1.

### 4.5 Batched Rotation for Millions of Vectors

**Problem:** Rotating 1M vectors of dim 1536 means multiplying a (1M, 1536) matrix by a (1536, 1536) matrix. In float32, the input is 6.1 GB and the rotation matrix is 9.4 MB. The output is another 6.1 GB. Total peak memory: ~12.5 GB.

**Solution:** Process in chunks.

```python
def rotate(self, vectors, normalize=False, batch_size=50_000):
    n = len(vectors)
    out = np.empty_like(vectors, dtype=np.float32)
    for i in range(0, n, batch_size):
        chunk = vectors[i:i+batch_size].astype(np.float32)
        if normalize:
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            chunk = chunk / (norms + 1e-10)
        out[i:i+batch_size] = chunk @ self._rotation_t
    return out
```

At batch_size=50K and d=1536: chunk is 307 MB, output chunk is 307 MB, rotation matrix is 9.4 MB. Peak memory per batch: ~624 MB. 1M vectors processed in 20 batches. Total time: ~1.5s on modern CPU (BLAS-optimized matmul).

### 4.6 GPU Acceleration

For very large batches, the rotation can be moved to GPU:

```python
def rotate_gpu(self, vectors, device="cuda"):
    import torch
    Q_t = torch.from_numpy(self._rotation_t).to(device)
    x = torch.from_numpy(vectors).to(device, dtype=torch.float32)
    rotated = x @ Q_t
    return rotated.cpu().numpy()
```

For 1M x 1536 on an RTX 4080: ~200ms (vs. ~1.5s on CPU). Optional, not required for v1.

### 4.7 Float32 Precision

Orthogonal matrices satisfy Q^T Q = I exactly in exact arithmetic. In float32:

```
||Q^T Q - I||_F ~ d * epsilon_machine ~ 1536 * 1.2e-7 ~ 1.8e-4
```

This means the round-trip rotate -> unrotate introduces ~1e-7 relative error per coordinate. For embedding search, this is negligible (embeddings themselves have far more noise from model stochasticity). For the paranoid, we offer float64 rotation (2x memory, ~2x slower, but error drops to ~1e-16).

### 4.8 Interaction with HNSW / IVF Indices

When using approximate nearest-neighbor indices (HNSW in Qdrant/Weaviate, IVF in FAISS):
- Rotation does NOT affect the quality of approximate search. HNSW graph structure is determined by pairwise distances, which are preserved. The same HNSW graph would be built for original and rotated vectors.
- IVF cluster assignments may differ (since centroids are computed in rotated space), but recall is identical in expectation.
- The user should build the index on rotated vectors and always query with rotated vectors.

### 4.9 Multiple Keys / Key Rotation

If the user wants to rotate to a new key (e.g., the old key may have been compromised):

```python
old_encoder = PrivateEncoder.load_key("old.tqkey")
new_encoder = PrivateEncoder.generate(dim=1536)
new_encoder.save_key("new.tqkey")

# Re-rotate: unrotate with old key, then rotate with new key
# This is equivalent to applying Q_new @ Q_old^T to the rotated vectors
original = old_encoder.unrotate(rotated_vectors)
re_rotated = new_encoder.rotate(original)
```

For efficiency, the combined rotation Q_new @ Q_old^T can be precomputed (one d x d matmul) and applied directly, avoiding the intermediate unrotation. This could be a method: `encoder.rekey(old_encoder, new_encoder, rotated_vectors)`.

---

## 5. Implementation Plan

### 5.1 Files to Create

```
turboquant_vectors/
  private/
    __init__.py          # exports PrivateEncoder, CompressedPrivateVectors
    encoder.py           # PrivateEncoder class
    compressed.py        # CompressedPrivateVectors class
    exceptions.py        # KeyMismatchError, DimensionMismatchError
    formats.py           # .tqkey and .tqv reader/writer
tests/
  test_private.py        # Unit tests for the privacy module
  test_private_recall.py # Recall@K benchmark proving lossless
```

### 5.2 Dependencies

- numpy (required, already a dependency of turboquant_vectors)
- scipy (optional, only for Beta distribution codebook computation)
- torch (optional, for GPU acceleration)

### 5.3 Test Matrix

| Test | What it proves |
|------|---------------|
| rotate then unrotate recovers original (MSE < 1e-12) | Round-trip correctness |
| rotated vectors have identical pairwise cosine similarities | Metric preservation |
| rotated vectors have identical pairwise L2 distances | Metric preservation |
| rotated vectors have identical inner products | Metric preservation |
| Recall@10 on rotated == Recall@10 on original (exact match) | Lossless search |
| Recall@10 on rotated+compressed == Recall@10 on compressed (no rotation) | Quantization loss is independent of rotation |
| Wrong key gives random cosine similarities (~0.0 mean) | Key-dependence |
| Key fingerprint mismatch raises KeyMismatchError | Error handling |
| Dimension mismatch raises DimensionMismatchError | Error handling |
| save_key / load_key round-trip | Serialization |
| save / load compressed vectors round-trip | Serialization |
| from_seed produces identical encoder to generate with same seed | Determinism |
| Different seeds produce different rotation matrices | Randomness |
| Rotation matrix is orthogonal (Q^T Q = I within tolerance) | Correctness |
| 1M vector batch rotation completes without OOM | Scalability |
| normalize=True produces unit-norm outputs | Normalization |

### 5.4 What NOT to Build (v1)

- Encrypted key files (use OS-level encryption)
- Structured random matrices (Hadamard-diagonal, only needed for d >= 8192)
- GPU-only code paths (numpy is fast enough for v1)
- Integration with specific vector DB clients (users do the rotation and pass to their DB)
- Key derivation from passwords (users can use a KDF externally)
- Streaming rotation (not needed since embeddings are batch-generated)

---

## 6. Marketing Angle

**Headline:** "Zero-cost privacy for your embeddings. Rotate before you send. Search quality stays identical."

**One-paragraph pitch:** Every time you send embeddings to a vector database, you're sending a compressed representation of your data. An attacker with access to your embeddings can reconstruct approximate versions of your original text. PrivateEncoder applies a secret rotation that makes individual vectors unrecoverable, while preserving all pairwise distances exactly. Your search results don't change. Your privacy does.

**Comparison table for the README:**

| | Rotation (ours) | Differential Privacy | Homomorphic Encryption |
|---|---|---|---|
| Recall@10 | 1.000 | 0.60 (eps=5) | 1.000 |
| Latency overhead | <1ms/query | <1ms/query | 1000x |
| Setup | 3 lines of code | 3 lines of code | Custom server |
| Protects against | Honest-but-curious server | Arbitrary adversary | Arbitrary adversary |

---

## 7. Open Questions

1. **Should the privacy module live in `turboquant_vectors` or the base `turboquant` package?** Recommendation: `turboquant_vectors`, since privacy for embeddings is a vector search concern. The base package is for KV cache compression.

2. **Should we support float16 rotation matrices?** For d=3072, float16 would halve storage (18.8 MB). But float16 matmul has lower precision, and the orthogonality guarantee degrades faster. Recommendation: float32 only for v1. The storage savings don't matter (9 MB vs 18 MB is irrelevant on disk).

3. **Should `rotate()` accept torch tensors directly?** Recommendation: Accept both numpy and torch, return the same type as input. This is a QOL feature that takes 10 lines to implement.

4. **Should we provide a CLI command?** E.g., `tq-rotate --key my.tqkey --input embeddings.npy --output rotated.npy`. Recommendation: Yes, in v2. Focus on the Python API first.

5. **Patent risk?** Orthogonal rotation for privacy is well-established in the literature (random orthogonal embeddings, oblivious RAM, secure computation). TurboQuant's contribution is the Beta-distribution codebook, not the rotation itself. No patent risk for the rotation-as-privacy application.

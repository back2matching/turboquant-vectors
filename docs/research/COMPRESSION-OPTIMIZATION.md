# Compression Optimization Research

> Research conducted 2026-03-25 by specialized agent. Findings inform EXECPLAN.md.

---

## 1. CRITICAL BUG: 3-bit Codebook is Wrong

**File:** `turboquant_vectors/_rotation.py`, line 93

The 3-bit Lloyd-Max codebook is **completely wrong**. It uses the inner 4 values from the 4-bit codebook instead of the correct 3-bit optimal centroids for N(0,1).

**Current (WRONG):**
```python
elif bits == 3:
    lloyd = [0.1284, 0.3882, 0.6568, 0.9423]
```

**Correct Lloyd-Max for N(0,1) with 8 levels:**
```python
elif bits == 3:
    lloyd = [0.2451, 0.7560, 1.3439, 2.1519]
```

The correct values are the positive centroids of the Lloyd-Max quantizer for N(0,1): `[-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519]`. These are confirmed by multiple independent sources including the `komm` library's LloydMaxQuantizer reference and my own numerical computation via scipy's `integrate.quad`.

**Impact:** The MSE with the wrong codebook is **627% worse** at d=1536 and **406% worse** at d=128 compared to the correct values. This means 3-bit compression quality is dramatically degraded -- likely losing several percentage points of recall. Fixing this single bug could yield the largest improvement to compression quality in the entire codebase.

The 1-bit, 2-bit, and 4-bit codebooks are all correct. The 5-8 bit uniform codebooks are suboptimal but functional.

**Paper details:** The TurboQuant paper (arXiv:2504.19874) does not publish explicit centroid tables. It states codebooks are "precomputed and stored...for a range of practically relevant bit-widths" by solving equation (4) via Lloyd-Max iteration on the coordinate distribution. The distribution is `f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)`, which is Beta((d-1)/2, (d-1)/2) on [-1,1], converging to N(0, 1/d) for large d. The Gaussian approximation is valid for all practical dimensions (d >= 64), so computing centroids for N(0,1) and scaling by `1/sqrt(d)` is correct.

**Recommended fix for `_rotation.py`:**
```python
elif bits == 3:
    lloyd = [0.2451, 0.7560, 1.3439, 2.1519]
```

Additionally, the 5-8 bit codebooks should use Lloyd-Max centroids instead of uniform spacing for better quality. Precomputed values for 5-bit through 8-bit can be generated once via the same Lloyd-Max iteration.

---

## 2. Competing Quantization Approaches

### SAQ (SIGMOD 2026, arXiv:2509.12086)
- **What it does:** PCA-projects vectors, then partitions dimensions into segments by variance. High-variance dimensions get more bits via dynamic programming optimization.
- **Claims:** 80% less quantization error than Extended-RaBitQ, 80x faster encoding than RaBitQ.
- **Data-dependent:** YES -- requires PCA + variance statistics from training data. Cannot be borrowed as-is for a data-oblivious method.
- **Borrowable technique:** The "code adjustment" step -- an iterative coordinate-descent refinement of quantized codes after initial assignment. This could be applied data-obliviously: after initial scalar quantization, adjust each coordinate's code to minimize global reconstruction error. This would be a post-quantization refinement pass.
- Source: [SAQ paper](https://arxiv.org/abs/2509.12086), [GitHub](https://github.com/howarlii/SAQ)

### RaBitQ / Extended-RaBitQ (SIGMOD 2024/2025)
- **Core innovation:** Randomized 1-bit quantization (RaBitQ) with theoretical error bounds. Extended to 2-8 bits per dimension (Extended-RaBitQ) with arbitrary compression rates.
- **Approach:** Uses random rotation (like TurboQuant) but with a different quantization strategy -- scalar quantization with learned codebooks and statistical distance estimators using Chebyshev inequality.
- **Implementation:** C++ with AVX-512 SIMD. Supports 3,4,5,7,8,9 bits.
- **Key differentiator from TurboQuant:** RaBitQ uses statistical bounds on distance estimation error, enabling asymptotically optimal ANNS guarantees. TurboQuant optimizes MSE directly.
- Source: [RaBitQ GitHub](https://github.com/gaoj0017/RaBitQ), [Extended-RaBitQ GitHub](https://github.com/VectorDB-NTU/Extended-RaBitQ)

### QJL (Quantized Johnson-Lindenstrauss, AAAI 2025)
- **What it does:** Dimensionality reduction via JL transform, then quantizes each dimension to a single sign bit (+1/-1). Used as a 1-bit residual compressor in the TurboQuant framework.
- **Relevance:** Could be added as an optional residual compression step on top of TurboQuant's scalar quantization, further reducing error at minimal storage cost.

### PolarQuant (AISTATS 2026)
- **What it does:** Converts coordinate pairs to polar form (radius + angle). After random rotation, the angular distribution is predictable, eliminating per-block normalization overhead.
- **Relevance:** Could replace the current norm-storage mechanism. Instead of storing `norms` separately, PolarQuant encodes radius implicitly in the polar representation, saving 1-2 bits per number that currently go to normalization constants.
- Source: [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

### Individualized Non-Uniform Quantization (arXiv:2509.18471)
- Per-vector non-uniform quantization that adapts codebook to each vector's distribution. Data-dependent but offers insights for adaptive codebook selection.

---

## 3. Performance Optimization for numpy Implementation

### 3a. Eliminate the Python loop in quantization (HIGH IMPACT)

The current quantization in `core.py` lines 132-138 uses a batched loop:
```python
for start in range(0, n, batch_size):
    ...
    dists = np.abs(batch[:, :, np.newaxis] - self.codebook[np.newaxis, np.newaxis, :])
    indices[start:end] = dists.argmin(axis=2).astype(np.uint8)
```

This can be replaced with `np.searchsorted` since the codebook is sorted and the optimal assignment for sorted centroids is determined by thresholds (midpoints between consecutive centroids):

```python
# Precompute thresholds (midpoints between consecutive centroids)
thresholds = (self.codebook[:-1] + self.codebook[1:]) / 2
# Vectorized quantization -- no loop, no 3D broadcast
indices = np.searchsorted(thresholds, rotated).astype(np.uint8)
```

This eliminates the O(n * d * 2^bits) distance computation and replaces it with O(n * d * log(2^bits)) = O(n * d * bits) binary searches. For 4-bit (16 centroids), this is roughly 4x faster and uses far less memory (no 3D tensor allocation).

### 3b. Bit-packing for actual storage savings (MEDIUM IMPACT)

Currently indices are stored as `uint8` regardless of bit width. For 4-bit, this wastes 50% of storage. numpy's `packbits` works for 1-bit but not for arbitrary bit widths. Implementation approach:

```python
def pack_indices(indices, bits):
    """Pack b-bit indices into uint8 array."""
    n, d = indices.shape
    flat = indices.ravel()
    if bits == 4:
        # Two 4-bit values per byte
        paired = flat.reshape(-1, 2)
        packed = (paired[:, 0] << 4) | paired[:, 1]
        return packed.astype(np.uint8)
    elif bits == 2:
        # Four 2-bit values per byte
        grouped = flat.reshape(-1, 4)
        packed = (grouped[:, 0] << 6) | (grouped[:, 1] << 4) | (grouped[:, 2] << 2) | grouped[:, 3]
        return packed.astype(np.uint8)
    elif bits == 1:
        return np.packbits(flat)
    # General case using bit shifting
    ...
```

For 4-bit, this cuts index storage in half. The `packed_memory_bytes` property already reports the theoretical packed size, but actual storage doesn't achieve it.

### 3c. GPU acceleration via CuPy (MEDIUM IMPACT, optional dependency)

CuPy is a drop-in replacement for numpy on NVIDIA GPUs. The key operations (matmul for rotation, searchsorted for quantization) map directly:

```python
try:
    import cupy as cp
    xp = cp  # Use GPU
except ImportError:
    xp = np  # Fallback to CPU

rotated = xp.asarray(unit_vectors) @ xp.asarray(self.rotation_t)
indices = xp.searchsorted(thresholds, rotated).astype(xp.uint8)
```

Expected speedup: 10-50x for large batches (10K+ vectors at d=1536). The rotation matmul is the bottleneck and maps perfectly to cuBLAS. This should be an optional dependency (`pip install turboquant-vectors[gpu]`).

### 3d. Memory-mapped quantization (LOW IMPACT for now)

For datasets that don't fit in RAM, use numpy memmap:
```python
indices_mmap = np.memmap('indices.bin', dtype=np.uint8, mode='w+', shape=(n, dim))
for start in range(0, n, batch_size):
    indices_mmap[start:end] = quantize_batch(vectors[start:end])
```

This is mainly useful at 1M+ scale. The current batching in the compress loop already manages memory for the 3D distance array.

---

## 4. Structured Rotations for d > 4096

### Randomized Hadamard Transform (HIGH IMPACT for large d)

The current rotation is a dense d x d matrix requiring O(d^2) storage and O(d^2) per-vector multiply. For d=4096, that's 67 MB for the matrix and significant compute.

A **randomized Hadamard rotation** provides equivalent statistical properties in O(d log d) time and O(d) storage:

```python
def randomized_hadamard_rotate(x, signs, d):
    """Apply D * H * D where D = diag(signs), H = Hadamard matrix.
    Uses fast Walsh-Hadamard transform: O(d log d) instead of O(d^2)."""
    # Step 1: Random sign flip (O(d))
    y = x * signs
    # Step 2: Fast Walsh-Hadamard transform (O(d log d))
    y = fwht(y) / np.sqrt(d)
    # Step 3: Another random sign flip (O(d))
    y = y * signs2
    return y
```

Available implementations:
- **FFHT** (C with Python bindings): [github.com/FALCONN-LIB/FFHT](https://github.com/FALCONN-LIB/FFHT) -- heavily optimized, works with numpy arrays
- **hadamard-transform** (PyPI): `pip install hadamard-transform` -- PyTorch-based with randomized variant
- **Pure numpy:** Recursive butterfly implementation in ~20 lines

**Caveat:** Requires d to be a power of 2. For d=1536, pad to 2048. The TurboQuant paper mentions this as future work. QuaRot (LLM quantization) already uses this approach successfully.

**Recommendation:** Add as optional fast path when d > 1024. Fall back to dense QR rotation for smaller d or non-power-of-2 dimensions. Store only the random sign vectors (2 * d bytes) instead of the d x d matrix (d^2 * 4 bytes).

### Block-diagonal rotations

For non-power-of-2 dimensions, use block-diagonal orthogonal matrices where each block is a small (e.g., 256x256) Hadamard rotation. Storage is O(d * block_size) and compute is O(d * block_size). Less random than full rotation but adequate for quantization purposes.

---

## 5. Product Quantization Hybrid

### TurboQuant + Sub-vector Quantization

The idea: divide the d-dimensional rotated vector into m sub-vectors of d/m dimensions each. Apply TurboQuant's scalar quantization within each sub-vector, but also learn per-subspace codebooks.

**However, this undermines TurboQuant's key advantage** (data-oblivious, no training). A simpler hybrid:

1. Apply TurboQuant rotation (makes coordinates near-independent)
2. Apply scalar quantization at b bits per coordinate (current approach)
3. Optionally apply a residual PQ step: quantize the residual (original - reconstructed) using a small product quantizer

This "TurboQuant + Residual PQ" approach could gain +1-2pp recall at the cost of a small training step on the residual. The first two steps remain data-oblivious.

### Interaction with Privacy Rotation

The privacy rotation (PrivateEncoder) and compression rotation (TurboQuantVectors) currently use **different** rotation matrices from different seeds. In `rotate_and_compress()`, these compose to `Q_compress @ Q_privacy^T`, which is itself a random orthogonal matrix. This is fine mathematically but wastes one matmul.

**Optimization:** Use a single rotation that serves both purposes. The privacy rotation already makes coordinates near-independent (it's from the Haar measure), so it's already a valid TurboQuant rotation. The `rotate_and_compress` pipeline could skip the second rotation entirely.

---

## Ranked Recommendations by Impact

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| **1** | Fix 3-bit codebook bug | 1 line change | ~6x MSE reduction at 3-bit |
| **2** | Replace argmin loop with `searchsorted` | ~10 lines | 3-5x quantization speedup |
| **3** | Compute and hardcode Lloyd-Max for 5-8 bit | ~20 lines | Better quality at 5-8 bit |
| **4** | Add bit-packing for actual storage savings | ~50 lines | 2x storage reduction at 4-bit |
| **5** | Add Hadamard fast rotation for d > 1024 | ~100 lines + optional dep | O(d log d) vs O(d^2), huge for d=3072+ |
| **6** | Add CuPy GPU backend | ~50 lines + optional dep | 10-50x speedup with GPU |
| **7** | Merge privacy + compression rotation | ~30 lines | 2x speedup for rotate_and_compress |
| **8** | Add residual QJL 1-bit layer | ~100 lines | +0.5-1pp recall, matching full TurboQuant paper |
| **9** | PolarQuant integration | ~200 lines | Eliminates norm storage overhead |
| **10** | Streaming/memmap for million-scale | ~100 lines | Enables 1M+ vector datasets |

## Sources

- [TurboQuant paper (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)
- [TurboQuant HTML version](https://arxiv.org/html/2504.19874)
- [SAQ paper (arXiv:2509.12086)](https://arxiv.org/abs/2509.12086)
- [SAQ GitHub](https://github.com/howarlii/SAQ)
- [Extended-RaBitQ GitHub](https://github.com/VectorDB-NTU/Extended-RaBitQ)
- [RaBitQ paper (arXiv:2405.12497)](https://arxiv.org/abs/2405.12497)
- [Google Research TurboQuant blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [turboquant-pytorch implementation](https://github.com/tonbistudio/turboquant-pytorch)
- [FFHT (Fast Hadamard Transform)](https://github.com/FALCONN-LIB/FFHT)
- [hadamard-transform PyPI](https://pypi.org/project/hadamard-transform/)
- [komm LloydMaxQuantizer reference](https://komm.dev/ref/LloydMaxQuantizer/)
- [Lloyd-Max quantizer Python implementation](https://github.com/ninfueng/lloyd-max-quantizer)
- [numpy.packbits documentation](https://numpy.org/doc/stable/reference/generated/numpy.packbits.html)
- [QuaRot: Hadamard rotations in LLM quantization](https://www.emergentmind.com/topics/quarot)
- [Weaviate 8-bit rotational quantization blog](https://weaviate.io/blog/8-bit-rotational-quantization)
- [Extended-RaBitQ DEV Community article](https://dev.to/gaoj0017/extended-rabitq-an-optimized-scalar-quantization-method-83m)

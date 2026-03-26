# turboquant-vectors v1.0: Architecture Blueprint

> Research conducted 2026-03-25 by specialized agent. Findings inform EXECPLAN.md.

---

## 1. Patterns and Conventions Found

### Current Architecture Inventory

**`/turboquant_vectors/core.py`**
- `TurboQuantVectors.__init__` (line 80): uses `np.random.RandomState(seed)` — the legacy Generator API
- `_rotation.py:generate_rotation_matrix` (line 31): uses `np.random.default_rng(seed)` — the modern Generator API
- These two paths produce different rotation matrices for identical integer seeds. There is no cross-module seed contract.

**`/turboquant_vectors/core.py` lines 224-234**: `decompress()` and `search()` convenience functions instantiate a `TurboQuantVectors` with `seed=42` by default. The rotation matrix burned into that instance is never applied during decompression — `TurboQuantVectors.decompress()` reads `compressed.rotation` directly from the `CompressedVectors` dataclass. The phantom seed=42 instance is pure overhead and misleads readers into thinking the rotation is being reconstructed from scratch.

**`/turboquant_vectors/private.py` line 582-594**: `CompressedPrivateVectors.search()` supports `metric="ip"` and `metric="l2"`. The decompressed vectors at that point are the product of: original vectors, divided by their norms, rotated, quantized, then multiplied back by the original norms. The norms are pre-rotation, so the decompressed vectors do not live in a coherent space for raw IP or L2 comparison against an unrotated query. The `cosine` branch is also wrong for IP queries. The EXECPLAN.md decision log (line 105) has already correctly decided: cosine-only.

**`/turboquant_vectors/private.py` line 518**: `_decompressed_cache = None` is mutable state on a data object. If a caller modifies `norms` or `indices` after construction, the cache is silently stale.

**`/turboquant_vectors/private.py` line 548**: `_decompress()` multiplies codebook values by `original_norms`, not by rotated norms. This is intentional and documented but creates a subtle correctness requirement: the search query must also be scaled appropriately. Currently the search method does not normalize the decompressed vectors before cosine comparison, but it does normalize the query. This works, but it means cosine search on `CompressedPrivateVectors` compares unit-query against norm-scaled database. That is numerically correct for cosine (cosine is scale-invariant), but the comment on line 578 that normalizes the decompressed data should be computing norms of those norm-scaled vectors, not of unit vectors. It does — but this is fragile.

**No `py.typed` marker exists.** `pyproject.toml` has no `[tool.mypy]` section and no `py.typed`. The package ships no type stubs.

**CLI (`cli.py`)**: Three commands, all compression-only. `keygen`, `rotate`, `keyinfo`, `verify` do not exist. The integration guide (`PRIVATE-ENCODER-INTEGRATIONS.md`) shows the rotation pattern is the primary use case for most users, but there is no CLI path for it.

**`dataclass` use**: `CompressedVectors` in `core.py` is a `@dataclass`. `CompressedPrivateVectors` in `private.py` is a plain class with `__init__`. No shared base class, no shared interface.

**Test coverage gap**: `test_cli.py` has no test for nonexistent privacy commands. `test_private.py` tests `rekey_vectors` and `verify_canary` but does not test the `normalize` parameter interaction with `rotate_and_compress`.

---

## 2. Architecture Decision

### Rejected: Merge PrivateEncoder and TurboQuantVectors into one class

FAISS, Annoy, and ScaNN all separate the index structure from the transformation pipeline. FAISS uses `faiss.IndexIVFPQ` for data plus `faiss.VectorTransform` subclasses (`PCAMatrix`, `OPQMatrix`) for preprocessing — they compose rather than merge. The equivalent here is keeping `PrivateEncoder` as the transformation and `CompressedVectors`/`CompressedPrivateVectors` as the index. Merging them would break the rotation-only use case (Pinecone users who rotate but do not compress) and would make the class too large to reason about.

### Chosen Approach: Layered composition with a shared Protocol

Define a `VectorIndex` Protocol that both `CompressedVectors` and `CompressedPrivateVectors` satisfy. Fix the four concrete bugs. Add streaming via memory-mapped numpy. Expand the CLI. Add `py.typed`. The class hierarchy stays flat.

**Trade-off accepted**: The API surface grows slightly (new `StreamingCompressor`), but no existing call sites break. The `from turboquant_vectors import compress, search` one-liner API remains identical.

---

## 3. Bug Fixes Required Before API Expansion

These must land before v1.0 API discussion is moot.

### Bug 1: RNG incompatibility (`core.py:90` vs `_rotation.py:31`)

`TurboQuantVectors.__init__` uses `np.random.RandomState`. `generate_rotation_matrix` uses `np.random.default_rng`. The integer seed `42` produces different matrices in these two generators. The EXECPLAN decision (line 105) says "keep legacy RNG in core.py to avoid breaking existing compressed indexes" — this is correct for backward compatibility with serialized `.npz` files that store the rotation matrix directly. The fix is not to change the RNG but to add a comment and a guard:

```python
# core.py line 90 — MUST remain RandomState, not default_rng.
# CompressedVectors stores the rotation matrix in the .npz file,
# so the seed only matters at compress time; loaded indexes are not affected.
# DO NOT change to default_rng without a migration plan.
rng = np.random.RandomState(seed)
```

**Action for v1.0**: Add the comment now. In v1.0, when `CompressedVectors` is rebuilt on top of the Protocol, the rotation matrix is always loaded from the file and the seed is deprecated. The `seed` parameter to `TurboQuantVectors.__init__` should emit a `DeprecationWarning` in v1.0 (it is a legacy API; callers should use `PrivateEncoder.generate()` for rotation and save the key explicitly).

### Bug 2: Dead instances in convenience functions (`core.py:226, 233`)

```python
# current (broken)
def decompress(compressed: CompressedVectors) -> np.ndarray:
    tq = TurboQuantVectors(dim=compressed.dim, bits=compressed.bits)  # rotation never used
    return tq.decompress(compressed)
```

The fix is to move the decompress/search logic onto `CompressedVectors` itself (it already has all data it needs), matching the pattern `CompressedPrivateVectors` already uses:

```python
# fixed
def decompress(compressed: CompressedVectors) -> np.ndarray:
    y_hat = compressed.codebook[compressed.indices]
    x_hat = (y_hat @ compressed.rotation) * compressed.norms[:, np.newaxis]
    return x_hat

def search(compressed: CompressedVectors, query: np.ndarray, top_k: int = 10):
    return compressed.search(query, top_k=top_k)
```

### Bug 3: `CompressedPrivateVectors.search()` metric confusion (`private.py:551`)

Remove `metric="ip"` and `metric="l2"` entirely. Replace them with a single cosine implementation and a `ValueError` for other metric strings:

```python
def search(self, query: np.ndarray, top_k: int = 10) -> tuple:
    """
    Cosine nearest-neighbor search on compressed rotated vectors.

    Query MUST be rotated with the same PrivateEncoder used to create
    this object. IP and L2 metrics are not supported: decompressed vectors
    are in rotated space with original norms applied, so only cosine
    similarity is metric-correct.
    """
```

### Bug 4: Stale `_decompressed_cache` (`private.py:518`)

The cache is on a mutable `__init__` attribute. Two fixes:
1. Make `indices`, `norms`, `codebook` read-only via `@property` backed by private attributes.
2. Or use `__slots__` and `__setattr__` guard.

The minimal-disruption fix for v0.3.1 is to rename the cache attribute to `__decompressed_cache` (name-mangling prevents accidental external write) and add a `cache_clear()` method for users who genuinely want to free the RAM.

---

## 4. v1.0 Component Design

### 4.1 File Structure

```
turboquant_vectors/
  __init__.py          — (modify) export Protocol, StreamingCompressor
  _rotation.py         — (no change needed)
  private.py           — (modify) fix bugs 3+4, add Protocol compliance
  core.py              — (modify) fix bugs 1+2, add search() to CompressedVectors
  compress_stream.py   — (create) StreamingCompressor
  _types.py            — (create) VectorIndex Protocol, DimError, KeyMismatchError
  cli.py               — (modify) add keygen, rotate, keyinfo, verify commands
  py.typed             — (create) empty marker file

tests/
  test_streaming.py    — (create)
  test_types.py        — (create) Protocol compliance
  test_cli_privacy.py  — (create)
```

### 4.2 `_types.py` — Shared Protocol and Exceptions

**File**: `/turboquant_vectors/_types.py`

This is the foundation for type safety and the plugin architecture.

```python
from typing import Protocol, runtime_checkable, Tuple
import numpy as np

class DimensionError(ValueError):
    """Raised when input vector dimension doesn't match the encoder's dimension."""
    def __init__(self, expected: int, got_shape: tuple):
        super().__init__(
            f"Dimension mismatch: encoder expects dim={expected}, "
            f"got array shape {got_shape}. "
            f"Did you use the right encoder? Check encoder.dim."
        )
        self.expected = expected
        self.got_shape = got_shape

class KeyMismatchError(ValueError):
    """Raised when a PrivateEncoder's key doesn't match a CompressedPrivateVectors object."""
    def __init__(self, index_fp: str, encoder_fp: str):
        super().__init__(
            f"Key fingerprint mismatch: index was created with key={index_fp!r}, "
            f"but encoder has key={encoder_fp!r}. Load the correct .tqkey file."
        )

@runtime_checkable
class VectorIndex(Protocol):
    """Protocol satisfied by CompressedVectors and CompressedPrivateVectors."""
    n_vectors: int
    dim: int
    bits: int
    compression_ratio: float

    def search(self, query: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def save(self, path) -> None:
        ...
```

`DimensionError` replaces the generic `ValueError(f"Expected vectors with dim=...")` in both `rotate()` and `compress()`. The structured exception means callers can catch specifically and the message always includes the remedy.

`KeyMismatchError` enables automatic key verification. When `CompressedPrivateVectors.search()` is called, it should optionally accept an `encoder` argument and verify `encoder.fingerprint() == self.key_fingerprint` immediately, raising `KeyMismatchError` before any computation.

### 4.3 `core.py` — `CompressedVectors` gains `search()`

Add search directly to the dataclass so it no longer depends on `TurboQuantVectors`:

```python
@dataclass
class CompressedVectors:
    # ... existing fields ...

    def decompress(self) -> np.ndarray:
        """Decompress to float32. Inverse rotation is applied using stored matrix."""
        y_hat = self.codebook[self.indices.astype(np.int32)]
        x_hat = (y_hat @ self.rotation) * self.norms[:, np.newaxis]
        return x_hat

    def search(self, query: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Cosine nearest-neighbor search. Query should be in the same space as the original vectors."""
        # Implementation uses decompress_normalized cache, same as TurboQuantVectors.search
        ...
```

The `TurboQuantVectors` class itself is not removed in v1.0 — it exists for users who want stateful control of compression parameters. But the convenience `decompress()` and `search()` module-level functions are rewritten to call methods on `CompressedVectors` directly.

### 4.4 `compress_stream.py` — Streaming API

**File**: `/turboquant_vectors/compress_stream.py`

The streaming design uses memory-mapped numpy arrays as the output buffer. This means the full index is never in RAM simultaneously. Input can be a file path (mmap read), a generator of batches, or any iterable of arrays.

```python
import numpy as np
from pathlib import Path
from typing import Iterator, Union
from turboquant_vectors._rotation import compute_codebook
from turboquant_vectors.private import PrivateEncoder

class StreamingCompressor:
    """
    Compress very large embedding collections that don't fit in RAM.

    Uses memory-mapped output so only one batch is in RAM at a time.
    Input can be a .npy file path, a generator of batches, or a numpy mmap.

    Example — compress a 10M vector file:
        sc = StreamingCompressor(encoder, bits=4, batch_size=50_000)
        sc.compress_file("embeddings.npy", "embeddings.tqv.npz")

    Example — streaming from a generator:
        def batches():
            for chunk in db.iter_embeddings(batch_size=10000):
                yield np.array(chunk, dtype=np.float32)

        sc.compress_stream(batches(), n_total=1_000_000, output="out.tqv.npz")
    """

    def __init__(
        self,
        encoder: PrivateEncoder,
        bits: int = 4,
        batch_size: int = 50_000,
    ):
        self.encoder = encoder
        self.bits = bits
        self.batch_size = batch_size
        self._codebook = compute_codebook(encoder.dim, bits)
        self._n_centroids = 2 ** bits

    def compress_file(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        """
        Compress a .npy file using memory mapping.
        Peak RAM = one batch of float32 + one batch of uint8 indices.
        """
        input_path = Path(input_path)
        vectors = np.load(input_path, mmap_mode='r')  # read-only mmap
        n, dim = vectors.shape
        if dim != self.encoder.dim:
            raise DimensionError(self.encoder.dim, vectors.shape)

        self._compress_mmap(vectors, n, dim, output_path)

    def compress_stream(
        self,
        batches: Iterator[np.ndarray],
        n_total: int,
        output_path: Union[str, Path],
    ) -> None:
        """
        Compress from a generator. n_total must be known upfront for mmap allocation.
        """
        output_path = Path(output_path)
        dim = self.encoder.dim

        # Pre-allocate output mmap
        indices_mmap = np.lib.format.open_memmap(
            str(output_path) + '.indices.npy',
            mode='w+',
            dtype=np.uint8,
            shape=(n_total, dim),
        )
        norms_mmap = np.lib.format.open_memmap(
            str(output_path) + '.norms.npy',
            mode='w+',
            dtype=np.float32,
            shape=(n_total,),
        )

        cursor = 0
        for batch in batches:
            batch = np.asarray(batch, dtype=np.float32)
            batch_n = batch.shape[0]
            norms, idxs = self._compress_batch(batch)
            indices_mmap[cursor:cursor + batch_n] = idxs
            norms_mmap[cursor:cursor + batch_n] = norms
            cursor += batch_n

        # Flush and package
        del indices_mmap, norms_mmap
        self._package_mmap_files(output_path, n_total, dim)

    def _compress_batch(self, batch: np.ndarray):
        """Rotate + quantize one batch. Returns (norms, indices)."""
        norms = np.linalg.norm(batch, axis=1)
        safe = np.maximum(norms, 1e-10)
        unit = batch / safe[:, np.newaxis]
        rotated = self.encoder.rotate(unit, normalize=False)
        dists = np.abs(rotated[:, :, np.newaxis] - self._codebook[np.newaxis, np.newaxis, :])
        indices = dists.argmin(axis=2).astype(np.uint8)
        return norms.astype(np.float32), indices
```

Key design choices:
- `mmap_mode='r'` on input: the input file is never fully loaded.
- `np.lib.format.open_memmap` for output: allocates a writable mmap backed by a `.npy` file on disk, filled batch by batch.
- `n_total` is required for stream mode because mmap requires knowing the shape upfront. For unknown-size streams, users should collect to disk first using a two-pass approach, or accept a temporary `.npy` file approach.
- FAISS handles large datasets through its `IndexIVFPQ.add_with_ids()` in batches and `write_index()` when done. The design mirrors this: add in batches, finalize to a single file.

### 4.5 `cli.py` — Expanded with Privacy Commands

**Full v1.0 command structure**:

```
tq-vectors
  compress     — existing
  search       — existing
  info         — existing (extend to show key fingerprint for .tqv files with privacy)
  keygen       — generate a new rotation key and save to .tqkey
  rotate       — rotate a .npy file with a key; outputs .npy
  unrotate     — reverse rotation on a .npy file (requires key)
  keyinfo      — show dim/fingerprint of a .tqkey file
  verify       — verify a .tqkey matches a canary blob or a compressed index's key_fingerprint
  rekey        — rotate vectors from old key to new key without exposing originals
```

**Pipe-friendly design**: `rotate` and `unrotate` operate on files. Pipe support via stdin/stdout for `.npy` format is not feasible without a framing protocol because `numpy.load` on a stream requires seeking. Instead, the pipe-friendly interface uses a `--stdin` / `--stdout` flag that writes raw float32 bytes:

```bash
# File mode (safe, recommended)
tq-vectors rotate --key secret.tqkey embeddings.npy -o rotated.npy

# Raw float32 pipe (advanced, requires knowing dim)
cat embeddings.f32 | tq-vectors rotate --key secret.tqkey --stdin --dim 1536 > rotated.f32
```

**Concrete `keygen` command**:

```python
# In cli.py, new subcommand
kg = sub.add_parser("keygen", help="Generate a new rotation key")
kg.add_argument("output", help="Output .tqkey file path")
kg.add_argument("-d", "--dim", type=int, required=True,
                help="Embedding dimension (e.g. 1536 for OpenAI, 768 for BERT)")
kg.add_argument("--from-seed", type=int, default=None,
                help="Deterministic key from integer seed (>= 2^64). "
                     "Omit to use OS entropy (recommended).")
```

```python
elif args.command == "keygen":
    from turboquant_vectors.private import PrivateEncoder
    import secrets
    if args.from_seed is not None:
        enc = PrivateEncoder.from_seed(dim=args.dim, seed=args.from_seed)
        print(f"  Generated deterministic key (seed-based)")
    else:
        enc = PrivateEncoder.generate(dim=args.dim)
        print(f"  Generated random key (OS entropy)")
    path = args.output
    enc.save_key(path)
    print(f"  Saved: {path}")
    print(f"  Dimension: {enc.dim}")
    print(f"  Fingerprint: {enc.fingerprint()}")
    print(f"  Canary (store with your index): {enc.make_canary().hex()}")
    if sys.platform != "win32":
        print(f"  Protect with: chmod 600 {path}")
```

**`verify` command** closes the debugging gap — it tells users immediately whether the key they loaded matches the data they have:

```bash
tq-vectors verify --key secret.tqkey --index my_index.tqv.npz
# Fingerprint match: YES (key=a3f2b1c0..., index=a3f2b1c0...)

tq-vectors verify --key wrong.tqkey --index my_index.tqv.npz
# ERROR: Key mismatch. Index expects key=a3f2b1c0..., but loaded key=deadbeef...
# Exit code: 1
```

**`info` command extended** to handle both compressed-only and private compressed files:

```bash
tq-vectors info my_index.tqv.npz
# File: my_index.tqv.npz
# Vectors: 1,000,000
# Dimensions: 1536
# Bits: 4
# Key fingerprint: a3f2b1c0...  (load with: tq-vectors verify --key YOUR.tqkey --index this_file)
# Original size: 5,859.4 MB
# Compressed size: 732.4 MB
# Compression ratio: 8.0x
```

---

## 5. Type Safety: `py.typed` and Protocol-Driven DB Wrappers

### `py.typed` Marker

Create `/turboquant_vectors/py.typed` as an empty file. Add to `pyproject.toml`:

```toml
[tool.setuptools.package-data]
turboquant_vectors = ["py.typed"]
```

Add type annotations to all public methods. The functions in `_rotation.py` already have annotations. The main gap is return type of `search()` — both classes return `Tuple[np.ndarray, np.ndarray]` but this is not annotated in `CompressedPrivateVectors`.

### Protocol-Driven DB Wrapper Architecture

Rather than requiring users to subclass an abstract base class, provide a `VectorDBAdapter` Protocol. This is lighter than ABC and matches the existing "no-framework" philosophy.

```python
# _types.py addition

from typing import Protocol, List, Any, runtime_checkable

@runtime_checkable
class VectorDBAdapter(Protocol):
    """
    Protocol for vector database integrations.

    Implement this protocol to make any vector DB work transparently
    with PrivateEncoder rotation. See turboquant_vectors.adapters for
    reference implementations (Pinecone, ChromaDB, FAISS).

    The pattern is:
        1. Wrap a DB client with a PrivateEncoder
        2. add() rotates before insert
        3. query() rotates the query before search
        4. Results come back in rotated space — IDs and scores are identical
    """

    def add(self, vectors: np.ndarray, ids: List[Any], **kwargs) -> None:
        """Insert vectors (rotation is applied internally)."""
        ...

    def query(self, vector: np.ndarray, top_k: int = 10, **kwargs) -> List[Any]:
        """Search (query rotation is applied internally)."""
        ...
```

A reference implementation for FAISS:

```python
# turboquant_vectors/adapters/faiss_adapter.py

import numpy as np
import faiss
from typing import List, Any, Optional
from turboquant_vectors.private import PrivateEncoder

class PrivateFAISSIndex:
    """
    FAISS index with automatic rotation via PrivateEncoder.

    Example:
        encoder = PrivateEncoder.load_key("secret.tqkey")
        index = PrivateFAISSIndex(encoder, dim=1536, index_type="Flat")
        index.add(embeddings, ids=list(range(len(embeddings))))
        results = index.query(query_vector, top_k=10)
    """

    def __init__(
        self,
        encoder: PrivateEncoder,
        dim: int,
        index_type: str = "Flat",
    ):
        if encoder.dim != dim:
            raise DimensionError(dim, (dim,))
        self.encoder = encoder
        self._index = faiss.IndexFlatIP(dim) if index_type == "Flat" else ...
        self._ids: List[Any] = []

    def add(self, vectors: np.ndarray, ids: Optional[List[Any]] = None) -> None:
        rotated = self.encoder.rotate(vectors)
        self._index.add(rotated)
        if ids is not None:
            self._ids.extend(ids)
        else:
            self._ids.extend(range(len(self._ids), len(self._ids) + len(vectors)))

    def query(self, vector: np.ndarray, top_k: int = 10):
        rotated_q = self.encoder.rotate(vector)
        if rotated_q.ndim == 1:
            rotated_q = rotated_q[np.newaxis, :]
        scores, faiss_ids = self._index.search(rotated_q, top_k)
        return [self._ids[i] for i in faiss_ids[0]], scores[0]
```

This pattern is a thin wrapper, not a heavy framework. Each adapter is a standalone file in `turboquant_vectors/adapters/`. This is an optional extra; the core package still has zero mandatory dependencies beyond numpy.

The adapter directory structure:

```
turboquant_vectors/adapters/
  __init__.py
  faiss_adapter.py     — requires faiss-cpu (optional dep already in pyproject.toml)
  chromadb_adapter.py  — requires chromadb
  langchain_adapter.py — requires langchain-core
```

These are importable but not included in the default `from turboquant_vectors import *`. They are import-guarded:

```python
# turboquant_vectors/adapters/__init__.py
def get_faiss_adapter():
    try:
        from turboquant_vectors.adapters.faiss_adapter import PrivateFAISSIndex
        return PrivateFAISSIndex
    except ImportError:
        raise ImportError(
            "FAISS adapter requires faiss-cpu: pip install turboquant-vectors[faiss]"
        )
```

---

## 6. Error Messages and Dimension Mismatch UX

The single most common user error is feeding a 1536-dim encoder a 768-dim array after switching embedding models. The current error from `private.py:282-285`:

```
ValueError: Expected vectors with dim=1536, got shape (100, 768)
```

The v1.0 `DimensionError` replaces this with:

```
turboquant_vectors.DimensionError: Dimension mismatch: encoder expects dim=1536,
got array shape (100, 768).
Did you use the right encoder? Check encoder.dim.

Common causes:
  - You changed embedding models (e.g., OpenAI ada-002 -> text-3-small)
  - You transposed your array (got (768, 100) instead of (100, 768))
  - You're passing a single vector — use shape (1, 1536), not (1536,)

Your encoder: PrivateEncoder(dim=1536, key=a3f2b1c0...)
```

The last two lines are added dynamically by catching the `ValueError` in `rotate()` and re-raising as `DimensionError`. The encoder's repr is embedded in the exception at raise time.

For key mismatches, `CompressedPrivateVectors.search()` should perform automatic fingerprint verification when an encoder is passed:

```python
def search(
    self,
    query: np.ndarray,
    top_k: int = 10,
    encoder: Optional["PrivateEncoder"] = None,  # NEW
) -> Tuple[np.ndarray, np.ndarray]:
    if encoder is not None:
        if encoder.fingerprint() != self.key_fingerprint:
            raise KeyMismatchError(self.key_fingerprint, encoder.fingerprint())
```

This makes the most common silent bug (loading the wrong `.tqkey`) into an immediate loud error.

---

## 7. dataclass vs attrs vs pydantic for CompressedVectors

**Decision: stay with `@dataclass`, add `__slots__`.**

`CompressedVectors` in `core.py` is already `@dataclass`. `CompressedPrivateVectors` is a plain class. In v1.0, convert both to `@dataclass(slots=True)` (Python 3.10+ required, which the project already enforces via `requires-python = ">=3.10"`).

`slots=True` gives:
- ~20% faster attribute access (matters for the inner loop of `search()` reading `self.codebook`, `self.indices`)
- Prevents accidental attribute addition (the stale cache bug in Bug 4 becomes impossible because `_decompressed_cache` is declared in `__slots__`)
- Type checkers understand `@dataclass` natively without stubs

```python
@dataclass(slots=True)
class CompressedPrivateVectors:
    indices: np.ndarray
    norms: np.ndarray
    codebook: np.ndarray
    bits: int
    dim: int
    key_fingerprint: str
    _decompressed_cache: Optional[np.ndarray] = field(default=None, repr=False)
```

`pydantic` is inappropriate here because numpy arrays are not pydantic-native types and the serialization story is already `np.savez_compressed`. `attrs` offers nothing beyond `dataclass(slots=True)` for this use case.

---

## 8. Implementation Map

### Phase 1: Bug Fixes for v0.3.1 (2 hours)

File: `/turboquant_vectors/private.py`
- Line 551-594: Remove `metric` parameter from `search()`. Always cosine. Raise `ValueError("CompressedPrivateVectors.search() is cosine-only")` if caller passes `metric`.
- Line 518: Rename `_decompressed_cache` to `__decompressed_cache`. Add `cache_clear()` method.
- Line 282: Replace `ValueError` with future-compatible message (pre-register `DimensionError` class location).

File: `/turboquant_vectors/core.py`
- Line 90: Add the "DO NOT change to default_rng" comment block.
- Line 224-234: Rewrite `decompress()` and `search()` to not instantiate `TurboQuantVectors`.
- Line 29: Add `search()` method to `CompressedVectors` dataclass (move logic from `TurboQuantVectors.search()`).

### Phase 2: Type Safety and `py.typed` (1 hour)

- Create `/turboquant_vectors/py.typed` (empty).
- Create `/turboquant_vectors/_types.py` with `DimensionError`, `KeyMismatchError`, `VectorIndex`, `VectorDBAdapter`.
- Update `pyproject.toml` to include `py.typed` in package data.
- Annotate all public method signatures in `private.py`, `core.py`, `_rotation.py`.
- Export `DimensionError`, `KeyMismatchError` from `__init__.py`.

### Phase 3: CLI Expansion for v0.4 (1-2 days)

File: `/turboquant_vectors/cli.py`
- Add `keygen` subcommand (args: `output`, `--dim`, `--from-seed`).
- Add `rotate` subcommand (args: `input`, `--key`, `-o/--output`, `--no-normalize`).
- Add `unrotate` subcommand (args: `input`, `--key`, `-o/--output`).
- Add `keyinfo` subcommand (args: `keyfile`).
- Add `verify` subcommand (args: `--key`, `--index`, `--canary`). Exit code 1 on mismatch.
- Add `rekey` subcommand (args: `input`, `--old-key`, `--new-key`, `-o/--output`).
- Extend `info` to detect and show `key_fingerprint` when present in `.npz`.

Create `/tests/test_cli_privacy.py` mirroring `test_cli.py` structure.

### Phase 4: Streaming API (2-3 days)

- Create `/turboquant_vectors/compress_stream.py` with `StreamingCompressor`.
- Create `/tests/test_streaming.py` with tests for:
  - Compress a 500K vector file via `compress_file()`, verify indices/norms match batch result.
  - Compress via generator, verify correctness.
  - RAM ceiling test: confirm peak RSS stays below 2x batch_size * dim * 4 bytes.

### Phase 5: Adapter Architecture (1 day)

- Create `/turboquant_vectors/adapters/__init__.py` (lazy import guards).
- Create `/turboquant_vectors/adapters/faiss_adapter.py` with `PrivateFAISSIndex`.
- Create `/turboquant_vectors/adapters/chromadb_adapter.py` with `PrivateChromaCollection`.
- Add `adapters = ["faiss-cpu"]` to `[project.optional-dependencies]` in `pyproject.toml`.

### Phase 6: `@dataclass(slots=True)` migration (1 hour, careful)

- Convert `CompressedPrivateVectors` from plain class to `@dataclass(slots=True)`.
- Add `__post_init__` where `__init__` currently does validation.
- Verify all 92 tests pass (especially `test_compression_private.py` which accesses `.indices`, `.norms`, `.codebook` directly).

---

## 9. Data Flow: Complete v1.0 Picture

```
[User vectors: (n, d) float32]
         |
         | PrivateEncoder.rotate()
         |   - DimensionError if shape mismatch
         |   - L2 normalize if normalize=True
         |   - matmul with Q^T
         v
[Rotated vectors: (n, d) float32]  --> Pinecone / Weaviate / Qdrant (direct)
         |
         | StreamingCompressor._compress_batch()  OR  PrivateEncoder.rotate_and_compress()
         |   - batch loop with mmap output  OR  single-shot in-memory
         |   - compute norms before normalize
         |   - argmin against codebook
         v
[CompressedPrivateVectors]
   .indices: (n, d) uint8
   .norms: (n,) float32
   .codebook: (2^bits,) float32
   .key_fingerprint: str
         |
         | .save("out.tqv.npz")  -- np.savez_compressed
         | .load("out.tqv.npz") -- np.load
         |
         | .search(rotated_query, top_k=10, encoder=enc)
         |   - KeyMismatchError if encoder.fingerprint() != self.key_fingerprint
         |   - _decompress() with cache
         |   - cosine similarity via unit-normalize both sides
         |   - argpartition top-k
         v
[(indices, scores): (k,) int, (k,) float32]
```

```
[CLI: tq-vectors keygen --dim 1536 -o secret.tqkey]
         |
         | PrivateEncoder.generate(dim=1536)
         | .save_key("secret.tqkey")
         v
[secret.tqkey: TQKEY magic + dim + rotation_matrix + SHA256]

[CLI: tq-vectors rotate --key secret.tqkey embeddings.npy -o rotated.npy]
         |
         | PrivateEncoder.load_key("secret.tqkey")
         | np.load("embeddings.npy", mmap_mode='r')  -- mmap for large files
         | encoder.rotate(vectors)  -- batched if needed
         | np.save("rotated.npy", rotated)
         v
[rotated.npy: (n, d) float32]

[CLI: tq-vectors verify --key secret.tqkey --index my_data.tqv.npz]
         |
         | encoder.fingerprint()  vs  npz['key_fp']
         | exit(0)  or  exit(1) with KeyMismatchError message
```

---

## 10. Critical Details

### What to never change without a migration plan

- The `RandomState(seed)` in `TurboQuantVectors.__init__`: changing it breaks all existing serialized `.npz` files where the rotation was generated this way. Since `CompressedVectors` stores the rotation matrix in the file, the only affected users are those who rely on the deterministic seed to regenerate (never saving the rotation to file). Those users are using the API incorrectly anyway.
- The 4-bit codebook values in `_rotation.py:95` (`lloyd = [0.1284, ...]`): these match the TurboQuant paper. The EXECPLAN notes explicitly: changing breaks reproducibility.
- The `.tqkey` file format magic bytes and layout: once users start storing keys, format changes break them. Add a version byte to the magic if format changes are needed in future.

### Performance

- `search()` in both classes decompresses once and caches. The cache is intentionally not pre-populated at `__init__` because for large collections the decompressed float32 may be larger than the user's available RAM. Cache-on-demand is correct.
- The `argpartition` trick for top-k (lines 199-206 in `core.py`, lines 599-605 in `private.py`) is correct and already optimal. The duplicate code should be extracted to `_rotation.py` as `_topk(scores, k)`.
- For the streaming compressor, the `np.abs(...).argmin(axis=2)` quantization step (private.py:476-477) is the hot path. For dim=1536, bits=4, and batch_size=50000, this is a `(50000, 1536, 16)` absolute-difference tensor — 4.7 GB float32. The existing batch loop prevents OOM; do not remove it.

### Security

- `verify_canary()` uses `==` not `hmac.compare_digest()`. This is a timing oracle on canary comparison. For the stated threat model (honest-but-curious server), this is acceptable. If the threat model expands to include active adversaries who can time canary verifications, replace with `hmac.compare_digest(self.make_canary(), canary)`.
- The `fingerprint()` function (16 hex chars of SHA-256) is adequate for identification but not for collision resistance. If two keys accidentally collide in fingerprint (probability 1 in 2^64), `KeyMismatchError` would not fire when it should. Extend fingerprint length to 32 chars (128 bits) in v1.0.
- `save_key()` already warns about Unix permissions. On Windows, the NTFS ACL warning path is skipped (line 229). Consider adding a `icacls` suggestion for Windows users with `sys.platform == "win32"`.

### Testing

The streaming tests need a RAM measurement. On Python, use `tracemalloc`:

```python
import tracemalloc
tracemalloc.start()
sc.compress_file("big_embeddings.npy", "out.tqv.npz")
current, peak = tracemalloc.get_traced_memory()
assert peak < 2 * batch_size * dim * 4 * 3  # 3x for input+output+temp
```

The adapter tests require optional deps. Mark them with `pytest.mark.skipif(not HAS_FAISS, ...)` using a module-level `HAS_FAISS = importlib.util.find_spec("faiss") is not None` check.

---

## 11. Build Sequence Checklist

**v0.3.1 (this week)**
- [ ] Fix `CompressedPrivateVectors.search()` metric parameter — remove ip/l2, raise `ValueError` with explanation
- [ ] Fix convenience `decompress()` and `search()` in `core.py` — remove dead TurboQuantVectors instantiation
- [ ] Add the RNG comment block to `TurboQuantVectors.__init__`
- [ ] Rename `_decompressed_cache` to name-mangled `__decompressed_cache` in `CompressedPrivateVectors`
- [ ] Add `cache_clear()` method to `CompressedPrivateVectors`
- [ ] Bump to 0.3.1 and publish

**v0.4 (next 2 weeks)**
- [ ] Create `/turboquant_vectors/py.typed`
- [ ] Create `/turboquant_vectors/_types.py` with `DimensionError`, `KeyMismatchError`, `VectorIndex`
- [ ] Update `__init__.py` to export new exception types
- [ ] Add `keygen` command to CLI
- [ ] Add `rotate` command to CLI (file-based, mmap input)
- [ ] Add `keyinfo` command to CLI
- [ ] Add `verify` command to CLI (exit code 1 on mismatch)
- [ ] Extend `info` command to show key fingerprint
- [ ] Add encoder parameter to `CompressedPrivateVectors.search()` for key verification
- [ ] Extend fingerprint from 16 to 32 hex chars (update `.tqkey` format: add version byte)
- [ ] Update all tests to use new fingerprint length
- [ ] Create `/tests/test_cli_privacy.py`
- [ ] Bump to 0.4.0 and publish

**v1.0 (next month)**
- [ ] Create `/turboquant_vectors/compress_stream.py` with `StreamingCompressor`
- [ ] Create `/tests/test_streaming.py` with RAM ceiling test
- [ ] Create `/turboquant_vectors/adapters/` directory and `__init__.py`
- [ ] Create `faiss_adapter.py` and `chromadb_adapter.py`
- [ ] Convert `CompressedPrivateVectors` to `@dataclass(slots=True)`
- [ ] Add `search()` method to `CompressedVectors` dataclass
- [ ] Extract `_topk()` deduplication from both `search()` implementations to `_rotation.py`
- [ ] Update `pyproject.toml` classifiers from `3 - Alpha` to `4 - Beta`
- [ ] Update `pyproject.toml` to include `py.typed` in package data
- [ ] Run mypy in strict mode on entire package; fix all errors
- [ ] Create `/tests/test_types.py` confirming both Compressed* classes satisfy `VectorIndex` Protocol
- [ ] Add rekey CLI command
- [ ] Bump to 1.0.0 and publish

---

## Relevant Files

- `/turboquant_vectors/core.py` — bugs 1 and 2 live here (lines 90, 224-234)
- `/turboquant_vectors/private.py` — bugs 3 and 4 live here (lines 551-594, 518)
- `/turboquant_vectors/_rotation.py` — stable; no changes needed
- `/turboquant_vectors/cli.py` — expand with privacy commands
- `/turboquant_vectors/__init__.py` — add `DimensionError`, `KeyMismatchError` to exports
- `/turboquant_vectors/_types.py` — create this file
- `/turboquant_vectors/compress_stream.py` — create this file
- `/turboquant_vectors/adapters/` — create this directory
- `/turboquant_vectors/py.typed` — create this empty file
- `/pyproject.toml` — add `py.typed` package data, adapters optional deps
- `/tests/test_cli_privacy.py` — create this file
- `/tests/test_streaming.py` — create this file
- `/tests/test_types.py` — create this file
- `/docs/plans/EXECPLAN.md` — decision log to update as each phase lands

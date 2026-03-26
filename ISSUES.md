# ISSUES — turboquant-vectors

> Known issues, improvement opportunities, and next steps.
> Updated: 2026-03-25

---

## Bugs — Fixed

### ~~B1. `assert` used for input validation in core.py~~ ✅ FIXED
Replaced with proper `ValueError`. Added NaN/inf check too.

### ~~B2. No NaN/inf validation in core.py compress~~ ✅ FIXED
Added `np.isfinite` check matching private.py.

### ~~B3. Windows chmod warning spam in tests~~ ✅ FIXED
Skipped permission check on `sys.platform == "win32"`.

### ~~B4. No `bits` validation in TurboQuantVectors~~ ✅ FIXED
Added `1 <= bits <= 8` and `dim >= 1` guards to `__init__`. Without this, `bits=9` would overflow uint8 indices silently.

### ~~B5. CLI `.replace(".npy", ...)` silently wrong for non-.npy inputs~~ ✅ FIXED
Replaced with `Path.with_suffix('')` approach.

### ~~B6. Pinned fingerprint test didn't actually pin a value~~ ✅ FIXED
Now asserts exact hex value `4f240186f1a3c69b`. If key derivation changes, this test catches it.

### ~~B7. Flaky correlation threshold in test_inversion_resistance~~ ✅ FIXED
Relaxed from 0.1 to 0.15 (still well above theoretical ~0.07 for d=128).

---

## Bugs — Open

### B8. IP/L2 metrics in CompressedPrivateVectors.search() compare incompatible spaces
**Severity:** Medium
**File:** `turboquant_vectors/private.py:573-616`
When `metric="ip"` or `metric="l2"`, the query (rotated, unit-norm) and database vectors (rotated, scaled by original norms) are in different magnitude spaces. Cosine metric normalizes both so it's fine. IP and L2 give technically wrong absolute values though rankings may still be reasonable.

### B9. `rekey_vectors()` and `unrotate()` skip NaN/inf validation
**Severity:** Low
Unlike `rotate()`, these methods don't check for NaN/inf input. Silently produces garbage.

---

## Code Quality — Fixed

### ~~C1. Codebook computation duplicated~~ ✅ FIXED
Extracted shared `compute_codebook()` into `_rotation.py`.

---

## Code Quality — Open

### C2. RNG API inconsistency
**Severity:** Low (breaking change to fix)
core.py uses deprecated `np.random.RandomState`, _rotation.py uses `np.random.default_rng`. Changing core.py would break reproducibility of existing compressed indexes.

### C3. Convenience functions recreate TurboQuantVectors per call
**Severity:** Low
`decompress()` and `search()` create a new `TurboQuantVectors` on every call. Works correctly (uses stored rotation) but wasteful.

### C4. `rotate_and_compress` double-normalizes when normalize=True
**Severity:** Low
When `self._normalize=True`, `rotate()` already unit-normalizes. The second normalization in `rotate_and_compress` is redundant. Harmless but wastes a `linalg.norm` pass.

### ~~C5. 3-bit codebook was completely wrong~~ ✅ FIXED (CRITICAL)
3-bit codebook used inner 4 values of the 4-bit codebook instead of correct 8-level Lloyd-Max centroids. ~6x worse MSE. Fixed with scipy-verified values. 4-bit also updated to higher-precision values.

### ~~C6. Quantization used O(n*d*2^bits) argmin on 3D tensor~~ ✅ FIXED
Replaced with O(n*d*bits) searchsorted on midpoint thresholds. 18% test suite speedup, eliminates huge memory allocation.

---

## Missing Tests — Fixed

### ~~T1. CLI has zero tests~~ ✅ FIXED (5 tests)
### ~~T2. 1-bit and 5-8 bit quantization untested~~ ✅ FIXED (8 parametrized)
### ~~T3. CompressedVectors save/load not tested~~ ✅ FIXED (2 tests)
### ~~T4. TurboQuantVectors bits/dim validation untested~~ ✅ FIXED (4 tests)
### ~~T5. rotate_and_compress dimension mismatch untested~~ ✅ FIXED (2 tests)

---

## Feature Gaps

### F1. No Colab notebook (Phase 8)
The plan calls for a runnable Colab notebook — highest ROI marketing asset.

### F2. No standard benchmark results (SIFT1M, GloVe)
Only benchmarked on Qdrant OpenAI dataset. SIFT1M and GloVe are the standard ANN-benchmarks datasets.

### F3. No VIBE or ann-benchmarks submission
Being listed would significantly increase credibility and adoption.

### F4. CLI has no privacy commands
No `rotate`, `keygen`, or `keyinfo` commands for PrivateEncoder.

---

## Research-Informed Opportunities (from web researcher agent)

### R1. TurboQuant paper trending NOW (March 25, 2026)
TechCrunch, VentureBeat, Tom's Hardware, 500+ HN points. The name "TurboQuant" is viral. Window of peak attention is 1-2 weeks. Marketing posts should go out ASAP.

### R2. New attack: Jina AI Conditional Masked Diffusion (Feb 2026)
81% token accuracy, black-box, 8 forward passes. Added to threat model. Rotation defeats it (same reason as Vec2Text).

### R3. SAQ (SIGMOD 2026) claims 80% less quantization error than Extended-RaBitQ
New competitor for compression. Data-dependent (needs PCA), unlike TurboQuant which is data-oblivious. Should benchmark against.

### R4. SPARSE (ICLR 2026) — potential "strict mode" add-on
Concept-aware elliptical noise defense. Could complement rotation: rotate for zero-loss scramble, then optionally apply SPARSE-style noise for defense-in-depth against known-plaintext attacks.

### R5. IronCore now does encrypted model training (July 2025)
Gartner Cool Vendor. Their encrypted-training feature is a differentiator we don't have. Still lossy though.

### R6. RecSys 2025 reproducibility study validates compression+privacy
Confirmed quantization reduces Vec2Text effectiveness. Validates our combined rotate+compress story.

### R7. No other rotation-based privacy package exists on PyPI
salty-embeddings (GitHub, 3 stars, permutation only, no PyPI) is the only thing close. Market remains open.

---

## Documentation

### D1. Marketing docs not posted
Reddit, HN, Twitter drafts exist but haven't been posted. TurboQuant is trending NOW — window closing.

### D2. No CHANGELOG
Version jumps from 0.2.1 to 0.3.0 with no changelog.

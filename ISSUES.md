# ISSUES — turboquant-vectors

> Known issues, improvement opportunities, and next steps.
> Updated: 2026-03-25

---

## Bugs

### ~~B1. `assert` used for input validation in core.py~~ ✅ FIXED
Replaced with proper `ValueError`. Added NaN/inf check too.

### ~~B2. No NaN/inf validation in core.py compress~~ ✅ FIXED
Added `np.isfinite` check matching private.py.

### ~~B3. Windows chmod warning spam in tests~~ ✅ FIXED
Skipped permission check on `sys.platform == "win32"`.

---

## Code Quality

### ~~C1. Codebook computation duplicated~~ ✅ FIXED
Extracted shared `compute_codebook()` into `_rotation.py`. Both modules import from there.

### C2. RNG API inconsistency
**Severity:** Low
**Files:** `turboquant_vectors/core.py:85-86`, `turboquant_vectors/_rotation.py:29-32`
core.py uses deprecated `np.random.RandomState` (legacy API). _rotation.py uses modern `np.random.default_rng`. Both work but changing core.py would break reproducibility of existing compressed indexes (different RNG = different rotation matrices for same seed).

### C3. Convenience functions recreate TurboQuantVectors per call
**Severity:** Low
**File:** `turboquant_vectors/core.py:230-249`
`decompress()` and `search()` convenience functions create a new `TurboQuantVectors` object on every call, regenerating the rotation matrix. Not a correctness issue (compressed data stores its own rotation) but wasteful. Users doing repeated searches should use the class API directly.

---

## Missing Tests

### ~~T1. CLI has zero tests~~ ✅ FIXED
Added `test_cli.py` with 5 tests covering compress, search, info, and help.

### ~~T2. 1-bit and 5-8 bit quantization untested~~ ✅ FIXED
Added parametrized test for all bit widths 1-8 in `test_core_extended.py`.

### ~~T3. CompressedVectors save/load not tested~~ ✅ FIXED
Added save/load roundtrip and search-after-load tests in `test_core_extended.py`.

---

## Feature Gaps

### F1. No Colab notebook (Phase 8)
The plan calls for a runnable Colab notebook — highest ROI marketing asset. Not built yet.

### F2. No standard benchmark results (SIFT1M, GloVe)
Only benchmarked on Qdrant OpenAI dataset. SIFT1M and GloVe are the standard ANN-benchmarks datasets that everyone compares against.

### F3. No VIBE or ann-benchmarks submission
Being listed on these platforms would significantly increase credibility and adoption.

### F4. CLI has no privacy commands
CLI only supports compression (`compress`, `search`, `info`). No `rotate`, `keygen`, or `keyinfo` commands for PrivateEncoder.

---

## Documentation

### D1. Marketing docs not posted
Reddit, HN, Twitter drafts exist but haven't been posted. Window of attention from TurboQuant paper may be closing.

### D2. No CHANGELOG
Version jumps from 0.2.1 to 0.3.0 with no changelog. Users can't tell what changed between versions.

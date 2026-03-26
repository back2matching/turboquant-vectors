# Changelog

All notable changes to turboquant-vectors.

## [0.3.1] - 2026-03-25

### Fixed
- **CRITICAL**: 3-bit codebook was completely wrong — used inner 4 values of 4-bit codebook instead of correct 8-level Lloyd-Max centroids (~6x worse MSE). Fixed with scipy-verified values.
- 4-bit codebook updated to higher-precision Lloyd-Max values
- `CompressedPrivateVectors.search()` now cosine-only (IP/L2 metrics removed — they produced incorrect results on compressed rotated data)
- `unrotate()` and `rekey_vectors()` now validate for NaN/inf input
- `TurboQuantVectors` rejects `bits` outside 1-8 and `dim < 1`
- CLI output path handling for non-.npy input files
- Windows: suppressed spurious chmod warning on `.tqkey` save
- `core.py`: replaced `assert` with `ValueError` for input validation
- Pinned fingerprint test to exact value (catches silent key derivation changes)
- Fixed flaky correlation threshold in statistical tests

### Changed
- Codebook computation extracted to shared `_rotation.compute_codebook()` (was duplicated in core.py and private.py)

### Added
- `ISSUES.md` — issue tracker
- `EXECPLAN.md` — active execution plan
- `test_cli.py` — 5 CLI tests (was untested)
- `test_core_extended.py` — 20 tests for validation, all bit widths, save/load
- Jina AI Conditional Masked Diffusion (Feb 2026) added to threat model
- SAQ, VIBE, RaBitQ references added to landscape doc

## [0.3.0] - 2026-03-25

### Added
- `demos/vec2text_demo.py` — Vec2Text inversion attack demo (BLEU drops ~0.90 to ~0.01)
- `benchmarks/real_data_benchmark.py` — TQ vs FAISS PQ on real OpenAI embeddings
- `benchmarks/adversarial_self_test.py` — Wasserstein-Procrustes self-attack (FAILS)
- Real-data benchmark results in README (10K OpenAI 1536-dim vectors)
- docs/ structure: research, marketing, plans

## [0.2.1] - 2026-03-25

### Fixed
- 18 new tests from audit round 2
- Bits validation, README precision improvements

## [0.2.0] - 2026-03-25

### Added
- `PrivateEncoder` — zero-loss embedding privacy via orthogonal rotation
- `rotate()`, `unrotate()`, `save_key()`, `load_key()`, `from_seed()`
- `rekey_vectors()`, `make_canary()`, `verify_canary()`, `fingerprint()`
- `rotate_and_compress()` — privacy + compression pipeline
- `CompressedPrivateVectors` with search and save/load
- 53 privacy tests, 11 demo tests, 12 compression-privacy tests
- `.tqkey` binary key format with SHA-256 checksum
- Seed enforcement >= 2^64, HMAC-SHA256 key derivation

## [0.1.0] - 2026-03-25

### Added
- `TurboQuantVectors` — core compression engine
- `compress()`, `decompress()`, `search()` convenience API
- `CompressedVectors` with save/load to .npz
- CLI: `tq-vectors compress`, `search`, `info`
- 14 core tests, 3 benchmark tests
- Lloyd-Max optimal codebooks for 1-4 bit

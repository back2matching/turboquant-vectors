# CLAUDE.md — turboquant-vectors

> Operating instructions for Claude Code on this repo.

## What Is This?

The privacy + compression layer for vector search. Two features:

1. **PrivateEncoder** — rotate embeddings with a secret orthogonal matrix before storing in third-party vector DBs. All distance metrics preserved exactly. Published inversion attacks (Vec2Text, ALGEN) fail completely.

2. **TurboQuant compression** — 8x compression of embeddings using rotation + optimal scalar quantization (ICLR 2026, data-oblivious, no training needed). Beats FAISS PQ on real OpenAI embeddings.

## Current State

| Metric | Value |
|--------|-------|
| Version | 0.3.0 (PyPI) |
| Tests | 123 |
| Dependencies | numpy only |
| Python | >= 3.10 |

## Key Files

```
turboquant_vectors/
  __init__.py       — exports PrivateEncoder, CompressedPrivateVectors, compress, search
  private.py        — PrivateEncoder class (privacy via orthogonal rotation)
  _rotation.py      — rotation matrix generation, HMAC seed derivation, validation, shared codebook
  core.py           — TurboQuantVectors compression engine
  cli.py            — CLI (compression only, no privacy commands yet)

tests/
  test_private.py           — 53 tests: math, security, key management, edge cases
  test_privacy_demo.py      — 11 tests: classifier attack, correlation, benchmarks
  test_compression_private.py — 12 tests: rotate_and_compress pipeline
  test_core.py              — 14 tests: core compression
  test_core_extended.py     — 14 tests: input validation, all bit widths, save/load, codebook
  test_cli.py               — 5 tests: CLI compress/search/info
  test_benchmark.py         — 3 tests: reproducibility

demos/
  inversion_demo.py    — proves classifier drops 88.9% -> 11.1% on rotated embeddings
  vec2text_demo.py     — Vec2Text inversion (requires torch >= 2.6 or workaround)

benchmarks/
  real_data_benchmark.py    — TQ vs FAISS PQ on real OpenAI embeddings
  adversarial_self_test.py  — Wasserstein-Procrustes attack (FAILS, same as random)
  BENCHMARK-DESIGN.md       — full benchmark plan

docs/
  research/     — threat model, API spec, landscape, integrations, benchmarks
  marketing/    — launch plan, Reddit posts, HN/Twitter drafts
  plans/
    EXECPLAN.md — ACTIVE plan (0.4 roadmap: marketing, harden, credibility, features)
    PLAN-private-embeddings.md — completed plan (0.1-0.3)
```

## Commands

```bash
pip install -e .                    # Dev install
python -m pytest tests/ -v         # Run all 117 tests
python demos/inversion_demo.py     # Privacy demo (needs sentence-transformers)
python benchmarks/real_data_benchmark.py  # Compression benchmark (needs faiss-cpu, datasets)
python benchmarks/adversarial_self_test.py  # Self-attack test
```

## Architecture

**Privacy:** PrivateEncoder generates a random orthogonal matrix Q via QR decomposition (float64 for stability, stored as float32). `rotate()` = matmul with Q^T. `unrotate()` = matmul with Q. Key stored in `.tqkey` binary format with SHA-256 checksum.

**Compression:** TurboQuantVectors applies the same rotation, then quantizes each coordinate to nearest centroid from a Beta-distribution-optimal codebook. No training data needed (data-oblivious).

**Combined:** `rotate_and_compress()` does both: rotate for privacy, then quantize for compression. Stores original norms for faithful decompression.

## Security Model

- **Threat model:** honest-but-curious vector DB provider
- **Seed enforcement:** >= 2^64, HMAC-SHA256 derivation with dimension
- **Known weakness:** d known-plaintext pairs recovers key via SVD (Procrustes)
- **Tested:** Wasserstein-Procrustes unsupervised attack FAILS (cos recovery = random)
- **Framing:** "privacy-preserving transform", never "encryption"

## Parent Project

This is a satellite project of [FlockRun](https://github.com/back2matching/flockrun). The FlockRun repo has the high-level TurboQuant ecosystem strategy in `docs/research/TURBOQUANT-NEXT-MOVES.md`.

## PyPI

- Account: back2matching
- Published: turboquant-vectors 0.3.0

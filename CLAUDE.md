# CLAUDE.md — turboquant-vectors

> Operating instructions for Claude Code on this repo.

## How To Work On This Repo

**On every conversation start:**
1. Read `.claude/WORKFLOW.md` for the current task queue
2. Run `python -m pytest tests/ -q` to confirm baseline
3. Check `git log --oneline -5` for recent context
4. Pick the next uncompleted task from WORKFLOW.md and execute it
5. After completing work: update WORKFLOW.md, update ISSUES.md if bugs found/fixed, commit, push

**Do NOT ask the user what to do.** Read the state, decide, execute. If blocked on something (e.g. needs credentials), document the blocker in ISSUES.md and move to the next task.

**After finishing a task**, always:
- Run tests to verify nothing broke
- Commit with descriptive message
- Mark the task done in WORKFLOW.md
- Check if the work revealed new tasks to add
- Start the next task immediately

**Research tasks:** When asked to research or plan, deploy multiple specialized agents in parallel, save findings as docs in `docs/research/`, update ISSUES.md and WORKFLOW.md with actionable items discovered, then start executing the highest-priority item.

**Quality bar:** Every claim in README/docs must be backed by code or tests. Run the honest assessment pattern (deploy a critical reviewer agent) periodically. Fix overclaiming immediately.

## What Is This?

Zero-loss embedding privacy via orthogonal rotation, with optional compression. Two features:

1. **PrivateEncoder** — rotate embeddings with a secret orthogonal matrix before storing in third-party vector DBs. All distance metrics preserved exactly. Published inversion attacks (Vec2Text, ALGEN) fail completely.

2. **Compression** — rotation + optimal scalar quantization inspired by TurboQuant (ICLR 2026). Data-oblivious, no training needed. Implements stage 1 only (rotation + Lloyd-Max codebook), not the full QJL pipeline.

## Current State

| Metric | Value |
|--------|-------|
| Version | 0.3.1 (dev, ready for PyPI) |
| Tests | 132 |
| Dependencies | numpy only |
| Python | >= 3.10 |
| Users | 0 (as of 2026-03-25) |
| Competitors on PyPI | 0 for rotation-based privacy |

## Key Files

```
.claude/WORKFLOW.md   — TASK QUEUE (read this first every session)
ISSUES.md             — bug tracker + open items
CHANGELOG.md          — version history
docs/plans/EXECPLAN.md — strategic roadmap

turboquant_vectors/
  __init__.py       — exports PrivateEncoder, CompressedPrivateVectors, compress, search
  private.py        — PrivateEncoder class (privacy via orthogonal rotation)
  _rotation.py      — rotation matrices, HMAC derivation, codebook, quantize()
  core.py           — TurboQuantVectors compression engine
  cli.py            — CLI (compression only, privacy commands planned)

tests/              — 124 tests across 7 files
notebooks/          — Colab demo notebook
docs/research/      — 11 research docs (threat model, landscape, architecture, etc.)
docs/marketing/     — launch drafts (Reddit, HN, Twitter)
```

## Commands

```bash
pip install -e .                           # Dev install
python -m pytest tests/ -q                 # Run all tests
python -m build                            # Build for PyPI
python demos/inversion_demo.py             # Privacy demo (needs sentence-transformers)
python benchmarks/real_data_benchmark.py   # Compression benchmark (needs faiss-cpu, datasets)
```

## Architecture

**Privacy:** PrivateEncoder generates a random orthogonal matrix Q via QR decomposition (float64 for stability, stored as float32). `rotate()` = matmul with Q^T. `unrotate()` = matmul with Q. Key stored in `.tqkey` binary format with SHA-256 checksum.

**Compression:** TurboQuantVectors applies rotation, then quantizes each coordinate to nearest centroid from a Lloyd-Max optimal codebook. No training needed (data-oblivious). Supports `stochastic=True` for formal Renyi DP via randomized rounding.

**Combined:** `rotate_and_compress()` does both: rotate for privacy, then quantize for compression. Stores original norms for faithful decompression.

## Security Model

- **Threat model:** honest-but-curious vector DB provider
- **Seed enforcement:** >= 2^64, HMAC-SHA256 derivation with dimension
- **Known weakness:** d known-plaintext pairs recovers key via SVD (Procrustes)
- **Tested:** Wasserstein-Procrustes unsupervised attack FAILS (cos recovery = random)
- **Framing:** "privacy-preserving transform", never "encryption"
- **Stochastic mode:** provides formal Renyi DP on top of rotation

## Honest Assessment (from 2026-03-25 audit)

- Privacy feature fills a genuine empty niche. No competitors on PyPI.
- Compression is NOT competitive with FAISS at scale. We implement stage 1 of TurboQuant only.
- The path to adoption is: one blog post, one LangChain PR, one real user.
- Do not over-plan. Execute the WORKFLOW.md task queue.

## Related Projects

Part of the TurboQuant family under [back2matching](https://github.com/back2matching):
- **turboquant** — KV cache compression (PyTorch/CUDA), 0.1.0 on PyPI
- **kvcache-bench** — KV cache benchmarking tool, 0.1.0 on PyPI
- **quant-sim** — quantization level benchmarking tool

## PyPI

- Account: back2matching
- Published: turboquant-vectors 0.3.0 (0.3.1 built but needs `twine upload`)

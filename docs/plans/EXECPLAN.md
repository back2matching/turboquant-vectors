# ExecPlan: turboquant-vectors — Full Roadmap to v1.0

> Synthesized from 7 specialized research agents + full codebase audit.
> This is the ACTIVE plan. Execute phase by phase.

**Updated:** 2026-03-25
**Current:** 0.3.0 on PyPI, 121 tests, dev branch (9 commits ahead of main)
**Agents consulted:** code-reviewer, test-reviewer, web-researcher, architecture-researcher, compression-researcher, privacy-researcher, distribution-researcher

---

## Strategic Context

TurboQuant paper went viral March 25, 2026 (TechCrunch, VentureBeat, Tom's Hardware, 500+ HN points, Morgan Stanley called it "breakthrough"). No competing rotation-based embedding privacy package exists on PyPI. IronCore Cloaked AI is the closest competitor but is lossy. The market window is open NOW.

**Three strategic advantages:**
1. **Zero recall loss** — mathematically proven, no other approach offers this
2. **Privacy + compression in one package** — unique combination
3. **Data-oblivious** — no training needed, works instantly on any domain

---

## Phase 1: Ship 0.3.1 (2-3 days) — HARDEN

> Fix remaining bugs, add CHANGELOG, publish to PyPI before marketing push.

### 1.1 Bug Fixes (all done on dev, need PyPI publish)

| Task | Status | Detail |
|------|--------|--------|
| B8: Remove broken IP/L2 metrics from CompressedPrivateVectors | ✅ Done | Cosine-only now |
| B9: NaN validation in unrotate/rekey | ✅ Done | Matches rotate() |
| B1-B7: Prior bug fixes | ✅ Done | assert→ValueError, bits validation, CLI path, etc. |
| RNG rationale comment | ✅ Done | Documents why RandomState must stay |
| CHANGELOG.md | ✅ Done | Full version history |

### 1.2 Remaining for 0.3.1

| Task | Effort | Detail |
|------|--------|--------|
| Bump version to 0.3.1 in pyproject.toml + __init__.py | 5min | |
| Build and publish to PyPI | 10min | `python -m build && twine upload dist/*` |
| Merge dev → main (code changes only, not docs/) | 30min | Set up .gitignore on main to exclude docs/ |
| Tag v0.3.1 on main | 5min | `git tag v0.3.1` |

### 1.3 Gate
- [ ] All tests pass on main
- [ ] `pip install turboquant-vectors==0.3.1` works clean
- [ ] README examples run without error

---

## Phase 2: Colab + Marketing (3-5 days) — LAUNCH

> Build the Colab notebook, then fire all marketing channels while TurboQuant is trending.

### 2.1 Colab Notebook (A4 — highest ROI asset)

**Target:** A single notebook that someone can run in 5 minutes and see the privacy claim proven.

**Structure:**
```
Cell 1: pip install turboquant-vectors sentence-transformers
Cell 2: Generate 5-category labeled embeddings (medical, financial, legal, personal, neutral)
Cell 3: Train category classifier → 88.9% accuracy
Cell 4: Rotate with PrivateEncoder
Cell 5: Same classifier on rotated → 11.1% (random chance)
Cell 6: Show search recall = 1.000 (identical results)
Cell 7: Compression demo: 8x smaller, recall > 96%
Cell 8: Key management: save/load/fingerprint
```

**Key design decisions:**
- Use `all-MiniLM-L6-v2` (384-dim, fast, runs on Colab free tier)
- Print side-by-side: original accuracy vs rotated accuracy
- No Gradio (adds complexity, slower load time, breaks on free Colab)
- Keep it under 2 minutes total runtime

### 2.2 Marketing Posts

| Day | Channel | Asset | Notes |
|-----|---------|-------|-------|
| Day 0 | Colab | Notebook published | Gate for all other posts |
| Day 1 | r/LocalLLaMA | Main post (REDDIT-POSTS.md) | Link to Colab, focus on Vec2Text threat |
| Day 1 | r/MachineLearning | Academic angle | Cite papers, honest limitations |
| Day 2 | Hacker News | Show HN (HN-TWITTER.md) | Link to GitHub + Colab |
| Day 2 | Twitter/X | 7-tweet thread | Code screenshots, benchmark table |
| Day 5 | Dev.to/Hashnode | Blog post | Long-form "Your RAG Embeddings Are Not Private" |

**Handling criticism (it will come):**
- "This is just a matrix multiply" → "Yes. And that's the point. Simplicity is a feature. One numpy matmul, zero recall loss, defeats all published attacks."
- "Not real encryption" → "Correct. We say that in the README. Threat model: honest-but-curious server."
- "Known-plaintext breaks it" → "Also in the README. 1,536 pairs for OpenAI embeddings. That's a high bar for the stated threat model."

### 2.3 Gate
- [ ] Colab runs end-to-end on free tier
- [ ] All 3 marketing drafts updated for 0.3.1 (version references)
- [ ] First post gets > 50 upvotes (signal to continue)

---

## Phase 3: CLI Privacy + Type Safety (1 week) — v0.4.0

> Make the privacy workflow first-class in the CLI, add proper types.

### 3.1 CLI Privacy Commands

```
tq-vectors keygen  --dim 1536 -o secret.tqkey      # Generate key
tq-vectors rotate  --key secret.tqkey emb.npy -o rotated.npy
tq-vectors keyinfo secret.tqkey                      # Show dim, fingerprint
tq-vectors verify  --key secret.tqkey --index data.tqv.npz  # Exit 1 on mismatch
```

**Implementation plan (from architecture agent):**
- `keygen`: calls `PrivateEncoder.generate()` or `.from_seed()`, saves with `save_key()`
- `rotate`: loads key, mmaps input .npy, rotates in batches, saves output
- `keyinfo`: loads key header without full matrix, prints dim + fingerprint
- `verify`: compares key fingerprint against index's stored `key_fp`

### 3.2 Type Safety

| Task | Effort | Detail |
|------|--------|--------|
| Create `_types.py` | 1h | `DimensionError`, `KeyMismatchError`, `VectorIndex` Protocol |
| Create `py.typed` marker | 5min | Empty file + pyproject.toml update |
| Add encoder param to `CompressedPrivateVectors.search()` | 30min | Auto-verify key fingerprint |
| Export new types from `__init__.py` | 10min | |

**`DimensionError` example:**
```
turboquant_vectors.DimensionError: Dimension mismatch: encoder expects dim=1536,
got array shape (100, 768).
Did you use the right encoder? Check encoder.dim.
```

### 3.3 Convenience Function Cleanup

Remove dead `TurboQuantVectors` instantiation from `decompress()` and `search()` convenience functions. Move search/decompress logic onto `CompressedVectors` itself (it already stores the rotation matrix).

### 3.4 Gate
- [ ] `tq-vectors keygen --dim 1536 -o test.tqkey` works
- [ ] `tq-vectors verify` returns exit code 0/1 correctly
- [ ] `DimensionError` raised on mismatch (not generic ValueError)
- [ ] mypy passes on public API
- [ ] Bump to 0.4.0, publish

---

## Phase 4: Credibility Benchmarks (1-2 weeks) — PROVE

> Get real numbers on standard datasets. Required for VIBE submission.

### 4.1 Standard ANN-Benchmark Datasets

| Dataset | Dims | Vectors | Why |
|---------|------|---------|-----|
| SIFT1M | 128 | 1M | The classic. Everyone cites it. |
| GloVe-100 | 100 | 1.2M | TurboQuant paper uses GloVe-200. |
| Qdrant OpenAI 100K | 1536 | 100K | Already have this benchmark. |

**Benchmark script:** `benchmarks/standard_benchmark.py`
- Downloads HDF5 from ann-benchmarks.com
- Runs TQ at 2/4/8-bit vs FAISS PQ at matched byte budgets
- Outputs recall@1, recall@10, recall@100, compression time
- Saves results to `benchmarks/results/` as JSON

### 4.2 Head-to-Head Comparisons

| Competitor | What to compare | Why |
|-----------|----------------|-----|
| FAISS PQ | Already done (we win by +0.4 to +1.2pp) | Baseline |
| RaBitQ / Extended-RaBitQ | At 1/2/4 bits per dim | Current SOTA, adopted by Elasticsearch |
| SAQ (SIGMOD 2026) | At 4-bit | Newest entrant, claims 80% less error |

**Key differentiator to highlight:** TurboQuant is data-oblivious (no PCA, no training). SAQ and RaBitQ require data-dependent preprocessing. This matters for:
1. Privacy (data-oblivious = no distribution leakage)
2. Deployment speed (instant, no training step)
3. Generalization (works on any domain without tuning)

### 4.3 VIBE Submission

Target: submit to [vector-index-bench/vibe](https://github.com/vector-index-bench/vibe) after getting numbers on their datasets. Being listed alongside FAISS, ScaNN, and SymphonyQG would be a major credibility boost.

### 4.4 Gate
- [ ] Beat FAISS PQ on SIFT1M at 4-bit (the headline number)
- [ ] Results table in README with standard datasets
- [ ] VIBE PR submitted

---

## Phase 5: Privacy Hardening (2 weeks) — STRENGTHEN

> Add optional noise layer, improve key management, explore formal guarantees.

### 5.1 "Strict Mode" — Rotation + Calibrated Noise

**Concept:** For users who want defense against known-plaintext attacks at the cost of ~1-3% recall.

```python
encoder = PrivateEncoder.generate(dim=1536, strict=True, noise_scale=0.01)
rotated = encoder.rotate(embeddings)  # rotation + small Gaussian noise
```

**Research basis:**
- SPARSE (ICLR 2026): concept-aware elliptical noise, 95% token protection, 98% task consistency
- TextCrafter: RL-learned adversarial perturbation orthogonal to embeddings
- DCPE (IronCore): scale-and-perturb, approximate distance preservation

**Our approach:** After rotation, add isotropic Gaussian noise with scale calibrated to defeat Procrustes alignment while keeping recall > 97%. The noise makes the SVD alignment attack return a noisy Q estimate, degrading inversion quality proportional to noise scale.

**Implementation:**
1. `PrivateEncoder.rotate(vectors, noise_scale=0.01)` — optional parameter
2. Noise drawn from `N(0, noise_scale^2 * I)` using the encoder's RNG seed
3. Document recall-vs-noise tradeoff curve
4. Never default to noise (zero-loss is our core value prop)

### 5.2 Key Rotation Protocol

Document and test a key rotation workflow:
1. Generate new key
2. `rekey_vectors(old_rotated, old_encoder)` with new encoder
3. Update all indexes
4. Destroy old key
5. Verify with canary

### 5.3 Formal Security Documentation

For compliance teams:
- Define IND-ROT security game (already in SPEC, needs polishing)
- Quantify information leakage: "server learns pairwise distances and norms, nothing else"
- GDPR assessment: are rotated embeddings still personal data? (Likely yes under GDPR's broad definition, but rotation is a valid "technical measure" under Art. 32)
- HIPAA: rotation qualifies as "access control" and "encryption" under the Administrative Safeguards, depending on interpretation

### 5.4 Gate
- [ ] `noise_scale` parameter works, tested
- [ ] Recall-vs-noise curve documented
- [ ] Key rotation guide in docs/
- [ ] Security whitepaper draft for compliance teams

---

## Phase 6: Streaming + Integrations (2 weeks) — SCALE

> Handle datasets that don't fit in RAM. Plug into popular frameworks.

### 6.1 StreamingCompressor

```python
from turboquant_vectors import StreamingCompressor, PrivateEncoder

encoder = PrivateEncoder.load_key("secret.tqkey")
sc = StreamingCompressor(encoder, bits=4, batch_size=50_000)
sc.compress_file("embeddings.npy", "compressed.tqv.npz")
# Peak RAM: ~50K * dim * 4 bytes, not N * dim * 4
```

**Design (from architecture agent):**
- Memory-mapped numpy input (`np.load(path, mmap_mode='r')`)
- Batch processing with configurable batch_size
- Output via `np.lib.format.open_memmap` for write-ahead
- Generator-based API for streaming from databases

### 6.2 Framework Adapters

```
turboquant_vectors/adapters/
  faiss_adapter.py      — PrivateFAISSIndex
  chromadb_adapter.py   — PrivateChromaCollection
  langchain_adapter.py  — PrivateEmbeddings wrapper
```

Each adapter is thin (~50 lines), import-guarded, optional dependency.

**LangChain integration is highest priority** — submit as community PR to `langchain-community`.

### 6.3 Gate
- [ ] `StreamingCompressor.compress_file()` handles 1M vectors with < 500MB peak RAM
- [ ] LangChain adapter works with `ChatOpenAI` + `PrivateEmbeddings`
- [ ] At least 2 adapter tests pass

---

## Phase 7: v1.0 Release (1 week) — SHIP

> Polish, final audit, promote to Beta status.

### 7.1 Final Cleanup
- Convert `CompressedPrivateVectors` to `@dataclass(slots=True)`
- Extract shared `_topk()` helper from both search implementations
- Run mypy strict on entire package
- Update pyproject.toml: `Development Status :: 4 - Beta`
- Comprehensive README update with all new features

### 7.2 Automation
- GitHub Actions: test on push (Python 3.10, 3.11, 3.12)
- GitHub Actions: auto-publish to PyPI on tag
- Dependabot for numpy version updates

### 7.3 Gate
- [ ] 150+ tests
- [ ] mypy clean
- [ ] CI/CD green
- [ ] v1.0.0 on PyPI

---

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-03-25 | Marketing before features | TurboQuant trending, attention window closing |
| 2026-03-25 | Cosine-only for CompressedPrivateVectors | IP/L2 on mixed-norm compressed data is incorrect |
| 2026-03-25 | Keep legacy RNG in core.py | Changing breaks existing compressed indexes |
| 2026-03-25 | Don't merge PrivateEncoder + TurboQuantVectors | FAISS pattern: separate transform from index |
| 2026-03-25 | Stay with @dataclass, add slots | No pydantic/attrs needed, numpy arrays aren't pydantic-native |
| 2026-03-25 | Noise as opt-in, never default | Zero-loss is core value prop |
| 2026-03-25 | Don't fix 4-bit codebook (C5) | Matches paper, changing breaks reproducibility |

---

## Competitive Landscape (as of 2026-03-25)

| Competitor | Approach | Recall Loss | Our Advantage |
|-----------|----------|-------------|---------------|
| IronCore Cloaked AI | Scale-and-perturb (DCPE) | 1-15% | Zero loss. They fuzz distances, we preserve them. |
| CyborgDB | HW-backed encryption | ~0% | We're a pip install. They need NVIDIA Hopper GPUs. |
| salty-embeddings | Permutation | 0% | Permutation is weaker than full rotation. 3 GitHub stars, no PyPI. |
| SAQ (SIGMOD 2026) | PCA + coordinate descent | 0% (compression) | Data-oblivious. They need PCA training. |
| RaBitQ | Randomized quantization | 0% (compression) | We combine privacy + compression. They do compression only. |

**No one else combines zero-loss privacy with data-oblivious compression in a pip package.**

---

## Timeline

```
Week 1 (Mar 25-31):  Phase 1 (ship 0.3.1) + Phase 2 (Colab + first posts)
Week 2 (Apr 1-7):    Phase 2 (remaining posts) + Phase 3 start (CLI)
Week 3 (Apr 8-14):   Phase 3 (v0.4.0) + Phase 4 start (benchmarks)
Week 4 (Apr 15-21):  Phase 4 (SIFT1M, RaBitQ comparison)
Week 5 (Apr 22-28):  Phase 5 (strict mode, key rotation)
Week 6 (Apr 29-May 5): Phase 6 (streaming, adapters)
Week 7 (May 6-12):   Phase 7 (v1.0 release)
```

---

## References

- [ISSUES.md](../../ISSUES.md) — bug tracker
- [CHANGELOG.md](../../CHANGELOG.md) — version history
- [CLAUDE.md](../../CLAUDE.md) — repo operating instructions
- [Threat model](../research/EMBEDDING-INVERSION-THREAT-MODEL.md)
- [Competitive landscape](../research/PRIVACY-PRESERVING-EMBEDDINGS-LANDSCAPE.md)
- [Benchmark research](../research/TURBOQUANT-VECTORS-REAL-BENCHMARKS.md)
- [Integration guide](../research/PRIVATE-ENCODER-INTEGRATIONS.md)
- [Architecture blueprint](../research/ARCHITECTURE-BLUEPRINT.md) — v1.0 API design (from architecture agent)

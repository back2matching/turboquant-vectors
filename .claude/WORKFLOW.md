# WORKFLOW — Automated Build Loop

> This file drives the `/loop` command. Each task is picked up in order.
> Mark tasks ✅ when done. Add new tasks as discovered. Commit after each task.
> Read ISSUES.md for bugs. Read EXECPLAN.md for strategy. Read research docs for context.

**Last updated:** 2026-03-25
**Branch:** dev
**Tests:** 128 passing, 8.6s

---

## Phase A: Honest Foundations (current)

> Incorporate the honest assessment findings. Fix overclaiming. Make the project genuinely solid.

- [x] A1. Fix README overclaiming — remove "First open-source implementation of TurboQuant" (we implement stage 1 only, not QJL). Reframe compression as "TurboQuant-inspired" or "rotation + scalar quantization". Keep privacy claims (those are accurate).
- [ ] A2. Save honest assessment as `docs/research/HONEST-ASSESSMENT.md` — the brutal findings from the assessment agent are currently only in temp files. Persist them.
- [ ] A3. Write educational blog post draft — `docs/marketing/BLOG-EMBEDDING-PRIVACY.md`. Not marketing hype. Educational: "Your Pinecone index leaks patient diagnoses: here's the math." Focus on the Vec2Text threat, show the Colab demo, link to package at the end.
- [x] A4. Add stochastic quantization option — one-line change in quantization loop. Privacy researcher said this gives formal Renyi DP from randomized rounding. Huge compliance value.
- [ ] A5. Compute and hardcode correct Lloyd-Max centroids for 5-8 bit — currently uses uniform quantization for these. Compression researcher said this is suboptimal. Use scipy to compute correct values.

## Phase B: Make It Actually Useful

> Focus on the ONE thing that gets real users: LangChain integration.

- [x] B1. Create `turboquant_vectors/adapters/langchain_adapter.py` — `PrivateEmbeddings(Embeddings)` wrapper that applies rotation to any base embeddings class. This is the 10,000x eyeballs play.
- [ ] B2. Create `turboquant_vectors/_types.py` — `DimensionError`, `KeyMismatchError` with helpful messages. Architecture agent designed these.
- [ ] B3. Add `keygen` and `rotate` CLI commands — make the privacy workflow first-class from command line.
- [ ] B4. Add `py.typed` marker — type safety for downstream users.

## Phase C: Credibility

> Get real numbers on standard datasets. One honest benchmark is worth 10 marketing docs.

- [ ] C1. Benchmark on SIFT1M (download HDF5, run TQ vs FAISS PQ at matched bits, save results)
- [ ] C2. Re-run real_data_benchmark.py with corrected codebooks — our benchmark numbers may have changed after the 3-bit/4-bit codebook fixes
- [ ] C3. Run the Colab notebook end-to-end locally and verify all outputs are correct

## Phase D: Distribution

> Only after the above is solid.

- [ ] D1. Publish 0.3.1 to PyPI — BLOCKED: no .pypirc, no keyring token, Playwright MCP disconnected. User needs to run: `python -m twine upload dist/turboquant_vectors-0.3.1*` or set up ~/.pypirc with API token first.
- [ ] D2. Submit LangChain PR (once B1 is tested)
- [ ] D3. Write one Hacker News comment in the existing TurboQuant thread explaining the privacy angle (not a Show HN — a relevant comment)

---

## Discovered Tasks (add here during loops)

<!-- New tasks found during work go here, then get sorted into phases -->

---

## Completed

<!-- Move completed tasks here with date -->

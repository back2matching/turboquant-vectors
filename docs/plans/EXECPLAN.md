# What's Next

**Updated:** 2026-03-25
**State:** 0.3.1 on dev (132 tests), 0.3.0 on PyPI

---

## Done

- 11 bugs fixed (including critical 3-bit codebook — was 6x worse MSE)
- Correct Lloyd-Max codebooks for all bit widths 1-8
- Quantization 3-5x faster via searchsorted
- Stochastic quantization option (formal Renyi DP)
- LangChain PrivateEmbeddings adapter
- CLI privacy commands (keygen, rotate, keyinfo, verify)
- DimensionError / KeyMismatchError types + py.typed
- Colab notebook + GitHub Actions CI
- CHANGELOG, honest README, all docs cleaned up

## Next

### Publish 0.3.1
- Build is ready in `dist/`
- Need PyPI credentials to upload (`twine upload dist/turboquant_vectors-0.3.1*`)
- Then merge dev → main and tag

### Get one real user
- Write a practical blog post explaining the embedding inversion threat
- Submit `PrivateEmbeddings` as a LangChain community integration
- Post in the existing TurboQuant HN thread with the privacy angle

### Benchmarks on standard datasets
- Re-run `real_data_benchmark.py` with corrected codebooks
- Run on SIFT1M for standard comparison numbers
- Verify Colab notebook produces correct outputs

## Later

- Streaming API for datasets > RAM
- SIFT1M / GloVe standard benchmarks
- Security whitepaper for compliance teams
- GDPR/HIPAA documentation templates

## Decisions made

| Decision | Why |
|----------|-----|
| Lead with privacy, not compression | Privacy niche is empty. Compression is weaker than full TurboQuant. |
| Cosine-only for CompressedPrivateVectors.search() | IP/L2 on mixed-norm compressed data is incorrect |
| Keep legacy RNG in core.py | Changing breaks existing compressed indexes |
| Don't fix 4-bit codebook sharing with 3-bit | Already fixed — all bit widths now have correct Lloyd-Max |
| Stochastic quantization opt-in only | Zero-loss is the default value prop |

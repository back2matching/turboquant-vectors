# TurboQuant: Next Moves Synthesis

> 8 novel, actionable ideas ranked by impact-to-effort ratio. Excludes all existing work (turboquant PyPI, kvcache-bench, llama.cpp PR, turboquant-vectors, GPU monitoring, production q8_0, consumer GPU benchmarking).

**Date:** 2026-03-25
**Sources:** 4 parallel research agents (community builds, novel applications, forum discussions, technique combinations)

---

## Summary Table

| Rank | Idea | Effort | Impact | Our Advantage |
|------|------|--------|--------|---------------|
| 1 | Persistent Agent KV Cache for FlockRun | 1-2 weeks | High | We own the agent runtime AND the compression library |
| 2 | TurboQuant + KV Eviction Hybrid Library | 2-3 weeks | Very High | First combo library, our PyTorch impl is ready |
| 3 | Privacy-Preserving Embeddings Module | 3-5 days | High | Nobody has packaged this, rotation = free obfuscation |
| 4 | VLM Visual Token Compressor | 2-3 weeks | Very High | Zero competition, visual tokens are ideal targets |
| 5 | vLLM TurboQuant KV Backend | 3-4 weeks | Enormous | vLLM only supports FP8 today, sub-4-bit is wide open |
| 6 | TurboQuant Self-Speculative Decoding | 2-3 weeks | High | QuantSpec proved it works at 4-bit, nobody tried rotation-based |
| 7 | Compressed KV Cache Persistence Format (.tqkv) | 1 week | Medium-High | Pairs with llama.cpp PR and FlockRun integration |
| 8 | TurboQuant + ParoQuant Weight Combo Demo | 3-5 days | Medium | 70B on 24GB headline, community already proposed it |

---

## 1. Persistent Agent KV Cache for FlockRun

**What:** Build a system where FlockRun agents serialize their TurboQuant-compressed KV caches to disk between invocations. When an agent wakes up (heartbeat, trigger, message), it restores its compressed cache instead of re-prefilling from scratch. The "Agent Memory Below the Prompt" paper (arxiv 2603.04428) proved this gives 136x TTFT speedup and fits 4x more agents. We combine their persistence idea with our TurboQuant compression for even smaller cache files.

**Why it's novel:** The arxiv paper used scalar Q4 quantization. Nobody has combined rotation-based compression with persistent agent caches. And nobody has integrated persistent KV caches into a real multi-agent runtime. This is where TurboQuant meets FlockRun's actual product.

**What we can reuse:**
- `turboquant` PyPI package (TurboQuantMSE class works on any tensors)
- `tests/kv-compression/turboquant_cache.py` (our DynamicCache subclass)
- FlockRun's existing agent invocation pipeline (buildAgentContext -> executeToolLoop)
- FlockRun's GPU monitoring (gpu.ts, VRAM-aware heartbeat scheduling)
- The paper's open-source code at github.com/yshk-mxim/agent-memory as reference

**Effort:** 1-2 weeks. Cache serialization/deserialization is straightforward. The hard part is cache invalidation (when does a stale cache hurt more than re-prefilling?) and integration with Ollama's context management.

**Impact:** Directly improves FlockRun's core product. 3 agents on RTX 4080 currently waste ~15 seconds per wake-up on re-prefill. With persistent TQ3 caches: sub-second restoration, 3x smaller cache files than Q4, and room for 5+ agents at 8K context. This is a genuine product differentiator.

**Risk:** Ollama manages its own KV cache internally. Bypassing it for persistent caches may require running a custom inference server (our existing `tests/kv-compression/server.py`) instead of Ollama, which adds operational complexity. Cache staleness detection is non-trivial: if the agent's system prompt changes, the cached KV is invalid.

---

## 2. TurboQuant + KV Eviction Hybrid Library

**What:** A Python library that combines TurboQuant rotation-based quantization with attention-score-based token eviction (H2O/StreamingLLM style). Keep only the important tokens in the KV cache, and store those at 3-4 bits via TurboQuant. Two layers of compression: fewer tokens AND smaller per-token storage. Target: 96-98% total KV reduction (32K context in ~100-200 MB instead of 4.6 GB).

**Why it's novel:** MiniKV (ACL 2025) proved that hybrid eviction+quantization works, achieving 86% KV reduction. But MiniKV uses scalar quantization. TurboQuant's rotation-based approach gives better quality at the same bit-width, and its data-oblivious property means evicting tokens doesn't affect the quantization quality of remaining tokens (no recalibration needed). Nobody has combined these two specific techniques.

**What we can reuse:**
- `turboquant` PyPI (core algorithm)
- Our research doc `TURBOQUANT-COMBINATIONS.md` has the full design
- MiniKV's Triton kernel approach as implementation template
- QJL CUDA kernels (Apache-2.0) for the residual correction step

**Effort:** 2-3 weeks. The eviction logic (cumulative attention scoring) is well-documented in H2O and StreamingLLM papers. Combining it with our existing TurboQuant quantization is architecturally clean. The custom attention kernel (handles sparse + quantized KV) is the hardest part.

**Impact:** Very high. 96-98% KV reduction enables 128K+ contexts on 16GB GPUs, which is currently impossible. The r/LocalLLaMA audience would love this. Publishable as a workshop paper.

**Risk:** The custom attention kernel is non-trivial. Performance may not match theoretical gains if the kernel isn't optimized. Eviction policy is model-dependent.

---

## 3. Privacy-Preserving Embeddings Module

**What:** Add a `turboquant.private` module to our existing PyPI package that uses TurboQuant's random rotation as a reversible obfuscation layer for embeddings. Users rotate their embeddings with a secret rotation matrix before sending to a vector database or API. Inner product similarity is preserved (TurboQuant's mathematical guarantee), but the raw embeddings can't be reconstructed without the secret matrix. Essentially: free privacy for vector search.

**Why it's novel:** TurboQuant's rotation is orthogonal, meaning it preserves all distance metrics exactly (not approximately). This is mathematically stronger than differential privacy approaches which add noise and degrade quality. Nobody has packaged this as a privacy tool. A zero-loss privacy approach is genuinely new.

**What we can reuse:**
- `turboquant` PyPI package (rotation matrix generation, the `TurboQuantMSE` class)
- turboquant-vectors plan (compress + search API, same target audience)
- Our existing recall benchmarks (prove that rotated search has identical recall)

**Effort:** 3-5 days. The rotation code already exists. This is mostly packaging, API design, and compelling documentation with a privacy angle.

**Impact:** High. "Zero-cost privacy for your embeddings" is a headline that gets clicks. The audience is every company sending embeddings to third-party vector DBs (Pinecone, Weaviate, Qdrant). Could get citations from the privacy-ML community.

**Risk:** Low technical risk. Positioning risk: people may not believe it's "real" privacy because the rotation is invertible if the matrix leaks. Need to be clear about the threat model (honest-but-curious server).

---

## 4. VLM Visual Token Compressor

**What:** Apply TurboQuant at aggressive bit-widths (2-3 bits) specifically to visual token KV cache entries in vision-language models, while keeping text tokens at higher precision (4-8 bits). Visual tokens are highly redundant (nearby image patches produce similar vectors). A single high-res image generates 2880+ KV cache entries; a 30-second video generates 10K+. Compressing these is the difference between "fits on consumer GPU" and "needs A100."

**Why it's novel:** VL-Cache (ICLR 2025) and MBQ (Tsinghua, 2025) proved modality-aware KV compression works, but both use scalar quantization or eviction. Nobody has applied rotation-based quantization to VLM KV caches. Visual token redundancy means TurboQuant's quantization error should be even lower than for text.

**What we can reuse:**
- `turboquant` PyPI (works on any tensors)
- `turboquant_cache.py` (DynamicCache subclass, needs per-modality extension)
- TURBOQUANT-COMBINATIONS research doc

**Effort:** 2-3 weeks. Modify DynamicCache to track token modality, apply different TQ bit-widths per modality, test on LLaVA or similar.

**Impact:** Very high. VLMs are the fastest-growing model category. Video understanding KV drops from 1.7 GB to ~215 MB. Enables real local video understanding on RTX 4080.

**Risk:** New domain for us. Testing requires downloading VLMs. Modality detection varies by architecture.

---

## 5. vLLM TurboQuant KV Backend

**What:** Contribute a TurboQuant KV cache quantization backend to vLLM. vLLM currently supports FP8 (E4M3/E5M2) and NVFP4 (Blackwell-only). Store TurboQuant-compressed KV entries in vLLM's paged memory blocks. Fits 4-5x more sequences than FP16, 2x more than FP8.

**Why it's novel:** Sub-8-bit KV cache on non-Blackwell GPUs doesn't exist in vLLM. This would bring TurboQuant to every vLLM deployment worldwide.

**What we can reuse:**
- `turboquant` PyPI, llama.cpp integration experience
- TURBOQUANT-COMBINATIONS research doc (full vLLM design)
- vLLM's existing FP8 integration as template

**Effort:** 3-4 weeks. Custom CUDA attention kernels needed. vLLM's architecture is complex.

**Impact:** Enormous. Front-page HN material. Would establish us as the go-to TurboQuant implementers.

**Risk:** High. Large codebase, long review cycles, need proven CUDA kernel skills. Only attempt after llama.cpp PR lands.

---

## 6. TurboQuant Self-Speculative Decoding

**What:** Use a single model with two KV cache quality levels for self-speculative decoding. The "draft" pass uses aggressively compressed TQ 3-bit KV cache (faster due to less memory bandwidth). The "verify" pass uses full-precision KV. QuantSpec (ICML 2025, Apple) proved this works at 4-bit.

**Why it's novel:** Nobody has tried rotation-based quantization for speculative decoding. TurboQuant's unbiased inner product property means the draft model's attention scores are unbiased estimators of the true scores, improving acceptance rates vs. scalar quantization's biased estimates.

**What we can reuse:**
- `turboquant` PyPI and `turboquant_cache.py`
- QuantSpec framework as template
- Consumer GPU benchmarking infrastructure

**Effort:** 2-3 weeks. Tricky part: hierarchical bit-sharing so TQ indices are derivable from higher-precision cache.

**Impact:** High. 2-2.8x generation speedup on single-GPU setups. Fills a gap for consumer GPUs where you can't fit separate draft + target models.

**Risk:** Rotation step latency could eat into speculative decoding gains. Needs careful kernel optimization.

---

## 7. Compressed KV Cache Persistence Format (.tqkv)

**What:** Define a compact binary format for serialized TurboQuant-compressed KV caches. Think GGUF but for KV state: stores rotation matrix, quantized indices, norms, and metadata. Enables saving/loading compressed KV caches to/from disk.

**Why it's novel:** GGUF standardized model weight serialization. No equivalent exists for KV cache state. A standard format with TurboQuant's compression makes cache files 5-6x smaller than FP16 serialization.

**What we can reuse:**
- `turboquant` PyPI, llama.cpp integration (block-based type systems)
- FlockRun's existing knowledge serialization patterns

**Effort:** 1 week. Straightforward format design + Python reader/writer.

**Impact:** Medium-high. Foundational infrastructure enabling #1 (persistent agent caches) and prompt caching workflows. "Compile" a long document into a .tqkv file and load in sub-second time.

**Risk:** Low. Main risk is premature standardization. Mitigate with versioning.

---

## 8. TurboQuant + ParoQuant Weight Combo Demo

**What:** PoC combining ParoQuant (4-bit weights) with TurboQuant (3-bit KV) to run 70B models on 24GB GPUs. Use an existing 4-bit quantized 70B model and add TurboQuant KV cache on top.

**Why it's novel:** The community proposed this combo on HN but nobody built it. Both use rotation-based approaches from the same ICLR 2026 cycle.

**What we can reuse:**
- `turboquant` PyPI, existing 4-bit models on HuggingFace
- Consumer GPU benchmarking infrastructure

**Effort:** 3-5 days for a demo.

**Impact:** Medium. "70B on a $1200 GPU" headline generates interest, but practical value is limited by speed.

**Risk:** Quality may be poor from stacking two aggressive quantization methods. Need honest benchmarks.

---

## Recommended Execution Order

**Quick wins first, bigger bets after:**

1. **Privacy-Preserving Embeddings (#3)** — 3-5 days, low risk, high shareability
2. **KV Cache Persistence Format (#7)** — 1 week, low risk, foundational
3. **ParoQuant + TurboQuant Demo (#8)** — 3-5 days, great headline
4. **Persistent Agent KV Cache (#1)** — 1-2 weeks, direct FlockRun value
5. **KV Eviction Hybrid (#2)** — 2-3 weeks, publishable result
6. **VLM Visual Token Compressor (#4)** — 2-3 weeks, new domain, very high ceiling
7. **Self-Speculative Decoding (#6)** — 2-3 weeks, needs kernel optimization
8. **vLLM Backend (#5)** — 3-4 weeks, highest risk, highest ceiling (after llama.cpp PR lands)

---

## Key Strategic Observations

**Our unique position:** We are the only team with (a) a working TurboQuant implementation, (b) a published PyPI package, (c) a multi-agent runtime that directly benefits from KV compression, and (d) ongoing llama.cpp integration. Nobody else has this combination.

**The privacy angle (#3) is underappreciated.** Every other team focused on memory savings. The rotation-as-obfuscation angle is mathematically elegant, practically useful, and completely uncontested. Could become our most-cited contribution despite being technically the simplest.

**The persistent agent cache (#1) is the highest-leverage FlockRun feature.** Turns TurboQuant from "interesting research" into "tangible product improvement." The arxiv paper validated the approach.

**vLLM (#5) is the moonshot.** Landing TurboQuant in both llama.cpp and vLLM makes us the de facto implementation team. But it requires CUDA kernel expertise. Don't rush it.

---

## Competitive Landscape Update (as of 2026-03-25)

| Who | What | Status |
|-----|------|--------|
| tonbistudio/turboquant-pytorch | PyTorch full pipeline | 48 stars, MIT, RTX 3060 tested |
| veritatisquaesitoressumus | C + CUDA kernels | 18/18 tests, submitted to ik_llama.cpp |
| TheTom/turboquant_plus | Metal/Apple Silicon | M5 Max benchmarks, 141 tests |
| mudler (LocalAI) | llama.cpp fork branch | Builds, LocalAI tracking issue open |
| mlx-optiq (PyPI) | MLX TurboQuant KV cache | Published, HF models available |
| NVIDIA KVTC | Competing approach (20x, needs calibration) | ICLR 2026, no open code |
| Apple CommVQ | Competing approach (1-bit KV, trained codebook) | ICML 2025 |
| ParoQuant (z-lab) | Weight compression via rotation | ICLR 2026, HF models available |
| **Us (back2matching)** | turboquant PyPI + turboquant-vectors PyPI + llama.cpp TQ4_0 + FlockRun | First to publish KV + vectors packages, first llama.cpp PR |

**Bottom line:** We have 3 packages on PyPI (turboquant, kvcache-bench, turboquant-vectors). turboquant-vectors 0.1.0b1 has fair FAISS benchmarks (+8pp at 4-bit). The field is moving fast. Our edge is the breadth of our ecosystem (PyPI packages + llama.cpp + FlockRun + these 8 new ideas).

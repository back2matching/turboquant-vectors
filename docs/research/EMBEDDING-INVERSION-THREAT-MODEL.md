# Embedding Inversion Threat Model

> Threat analysis for the Privacy-Preserving Embeddings Module (TurboQuant `private` submodule).
> Covers known attacks, defense properties of orthogonal rotation, and honest security claims.

**Date:** 2026-03-25
**Context:** Item #3 from TURBOQUANT-NEXT-MOVES.md. TurboQuant's random orthogonal rotation, originally designed for quantization quality, doubles as a privacy mechanism for embeddings stored in third-party vector databases.

---

## 1. Text-to-Embedding Inversion Attacks

### 1.1 Vec2Text (Morris et al., Cornell, 2023-2024)

The foundational attack. Trains a correction model that iteratively refines text hypotheses to match a target embedding.

**How it works:**
1. Train a "zero-step" model: embedding -> initial text guess
2. Re-embed the guess, compute embedding residual
3. Train a correction model conditioned on (target embedding, current guess embedding, current guess text)
4. Iterate 50-100 correction steps, re-embedding each time

**Published accuracy (GTR-base, 768-dim):**

| Input length | BLEU | Exact match |
|-------------|------|-------------|
| 32 tokens | 97.3 | 92% |
| 128 tokens | 55.0 | 8% |

**On OpenAI text-embedding-ada-002 (1536-dim):**

| Input length | BLEU | Exact match |
|-------------|------|-------------|
| 32 tokens | 83.4 | 60.9% |
| 128 tokens | 55.0 | 8.0% |

**Key limitations:**
- Requires training a separate inversion model per target embedding model
- Performance degrades sharply with text length (92% at 32 tokens, 8% at 128 tokens)
- Trained on fixed sequence lengths; out-of-distribution lengths degrade further
- Needs millions of text-embedding pairs for training (white-box or query access to the embedding API)

**PII recovery:** Vec2Text can recover full names from clinical notes (MIMIC-III dataset). This is the headline privacy risk that motivates our module.

### 1.2 ALGEN (Chen et al., ACL 2025)

Few-shot attack. Aligns the victim embedding space to an attacker's known embedding space, then uses a generative model to reconstruct text.

**Breakthrough threat level:** With only 1,000 text-embedding pairs from the victim model, ALGEN reaches near-optimal inversion performance across a range of black-box encoders. A single data point is sufficient for a partially successful attack.

**Why this matters:** ALGEN proves that an attacker does NOT need full training access. A small number of leaked pairs is enough to mount a meaningful inversion attack. Defenses that assume the attacker has zero knowledge of the embedding space are insufficient.

**Tested defenses:** The ALGEN paper tested multiple defense mechanisms and found none effective against their attack.

### 1.3 ZSinvert (Zhang & Morris, March 2025)

Zero-shot, universal inversion. Uses adversarial decoding without training a model-specific inverter.

- Works on ANY embedding model without training
- Fewer encoder queries than Vec2Text
- Same algorithm for all embeddings (past, present, future models)
- Lower accuracy than Vec2Text but non-trivial; the "free" nature is the threat

### 1.4 Zero2Text (February 2026)

The current state of the art for strict black-box inversion.

- Zero training, zero leaked alignment data
- Uses a pre-trained LLM as a universal generator
- Recursive online alignment via dynamic ridge regression
- Issues a limited number of online API queries to the embedding model
- Achieves state-of-the-art fidelity in strict black-box settings

### 1.5 Transferable Embedding Inversion (Huang et al., ACL 2024)

Transfer attack trained on one embedding model, applied to another without querying the target.

- Recovers 98% of age and 99% of sex attributes from clinical embeddings
- Identifies sensitive attributes (age, sex, disease) with 80-99% accuracy
- Works cross-model: train on Model A, attack Model B

### 1.6 Model Vulnerability Summary

| Embedding Model | Dims | Vec2Text Trained? | Inversion Risk |
|----------------|------|-------------------|----------------|
| GTR-base | 768 | Yes (original paper) | **CRITICAL** - 92% exact match at 32 tokens |
| OpenAI text-embedding-ada-002 | 1536 | Yes (original paper) | **HIGH** - 60.9% exact match at 32 tokens |
| OpenAI text-embedding-3-small | 1536 | No Vec2Text model published; ZSinvert/Zero2Text applicable | **MEDIUM-HIGH** - universal attacks apply |
| OpenAI text-embedding-3-large | 3072 | No Vec2Text model published | **MEDIUM** - higher dimensionality helps slightly |
| Cohere embed-v3 | 1024 | No Vec2Text model published | **MEDIUM-HIGH** - universal attacks apply |
| all-MiniLM-L6-v2 | 384 | Likely trainable (open model) | **HIGH** - open weights enable white-box training |
| Sentence-transformers (general) | 384-1024 | Multiple models attacked in literature | **HIGH** - open weights, well-studied |

**Dimensionality effect:** Higher dimensions do NOT prevent inversion. They slightly increase the difficulty (more information compressed into each dimension at lower dims, but the embedding still encodes the same semantic content). The 92% result was on 768-dim GTR. The 60.9% result on 1536-dim ada-002 suggests dimension helps marginally, but not decisively. The real factors are: text length, model architecture openness, and availability of training pairs.

---

## 2. Membership Inference Attacks

**Question:** Can an attacker determine if a specific document is in the embedding index?

### 2.1 RAG-Specific Membership Inference

Recent work (2024-2025) demonstrates effective membership inference attacks against RAG systems:

- **S2MIA:** Provide the first half of a document, request completion. If the RAG system completes it accurately, the document is likely in the database.
- **MBA (Masked-Based Attack):** Prompt the model to predict masked tokens. Higher accuracy implies membership.
- **Interrogation Attack (IA):** Ask multiple questions that are hard to answer without the specific document.

These attacks target the RAG system's LLM output, not the raw embeddings. They are **not mitigated by embedding rotation** because the attack vector is the retrieval+generation pipeline, not the embedding space.

### 2.2 Direct Embedding Membership Inference

For raw embedding access (e.g., a compromised vector database):

- An attacker with the embedding model can embed a candidate text and check if a near-exact match exists in the index
- Cosine similarity > 0.999 is a strong membership signal
- **Rotation DOES mitigate this:** The attacker's freshly computed embedding is in the original space; the stored embeddings are in the rotated space. Without the rotation matrix, the attacker cannot compute the correct rotated embedding to check against.

### 2.3 Membership Inference Risk Assessment

| Attack Vector | Rotation Mitigates? | Notes |
|--------------|---------------------|-------|
| Direct embedding comparison | **YES** | Attacker can't produce rotated embeddings without the key |
| RAG output analysis (S2MIA, MBA, IA) | **NO** | Attack is on the LLM output, not embeddings |
| Timing side-channels | **NO** | Query latency may leak membership |

---

## 3. Attribute Inference Attacks

### 3.1 Sensitive Attributes from Embeddings

Embeddings encode far more than semantic similarity. Published results show:

- **Demographics:** Age (98% accuracy), sex (99%), ethnicity (high accuracy) recoverable from clinical note embeddings via transfer attacks
- **Medical conditions:** Disease presence/absence inferable with 80-99% accuracy
- **PII:** Full names recovered from MIMIC-III clinical note embeddings
- **Location:** Geographic information inferable from language patterns in embeddings

### 3.2 The "Leak Auditor" Framework

Attribute inference works even without full text reconstruction. An attacker trains a classifier: embedding -> attribute. This is simpler than full inversion and requires fewer resources.

Example: Train a binary classifier (embedding -> "has diabetes" / "no diabetes") using a small labeled dataset. This works because the embedding faithfully encodes the semantic content, including medical conditions.

### 3.3 Attribute Inference Risk with Rotation

| Attack | Rotation Mitigates? | Why |
|--------|---------------------|-----|
| Train classifier on original embeddings, apply to rotated | **YES** | Classifier expects original space; rotated vectors are gibberish to it |
| Train classifier on rotated embeddings (attacker has rotation key) | **NO** | If key is compromised, all bets are off |
| Train classifier on rotated embeddings (attacker has some labeled pairs) | **PARTIAL** | See Section 4.4 on known-plaintext attacks |

---

## 4. How Orthogonal Rotation Defeats (and Doesn't Defeat) These Attacks

### 4.1 The Core Mathematical Guarantee

Let R be a d x d orthogonal matrix (R^T R = I, det(R) = +/-1). For any vectors x, y:

```
<Rx, Ry> = x^T R^T R y = x^T y = <x, y>
```

**Inner products are exactly preserved.** This means:
- Cosine similarity: identical before and after rotation
- Euclidean distance: identical (for normalized vectors, a monotonic transform of cosine)
- Ranking: identical. Top-k search on rotated embeddings returns the same results as on originals.

**This is not approximate. It is exact. Zero quality loss.**

### 4.2 Why Vec2Text Fails on Rotated Embeddings

Vec2Text's inversion model is trained on (text, embedding) pairs in the original embedding space. The model learns a mapping: E_original -> text.

After rotation with secret matrix R:
- Stored embeddings: E_rotated = R * E_original
- The inversion model receives E_rotated but expects E_original
- E_rotated is a valid point in R^d but does NOT correspond to any text in the original embedding space
- The inversion model produces garbage

**This is not a marginal degradation. The attack completely fails.** The rotated embedding is as useful to the original inversion model as a random vector of the same norm.

The same logic applies to:
- ZSinvert (adversarial decoding against the original encoder: rotated vectors don't match)
- Zero2Text (online alignment against the original encoder: same failure)
- ALGEN with an alignment trained on original-space pairs

### 4.3 Could an Attacker Train a New Inversion Model on Rotated Embeddings?

**No, not without the rotation matrix or equivalent information.** Here's why:

To train Vec2Text on the rotated space, the attacker needs (text, rotated_embedding) pairs. Producing these requires:
1. Access to the original embedding model (to embed text -> E_original), AND
2. The secret rotation matrix R (to compute R * E_original = E_rotated)

Without R, the attacker cannot generate training data for the rotated space. The attacker could embed texts in the original space, but those embeddings don't match the rotated ones.

### 4.4 Known-Plaintext Attack (The Real Threat)

**Scenario:** The attacker knows both the original text and the corresponding rotated embedding for n documents.

**Mathematical reality:** This is the Orthogonal Procrustes Problem. Given:
- A = matrix of original embeddings (attacker computes these from known texts)
- B = matrix of rotated embeddings (attacker observes these in the database)

The attacker solves: minimize ||R*A - B||_F subject to R^T R = I

**Solution:** SVD of A * B^T = U Sigma V^T, then R_recovered = V * U^T

**How many pairs needed:** Theoretically, d linearly independent pairs fully determine the d x d rotation matrix. For 384-dim embeddings (all-MiniLM-L6-v2), that's 384 known pairs. For 1536-dim (OpenAI), that's 1,536 known pairs.

**In practice:**
- With n < d pairs: the attacker recovers a partial rotation (projects onto the subspace spanned by the known pairs). Attack partially succeeds in that subspace.
- With n >= d linearly independent pairs: the attacker recovers R exactly. Game over. All stored embeddings can be un-rotated and fed to standard inversion models.
- The pairs must be (original_text -> rotated_embedding) mappings. The attacker needs to know which specific text maps to which specific stored embedding.

**This is the Hill Cipher analogy.** Pure orthogonal rotation is a linear cipher. Linear ciphers are vulnerable to known-plaintext attacks. This is well-established in cryptography.

### 4.5 Mitigations for the Known-Plaintext Threat

| Mitigation | Effectiveness | Trade-off |
|-----------|--------------|-----------|
| Keep the rotation matrix secret (key management) | Essential baseline | Operational complexity |
| Rotate per-tenant (different R per user/org) | Limits blast radius | More keys to manage |
| Add calibrated noise after rotation (DCPE approach) | Blocks exact Procrustes recovery | Approximate (not exact) similarity; small quality loss |
| Periodically re-rotate with new matrix | Limits window of vulnerability | Requires re-encrypting all stored embeddings |
| Don't expose raw embeddings (API-only access) | Eliminates direct embedding theft | Limits use cases |

### 4.6 What Rotation Does NOT Defeat

1. **Attacks on the retrieval pipeline** (membership inference via RAG output analysis)
2. **Side-channel attacks** (timing, access patterns revealing query/document relationships)
3. **Compromised key** (if R leaks, all protections vanish instantly)
4. **Known-plaintext with sufficient pairs** (>= d pairs fully recovers R)
5. **Attacks by the data owner** (the owner knows R; rotation protects against third-party storage providers, not self)

---

## 5. Quantitative Security Claims

### 5.1 Claims We CAN Make (Honestly)

**Claim 1: Rotation provides exact-zero-loss privacy against honest-but-curious vector database providers.**

A storage provider who sees only the rotated embeddings cannot:
- Run Vec2Text, ZSinvert, Zero2Text, or any published inversion attack
- Train attribute classifiers (age, sex, medical conditions)
- Perform direct membership inference
- Determine the original embedding model used

This holds as long as the provider has no access to original-space embeddings or the rotation matrix.

**Claim 2: Similarity search works identically on rotated embeddings.**

Inner product, cosine similarity, and Euclidean distance are all exactly preserved. Recall@k is 100% of unrotated performance. This is a mathematical identity, not an approximation.

**Claim 3: The rotation is computationally cheap.**

Matrix-vector multiplication: O(d^2) per embedding. For d=1536: ~2.4M multiply-adds. Negligible compared to the embedding model inference cost (tens of milliseconds). Batch rotation of 100K embeddings takes under 1 second on CPU.

**Claim 4: Rotation is strictly stronger than dimensionality reduction for privacy.**

Truncating dimensions (OpenAI's dimension shortening) loses information permanently and degrades search quality. Rotation preserves all information and loses zero search quality.

### 5.2 Claims We CANNOT Make

**Cannot claim: "Rotation provides cryptographic security."**

It doesn't. The rotation matrix is a single secret with no computational hardness assumption. If the matrix is recovered (via known-plaintext or key compromise), all past and future embeddings encrypted with that matrix are exposed. There is no forward secrecy.

**Cannot claim: "Rotation protects against attackers with known text-embedding pairs."**

With >= d known pairs, the rotation is fully recoverable via SVD (Procrustes). This is O(d^3) computation, trivial for d <= 3072.

**Cannot claim: "Rotation provides differential privacy guarantees."**

Rotation is deterministic and reversible. It does not satisfy epsilon-differential privacy. It provides no formal privacy budget.

**Cannot claim: "Rotation protects the data owner's privacy from themselves."**

The data owner holds the rotation matrix. Rotation protects against third-party infrastructure providers, not insiders.

### 5.3 Comparison with Alternative Defenses

| Defense | Quality Loss | Inversion Protection | Membership Protection | Known-Plaintext Resistant | Formal Guarantee |
|---------|-------------|---------------------|----------------------|--------------------------|-----------------|
| **Orthogonal rotation (ours)** | **Zero** | Yes (without key) | Yes (direct) | **No** (d pairs breaks it) | None (security-by-obscurity of key) |
| Gaussian noise (sigma=0.1) | ~2-5% recall drop | Partial (degrades inversion) | Partial | Yes (noise is non-deterministic) | Approximate DP possible |
| DCPE (IronCore Cloaked AI) | ~1-3% recall drop | Yes | Partial (approximate distances) | Partially (noise component) | RoR indistinguishability |
| Dimensionality reduction (PCA) | 5-15% recall drop | Weak (lower-dim still invertible) | Weak | N/A | None |
| Concept-aware elliptical noise | ~1-3% recall drop | Yes (targeted at PII dimensions) | Partial | Yes | Concept-specific DP |
| Full homomorphic encryption | Zero (exact) | Yes | Yes | Yes | Cryptographic |
| Don't store embeddings at all | N/A | N/A | N/A | N/A | Perfect |

### 5.4 Recommended Framing for Documentation

**Do say:**
- "Zero-cost privacy layer for embeddings stored with third-party providers"
- "Preserves all distance metrics exactly. No quality trade-off."
- "Defeats all published embedding inversion attacks when the rotation key is secret"
- "Threat model: honest-but-curious storage provider without access to original embeddings"

**Don't say:**
- "Encrypted embeddings" (rotation is not encryption in the cryptographic sense)
- "Unbreakable" or "secure" without qualification
- "Privacy-preserving" without specifying the threat model
- "Differential privacy" (rotation provides none)

**Accurate one-liner:**
> "Rotate your embeddings with a secret orthogonal matrix before storing them. Search works identically. Published inversion attacks fail completely. The catch: if an attacker gets d original-text-to-rotated-embedding pairs, they can recover the rotation matrix via SVD."

---

## 6. Threat Model Summary

### Actors

| Actor | Access | Goal |
|-------|--------|------|
| **Honest-but-curious DB provider** | Sees rotated embeddings only | Reconstruct text, infer attributes, sell data |
| **External attacker (DB breach)** | Steals rotated embeddings | Same as above |
| **Attacker with partial knowledge** | Has some original texts + knows they're in the DB | Recover rotation matrix, then invert all embeddings |
| **Insider (data owner employee)** | Has rotation matrix + DB access | Full access (rotation provides no protection) |

### Attack-Defense Matrix

| Attack | No Defense | Rotation Only | Rotation + Noise | Rotation + Per-Tenant Keys |
|--------|-----------|---------------|------------------|---------------------------|
| Vec2Text (white-box) | **BROKEN** (92% at 32 tok) | **SAFE** | **SAFE** | **SAFE** |
| Vec2Text (on rotated, no key) | N/A | **SAFE** | **SAFE** | **SAFE** |
| ZSinvert / Zero2Text (universal) | **BROKEN** (partial) | **SAFE** | **SAFE** | **SAFE** |
| ALGEN (few-shot, 1K pairs from original space) | **BROKEN** | **SAFE** | **SAFE** | **SAFE** |
| Attribute classifier (trained on original space) | **BROKEN** (80-99%) | **SAFE** | **SAFE** | **SAFE** |
| Direct membership inference | **BROKEN** | **SAFE** | **SAFE** | **SAFE** |
| Known-plaintext (< d pairs) | **BROKEN** | **PARTIAL** (subspace leak) | **SAFE** (noise blocks Procrustes) | **SAFE** (per-tenant limits pairs) |
| Known-plaintext (>= d pairs) | **BROKEN** | **BROKEN** (R fully recovered) | **PARTIAL** (approximate R recovery) | **SAFE** if pairs < d per tenant |
| RAG output membership inference | **BROKEN** | **NOT MITIGATED** | **NOT MITIGATED** | **NOT MITIGATED** |
| Key compromise | N/A | **BROKEN** (all embeddings exposed) | **PARTIAL** (noise still helps) | **CONTAINED** (only one tenant) |
| Timing side-channel | **BROKEN** | **NOT MITIGATED** | **NOT MITIGATED** | **NOT MITIGATED** |

### Bottom Line

Orthogonal rotation is an excellent first layer of defense with a genuinely unique property: **zero quality loss.** No other defense can make this claim. It completely defeats all published embedding inversion attacks under the honest-but-curious threat model. Its weakness is the known-plaintext scenario, which is addressable by adding a small noise component (at the cost of approximate rather than exact similarity). For the target use case (protecting embeddings stored in third-party vector databases like Pinecone, Weaviate, Qdrant), rotation alone provides meaningful and practical privacy.

---

## Sources

- [Vec2Text: Text Embeddings Reveal (Almost) As Much As Text (Morris et al., EMNLP 2023)](https://github.com/vec2text/vec2text)
- [ALGEN: Few-shot Inversion Attacks on Textual Embeddings (Chen et al., ACL 2025)](https://aclanthology.org/2025.acl-long.1185/)
- [ZSinvert: Universal Zero-shot Embedding Inversion (Zhang & Morris, 2025)](https://arxiv.org/abs/2504.00147)
- [Zero2Text: Zero-Training Cross-Domain Inversion Attacks (2026)](https://arxiv.org/abs/2602.01757)
- [Transferable Embedding Inversion Attack (Huang et al., ACL 2024)](https://aclanthology.org/2024.acl-long.230/)
- [Concept-Aware Privacy Mechanisms for Defending Embedding Inversion (ICLR 2026)](https://arxiv.org/html/2602.07090)
- [TextCrafter: Optimization-Calibrated Noise for Defending Against Text Embedding Inversion](https://arxiv.org/html/2509.17302)
- [Mitigating Privacy Risks in LLM Embeddings from Embedding Inversion](https://arxiv.org/html/2411.05034v1)
- [Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](https://arxiv.org/html/2402.12784)
- [Membership Inference Attacks Against RAG Systems](https://arxiv.org/abs/2405.20446)
- [IronCore Labs Cloaked AI (DCPE approach)](https://ironcorelabs.com/products/cloaked-ai/)
- [Approximate Distance-Comparison-Preserving Symmetric Encryption](https://eprint.iacr.org/2021/1666.pdf)
- [Orthogonal Procrustes Problem (Wikipedia)](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
- [IronCore Labs: Text Embedding Privacy Risks](https://ironcorelabs.com/blog/2024/text-embedding-privacy-risks/)
- [Rethinking the Privacy of Text Embeddings: A Reproducibility Study](https://arxiv.org/html/2507.07700v1)
- [Embeddings, Privacy, and the Leak Auditor](https://medium.com/@yassien/embeddings-privacy-and-the-leak-auditor-auditing-the-hidden-memory-of-ai-2e7c78339ad9)

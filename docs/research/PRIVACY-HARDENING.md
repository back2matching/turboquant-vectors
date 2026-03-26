# Privacy Hardening Research

> Research conducted 2026-03-25 by specialized agent. Findings inform EXECPLAN.md.

---

## 1. SPARSE / Concept-Aware Privacy (ICLR 2026, arXiv:2602.07090)

### How It Works

SPARSE solves the fundamental problem of uniform noise in differential privacy: instead of adding the same noise to every embedding dimension, it identifies which dimensions encode privacy-sensitive concepts and applies disproportionate noise there.

**Stage 1: Differentiable Mask Learning.** The framework learns a binary mask `m` over embedding dimensions using Hard Concrete distributions:

```
s_i = sigma((1/beta_i) * (log(mu_i/(1-mu_i)) + log(alpha_i)))
m_i = min(1, max(0, s_i * (1.1 - (-0.1)) + (-0.1)))
```

The training objective jointly optimizes a classifier (2-layer MLP, 256+128 hidden units) and the mask:

```
min_{m,theta} L_cls(m, theta) + lambda * L_reg(m)
```

L_cls trains the MLP to distinguish embeddings containing the target privacy concept (e.g., "person name") from those that don't, while L_reg is an L0 sparsity regularizer. Dimensions that the classifier needs to succeed are exactly the privacy-sensitive dimensions.

**Stage 2: Mahalanobis Mechanism.** Instead of spherical Gaussian noise, inject elliptical noise calibrated by the learned mask:

```
M_Mah(x) = x + Z_Mah,   where Z_Mah ~ exp(-epsilon * ||z||_M)
||v||_M = sqrt(v^T Sigma^{-1} v),   Sigma = diag(m_1 + delta, ..., m_n + delta)
```

Privacy-sensitive dimensions (high mask value) get large noise; non-sensitive dimensions get minimal noise. Theorem 1 proves epsilon*d-LDP with respect to the Mahalanobis norm.

### Results

| Setting | Before | After SPARSE |
|---------|--------|-------------|
| STS12 leakage (epsilon=10) | 60% | 19% |
| Vec2Text attack (epsilon=5) | baseline | 92% reduction |
| MIMIC-III sex attribute (epsilon=10) | 88% | 28% |

Tested on GTR-base, Sentence-T5, SBERT across 6 datasets + MIMIC-III. The paper claims 46.2% mitigation for semantically similar tokens even when only the target concept is protected.

### Recommendation: Add SPARSE on Top of Rotation

Pipeline: `embed -> rotate (lossless) -> SPARSE noise (concept-specific DP)`. This would be offered as a "strict mode". At epsilon=10, task consistency is approximately 65%, but most of that loss is in dimensions irrelevant to retrieval, so actual recall@10 impact would be smaller.

We could ship pre-trained masks for common concepts (PII, medical, financial) on common embedding models. The mask is a trivially small artifact (a vector of d floats). The noise injection is a single vector addition.

**Feasibility: HIGH. Impact: HIGH.**

---

## 2. DCPE Formal Security Analysis (Fuchsbauer et al. 2021)

### The SAP Scheme

Scale-And-Perturb encryption:
1. Scale each element of vector x by a secret factor s derived from the key
2. Add a pseudorandom perturbation vector p, each element uniform within a range determined by the approximation factor beta
3. Perturbation sampled from an n-dimensional sphere centered on the scaled vector

### Formal Security Properties

SAP achieves **Real-or-Replaced (RoR) indistinguishability**: the adversary cannot distinguish between encryptions of actual plaintexts and encryptions of random replacements from the same distribution. The authors proved that the stronger Left-or-Right (LoR) IND-CPA is NOT achievable for approximate DCP with practical approximation factors.

RoR implies:
- Membership inference resistance (can't tell if a specific vector is in the database)
- For i.i.d. multivariate normal plaintexts, security against approximate frequency-finding attacks
- Better bit-security than Order-Preserving Encryption

**Key difference from pure rotation:**

| Property | Rotation | DCPE/SAP |
|----------|----------|----------|
| Formal security game | IND-ROT (single-vector perfect) | RoR (multi-vector, established PPE framework) |
| Membership inference resistance | No (exact distances preserved) | Yes (perturbation fuzzes distances) |
| Known-plaintext resistance | No (d pairs recovers key) | Partial (noise disrupts exact Procrustes) |
| Distance preservation | Exact | Approximate (within beta factor) |
| Recall loss | Zero | 1-15% depending on beta |

DCPE is also weak against chosen-plaintext attacks (attacker who can encrypt arbitrary plaintexts under the key), but IronCore mitigates this operationally by keeping keys in KMS/HSMs.

### Can We Adopt the Framework?

We cannot achieve RoR indistinguishability AND zero recall loss simultaneously -- the perturbation is what provides the formal property. But we can adopt a **two-tier model**:

- **Standard mode** (current): Pure rotation. IND-ROT. Zero recall loss.
- **Strict mode**: Rotation + calibrated perturbation. RoR-like indistinguishability. 1-3% recall loss. Resistant to partial known-plaintext.

**Feasibility: MEDIUM. Impact: HIGH for compliance storytelling.**

---

## 3. Quantization as Privacy Defense

### RecSys 2025 Reproducibility Results (arXiv:2507.07700)

8-bit quantization dramatically degrades Vec2Text:

| Dataset | BLEU (original) | BLEU (8-bit quant) | Drop |
|---------|-----------------|---------------------|------|
| ArguAna | 59.1 | 19.0 | 68% |
| FiQA | 38.5 | 16.0 | 58% |
| NFCorpus | 63.1 | 20.7 | 67% |
| SciFact | 56.6 | 24.2 | 57% |

Retrieval quality (nDCG@10) was essentially unchanged. Both Absolute Maximum and Zeropoint quantization performed identically. The authors recommend quantization as "lightweight, hyperparameter-free" vs. Gaussian noise which requires tuning.

### Critical Finding: Deterministic Quantization is NOT DP

From arXiv:2306.11913: "a value of x would always map to the fixed set of two quantization levels deterministically. This immediately breaks differential privacy." TurboQuant's current argmin quantization does not provide DP.

However, **randomized quantization** (stochastic rounding) provides formal Renyi DP. The Randomized Quantization Mechanism achieves (alpha, epsilon)-Renyi DP through random sub-sampling of quantization bins + randomized rounding. Privacy epsilon scales linearly with the number of quantization levels (fewer bits = more privacy).

### Recommendation

Add a `stochastic=True` parameter to `rotate_and_compress()`. Instead of `dists.argmin(axis=2)`, probabilistically select between the two nearest centroids proportional to inverse distance. This gives:
- A legitimate Renyi DP claim
- Only marginal additional recall loss vs. deterministic
- A real epsilon to quote in compliance documents

**Feasibility: HIGH (one-line code change in the quantization loop). Impact: MEDIUM-HIGH.**

---

## 4. Known-Plaintext Attack Mitigation

The Procrustes attack (d linearly independent pairs recovers Q via SVD) is the primary theoretical weakness.

### Ranked Strategies

**1. Per-tenant keys (already supported).** Different rotation matrix per organization. Limits blast radius. An attacker needs d pairs from a single tenant. **Impact: HIGH. Effort: Documentation only.**

**2. Key rotation with `rekey_vectors()` (already implemented).** Generate new Q periodically. Old known-plaintext pairs become useless. The existing `rekey_vectors()` does the transform in one step without materializing unrotated vectors. **Impact: HIGH. Effort: Operational docs.**

**3. Optional Gaussian noise injection.** After rotation, add `eta ~ N(0, sigma^2 I)`. This turns the Procrustes system from `B = QA` to `B = QA + E`. The SVD recovery error is approximately `||Q_hat - Q||_F ~ O(sigma * sqrt(d) / sigma_min(A))`. At sigma=0.003 on d=1536, recall loss is <1% but Procrustes recovery is significantly degraded with fewer than ~10*d pairs. **Impact: MEDIUM-HIGH. Effort: Easy code change.**

**4. SPARSE concept-aware noise (described in section 1).** Smarter version of strategy 3 -- noise only on privacy-sensitive dimensions. **Impact: HIGH. Effort: Medium.**

**5. Non-linear component.** Adding a secret permutation or per-coordinate scaling would break the linearity that enables Procrustes, but it also breaks exact distance preservation. Not recommended for the standard mode. **Impact: LOW (breaks core value proposition).**

---

## 5. Formal Privacy Definitions

### IND-ROT (Already Defined in Spec)

For any two vectors x_0, x_1 with equal norms, given x' = Qx_b for random bit b and uniform Q from Haar measure on O(d), no adversary can determine b better than 1/2. The advantage is exactly 0. This is information-theoretically perfect for single vectors.

### Multi-Vector Leakage: The Gram Matrix

For n vectors under the same Q, the server can compute all pairwise inner products `<x'_i, x'_j> = <x_i, x_j>`. The leaked information is exactly the Gram matrix. Formally:

```
I({x_1,...,x_n}; {x'_1,...,x'_n}) = H(Gram matrix)
```

This reveals clustering structure, outliers, and data manifold dimensionality. It does NOT reveal individual vector content or which embedding model was used. For n=100K vectors in d=1536, the server learns ~5 billion pairwise relationships, which is structurally significant but does not enable reconstruction of individual vectors.

### Recommendation

Formalize IND-ROT-MULTI: "The adversary's advantage in distinguishing databases D_0 and D_1 is bounded by the statistical distance between their Gram matrices. If Gram(D_0) = Gram(D_1), advantage is exactly 0." This gives compliance officers a precise statement: "The server learns pairwise similarities and nothing else."

**Feasibility: HIGH (analysis/documentation only). Impact: HIGH for credibility.**

---

## 6. Regulatory Compliance

### GDPR

**Rotated embeddings are still personal data** (pseudonymised, not anonymised). Under GDPR Article 4(5), rotation is pseudonymisation: data can't be attributed to a subject without the key, provided the key is kept separately. The EDPB Guidelines 01/2025 on Pseudonymisation (adopted January 2025) confirm pseudonymised data remains personal data.

However, pseudonymisation is explicitly valuable:
- Article 32(1)(a) lists it as a technical safeguard
- Article 34(3)(a): if breached data is pseudonymised and the key wasn't compromised, the controller may not need to notify data subjects
- Recital 28 encourages pseudonymisation to reduce risks

**Key compliance argument:** "Rotated embeddings stored with a third-party vector database constitute pseudonymised data under GDPR Article 4(5). The rotation key is kept separately under strict access controls. Without the key, published embedding inversion attacks fail completely. This constitutes a technical safeguard under Article 32 and may reduce breach notification obligations under Article 34(3)(a)."

### HIPAA

Rotation qualifies as a **technical safeguard** under 45 CFR 164.312 (access control: the key controls access to original embeddings). Rotated embeddings are NOT de-identified under Safe Harbor (doesn't remove 18 identifiers) or Expert Determination. They remain PHI with a technical safeguard applied.

An Expert Determination under 164.514(b)(1) could potentially certify "very small" re-identification risk if the expert is satisfied that the key is secured and the known-plaintext attack requires infeasible pairs in the operational context.

### What Compliance Officers Need

1. Documentation that rotation key is stored separately from rotated embeddings
2. Key management policy (access control, rotation schedule, incident response)
3. Business Associate Agreement with the vector DB provider that does NOT give them key access
4. Demonstration that Vec2Text fails on rotated embeddings (existing `inversion_demo.py`)
5. Formal statement of what the server learns (Gram matrix only)
6. DPIA covering residual risk

---

## Top Priority Implementation Actions

| # | Action | Recall Cost | Effort | Why |
|---|--------|-------------|--------|-----|
| 1 | Stochastic quantization option | ~0.5% | 1 day | Formal Renyi DP claim from a one-line change |
| 2 | Gaussian noise option (sigma param on rotate) | 0.5-2% | 1 day | Degrades Procrustes attack significantly |
| 3 | Key rotation operational docs | 0% | 2 days | `rekey_vectors()` exists but is undocumented operationally |
| 4 | IND-ROT-MULTI formalization | 0% | 2 days | Precise statement for compliance docs |
| 5 | GDPR/HIPAA compliance templates | 0% | 3 days | Ready-made documents for regulated users |
| 6 | SPARSE concept masks (strict mode) | 1-5% | 2 weeks | Best-in-class targeted privacy with minimal utility loss |

## Sources

- [SPARSE: Concept-Aware Privacy (ICLR 2026)](https://arxiv.org/abs/2602.07090)
- [SPARSE Full Paper](https://arxiv.org/html/2602.07090)
- [Fuchsbauer et al. DCPE](https://eprint.iacr.org/2021/1666)
- [Fuchsbauer et al. DCPE (Springer)](https://link.springer.com/chapter/10.1007/978-3-031-14791-3_6)
- [Rethinking Privacy of Text Embeddings (RecSys 2025)](https://arxiv.org/abs/2507.07700)
- [Randomized Quantization for DP](https://arxiv.org/abs/2306.11913)
- [IronCore Labs Cloaked AI](https://ironcorelabs.com/products/cloaked-ai/)
- [IronCore Labs How It Works](https://ironcorelabs.com/docs/cloaked-ai/how-it-works/)
- [TextCrafter Defense](https://arxiv.org/abs/2509.17302)
- [Eguard](https://arxiv.org/abs/2411.05034)
- [Salty Embeddings](https://github.com/vkleban/salty-embeddings)
- [EDPB Guidelines 01/2025 on Pseudonymisation](https://www.edpb.europa.eu/our-work-tools/documents/public-consultations/2025/guidelines-012025-pseudonymisation_en)
- [EDPB Pseudonymisation Analysis](https://www.hunton.com/insights/publications/edpb-advises-on-pseudonymisation-for-gdpr-compliance)
- [HIPAA De-Identification Guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html)
- [IronCore Labs DCPE Chosen-Plaintext Weakness](https://ironcorelabs.com/blog/2025/nist-standards-ai-encryption/)
- [Securing Vector Databases (Cisco)](https://sec.cloudapps.cisco.com/security/center/resources/securing-vector-databases/)
- [IBM: Protecting Embeddings with DCPE](https://developer.ibm.com/articles/java-vector-embeddings-encryption/)

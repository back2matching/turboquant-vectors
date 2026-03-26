# turboquant-vectors Launch Plan

> Announce the PrivateEncoder privacy module. Published as 0.3.0. Ready to post.

**Package:** https://pypi.org/project/turboquant-vectors/0.3.0/
**GitHub:** https://github.com/back2matching/turboquant-vectors
**What's new:** `PrivateEncoder` — zero-loss embedding privacy via orthogonal rotation

---

## Launch Sequence

| Day | Platform | Action |
|-----|----------|--------|
| 1 | PyPI | Already published (0.3.0) |
| 1 | GitHub | README updated, pushed |
| 2 | r/LocalLLaMA | Main post (see REDDIT-POSTS.md) |
| 2 | r/MachineLearning | Academic angle post |
| 3 | Hacker News | "Show HN" submission |
| 3 | Twitter/X | Thread (7 tweets) |
| 5 | Dev.to / Hashnode | Blog post |
| 7+ | LangChain | Community integration PR (v1.1 scope) |

**Timing note:** TurboQuant blog post dropped March 24-25. Window of peak attention is 1-2 weeks. Post r/LocalLLaMA by March 27 latest.

---

## Key Assets Needed

- ⬜ Colab notebook (highest ROI asset — runnable proof beats written claims)
- ✅ README with honest threat model and integration examples
- ✅ 92 passing tests proving all claims
- ⬜ Reddit posts drafted (see REDDIT-POSTS.md)
- ⬜ HN submission text
- ⬜ Twitter thread

---

## Messaging (from positioning research)

**One-line pitch:** "Rotate your embeddings with a secret key. Search works identically. Inversion attacks fail."

**The hook:** Vec2Text recovers 92% of text from embeddings. OWASP made this LLM08. Here's a one-line fix with zero recall loss.

**Differentiation:**
- vs IronCore Cloaked AI: "They fuzz values AND distances (lossy). We rotate (lossless). Different tradeoffs."
- vs salty-embeddings: "They permute (reorder elements). We rotate (mix all dimensions). Strictly stronger."
- vs doing nothing: "Vec2Text, ALGEN, and Zero2Text can reconstruct your text. It's an OWASP top-10 risk."

**What to NEVER say:**
- "Encrypted embeddings" (it's not encryption)
- "Unbreakable" or "secure" without qualification
- "Privacy-preserving" without specifying the threat model

**Honest one-liner for the threat model:**
> "Rotate your embeddings with a secret orthogonal matrix before storing them. Search works identically. Published inversion attacks fail completely. The catch: if an attacker gets d original-to-rotated pairs, they can recover the rotation matrix via SVD."

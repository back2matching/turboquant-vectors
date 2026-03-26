"""
Adversarial Self-Test: Can Wasserstein-Procrustes Recover Our Rotation?

Tests whether an attacker who has a large corpus of unrotated embeddings
(from the same model, different texts) can recover the rotation matrix
using unsupervised distribution alignment — WITHOUT any known pairs.

This is the attack described in Conneau et al. 2018 / Grave et al. 2019
for cross-lingual word embedding alignment.

We run this attack on ourselves and honestly document the results.
"""

import numpy as np
import time
from turboquant_vectors import PrivateEncoder


def wasserstein_procrustes_attack(
    source: np.ndarray,  # Attacker's reference embeddings (unrotated, different texts)
    target: np.ndarray,  # Observed rotated embeddings in the database
    n_iter: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Wasserstein-Procrustes unsupervised alignment attack.

    Attempts to recover the rotation matrix Q such that target ≈ source @ Q
    using only distributional alignment (no matched pairs).

    Based on Conneau et al. (2018) "Word Translation Without Parallel Data"
    and Grave et al. (2019) "Unsupervised Alignment of Embeddings with
    Wasserstein Procrustes."

    The algorithm alternates between:
    1. Nearest-neighbor matching (find correspondences)
    2. Procrustes alignment (find best rotation given correspondences)

    Returns:
        Estimated rotation matrix Q_hat, shape (dim, dim)
    """
    rng = np.random.default_rng(seed)
    dim = source.shape[1]

    # Normalize both sets
    src_norms = np.linalg.norm(source, axis=1, keepdims=True)
    tgt_norms = np.linalg.norm(target, axis=1, keepdims=True)
    src = source / np.maximum(src_norms, 1e-10)
    tgt = target / np.maximum(tgt_norms, 1e-10)

    # Initialize with a random orthogonal matrix
    G = rng.standard_normal((dim, dim))
    Q_hat, _ = np.linalg.qr(G)
    Q_hat = Q_hat.astype(np.float32)

    for iteration in range(n_iter):
        # Step 1: Find nearest-neighbor correspondences
        # For each source vector, find its nearest neighbor in (target @ Q_hat^T)
        # This is equivalent to: for each src, find argmax(src @ Q_hat @ tgt^T)
        transformed_src = src @ Q_hat  # (n_src, dim)
        similarities = transformed_src @ tgt.T  # (n_src, n_tgt)

        # Greedy matching (not optimal transport, but much faster)
        src_to_tgt = similarities.argmax(axis=1)  # For each src, best tgt

        # Step 2: Procrustes alignment on matched pairs
        matched_src = src
        matched_tgt = tgt[src_to_tgt]

        # Solve: Q_hat = argmin ||matched_src @ Q - matched_tgt||
        # Solution: SVD of matched_tgt^T @ matched_src
        M = matched_tgt.T @ matched_src  # (dim, dim)
        U, _, Vt = np.linalg.svd(M)
        Q_hat = (U @ Vt).astype(np.float32)

    return Q_hat


def measure_attack_quality(Q_true, Q_estimated, test_embeddings):
    """
    Measure how well the estimated rotation matches the true one.

    Returns multiple metrics:
    - frobenius_error: ||Q_true - Q_estimated||_F (0 = perfect recovery)
    - cosine_recovery: avg cosine sim between true and estimated rotations of test vectors
    - recall_overlap: what fraction of top-K results match between true and estimated
    """
    dim = Q_true.shape[0]

    # Frobenius error
    frob_error = np.linalg.norm(Q_true - Q_estimated, 'fro')
    max_frob = np.sqrt(2 * dim)  # Max possible for two orthogonal matrices

    # Cosine recovery: how close are the estimated rotations to the true ones?
    true_rot = test_embeddings @ Q_true.T
    est_rot = test_embeddings @ Q_estimated.T

    cosines = []
    for i in range(len(test_embeddings)):
        cos = np.dot(true_rot[i], est_rot[i]) / (
            np.linalg.norm(true_rot[i]) * np.linalg.norm(est_rot[i]) + 1e-10
        )
        cosines.append(cos)
    avg_cosine = np.mean(cosines)

    # Recall overlap: if attacker uses Q_estimated to "unrotate" and search
    # vs using Q_true, how many top-10 results overlap?
    corpus = true_rot  # The actual rotated corpus
    n_queries = min(50, len(test_embeddings))
    k = 10
    recalls = []
    for i in range(n_queries):
        # True query rotation
        true_query = true_rot[i]
        true_scores = corpus @ true_query
        true_topk = set(np.argsort(-true_scores)[:k])

        # Attacker's estimated query rotation
        est_query = est_rot[i]
        est_scores = corpus @ est_query
        est_topk = set(np.argsort(-est_scores)[:k])

        recalls.append(len(true_topk & est_topk) / k)

    return {
        "frobenius_error": frob_error,
        "frobenius_normalized": frob_error / max_frob,
        "avg_cosine_recovery": avg_cosine,
        "recall_overlap": np.mean(recalls),
    }


def run_adversarial_test():
    """Run the full adversarial self-test."""
    print("=" * 70)
    print("ADVERSARIAL SELF-TEST: Wasserstein-Procrustes Attack")
    print("=" * 70)
    print()
    print("Can an attacker recover our rotation matrix WITHOUT any known pairs?")
    print("Using only: (1) rotated embeddings from our DB, (2) their own unrotated")
    print("embeddings from the same model on different texts.")
    print()

    # Generate two sets of embeddings (simulating attacker having their own corpus)
    dim = 384  # MiniLM-L6-v2 dimension (fast)
    rng = np.random.default_rng(42)

    # Simulate real embedding distribution (clustered, not uniform)
    n_clusters = 20
    n_per_cluster = 100
    n_total = n_clusters * n_per_cluster

    def make_clustered_embeddings(seed):
        r = np.random.default_rng(seed)
        centers = r.standard_normal((n_clusters, dim)).astype(np.float32)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        vecs = []
        for c in centers:
            pts = c + r.standard_normal((n_per_cluster, dim)).astype(np.float32) * 0.2
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            vecs.append(pts)
        return np.vstack(vecs)

    # Defender's embeddings (will be rotated and stored)
    defender_embeddings = make_clustered_embeddings(seed=100)

    # Attacker's embeddings (different texts, same model, not rotated)
    attacker_embeddings = make_clustered_embeddings(seed=200)

    # Test embeddings (for measuring quality)
    test_embeddings = make_clustered_embeddings(seed=300)

    # Create secret rotation
    encoder = PrivateEncoder.generate(dim=dim, normalize=False)
    Q_true = encoder._rotation

    # Rotate defender's embeddings (this is what the attacker sees in the DB)
    rotated_defender = encoder.rotate(defender_embeddings)

    print(f"Setup:")
    print(f"  Dimension: {dim}")
    print(f"  Defender corpus: {n_total} vectors (rotated, stored in DB)")
    print(f"  Attacker corpus: {n_total} vectors (unrotated, their own texts)")
    print(f"  Test corpus: {n_total} vectors")
    print()

    # Run attack at different sample sizes
    sample_sizes = [100, 500, 1000, 2000]

    print(f"{'Samples':>8} {'Frob Err':>10} {'Frob Norm':>10} {'Cos Recovery':>13} {'Recall Overlap':>15} {'Time':>8}")
    print("-" * 70)

    for n_samples in sample_sizes:
        t0 = time.perf_counter()

        Q_estimated = wasserstein_procrustes_attack(
            source=attacker_embeddings[:n_samples],
            target=rotated_defender[:n_samples],
            n_iter=30,
        )

        elapsed = time.perf_counter() - t0

        metrics = measure_attack_quality(Q_true, Q_estimated, test_embeddings[:200])

        print(f"{n_samples:>8} {metrics['frobenius_error']:>10.2f} "
              f"{metrics['frobenius_normalized']:>10.4f} "
              f"{metrics['avg_cosine_recovery']:>13.4f} "
              f"{metrics['recall_overlap']:>15.3f} "
              f"{elapsed:>7.1f}s")

    print()

    # Baseline: random rotation (no attack at all)
    Q_random, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    Q_random = Q_random.astype(np.float32)
    baseline = measure_attack_quality(Q_true, Q_random, test_embeddings[:200])

    print(f"Baselines:")
    print(f"  Random guess:    Frob={baseline['frobenius_error']:.2f}, "
          f"Cos={baseline['avg_cosine_recovery']:.4f}, "
          f"Recall={baseline['recall_overlap']:.3f}")

    # Perfect recovery
    perfect = measure_attack_quality(Q_true, Q_true, test_embeddings[:200])
    print(f"  Perfect recovery: Frob={perfect['frobenius_error']:.2f}, "
          f"Cos={perfect['avg_cosine_recovery']:.4f}, "
          f"Recall={perfect['recall_overlap']:.3f}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("If 'Cos Recovery' is close to 1.0 and 'Recall Overlap' is high,")
    print("the attack is succeeding — the attacker can partially recover")
    print("the rotation and use it to break privacy.")
    print()
    print("If values are close to the 'Random guess' baseline, the attack")
    print("is failing — rotation provides meaningful privacy.")


if __name__ == "__main__":
    run_adversarial_test()

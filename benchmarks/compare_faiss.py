"""
Reproducible benchmark: TurboQuant vs FAISS Product Quantization.

Compares recall@K at MATCHED bit budgets (same bytes per vector).
Uses synthetic clustered data with fixed seeds for full reproducibility.

Usage:
    python benchmarks/compare_faiss.py
    python benchmarks/compare_faiss.py --n 50000 --dim 1536 --top-k 10

Requirements:
    pip install turboquant-vectors faiss-cpu
"""

import argparse
import time
import numpy as np
import sys

try:
    import faiss
except ImportError:
    print("faiss-cpu required: pip install faiss-cpu")
    sys.exit(1)

from turboquant_vectors import compress, search


def generate_clustered_data(n: int, dim: int, n_clusters: int = 50,
                            noise: float = 0.3, seed: int = 42):
    """Generate reproducible clustered data (simulates real embeddings)."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    labels = rng.randint(0, n_clusters, n)
    vectors = centers[labels] + rng.randn(n, dim).astype(np.float32) * noise
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)
    return vectors


def generate_queries(vectors: np.ndarray, n_queries: int = 100,
                     noise: float = 0.05, seed: int = 99):
    """Generate queries near existing vectors (simulates realistic queries)."""
    rng = np.random.RandomState(seed)
    base_indices = rng.choice(len(vectors), n_queries, replace=False)
    queries = vectors[base_indices] + rng.randn(n_queries, vectors.shape[1]).astype(np.float32) * noise
    queries = (queries / np.linalg.norm(queries, axis=1, keepdims=True)).astype(np.float32)
    return queries


def exact_search(vectors: np.ndarray, queries: np.ndarray, top_k: int):
    """Brute-force exact cosine similarity search (ground truth)."""
    scores = queries @ vectors.T
    top_indices = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
    for i in range(len(top_indices)):
        order = np.argsort(-scores[i, top_indices[i]])
        top_indices[i] = top_indices[i, order]
    return top_indices


def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Mean recall@K across all queries."""
    recalls = []
    for pred_row, gt_row in zip(predicted, ground_truth):
        pred_set = set(pred_row[:k].tolist())
        gt_set = set(gt_row[:k].tolist())
        recalls.append(len(pred_set & gt_set) / k)
    return np.mean(recalls)


def benchmark_turboquant(vectors, queries, bits, top_k):
    """Run TurboQuant compression + search, return recall and timing."""
    from turboquant_vectors.core import TurboQuantVectors

    dim = vectors.shape[1]
    t0 = time.time()
    tq = TurboQuantVectors(dim=dim, bits=bits)
    compressed = tq.compress(vectors)
    compress_time = time.time() - t0

    bytes_per_vec = (dim * bits + 7) // 8

    # Batch search: reuse TQ instance so decompression cache works
    t0 = time.time()
    all_indices, all_scores = tq.search(compressed, queries, top_k=top_k)
    search_time = time.time() - t0

    return np.atleast_2d(all_indices), bytes_per_vec, compress_time, search_time


def benchmark_faiss_pq(vectors, queries, m, top_k, nbits=8):
    """Run FAISS Product Quantization, return recall and timing."""
    dim = vectors.shape[1]

    t0 = time.time()
    index = faiss.IndexPQ(dim, m, nbits)
    index.train(vectors)
    index.add(vectors)
    compress_time = time.time() - t0

    bytes_per_vec = m * nbits // 8  # m subvectors * nbits per code

    t0 = time.time()
    scores, indices = index.search(queries, top_k)
    search_time = time.time() - t0

    return indices, bytes_per_vec, compress_time, search_time


def main():
    parser = argparse.ArgumentParser(description="TurboQuant vs FAISS PQ benchmark")
    parser.add_argument("--n", type=int, default=10000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimension")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for recall")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    n, dim, n_queries, top_k = args.n, args.dim, args.n_queries, args.top_k

    print(f"Benchmark: TurboQuant vs FAISS PQ")
    print(f"  Vectors: {n:,} x {dim}-dim (float32)")
    print(f"  Queries: {n_queries}")
    print(f"  Metric: Recall@{top_k}")
    print(f"  Seed: {args.seed}")
    print()

    # Generate data
    print("Generating data...")
    vectors = generate_clustered_data(n, dim, seed=args.seed)
    queries = generate_queries(vectors, n_queries, seed=args.seed + 57)

    # Ground truth
    print("Computing ground truth (exact search)...")
    gt = exact_search(vectors, queries, top_k)

    # Storage baselines
    float32_bytes = n * dim * 4
    print(f"  Float32 storage: {float32_bytes / 1e6:.1f} MB")
    print()

    # Define matched configurations
    # Each config: (label, TQ bits, FAISS m) where both give same bytes/vector
    configs = []

    # At 1-bit: TQ=1bit (dim/8 bytes), FAISS m=dim/8
    m_1bit = dim // 8
    if dim % m_1bit == 0:
        configs.append(("1-bit equivalent", 1, m_1bit))

    # At 2-bit: TQ=2bit (dim/4 bytes), FAISS m=dim/4
    m_2bit = dim // 4
    if dim % m_2bit == 0:
        configs.append(("2-bit equivalent", 2, m_2bit))

    # At 4-bit: TQ=4bit (dim/2 bytes), FAISS m=dim/2
    m_4bit = dim // 2
    if dim % m_4bit == 0:
        configs.append(("4-bit equivalent", 4, m_4bit))

    # Also test FAISS at its common settings for context
    common_faiss_m = [48, 96]

    # Run benchmarks
    results = []

    print("=" * 85)
    print(f"{'Config':<22} {'Method':<20} {'Bytes/vec':<12} {'Recall@'+str(top_k):<12} {'Compress':<10} {'Search':<10}")
    print("=" * 85)

    # Matched-budget comparisons
    for label, tq_bits, faiss_m in configs:
        # TurboQuant
        tq_idx, tq_bpv, tq_ct, tq_st = benchmark_turboquant(vectors, queries, tq_bits, top_k)
        tq_recall = recall_at_k(tq_idx, gt, top_k)

        # FAISS PQ
        pq_idx, pq_bpv, pq_ct, pq_st = benchmark_faiss_pq(vectors, queries, faiss_m, top_k)
        pq_recall = recall_at_k(pq_idx, gt, top_k)

        assert tq_bpv == pq_bpv, f"Byte mismatch: TQ={tq_bpv}, PQ={pq_bpv}"

        print(f"{label:<22} {'TurboQuant '+str(tq_bits)+'bit':<20} {tq_bpv:<12} {tq_recall*100:>8.1f}%    {tq_ct:>7.2f}s   {tq_st:>7.2f}s")
        print(f"{'':<22} {'FAISS PQ m='+str(faiss_m):<20} {pq_bpv:<12} {pq_recall*100:>8.1f}%    {pq_ct:>7.2f}s   {pq_st:>7.2f}s")

        delta = tq_recall - pq_recall
        winner = "TurboQuant" if delta > 0 else "FAISS PQ"
        print(f"{'':<22} {'-> '+winner+' by '+f'{abs(delta)*100:.1f}pp':<40}")
        print("-" * 85)

        results.append({
            "config": label,
            "tq_bits": tq_bits, "faiss_m": faiss_m,
            "bytes_per_vec": tq_bpv,
            "tq_recall": tq_recall, "pq_recall": pq_recall,
            "tq_compress_time": tq_ct, "pq_compress_time": pq_ct,
            "tq_search_time": tq_st, "pq_search_time": pq_st,
        })

    # FAISS at typical settings (for context)
    print()
    print("FAISS PQ at common settings (for context, NOT matched to TurboQuant):")
    print("-" * 85)
    for m in common_faiss_m:
        if dim % m != 0:
            continue
        pq_idx, pq_bpv, pq_ct, pq_st = benchmark_faiss_pq(vectors, queries, m, top_k)
        pq_recall = recall_at_k(pq_idx, gt, top_k)
        compression = float32_bytes / (n * pq_bpv)
        print(f"  FAISS PQ m={m:<4}  {pq_bpv:>4} bytes/vec  ({compression:>5.1f}x compression)  Recall@{top_k}: {pq_recall*100:.1f}%  [{pq_ct:.2f}s compress, {pq_st:.3f}s search]")

    # Summary
    print()
    print("=" * 85)
    print("SUMMARY: Recall@{} at matched storage budgets".format(top_k))
    print("=" * 85)
    print()
    print(f"{'Budget':<22} {'TurboQuant':<15} {'FAISS PQ':<15} {'Delta':<15} {'Winner':<15}")
    print("-" * 75)
    for r in results:
        delta = r["tq_recall"] - r["pq_recall"]
        winner = "TurboQuant" if delta > 0 else "FAISS PQ" if delta < 0 else "Tie"
        print(f"{r['config']:<22} {r['tq_recall']*100:>8.1f}%      {r['pq_recall']*100:>8.1f}%      {delta*100:>+7.1f}pp      {winner}")

    print()
    print(f"Note: TurboQuant requires NO training (data-oblivious). FAISS PQ runs k-means.")
    print(f"Reproduce: python benchmarks/compare_faiss.py --n {n} --dim {dim} --seed {args.seed}")


if __name__ == "__main__":
    main()

"""
Real-Data Compression Benchmark: TurboQuant vs FAISS PQ

Tests on real OpenAI text-embedding-3-small embeddings (1536-dim, 100K vectors)
from Qdrant's HuggingFace dataset.

Requirements:
    pip install turboquant-vectors datasets numpy faiss-cpu
"""

import time
import numpy as np
import sys


def load_real_embeddings(n_vectors=10000):
    """Load real OpenAI embeddings from HuggingFace."""
    print(f"Loading {n_vectors} real OpenAI embeddings from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "Qdrant/dbpedia-entities-openai3-text-embedding-3-small-1536-100K",
            split=f"train[:{n_vectors}]",
        )
        embeddings = np.array(ds["text-embedding-3-small-1536-embedding"], dtype=np.float32)
        print(f"Loaded: shape={embeddings.shape}, dtype={embeddings.dtype}")
        return embeddings
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        print("Falling back to synthetic OpenAI-like embeddings...")
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n_vectors, 1536)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        print(f"Generated: shape={embeddings.shape}")
        return embeddings


def benchmark_turboquant(embeddings, queries, ground_truth, bits, k=10):
    """Benchmark TurboQuant compression."""
    from turboquant_vectors import compress, search

    n, dim = embeddings.shape

    # Compress
    t0 = time.perf_counter()
    compressed = compress(embeddings, bits=bits)
    compress_time = time.perf_counter() - t0

    # Search
    recalls = []
    search_times = []
    for i in range(len(queries)):
        t0 = time.perf_counter()
        idx, scores = search(compressed, queries[i], top_k=k)
        search_times.append(time.perf_counter() - t0)
        recalls.append(len(set(idx) & set(ground_truth[i])) / k)

    return {
        "method": f"TurboQuant {bits}-bit",
        "bits": bits,
        "recall_at_k": np.mean(recalls),
        "compress_time_s": compress_time,
        "search_time_ms": np.median(search_times) * 1000,
        "memory_bytes": compressed.memory_bytes,
        "compression_ratio": compressed.compression_ratio,
    }


def benchmark_faiss_pq(embeddings, queries, ground_truth, bits, k=10):
    """Benchmark FAISS Product Quantization at matched byte budget."""
    try:
        import faiss
    except ImportError:
        return None

    n, dim = embeddings.shape

    # Match the byte budget: TurboQuant uses `bits` per dimension
    # FAISS PQ: m subquantizers, each with nbits=8 by default
    # Bytes per vector = m (one byte per subquantizer)
    # TurboQuant bytes per vector = dim * bits / 8
    tq_bytes = dim * bits / 8
    m = max(1, int(tq_bytes))  # Number of PQ subquantizers
    # m must divide dim
    while dim % m != 0 and m > 1:
        m -= 1

    t0 = time.perf_counter()
    index = faiss.IndexPQ(dim, m, 8)  # m subquantizers, 8 bits each
    index.train(embeddings)
    index.add(embeddings)
    compress_time = time.perf_counter() - t0

    recalls = []
    search_times = []
    for i in range(len(queries)):
        t0 = time.perf_counter()
        D, I = index.search(queries[i:i+1], k)
        search_times.append(time.perf_counter() - t0)
        recalls.append(len(set(I[0]) & set(ground_truth[i])) / k)

    bytes_per_vec = m  # 1 byte per subquantizer
    total_bytes = n * bytes_per_vec

    return {
        "method": f"FAISS PQ (m={m})",
        "bits": bits,
        "recall_at_k": np.mean(recalls),
        "compress_time_s": compress_time,
        "search_time_ms": np.median(search_times) * 1000,
        "memory_bytes": total_bytes,
        "compression_ratio": (n * dim * 4) / total_bytes,
    }


def compute_ground_truth(embeddings, queries, k=10):
    """Brute-force exact top-K for ground truth."""
    ground_truth = []
    for q in queries:
        scores = embeddings @ q
        top_k = np.argsort(-scores)[:k]
        ground_truth.append(top_k)
    return ground_truth


def run_benchmark():
    """Run the full benchmark suite."""
    # Parameters
    n_vectors = 10000  # Start smaller for speed, scale up later
    n_queries = 100
    k = 10

    print("=" * 70)
    print("REAL-DATA BENCHMARK: TurboQuant vs FAISS PQ")
    print(f"Dataset: Qdrant OpenAI text-embedding-3-small (1536-dim)")
    print(f"Vectors: {n_vectors}, Queries: {n_queries}, k={k}")
    print("=" * 70)
    print()

    # Load data
    embeddings = load_real_embeddings(n_vectors)
    dim = embeddings.shape[1]

    # Split queries
    rng = np.random.default_rng(42)
    query_idx = rng.choice(n_vectors, n_queries, replace=False)
    queries = embeddings[query_idx].copy()

    # Ground truth (brute-force on full-precision data)
    print("Computing ground truth (brute-force)...")
    gt = compute_ground_truth(embeddings, queries, k)
    print()

    # Run benchmarks
    results = []

    for bits in [2, 3, 4, 8]:
        print(f"--- {bits}-bit ---")

        # TurboQuant
        tq_result = benchmark_turboquant(embeddings, queries, gt, bits, k)
        results.append(tq_result)
        print(f"  TurboQuant: Recall@{k}={tq_result['recall_at_k']:.3f}, "
              f"ratio={tq_result['compression_ratio']:.1f}x, "
              f"compress={tq_result['compress_time_s']:.2f}s")

        # FAISS PQ
        faiss_result = benchmark_faiss_pq(embeddings, queries, gt, bits, k)
        if faiss_result:
            results.append(faiss_result)
            print(f"  FAISS PQ:   Recall@{k}={faiss_result['recall_at_k']:.3f}, "
                  f"ratio={faiss_result['compression_ratio']:.1f}x, "
                  f"compress={faiss_result['compress_time_s']:.2f}s")

            delta = tq_result['recall_at_k'] - faiss_result['recall_at_k']
            print(f"  Delta:      {delta:+.3f} ({'TQ wins' if delta > 0 else 'FAISS wins'})")
        else:
            print(f"  FAISS PQ:   (faiss-cpu not installed)")

        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<25} {'Recall@10':>10} {'Ratio':>8} {'Compress':>10} {'Search':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['method']:<25} {r['recall_at_k']:>10.3f} {r['compression_ratio']:>7.1f}x "
              f"{r['compress_time_s']:>9.2f}s {r['search_time_ms']:>8.1f}ms")

    return results


if __name__ == "__main__":
    results = run_benchmark()

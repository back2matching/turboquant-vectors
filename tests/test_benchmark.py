"""Tests for benchmark reproducibility."""
import numpy as np
import pytest
from turboquant_vectors import compress, search
from turboquant_vectors.core import TurboQuantVectors


class TestBenchmarkReproducibility:
    def test_same_seed_same_results(self):
        """Compression with same seed gives identical output."""
        vecs = np.random.randn(100, 64).astype(np.float32)
        c1 = compress(vecs, bits=4, seed=42)
        c2 = compress(vecs, bits=4, seed=42)
        assert np.array_equal(c1.indices, c2.indices)
        assert np.array_equal(c1.norms, c2.norms)

    def test_batch_search_matches_single(self):
        """Batch query search matches individual query results."""
        vecs = np.random.randn(200, 64).astype(np.float32)
        queries = np.random.randn(5, 64).astype(np.float32)

        tq = TurboQuantVectors(dim=64, bits=4)
        compressed = tq.compress(vecs)

        # Batch search
        batch_idx, batch_scores = tq.search(compressed, queries, top_k=5)

        # Individual searches
        for i in range(len(queries)):
            single_idx, single_scores = tq.search(compressed, queries[i], top_k=5)
            assert np.array_equal(batch_idx[i], single_idx)
            assert np.allclose(batch_scores[i], single_scores, atol=1e-6)

    def test_recall_above_minimum(self):
        """4-bit compression on clustered data gets reasonable recall."""
        np.random.seed(42)
        n, dim = 5000, 128
        centers = np.random.randn(20, dim).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        labels = np.random.randint(0, 20, n)
        vecs = centers[labels] + np.random.randn(n, dim).astype(np.float32) * 0.3
        vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)

        queries = vecs[:10] + np.random.randn(10, dim).astype(np.float32) * 0.05
        queries = (queries / np.linalg.norm(queries, axis=1, keepdims=True)).astype(np.float32)

        # Exact ground truth
        scores = queries @ vecs.T
        gt = np.argsort(-scores, axis=1)[:, :10]

        # Compressed search
        tq = TurboQuantVectors(dim=dim, bits=4)
        compressed = tq.compress(vecs)
        pred, _ = tq.search(compressed, queries, top_k=10)

        # Recall
        recalls = []
        for p, g in zip(pred, gt):
            recalls.append(len(set(p.tolist()) & set(g.tolist())) / 10)
        mean_recall = np.mean(recalls)
        assert mean_recall >= 0.5, f"Recall too low: {mean_recall:.1%}"

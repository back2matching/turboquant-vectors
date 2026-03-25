"""Tests for TurboQuant vector compression."""
import numpy as np
import pytest
from turboquant_vectors import compress, decompress, search
from turboquant_vectors.core import TurboQuantVectors, CompressedVectors


class TestCompression:
    def test_compress_returns_correct_type(self):
        vecs = np.random.randn(100, 64).astype(np.float32)
        c = compress(vecs, bits=4)
        assert isinstance(c, CompressedVectors)
        assert c.n_vectors == 100
        assert c.dim == 64

    def test_compress_decompress_roundtrip(self):
        vecs = np.random.randn(100, 128).astype(np.float32)
        c = compress(vecs, bits=4)
        restored = decompress(c)
        assert restored.shape == vecs.shape
        mse = ((vecs - restored) ** 2).mean()
        assert mse < 1.0  # Reasonable error

    def test_compression_ratio_4bit(self):
        vecs = np.random.randn(10000, 768).astype(np.float32)
        c = compress(vecs, bits=4)
        # packed_memory_bytes excludes one-time rotation matrix overhead
        ratio = vecs.nbytes / c.packed_memory_bytes
        assert ratio > 5.0  # At least 5x at 4-bit

    def test_compression_ratio_2bit(self):
        vecs = np.random.randn(10000, 768).astype(np.float32)
        c = compress(vecs, bits=2)
        ratio = vecs.nbytes / c.packed_memory_bytes
        assert ratio > 10.0  # At least 10x at 2-bit

    def test_all_bit_widths(self):
        vecs = np.random.randn(50, 64).astype(np.float32)
        for bits in [2, 3, 4, 8]:
            c = compress(vecs, bits=bits)
            restored = decompress(c)
            assert restored.shape == vecs.shape

    def test_norm_preservation(self):
        vecs = np.random.randn(100, 128).astype(np.float32) * 5.0
        c = compress(vecs, bits=4)
        restored = decompress(c)
        orig_norms = np.linalg.norm(vecs, axis=1)
        rest_norms = np.linalg.norm(restored, axis=1)
        norm_error = np.abs(orig_norms - rest_norms).mean() / orig_norms.mean()
        assert norm_error < 0.2  # Less than 20% norm error

    def test_large_batch(self):
        vecs = np.random.randn(50000, 256).astype(np.float32)
        c = compress(vecs, bits=4)
        assert c.n_vectors == 50000


class TestSearch:
    def test_search_returns_correct_shape(self):
        vecs = np.random.randn(100, 64).astype(np.float32)
        c = compress(vecs, bits=4)
        query = np.random.randn(64).astype(np.float32)
        indices, scores = search(c, query, top_k=5)
        assert len(indices) == 5
        assert len(scores) == 5

    def test_search_finds_exact_match(self):
        vecs = np.random.randn(100, 64).astype(np.float32)
        # Query IS one of the vectors
        query = vecs[42].copy()
        c = compress(vecs, bits=8)  # High bit = low distortion
        indices, scores = search(c, query, top_k=5)
        assert 42 in indices  # Should find itself

    def test_recall_on_clustered_data(self):
        np.random.seed(42)
        n, dim = 1000, 128
        centers = np.random.randn(10, dim).astype(np.float32)
        labels = np.random.randint(0, 10, n)
        vecs = centers[labels] + np.random.randn(n, dim).astype(np.float32) * 0.3
        vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)

        query = centers[0] + np.random.randn(dim).astype(np.float32) * 0.1
        query = (query / np.linalg.norm(query)).astype(np.float32)

        # Exact search
        exact_scores = vecs @ query
        exact_top10 = np.argsort(-exact_scores)[:10]

        # Compressed search
        c = compress(vecs, bits=4)
        approx_idx, _ = search(c, query, top_k=10)

        recall = len(set(approx_idx) & set(exact_top10)) / 10
        assert recall >= 0.3  # At least 30% recall at 4-bit


class TestTurboQuantVectors:
    def test_deterministic_rotation(self):
        tq1 = TurboQuantVectors(dim=64, bits=4, seed=42)
        tq2 = TurboQuantVectors(dim=64, bits=4, seed=42)
        assert np.allclose(tq1.rotation, tq2.rotation)

    def test_different_seeds_different_rotations(self):
        tq1 = TurboQuantVectors(dim=64, bits=4, seed=42)
        tq2 = TurboQuantVectors(dim=64, bits=4, seed=99)
        assert not np.allclose(tq1.rotation, tq2.rotation)

    def test_rotation_is_orthogonal(self):
        tq = TurboQuantVectors(dim=64, bits=4)
        identity = tq.rotation @ tq.rotation.T
        assert np.allclose(identity, np.eye(64), atol=1e-5)

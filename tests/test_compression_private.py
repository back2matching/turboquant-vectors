"""
Tests for rotate_and_compress + CompressedPrivateVectors.

Proves:
1. Compression + privacy pipeline works end-to-end
2. Search on compressed vectors has acceptable recall
3. Save/load round-trips correctly
4. Compression ratios match expectations
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from turboquant_vectors.private import PrivateEncoder, CompressedPrivateVectors


@pytest.fixture
def encoder():
    return PrivateEncoder.generate(dim=256, normalize=False)


@pytest.fixture
def unit_vectors():
    rng = np.random.default_rng(42)
    v = rng.standard_normal((1000, 256)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


class TestRotateAndCompress:

    def test_basic_pipeline(self, encoder, unit_vectors):
        """rotate_and_compress returns a CompressedPrivateVectors."""
        cpv = encoder.rotate_and_compress(unit_vectors, bits=4)
        assert isinstance(cpv, CompressedPrivateVectors)
        assert cpv.n_vectors == 1000
        assert cpv.dim == 256
        assert cpv.bits == 4
        assert cpv.key_fingerprint == encoder.fingerprint()

    def test_compression_ratio_4bit(self, encoder, unit_vectors):
        """4-bit compression gives ~8x ratio from float32."""
        cpv = encoder.rotate_and_compress(unit_vectors, bits=4)
        ratio = cpv.compression_ratio
        assert ratio > 5.0, f"Compression ratio {ratio:.1f}x too low (expected > 5x at 4-bit)"

    def test_compression_ratio_2bit(self, encoder, unit_vectors):
        """2-bit compression gives higher ratio."""
        cpv = encoder.rotate_and_compress(unit_vectors, bits=2)
        ratio = cpv.compression_ratio
        assert ratio > 10.0, f"Compression ratio {ratio:.1f}x too low at 2-bit"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_search_recall(self, encoder, unit_vectors, bits):
        """Search on compressed vectors has acceptable recall."""
        data = unit_vectors[:900]
        queries = unit_vectors[900:]  # 100 queries

        cpv = encoder.rotate_and_compress(data, bits=bits)

        # Rotate queries with same key
        rotated_queries = encoder.rotate(queries, normalize=False)

        # Ground truth: brute-force on rotated (uncompressed) data
        rotated_data = encoder.rotate(data, normalize=False)

        recall_sum = 0
        k = 10
        for i in range(len(queries)):
            # Ground truth top-k
            gt_scores = rotated_data @ rotated_queries[i]
            gt_topk = set(np.argsort(-gt_scores)[:k])

            # Compressed search top-k
            idx, _ = cpv.search(rotated_queries[i], top_k=k)
            comp_topk = set(idx.tolist())

            recall_sum += len(gt_topk & comp_topk) / k

        avg_recall = recall_sum / len(queries)
        # Lower dims have worse quantization quality, so thresholds are lower
        min_recall = {2: 0.40, 3: 0.55, 4: 0.75}[bits]
        assert avg_recall >= min_recall, (
            f"Recall@{k} at {bits}-bit: {avg_recall:.3f} (expected >= {min_recall})"
        )

    def test_decompress_with_encoder(self, encoder, unit_vectors):
        """Full decompress + unrotate recovers approximate originals."""
        cpv = encoder.rotate_and_compress(unit_vectors[:100], bits=4)
        recovered = cpv.decompress(encoder)

        # Should be close but not exact (quantization loss)
        mse = np.mean((recovered - unit_vectors[:100]) ** 2)
        assert mse < 0.1, f"Decompress MSE {mse:.4f} too high"

    def test_batch_search(self, encoder, unit_vectors):
        """Batch query search works."""
        cpv = encoder.rotate_and_compress(unit_vectors[:500], bits=4)
        queries = encoder.rotate(unit_vectors[500:510], normalize=False)

        idx, scores = cpv.search(queries, top_k=5)
        assert idx.shape == (10, 5)
        assert scores.shape == (10, 5)


class TestSaveLoad:

    def test_npz_roundtrip(self, encoder, unit_vectors):
        """Save and load compressed vectors preserves data."""
        cpv = encoder.rotate_and_compress(unit_vectors[:200], bits=4)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = Path(f.name)

        try:
            cpv.save(path)
            loaded = CompressedPrivateVectors.load(path)

            assert loaded.n_vectors == cpv.n_vectors
            assert loaded.dim == cpv.dim
            assert loaded.bits == cpv.bits
            assert loaded.key_fingerprint == cpv.key_fingerprint
            np.testing.assert_array_equal(loaded.indices, cpv.indices)
            np.testing.assert_array_equal(loaded.norms, cpv.norms)
            np.testing.assert_array_equal(loaded.codebook, cpv.codebook)
        finally:
            path.unlink(missing_ok=True)

    def test_search_after_load(self, encoder, unit_vectors):
        """Search works after save/load cycle."""
        cpv = encoder.rotate_and_compress(unit_vectors[:500], bits=4)
        query = encoder.rotate(unit_vectors[500:501], normalize=False)

        idx_before, scores_before = cpv.search(query, top_k=5)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = Path(f.name)

        try:
            cpv.save(path)
            loaded = CompressedPrivateVectors.load(path)
            idx_after, scores_after = loaded.search(query, top_k=5)

            np.testing.assert_array_equal(idx_before, idx_after)
            np.testing.assert_allclose(scores_before, scores_after, atol=1e-6)
        finally:
            path.unlink(missing_ok=True)


class TestReprAndProperties:

    def test_repr(self, encoder, unit_vectors):
        cpv = encoder.rotate_and_compress(unit_vectors[:100], bits=4)
        r = repr(cpv)
        assert "CompressedPrivateVectors" in r
        assert "100" in r
        assert encoder.fingerprint() in r

    def test_memory_bytes(self, encoder, unit_vectors):
        cpv = encoder.rotate_and_compress(unit_vectors[:100], bits=4)
        assert cpv.memory_bytes > 0
        assert cpv.memory_bytes < cpv.original_bytes


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

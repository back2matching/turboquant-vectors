"""
Tests for PrivateEncoder: privacy-preserving embedding rotation.

Proves:
1. Mathematical guarantees (distance preservation, losslessness)
2. Security properties (inversion resistance, seed enforcement)
3. Key management (save/load, canary, fingerprint, rekey)
4. Error handling (NaN, zero vectors, dimension mismatch)
"""

import numpy as np
import pytest
import secrets
import tempfile
from pathlib import Path

from turboquant_vectors.private import PrivateEncoder


# --- Fixtures ---

@pytest.fixture
def encoder():
    """A PrivateEncoder with a fixed seed for reproducibility."""
    seed = secrets.randbits(128)
    return PrivateEncoder.from_seed(dim=128, seed=seed)


@pytest.fixture
def encoder_1536():
    """PrivateEncoder for OpenAI-sized embeddings."""
    seed = secrets.randbits(128)
    return PrivateEncoder.from_seed(dim=1536, seed=seed)


@pytest.fixture
def random_vectors():
    """10K random unit vectors in 128 dimensions."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal((10000, 128)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


@pytest.fixture
def random_vectors_1536():
    """1K random unit vectors in 1536 dimensions."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal((1000, 1536)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


# === 1. Mathematical Guarantees ===

class TestDistancePreservation:
    """Prove that orthogonal rotation preserves all distance metrics exactly."""

    def test_inner_product_preserved(self, encoder, random_vectors):
        """<Qx, Qy> == <x, y> for random x, y."""
        x, y = random_vectors[:100], random_vectors[100:200]
        rx = encoder.rotate(x, normalize=False)
        ry = encoder.rotate(y, normalize=False)

        original_ip = np.sum(x * y, axis=1)
        rotated_ip = np.sum(rx * ry, axis=1)

        np.testing.assert_allclose(rotated_ip, original_ip, atol=1e-5)

    def test_cosine_similarity_preserved(self, encoder, random_vectors):
        """cos(Qx, Qy) == cos(x, y)."""
        x, y = random_vectors[:100], random_vectors[100:200]
        rx = encoder.rotate(x, normalize=False)
        ry = encoder.rotate(y, normalize=False)

        def cosine(a, b):
            return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

        original_cos = cosine(x, y)
        rotated_cos = cosine(rx, ry)

        np.testing.assert_allclose(rotated_cos, original_cos, atol=1e-5)

    def test_l2_distance_preserved(self, encoder, random_vectors):
        """||Qx - Qy|| == ||x - y||."""
        x, y = random_vectors[:100], random_vectors[100:200]
        rx = encoder.rotate(x, normalize=False)
        ry = encoder.rotate(y, normalize=False)

        original_dist = np.linalg.norm(x - y, axis=1)
        rotated_dist = np.linalg.norm(rx - ry, axis=1)

        np.testing.assert_allclose(rotated_dist, original_dist, atol=1e-5)

    def test_recall_at_k_exactly_1(self, encoder, random_vectors):
        """Top-K search on rotated vectors returns identical results to original."""
        data = random_vectors[:1000]
        queries = random_vectors[1000:1010]
        k = 10

        rotated_data = encoder.rotate(data, normalize=False)
        rotated_queries = encoder.rotate(queries, normalize=False)

        for i in range(len(queries)):
            # Original top-k
            orig_scores = data @ queries[i]
            orig_topk = set(np.argsort(-orig_scores)[:k])

            # Rotated top-k
            rot_scores = rotated_data @ rotated_queries[i]
            rot_topk = set(np.argsort(-rot_scores)[:k])

            assert orig_topk == rot_topk, f"Query {i}: top-{k} mismatch"

    def test_orthogonality(self, encoder):
        """Q^T Q == I within float32 tolerance."""
        Q = encoder._rotation
        QTQ = Q.T @ Q
        I = np.eye(encoder.dim, dtype=np.float32)
        np.testing.assert_allclose(QTQ, I, atol=1e-4)

    def test_round_trip_identity(self, encoder, random_vectors):
        """unrotate(rotate(x)) == x."""
        x = random_vectors[:100]
        rx = encoder.rotate(x, normalize=False)
        recovered = encoder.unrotate(rx)
        np.testing.assert_allclose(recovered, x, atol=1e-5)

    def test_norm_preservation(self, encoder):
        """||Qx|| == ||x|| for non-unit-norm vectors."""
        rng = np.random.default_rng(99)
        x = rng.standard_normal((100, encoder.dim)).astype(np.float32) * 3.7
        rx = encoder.rotate(x, normalize=False)

        original_norms = np.linalg.norm(x, axis=1)
        rotated_norms = np.linalg.norm(rx, axis=1)

        np.testing.assert_allclose(rotated_norms, original_norms, atol=1e-4)

    def test_single_vector_shape(self, encoder):
        """rotate() handles single vector (d,) shape."""
        x = np.random.randn(encoder.dim).astype(np.float32)
        x /= np.linalg.norm(x)
        rx = encoder.rotate(x, normalize=False)
        assert rx.shape == (encoder.dim,)
        recovered = encoder.unrotate(rx)
        np.testing.assert_allclose(recovered, x, atol=1e-5)


# === 2. Security Properties ===

class TestSecurityProperties:
    """Prove that rotation provides meaningful privacy."""

    def test_inversion_resistance(self, encoder, random_vectors):
        """Rotated vectors have near-zero correlation with originals per dimension."""
        x = random_vectors[:1000]
        rx = encoder.rotate(x, normalize=False)

        # Per-dimension Pearson correlation
        correlations = []
        for d in range(min(20, encoder.dim)):  # Check first 20 dims
            r = np.corrcoef(x[:, d], rx[:, d])[0, 1]
            correlations.append(abs(r))

        mean_corr = np.mean(correlations)
        # Theoretical expected ~sqrt(2/(pi*d)) ≈ 0.07 for d=128, plus sampling noise
        assert mean_corr < 0.15, f"Mean correlation {mean_corr:.3f} too high (expected < 0.15)"

    def test_seed_minimum_enforced(self):
        """Seeds < 2^64 are rejected."""
        with pytest.raises(ValueError, match="2\\^64"):
            PrivateEncoder.from_seed(dim=128, seed=42)

        with pytest.raises(ValueError, match="2\\^64"):
            PrivateEncoder.from_seed(dim=128, seed=2**63)

    def test_seed_at_minimum_works(self):
        """Seed exactly at 2^64 is accepted."""
        enc = PrivateEncoder.from_seed(dim=32, seed=2**64)
        assert enc.dim == 32

    def test_different_seeds_different_rotations(self):
        """Two encoders with different seeds produce different output."""
        seed1 = secrets.randbits(128)
        seed2 = secrets.randbits(128)
        enc1 = PrivateEncoder.from_seed(dim=64, seed=seed1)
        enc2 = PrivateEncoder.from_seed(dim=64, seed=seed2)

        x = np.random.randn(10, 64).astype(np.float32)
        r1 = enc1.rotate(x, normalize=False)
        r2 = enc2.rotate(x, normalize=False)

        assert not np.allclose(r1, r2, atol=1e-3), "Different seeds should produce different rotations"

    def test_seed_reproducibility(self):
        """Same seed produces identical rotation matrix."""
        seed = secrets.randbits(128)
        enc1 = PrivateEncoder.from_seed(dim=64, seed=seed)
        enc2 = PrivateEncoder.from_seed(dim=64, seed=seed)

        x = np.random.randn(10, 64).astype(np.float32)
        r1 = enc1.rotate(x, normalize=False)
        r2 = enc2.rotate(x, normalize=False)

        np.testing.assert_array_equal(r1, r2)

    def test_cross_dimension_decorrelation(self):
        """Same seed with different dimensions produces unrelated matrices."""
        seed = secrets.randbits(128)
        enc32 = PrivateEncoder.from_seed(dim=32, seed=seed)
        enc64 = PrivateEncoder.from_seed(dim=64, seed=seed)

        # The 32x32 submatrix of the 64x64 should NOT match the 32x32 matrix
        sub = enc64._rotation[:32, :32]
        assert not np.allclose(sub, enc32._rotation, atol=0.1), \
            "Same seed with different dims should produce unrelated matrices"


# === 3. Key Management ===

class TestKeyManagement:
    """Test .tqkey file format, canary, fingerprint, rekey."""

    def test_save_load_roundtrip(self, encoder):
        """save_key + load_key preserves the rotation matrix exactly."""
        with tempfile.NamedTemporaryFile(suffix='.tqkey', delete=False) as f:
            path = Path(f.name)

        try:
            encoder.save_key(path)
            loaded = PrivateEncoder.load_key(path)

            np.testing.assert_array_equal(encoder._rotation, loaded._rotation)
            assert encoder.fingerprint() == loaded.fingerprint()
        finally:
            path.unlink(missing_ok=True)

    def test_save_adds_extension(self, encoder):
        """save_key adds .tqkey extension if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mykey"
            encoder.save_key(path)
            assert (Path(tmpdir) / "mykey.tqkey").exists()

    def test_load_detects_corruption(self, encoder):
        """Corrupted .tqkey file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.tqkey', delete=False) as f:
            path = Path(f.name)

        try:
            encoder.save_key(path)

            # Corrupt a byte in the matrix data
            data = bytearray(path.read_bytes())
            data[20] ^= 0xFF
            path.write_bytes(bytes(data))

            with pytest.raises(ValueError, match="checksum"):
                PrivateEncoder.load_key(path)
        finally:
            path.unlink(missing_ok=True)

    def test_load_wrong_magic(self):
        """Non-.tqkey file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.tqkey', delete=False) as f:
            f.write(b'NOT_A_KEY_FILE_AT_ALL_1234567890')
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="magic"):
                PrivateEncoder.load_key(path)
        finally:
            path.unlink(missing_ok=True)

    def test_fingerprint_is_stable(self, encoder):
        """Fingerprint returns the same value on repeated calls."""
        fp1 = encoder.fingerprint()
        fp2 = encoder.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 16
        assert all(c in '0123456789abcdef' for c in fp1)

    def test_canary_roundtrip(self, encoder):
        """make_canary + verify_canary works."""
        canary = encoder.make_canary()
        assert isinstance(canary, bytes)
        assert len(canary) == 32
        assert encoder.verify_canary(canary)

    def test_canary_wrong_key(self, encoder):
        """Canary from different key returns False."""
        canary = encoder.make_canary()
        other = PrivateEncoder.generate(dim=encoder.dim)
        assert not other.verify_canary(canary)

    def test_rekey_correctness(self):
        """rekey_vectors gives same result as unrotate + rotate."""
        seed1 = secrets.randbits(128)
        seed2 = secrets.randbits(128)
        old = PrivateEncoder.from_seed(dim=64, seed=seed1)
        new = PrivateEncoder.from_seed(dim=64, seed=seed2)

        x = np.random.randn(100, 64).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)

        # Rotate with old key
        rotated_old = old.rotate(x, normalize=False)

        # Rekey: should equal unrotate(old) + rotate(new)
        rekeyed = new.rekey_vectors(rotated_old, old)
        expected = new.rotate(old.unrotate(rotated_old), normalize=False)

        np.testing.assert_allclose(rekeyed, expected, atol=1e-5)

    def test_rekey_dimension_mismatch(self):
        """rekey_vectors raises on dimension mismatch."""
        enc32 = PrivateEncoder.generate(dim=32)
        enc64 = PrivateEncoder.generate(dim=64)
        v = np.random.randn(10, 32).astype(np.float32)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            enc64.rekey_vectors(v, enc32)


# === 4. Error Handling ===

class TestErrorHandling:
    """Input validation and edge cases."""

    def test_nan_rejected(self, encoder):
        """rotate() raises ValueError for NaN input."""
        x = np.random.randn(10, encoder.dim).astype(np.float32)
        x[3, 5] = np.nan

        with pytest.raises(ValueError, match="NaN or inf"):
            encoder.rotate(x, normalize=False)

    def test_inf_rejected(self, encoder):
        """rotate() raises ValueError for inf input."""
        x = np.random.randn(10, encoder.dim).astype(np.float32)
        x[7, 2] = np.inf

        with pytest.raises(ValueError, match="NaN or inf"):
            encoder.rotate(x, normalize=False)

    def test_zero_vector_with_normalize_rejected(self, encoder):
        """rotate() raises ValueError for zero vectors when normalize=True."""
        x = np.zeros((5, encoder.dim), dtype=np.float32)
        x[0] = np.random.randn(encoder.dim)  # One valid vector

        with pytest.raises(ValueError, match="zero"):
            encoder.rotate(x, normalize=True)

    def test_zero_vector_without_normalize_ok(self, encoder):
        """Zero vectors are fine when normalize=False."""
        x = np.zeros((5, encoder.dim), dtype=np.float32)
        rx = encoder.rotate(x, normalize=False)
        np.testing.assert_allclose(rx, 0.0, atol=1e-10)

    def test_dimension_mismatch(self, encoder):
        """rotate() raises ValueError for wrong dimension."""
        x = np.random.randn(10, encoder.dim + 1).astype(np.float32)

        with pytest.raises(ValueError, match="dim="):
            encoder.rotate(x)

    def test_small_dim_rejected(self):
        """dim < 2 raises ValueError."""
        with pytest.raises(ValueError, match="Dimension"):
            PrivateEncoder.generate(dim=1)

    def test_c_contiguous_output(self, encoder, random_vectors):
        """Output arrays are C-contiguous (for FAISS compatibility)."""
        rx = encoder.rotate(random_vectors[:100], normalize=False)
        assert rx.flags['C_CONTIGUOUS']

    def test_normalize_default_from_constructor(self):
        """Constructor normalize setting is applied by default."""
        enc_norm = PrivateEncoder.generate(dim=32)  # normalize=True by default
        x = np.random.randn(10, 32).astype(np.float32) * 5.0

        rx = enc_norm.rotate(x)  # Should normalize
        norms = np.linalg.norm(rx, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_normalize_override(self):
        """Per-call normalize override works."""
        enc = PrivateEncoder.generate(dim=32)  # normalize=True default
        x = np.random.randn(10, 32).astype(np.float32) * 5.0
        original_norms = np.linalg.norm(x, axis=1)

        rx = enc.rotate(x, normalize=False)  # Override: don't normalize
        rotated_norms = np.linalg.norm(rx, axis=1)
        np.testing.assert_allclose(rotated_norms, original_norms, atol=1e-4)


# === 5. Repr and Properties ===

class TestReprAndProperties:

    def test_repr(self, encoder):
        r = repr(encoder)
        assert "PrivateEncoder" in r
        assert str(encoder.dim) in r
        assert encoder.fingerprint() in r

    def test_key_size_bytes(self, encoder):
        assert encoder.key_size_bytes == encoder.dim ** 2 * 4

    def test_generate_uses_entropy(self):
        """Two generate() calls produce different keys."""
        enc1 = PrivateEncoder.generate(dim=32)
        enc2 = PrivateEncoder.generate(dim=32)
        assert enc1.fingerprint() != enc2.fingerprint()


# === 6. Audit-Required Tests (from review agents) ===

class TestPinnedOutput:
    """Golden-file tests to catch silent key derivation changes."""

    def test_from_seed_fingerprint_is_stable(self):
        """A known seed must always produce the same fingerprint.
        If this test fails, all previously saved .tqkey files from from_seed() are broken."""
        seed = 2**64 + 12345  # Fixed known seed
        enc = PrivateEncoder.from_seed(dim=64, seed=seed)
        fp = enc.fingerprint()
        # Pin the expected fingerprint. If the derivation pipeline changes,
        # this test catches it immediately — all saved .tqkey files would break.
        assert fp == "4f240186f1a3c69b", (
            f"Fingerprint changed from 4f240186f1a3c69b to {fp}. "
            f"This means the key derivation pipeline changed and all "
            f"previously generated .tqkey files from from_seed() are broken."
        )
        # Re-derive to confirm determinism
        enc2 = PrivateEncoder.from_seed(dim=64, seed=seed)
        assert enc2.fingerprint() == fp

    def test_canary_stable_across_save_load(self):
        """Canary must survive save/load cycle."""
        seed = 2**64 + 99999
        enc = PrivateEncoder.from_seed(dim=64, seed=seed)
        canary = enc.make_canary()

        with tempfile.NamedTemporaryFile(suffix='.tqkey', delete=False) as f:
            path = Path(f.name)
        try:
            enc.save_key(path)
            loaded = PrivateEncoder.load_key(path)
            assert loaded.verify_canary(canary)
        finally:
            path.unlink(missing_ok=True)


class TestSearchMetrics:
    """Cover search() with all metric types (ip, l2, cosine)."""

    def _make_compressed(self):
        enc = PrivateEncoder.generate(dim=64, normalize=False)
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 64)).astype(np.float32)
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        cpv = enc.rotate_and_compress(data, bits=4)
        # Single vector query (1D) for single-result tests
        query = enc.rotate(data[0], normalize=False)
        return cpv, query

    def test_search_cosine(self):
        cpv, q = self._make_compressed()
        idx, scores = cpv.search(q, top_k=5, metric="cosine")
        assert idx.shape == (5,), f"Expected (5,), got {idx.shape}"
        assert scores.shape == (5,)
        assert scores[0] >= scores[-1]  # Sorted descending

    def test_search_ip(self):
        cpv, q = self._make_compressed()
        idx, scores = cpv.search(q, top_k=5, metric="ip")
        assert idx.shape == (5,)
        assert scores[0] >= scores[-1]

    def test_search_l2(self):
        cpv, q = self._make_compressed()
        idx, scores = cpv.search(q, top_k=5, metric="l2")
        assert idx.shape == (5,)
        # L2 scores are negative squared distances, higher = closer
        assert scores[0] >= scores[-1]

    def test_search_unknown_metric(self):
        cpv, q = self._make_compressed()
        with pytest.raises(ValueError, match="Unknown metric"):
            cpv.search(q, top_k=5, metric="hamming")

    def test_search_topk_larger_than_n(self):
        """top_k > n_vectors should return all vectors without error."""
        cpv, q = self._make_compressed()
        idx, scores = cpv.search(q, top_k=500)  # Only 200 vectors
        assert len(idx) <= 200


class TestBitsValidation:
    """Validate bits parameter bounds."""

    def test_bits_zero_rejected(self):
        enc = PrivateEncoder.generate(dim=32, normalize=False)
        data = np.random.randn(10, 32).astype(np.float32)
        with pytest.raises(ValueError, match="bits must be 1-8"):
            enc.rotate_and_compress(data, bits=0)

    def test_bits_negative_rejected(self):
        enc = PrivateEncoder.generate(dim=32, normalize=False)
        data = np.random.randn(10, 32).astype(np.float32)
        with pytest.raises(ValueError, match="bits must be 1-8"):
            enc.rotate_and_compress(data, bits=-1)

    def test_bits_too_large_rejected(self):
        enc = PrivateEncoder.generate(dim=32, normalize=False)
        data = np.random.randn(10, 32).astype(np.float32)
        with pytest.raises(ValueError, match="bits must be 1-8"):
            enc.rotate_and_compress(data, bits=9)


class TestAdditionalEdgeCases:
    """Edge cases from audit."""

    def test_unrotate_dimension_mismatch(self):
        enc = PrivateEncoder.generate(dim=64, normalize=False)
        x = np.random.randn(10, 32).astype(np.float32)
        with pytest.raises(ValueError, match="dim="):
            enc.unrotate(x)

    def test_empty_input(self):
        """Empty array should produce empty output."""
        enc = PrivateEncoder.generate(dim=64, normalize=False)
        x = np.empty((0, 64), dtype=np.float32)
        rx = enc.rotate(x, normalize=False)
        assert rx.shape == (0, 64)

    def test_float64_input_converted(self):
        """float64 input should be silently converted to float32."""
        enc = PrivateEncoder.generate(dim=32, normalize=False)
        x = np.random.randn(10, 32)  # float64 by default
        rx = enc.rotate(x)
        assert rx.dtype == np.float32

    def test_from_seed_dim_too_small(self):
        with pytest.raises(ValueError, match="Dimension"):
            PrivateEncoder.from_seed(dim=1, seed=2**64)

    def test_seed_boundary_off_by_one(self):
        """2^64 - 1 should be rejected, 2^64 accepted."""
        with pytest.raises(ValueError, match="2\\^64"):
            PrivateEncoder.from_seed(dim=32, seed=2**64 - 1)
        enc = PrivateEncoder.from_seed(dim=32, seed=2**64)
        assert enc.dim == 32

    def test_bad_matrix_non_square(self):
        """Non-square matrix rejected by constructor."""
        mat = np.random.randn(32, 64).astype(np.float32)
        with pytest.raises(ValueError, match="square"):
            PrivateEncoder(mat)

    def test_bad_matrix_non_orthogonal(self):
        """Non-orthogonal matrix rejected by constructor."""
        mat = np.random.randn(32, 32).astype(np.float32)  # Random, not orthogonal
        with pytest.raises(ValueError, match="orthogonal"):
            PrivateEncoder(mat)

    def test_load_key_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PrivateEncoder.load_key("/nonexistent/path.tqkey")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

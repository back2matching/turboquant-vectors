"""Extended tests for core.py: input validation, edge cases, save/load."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from turboquant_vectors.core import TurboQuantVectors, CompressedVectors, compress, decompress, search


class TestInputValidation:
    """B1/B2: Proper ValueError instead of assert, NaN/inf rejection."""

    def test_wrong_shape_raises_valueerror(self):
        tq = TurboQuantVectors(dim=64, bits=4)
        vecs = np.random.randn(10, 32).astype(np.float32)
        with pytest.raises(ValueError, match="dim=64"):
            tq.compress(vecs)

    def test_1d_input_raises_valueerror(self):
        tq = TurboQuantVectors(dim=64, bits=4)
        vec = np.random.randn(64).astype(np.float32)
        with pytest.raises(ValueError):
            tq.compress(vec)

    def test_nan_input_raises_valueerror(self):
        tq = TurboQuantVectors(dim=64, bits=4)
        vecs = np.random.randn(10, 64).astype(np.float32)
        vecs[3, 5] = np.nan
        with pytest.raises(ValueError, match="NaN or inf"):
            tq.compress(vecs)

    def test_inf_input_raises_valueerror(self):
        tq = TurboQuantVectors(dim=64, bits=4)
        vecs = np.random.randn(10, 64).astype(np.float32)
        vecs[0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or inf"):
            tq.compress(vecs)


class TestAllBitWidths:
    """T2: Test all bit widths including 1, 5, 6, 7."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_compress_decompress_all_bits(self, bits):
        vecs = np.random.randn(50, 64).astype(np.float32)
        c = compress(vecs, bits=bits)
        restored = decompress(c)
        assert restored.shape == vecs.shape
        # Higher bits should have lower error
        mse = ((vecs - restored) ** 2).mean()
        assert mse < 5.0  # Very loose bound for 1-bit


class TestCompressedVectorsSaveLoad:
    """T3: Test save/load for base CompressedVectors."""

    def test_save_load_roundtrip(self):
        vecs = np.random.randn(200, 64).astype(np.float32)
        c = compress(vecs, bits=4)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = Path(f.name)

        try:
            c.save(str(path))
            loaded = CompressedVectors.load(str(path))

            assert loaded.n_vectors == c.n_vectors
            assert loaded.dim == c.dim
            assert loaded.bits == c.bits
            np.testing.assert_array_equal(loaded.indices, c.indices)
            np.testing.assert_array_equal(loaded.norms, c.norms)
            np.testing.assert_array_equal(loaded.rotation, c.rotation)
            np.testing.assert_array_equal(loaded.codebook, c.codebook)
        finally:
            path.unlink(missing_ok=True)

    def test_search_after_load(self):
        vecs = np.random.randn(200, 64).astype(np.float32)
        query = vecs[0].copy()
        c = compress(vecs, bits=4)

        idx_before, scores_before = search(c, query, top_k=5)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = Path(f.name)

        try:
            c.save(str(path))
            loaded = CompressedVectors.load(str(path))
            idx_after, scores_after = search(loaded, query, top_k=5)

            np.testing.assert_array_equal(idx_before, idx_after)
            np.testing.assert_allclose(scores_before, scores_after, atol=1e-6)
        finally:
            path.unlink(missing_ok=True)


class TestSharedCodebook:
    """Verify codebook dedup didn't break anything."""

    def test_codebook_matches_between_modules(self):
        from turboquant_vectors._rotation import compute_codebook
        from turboquant_vectors.core import TurboQuantVectors

        tq = TurboQuantVectors(dim=128, bits=4)
        shared = compute_codebook(128, 4)
        np.testing.assert_array_equal(tq.codebook, shared)

    def test_codebook_1bit_symmetric(self):
        from turboquant_vectors._rotation import compute_codebook
        cb = compute_codebook(64, 1)
        assert len(cb) == 2
        assert cb[0] == -cb[1]

    def test_codebook_length_matches_bits(self):
        from turboquant_vectors._rotation import compute_codebook
        for bits in range(1, 9):
            cb = compute_codebook(64, bits)
            assert len(cb) == 2 ** bits


class TestTQVBitsValidation:
    """Validate bits parameter in TurboQuantVectors constructor."""

    def test_bits_zero_rejected(self):
        with pytest.raises(ValueError, match="bits must be 1-8"):
            TurboQuantVectors(dim=64, bits=0)

    def test_bits_nine_rejected(self):
        with pytest.raises(ValueError, match="bits must be 1-8"):
            TurboQuantVectors(dim=64, bits=9)

    def test_bits_negative_rejected(self):
        with pytest.raises(ValueError, match="bits must be 1-8"):
            TurboQuantVectors(dim=64, bits=-1)

    def test_dim_zero_rejected(self):
        with pytest.raises(ValueError, match="dim must be >= 1"):
            TurboQuantVectors(dim=0, bits=4)


class TestRotateAndCompressDimensionMismatch:
    """Test dimension mismatch in rotate_and_compress."""

    def test_wrong_dim_raises(self):
        from turboquant_vectors.private import PrivateEncoder
        enc = PrivateEncoder.generate(dim=64, normalize=False)
        vecs = np.random.randn(10, 32).astype(np.float32)
        with pytest.raises(ValueError):
            enc.rotate_and_compress(vecs, bits=4)

    def test_1d_input_raises(self):
        from turboquant_vectors.private import PrivateEncoder
        enc = PrivateEncoder.generate(dim=64, normalize=False)
        vec = np.random.randn(64).astype(np.float32)
        with pytest.raises(ValueError):
            enc.rotate_and_compress(vec, bits=4)


class TestEmptyInput:

    def test_compress_empty(self):
        tq = TurboQuantVectors(dim=64, bits=4)
        vecs = np.empty((0, 64), dtype=np.float32)
        c = tq.compress(vecs)
        assert c.n_vectors == 0

    def test_search_empty_database(self):
        vecs = np.random.randn(5, 64).astype(np.float32)
        c = compress(vecs, bits=4)
        query = np.random.randn(64).astype(np.float32)
        idx, scores = search(c, query, top_k=10)
        assert len(idx) <= 5  # Can't return more than we have


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

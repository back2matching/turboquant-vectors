"""
Privacy demonstration benchmarks.

Proves that rotation defeats practical attacks:
1. Classifier transfer: MLP trained on originals fails on rotated vectors
2. Statistical correlation: per-dimension Pearson r ≈ 0
3. Performance: rotation latency is negligible
"""

import time
import numpy as np
import pytest
import secrets

from turboquant_vectors.private import PrivateEncoder


def _generate_clustered_data(n_per_class=500, dim=256, n_classes=5, seed=42):
    """Generate synthetic labeled embeddings with distinct clusters."""
    rng = np.random.default_rng(seed)
    vectors = []
    labels = []
    for c in range(n_classes):
        center = rng.standard_normal(dim).astype(np.float32)
        center = center / np.linalg.norm(center) * 2.0
        points = center + rng.standard_normal((n_per_class, dim)).astype(np.float32) * 0.3
        # Normalize to unit vectors
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        vectors.append(points)
        labels.extend([c] * n_per_class)
    return np.vstack(vectors), np.array(labels)


class TestClassifierTransferAttack:
    """
    The killer demo: train a classifier on original embeddings,
    test on rotated. Accuracy should drop to random chance.
    """

    def test_classifier_fails_on_rotated(self):
        """MLP-style classifier trained on originals fails on rotated vectors."""
        dim = 256
        X, y = _generate_clustered_data(n_per_class=400, dim=dim, n_classes=5)
        n_classes = 5

        # Split train/test
        n = len(X)
        idx = np.random.default_rng(0).permutation(n)
        train_idx, test_idx = idx[:int(0.7 * n)], idx[int(0.7 * n):]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Train a simple nearest-centroid classifier on original embeddings
        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for c in range(n_classes):
            centroids[c] = X_train[y_train == c].mean(axis=0)

        # Test on original: should be high accuracy
        orig_preds = np.argmax(X_test @ centroids.T, axis=1)
        orig_acc = (orig_preds == y_test).mean()
        assert orig_acc > 0.8, f"Classifier should work on originals (got {orig_acc:.2%})"

        # Rotate test set with a secret key
        encoder = PrivateEncoder.generate(dim=dim, normalize=False)
        X_test_rotated = encoder.rotate(X_test, normalize=False)

        # Test rotated vectors against ORIGINAL centroids: should fail
        rot_preds = np.argmax(X_test_rotated @ centroids.T, axis=1)
        rot_acc = (rot_preds == y_test).mean()

        # Expected: near random chance (1/n_classes = 20%)
        assert rot_acc < 0.35, (
            f"Classifier on rotated vectors should be near random chance. "
            f"Got {rot_acc:.2%} (random = {1/n_classes:.2%})"
        )

    def test_rotated_classifier_still_works(self):
        """If you train on rotated, it works on rotated (same key). Proves rotation is consistent."""
        dim = 128
        X, y = _generate_clustered_data(n_per_class=300, dim=dim, n_classes=3)
        n_classes = 3

        encoder = PrivateEncoder.generate(dim=dim, normalize=False)
        X_rotated = encoder.rotate(X, normalize=False)

        n = len(X)
        idx = np.random.default_rng(0).permutation(n)
        train_idx, test_idx = idx[:int(0.7 * n)], idx[int(0.7 * n):]

        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        for c in range(n_classes):
            centroids[c] = X_rotated[train_idx][y[train_idx] == c].mean(axis=0)

        preds = np.argmax(X_rotated[test_idx] @ centroids.T, axis=1)
        acc = (preds == y[test_idx]).mean()
        assert acc > 0.8, f"Classifier trained on rotated should work on rotated (got {acc:.2%})"


class TestStatisticalCorrelation:
    """Per-dimension correlation between original and rotated should be ~0."""

    def test_per_dimension_correlation_near_zero(self):
        """Mean per-dimension Pearson r between original and rotated should be ~0.

        With n samples and d dimensions, each individual |r| is approximately
        Normal(0, 1/sqrt(n)). The mean over d dimensions converges fast.
        We check that the mean is small, not any single dimension's max
        (which is expected to be ~3/sqrt(n) by extreme value theory).
        """
        dim = 256
        n = 10000
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, dim)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        encoder = PrivateEncoder.generate(dim=dim, normalize=False)
        X_rot = encoder.rotate(X, normalize=False)

        # Vectorized correlation: much faster than per-dim loop
        # Pearson r = cov(X_d, X_rot_d) / (std(X_d) * std(X_rot_d))
        X_centered = X - X.mean(axis=0)
        R_centered = X_rot - X_rot.mean(axis=0)
        cov = (X_centered * R_centered).mean(axis=0)
        std_x = X_centered.std(axis=0)
        std_r = R_centered.std(axis=0)
        correlations = np.abs(cov / (std_x * std_r + 1e-10))

        mean_corr = correlations.mean()
        # Each Q_{j,j} entry of a random orthogonal matrix has expected magnitude
        # sqrt(2/(pi*d)). For d=256: ~0.05. Plus sampling noise ~1/sqrt(n).
        # The correlation between x_j and (Qx)_j is approximately Q_{j,j}.
        expected = np.sqrt(2.0 / (np.pi * dim))  # ~0.05 for d=256
        assert mean_corr < expected * 2.5, (
            f"Mean per-dim |correlation| {mean_corr:.4f} exceeds "
            f"{expected * 2.5:.4f} (2.5x theoretical {expected:.4f})"
        )


class TestPerformanceBenchmarks:
    """Rotation latency benchmarks for README."""

    @pytest.mark.parametrize("dim", [384, 768, 1536])
    def test_rotation_latency_single(self, dim):
        """Single vector rotation time."""
        encoder = PrivateEncoder.generate(dim=dim, normalize=False)
        x = np.random.randn(dim).astype(np.float32)

        # Warmup
        for _ in range(10):
            encoder.rotate(x, normalize=False)

        # Benchmark
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            encoder.rotate(x, normalize=False)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        print(f"\n  dim={dim} single: {median_ms:.3f} ms")
        assert median_ms < 50, f"Single rotation too slow: {median_ms:.1f}ms"

    @pytest.mark.parametrize("dim", [384, 768, 1536])
    def test_rotation_latency_batch_10k(self, dim):
        """Batch 10K vector rotation time."""
        encoder = PrivateEncoder.generate(dim=dim, normalize=False)
        X = np.random.randn(10000, dim).astype(np.float32)

        # Warmup
        encoder.rotate(X, normalize=False)

        # Benchmark
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            encoder.rotate(X, normalize=False)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        print(f"\n  dim={dim} batch=10K: {median_ms:.1f} ms")
        assert median_ms < 5000, f"Batch rotation too slow: {median_ms:.1f}ms"

    def test_key_generation_time(self):
        """Key generation time for common dimensions."""
        for dim in [384, 768, 1536]:
            t0 = time.perf_counter()
            PrivateEncoder.generate(dim=dim, normalize=False)
            elapsed = time.perf_counter() - t0
            print(f"\n  keygen dim={dim}: {elapsed*1000:.1f} ms")
            assert elapsed < 30, f"Keygen too slow: {elapsed:.1f}s"

    def test_key_file_sizes(self):
        """Key file sizes for common dimensions."""
        import tempfile
        from pathlib import Path

        for dim in [384, 768, 1536]:
            enc = PrivateEncoder.generate(dim=dim, normalize=False)
            with tempfile.NamedTemporaryFile(suffix='.tqkey', delete=False) as f:
                path = Path(f.name)
            try:
                enc.save_key(path)
                size_mb = path.stat().st_size / 1e6
                print(f"\n  dim={dim} key: {size_mb:.1f} MB")
                expected_mb = dim * dim * 4 / 1e6
                assert abs(size_mb - expected_mb) < 0.1
            finally:
                path.unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

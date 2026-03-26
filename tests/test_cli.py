"""Tests for CLI: tq-vectors compress/search/info."""

import numpy as np
import pytest
import subprocess
import sys
import tempfile
from pathlib import Path


@pytest.fixture
def sample_vectors():
    """Create a temporary .npy file with sample vectors."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((100, 64)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        path = Path(f.name)
    np.save(path, vecs)
    yield path, vecs
    path.unlink(missing_ok=True)


def run_cli(*args):
    """Run tq-vectors CLI and return stdout."""
    result = subprocess.run(
        [sys.executable, "-m", "turboquant_vectors.cli"] + list(args),
        capture_output=True, text=True, timeout=60,
    )
    return result


class TestCLICompress:

    def test_compress_creates_output(self, sample_vectors):
        path, _ = sample_vectors
        output = str(path).replace('.npy', '.tqv.npz')
        try:
            result = run_cli("compress", str(path))
            assert result.returncode == 0, result.stderr
            assert Path(output).exists()
            assert "compression" in result.stdout.lower() or "saved" in result.stdout.lower()
        finally:
            Path(output).unlink(missing_ok=True)

    def test_compress_custom_output(self, sample_vectors):
        path, _ = sample_vectors
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            output = Path(f.name)
        try:
            result = run_cli("compress", str(path), "-o", str(output))
            assert result.returncode == 0, result.stderr
            assert output.exists()
        finally:
            output.unlink(missing_ok=True)

    def test_compress_2bit(self, sample_vectors):
        path, _ = sample_vectors
        output = str(path).replace('.npy', '.tqv.npz')
        try:
            result = run_cli("compress", str(path), "-b", "2")
            assert result.returncode == 0, result.stderr
            assert "2-bit" in result.stdout
        finally:
            Path(output).unlink(missing_ok=True)


class TestCLIInfo:

    def test_info_shows_metadata(self, sample_vectors):
        path, _ = sample_vectors
        output = str(path).replace('.npy', '.tqv.npz')
        try:
            run_cli("compress", str(path))
            result = run_cli("info", output)
            assert result.returncode == 0, result.stderr
            assert "100" in result.stdout  # n_vectors
            assert "64" in result.stdout   # dim
        finally:
            Path(output).unlink(missing_ok=True)


class TestCLISearch:

    def test_search_returns_results(self, sample_vectors):
        path, vecs = sample_vectors
        output = str(path).replace('.npy', '.tqv.npz')
        # Create a single-vector query
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            query_path = Path(f.name)
        np.save(query_path, vecs[0])
        try:
            run_cli("compress", str(path))
            result = run_cli("search", output, str(query_path), "-k", "5")
            assert result.returncode == 0, result.stderr
            assert "score=" in result.stdout
        finally:
            Path(output).unlink(missing_ok=True)
            query_path.unlink(missing_ok=True)


class TestCLIHelp:

    def test_no_args_shows_help(self):
        result = run_cli()
        # Should not crash, should show usage
        assert result.returncode == 0

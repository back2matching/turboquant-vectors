"""Tests for LangChain PrivateEmbeddings adapter."""

import numpy as np
import pytest
import sys
import types


# Mock langchain_core.embeddings so tests work without langchain installed
def setup_langchain_mock():
    """Create a minimal mock of langchain_core.embeddings.Embeddings."""
    if "langchain_core" not in sys.modules:
        # Create mock module hierarchy
        langchain_core = types.ModuleType("langchain_core")
        langchain_core_embeddings = types.ModuleType("langchain_core.embeddings")

        class MockEmbeddings:
            """Mock base class matching LangChain's Embeddings interface."""
            def embed_documents(self, texts):
                raise NotImplementedError
            def embed_query(self, text):
                raise NotImplementedError

        langchain_core_embeddings.Embeddings = MockEmbeddings
        langchain_core.embeddings = langchain_core_embeddings
        sys.modules["langchain_core"] = langchain_core
        sys.modules["langchain_core.embeddings"] = langchain_core_embeddings
        return MockEmbeddings
    else:
        from langchain_core.embeddings import Embeddings
        return Embeddings


MockEmbeddings = setup_langchain_mock()

from turboquant_vectors.private import PrivateEncoder
from turboquant_vectors.adapters.langchain_adapter import PrivateEmbeddings


class FakeEmbeddings(MockEmbeddings):
    """Fake embedding model that returns deterministic vectors."""

    def __init__(self, dim=64):
        self.dim = dim
        self.rng = np.random.default_rng(42)

    def embed_documents(self, texts):
        vecs = self.rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs.tolist()

    def embed_query(self, text):
        vec = self.rng.standard_normal(self.dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()


class TestPrivateEmbeddings:

    def test_embed_documents_applies_rotation(self):
        base = FakeEmbeddings(dim=64)
        encoder = PrivateEncoder.generate(dim=64)
        private = PrivateEmbeddings(base_embeddings=base, encoder=encoder)

        result = private.embed_documents(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 64

        # Results should be different from what base returns (rotated)
        base2 = FakeEmbeddings(dim=64)  # same seed = same output
        raw = base2.embed_documents(["hello", "world"])
        assert not np.allclose(result, raw, atol=0.1)

    def test_embed_query_applies_rotation(self):
        base = FakeEmbeddings(dim=64)
        encoder = PrivateEncoder.generate(dim=64)
        private = PrivateEmbeddings(base_embeddings=base, encoder=encoder)

        result = private.embed_query("hello")
        assert len(result) == 64

    def test_cosine_similarity_preserved(self):
        """Cosine between two rotated vectors equals cosine between originals."""
        base = FakeEmbeddings(dim=64)
        encoder = PrivateEncoder.generate(dim=64)
        private = PrivateEmbeddings(base_embeddings=base, encoder=encoder)

        # Get raw embeddings
        base_copy = FakeEmbeddings(dim=64)
        raw = np.array(base_copy.embed_documents(["hello", "world"]))
        cos_raw = np.dot(raw[0], raw[1]) / (np.linalg.norm(raw[0]) * np.linalg.norm(raw[1]))

        # Get rotated embeddings
        rotated = np.array(private.embed_documents(["hello", "world"]))
        cos_rot = np.dot(rotated[0], rotated[1]) / (np.linalg.norm(rotated[0]) * np.linalg.norm(rotated[1]))

        np.testing.assert_allclose(cos_rot, cos_raw, atol=1e-5)

    def test_same_key_required(self):
        """Different keys produce incompatible embeddings."""
        base1 = FakeEmbeddings(dim=64)
        base2 = FakeEmbeddings(dim=64)
        enc1 = PrivateEncoder.generate(dim=64)
        enc2 = PrivateEncoder.generate(dim=64)
        private1 = PrivateEmbeddings(base_embeddings=base1, encoder=enc1)
        private2 = PrivateEmbeddings(base_embeddings=base2, encoder=enc2)

        r1 = private1.embed_documents(["hello"])
        r2 = private2.embed_documents(["hello"])
        assert not np.allclose(r1, r2, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

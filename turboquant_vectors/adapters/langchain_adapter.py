"""
LangChain adapter: wrap any Embeddings class with PrivateEncoder rotation.

Usage:
    from langchain_openai import OpenAIEmbeddings
    from turboquant_vectors import PrivateEncoder
    from turboquant_vectors.adapters.langchain_adapter import PrivateEmbeddings

    encoder = PrivateEncoder.load_key("secret.tqkey")
    base = OpenAIEmbeddings(model="text-embedding-3-small")
    private = PrivateEmbeddings(base_embeddings=base, encoder=encoder)

    # Use anywhere LangChain expects Embeddings:
    vectorstore = Chroma.from_documents(docs, private)
    results = vectorstore.similarity_search("query")

Requires: pip install langchain-core
"""

import numpy as np
from typing import List

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    raise ImportError(
        "LangChain adapter requires langchain-core: "
        "pip install langchain-core"
    )

from turboquant_vectors.private import PrivateEncoder


class PrivateEmbeddings(Embeddings):
    """
    Wraps any LangChain Embeddings with orthogonal rotation for privacy.

    All distance metrics (cosine, L2, inner product) are preserved exactly.
    The vector DB sees only rotated vectors. Published inversion attacks
    (Vec2Text, ALGEN, ZSinvert) fail completely on rotated embeddings.

    The same encoder (same key) must be used for both documents and queries.
    """

    def __init__(self, base_embeddings: Embeddings, encoder: PrivateEncoder):
        """
        Args:
            base_embeddings: Any LangChain Embeddings instance (OpenAI, Cohere, etc.)
            encoder: PrivateEncoder with the secret rotation key.
        """
        self.base_embeddings = base_embeddings
        self.encoder = encoder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents and rotate for privacy."""
        raw = self.base_embeddings.embed_documents(texts)
        raw_np = np.array(raw, dtype=np.float32)
        rotated = self.encoder.rotate(raw_np)
        return rotated.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a query and rotate with the same key."""
        raw = self.base_embeddings.embed_query(text)
        raw_np = np.array([raw], dtype=np.float32)
        rotated = self.encoder.rotate(raw_np)
        return rotated[0].tolist()

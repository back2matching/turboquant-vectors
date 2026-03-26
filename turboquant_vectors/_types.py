"""Shared types, exceptions, and protocols for turboquant-vectors."""

import numpy as np
from typing import Protocol, Tuple, runtime_checkable


class DimensionError(ValueError):
    """Raised when input vector dimension doesn't match the encoder's dimension."""

    def __init__(self, expected: int, got_shape: tuple):
        super().__init__(
            f"Dimension mismatch: encoder expects dim={expected}, "
            f"got array shape {got_shape}. "
            f"Did you use the right encoder? Check encoder.dim."
        )
        self.expected = expected
        self.got_shape = got_shape


class KeyMismatchError(ValueError):
    """Raised when a PrivateEncoder's key doesn't match a compressed index."""

    def __init__(self, index_fp: str, encoder_fp: str):
        super().__init__(
            f"Key fingerprint mismatch: index was created with key={index_fp!r}, "
            f"but encoder has key={encoder_fp!r}. Load the correct .tqkey file."
        )
        self.index_fp = index_fp
        self.encoder_fp = encoder_fp


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol satisfied by both CompressedVectors and CompressedPrivateVectors."""

    n_vectors: int
    dim: int
    bits: int

    def search(self, query: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def save(self, path) -> None:
        ...

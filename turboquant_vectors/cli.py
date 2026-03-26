"""CLI: compress and search embedding files."""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
from turboquant_vectors.core import TurboQuantVectors, CompressedVectors


def main():
    parser = argparse.ArgumentParser(
        prog="tq-vectors",
        description="Compress embeddings with TurboQuant. 6x smaller, instant, no training.",
    )
    sub = parser.add_subparsers(dest="command")

    # compress
    cp = sub.add_parser("compress", help="Compress a numpy embedding file")
    cp.add_argument("input", help="Input .npy file (float32, shape n x dim)")
    cp.add_argument("-o", "--output", help="Output .npz file (default: input.tqv.npz)")
    cp.add_argument("-b", "--bits", type=int, default=4, help="Bits per dimension (2,3,4,8)")

    # search
    sp = sub.add_parser("search", help="Search compressed vectors")
    sp.add_argument("index", help="Compressed .npz file")
    sp.add_argument("query", help="Query .npy file (single vector or batch)")
    sp.add_argument("-k", "--top-k", type=int, default=10)

    # info
    ip = sub.add_parser("info", help="Show info about compressed file")
    ip.add_argument("file", help="Compressed .npz file")

    args = parser.parse_args()

    if args.command == "compress":
        print(f"Loading {args.input}...")
        vectors = np.load(args.input)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        n, dim = vectors.shape
        orig_mb = vectors.nbytes / 1024**2

        print(f"  {n:,} vectors x {dim} dims = {orig_mb:.1f} MB")
        print(f"  Compressing at {args.bits}-bit...")

        t0 = time.time()
        tq = TurboQuantVectors(dim=dim, bits=args.bits)
        compressed = tq.compress(vectors)
        elapsed = time.time() - t0

        output = args.output or str(Path(args.input).with_suffix('')) + '.tqv.npz'
        compressed.save(output)

        comp_mb = compressed.packed_memory_bytes / 1024**2
        ratio = vectors.nbytes / compressed.packed_memory_bytes
        print(f"  Saved: {output}")
        print(f"  {orig_mb:.1f} MB -> {comp_mb:.1f} MB ({ratio:.1f}x compression)")
        print(f"  Time: {elapsed:.1f}s")

    elif args.command == "search":
        print(f"Loading index {args.index}...")
        compressed = CompressedVectors.load(args.index)
        print(f"  {compressed.n_vectors:,} vectors, {compressed.dim} dims, {compressed.bits}-bit")

        query = np.load(args.query).astype(np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]

        tq = TurboQuantVectors(dim=compressed.dim, bits=compressed.bits)

        t0 = time.time()
        indices, scores = tq.search(compressed, query, top_k=args.top_k)
        elapsed = time.time() - t0

        indices = np.atleast_2d(indices)
        scores = np.atleast_2d(scores)

        if query.shape[0] == 1:
            print(f"\nTop {args.top_k} results ({elapsed*1000:.0f}ms):")
            for i in range(min(args.top_k, indices.shape[1])):
                print(f"  {i+1}. index={int(indices[0, i])}, score={float(scores[0, i]):.4f}")
        else:
            print(f"\n{query.shape[0]} queries, {args.top_k} results each ({elapsed*1000:.0f}ms total)")
            for q in range(min(3, query.shape[0])):
                print(f"  Query {q}: top={indices[q, :3].tolist()}")

    elif args.command == "info":
        compressed = CompressedVectors.load(args.file)
        orig = compressed.original_bytes / 1024**2
        packed = compressed.packed_memory_bytes / 1024**2
        ratio = compressed.original_bytes / compressed.packed_memory_bytes
        print(f"File: {args.file}")
        print(f"Vectors: {compressed.n_vectors:,}")
        print(f"Dimensions: {compressed.dim}")
        print(f"Bits: {compressed.bits}")
        print(f"Original size: {orig:.1f} MB")
        print(f"Compressed size: {packed:.1f} MB")
        print(f"Compression ratio: {ratio:.1f}x")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

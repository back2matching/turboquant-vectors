# PrivateEncoder Integration Guide: Vector Databases, Embedding Pipelines, and Data Formats

> How PrivateEncoder plugs into every major vector DB, embedding framework, and data format.
> Research for turboquant-vectors PyPI package (Phase 4: integration examples).

**Date:** 2026-03-25
**Status:** Research complete
**Source APIs verified:** Pinecone, Weaviate v4, Qdrant, ChromaDB, Milvus/Zilliz, pgvector, sqlite-vec, LangChain, LlamaIndex, Haystack 2.0, sentence-transformers, OpenAI, FAISS, numpy, Parquet/Arrow, HuggingFace datasets

---

## Core Principle

PrivateEncoder has one job: multiply vectors by a secret orthogonal matrix before they leave the client.

```python
from turboquant_vectors import PrivateEncoder

encoder = PrivateEncoder.generate(dim=1536, seed=42)
encoder.save_key("secret.tqkey")

# Two operations. That's it.
rotated = encoder.rotate(vectors)        # before storing
rotated_query = encoder.rotate(query)    # before searching
```

Every integration below follows the same pattern:
1. **On insert:** call `encoder.rotate(vectors)` before passing to the DB
2. **On query:** call `encoder.rotate(query_vector)` before passing to the DB
3. **Results:** come back in rotated space (IDs and distances are identical, no unrotation needed)

---

## Part 1: Vector Databases

### 1.1 Pinecone

**Vector format:** `list[float]` or `numpy.ndarray`. Passed as `values` field in dict or tuple.

**Insert interception point:** Before `index.upsert()`.

**Query interception point:** Before `index.query(vector=...)`.

```python
from pinecone.grpc import PineconeGRPC as Pinecone
from turboquant_vectors import PrivateEncoder
import numpy as np

# Setup
encoder = PrivateEncoder.load_key("secret.tqkey")
pc = Pinecone(api_key="...")
index = pc.Index("my-index")

# --- INSERT ---
# Original embeddings from your model
embeddings = np.array([...])  # shape (n, 1536)
ids = ["doc-1", "doc-2", ...]
metadata = [{"title": "..."}, {"title": "..."}, ...]

# Rotate before upsert
rotated = encoder.rotate(embeddings, normalize=True)

# Upsert with dict format (current SDK)
vectors = [
    {"id": ids[i], "values": rotated[i].tolist(), "metadata": metadata[i]}
    for i in range(len(ids))
]
index.upsert(vectors=vectors, namespace="private")

# --- QUERY ---
query_embedding = np.array([...])  # shape (1536,)
rotated_query = encoder.rotate(query_embedding, normalize=True)

results = index.query(
    vector=rotated_query.tolist(),
    top_k=10,
    namespace="private",
    include_metadata=True,
)
# results.matches has identical ordering to unrotated search
```

**Batch upsert with tuple format (also supported):**
```python
vectors = [
    (ids[i], rotated[i].tolist(), metadata[i])
    for i in range(len(ids))
]
index.upsert(vectors=vectors)
```

**What works:** All distance metrics (cosine, L2, dotproduct) produce identical results. Metadata, namespaces, filtering all work normally.

**What to watch:** Pinecone normalizes vectors internally for cosine metric. Since rotation preserves norms, this is fine. Use `normalize=True` if your embeddings aren't already unit-norm to prevent norm leakage.

---

### 1.2 Weaviate (v4 client)

**Vector format:** `list[float]` passed as `vector` kwarg to `DataObject`.

**Insert interception point:** Before `collection.data.insert_many()`.

**Query interception point:** Before `collection.query.near_vector()`.

```python
import weaviate
import weaviate.classes as wvc
from turboquant_vectors import PrivateEncoder
import numpy as np

encoder = PrivateEncoder.load_key("secret.tqkey")

client = weaviate.connect_to_local()
collection = client.collections.get("Documents")

# --- INSERT ---
embeddings = np.array([...])  # shape (n, 1536)
rotated = encoder.rotate(embeddings, normalize=True)

objects = [
    wvc.data.DataObject(
        properties={"title": titles[i], "body": bodies[i]},
        vector=rotated[i].tolist(),
    )
    for i in range(len(titles))
]
collection.data.insert_many(objects)

# --- QUERY ---
query_embedding = np.array([...])  # shape (1536,)
rotated_query = encoder.rotate(query_embedding, normalize=True)

results = collection.query.near_vector(
    near_vector=rotated_query.tolist(),
    limit=10,
    return_metadata=wvc.query.MetadataQuery(distance=True),
)
```

**Collection config:** Set vectorizer to `none` (bring-your-own vectors):
```python
collection = client.collections.create(
    name="Documents",
    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
        distance_metric=wvc.config.VectorDistances.COSINE
    ),
)
```

**What works:** All distance metrics. Named vectors (multi-vector objects) work too, just rotate each named vector separately.

**What to watch:** Don't put the vector in the `properties` dict. It must be the `vector` kwarg on `DataObject`.

---

### 1.3 Qdrant

**Vector format:** `list[float]` in `PointStruct.vector`.

**Insert interception point:** Before `client.upsert()`.

**Query interception point:** Before `client.query_points()` or `client.search()`.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from turboquant_vectors import PrivateEncoder
import numpy as np

encoder = PrivateEncoder.load_key("secret.tqkey")
client = QdrantClient(url="http://localhost:6333")

# Create collection
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# --- INSERT ---
embeddings = np.array([...])  # shape (n, 1536)
rotated = encoder.rotate(embeddings, normalize=True)

points = [
    PointStruct(
        id=i,
        vector=rotated[i].tolist(),
        payload={"title": titles[i]},
    )
    for i in range(len(titles))
]
client.upsert(collection_name="docs", points=points)

# --- QUERY ---
query_embedding = np.array([...])  # shape (1536,)
rotated_query = encoder.rotate(query_embedding, normalize=True)

results = client.query_points(
    collection_name="docs",
    query=rotated_query.tolist(),
    limit=10,
)
```

**What works:** All distance metrics (Cosine, Euclid, Dot). Named vectors, sparse vectors (rotate only the dense part), payload filtering, scroll, batch operations.

**What to watch:** Qdrant supports both REST and gRPC. Rotation works with both since the vector format is the same.

---

### 1.4 ChromaDB

**Vector format:** `list[list[float]]` passed as `embeddings` kwarg.

**Insert interception point:** Before `collection.add()` or `collection.upsert()`.

**Query interception point:** Before `collection.query()`.

```python
import chromadb
from turboquant_vectors import PrivateEncoder
import numpy as np

encoder = PrivateEncoder.load_key("secret.tqkey")
client = chromadb.PersistentClient(path="./chroma_db")

# Create collection without an embedding function (bring-your-own)
collection = client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},
)

# --- INSERT ---
embeddings = np.array([...])  # shape (n, 1536)
rotated = encoder.rotate(embeddings, normalize=True)

collection.add(
    ids=["doc-1", "doc-2"],
    embeddings=rotated.tolist(),
    documents=["First doc text", "Second doc text"],
    metadatas=[{"source": "a"}, {"source": "b"}],
)

# --- QUERY ---
query_embedding = np.array([...])  # shape (1536,)
rotated_query = encoder.rotate(query_embedding, normalize=True)

results = collection.query(
    query_embeddings=[rotated_query.tolist()],
    n_results=10,
)
```

**Simplest integration.** ChromaDB accepts raw embedding lists directly. No schema config needed. Just rotate before passing.

**What works:** Cosine, L2, inner product (set via collection metadata `hnsw:space`). Metadata filtering, `where` clauses all work.

**What to watch:** If you use ChromaDB's built-in embedding function (e.g., `SentenceTransformerEmbeddingFunction`), you need to disable it and pass pre-rotated embeddings instead. Can't rotate inside ChromaDB's embedding function easily.

---

### 1.5 Milvus / Zilliz

**Vector format:** `list[list[float]]` in a dict per entity, or `list[float]` per row.

**Insert interception point:** Before `collection.insert()`.

**Query interception point:** Before `collection.search()`.

```python
from pymilvus import MilvusClient
from turboquant_vectors import PrivateEncoder
import numpy as np

encoder = PrivateEncoder.load_key("secret.tqkey")
client = MilvusClient(uri="http://localhost:19530")

# Create collection
client.create_collection(
    collection_name="docs",
    dimension=1536,
    metric_type="COSINE",
)

# --- INSERT ---
embeddings = np.array([...])  # shape (n, 1536)
rotated = encoder.rotate(embeddings, normalize=True)

data = [
    {"id": i, "vector": rotated[i].tolist(), "title": titles[i]}
    for i in range(len(titles))
]
client.insert(collection_name="docs", data=data)

# --- QUERY ---
query_embedding = np.array([...])  # shape (1536,)
rotated_query = encoder.rotate(query_embedding, normalize=True)

results = client.search(
    collection_name="docs",
    data=[rotated_query.tolist()],
    limit=10,
    output_fields=["title"],
)
```

**What works:** All metric types (COSINE, L2, IP). Partition keys, dynamic schema, hybrid search (rotate only the dense vectors, leave sparse alone).

**What to watch:** Milvus can generate embeddings server-side via built-in embedding functions. Don't use those if you want rotation privacy. Bring your own vectors.

---

### 1.6 pgvector (PostgreSQL)

**Vector format:** `list[float]` or string `'[0.1, 0.2, ...]'` in SQL.

**Insert interception point:** Before the INSERT statement.

**Query interception point:** Before the SELECT with distance operator.

```python
import psycopg2
from pgvector.psycopg2 import register_vector
from turboquant_vectors import PrivateEncoder
import numpy as np

encoder = PrivateEncoder.load_key("secret.tqkey")

conn = psycopg2.connect("postgresql://localhost/mydb")
register_vector(conn)

cur = conn.cursor()
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        title TEXT,
        embedding vector(1536)
    )
""")

# --- INSERT ---
embeddings = np.array([...])  # shape (n, 1536)
rotated = encoder.rotate(embeddings, normalize=True)

for i in range(len(rotated)):
    cur.execute(
        "INSERT INTO documents (title, embedding) VALUES (%s, %s)",
        (titles[i], rotated[i].tolist()),
    )
conn.commit()

# --- QUERY ---
query_embedding = np.array([...])  # shape (1536,)
rotated_query = encoder.rotate(query_embedding, normalize=True)

cur.execute(
    "SELECT id, title, embedding <=> %s AS distance FROM documents ORDER BY distance LIMIT 10",
    (rotated_query.tolist(),),
)
results = cur.fetchall()
```

**SQLAlchemy version:**
```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, Session
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    embedding = Column(Vector(1536))

# Insert
with Session(engine) as session:
    doc = Document(title="...", embedding=rotated[0].tolist())
    session.add(doc)
    session.commit()

# Query (cosine distance)
from sqlalchemy import select
stmt = select(Document).order_by(
    Document.embedding.cosine_distance(rotated_query.tolist())
).limit(10)
```

**What works:** All pgvector operators: `<->` (L2), `<=>` (cosine), `<#>` (inner product). IVFFlat and HNSW indices work normally.

**What to watch:** pgvector stores vectors as PostgreSQL arrays. The rotation doesn't change the dimension, so no schema changes needed. Just rotate before INSERT and before the distance operator in SELECT.

---

### 1.7 sqlite-vec (FlockRun)

**Vector format:** `Buffer` (Node.js) or `bytes` (Python) wrapping Float32Array.

**How FlockRun does it today** (from `src/core/infra/vector-search.ts`):

```typescript
// Insert: embed text, store as Buffer
const [embedding] = await embed([text]);
await driver.run(
  'INSERT INTO vec_knowledge(entry_id, embedding) VALUES (?, ?)',
  [entryId, Buffer.from(embedding.buffer)],
);

// Query: embed query, MATCH operator
const [queryEmbedding] = await embed([query]);
const rows = await driver.all(
  'SELECT entry_id, distance FROM vec_knowledge WHERE embedding MATCH ? ORDER BY distance LIMIT ?',
  [Buffer.from(queryEmbedding.buffer), maxResults],
);
```

**Where rotation fits:** Between `embed()` and the SQL insert/query.

```typescript
// Hypothetical FlockRun integration (TypeScript)
import { PrivateEncoder } from './private-encoder';  // JS/TS port or WASM

const encoder = PrivateEncoder.loadKey('./secret.tqkey');

// Insert
const [embedding] = await embed([text]);
const rotated = encoder.rotate(embedding);  // Float32Array -> Float32Array
await driver.run(
  'INSERT INTO vec_knowledge(entry_id, embedding) VALUES (?, ?)',
  [entryId, Buffer.from(rotated.buffer)],
);

// Query
const [queryEmbedding] = await embed([query]);
const rotatedQuery = encoder.rotate(queryEmbedding);
const rows = await driver.all(
  'SELECT entry_id, distance FROM vec_knowledge WHERE embedding MATCH ? ORDER BY distance LIMIT ?',
  [Buffer.from(rotatedQuery.buffer), maxResults],
);
```

**Python version (sqlite-vec via apsw or sqlite3):**
```python
import sqlite3
import struct
from turboquant_vectors import PrivateEncoder
import numpy as np

encoder = PrivateEncoder.load_key("secret.tqkey")

def serialize_f32(vector: np.ndarray) -> bytes:
    """Convert numpy array to little-endian float32 bytes for sqlite-vec."""
    return vector.astype(np.float32).tobytes()

conn = sqlite3.connect("flockrun.db")
# Assumes sqlite-vec extension is loaded and vec_knowledge table exists

# --- INSERT ---
embedding = np.array([...], dtype=np.float32)  # shape (384,) for all-MiniLM-L6-v2
rotated = encoder.rotate(embedding, normalize=True)
conn.execute(
    "INSERT INTO vec_knowledge(entry_id, embedding) VALUES (?, ?)",
    (entry_id, serialize_f32(rotated)),
)

# --- QUERY ---
query_embedding = np.array([...], dtype=np.float32)
rotated_query = encoder.rotate(query_embedding, normalize=True)
rows = conn.execute(
    "SELECT entry_id, distance FROM vec_knowledge WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
    (serialize_f32(rotated_query), 10),
).fetchall()
```

**FlockRun note:** The default embedding model is `all-MiniLM-L6-v2` (384 dimensions). The `.tqkey` file would be 384 x 384 x 4 = 589 KB. The `vec_knowledge` virtual table uses `float[384]`.

---

## Part 2: Embedding Pipelines

### 2.1 LangChain

**Hook point:** Wrap the `Embeddings` base class. Two methods to override:
- `embed_documents(texts: list[str]) -> list[list[float]]`
- `embed_query(text: str) -> list[float]`

```python
from langchain_core.embeddings import Embeddings
from turboquant_vectors import PrivateEncoder
import numpy as np


class PrivateEmbeddings(Embeddings):
    """Wraps any LangChain Embeddings with PrivateEncoder rotation."""

    def __init__(self, base: Embeddings, encoder: PrivateEncoder, normalize: bool = True):
        self._base = base
        self._encoder = encoder
        self._normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raw = self._base.embed_documents(texts)
        vectors = np.array(raw, dtype=np.float32)
        rotated = self._encoder.rotate(vectors, normalize=self._normalize)
        return rotated.tolist()

    def embed_query(self, text: str) -> list[float]:
        raw = self._base.embed_query(text)
        vector = np.array(raw, dtype=np.float32)
        rotated = self._encoder.rotate(vector, normalize=self._normalize)
        return rotated.tolist()
```

**Usage with any VectorStore:**
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Original
base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Wrapped with rotation
encoder = PrivateEncoder.load_key("secret.tqkey")
private_embeddings = PrivateEmbeddings(base_embeddings, encoder)

# Use anywhere LangChain expects an Embeddings object
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=private_embeddings,  # drop-in replacement
    persist_directory="./chroma_private",
)

# Queries are automatically rotated too
results = vectorstore.similarity_search("my query", k=10)
```

**This is the cleanest integration.** One wrapper class, works with every LangChain VectorStore (Pinecone, Weaviate, Qdrant, Chroma, pgvector, Milvus, FAISS, etc.) because they all accept an `Embeddings` object.

---

### 2.2 LlamaIndex

**Hook point:** Wrap the `BaseEmbedding` class. Two methods to override:
- `_get_text_embedding(text: str) -> list[float]`
- `_get_query_embedding(text: str) -> list[float]`

```python
from llama_index.core.embeddings import BaseEmbedding
from turboquant_vectors import PrivateEncoder
import numpy as np


class PrivateEmbedding(BaseEmbedding):
    """Wraps any LlamaIndex embedding model with PrivateEncoder rotation."""

    def __init__(self, base: BaseEmbedding, encoder: PrivateEncoder, normalize: bool = True):
        super().__init__(model_name=f"private({base.model_name})")
        self._base = base
        self._encoder = encoder
        self._normalize = normalize

    def _get_text_embedding(self, text: str) -> list[float]:
        raw = self._base.get_text_embedding(text)
        vector = np.array(raw, dtype=np.float32)
        rotated = self._encoder.rotate(vector, normalize=self._normalize)
        return rotated.tolist()

    def _get_query_embedding(self, text: str) -> list[float]:
        raw = self._base.get_query_embedding(text)
        vector = np.array(raw, dtype=np.float32)
        rotated = self._encoder.rotate(vector, normalize=self._normalize)
        return rotated.tolist()

    async def _aget_query_embedding(self, text: str) -> list[float]:
        return self._get_query_embedding(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
```

**Usage:**
```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

base = OpenAIEmbedding(model="text-embedding-3-small")
encoder = PrivateEncoder.load_key("secret.tqkey")

Settings.embed_model = PrivateEmbedding(base, encoder)

documents = SimpleDirectoryReader("./data/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Queries automatically rotated
query_engine = index.as_query_engine()
response = query_engine.query("What is the revenue?")
```

**Same pattern as LangChain.** One wrapper, works with every LlamaIndex VectorStore backend.

---

### 2.3 Haystack 2.0

**Hook point:** Create custom `DocumentEmbedder` and `TextEmbedder` components using the `@component` decorator.

```python
from haystack import Document, component
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from turboquant_vectors import PrivateEncoder
import numpy as np


@component
class PrivateDocumentEmbedder:
    """Wraps a Haystack DocumentEmbedder with PrivateEncoder rotation."""

    def __init__(self, base_model: str, encoder: PrivateEncoder, normalize: bool = True):
        self._embedder = SentenceTransformersDocumentEmbedder(model=base_model)
        self._encoder = encoder
        self._normalize = normalize

    def warm_up(self):
        self._embedder.warm_up()

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        result = self._embedder.run(documents=documents)
        for doc in result["documents"]:
            if doc.embedding is not None:
                vec = np.array(doc.embedding, dtype=np.float32)
                doc.embedding = self._encoder.rotate(vec, normalize=self._normalize).tolist()
        return result


@component
class PrivateTextEmbedder:
    """Wraps a Haystack TextEmbedder with PrivateEncoder rotation."""

    def __init__(self, base_model: str, encoder: PrivateEncoder, normalize: bool = True):
        self._embedder = SentenceTransformersTextEmbedder(model=base_model)
        self._encoder = encoder
        self._normalize = normalize

    def warm_up(self):
        self._embedder.warm_up()

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict:
        result = self._embedder.run(text=text)
        vec = np.array(result["embedding"], dtype=np.float32)
        rotated = self._encoder.rotate(vec, normalize=self._normalize)
        return {"embedding": rotated.tolist()}
```

**Usage in a pipeline:**
```python
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

encoder = PrivateEncoder.load_key("secret.tqkey")
doc_store = QdrantDocumentStore(location="http://localhost:6333")

indexing = Pipeline()
indexing.add_component("embedder", PrivateDocumentEmbedder("all-MiniLM-L6-v2", encoder))
indexing.add_component("writer", DocumentWriter(document_store=doc_store))
indexing.connect("embedder", "writer")

indexing.run({"embedder": {"documents": documents}})
```

---

### 2.4 Sentence-Transformers

**Hook point:** After `model.encode()`. Returns `numpy.ndarray` by default.

```python
from sentence_transformers import SentenceTransformer
from turboquant_vectors import PrivateEncoder
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
encoder = PrivateEncoder.load_key("secret.tqkey")

# --- EMBED + ROTATE (batch) ---
sentences = ["First document", "Second document", "Third document"]
embeddings = model.encode(sentences)  # numpy array, shape (3, 384)
rotated = encoder.rotate(embeddings, normalize=True)

# --- EMBED + ROTATE (single query) ---
query = "search query"
query_embedding = model.encode(query)  # numpy array, shape (384,)
rotated_query = encoder.rotate(query_embedding, normalize=True)
```

**This is the simplest integration.** `model.encode()` returns numpy, `encoder.rotate()` accepts numpy. One line of code.

**Convenience wrapper (optional):**
```python
def private_encode(model, encoder, texts, normalize=True):
    """Encode and rotate in one call."""
    embeddings = model.encode(texts, convert_to_numpy=True)
    return encoder.rotate(embeddings, normalize=normalize)
```

---

### 2.5 OpenAI Embeddings API

**Hook point:** After `client.embeddings.create()`. Response is `list[float]` per embedding.

```python
from openai import OpenAI
from turboquant_vectors import PrivateEncoder
import numpy as np

client = OpenAI()
encoder = PrivateEncoder.load_key("secret.tqkey")

# --- SINGLE ---
response = client.embeddings.create(
    input="Your text here",
    model="text-embedding-3-small",
)
raw = np.array(response.data[0].embedding, dtype=np.float32)  # shape (1536,)
rotated = encoder.rotate(raw, normalize=True)

# --- BATCH ---
response = client.embeddings.create(
    input=["Text 1", "Text 2", "Text 3"],
    model="text-embedding-3-small",
)
raw = np.array([d.embedding for d in response.data], dtype=np.float32)  # shape (3, 1536)
rotated = encoder.rotate(raw, normalize=True)
```

**Convenience wrapper:**
```python
def private_embed(client, encoder, texts, model="text-embedding-3-small", normalize=True):
    """Call OpenAI embeddings API and rotate the result."""
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(input=texts, model=model)
    raw = np.array([d.embedding for d in response.data], dtype=np.float32)
    rotated = encoder.rotate(raw, normalize=normalize)
    if len(texts) == 1:
        return rotated[0]
    return rotated
```

**OpenAI note:** `text-embedding-3-small` produces 1536-dim vectors. `text-embedding-3-large` produces 3072-dim. OpenAI already L2-normalizes the output, so `normalize=True` is a no-op but harmless.

---

## Part 3: Data Formats

### 3.1 NumPy .npy Files

**Trivial.** Load, rotate, save.

```python
import numpy as np
from turboquant_vectors import PrivateEncoder

encoder = PrivateEncoder.load_key("secret.tqkey")

# Rotate an existing embedding file
vectors = np.load("embeddings.npy")          # shape (n, d)
rotated = encoder.rotate(vectors, normalize=True)
np.save("embeddings_private.npy", rotated)

# Rotate back (requires the key)
recovered = encoder.unrotate(rotated)
assert np.allclose(vectors, recovered, atol=1e-6)
```

**Batch processing large files with memmap:**
```python
# For files that don't fit in RAM
vectors = np.load("huge_embeddings.npy", mmap_mode="r")
n, d = vectors.shape
out = np.memmap("huge_private.npy", dtype=np.float32, mode="w+", shape=(n, d))

batch_size = 100_000
for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    out[start:end] = encoder.rotate(vectors[start:end], normalize=True)
    out.flush()
```

---

### 3.2 FAISS Indices

**Can you rotate vectors inside an existing FAISS index?** Not in-place. FAISS treats vectors as immutable once added.

**Strategy: rotate before building the index.**

```python
import faiss
import numpy as np
from turboquant_vectors import PrivateEncoder

encoder = PrivateEncoder.load_key("secret.tqkey")

# Original vectors
vectors = np.load("embeddings.npy").astype(np.float32)
n, d = vectors.shape

# Rotate before building
rotated = encoder.rotate(vectors, normalize=True)

# Build FAISS index on rotated vectors
index = faiss.IndexFlatIP(d)  # inner product (cosine if normalized)
index.add(rotated)

# Query with rotated query
query = np.array([...], dtype=np.float32).reshape(1, d)
rotated_query = encoder.rotate(query, normalize=True)
distances, indices = index.search(rotated_query, k=10)

# Save the index (contains only rotated vectors)
faiss.write_index(index, "private_index.faiss")
```

**Converting an existing FAISS index:**
```python
# Extract vectors from existing index, rotate, rebuild
existing = faiss.read_index("original.faiss")
n = existing.ntotal
d = existing.d

# Reconstruct all vectors (only works with IndexFlat, IndexIVFFlat with direct_map, etc.)
original_vectors = existing.reconstruct_n(0, n)
rotated = encoder.rotate(original_vectors, normalize=True)

# Build new index
new_index = faiss.IndexFlatIP(d)
new_index.add(rotated)
faiss.write_index(new_index, "private_index.faiss")
```

**What works:** IndexFlat, IndexIVFFlat, IndexHNSW, IndexPQ, IndexIVFPQ. The rotation doesn't change the dimension, so all index types work. Distance metrics (L2, IP) produce identical results.

**What doesn't work:** You can't rotate vectors inside an IndexIVFPQ without retraining the PQ codebook. Reconstruct + rotate + rebuild from scratch is required.

**FAISS + IVF gotcha:** IVF indices train cluster centroids on the data. If you rotate the data, you must retrain the centroids on the rotated data. Don't train on originals and then add rotated vectors.

---

### 3.3 Parquet / Apache Arrow

**For large embedding datasets stored as Parquet files.**

```python
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from turboquant_vectors import PrivateEncoder

encoder = PrivateEncoder.load_key("secret.tqkey")

# --- READ + ROTATE + WRITE ---
table = pq.read_table("embeddings.parquet")

# Extract embedding column (stored as list<float32>)
embedding_col = table.column("embedding")

# Convert to numpy, rotate, convert back
batch_size = 100_000
rotated_chunks = []

for i in range(0, len(embedding_col), batch_size):
    chunk = embedding_col[i:i + batch_size]
    # Convert Arrow list column to numpy 2D array
    vectors = np.array([row.as_py() for row in chunk], dtype=np.float32)
    rotated = encoder.rotate(vectors, normalize=True)
    # Convert back to Arrow list column
    rotated_list = [row.tolist() for row in rotated]
    rotated_chunks.extend(rotated_list)

# Replace the embedding column
new_col = pa.array(rotated_chunks, type=pa.list_(pa.float32()))
col_idx = table.schema.get_field_index("embedding")
new_table = table.set_column(col_idx, "embedding", new_col)

pq.write_table(new_table, "embeddings_private.parquet")
```

**Efficient version with pyarrow fixed-size lists:**
```python
# If embeddings are stored as fixed_size_list<float32, 1536>
# (more memory-efficient than variable-length lists)
import pyarrow.compute as pc

# Read as fixed-size list
table = pq.read_table("embeddings.parquet")
emb_col = table.column("embedding")

# Flatten to 1D, reshape to (n, d), rotate, reshape back
flat = emb_col.values.to_numpy()
d = 1536  # known dimension
n = len(flat) // d
vectors = flat.reshape(n, d)
rotated = encoder.rotate(vectors, normalize=True)

# Rebuild the fixed-size list column
flat_rotated = pa.array(rotated.flatten(), type=pa.float32())
new_col = pa.FixedSizeListArray.from_arrays(flat_rotated, d)
col_idx = table.schema.get_field_index("embedding")
new_table = table.set_column(col_idx, "embedding", new_col)
pq.write_table(new_table, "embeddings_private.parquet")
```

---

### 3.4 HuggingFace Datasets

**Many open embedding datasets are published this way.**

```python
from datasets import load_dataset, Features, Sequence, Value
from turboquant_vectors import PrivateEncoder
import numpy as np

encoder = PrivateEncoder.load_key("secret.tqkey")

# Load dataset with embeddings
ds = load_dataset("mteb/stsbenchmark-sts", split="test")

# If embeddings are pre-computed as a column
# ds = ds.map(lambda x: {"embedding": model.encode(x["sentence"])})

def rotate_embeddings(batch):
    """Map function to rotate all embeddings in a batch."""
    vectors = np.array(batch["embedding"], dtype=np.float32)
    rotated = encoder.rotate(vectors, normalize=True)
    batch["embedding"] = rotated.tolist()
    return batch

# Apply rotation in batches (efficient)
ds_private = ds.map(rotate_embeddings, batched=True, batch_size=10000)

# Save to disk
ds_private.save_to_disk("./private_embeddings")

# Or push to hub (vectors are rotated, originals never uploaded)
ds_private.push_to_hub("your-name/private-stsbenchmark")
```

**Arrow format note:** HuggingFace datasets are backed by Arrow. Embeddings are typically stored as `Sequence(Value('float32'), length=dim)`. The rotation doesn't change the type or dimension, so no schema changes needed.

---

## Part 4: Architecture Decision: Wrapper Classes vs. Utility Functions

### Option A: Utility function only (recommended for v1)

```python
from turboquant_vectors import PrivateEncoder

encoder = PrivateEncoder.load_key("secret.tqkey")
rotated = encoder.rotate(vectors)
```

**Pros:**
- Zero coupling to any specific DB or framework
- Users know exactly what's happening
- No maintenance burden when DB APIs change
- Works with every DB, present and future

**Cons:**
- User must remember to rotate both inserts and queries
- Easy to accidentally insert unrotated vectors

### Option B: Framework wrappers (recommended for v2)

```python
from turboquant_vectors.integrations.langchain import PrivateEmbeddings
from turboquant_vectors.integrations.llamaindex import PrivateEmbedding
```

**Pros:**
- Can't forget to rotate (it's automatic)
- Drop-in replacement pattern
- More Pythonic

**Cons:**
- Maintenance burden (track API changes for each framework)
- Extra dependencies (langchain-core, llama-index-core, etc.)
- Must be optional extras: `pip install turboquant-vectors[langchain]`

### Option C: DB-specific wrappers (NOT recommended)

```python
from turboquant_vectors.integrations.pinecone import PrivatePineconeIndex
```

**Why not:** DB clients change APIs frequently. Pinecone v3 SDK was a breaking rewrite. Weaviate v3 -> v4 was a breaking rewrite. Maintaining wrapper classes for 7+ DBs is a support nightmare. The rotation is a one-liner; the wrapper adds complexity for marginal benefit.

### Recommendation

**v1.0: Utility function only + code examples.** Ship `PrivateEncoder.rotate()` and include the examples from this document in the README. Users copy 2 lines of code.

**v1.1+: LangChain and LlamaIndex wrappers only.** These two cover 80%+ of the Python RAG ecosystem. They wrap the Embeddings interface, not the DB interface, so they work with all backends automatically. Ship as optional extras.

**Never ship DB-specific wrappers.** The examples in this doc are the documentation. Users adapt them to their specific SDK version.

---

## Part 5: Quick Reference Matrix

| Platform | Vector Format | Insert Hook | Query Hook | Difficulty |
|----------|--------------|-------------|------------|------------|
| Pinecone | `list[float]` in dict/tuple | Before `index.upsert()` | Before `index.query()` | Trivial |
| Weaviate v4 | `list[float]` in `DataObject.vector` | Before `insert_many()` | Before `near_vector()` | Trivial |
| Qdrant | `list[float]` in `PointStruct.vector` | Before `client.upsert()` | Before `client.query_points()` | Trivial |
| ChromaDB | `list[list[float]]` as `embeddings` kwarg | Before `collection.add()` | Before `collection.query()` | Trivial |
| Milvus | `list[float]` in entity dict | Before `collection.insert()` | Before `collection.search()` | Trivial |
| pgvector | `list[float]` in SQL param | Before INSERT | Before SELECT with distance op | Trivial |
| sqlite-vec | `bytes` (Float32Array buffer) | Before INSERT | Before MATCH query | Trivial |
| LangChain | Wrap `Embeddings` class | `embed_documents()` | `embed_query()` | Easy (15 LOC) |
| LlamaIndex | Wrap `BaseEmbedding` class | `_get_text_embedding()` | `_get_query_embedding()` | Easy (20 LOC) |
| Haystack 2.0 | `@component` decorator | Custom `DocumentEmbedder` | Custom `TextEmbedder` | Medium (30 LOC) |
| sentence-transformers | `numpy.ndarray` from `model.encode()` | After `encode()` | After `encode()` | Trivial (1 line) |
| OpenAI API | `list[float]` from response | After `embeddings.create()` | After `embeddings.create()` | Trivial (2 lines) |
| NumPy .npy | `numpy.ndarray` | `np.load()` -> rotate -> `np.save()` | Same | Trivial |
| FAISS | `numpy.ndarray` added to index | Before `index.add()` | Before `index.search()` | Easy (must rebuild) |
| Parquet/Arrow | `list<float32>` column | Read -> rotate -> write | N/A | Medium (batched) |
| HF Datasets | `Sequence(float32)` column | `ds.map()` with rotation | N/A | Easy |

**Bottom line:** Every integration is either "rotate before insert, rotate before query" (for DBs) or "rotate after embed" (for pipelines). No platform requires special handling. The math guarantees this: rotation is a linear transformation that commutes with distance computation.

---

## Appendix: What Doesn't Work

1. **L1 (Manhattan) distance:** Not preserved by rotation. Users of L1 distance can't use PrivateEncoder. This is rare in practice (cosine and L2 dominate vector search).

2. **Server-side embedding:** If the DB generates embeddings server-side (Weaviate with `text2vec-openai`, Milvus with built-in functions), you can't rotate because you never see the raw vectors. Solution: disable server-side embedding, embed client-side, rotate, then insert.

3. **Quantization-aware search (e.g., FAISS PQ):** Product Quantization trains codebooks on the data distribution. Rotating the data changes which coordinates are grouped together, so you must retrain PQ after rotation. This isn't a problem (train on rotated data), but you can't rotate vectors inside an already-trained PQ index.

4. **Matryoshka embeddings (dimension truncation):** If you use OpenAI's dimension reduction (`dimensions=512` for a 1536-dim model), rotation must happen on the truncated vector, not the full-dim vector. Generate a key matching the truncated dimension.

5. **Sparse vectors:** Rotation is defined for dense vectors only. For hybrid search with sparse+dense, rotate only the dense component. Leave sparse (BM25, SPLADE) untouched.

# trysac-vector-store-lite

Embedded vector store on stdlib sqlite3 with float32 blobs, metadata filters compiled to parameterised SQL, exact search and seeded random-hyperplane LSH.

## Problem

Small RAG projects need persistent vector search with metadata filters, but a dedicated vector database server is heavy, and an in-memory list is lost on restart. SQLite is ubiquitous, yet ad-hoc vector implementations usually interpolate filter values into SQL queries, leading to SQL injection risks and type confusion.

## Installation

```bash
pip install trysac-vector-store-lite
```

## Usage

```python
from vector_store_lite import Record, VectorStore

with VectorStore.open(":memory:", dim=3, metric="cosine") as store:
    store.upsert(
        [
            Record(id="doc1", vector=[1.0, 0.0, 0.0], metadata={"topic": "ai"}),
            Record(id="doc2", vector=[0.0, 1.0, 0.0], metadata={"topic": "db"}),
        ]
    )
    hits = store.search([1.0, 0.1, 0.0], k=1, where={"topic": "ai"})
    print(hits[0].id, hits[0].score)
```

## Public API Summary

- `VectorStore.open(path, *, dim, metric='cosine', lsh_bits=16, lsh_tables=4, seed=0, on_event=None)`: Open/create a SQLite-backed store.
- `upsert(records: Iterable[Record]) -> int`: Atomic insertion/update.
- `delete(ids: Iterable[str]) -> int`: Atomic deletion.
- `search(vector, *, k=10, where=None, exact=False) -> list[Hit]`: Perform exact scan or LSH index search.
- `count(where=None) -> int`: Returns count of records matching filter.
- `VectorStoreError`: Raised for any operation failure (codes: `dimension_mismatch`, `dimension_conflict`, `invalid_vector`, `invalid_filter`, `invalid_id`, `invalid_metadata`, `invalid_k`, `invalid_config`).

## Why not Chroma or sqlite-vec?

Unlike Chroma, `trysac-vector-store-lite` requires no server/daemon. Unlike `sqlite-vec`, it relies solely on the standard library (`sqlite3` with bundled `json1`) and requires no compilation of native C/Rust extensions, making it portable and zero-dependency.

See [docs/SECURITY.md](docs/SECURITY.md) for vulnerability policy.
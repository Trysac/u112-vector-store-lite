# Project record

## Objective

Embedded vector store on stdlib sqlite3 with float32 blobs, parameterised metadata filters, exact search and seeded random-hyperplane LSH.

## Scope

- The public Python API (`VectorStore`, `Record`, `Hit`, `VectorStoreError`).
- Persistence via SQLite.
- Metadata filtering using parameterized SQL (via `json_extract`/`json_type`).
- LSH approximate search.

## Out-of-scope

- HNSW or IVF indexing.
- Third-party extensions like `sqlite-vec`.
- Hybrid keyword/vector search.
- Concurrency across processes beyond SQLite's standard file locking.
- Exact search performance for datasets > 100,000 vectors.

## Effort metrics

| Metric | Value |
|---|---|
| Tokens (input / output) | 1110346 / 58914 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.3131 |

## Factory run u112 (measured from API usage)

| Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|
| generation | gemini-3.8-flash | yes | 4686 | 48061 | 0 | 0 | 0.0919 |
| fix-1 | gemini-3.8-flash | no | 80150 | 7996 | 0 | 662859 | 0.1398 |
| fix-2 | gemini-3.8-flash | no | 29891 | 716 | 0 | 88117 | 0.0317 |
| audit | gemini-3.1-flash-lite | no | 45983 | 267 | 0 | 65177 | 0.0135 |
| docs | gemini-3.1-flash-lite | no | 133483 | 1874 | 0 | 0 | 0.0362 |

Total: $0.3131. Prices: Anthropic list prices.

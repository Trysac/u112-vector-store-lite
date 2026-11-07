# Architecture

`trysac-vector-store-lite` is a zero-dependency wrapper around `sqlite3`.

## Tables

1. `meta`: Stores store configuration (dim, metric, LSH params, seed, hyperplanes).
2. `items`: Stores `id`, raw `vec` (as little-endian float32 byte blob), and `meta` (as serialized JSON string).
3. `buckets`: Maps signatures to IDs for LSH. Indexed by `(t, sig)`.

## Search mechanism

- **Exact**: Scans the `items` table directly, filtering by the `where` clause.
- **Approximate**: Uses `buckets` to identify IDs with matching LSH signatures in any LSH table, joins with `items` to filter by the `where` clause, then performs exact re-ranking of the candidate set.
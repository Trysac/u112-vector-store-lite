# Security notes

## Threat model

### Defended

- **SQL Injection**: Metadata filters are compiled into parameterized SQL using `json_extract()` and `json_type()`. No user-provided field names or values are interpolated as strings into the raw SQL query.
- **Type Confusion**: Boolean values are compared using `json_type(..., '$.field')`, ensuring that a metadata boolean `True` never matches an integer `1`.

### Not defended

- **Concurrency**: The library is not thread-safe. Using the same `VectorStore` instance across threads may corrupt state.
- **File Integrity**: Malicious modification of the underlying SQLite file can lead to arbitrary data return or crash the reader.
- **Memory Exhaustion**: The library does not limit the size of loaded metadata dicts or result lists beyond standard Python/SQLite resource limits.
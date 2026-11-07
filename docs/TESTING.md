# Testing

Tests are located in `tests/test_vector_store_lite.py`.

## Coverage

- **Acceptance tests**: Verifying vector search accuracy (cosine, L2) using `pytest.approx`.
- **CRUD operations**: Upserting, replacing, and deleting records; verifying `count()` consistency.
- **Filtering**: `where` clause support (`$and`, `$eq`, `$in`, etc.), including edge cases like `null` handling and field injection protection.
- **Validation**: Error raising for invalid vector lengths, NaNs, zero vectors (in cosine mode), invalid IDs, metadata sizes, and parameter bounds (`k`, `dim`).
- **File persistence**: Verifying LSH hyperplanes are preserved and behavior remains deterministic upon reopening a store file.
- **Events**: Verifying `on_event` reports correctly on search statistics.
"""Comprehensive tests for trysac-vector-store-lite."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from vector_store_lite import Record, VectorStore, VectorStoreError


def test_acceptance_cosine_search() -> None:
    store = VectorStore.open(":memory:", dim=3)
    store.upsert(
        [
            Record(id="a", vector=[1, 0, 0]),
            Record(id="b", vector=[1, 1, 0]),
            Record(id="c", vector=[0, 0, 1]),
        ]
    )
    hits = store.search([1, 0, 0], k=2, exact=True)
    assert [h.id for h in hits] == ["a", "b"]
    assert [h.score for h in hits] == pytest.approx([1.0, 0.7071067811865475], abs=1e-6)
    store.close()


def test_acceptance_upsert_replace_and_delete() -> None:
    store = VectorStore.open(":memory:", dim=3)
    store.upsert(
        [
            Record(id="a", vector=[1, 0, 0]),
            Record(id="b", vector=[0, 1, 0]),
            Record(id="c", vector=[0, 0, 1]),
        ]
    )
    assert store.count() == 3

    # Upserting existing id replaces vector and metadata, count stays 3
    store.upsert([Record(id="a", vector=[1, 1, 1], metadata={"updated": True})])
    assert store.count() == 3

    # delete(['a', 'zzz']) returns 1
    del_count = store.delete(["a", "zzz"])
    assert del_count == 1
    assert store.count() == 2
    store.close()


def test_acceptance_metadata_filters() -> None:
    store = VectorStore.open(":memory:", dim=2)
    v = [1.0, 0.0]
    store.upsert(
        [
            Record(id="r1", vector=v, metadata={"lang": "en", "year": 2024, "flag": True}),
            Record(id="r2", vector=v, metadata={"lang": "es", "year": 2023, "flag": 1}),
            Record(id="r3", vector=v, metadata={"lang": "fr", "year": 2025, "flag": False}),
            Record(id="r4", vector=v, metadata={"lang": "en", "year": 2022}),
        ]
    )

    # search(v, where={'lang': 'en', 'year': {'$gte': 2024}}, exact=True) returns only matching ids
    hits = store.search(v, where={"lang": "en", "year": {"$gte": 2024}}, exact=True)
    assert [h.id for h in hits] == ["r1"]

    # count(where={'lang': {'$in': ['en', 'es']}}) counts both languages
    c = store.count(where={"lang": {"$in": ["en", "es"]}})
    assert c == 3  # r1, r2, r4

    # where={'flag': True} never matches a record whose flag is 1
    hits_flag = store.search(v, where={"flag": True}, exact=True)
    assert [h.id for h in hits_flag] == ["r1"]
    hits_flag_int = store.search(v, where={"flag": 1}, exact=True)
    assert [h.id for h in hits_flag_int] == ["r2"]
    store.close()


def test_acceptance_invalid_filters() -> None:
    store = VectorStore.open(":memory:", dim=2)
    v = [1.0, 0.0]
    store.upsert([Record(id="r1", vector=v, metadata={"val": 10})])

    # where={'$or': []} raises VectorStoreError with code 'invalid_filter'
    with pytest.raises(VectorStoreError) as exc:
        store.search(v, where={"$or": []})
    assert exc.value.code == "invalid_filter"

    # a field name 'a"; DROP' raises 'invalid_filter'
    with pytest.raises(VectorStoreError) as exc:
        store.search(v, where={'a"; DROP': 1})
    assert exc.value.code == "invalid_filter"

    # 17 conditions raise 'invalid_filter'
    conds_17: dict[str, Any] = {f"f_{i}": i for i in range(17)}
    with pytest.raises(VectorStoreError) as exc:
        store.search(v, where=conds_17)
    assert exc.value.code == "invalid_filter"
    store.close()


def test_acceptance_sql_injection_safety() -> None:
    store = VectorStore.open(":memory:", dim=2)
    v = [1.0, 0.0]
    store.upsert(
        [
            Record(id="victim", vector=v, metadata={"name": "x' OR 1=1 --"}),
            Record(id="other", vector=v, metadata={"name": "safe"}),
        ]
    )

    hits = store.search(v, where={"name": "x' OR 1=1 --"}, exact=True)
    assert len(hits) == 1
    assert hits[0].id == "victim"
    assert hits[0].metadata["name"] == "x' OR 1=1 --"
    store.close()


def test_acceptance_vector_and_dimension_validation(tmp_path: Path) -> None:
    db_path = tmp_path / "store.db"
    store = VectorStore.open(db_path, dim=3, metric="cosine")

    # A vector of length 4 in a dim=3 store raises code 'dimension_mismatch'
    with pytest.raises(VectorStoreError) as exc:
        store.upsert([Record(id="x", vector=[1, 0, 0, 0])])
    assert exc.value.code == "dimension_mismatch"

    # a NaN component raises 'invalid_vector'
    with pytest.raises(VectorStoreError) as exc:
        store.upsert([Record(id="x", vector=[1.0, float("nan"), 0.0])])
    assert exc.value.code == "invalid_vector"

    # a zero vector with metric 'cosine' raises 'invalid_vector'
    with pytest.raises(VectorStoreError) as exc:
        store.upsert([Record(id="x", vector=[0.0, 0.0, 0.0])])
    assert exc.value.code == "invalid_vector"

    store.close()

    # reopening a file store with dim=4 raises 'dimension_conflict'
    with pytest.raises(VectorStoreError) as exc:
        VectorStore.open(db_path, dim=4, metric="cosine")
    assert exc.value.code == "dimension_conflict"


def test_acceptance_lsh_approximate_recall_and_candidates() -> None:
    # "For 2_000 vectors of dim 32 from random.Random(1) and 50 queries made by adding
    # Gaussian noise 0.05 to stored vectors, approximate search with k=1 returns the source id
    # for at least 45 of the 50 queries, and the mean 'candidates' value reported to on_event is below 100"
    events: list[dict[str, Any]] = []
    store = VectorStore.open(
        ":memory:",
        dim=32,
        metric="cosine",
        lsh_bits=16,
        lsh_tables=4,
        seed=0,
        on_event=events.append,
    )

    rng = random.Random(1)  # noqa: S311
    records: list[Record] = []
    stored_vectors: list[list[float]] = []
    for i in range(2_000):
        vec = [rng.gauss(0.0, 1.0) for _ in range(32)]
        records.append(Record(id=f"rec_{i}", vector=vec))
        stored_vectors.append(vec)

    store.upsert(records)

    correct_ids = 0
    num_queries = 50
    for i in range(num_queries):
        source_id = f"rec_{i}"
        query_vec = [x + rng.gauss(0.0, 0.05) for x in stored_vectors[i]]
        hits = store.search(query_vec, k=1, exact=False)
        if hits and hits[0].id == source_id:
            correct_ids += 1

    assert correct_ids >= 45, f"Expected >= 45 source matches, got {correct_ids}"
    mean_candidates = sum(e["candidates"] for e in events) / len(events)
    assert mean_candidates < 100, f"Expected mean candidates < 100, got {mean_candidates}"
    store.close()


def test_acceptance_closing_and_reopening_file_store(tmp_path: Path) -> None:
    db_path = tmp_path / "store.db"
    store = VectorStore.open(db_path, dim=4, metric="cosine", seed=0)
    store.upsert(
        [
            Record(id="1", vector=[1.0, 0.0, 0.0, 0.0], metadata={"k": "v1"}),
            Record(id="2", vector=[0.0, 1.0, 0.0, 0.0], metadata={"k": "v2"}),
            Record(id="3", vector=[0.0, 0.0, 1.0, 0.0], metadata={"k": "v3"}),
        ]
    )
    q = [0.9, 0.1, 0.0, 0.0]
    hits_exact_1 = store.search(q, k=2, exact=True)
    hits_approx_1 = store.search(q, k=2, exact=False)
    store.close()

    # Reopen
    store2 = VectorStore.open(db_path, dim=4, metric="cosine")
    hits_exact_2 = store2.search(q, k=2, exact=True)
    hits_approx_2 = store2.search(q, k=2, exact=False)
    assert [(h.id, h.score) for h in hits_exact_1] == [(h.id, h.score) for h in hits_exact_2]
    assert [(h.id, h.score) for h in hits_approx_1] == [(h.id, h.score) for h in hits_approx_2]
    store2.close()


def test_acceptance_upsert_transaction_atomicity_and_bounds() -> None:
    store = VectorStore.open(":memory:", dim=2)
    store.upsert([Record(id="existing", vector=[1.0, 0.0])])
    assert store.count() == 1

    # An upsert batch containing one invalid record raises and leaves count() unchanged (single transaction)
    with pytest.raises(VectorStoreError):
        store.upsert(
            [
                Record(id="valid1", vector=[0.0, 1.0]),
                Record(id="bad", vector=[0.0, 0.0, 0.0]),  # dim mismatch
            ]
        )
    assert store.count() == 1

    # an empty id raises 'invalid_id'
    with pytest.raises(VectorStoreError) as exc:
        store.upsert([Record(id="", vector=[1.0, 0.0])])
    assert exc.value.code == "invalid_id"

    # id longer than 256 characters raises 'invalid_id'
    with pytest.raises(VectorStoreError) as exc:
        store.upsert([Record(id="a" * 257, vector=[1.0, 0.0])])
    assert exc.value.code == "invalid_id"

    # metadata whose JSON exceeds 16_384 bytes raises 'invalid_metadata'
    huge_str = "x" * 17_000
    with pytest.raises(VectorStoreError) as exc:
        store.upsert([Record(id="huge_meta", vector=[1.0, 0.0], metadata={"key": huge_str})])
    assert exc.value.code == "invalid_metadata"

    # k=1001 raises 'invalid_k'
    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], k=1001)
    assert exc.value.code == "invalid_k"

    # k=0 raises 'invalid_k'
    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], k=0)
    assert exc.value.code == "invalid_k"

    store.close()


def test_edge_cases_and_features(tmp_path: Path) -> None:
    # Empty store search (returns [])
    store = VectorStore.open(":memory:", dim=3)
    assert store.search([1.0, 0.0, 0.0], k=5) == []
    assert store.search([1.0, 0.0, 0.0], k=5, exact=True) == []
    assert store.count() == 0

    # Unicode ids
    store.upsert(
        [Record(id="こんにちは世界_🚀", vector=[1.0, 0.0, 0.0], metadata={"text": "hello"})]
    )
    assert store.count() == 1
    hits = store.search([1.0, 0.0, 0.0], k=1)
    assert len(hits) == 1
    assert hits[0].id == "こんにちは世界_🚀"

    # k larger than count
    hits_large_k = store.search([1.0, 0.0, 0.0], k=10, exact=True)
    assert len(hits_large_k) == 1

    # Filter on a missing field (no match)
    hits_missing = store.search([1.0, 0.0, 0.0], where={"non_existent_key": "val"}, exact=True)
    assert len(hits_missing) == 0

    # Delete then search
    store.delete(["こんにちは世界_🚀"])
    assert store.count() == 0
    assert store.search([1.0, 0.0, 0.0], k=1) == []

    store.close()


def test_filters_and_operators() -> None:
    store = VectorStore.open(":memory:", dim=2)
    v = [1.0, 0.0]
    store.upsert(
        [
            Record(id="1", vector=v, metadata={"age": 20, "name": "alice", "active": True}),
            Record(id="2", vector=v, metadata={"age": 30, "name": "bob", "active": False}),
            Record(id="3", vector=v, metadata={"age": 40, "name": "charlie", "active": None}),
        ]
    )

    # $eq None
    h = store.search(v, where={"active": None}, exact=True)
    assert [x.id for x in h] == ["3"]

    # $ne None
    h = store.search(v, where={"active": {"$ne": None}}, exact=True)
    assert sorted([x.id for x in h]) == ["1", "2"]

    # $ne bool
    h = store.search(v, where={"active": {"$ne": True}}, exact=True)
    assert sorted([x.id for x in h]) == ["2", "3"]

    # $ne int
    h = store.search(v, where={"age": {"$ne": 30}}, exact=True)
    assert sorted([x.id for x in h]) == ["1", "3"]

    # $ne str
    h = store.search(v, where={"name": {"$ne": "bob"}}, exact=True)
    assert sorted([x.id for x in h]) == ["1", "3"]

    # $gt, $gte, $lt, $lte
    assert len(store.search(v, where={"age": {"$gt": 20}}, exact=True)) == 2
    assert len(store.search(v, where={"age": {"$gte": 20}}, exact=True)) == 3
    assert len(store.search(v, where={"age": {"$lt": 30}}, exact=True)) == 1
    assert len(store.search(v, where={"age": {"$lte": 30}}, exact=True)) == 2

    # String comparison with operators
    assert len(store.search(v, where={"name": {"$gt": "b"}}, exact=True)) == 2

    # $and
    h = store.search(v, where={"$and": [{"age": {"$gte": 20}}, {"age": {"$lt": 40}}]}, exact=True)
    assert sorted([x.id for x in h]) == ["1", "2"]

    # empty $in returns no matches
    assert store.count(where={"name": {"$in": []}}) == 0

    # $in with boolean and None and mixed
    store.upsert([Record(id="4", vector=v, metadata={"active": False})])
    assert store.count(where={"active": {"$in": [True, None]}}) == 2

    store.close()


def test_metrics_and_context_manager() -> None:
    # l2 metric
    with VectorStore.open(":memory:", dim=2, metric="l2") as store:
        store.upsert(
            [
                Record(id="p1", vector=[0.0, 0.0]),
                Record(id="p2", vector=[3.0, 4.0]),
            ]
        )
        hits = store.search([0.0, 0.0], k=2, exact=True)
        assert hits[0].id == "p1"
        assert hits[0].score == pytest.approx(0.0)
        assert hits[1].id == "p2"
        assert hits[1].score == pytest.approx(-5.0)

    # dot metric
    with VectorStore.open(":memory:", dim=2, metric="dot") as store:
        store.upsert(
            [
                Record(id="p1", vector=[1.0, 2.0]),
                Record(id="p2", vector=[2.0, 3.0]),
            ]
        )
        hits = store.search([1.0, 1.0], k=2, exact=True)
        assert hits[0].id == "p2"
        assert hits[0].score == pytest.approx(5.0)
        assert hits[1].id == "p1"
        assert hits[1].score == pytest.approx(3.0)


def test_errors_and_validations(tmp_path: Path) -> None:
    # Invalid config
    with pytest.raises(VectorStoreError) as exc:
        VectorStore.open(":memory:", dim=0)
    assert exc.value.code == "invalid_config"

    with pytest.raises(VectorStoreError) as exc:
        VectorStore.open(":memory:", dim=3, metric="unknown")  # type: ignore[arg-type]
    assert exc.value.code == "invalid_config"

    with pytest.raises(VectorStoreError) as exc:
        VectorStore.open(":memory:", dim=3, lsh_bits=0)
    assert exc.value.code == "invalid_config"

    with pytest.raises(VectorStoreError) as exc:
        VectorStore.open(":memory:", dim=3, lsh_tables=0)
    assert exc.value.code == "invalid_config"

    # Conflict on metric
    db_file = tmp_path / "metric_test.db"
    s1 = VectorStore.open(db_file, dim=3, metric="l2")
    s1.close()
    with pytest.raises(VectorStoreError) as exc:
        VectorStore.open(db_file, dim=3, metric="cosine")
    assert exc.value.code == "dimension_conflict"

    # Closed store operations
    store = VectorStore.open(":memory:", dim=2)
    store.close()
    with pytest.raises(VectorStoreError) as exc:
        store.upsert([Record(id="1", vector=[1.0, 0.0])])
    assert exc.value.code == "invalid_config"

    # Invalid metadata
    store = VectorStore.open(":memory:", dim=2)
    with pytest.raises(VectorStoreError) as exc:
        store.upsert([Record(id="1", vector=[1.0, 0.0], metadata="not-mapping")])  # type: ignore[arg-type]
    assert exc.value.code == "invalid_metadata"

    with pytest.raises(VectorStoreError) as exc:
        store.upsert([Record(id="1", vector=[1.0, 0.0], metadata={123: "val"})])  # type: ignore[dict-item]
    assert exc.value.code == "invalid_metadata"

    with pytest.raises(VectorStoreError) as exc:
        store.upsert([Record(id="1", vector=[1.0, 0.0], metadata={"key": [1, 2]})])  # type: ignore[dict-item]
    assert exc.value.code == "invalid_metadata"

    with pytest.raises(VectorStoreError) as exc:
        store.delete([""])
    assert exc.value.code == "invalid_id"

    with pytest.raises(VectorStoreError) as exc:
        store.delete([123])  # type: ignore[list-item]
    assert exc.value.code == "invalid_id"

    # Invalid filters
    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], where="not a mapping")  # type: ignore[arg-type]
    assert exc.value.code == "invalid_filter"

    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], where={"$and": "not list"})  # type: ignore[dict-item]
    assert exc.value.code == "invalid_filter"

    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], where={"$and": ["not mapping"]})  # type: ignore[list-item]
    assert exc.value.code == "invalid_filter"

    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], where={"age": {}})
    assert exc.value.code == "invalid_filter"

    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], where={"age": {"$unsupported": 1}})
    assert exc.value.code == "invalid_filter"

    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], where={"age": {"$gt": True}})
    assert exc.value.code == "invalid_filter"

    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], where={"age": {"$in": "not list"}})  # type: ignore[dict-item]
    assert exc.value.code == "invalid_filter"

    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], where={"age": {"$in": [object()]}})
    assert exc.value.code == "invalid_filter"

    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], where={"age": {"$in": list(range(101))}})
    assert exc.value.code == "invalid_filter"

    with pytest.raises(VectorStoreError) as exc:
        store.search([1.0, 0.0], where={123: 1})  # type: ignore[dict-item]
    assert exc.value.code == "invalid_filter"

    store.close()

"""Embedded vector store on stdlib sqlite3 with float32 blobs and seeded LSH."""

from __future__ import annotations

import array
import json
import logging
import math
import os
import random
import re
import sqlite3
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "Hit",
    "Record",
    "VectorStore",
    "VectorStoreError",
    "__version__",
]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ErrorCode = Literal[
    "dimension_mismatch",
    "dimension_conflict",
    "invalid_vector",
    "invalid_filter",
    "invalid_id",
    "invalid_metadata",
    "invalid_k",
    "invalid_config",
]

Metric = Literal["cosine", "l2", "dot"]

_VALID_METRICS: set[str] = {"cosine", "l2", "dot"}
_VALID_OPS: set[str] = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in"}
_FIELD_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

_MAX_ID_LEN = 256
_MAX_METADATA_BYTES = 16_384
_MIN_DIM = 1
_MAX_DIM = 4096
_MIN_LSH_BITS = 1
_MAX_LSH_BITS = 62
_MIN_LSH_TABLES = 1
_MAX_LSH_TABLES = 16
_MIN_K = 1
_MAX_K = 1000
_MAX_CONDITIONS = 16
_MAX_FILTER_DEPTH = 8
_MAX_IN_SCALARS = 100


class VectorStoreError(Exception):
    """Base exception for vector_store_lite errors with a machine-readable code."""

    def __init__(self, code: ErrorCode, message: str | None = None) -> None:
        self.code: ErrorCode = code
        msg = message if message is not None else code
        if len(msg) > 200:
            msg = msg[:200]
        super().__init__(msg)


@dataclass(frozen=True)
class Record:
    """A record to store in the VectorStore."""

    id: str
    vector: Sequence[float]
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    """Search result hit."""

    id: str
    score: float
    metadata: dict[str, Any]


def _is_valid_scalar(val: Any) -> bool:
    if val is None or isinstance(val, (bool, str)):
        return True
    if isinstance(val, (int, float)):
        return math.isfinite(val)
    return False


def _compile_leaf_condition(field_name: str, op: str, val: Any) -> tuple[str, list[Any]]:
    if op == "$in":
        if not isinstance(val, list):
            raise VectorStoreError("invalid_filter", "$in requires a list")
        if len(val) > _MAX_IN_SCALARS:
            raise VectorStoreError("invalid_filter", f"$in list exceeds {_MAX_IN_SCALARS} scalars")
        for item in val:
            if not _is_valid_scalar(item):
                raise VectorStoreError("invalid_filter", "invalid scalar in $in")
        if not val:
            return ("(0 = 1)", [])

        subconds: list[str] = []
        params: list[Any] = []
        has_null = None in val
        has_true = True in val
        has_false = False in val
        numbers = [x for x in val if isinstance(x, (int, float)) and not isinstance(x, bool)]
        strings = [x for x in val if isinstance(x, str)]

        if has_null:
            subconds.append("json_type(items.meta, '$.' || ?) = 'null'")
            params.append(field_name)
        if has_true:
            subconds.append("json_type(items.meta, '$.' || ?) = 'true'")
            params.append(field_name)
        if has_false:
            subconds.append("json_type(items.meta, '$.' || ?) = 'false'")
            params.append(field_name)
        if numbers:
            q = ", ".join(["?"] * len(numbers))
            subconds.append(
                f"(json_type(items.meta, '$.' || ?) NOT IN ('true', 'false') "
                f"AND json_extract(items.meta, '$.' || ?) IN ({q}))"
            )
            params.extend([field_name, field_name, *numbers])
        if strings:
            q = ", ".join(["?"] * len(strings))
            subconds.append(
                f"(json_type(items.meta, '$.' || ?) = 'text' "
                f"AND json_extract(items.meta, '$.' || ?) IN ({q}))"
            )
            params.extend([field_name, field_name, *strings])

        return ("(" + " OR ".join(subconds) + ")", params)

    if not _is_valid_scalar(val):
        raise VectorStoreError("invalid_filter", "filter value must be a valid scalar")

    if op in {"$gt", "$gte", "$lt", "$lte"}:
        if val is None or isinstance(val, bool):
            raise VectorStoreError("invalid_filter", f"{op} not supported for bool or null")
        sql_op = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}[op]
        if isinstance(val, (int, float)):
            return (
                f"(json_type(items.meta, '$.' || ?) NOT IN ('true', 'false') "
                f"AND json_extract(items.meta, '$.' || ?) {sql_op} ?)",
                [field_name, field_name, val],
            )
        return (
            f"(json_type(items.meta, '$.' || ?) = 'text' "
            f"AND json_extract(items.meta, '$.' || ?) {sql_op} ?)",
            [field_name, field_name, val],
        )

    if op == "$eq":
        if val is None:
            return ("json_type(items.meta, '$.' || ?) = 'null'", [field_name])
        if isinstance(val, bool):
            t_str = "true" if val else "false"
            return ("json_type(items.meta, '$.' || ?) = ?", [field_name, t_str])
        if isinstance(val, (int, float)):
            return (
                "(json_type(items.meta, '$.' || ?) NOT IN ('true', 'false') "
                "AND json_extract(items.meta, '$.' || ?) = ?)",
                [field_name, field_name, val],
            )
        return (
            "(json_type(items.meta, '$.' || ?) = 'text' "
            "AND json_extract(items.meta, '$.' || ?) = ?)",
            [field_name, field_name, val],
        )

    # op == "$ne"
    if val is None:
        return (
            "(json_type(items.meta, '$.' || ?) IS NOT NULL "
            "AND json_type(items.meta, '$.' || ?) != 'null')",
            [field_name, field_name],
        )
    if isinstance(val, bool):
        t_str = "true" if val else "false"
        return (
            "(json_type(items.meta, '$.' || ?) IS NOT NULL "
            "AND json_type(items.meta, '$.' || ?) != ?)",
            [field_name, field_name, t_str],
        )
    if isinstance(val, (int, float)):
        return (
            "(json_type(items.meta, '$.' || ?) IS NOT NULL "
            "AND (json_type(items.meta, '$.' || ?) IN ('true', 'false') "
            "OR json_extract(items.meta, '$.' || ?) != ?))",
            [field_name, field_name, field_name, val],
        )
    return (
        "(json_type(items.meta, '$.' || ?) IS NOT NULL "
        "AND (json_type(items.meta, '$.' || ?) != 'text' "
        "OR json_extract(items.meta, '$.' || ?) != ?))",
        [field_name, field_name, field_name, val],
    )


def _compile_filter_node(
    node: Mapping[str, Any], depth: int = 0
) -> tuple[list[str], list[Any], int]:
    if depth > _MAX_FILTER_DEPTH:
        raise VectorStoreError("invalid_filter", "filter depth limit exceeded")

    clauses: list[str] = []
    params: list[Any] = []
    cond_count = 0

    for key, val in node.items():
        if not isinstance(key, str):
            raise VectorStoreError("invalid_filter", "filter key must be a string")

        if key == "$and":
            if not isinstance(val, list):
                raise VectorStoreError("invalid_filter", "$and must be a list")
            for subfilter in val:
                if not isinstance(subfilter, Mapping):
                    raise VectorStoreError("invalid_filter", "$and elements must be mappings")
                sub_clauses, sub_params, sub_count = _compile_filter_node(subfilter, depth + 1)
                cond_count += sub_count
                if sub_clauses:
                    if len(sub_clauses) == 1:
                        clauses.append(sub_clauses[0])
                    else:
                        clauses.append("(" + " AND ".join(sub_clauses) + ")")
                    params.extend(sub_params)
        elif key.startswith("$"):
            raise VectorStoreError("invalid_filter", f"unsupported top-level operator: {key}")
        else:
            if not _FIELD_REGEX.match(key):
                raise VectorStoreError("invalid_filter", f"invalid field name: {key}")
            if isinstance(val, Mapping):
                if not val:
                    raise VectorStoreError("invalid_filter", "empty operator mapping")
                for op, op_val in val.items():
                    if op not in _VALID_OPS:
                        raise VectorStoreError("invalid_filter", f"unsupported operator: {op}")
                    cond_count += 1
                    leaf_sql, leaf_params = _compile_leaf_condition(key, op, op_val)
                    clauses.append(leaf_sql)
                    params.extend(leaf_params)
            else:
                cond_count += 1
                leaf_sql, leaf_params = _compile_leaf_condition(key, "$eq", val)
                clauses.append(leaf_sql)
                params.extend(leaf_params)

    return clauses, params, cond_count


def _compile_where(where: Mapping[str, Any] | None) -> tuple[str, list[Any]]:
    if where is None or len(where) == 0:
        return "", []
    if not isinstance(where, Mapping):
        raise VectorStoreError("invalid_filter", "where must be a mapping")

    clauses, params, cond_count = _compile_filter_node(where, 0)
    if cond_count > _MAX_CONDITIONS:
        raise VectorStoreError(
            "invalid_filter", f"filter has {cond_count} conditions, max is {_MAX_CONDITIONS}"
        )
    if not clauses:
        return "", []
    return " AND ".join(clauses), params


class VectorStore:
    """Embedded vector store backed by SQLite with exact and LSH approximate search."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        dim: int,
        metric: Metric,
        lsh_bits: int,
        lsh_tables: int,
        seed: int,
        hyperplanes: list[list[list[float]]],
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._conn = conn
        self.dim = dim
        self.metric: Metric = metric
        self.lsh_bits = lsh_bits
        self.lsh_tables = lsh_tables
        self.seed = seed
        self.hyperplanes = hyperplanes
        self.on_event = on_event
        self._closed = False

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str] | os.PathLike[bytes] | os.PathLike[Any],
        *,
        dim: int,
        metric: Metric = "cosine",
        lsh_bits: int = 16,
        lsh_tables: int = 4,
        seed: int = 0,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> VectorStore:
        """Open or create a SQLite-backed vector store."""
        if not isinstance(dim, int) or dim < _MIN_DIM or dim > _MAX_DIM:
            raise VectorStoreError(
                "invalid_config", f"dim must be between {_MIN_DIM} and {_MAX_DIM}"
            )
        if metric not in _VALID_METRICS:
            raise VectorStoreError("invalid_config", f"invalid metric: {metric}")
        if not isinstance(lsh_bits, int) or lsh_bits < _MIN_LSH_BITS or lsh_bits > _MAX_LSH_BITS:
            raise VectorStoreError(
                "invalid_config", f"lsh_bits must be between {_MIN_LSH_BITS} and {_MAX_LSH_BITS}"
            )
        if (
            not isinstance(lsh_tables, int)
            or lsh_tables < _MIN_LSH_TABLES
            or lsh_tables > _MAX_LSH_TABLES
        ):
            raise VectorStoreError(
                "invalid_config",
                f"lsh_tables must be between {_MIN_LSH_TABLES} and {_MAX_LSH_TABLES}",
            )

        path_str = str(os.fspath(path))
        conn = sqlite3.connect(path_str)

        # Check if meta table already exists with data
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='meta'")
        has_meta_table = cur.fetchone()[0] > 0
        stored_meta: dict[str, str] = {}
        if has_meta_table:
            cur.execute("SELECT key, value FROM meta")
            for k_row, v_row in cur.fetchall():
                stored_meta[k_row] = v_row

        if stored_meta:
            stored_dim = int(stored_meta["dim"])
            stored_metric = stored_meta["metric"]
            if stored_dim != dim:
                conn.close()
                raise VectorStoreError(
                    "dimension_conflict",
                    f"store dim ({stored_dim}) conflicts with requested dim ({dim})",
                )
            if stored_metric != metric:
                conn.close()
                raise VectorStoreError(
                    "dimension_conflict",
                    f"store metric ({stored_metric}) conflicts with requested metric ({metric})",
                )
            # Stored LSH parameters win over arguments
            active_lsh_bits = int(stored_meta["lsh_bits"])
            active_lsh_tables = int(stored_meta["lsh_tables"])
            active_seed = int(stored_meta["seed"])
            hyperplanes: list[list[list[float]]] = json.loads(stored_meta["hyperplanes"])
        else:
            # New store: initialize tables and hyperplanes
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS items ("
                    "id TEXT PRIMARY KEY, vec BLOB NOT NULL, meta TEXT NOT NULL"
                    ")"
                )
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS buckets ("
                    "t INTEGER NOT NULL, sig INTEGER NOT NULL, id TEXT NOT NULL, "
                    "PRIMARY KEY (t, sig, id)"
                    ")"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_buckets_t_sig ON buckets (t, sig)")

                rng = random.Random(seed)  # noqa: S311
                hyperplanes = []
                for _ in range(lsh_tables):
                    table_planes: list[list[float]] = []
                    for _ in range(lsh_bits):
                        table_planes.append([rng.gauss(0.0, 1.0) for _ in range(dim)])
                    hyperplanes.append(table_planes)

                meta_entries = [
                    ("dim", str(dim)),
                    ("metric", metric),
                    ("lsh_bits", str(lsh_bits)),
                    ("lsh_tables", str(lsh_tables)),
                    ("seed", str(seed)),
                    ("hyperplanes", json.dumps(hyperplanes)),
                ]
                conn.executemany(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    meta_entries,
                )

            active_lsh_bits = lsh_bits
            active_lsh_tables = lsh_tables
            active_seed = seed

        return cls(
            conn=conn,
            dim=dim,
            metric=metric,
            lsh_bits=active_lsh_bits,
            lsh_tables=active_lsh_tables,
            seed=active_seed,
            hyperplanes=hyperplanes,
            on_event=on_event,
        )

    def _check_open(self) -> None:
        if self._closed:
            raise VectorStoreError("invalid_config", "VectorStore is closed")

    def _validate_vector(self, raw_vec: Sequence[float]) -> list[float]:
        if len(raw_vec) != self.dim:
            raise VectorStoreError(
                "dimension_mismatch",
                f"vector length {len(raw_vec)} does not match store dimension {self.dim}",
            )
        vec: list[float] = []
        for x in raw_vec:
            if not isinstance(x, (int, float)) or not math.isfinite(x):
                raise VectorStoreError("invalid_vector", "vector elements must be finite numbers")
            vec.append(float(x))

        if self.metric == "cosine":
            norm = math.sqrt(sum(x * x for x in vec))
            if norm == 0.0:
                raise VectorStoreError("invalid_vector", "zero vector invalid for cosine metric")
            vec = [x / norm for x in vec]
        return vec

    def _compute_signatures(self, vec: Sequence[float]) -> list[int]:
        sigs: list[int] = []
        for t in range(self.lsh_tables):
            sig = 0
            for i, plane in enumerate(self.hyperplanes[t]):
                dot = sum(p * x for p, x in zip(plane, vec, strict=False))
                if dot >= 0.0:
                    sig |= 1 << i
            sigs.append(sig)
        return sigs

    def _encode_vector(self, vec: Sequence[float]) -> bytes:
        arr = array.array("f", vec)
        if sys.byteorder != "little":
            arr.byteswap()
        return arr.tobytes()

    def _decode_vector(self, blob: bytes) -> list[float]:
        arr = array.array("f")
        arr.frombytes(blob)
        if sys.byteorder != "little":
            arr.byteswap()
        return arr.tolist()

    def upsert(self, records: Iterable[Record]) -> int:
        """Insert or replace records in a single atomic transaction."""
        self._check_open()

        validated: list[tuple[str, bytes, str, list[int]]] = []
        for rec in records:
            if not isinstance(rec.id, str) or len(rec.id) < 1 or len(rec.id) > _MAX_ID_LEN:
                raise VectorStoreError(
                    "invalid_id", f"id must be a string between 1 and {_MAX_ID_LEN} characters"
                )

            vec = self._validate_vector(rec.vector)
            if not isinstance(rec.metadata, Mapping):
                raise VectorStoreError("invalid_metadata", "metadata must be a mapping")

            for k, val in rec.metadata.items():
                if not isinstance(k, str):
                    raise VectorStoreError("invalid_metadata", "metadata keys must be strings")
                if not _is_valid_scalar(val):
                    raise VectorStoreError(
                        "invalid_metadata", f"metadata value for key '{k}' must be a scalar or null"
                    )

            meta_json = json.dumps(rec.metadata, ensure_ascii=False, separators=(",", ":"))
            if len(meta_json.encode("utf-8")) > _MAX_METADATA_BYTES:
                raise VectorStoreError(
                    "invalid_metadata",
                    f"metadata JSON exceeds {_MAX_METADATA_BYTES} bytes",
                )

            sigs = self._compute_signatures(vec)
            vec_bytes = self._encode_vector(vec)
            validated.append((rec.id, vec_bytes, meta_json, sigs))

        with self._conn:
            cur = self._conn.cursor()
            for rec_id, vec_bytes, meta_json, sigs in validated:
                cur.execute("DELETE FROM buckets WHERE id = ?", (rec_id,))
                cur.execute(
                    "INSERT OR REPLACE INTO items (id, vec, meta) VALUES (?, ?, ?)",
                    (rec_id, vec_bytes, meta_json),
                )
                for t, sig in enumerate(sigs):
                    cur.execute(
                        "INSERT OR REPLACE INTO buckets (t, sig, id) VALUES (?, ?, ?)",
                        (t, sig, rec_id),
                    )

        return len(validated)

    def delete(self, ids: Iterable[str]) -> int:
        """Delete records by ID in a single atomic transaction, returning the count deleted."""
        self._check_open()
        ids_list: list[str] = []
        for id_val in ids:
            if not isinstance(id_val, str) or len(id_val) < 1 or len(id_val) > _MAX_ID_LEN:
                raise VectorStoreError(
                    "invalid_id", f"id must be a string between 1 and {_MAX_ID_LEN} characters"
                )
            ids_list.append(id_val)

        deleted = 0
        with self._conn:
            cur = self._conn.cursor()
            for id_val in ids_list:
                cur.execute("DELETE FROM items WHERE id = ?", (id_val,))
                if cur.rowcount > 0:
                    deleted += cur.rowcount
                    cur.execute("DELETE FROM buckets WHERE id = ?", (id_val,))
        return deleted

    def count(self, where: Mapping[str, Any] | None = None) -> int:
        """Count items matching the given filter."""
        self._check_open()
        filter_sql, filter_params = _compile_where(where)
        cur = self._conn.cursor()
        if filter_sql:
            cur.execute(f"SELECT COUNT(*) FROM items WHERE {filter_sql}", filter_params)  # noqa: S608
        else:
            cur.execute("SELECT COUNT(*) FROM items")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def search(
        self,
        vector: Sequence[float],
        *,
        k: int = 10,
        where: Mapping[str, Any] | None = None,
        exact: bool = False,
    ) -> list[Hit]:
        """Search for the top k nearest items using exact scan or LSH index."""
        self._check_open()
        if not isinstance(k, int) or k < _MIN_K or k > _MAX_K:
            raise VectorStoreError(
                "invalid_k", f"k must be an integer between {_MIN_K} and {_MAX_K}"
            )

        q_vec = self._validate_vector(vector)
        filter_sql, filter_params = _compile_where(where)

        cur = self._conn.cursor()
        rows: list[tuple[str, bytes, str]]

        if exact:
            if filter_sql:
                query = f"SELECT id, vec, meta FROM items WHERE {filter_sql}"  # noqa: S608
                cur.execute(query, filter_params)
            else:
                query = "SELECT id, vec, meta FROM items"
                cur.execute(query)
            rows = cur.fetchall()
        else:
            # LSH approximate search
            sigs = self._compute_signatures(q_vec)
            lsh_clauses = ["(buckets.t = ? AND buckets.sig = ?)" for _ in range(self.lsh_tables)]
            lsh_params: list[Any] = []
            for t, sig in enumerate(sigs):
                lsh_params.extend([t, sig])

            base_lsh = "(" + " OR ".join(lsh_clauses) + ")"
            if filter_sql:
                query = (
                    f"SELECT DISTINCT items.id, items.vec, items.meta FROM buckets "  # noqa: S608
                    f"JOIN items ON buckets.id = items.id "
                    f"WHERE {base_lsh} AND ({filter_sql})"
                )
                cur.execute(query, [*lsh_params, *filter_params])
            else:
                query = (
                    f"SELECT DISTINCT items.id, items.vec, items.meta FROM buckets "  # noqa: S608
                    f"JOIN items ON buckets.id = items.id "
                    f"WHERE {base_lsh}"
                )
                cur.execute(query, lsh_params)
            rows = cur.fetchall()

        candidates_count = len(rows)
        scored_items: list[tuple[str, float, dict[str, Any]]] = []

        for row_id, vec_blob, meta_text in rows:
            stored_v = self._decode_vector(vec_blob)
            if self.metric == "cosine" or self.metric == "dot":
                score = sum(qi * vi for qi, vi in zip(q_vec, stored_v, strict=False))
            else:  # l2
                dist = math.sqrt(
                    sum((qi - vi) ** 2 for qi, vi in zip(q_vec, stored_v, strict=False))
                )
                score = -dist
            meta_dict = json.loads(meta_text)
            scored_items.append((row_id, score, meta_dict))

        # Sort descending by score, ascending by id on ties
        scored_items.sort(key=lambda item: (-item[1], item[0]))
        top_items = scored_items[:k]

        hits = [Hit(id=item_id, score=score, metadata=meta) for item_id, score, meta in top_items]

        if self.on_event is not None:
            self.on_event(
                {
                    "kind": "search",
                    "exact": exact,
                    "candidates": candidates_count,
                    "returned": len(hits),
                }
            )

        return hits

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if not self._closed:
            self._conn.close()
            self._closed = True

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

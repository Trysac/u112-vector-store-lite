# Use stdlib SQLite with JSON1 and Seeded Hyperplane LSH for Embedded Vector Search

Status: accepted (generated, pending human review)

## Context

Lightweight RAG applications need persistent vector storage and metadata filtering without external server processes or compiled C extensions.

## Decision

We use SQLite float32 BLOB storage with JSON1 parameterised queries for metadata filtering, combined with seeded random-hyperplane Locality-Sensitive Hashing stored in the database meta table.

## Consequences

Zero external dependencies, safe parameterized SQL queries immune to SQL injection, deterministic persistence across restarts, and atomic single-transaction upsert/delete operations.

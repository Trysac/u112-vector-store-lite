# vector-store-lite

[![PyPI version](https://img.shields.io/pypi/v/trysac-vector-store-lite)](https://pypi.org/project/trysac-vector-store-lite/)
[![License: MIT](https://img.shields.io/pypi/l/trysac-vector-store-lite)](LICENSE)
[![CI](https://github.com/Trysac/u112-vector-store-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/Trysac/u112-vector-store-lite/actions/workflows/ci.yml)

Embedded vector store on stdlib sqlite3 with float32 blobs, metadata filters compiled to parameterised SQL, exact search and seeded random-hyperplane LSH

## Problem

Describe in two or three sentences the concrete problem this utility solves, who hits it, and what
they do today without it. Replace this paragraph when the first real feature lands.

## Install

```sh
pip install trysac-vector-store-lite
# or
uv add trysac-vector-store-lite
```

Requires Python 3.10 or newer. No runtime dependencies.

## Usage

```python
import vector_store_lite

# normalize() trims surrounding whitespace and lower-cases the text.
print(vector_store_lite.normalize("  Hello World "))  # -> "hello world"

# Blank input is rejected explicitly instead of returning an empty string.
try:
    vector_store_lite.normalize("   ")
except ValueError as exc:
    print(f"rejected: {exc}")
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [decisions](docs/decisions/)
- [Testing](docs/TESTING.md) and [security](docs/SECURITY.md)
- [Project record](docs/PROJECT.md) and [stack](docs/STACK.md)
- [Changelog](CHANGELOG.md) and [contributing guide](CONTRIBUTING.md)

## License

[MIT](LICENSE) (c) 2026 Miguel Gomez

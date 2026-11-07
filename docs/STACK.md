# Stack

- **Python**: `>=3.10` (required for modern typing).
- **Build System**: `uv`.
- **Quality Tools**: `ruff` (lint/format), `mypy` (strict type checking), `pytest`.
- **Infrastructure**: GitHub Actions, Dependabot.

## Discarded

- **Poetry**: Too slow/heavy; `uv` preferred.
- **tox**: Replaced by custom `scripts/check.py` for simpler, portable CI gates.
"""``python -m worker`` support."""

from .main import main

if __name__ == "__main__":  # pragma: no cover - process entry
    raise SystemExit(main())

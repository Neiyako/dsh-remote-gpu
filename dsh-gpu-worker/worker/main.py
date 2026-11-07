"""``python -m worker`` entrypoint for the HTTP server."""

from __future__ import annotations

import logging
import sys

import uvicorn

from .api import create_app
from .config import ConfigurationError, Settings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    app = create_app(settings)
    logging.getLogger("worker").info(
        "starting dsh-gpu-worker on %s:%s (root=%s, max_concurrent=%s)",
        settings.host,
        settings.port,
        settings.root,
        settings.max_concurrent_tasks,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info", access_log=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry
    raise SystemExit(main())

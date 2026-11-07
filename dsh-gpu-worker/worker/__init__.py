"""DSH remote GPU worker (FastAPI) — independent project, contract v1."""

from .config import API_VERSION, WORKER_VERSION

__all__ = ["API_VERSION", "WORKER_VERSION", "__version__"]
__version__ = WORKER_VERSION

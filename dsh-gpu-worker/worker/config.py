"""Runtime configuration for the DSH remote GPU worker.

All configuration comes from environment variables.  A token is mandatory and
startup must fail loudly when it is missing (see ``Settings.from_env``).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

#: Directory that contains the ``worker`` package (the project root).
PROJECT_DIR = Path(__file__).resolve().parent.parent

API_VERSION = "1"
WORKER_VERSION = "0.1.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_TIMEOUT_SECONDS = 3600
MIN_TIMEOUT_SECONDS = 10
MAX_TIMEOUT_SECONDS = 86400


class ConfigurationError(RuntimeError):
    """Raised when the worker is misconfigured and must not start."""


class Settings(BaseModel):
    """Immutable worker settings resolved from the environment."""

    model_config = ConfigDict(frozen=True)

    token: str
    root: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_concurrent_tasks: int = 1
    allow_generic_python: bool = True
    db_path: Path | None = None
    poll_seconds: float = 0.05
    kill_grace_seconds: float = 3.0
    token_source: str = "DSH_GPU_WORKER_TOKEN"

    @field_validator("root")
    @classmethod
    def _absolute_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def generic_jobs_dir(self) -> Path:
        return self.scripts_dir / "jobs"

    @property
    def resolved_db_path(self) -> Path:
        return self.db_path or (self.root / "worker.db")

    def ensure_directories(self) -> None:
        """Create the directories the worker owns, all below ``root``."""
        for directory in (
            self.root,
            self.experiments_dir,
            self.datasets_dir,
            self.scripts_dir,
            self.generic_jobs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.environ.get("DSH_GPU_WORKER_TOKEN", "").strip()
        if not token:
            raise ConfigurationError(
                "DSH_GPU_WORKER_TOKEN is not set. Refusing to start without an "
                "authentication token. Generate one with: python -c "
                "\"import secrets;print(secrets.token_urlsafe(32))\""
            )
        if len(token) < 8:
            raise ConfigurationError(
                "DSH_GPU_WORKER_TOKEN must be at least 8 characters long."
            )

        root_env = os.environ.get("DSH_GPU_WORKER_ROOT", "").strip()
        root = Path(root_env).expanduser() if root_env else PROJECT_DIR

        host = os.environ.get("DSH_GPU_WORKER_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
        try:
            port = int(os.environ.get("DSH_GPU_WORKER_PORT", str(DEFAULT_PORT)))
        except ValueError as exc:  # pragma: no cover - defensive
            raise ConfigurationError("DSH_GPU_WORKER_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ConfigurationError("DSH_GPU_WORKER_PORT must be within 1..65535")

        def _int_env(name: str, default: int) -> int:
            raw = os.environ.get(name, "").strip()
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ConfigurationError(f"{name} must be an integer") from exc

        max_concurrent = _int_env("DSH_GPU_WORKER_MAX_CONCURRENT_TASKS", 1)
        if max_concurrent < 1:
            raise ConfigurationError("DSH_GPU_WORKER_MAX_CONCURRENT_TASKS must be >= 1")

        allow_generic = os.environ.get(
            "DSH_GPU_WORKER_ALLOW_GENERIC_PYTHON", "true"
        ).strip().lower() not in {"0", "false", "no", "off"}

        db_env = os.environ.get("DSH_GPU_WORKER_DB", "").strip()

        return cls(
            token=token,
            root=root,
            host=host,
            port=port,
            max_concurrent_tasks=max_concurrent,
            allow_generic_python=allow_generic,
            db_path=Path(db_env).expanduser().resolve() if db_env else None,
        )

"""FastAPI application: the complete HTTP contract from docs/CONTRACT.md §2.

Every endpoint (including ``/health`` and ``/gpu``) requires the worker token.
Errors are always rendered as ``{"error":{"code","message","details"}}``.
"""

from __future__ import annotations

import hmac
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import gpu
from .artifacts import scan_artifacts
from .config import API_VERSION, WORKER_VERSION, Settings
from .db import Store
from .errors import WorkerError, bad_request, conflict, not_found, unauthorized
from .logging_utils import read_log
from .models import TERMINAL_STATUSES, SubmitRequest
from .paths import ensure_under, experiment_dir, resolve_artifact_path
from .runner import TaskRunner, row_to_task_view
from .utils import utcnow

logger = logging.getLogger("worker.api")

_ALL_STATUSES = {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"}


def require_token(request: Request) -> None:
    """Authenticate every request via Bearer or ``X-Worker-Token``."""
    settings: Settings = request.app.state.settings
    provided: str | None = None
    header = request.headers.get("authorization")
    if header:
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            provided = value.strip()
    if not provided:
        header_token = request.headers.get("x-worker-token")
        if header_token and header_token.strip():
            provided = header_token.strip()
    if not provided or not hmac.compare_digest(provided, settings.token):
        raise unauthorized()


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _store(request: Request) -> Store:
    return request.app.state.store


def _runner(request: Request) -> TaskRunner:
    return request.app.state.runner


def _parse_status_filter(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if not normalized:
        return None
    if normalized not in _ALL_STATUSES:
        raise bad_request(
            "INVALID_REQUEST",
            f"Unknown status filter: {value!r}",
            {"supported": sorted(_ALL_STATUSES)},
        )
    return normalized


def create_app(
    settings: Settings,
    *,
    store: Store | None = None,
    runner: TaskRunner | None = None,
) -> FastAPI:
    """Build the FastAPI app; ``store``/``runner`` exist for tests."""
    owns_resources = store is None and runner is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.ensure_directories()
        actual_store = store or (runner.store if runner is not None else None)
        if actual_store is None:
            actual_store = Store(settings.resolved_db_path)
        actual_runner = runner or TaskRunner(settings, actual_store)
        app.state.store = actual_store
        app.state.runner = actual_runner
        actual_runner.start()
        logger.info(
            "worker ready: root=%s max_concurrent=%s recovered=%d",
            settings.root,
            settings.max_concurrent_tasks,
            len(actual_runner.recovery),
        )
        try:
            yield
        finally:
            try:
                app.state.runner.shutdown()
            finally:
                if owns_resources:
                    app.state.store.close()

    app = FastAPI(
        title="DSH Remote GPU Worker",
        version=WORKER_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
        dependencies=[Depends(require_token)],
    )
    app.state.settings = settings

    # -- error handlers -------------------------------------------------------

    @app.exception_handler(WorkerError)
    async def _worker_error_handler(request: Request, exc: WorkerError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content=jsonable_encoder(exc.to_body())
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Request validation failed",
                    "details": {"errors": jsonable_encoder(exc.errors())},
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        codes = {
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "ROUTE_NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
        }
        code = codes.get(exc.status_code, f"HTTP_{exc.status_code}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": str(exc.detail), "details": {}}},
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {},
                }
            },
        )

    # -- health / gpu / version ----------------------------------------------

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        settings = _settings(request)
        store = _store(request)
        runner = _runner(request)
        info = gpu.gpu_info()
        return {
            "status": "ok",
            "worker_id": store.worker_id(),
            "version": WORKER_VERSION,
            "time": utcnow(),
            "uptime_seconds": round(runner.uptime_seconds, 3),
            "cuda_available": info["cuda_available"],
            "active_tasks": runner.active_tasks,
            "queued_tasks": store.count_by_status("PENDING"),
            "max_concurrent_tasks": settings.max_concurrent_tasks,
            "allow_generic_python": settings.allow_generic_python,
            "torch_installed": info["torch_installed"],
            "recovered_tasks": len(runner.recovery),
        }

    @app.get("/gpu")
    def gpu_info(request: Request) -> dict[str, Any]:
        payload = dict(gpu.gpu_info())
        payload["time"] = utcnow()
        return payload

    @app.get("/version")
    def version(request: Request) -> dict[str, Any]:
        settings = _settings(request)
        return {
            "name": "dsh-gpu-worker",
            "version": WORKER_VERSION,
            "api_version": API_VERSION,
            "worker_id": _store(request).worker_id(),
            "time": utcnow(),
            "root": str(settings.root),
            "max_concurrent_tasks": settings.max_concurrent_tasks,
            "allow_generic_python": settings.allow_generic_python,
        }

    # -- tasks ---------------------------------------------------------------

    @app.post("/tasks")
    def submit_task(request: Request, payload: SubmitRequest) -> JSONResponse:
        view, created = _runner(request).submit(payload)
        return JSONResponse(status_code=201 if created else 200, content=view)

    @app.get("/tasks")
    def list_tasks(
        request: Request,
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        parsed_status = _parse_status_filter(status)
        rows = _store(request).list_tasks(parsed_status, limit=limit)
        views = [row_to_task_view(row) for row in rows]
        return {"tasks": views, "count": len(views)}

    @app.get("/tasks/{task_id}")
    def get_task(request: Request, task_id: str) -> dict[str, Any]:
        return row_to_task_view(_store(request).get_task(task_id))

    @app.post("/tasks/{task_id}/cancel")
    def cancel_task(request: Request, task_id: str) -> dict[str, Any]:
        return _runner(request).cancel(task_id)

    @app.get("/tasks/{task_id}/logs")
    def task_logs(
        request: Request,
        task_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=500, ge=1, le=2000),
    ) -> dict[str, Any]:
        row = _store(request).get_task(task_id)
        log_path: Path | None = None
        if row["log_path"]:
            log_path = ensure_under(
                _settings(request).root,
                _settings(request).root / row["log_path"],
                what="log path",
            )
        page = read_log(log_path, offset, limit)
        return {"task_id": task_id, **page}

    @app.get("/tasks/{task_id}/result")
    def task_result(request: Request, task_id: str) -> dict[str, Any]:
        row = _store(request).get_task(task_id)
        status = row["status"]
        if status not in TERMINAL_STATUSES:
            raise conflict(
                "TASK_NOT_FINISHED",
                f"Task {task_id} is not finished yet (status={status})",
                {"status": status},
            )
        return {
            "task_id": task_id,
            "status": status,
            "result": _loads(row["result_json"]),
            "error": _loads(row["error_json"]),
        }

    @app.get("/tasks/{task_id}/artifacts")
    def task_artifacts(request: Request, task_id: str) -> dict[str, Any]:
        _store(request).get_task(task_id)
        experiment = experiment_dir(_settings(request).root, task_id)
        return {"task_id": task_id, "artifacts": scan_artifacts(experiment, task_id)}

    @app.get("/tasks/{task_id}/artifacts/{artifact_path:path}")
    def download_artifact(request: Request, task_id: str, artifact_path: str) -> FileResponse:
        _store(request).get_task(task_id)
        root = _settings(request).root
        target = resolve_artifact_path(root, task_id, artifact_path)
        if not target.is_file():
            raise not_found(
                "ARTIFACT_NOT_FOUND",
                f"Artifact not found: {artifact_path}",
                {"task_id": task_id, "path": artifact_path},
            )
        return FileResponse(
            path=str(target),
            media_type="application/octet-stream",
            filename=target.name,
        )

    # -- datasets ------------------------------------------------------------

    @app.get("/datasets")
    def list_datasets(request: Request) -> dict[str, Any]:
        datasets_dir = _settings(request).datasets_dir
        entries: list[dict[str, Any]] = []
        if datasets_dir.is_dir():
            for child in sorted(datasets_dir.iterdir()):
                if child.is_symlink() or not child.is_dir():
                    continue
                files: list[dict[str, Any]] = []
                for file_path in sorted(child.rglob("*")):
                    if file_path.is_symlink() or not file_path.is_file():
                        continue
                    files.append(
                        {
                            "name": file_path.relative_to(child).as_posix(),
                            "size_bytes": file_path.stat().st_size,
                        }
                    )
                formats = sorted(
                    {
                        Path(item["name"]).suffix.lstrip(".").lower()
                        for item in files
                        if Path(item["name"]).suffix
                    }
                )
                entries.append(
                    {
                        "dataset_id": child.name,
                        "path": f"datasets/{child.name}",
                        "num_files": len(files),
                        "formats": formats,
                        "files": files,
                    }
                )
        return {"datasets": entries, "count": len(entries)}

    return app


def _loads(value: Any) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):  # pragma: no cover
        return None

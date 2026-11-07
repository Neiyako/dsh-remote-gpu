"""Typed API errors with the unified error body required by the contract."""

from __future__ import annotations

from typing import Any


class WorkerError(Exception):
    """An error that is rendered as ``{"error": {"code", "message", "details"}}``."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}

    def to_body(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


# --- frequently used constructors -------------------------------------------------

def unauthorized(message: str = "Missing or invalid worker token") -> WorkerError:
    return WorkerError(401, "UNAUTHORIZED", message)


def forbidden(message: str) -> WorkerError:
    return WorkerError(403, "FORBIDDEN", message)


def not_found(code: str, message: str, details: dict[str, Any] | None = None) -> WorkerError:
    return WorkerError(404, code, message, details)


def task_not_found(task_id: str) -> WorkerError:
    return not_found("TASK_NOT_FOUND", f"Task {task_id!r} does not exist", {"task_id": task_id})


def conflict(code: str, message: str, details: dict[str, Any] | None = None) -> WorkerError:
    return WorkerError(409, code, message, details)


def bad_request(code: str, message: str, details: dict[str, Any] | None = None) -> WorkerError:
    return WorkerError(400, code, message, details)


def path_escape(message: str = "Resolved path escapes the allowed root") -> WorkerError:
    return bad_request("PATH_ESCAPE", message)


class ExecutorError(Exception):
    """Raised inside a job subprocess; becomes ``FAILED`` with ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

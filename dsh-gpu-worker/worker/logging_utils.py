"""Structured JSONL log files + internal progress/result marker protocol."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .utils import utcnow

PROGRESS_MARKER = "@@PROGRESS@@"
RESULT_MARKER = "@@RESULT@@"
ALL_MARKERS = (PROGRESS_MARKER, RESULT_MARKER)


def sanitize_log_text(text: str) -> str:
    """Internal markers must never leak through the public logs endpoint."""
    for marker in ALL_MARKERS:
        if marker in text:
            text = text.replace(marker, "[internal-marker-removed]")
    return text


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


class LogWriter:
    """Appends ``{"seq","ts","stream","text"}`` records to a JSONL file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._seq = _count_lines(path)
        self._handle = path.open("a", encoding="utf-8")

    @property
    def next_seq(self) -> int:
        with self._lock:
            return self._seq + 1

    def write(self, stream: str, text: str) -> None:
        text = sanitize_log_text(text.rstrip("\n"))
        with self._lock:
            if self._handle.closed:
                return
            self._seq += 1
            record = {"seq": self._seq, "ts": utcnow(), "stream": stream, "text": text}
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()


def read_log(path: Path | None, offset: int, limit: int) -> dict[str, Any]:
    """Read a page of log records; returns the contract §2.6 payload."""
    lines: list[dict[str, Any]] = []
    total = 0
    if path is not None and path.is_file():
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, raw in enumerate(handle):
                total = index + 1
                if index < offset:
                    continue
                if len(lines) >= limit:
                    continue
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    record = {"seq": index + 1, "ts": utcnow(), "stream": "stdout", "text": raw}
                record["text"] = sanitize_log_text(str(record.get("text", "")))
                lines.append(record)
    return {
        "offset": offset,
        "next_offset": offset + len(lines),
        "lines": lines,
        "eof": offset + len(lines) >= total,
    }
